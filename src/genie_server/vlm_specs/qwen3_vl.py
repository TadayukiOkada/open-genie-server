"""Qwen3-VL family: preprocessing, chat template, and the bundle-specific
resolution (resolution/patch grid, spatial merge) that varies across the
three known layouts (AI Hub 4B, the DeepStack tutorial 4B, and 2B) but
nothing else about the model does.

The patch ordering follows transformers' Qwen3VLVideoProcessor. The values
are not byte-identical to that processor's output: it resamples with
interpolation, this resizes once and normalizes. Recognition on the device
is correct with it, which is why the difference is accepted; nothing in
tests/ compares against the processor.
"""
from dataclasses import replace

import numpy as np

from .base import VLMFamily, VLMSpec

# ---------------------------------------------------------------- preprocessing

def _qwen3vl_normalize(rgb: np.ndarray, mean: tuple, std: tuple) -> np.ndarray:
    """(H, W, 3) uint8 -> (3, H, W) float32"""
    x = rgb.astype(np.float32) * (1.0 / 255.0)
    mean_arr = np.array(mean, np.float32)
    istd_arr = 1.0 / np.array(std, np.float32)
    return ((x - mean_arr) * istd_arr).transpose(2, 0, 1)


def _qwen3vl_patchify(frame0: np.ndarray, frame1: np.ndarray, spec: "VLMSpec") -> np.ndarray:
    """2 frames (H,W,3 uint8) -> (rows, cols) float32.

    Ordering: row = [hb][wb][mh][mw], within a row = [c][t][ph][pw]
    (transformers' Qwen3VLVideoProcessor ordering; the values themselves
    differ from its output, see the module docstring.)
    """
    patch, merge, temporal = spec.patch_size, spec.spatial_merge_size, spec.temporal_patch_size
    grid_h = spec.image_height // patch
    grid_w = spec.image_width // patch

    a = np.stack([
        _qwen3vl_normalize(frame0, spec.normalize_mean, spec.normalize_std),
        _qwen3vl_normalize(frame1, spec.normalize_mean, spec.normalize_std),
    ], axis=1)  # (3, T, H, W)
    a = a.reshape(3, temporal, grid_h, patch, grid_w, patch)
    a = a.reshape(3, temporal,
                  grid_h // merge, merge, patch,
                  grid_w // merge, merge, patch)
    #             0  1        2         3      4         5      6      7
    #             c  t        hb        mh     ph        wb     mw     pw
    a = a.transpose(2, 5, 3, 6, 0, 1, 4, 7)
    rows = grid_h * grid_w
    cols = 3 * temporal * patch * patch
    return np.ascontiguousarray(a).reshape(rows, cols)


def _resize_to_spec(pil_image, spec: "VLMSpec") -> np.ndarray:
    """Arbitrary-size PIL image -> RGB uint8 ndarray at the spec's fixed resolution.

    A plain resize (not an aspect-ratio-preserving letterbox/center-crop).
    Verify separately if you need strict parity with the real transformers
    processor.
    """
    from PIL import Image
    img = pil_image.convert("RGB").resize(
        (spec.image_width, spec.image_height), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def qwen3vl_preprocess_step(images: list, payload, spec: "VLMSpec") -> np.ndarray:
    """One step's frames -> pixel_values (rows, cols) float32.

    payload is the tuple of temporal_patch_size indices into `images` that
    qwen3vl_build_prompt_segments emitted for this step. A still image
    duplicates its own index, which is the standard way Qwen-family
    processors fill the temporal dimension for a single picture; consecutive
    video frames give the ViT two genuinely different frames, which is what
    lets it see motion at all.
    """
    if len(payload) != spec.temporal_patch_size:
        raise ValueError(
            f"step payload has {len(payload)} frames, but this spec's ViT "
            f"takes {spec.temporal_patch_size} per execution")
    frames = [_resize_to_spec(images[i], spec) for i in payload]
    return _qwen3vl_patchify(frames[0], frames[1], spec)


def _frame_times(n_frames: int, video_meta: dict):
    """Per-frame timestamps in seconds, or None when the request carried no
    timeline to derive them from.

    Reads what vLLM accepts in `media_io_kwargs.video` for client-side frame
    extraction: `fps` (a float, or a single-element list — the Qwen examples
    pass `[3.0]`) and optionally `frames_indices`, the position of each
    supplied frame in the source video.

    **`fps` means two different things, and which one depends on
    `frames_indices`** — vLLM keeps them in separate fields and this one key
    has to carry both:

      with frames_indices     the SOURCE video's frame rate, because the
                              indices are positions in that video and the
                              time of one is idx / fps. This is vLLM's
                              VideoMetadata["fps"], the number
                              _calculate_timestamps divides by.
      without frames_indices  the rate the frames were SAMPLED at, because
                              evenly spaced frames are all there is to go on
                              and frame k is then at k / fps. This is the
                              `fps` of media_io_kwargs.video itself, which in
                              vLLM asks a backend to sample at that rate.

    Sending a source fps without indices would therefore date the whole clip
    wrong (30 fps reads as frames 33 ms apart), so a `frames_indices` whose
    length does not match the frames supplied returns None instead of
    quietly falling back to the other meaning: the count disagreeing is the
    one signal available that the two are out of step.

    Returning None rather than inventing a default fps is the same rule: the
    `<t seconds>` markers claim a real timeline to the model, so a wrong one
    is worse than none.
    """
    fps = video_meta.get("fps")
    if isinstance(fps, (list, tuple)):
        fps = fps[0] if fps else None
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        return None
    if fps <= 0:
        return None

    indices = video_meta.get("frames_indices")
    if indices is not None:
        if not isinstance(indices, (list, tuple)) or len(indices) != n_frames:
            return None
        try:
            return [float(i) / fps for i in indices]
        except (TypeError, ValueError):
            return None
    return [k / fps for k in range(n_frames)]


def _qwen3vl_step_time(times: list, start: int, per_step: int) -> float:
    """The timestamp Qwen3-VL trained on for the step packing frames
    [start, start + per_step).

    The midpoint of the window, not its first frame: the ViT collapses the
    whole group into one visual chunk, and the reference implementation dates
    that chunk by averaging the group's first and last frame times (vLLM's
    Qwen3VLProcessor._calculate_timestamps, which pads the index list with
    its last entry before pairing). Clamping to the final frame reproduces
    that padding — the same repeat the odd tail's pixels get — so a two-frame
    step at 2 fps is 0.2s rather than 0.0s, and an unevenly sampled pair is
    dated between its frames rather than at the earlier one.
    """
    end = min(start + per_step - 1, len(times) - 1)
    return (times[start] + times[end]) / 2.0


def qwen3vl_build_prompt_segments(system_text: str, parts: list,
                                  video_meta: dict, spec: "VLMSpec") -> list:
    """Converts OpenAI content parts (an ordered list of text/image/video
    tuples) into Accumulator-feed-order segments, including the Qwen3-VL chat
    template.

    Each returned element: ("text", str) | ("step", (frame_idx, ...))
    Text segments are complete fragments with <|vision_start|>/<|vision_end|>
    already inserted around each step (callers can pass them straight to the
    text_encoder's setData).

    A "video" part becomes ceil(frames / temporal_patch_size) steps, each
    carrying two consecutive frames, prefixed with a `<t seconds>` marker
    when the request supplied an fps to derive one from (the midpoint of the
    frames in that step — see _qwen3vl_step_time). An odd frame count repeats
    the final frame to fill the last step, the same padding a still image
    gets.
    """
    video_meta = video_meta or {}
    per_step = spec.temporal_patch_size

    segments = []
    buf = "<|im_start|>system\n" + system_text + "<|im_end|>\n" if system_text else ""
    buf += "<|im_start|>user\n"

    def emit_step(payload):
        nonlocal buf
        buf += "<|vision_start|>"
        segments.append(("text", buf))
        buf = ""
        segments.append(("step", tuple(payload)))
        buf = "<|vision_end|>"

    for kind, value in parts:
        if kind == "text":
            buf += value
        elif kind == "image":
            emit_step([value] * per_step)
        elif kind == "video":
            frames = list(value)
            times = _frame_times(len(frames), video_meta)
            for start in range(0, len(frames), per_step):
                window = frames[start:start + per_step]
                while len(window) < per_step:      # odd tail: repeat the last frame
                    window.append(window[-1])
                if times is not None:
                    t = _qwen3vl_step_time(times, start, per_step)
                    buf += f"<{t:.1f} seconds>"
                emit_step(window)
        else:
            raise ValueError(f"unknown content part kind: {kind}")

    buf += "<|im_end|>\n<|im_start|>assistant\n"
    segments.append(("text", buf))
    return segments


# ---------------------------------------------------------------- bind / detect

def _dig(cfg: dict, *keys):
    for k in keys:
        if not isinstance(cfg, dict):
            return None
        cfg = cfg.get(k)
    return cfg


def _meta_int(vp: dict, key: str, default: int) -> int:
    """An integer from metadata.json genie.vision_preprocessing. int() used
    to truncate a fraction without a word: temporal_patch_size 2.7 became 2
    and passed, image_width 2.5 became 2 and failed every request."""
    value = vp.get(key, default)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    elif isinstance(value, str):
        try:
            value = int(value.strip())
        except ValueError:
            pass
    if isinstance(value, bool) or not isinstance(value, int):
        # ValueError, like every other bad bundle value bind reports.
        raise ValueError(  # noqa: TRY004
            f"qwen3_vl: metadata.json genie.vision_preprocessing.{key} must be "
            f"an integer, got {vp.get(key)!r}")
    return value


def qwen3vl_bind(spec: "VLMSpec", node_cfgs: dict, layout) -> "VLMSpec":
    """Resolves the resolution/patch grid from whatever the bundle states,
    in this order:

      1. The image-encoder's own `vision-param.height/width` (in patches —
         the AI Hub export, the DeepStack tutorial export, and the 2B export
         all carry this). Resolution = h*patch x w*patch.
      2. Failing that, metadata.json's `genie.vision_preprocessing` block
         (a plain width/height/patch/temporal/merge/mean/std dict).
      3. Failing that, the family defaults (512x512, patch 16, merge 2,
         temporal 2, mean/std 0.5) — the AI Hub export's own numbers, kept as
         the fallback because they are the ones every other value in this
         module was verified against.

    spatial_merge_size additionally comes from the ViT's own
    `rope-scaling.spatial-merge-size` when the image-encoder config states
    one (every export that also states vision-param does); the family
    default otherwise.

    The processor's normalization is fixed at 0.5/0.5/0.5 for every known
    export, 4B or 2B alike — see the HF `preprocessor_config.json`:
    ``{"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], ...}``.
    """
    image_cfg = next(iter(node_cfgs.get("image_encoder", {}).values()), {})
    model = image_cfg.get("engine", {}).get("model", {})
    vision = model.get("vision-param") or {}
    height_p, width_p = vision.get("height"), vision.get("width")

    if isinstance(height_p, int) and isinstance(width_p, int) and height_p > 0 and width_p > 0:
        spec = replace(spec, image_height=height_p * spec.patch_size,
                       image_width=width_p * spec.patch_size)
    else:
        vp = _dig(layout.metadata, "genie", "vision_preprocessing") or {}
        if vp:
            spec = replace(
                spec,
                image_width=_meta_int(vp, "image_width", spec.image_width),
                image_height=_meta_int(vp, "image_height", spec.image_height),
                patch_size=_meta_int(vp, "patch_size", spec.patch_size),
                temporal_patch_size=_meta_int(vp, "temporal_patch_size",
                                              spec.temporal_patch_size),
                spatial_merge_size=_meta_int(vp, "spatial_merge_size",
                                             spec.spatial_merge_size),
                normalize_mean=tuple(vp.get("normalize_mean", spec.normalize_mean)),
                normalize_std=tuple(vp.get("normalize_std", spec.normalize_std)),
            )
        # else: keep the family defaults set above.

    merge = _dig(model, "positional-encoding", "rope-scaling", "spatial-merge-size")
    if isinstance(merge, int) and merge > 0:
        spec = replace(spec, spatial_merge_size=merge)

    # _qwen3vl_patchify stacks exactly two frames -- the only temporal
    # patch size any export has, and the only one the patchify has run on a
    # device with. Anything else used to pass here and then fail the reshape
    # (a 500) on every request; refuse it at startup instead.
    if spec.temporal_patch_size != 2:
        raise ValueError(
            f"qwen3_vl: temporal_patch_size {spec.temporal_patch_size} (from "
            "metadata.json genie.vision_preprocessing) is not supported; the "
            "patchify takes exactly 2 frames per ViT execution")
    if spec.image_width <= 0 or spec.image_height <= 0:
        raise ValueError(
            "qwen3_vl: could not determine the image resolution from "
            "vision-param, metadata.json genie.vision_preprocessing, or the "
            "family defaults")
    if spec.patch_size <= 0 or spec.spatial_merge_size <= 0:
        raise ValueError(
            f"qwen3_vl: patch_size ({spec.patch_size}) and spatial_merge_size "
            f"({spec.spatial_merge_size}) must be positive")
    # The patchify reshapes each side into (grid, patch): a side that is not
    # a whole number of patches passed here and then failed that reshape (a
    # 500) on every request, like a temporal_patch_size other than 2.
    if spec.image_height % spec.patch_size or spec.image_width % spec.patch_size:
        raise ValueError(
            f"qwen3_vl: image {spec.image_height}x{spec.image_width} is not a "
            f"whole number of {spec.patch_size}-pixel patches")
    grid_h = spec.image_height // spec.patch_size
    grid_w = spec.image_width // spec.patch_size
    # _qwen3vl_patchify reshapes each of grid_h and grid_w into
    # (grid // merge, merge) separately, so each one — not just their
    # product — must be divisible by merge. A product that happens to be
    # divisible by merge**2 (e.g. grid_h=27, grid_w=36, merge=2: 972 % 4 == 0)
    # can still fail the reshape if only one side carries the factor.
    if grid_h % spec.spatial_merge_size or grid_w % spec.spatial_merge_size:
        raise ValueError(
            f"qwen3_vl: {grid_h}x{grid_w} patches ({spec.image_height}x"
            f"{spec.image_width} at patch {spec.patch_size}) is not "
            f"divisible by spatial_merge_size ({spec.spatial_merge_size}) "
            "in both dimensions")
    return spec


def qwen3vl_detect(tokenizer_json: dict, node_cfgs: dict) -> bool:
    """Auto-detection signal for VLM_SLOTS[].spec when it is not given: the
    tokenizer's own vision marker, or (belt and suspenders) the
    text-generator's mRoPE rope-type."""
    added = {t.get("content") for t in (tokenizer_json or {}).get("added_tokens", [])}
    if "<|vision_start|>" in added:
        return True
    tg = next(iter(node_cfgs.get("text_generator", {}).values()), {})
    rope_type = _dig(tg, "engine", "model", "positional-encoding", "rope-scaling", "rope-type")
    return rope_type == "qwen3vl-mrope"


QWEN3_VL_FAMILY = VLMFamily(
    name="qwen3_vl",
    build_prompt_segments=qwen3vl_build_prompt_segments,
    preprocess_step=qwen3vl_preprocess_step,
    bind=qwen3vl_bind,
    detect=qwen3vl_detect,
    image_width=512,
    image_height=512,
    patch_size=16,
    spatial_merge_size=2,
    temporal_patch_size=2,
    # From MODELS/qwen3_vl_4b_instruct-genie-w4a16-qualcomm_sa8775p/metadata.json's
    # genie.vision_preprocessing (note: not the standard CLIP constants) —
    # also true for the 2B and the DeepStack tutorial export.
    normalize_mean=(0.5, 0.5, 0.5),
    normalize_std=(0.5, 0.5, 0.5),
)
