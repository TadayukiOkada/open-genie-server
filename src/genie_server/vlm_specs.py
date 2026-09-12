"""Per-VLM-model specs (preprocessing, node topology, prompt template).

This is the "model-specific layer" paired with genie_node.py in this package (the generic
plumbing layer) — the same split as GenieX's `core/` vs `models/*.h`
pattern. Adding support for a new VLM just means registering a new VLMSpec
instance in this file (no changes needed to the generic code in
genie_server/).

The normalization constants, fixed resolution, and patch ordering can only
be trusted once this module's patchify implementation is verified to be
byte-identical to the real transformers processor's output. The values in
this module come from the QAIRT AI Hub Qwen3-VL-4B export's
(MODELS/qwen3_vl_4b_instruct-genie-w4a16-qualcomm_sa8775p/metadata.json)
genie.vision_preprocessing block — for a different export/resolution, check
that model's own metadata.json and build a new VLMSpec accordingly.

The Gemma 4 spec's constants come from the model's own processor config
(Gemma4ImageProcessor), and its patch grid from the bundle's image-encoder
config at load time — see gemma4_bind.
"""
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np


# ---------------------------------------------------------------- VLMSpec

@dataclass
class VLMSpec:
    name: str

    # Paths relative to the model directory root. Passed straight to
    # genie_node.Node(...).
    node_config_files: dict          # {"image_encoder": ..., "text_encoder": ..., "text_generator": ...}

    # List of (producer_key, producer_io, consumer_key, consumer_io).
    # Each key corresponds to a key in node_config_files.
    connections: list

    # Each node's primary IO name (a key string from genie_node.NODE_IO).
    text_encoder_text_input_io: str
    text_encoder_embedding_output_io: str
    image_encoder_image_input_io: str
    image_encoder_embedding_output_io: str
    text_generator_embedding_input_io: str
    text_generator_text_output_io: str

    # Auxiliary tensors that don't depend on image content (positional
    # encoding / attention masks). {IO name: path relative to the model
    # directory}. Assumes a fixed resolution — loaded once at startup and
    # reused across every subsequent request.
    static_tensor_files: dict

    # Preprocessing parameters
    image_width: int
    image_height: int
    patch_size: int
    spatial_merge_size: int
    temporal_patch_size: int
    normalize_mean: tuple
    normalize_std: tuple

    # Prompt template function:
    #   (system_text, parts, video_meta, spec) -> list[("text", str) | ("step", payload)]
    # parts is exactly the ("text"|"image"|"video", value) list returned by
    # vlm.extract_multimodal_parts(); video_meta is the dict from
    # vlm.extract_video_meta(). The return value is the exact order to feed
    # into the Accumulator (the final text/step interleave order).
    #
    # A "step" is one image-encoder execution, NOT one input image: the ViT
    # consumes temporal_patch_size frames at a time. The payload is whatever
    # this spec's own preprocess_step understands (for Qwen3-VL, a tuple of
    # indices into the images list) — the generic code in vlm.py passes it
    # straight through without interpreting it.
    build_prompt_segments: Callable = field(repr=False)

    # Preprocessing function: (images, payload, spec) -> pixel_values ndarray
    # for one step. Takes the whole images list plus the payload its own
    # build_prompt_segments emitted, so that packing several frames into one
    # step stays entirely inside this module. Takes spec as an argument
    # (rather than a self-referential closure bound at dataclass
    # construction time).
    preprocess_step: Callable = field(repr=False)

    # Rows pixel_values is zero-padded to, for an encoder exported with a
    # fixed maximum and told at load time how many of them are real
    # (Gemma 4). 0 = the input is exactly the resolution's own patch count
    # (Qwen3-VL).
    max_patches: int = 0

    # True when the bundle's text-encoder config names a bos-token, so
    # libGenie prepends a BOS to every text segment itself. Set by VLMSlot
    # from the loaded config; a template with a BOS of its own leaves it out.
    text_encoder_adds_bos: bool = False

    # (spec, node_cfgs) -> VLMSpec, or None. Called once per slot with every
    # node config loaded (paths already resolved) but before any node is
    # created, so a spec can read what the bundle was exported with and
    # adjust the configs it is about to be built from. The slot uses the spec
    # it returns.
    bind: Callable | None = field(default=None, repr=False)

    @property
    def vision_tokens_per_step(self) -> int:
        """How much context one step costs the text-generator.

        The ViT emits one embedding per patch, and the spatial merge folds
        each spatial_merge_size**2 block into a single token before the LLM
        sees it. For the 512x512 / patch 16 / merge 2 export that is
        (512/16)**2 / 2**2 = 256 — a quarter of the 1024 rows in
        pixel_values, which is the easy number to mistake it for.
        """
        patches = ((self.image_height // self.patch_size)
                   * (self.image_width // self.patch_size))
        return patches // (self.spatial_merge_size ** 2)


# ---------------------------------------------------------------- preprocessing (Qwen3-VL)

def _qwen3vl_normalize(rgb: np.ndarray, mean: tuple, std: tuple) -> np.ndarray:
    """(H, W, 3) uint8 -> (3, H, W) float32"""
    x = rgb.astype(np.float32) * (1.0 / 255.0)
    mean_arr = np.array(mean, np.float32)
    istd_arr = 1.0 / np.array(std, np.float32)
    return ((x - mean_arr) * istd_arr).transpose(2, 0, 1)


def _qwen3vl_patchify(frame0: np.ndarray, frame1: np.ndarray, spec: "VLMSpec") -> np.ndarray:
    """2 frames (H,W,3 uint8) -> (rows, cols) float32.

    Ordering: row = [hb][wb][mh][mw], within a row = [c][t][ph][pw]
    (implemented to match transformers' Qwen3VLVideoProcessor ordering —
    verify separately via a byte-for-byte comparison against golden output.
    See preprocess.py.)
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


def _qwen3vl_frame_times(n_frames: int, video_meta: dict):
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
            times = _qwen3vl_frame_times(len(frames), video_meta)
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


# ---------------------------------------------------------------- Gemma 4

def gemma4_bind(spec: "VLMSpec", node_cfgs: dict) -> "VLMSpec":
    """Fixes the spec to the patch grid the bundle's image-encoder config
    names. The configs themselves are left as they are.

    **The grid is per slot, not per image.** Gemma 4's own processor resizes
    each image to the largest grid of its aspect ratio that fits the patch
    budget (640x427 becomes 60x39 patches). On the device the encoder's
    position ids and pooling index are computed from `vision-param` (height
    and width in patches, pooling-kernel-size), but that block is read once,
    when the node is created, and nothing changes it per request. So every
    image this slot sees is resized to that one grid, stretched if its aspect
    ratio differs; set vision-param to the grid that suits the input.
    """
    vision = (next(iter(node_cfgs["image_encoder"].values()))
              .get("engine", {}).get("model", {}).get("vision-param") or {})
    rows, cols = vision.get("height"), vision.get("width")
    pool = vision.get("pooling-kernel-size")
    if not all(isinstance(v, int) and v > 0 for v in (rows, cols, pool)):
        raise ValueError(
            "gemma4: the image-encoder config needs engine.model.vision-param "
            "with positive integer height, width and pooling-kernel-size "
            f"(got {vision!r})")
    if rows % pool or cols % pool:
        raise ValueError(
            f"gemma4: vision-param {rows}x{cols} patches is not divisible by "
            f"pooling-kernel-size {pool}")
    if rows * cols > spec.max_patches:
        raise ValueError(
            f"gemma4: vision-param {rows}x{cols} = {rows * cols} patches "
            f"exceeds the {spec.max_patches} the encoder takes")

    return replace(spec, image_height=rows * spec.patch_size,
                   image_width=cols * spec.patch_size, spatial_merge_size=pool)


def gemma4_preprocess_step(images: list, payload, spec: "VLMSpec") -> np.ndarray:
    """One image -> pixel_values (max_patches, 3 * patch * patch) float32.

    What Gemma4ImageProcessor does apart from choosing the grid (see
    gemma4_bind): bicubic resize, rescale to [0, 1] with no normalization
    (Gemma 4 was trained on [0, 1] pixels), patches in raster order with each
    patch flattened channels-last, zero rows out to max_patches. Compared
    against the transformers processor's output for a 640x427 image at the
    grid it picks (60x39): same layout, every value within one 8-bit step.
    """
    from PIL import Image

    if len(payload) != 1:
        raise ValueError(
            f"step payload has {len(payload)} frames, but the gemma4 encoder "
            "takes one image per execution")
    patch = spec.patch_size
    rows, cols = spec.image_height // patch, spec.image_width // patch
    img = images[payload[0]].convert("RGB").resize(
        (spec.image_width, spec.image_height), Image.BICUBIC)
    x = np.asarray(img, dtype=np.float32) / 255.0                  # (H, W, 3)
    patches = (x.reshape(rows, patch, cols, patch, 3)
               .transpose(0, 2, 1, 3, 4)                             # r c ph pw ch
               .reshape(rows * cols, -1))
    out = np.zeros((spec.max_patches, patches.shape[1]), np.float32)
    out[:rows * cols] = patches
    return out


def gemma4_build_prompt_segments(system_text: str, parts: list,
                                 video_meta: dict, spec: "VLMSpec") -> list:
    """OpenAI content parts -> Accumulator-order segments in Gemma 4's chat
    format.

    Follows the model's chat template: `<bos>` (unless the text-encoder adds
    one to every segment itself — spec.text_encoder_adds_bos — in which case
    the prompt carries the SDK's BOS tokens and none of ours), a system turn
    only when there is a system message, text parts trimmed and joined with nothing between
    them, and each image as `<|image>` + its soft tokens + `<image|>` — what
    the processor expands the template's image placeholder into. The soft
    tokens are the encoder's output, so the text before an image ends at
    `<|image>` and the text after it starts at `<image|>`.

    Video is refused. Gemma 4 does take video, but as frames encoded at a
    smaller budget than still images (70 soft tokens rather than 280), each
    after an mm:ss timestamp: a different encoder resolution from the one
    this slot's grid is fixed to.
    """
    segments = []
    buf = "" if spec.text_encoder_adds_bos else "<bos>"
    if system_text:
        buf += "<|turn>system\n" + system_text.strip() + "<turn|>\n"
    buf += "<|turn>user\n"

    for kind, value in parts:
        if kind == "text":
            buf += value.strip()
        elif kind == "image":
            buf += "<|image>"
            segments.append(("text", buf))
            segments.append(("step", (value,)))
            buf = "<image|>"
        elif kind == "video":
            raise ValueError(
                "the gemma4 VLM spec does not take video_url parts; send "
                "still images as image_url parts")
        else:
            raise ValueError(f"unknown content part kind: {kind}")

    buf += "<turn|>\n<|turn>model\n"
    segments.append(("text", buf))
    return segments


# ---------------------------------------------------------------- registry

QWEN3_VL_SPEC = VLMSpec(
    name="qwen3_vl",
    node_config_files={
        "image_encoder": "img-enc-htp.json",
        "text_encoder": "text-encoder.json",
        "text_generator": "text-generator.json",
    },
    connections=[
        ("image_encoder", "IMAGE_ENCODER_EMBEDDING_OUTPUT",
         "text_generator", "TEXT_GENERATOR_EMBEDDING_INPUT"),
        ("text_encoder", "TEXT_ENCODER_EMBEDDING_OUTPUT",
         "text_generator", "TEXT_GENERATOR_EMBEDDING_INPUT"),
        # Needed to auto-wire same-named tensors other than the main
        # embedding, e.g. deepstack_visual_embeds_* (dropped without WILDCARD).
        ("image_encoder", "WILDCARD", "text_generator", "WILDCARD"),
    ],
    text_encoder_text_input_io="TEXT_ENCODER_TEXT_INPUT",
    text_encoder_embedding_output_io="TEXT_ENCODER_EMBEDDING_OUTPUT",
    image_encoder_image_input_io="IMAGE_ENCODER_IMAGE_INPUT",
    image_encoder_embedding_output_io="IMAGE_ENCODER_EMBEDDING_OUTPUT",
    text_generator_embedding_input_io="TEXT_GENERATOR_EMBEDDING_INPUT",
    text_generator_text_output_io="TEXT_GENERATOR_TEXT_OUTPUT",
    static_tensor_files={
        "IMAGE_ENCODER_IMAGE_POS_COS": "sample_inputs/position_ids_cos.raw",
        "IMAGE_ENCODER_IMAGE_POS_SIN": "sample_inputs/position_ids_sin.raw",
        "IMAGE_ENCODER_IMAGE_FULL_ATTN_MASK": "sample_inputs/full_attention_mask.raw",
        "IMAGE_ENCODER_IMAGE_WINDOW_ATTN_MASK": "sample_inputs/window_attention_mask.raw",
    },
    image_width=512,
    image_height=512,
    patch_size=16,
    spatial_merge_size=2,
    temporal_patch_size=2,
    # From MODELS/qwen3_vl_4b_instruct-genie-w4a16-qualcomm_sa8775p/metadata.json's
    # genie.vision_preprocessing (note: not the standard CLIP constants).
    normalize_mean=(0.5, 0.5, 0.5),
    normalize_std=(0.5, 0.5, 0.5),
    build_prompt_segments=qwen3vl_build_prompt_segments,
    preprocess_step=qwen3vl_preprocess_step,
)

GEMMA4_SPEC = VLMSpec(
    name="gemma4",
    node_config_files={
        "image_encoder": "image-encoder.json",
        "text_encoder": "text-encoder.json",
        "text_generator": "text-generator.json",
    },
    # No WILDCARD: the encoder's only output is the image embedding, and the
    # position ids and pooling index it takes are made on the device.
    connections=[
        ("image_encoder", "IMAGE_ENCODER_EMBEDDING_OUTPUT",
         "text_generator", "TEXT_GENERATOR_EMBEDDING_INPUT"),
        ("text_encoder", "TEXT_ENCODER_EMBEDDING_OUTPUT",
         "text_generator", "TEXT_GENERATOR_EMBEDDING_INPUT"),
    ],
    text_encoder_text_input_io="TEXT_ENCODER_TEXT_INPUT",
    text_encoder_embedding_output_io="TEXT_ENCODER_EMBEDDING_OUTPUT",
    image_encoder_image_input_io="IMAGE_ENCODER_IMAGE_INPUT",
    image_encoder_embedding_output_io="IMAGE_ENCODER_EMBEDDING_OUTPUT",
    text_generator_embedding_input_io="TEXT_GENERATOR_EMBEDDING_INPUT",
    text_generator_text_output_io="TEXT_GENERATOR_TEXT_OUTPUT",
    static_tensor_files={},
    # Placeholders until gemma4_bind reads the grid from vision-param.
    image_width=0,
    image_height=0,
    patch_size=16,
    # pooling-kernel-size plays the part of the spatial merge: each k x k
    # block of patches becomes one soft token (39x60 patches -> 260).
    spatial_merge_size=3,
    temporal_patch_size=1,
    # The processor config's image_processor: rescale only, no normalization.
    normalize_mean=(0.0, 0.0, 0.0),
    normalize_std=(1.0, 1.0, 1.0),
    build_prompt_segments=gemma4_build_prompt_segments,
    preprocess_step=gemma4_preprocess_step,
    # max_soft_tokens 280 x pooling 3**2: the encoder's input length in the
    # E2B export (pixel_values [1, 2520, 768]).
    max_patches=2520,
    bind=gemma4_bind,
)

VLM_SPECS = {
    "qwen3_vl": QWEN3_VL_SPEC,
    "gemma4": GEMMA4_SPEC,
}


def get_spec(name: str) -> VLMSpec:
    try:
        return VLM_SPECS[name]
    except KeyError:
        raise KeyError(
            f"Unknown VLM spec '{name}'. Registered: {sorted(VLM_SPECS)}"
        ) from None
