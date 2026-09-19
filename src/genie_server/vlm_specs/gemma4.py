"""Gemma 4 family: preprocessing, chat template, and the bundle-specific
patch grid (read from the image-encoder's own vision-param at bind time).
"""
from dataclasses import replace

import numpy as np

from .base import VLMFamily, VLMSpec
from .qwen3_vl import _frame_times

# ---------------------------------------------------------------- bind / detect

def gemma4_bind(spec: "VLMSpec", node_cfgs: dict, layout) -> "VLMSpec":
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


def gemma4_detect(tokenizer_json: dict, node_cfgs: dict) -> bool:
    """Auto-detection signal for VLM_SLOTS[].spec when it is not given: the
    tokenizer's own image markers, or (belt and suspenders) the
    image-encoder's pooling-kernel-size — Qwen3-VL's vision-param has no such
    key."""
    added = {t.get("content") for t in (tokenizer_json or {}).get("added_tokens", [])}
    if "<|image>" in added or "<image|>" in added:
        return True
    image_cfg = next(iter(node_cfgs.get("image_encoder", {}).values()), {})
    pool = image_cfg.get("engine", {}).get("model", {}).get("vision-param", {}).get(
        "pooling-kernel-size")
    return isinstance(pool, int) and pool > 0


# ---------------------------------------------------------------- preprocessing

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


def _mmss(seconds: float) -> str:
    """Gemma 4's video frame timestamp: minutes and whole seconds, as
    Gemma4Processor.replace_video_token formats them."""
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


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

    Video is one step per frame — Gemma 4 has no temporal packing — laid out
    as Gemma4Processor does: `mm:ss <|image>` + that frame's soft tokens +
    `<image|>`, frames joined by a space. The timestamps come from
    media_io_kwargs.video exactly as Qwen3-VL's do (_frame_times), truncated
    to whole seconds; with no fps none is written, where the processor would
    assume 24 fps. Every frame is encoded at this slot's grid, and Gemma 4's
    video processor budgets 70 soft tokens a frame against 280 for a still:
    a slot meant for video wants the smaller grid (24x24 for 512x512 frames).
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
            frames = list(value)
            times = _frame_times(len(frames), video_meta)
            for k, index in enumerate(frames):
                if k:
                    buf += " "
                if times is not None:
                    buf += _mmss(times[k]) + " "
                buf += "<|image>"
                segments.append(("text", buf))
                segments.append(("step", (index,)))
                buf = "<image|>"
        else:
            raise ValueError(f"unknown content part kind: {kind}")

    buf += "<turn|>\n<|turn>model\n"
    segments.append(("text", buf))
    return segments


GEMMA4_FAMILY = VLMFamily(
    name="gemma4",
    build_prompt_segments=gemma4_build_prompt_segments,
    preprocess_step=gemma4_preprocess_step,
    bind=gemma4_bind,
    detect=gemma4_detect,
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
    # max_soft_tokens 280 x pooling 3**2: the encoder's input length in the
    # E2B export (pixel_values [1, 2520, 768]).
    max_patches=2520,
)
