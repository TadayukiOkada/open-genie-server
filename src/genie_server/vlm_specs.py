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
"""
from dataclasses import dataclass, field
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

    # Prompt template function: (system_text, parts) -> list[("text", str) | ("image", int)]
    # parts is exactly the ("text"|"image", value) list returned by
    # _extract_image_parts(). The return value is the exact order to feed
    # into the Accumulator (the final text/image interleave order).
    build_prompt_segments: Callable = field(repr=False)

    # Preprocessing function: (pil_image, spec) -> pixel_values ndarray.
    # Takes spec as an argument (rather than a self-referential closure
    # bound at dataclass construction time).
    preprocess_image: Callable = field(repr=False)


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


def qwen3vl_preprocess_image(pil_image, spec: "VLMSpec") -> np.ndarray:
    """A single still image -> pixel_values (rows, cols) float32.

    Since the ViT expects temporal_patch_size=2, the same frame is
    duplicated to fill the temporal dimension (the standard way Qwen-family
    processors handle a still image).
    """
    frame = _resize_to_spec(pil_image, spec)
    return _qwen3vl_patchify(frame, frame, spec)


def qwen3vl_build_prompt_segments(system_text: str, parts: list) -> list:
    """Converts OpenAI content parts (an ordered list of text/image tuples)
    into Accumulator-feed-order segments, including the Qwen3-VL chat
    template.

    Each returned element: ("text", str) | ("image", index)
    Text segments are complete fragments with <|vision_start|>/<|vision_end|>
    already inserted around images (callers can pass them straight to the
    text_encoder's setData).
    """
    segments = []
    buf = "<|im_start|>system\n" + system_text + "<|im_end|>\n" if system_text else ""
    buf += "<|im_start|>user\n"
    image_idx = 0

    for kind, value in parts:
        if kind == "text":
            buf += value
        elif kind == "image":
            buf += "<|vision_start|>"
            segments.append(("text", buf))
            buf = ""
            segments.append(("image", image_idx))
            image_idx += 1
            buf = "<|vision_end|>"
        else:
            raise ValueError(f"unknown content part kind: {kind}")

    buf += "<|im_end|>\n<|im_start|>assistant\n"
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
    preprocess_image=qwen3vl_preprocess_image,
)

VLM_SPECS = {
    "qwen3_vl": QWEN3_VL_SPEC,
}


def get_spec(name: str) -> VLMSpec:
    try:
        return VLM_SPECS[name]
    except KeyError:
        raise KeyError(
            f"Unknown VLM spec '{name}'. Registered: {sorted(VLM_SPECS)}"
        ) from None
