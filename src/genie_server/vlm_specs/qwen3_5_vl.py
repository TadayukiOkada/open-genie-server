"""Qwen3.5 family (image input): Qwen3-VL's vision side with
Qwen3.5's own chat template.

The ViT, its preprocessing and the <|vision_start|>/<|vision_end|> markers
are Qwen3-VL's, so this module reuses qwen3_vl.py for all of them. What
differs is the text side, taken from Qwen3.5's chat_template.jinja:

  - The generation header goes past "<|im_start|>assistant\\n": "<think>\\n"
    when thinking, the empty block "<think>\\n\\n</think>\\n\\n" when not.
    Without either, the model writes its own "<think>" and, decoding
    greedily, ends the reply there: an SA8255P Qwen3.5-4B bundle returned
    "<think>\\n</think>\\n\\n" and nothing else on every greedy request while
    it was served by the qwen3_vl family.
  - The system text and the user turn are trimmed as a whole (`|trim` on the
    rendered content), so only the outer ends of the turn lose whitespace.

Which of the two headers a request gets follows the text slot's "qwen3_5"
template (chat_template below): templates.default_thinking when the request
does not say, and templates.generation_prefix gives a thinking reply back
with the "<think>\\n" the prompt opened.
"""
from .base import VLMFamily, VLMSpec
from .qwen3_vl import (QWEN3_VL_FAMILY, _qwen_vision_markers, is_linear_attention,
                       qwen3vl_build_prompt_segments)


def _trim_turn(parts: list) -> list:
    """`|trim` on the rendered user content: leading whitespace off the first
    part and trailing off the last, when those are text. A part in between,
    or text next to an image, keeps its own."""
    parts = list(parts)
    if parts and parts[0][0] == "text":
        parts[0] = ("text", parts[0][1].lstrip())
    if parts and parts[-1][0] == "text":
        parts[-1] = ("text", parts[-1][1].rstrip())
    return parts


def qwen3_5_vl_build_prompt_segments(system_text: str, parts: list,
                                     video_meta: dict, spec: "VLMSpec",
                                     enable_thinking: bool = False) -> list:
    """Qwen3-VL's segments (see qwen3vl_build_prompt_segments) for the
    trimmed turn, with Qwen3.5's generation header after the assistant one."""
    segments = qwen3vl_build_prompt_segments(
        (system_text or "").strip(), _trim_turn(parts), video_meta, spec)
    kind, text = segments[-1]
    header = "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"
    segments[-1] = (kind, text + header)
    return segments


def qwen3_5_vl_detect(tokenizer_json: dict, node_cfgs: dict) -> bool:
    """The Qwen vision markers on a bundle with linear attention."""
    return (_qwen_vision_markers(tokenizer_json, node_cfgs)
            and is_linear_attention(node_cfgs))


QWEN3_5_VL_FAMILY = VLMFamily(
    name="qwen3_5_vl",
    build_prompt_segments=qwen3_5_vl_build_prompt_segments,
    preprocess_step=QWEN3_VL_FAMILY.preprocess_step,
    bind=QWEN3_VL_FAMILY.bind,
    detect=qwen3_5_vl_detect,
    image_width=QWEN3_VL_FAMILY.image_width,
    image_height=QWEN3_VL_FAMILY.image_height,
    patch_size=QWEN3_VL_FAMILY.patch_size,
    spatial_merge_size=QWEN3_VL_FAMILY.spatial_merge_size,
    temporal_patch_size=QWEN3_VL_FAMILY.temporal_patch_size,
    normalize_mean=QWEN3_VL_FAMILY.normalize_mean,
    normalize_std=QWEN3_VL_FAMILY.normalize_std,
    chat_template="qwen3_5",
)
