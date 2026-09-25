"""Tests for vlm_layout.py (bundle layout auto-read) and the vlm_specs
family/resolve split it feeds. Covers the four known bundle layouts (AI Hub
Qwen3-VL-4B, the DeepStack tutorial 4B, the 2B LMM bundle, Gemma 4 E2B) via
tests/data/vlm_bundles/, plus the priority order and error cases against
synthetic tmp_path bundles.
"""
import json
from pathlib import Path

import pytest

from genie_server import vlm_layout, vlm_specs

FIXTURES = Path(__file__).parent / "data" / "vlm_bundles"


def _load_node_cfgs(model_root: Path, node_config_files: dict) -> dict:
    """The shape vlm.py hands to a family's bind()/detect(): {role: parsed
    JSON}, paths resolved but otherwise unmodified (no HTP pinning, no
    max-num-tokens patch — irrelevant to layout/family resolution)."""
    out = {}
    for role, rel_path in node_config_files.items():
        with open(model_root / rel_path) as f:
            out[role] = json.load(f)
    return out


def _tokenizer_json(model_root: Path) -> dict:
    with open(model_root / "tokenizer.json") as f:
        return json.load(f)


# ---------------------------------------------------------------- AI Hub (metadata.json path)

def test_ai_hub_layout_comes_from_metadata_pipeline():
    layout = vlm_layout.read_layout(FIXTURES / "ai_hub")
    assert layout.source == "metadata.json genie.pipeline"
    assert layout.node_config_files == {
        "image_encoder": "img-enc-htp.json",
        "text_encoder": "text-encoder.json",
        "text_generator": "text-generator.json",
    }
    assert len(layout.connections) == 3
    assert ("image_encoder", "WILDCARD", "text_generator", "WILDCARD") in layout.connections
    assert set(layout.static_tensor_files) == {
        "IMAGE_ENCODER_IMAGE_POS_COS", "IMAGE_ENCODER_IMAGE_POS_SIN",
        "IMAGE_ENCODER_IMAGE_FULL_ATTN_MASK", "IMAGE_ENCODER_IMAGE_WINDOW_ATTN_MASK",
    }
    # request-dependent IO never becomes a static tensor
    assert "TEXT_ENCODER_TEXT_INPUT" not in layout.static_tensor_files
    assert "IMAGE_ENCODER_IMAGE_INPUT" not in layout.static_tensor_files


def test_ai_hub_resolves_to_the_known_qwen3_vl_parameters():
    model_root = FIXTURES / "ai_hub"
    layout = vlm_layout.read_layout(model_root)
    node_cfgs = _load_node_cfgs(model_root, layout.node_config_files)
    spec = vlm_specs.resolve(vlm_specs.get_family("qwen3_vl"), layout, node_cfgs)

    assert (spec.image_width, spec.image_height) == (512, 512)
    assert spec.patch_size == 16
    assert spec.spatial_merge_size == 2
    assert spec.temporal_patch_size == 2
    assert spec.normalize_mean == (0.5, 0.5, 0.5)
    assert spec.normalize_std == (0.5, 0.5, 0.5)
    assert spec.vision_tokens_per_step == 256


def test_ai_hub_family_auto_detects_as_qwen3_vl():
    model_root = FIXTURES / "ai_hub"
    layout = vlm_layout.read_layout(model_root)
    node_cfgs = _load_node_cfgs(model_root, layout.node_config_files)
    family = vlm_specs.detect_family(_tokenizer_json(model_root), node_cfgs)
    assert family.name == "qwen3_vl"


# ---------------------------------------------------------------- DeepStack tutorial (script path)

def test_deepstack_layout_comes_from_the_named_script():
    layout = vlm_layout.read_layout(FIXTURES / "deepstack")
    assert layout.source == "script VLMScript_base"
    assert layout.node_config_files == {
        "image_encoder": "image_encoder.json",
        "text_encoder": "text_encoder.json",
        "text_generator": "text_decoder_vlm_base.json",
    }
    assert ("image_encoder", "WILDCARD", "text_generator", "WILDCARD") in layout.connections
    # no "node set embedding" lines in this script -> no static tensors
    assert layout.static_tensor_files == {}


def test_deepstack_resolves_resolution_from_vision_param():
    model_root = FIXTURES / "deepstack"
    layout = vlm_layout.read_layout(model_root)
    node_cfgs = _load_node_cfgs(model_root, layout.node_config_files)
    spec = vlm_specs.resolve(vlm_specs.get_family("qwen3_vl"), layout, node_cfgs)

    assert (spec.image_width, spec.image_height) == (34 * 16, 34 * 16)
    assert spec.spatial_merge_size == 2
    # normalization/patch/temporal are not stated by this bundle -> family defaults
    assert spec.normalize_mean == (0.5, 0.5, 0.5)
    assert spec.temporal_patch_size == 2


def test_deepstack_family_auto_detects_as_qwen3_vl():
    model_root = FIXTURES / "deepstack"
    layout = vlm_layout.read_layout(model_root)
    node_cfgs = _load_node_cfgs(model_root, layout.node_config_files)
    family = vlm_specs.detect_family(_tokenizer_json(model_root), node_cfgs)
    assert family.name == "qwen3_vl"


def test_qwen3_vl_deepstack_alias_still_selects_the_qwen3_vl_family():
    assert vlm_specs.get_family("qwen3_vl_deepstack") is vlm_specs.get_family("qwen3_vl")


# ---------------------------------------------------------------- 2B LMM bundle (script path)

def test_2b_layout_comes_from_the_named_script():
    layout = vlm_layout.read_layout(FIXTURES / "qwen3vl_2b")
    assert layout.source == "script LMMScript"
    assert layout.node_config_files == {
        "image_encoder": "image_encoder.json",
        "text_encoder": "text_encoder.json",
        "text_generator": "text_decoder.json",
    }
    assert layout.static_tensor_files == {}


def test_2b_resolves_resolution_from_vision_param():
    model_root = FIXTURES / "qwen3vl_2b"
    layout = vlm_layout.read_layout(model_root)
    node_cfgs = _load_node_cfgs(model_root, layout.node_config_files)
    spec = vlm_specs.resolve(vlm_specs.get_family("qwen3_vl"), layout, node_cfgs)
    assert (spec.image_width, spec.image_height) == (40 * 16, 30 * 16)


# ---------------------------------------------------------------- Gemma 4 E2B (script path)

def test_gemma4_layout_comes_from_the_named_script():
    layout = vlm_layout.read_layout(FIXTURES / "gemma4")
    assert layout.source == "script genie_app_image.txt"
    assert layout.node_config_files == {
        "image_encoder": "image-encoder.json",
        "text_encoder": "text-encoder.json",
        "text_generator": "text-generator.json",
    }
    # this script has no WILDCARD connect line
    assert ("image_encoder", "WILDCARD", "text_generator", "WILDCARD") not in layout.connections
    assert len(layout.connections) == 2
    assert layout.static_tensor_files == {}


def test_gemma4_resolves_the_grid_from_vision_param():
    model_root = FIXTURES / "gemma4"
    layout = vlm_layout.read_layout(model_root)
    node_cfgs = _load_node_cfgs(model_root, layout.node_config_files)
    spec = vlm_specs.resolve(vlm_specs.get_family("gemma4"), layout, node_cfgs)
    assert (spec.image_height, spec.image_width) == (39 * 16, 60 * 16)
    assert spec.spatial_merge_size == 3
    assert spec.vision_tokens_per_step == 260


def test_gemma4_family_auto_detects():
    model_root = FIXTURES / "gemma4"
    layout = vlm_layout.read_layout(model_root)
    node_cfgs = _load_node_cfgs(model_root, layout.node_config_files)
    family = vlm_specs.detect_family(_tokenizer_json(model_root), node_cfgs)
    assert family.name == "gemma4"


# ---------------------------------------------------------------- GenieX rejection

def test_genie_x_dialog_only_bundle_is_rejected_with_a_clear_reason():
    with pytest.raises(ValueError, match="GenieX"):
        vlm_layout.read_layout(FIXTURES / "genie_x")


def test_a_bundle_with_nothing_recognizable_gets_a_generic_error(tmp_path):
    (tmp_path / "readme.txt").write_text("not a genie-app script")
    with pytest.raises(ValueError, match="could not determine"):
        vlm_layout.read_layout(tmp_path)


# ---------------------------------------------------------------- priority order / escape hatches

def test_explicit_node_configs_skip_layout_detection(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"image-encoder": {}}))
    (tmp_path / "b.json").write_text(json.dumps({"text-encoder": {}}))
    (tmp_path / "c.json").write_text(json.dumps({"text-generator": {}}))
    layout = vlm_layout.read_layout(
        tmp_path, node_configs={"image_encoder": "a.json", "text_encoder": "b.json",
                                "text_generator": "c.json"})
    assert layout.source == "explicit VLM_SLOTS[].node_configs"
    assert layout.node_config_files["image_encoder"] == "a.json"
    assert ("image_encoder", "WILDCARD", "text_generator", "WILDCARD") in layout.connections


def test_explicit_pipeline_script_names_which_file_to_parse(tmp_path):
    """Even with a metadata.json genie.pipeline present, an explicit
    pipeline_script wins — the escape hatch is for when auto-detection picks
    the wrong thing, so it has to be able to override it."""
    (tmp_path / "metadata.json").write_text(json.dumps(
        {"genie": {"pipeline": {"nodes": {}, "connections": []}}}))
    (tmp_path / "img.json").write_text(json.dumps({"image-encoder": {}}))
    (tmp_path / "txt_enc.json").write_text(json.dumps({"text-encoder": {}}))
    (tmp_path / "txt_gen.json").write_text(json.dumps({"text-generator": {}}))
    (tmp_path / "custom_script.txt").write_text(
        "version\n"
        "node config create c1 img.json\nnode create imageEncoder c1\n"
        "node config create c2 txt_enc.json\nnode create lutEncoder c2\n"
        "node config create c3 txt_gen.json\nnode create textGenerator c3\n"
        "pipeline connect GeniePipeline imageEncoder GENIE_NODE_IMAGE_ENCODER_EMBEDDING_OUTPUT "
        "textGenerator GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT\n"
        "pipeline connect GeniePipeline lutEncoder GENIE_NODE_TEXT_ENCODER_EMBEDDING_OUTPUT "
        "textGenerator GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT\n"
        "pipeline create GeniePipeline pipelineConfig\n")
    layout = vlm_layout.read_layout(tmp_path, pipeline_script="custom_script.txt")
    assert layout.source == "script custom_script.txt"
    assert layout.node_config_files["image_encoder"] == "img.json"


def test_static_tensors_override_applies_regardless_of_source(tmp_path):
    (tmp_path / "img.json").write_text(json.dumps({"image-encoder": {}}))
    (tmp_path / "b.json").write_text(json.dumps({"text-encoder": {}}))
    (tmp_path / "c.json").write_text(json.dumps({"text-generator": {}}))
    layout = vlm_layout.read_layout(
        tmp_path,
        node_configs={"image_encoder": "img.json", "text_encoder": "b.json",
                     "text_generator": "c.json"},
        static_tensors={"IMAGE_ENCODER_IMAGE_POS_COS": "cos.raw"})
    assert layout.static_tensor_files == {"IMAGE_ENCODER_IMAGE_POS_COS": "cos.raw"}


def test_static_tensors_override_strips_a_genie_node_prefix(tmp_path):
    """An operator copying a name straight out of a genie-app script
    ('node set embedding ... GENIE_NODE_IMAGE_ENCODER_IMAGE_POS_COS ...')
    still resolves to the right IO — matching what the script parser itself
    does — instead of silently storing an unusable key."""
    (tmp_path / "img.json").write_text(json.dumps({"image-encoder": {}}))
    (tmp_path / "b.json").write_text(json.dumps({"text-encoder": {}}))
    (tmp_path / "c.json").write_text(json.dumps({"text-generator": {}}))
    layout = vlm_layout.read_layout(
        tmp_path,
        node_configs={"image_encoder": "img.json", "text_encoder": "b.json",
                     "text_generator": "c.json"},
        static_tensors={"GENIE_NODE_IMAGE_ENCODER_IMAGE_POS_COS": "cos.raw"})
    assert layout.static_tensor_files == {"IMAGE_ENCODER_IMAGE_POS_COS": "cos.raw"}


def test_static_tensors_override_rejects_an_io_the_image_encoder_cannot_take(tmp_path):
    """A name that is not a real image-encoder auxiliary IO (wrong node, a
    typo, or an output name) must fail at startup — not with a KeyError from
    genie_node.NODE_IO on the first request that reaches the image encoder."""
    (tmp_path / "img.json").write_text(json.dumps({"image-encoder": {}}))
    (tmp_path / "b.json").write_text(json.dumps({"text-encoder": {}}))
    (tmp_path / "c.json").write_text(json.dumps({"text-generator": {}}))
    for bad_io in ("TEXT_ENCODER_EMBEDDING_OUTPUT",   # belongs to the wrong node
                  "IMAGE_ENCODER_EMBEDDING_OUTPUT",   # the node's own output
                  "IMAGE_ENCODER_IMAGE_POS_CSO"):     # typo
        with pytest.raises(vlm_layout.LayoutError, match="invalid static tensor"):
            vlm_layout.read_layout(
                tmp_path,
                node_configs={"image_encoder": "img.json", "text_encoder": "b.json",
                             "text_generator": "c.json"},
                static_tensors={bad_io: "x.raw"})


def test_metadata_sample_inputs_only_take_image_encoder_entries():
    """A sample_inputs entry that names the text-encoder or text-generator
    node must never end up in static_tensor_files — vlm.py only ever feeds
    these to the image encoder, so a wrongly-attributed entry would be set
    on the wrong node instead of being dropped."""
    model_root = FIXTURES / "ai_hub"
    metadata = json.loads((model_root / "metadata.json").read_text())
    # Same IO name as the genuine imageEncoder entry, but attributed to
    # lutEncoder and pointing at a different file — appended last, so
    # without the node-role filter it would clobber the correct value.
    metadata["genie"]["sample_inputs"].append({
        "node": "lutEncoder", "node_io": "GENIE_NODE_IMAGE_ENCODER_IMAGE_POS_SIN",
        "file": "bogus.raw"})
    from genie_server.vlm_layout import _layout_from_metadata_pipeline
    layout = _layout_from_metadata_pipeline(model_root, metadata)
    assert layout.static_tensor_files["IMAGE_ENCODER_IMAGE_POS_SIN"] == \
        "sample_inputs/position_ids_sin.raw"
    assert len(layout.static_tensor_files) == 4


def test_script_embedding_lines_only_take_image_encoder_lines(tmp_path):
    """A 'node set embedding' line naming the text-encoder or text-generator
    alias must never end up in static_tensor_files, for the same reason as
    the metadata.json case above."""
    (tmp_path / "img.json").write_text(json.dumps({"image-encoder": {}}))
    (tmp_path / "txt.json").write_text(json.dumps({"text-encoder": {}}))
    (tmp_path / "gen.json").write_text(json.dumps({"text-generator": {}}))
    (tmp_path / "s.txt").write_text(
        "version\n"
        "node config create c1 img.json\nnode create imageEncoder c1\n"
        "node config create c2 txt.json\nnode create lutEncoder c2\n"
        "node config create c3 gen.json\nnode create textGenerator c3\n"
        "node set embedding imageEncoder GENIE_NODE_IMAGE_ENCODER_IMAGE_POS_COS cos.raw\n"
        "node set embedding lutEncoder GENIE_NODE_IMAGE_ENCODER_IMAGE_POS_SIN sin.raw\n"
        "pipeline create GeniePipeline pipelineConfig\n")
    layout = vlm_layout.read_layout(tmp_path, pipeline_script="s.txt")
    assert layout.static_tensor_files == {"IMAGE_ENCODER_IMAGE_POS_COS": "cos.raw"}


def test_generic_script_sniff_finds_an_unconventionally_named_script(tmp_path):
    (tmp_path / "image-encoder.json").write_text(json.dumps({"image-encoder": {}}))
    (tmp_path / "text-encoder.json").write_text(json.dumps({"text-encoder": {}}))
    (tmp_path / "text-generator.json").write_text(json.dumps({"text-generator": {}}))
    (tmp_path / "run_this.sh").write_text(
        "version\n"
        "node config create c1 image-encoder.json\nnode create imageEncoder c1\n"
        "node config create c2 text-encoder.json\nnode create lutEncoder c2\n"
        "node config create c3 text-generator.json\nnode create textGenerator c3\n"
        "pipeline connect GeniePipeline imageEncoder GENIE_NODE_IMAGE_ENCODER_EMBEDDING_OUTPUT "
        "textGenerator GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT\n"
        "pipeline connect GeniePipeline lutEncoder GENIE_NODE_TEXT_ENCODER_EMBEDDING_OUTPUT "
        "textGenerator GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT\n"
        "pipeline create GeniePipeline pipelineConfig\npipeline execute GeniePipeline\n")
    layout = vlm_layout.read_layout(tmp_path)
    assert layout.source == "script run_this.sh"


def test_legacy_fallback_ai_hub_style_keeps_wildcard_and_sample_inputs(tmp_path):
    (tmp_path / "img-enc-htp.json").write_text(json.dumps({"image-encoder": {}}))
    (tmp_path / "text-encoder.json").write_text(json.dumps({"text-encoder": {}}))
    (tmp_path / "text-generator.json").write_text(json.dumps({"text-generator": {}}))
    (tmp_path / "sample_inputs").mkdir()
    (tmp_path / "sample_inputs" / "position_ids_cos.raw").write_bytes(b"")
    layout = vlm_layout.read_layout(tmp_path)
    assert layout.source == "legacy fallback (img-enc-htp.json)"
    assert ("image_encoder", "WILDCARD", "text_generator", "WILDCARD") in layout.connections
    assert layout.static_tensor_files == {
        "IMAGE_ENCODER_IMAGE_POS_COS": "sample_inputs/position_ids_cos.raw"}


def test_legacy_fallback_gemma4_style_has_no_wildcard_or_static_tensors(tmp_path):
    (tmp_path / "image-encoder.json").write_text(json.dumps({"image-encoder": {}}))
    (tmp_path / "text-encoder.json").write_text(json.dumps({"text-encoder": {}}))
    (tmp_path / "text-generator.json").write_text(json.dumps({"text-generator": {}}))
    layout = vlm_layout.read_layout(tmp_path)
    assert layout.source == "legacy fallback (image-encoder.json)"
    assert len(layout.connections) == 2
    assert layout.static_tensor_files == {}


# ---------------------------------------------------------------- validation errors

def test_a_missing_role_is_a_startup_error(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"image-encoder": {}}))
    (tmp_path / "b.json").write_text(json.dumps({"text-encoder": {}}))
    with pytest.raises(ValueError, match="missing"):
        vlm_layout.read_layout(
            tmp_path, node_configs={"image_encoder": "a.json", "text_encoder": "b.json"})


def test_two_aliases_claiming_the_same_role_is_ambiguous(tmp_path):
    (tmp_path / "img1.json").write_text(json.dumps({"image-encoder": {}}))
    (tmp_path / "img2.json").write_text(json.dumps({"image-encoder": {}}))
    (tmp_path / "b.json").write_text(json.dumps({"text-encoder": {}}))
    (tmp_path / "c.json").write_text(json.dumps({"text-generator": {}}))
    script = tmp_path / "s.txt"
    script.write_text(
        "version\n"
        "node config create c1 img1.json\nnode create imageEncoder1 c1\n"
        "node config create c2 img2.json\nnode create imageEncoder2 c2\n"
        "node config create c3 b.json\nnode create lutEncoder c3\n"
        "node config create c4 c.json\nnode create textGenerator c4\n"
        "pipeline create GeniePipeline pipelineConfig\n")
    with pytest.raises(vlm_layout.LayoutError, match="ambiguous"):
        vlm_layout.read_layout(tmp_path, pipeline_script="s.txt")


def test_an_unknown_role_key_in_an_explicit_override_is_rejected(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"image-encoder": {}}))
    with pytest.raises(ValueError, match="unknown VLM node role"):
        vlm_layout.read_layout(tmp_path, node_configs={"video_encoder": "a.json"})


def test_a_top_level_key_mismatch_in_an_explicit_override_is_rejected(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"text-encoder": {}}))
    (tmp_path / "b.json").write_text(json.dumps({"text-encoder": {}}))
    (tmp_path / "c.json").write_text(json.dumps({"text-generator": {}}))
    with pytest.raises(ValueError, match="does not match role"):
        vlm_layout.read_layout(
            tmp_path, node_configs={"image_encoder": "a.json", "text_encoder": "b.json",
                                    "text_generator": "c.json"})


# ---------------------------------------------------------------- qwen3_vl bind errors

def test_qwen3vl_bind_refuses_a_merge_size_that_does_not_divide_the_grid():
    from genie_server.vlm_specs.qwen3_vl import qwen3vl_bind

    spec = vlm_specs.get_spec("qwen3_vl")
    node_cfgs = {
        "image_encoder": {"image-encoder": {"engine": {"model": {
            "vision-param": {"height": 3, "width": 3},
            "positional-encoding": {"rope-scaling": {"spatial-merge-size": 2}}}}}},
        "text_generator": {"text-generator": {}},
    }
    layout = vlm_layout.BundleLayout(node_config_files={}, connections=[],
                                     static_tensor_files={}, source="test")
    with pytest.raises(ValueError, match="divisible"):
        qwen3vl_bind(spec, node_cfgs, layout)


def test_qwen3vl_bind_catches_a_lopsided_grid_the_product_check_would_miss():
    """27x36 patches at merge 2: the product (972) is divisible by merge**2
    (4), so a check on the product alone would wrongly accept it — but
    _qwen3vl_patchify reshapes grid_h and grid_w separately, and grid_h=27
    is not divisible by 2 on its own. Must fail."""
    from genie_server.vlm_specs.qwen3_vl import qwen3vl_bind

    spec = vlm_specs.get_spec("qwen3_vl")
    node_cfgs = {
        "image_encoder": {"image-encoder": {"engine": {"model": {
            "vision-param": {"height": 27, "width": 36},
            "positional-encoding": {"rope-scaling": {"spatial-merge-size": 2}}}}}},
        "text_generator": {"text-generator": {}},
    }
    layout = vlm_layout.BundleLayout(node_config_files={}, connections=[],
                                     static_tensor_files={}, source="test")
    assert (27 * 16 * 36 * 16 // (16 * 16)) % (2 ** 2) == 0    # the product check would pass
    with pytest.raises(ValueError, match="divisible"):
        qwen3vl_bind(spec, node_cfgs, layout)


def test_qwen3vl_bind_falls_back_to_metadata_vision_preprocessing():
    from genie_server.vlm_specs.qwen3_vl import qwen3vl_bind

    spec = vlm_specs.get_spec("qwen3_vl")
    node_cfgs = {
        "image_encoder": {"image-encoder": {"engine": {"model": {}}}},
        "text_generator": {"text-generator": {}},
    }
    layout = vlm_layout.BundleLayout(
        node_config_files={}, connections=[], static_tensor_files={}, source="test",
        metadata={"genie": {"vision_preprocessing": {
            "image_width": 384, "image_height": 384, "patch_size": 16,
            "temporal_patch_size": 2, "spatial_merge_size": 2,
            "normalize_mean": [0.5, 0.5, 0.5], "normalize_std": [0.5, 0.5, 0.5]}}})
    resolved = qwen3vl_bind(spec, node_cfgs, layout)
    assert (resolved.image_width, resolved.image_height) == (384, 384)


# ---------------------------------------------------------------- family auto-detection edge cases

def test_ambiguous_family_detection_names_both_matches():
    tokenizer_json = {"added_tokens": [{"content": "<|vision_start|>"}, {"content": "<|image>"}]}
    node_cfgs = {"image_encoder": {"image-encoder": {}}, "text_generator": {"text-generator": {}}}
    with pytest.raises(ValueError, match="ambiguous"):
        vlm_specs.detect_family(tokenizer_json, node_cfgs)


def test_no_family_matches_names_the_registered_ones():
    with pytest.raises(ValueError, match="qwen3_vl"):
        vlm_specs.detect_family({"added_tokens": []}, {"image_encoder": {"image-encoder": {}},
                                                       "text_generator": {"text-generator": {}}})


# ---------------------------------------------------------------- resolve_static_tensors

def test_positional_encoding_conflict_drops_the_overlapping_static_tensors(caplog):
    node_cfgs = {"image_encoder": {"image-encoder": {"engine": {"model": {
        "positional-encoding": {"type": "rope"}}}}}}
    static = {"IMAGE_ENCODER_IMAGE_POS_COS": "a.raw", "IMAGE_ENCODER_IMAGE_POS_SIN": "b.raw"}
    with caplog.at_level("WARNING"):
        result = vlm_layout.resolve_static_tensors(node_cfgs, static, "vlm0")
    assert result == {}
    assert "positional-encoding" in caplog.text


def test_no_conflict_when_the_config_has_no_positional_encoding():
    node_cfgs = {"image_encoder": {"image-encoder": {"engine": {"model": {}}}}}
    static = {"IMAGE_ENCODER_IMAGE_POS_COS": "a.raw"}
    assert vlm_layout.resolve_static_tensors(node_cfgs, static, "vlm0") == static


# ---------------------------------------------------------------- VLMSlot builds (fake genie_node)

def _patch_genie_node(monkeypatch):
    pytest.importorskip("numpy")
    from genie_server import genie_node
    from fake_genie import FakeVLMNode, FakeVLMPipeline
    monkeypatch.setattr(genie_node, "Node", FakeVLMNode)
    monkeypatch.setattr(genie_node, "Pipeline", FakeVLMPipeline)


def test_vlm_slot_builds_from_the_ai_hub_layout(monkeypatch, tmp_path):
    """The layout with the most moving parts: WILDCARD, static tensors read
    from disk, and both text nodes naming the same tokenizer."""
    _patch_genie_node(monkeypatch)
    from genie_server import vlm

    slot = vlm.VLMSlot(name="vlm0", device_id=None, model_root=FIXTURES / "ai_hub",
                       spec_name=None, htp_ext_cache_dir=tmp_path / "htpcache")

    assert slot.spec.name == "qwen3_vl"
    assert slot.layout.source == "metadata.json genie.pipeline"
    assert len(slot.pipeline.nodes) == 3
    assert len(slot.pipeline.connections) == 3
    assert set(slot.static_tensors) == {
        "IMAGE_ENCODER_IMAGE_POS_COS", "IMAGE_ENCODER_IMAGE_POS_SIN",
        "IMAGE_ENCODER_IMAGE_FULL_ATTN_MASK", "IMAGE_ENCODER_IMAGE_WINDOW_ATTN_MASK"}
    assert (slot.spec.image_width, slot.spec.image_height) == (512, 512)


def test_vlm_slot_builds_from_the_gemma4_layout_with_auto_detected_family(monkeypatch, tmp_path):
    """No WILDCARD, no static tensors, and spec_name left out — exercises
    detect_family end to end through a real VLMSlot construction."""
    _patch_genie_node(monkeypatch)
    from genie_server import vlm

    slot = vlm.VLMSlot(name="vlm0", device_id=None, model_root=FIXTURES / "gemma4",
                       spec_name=None, htp_ext_cache_dir=tmp_path / "htpcache")

    assert slot.spec.name == "gemma4"
    assert slot.layout.source == "script genie_app_image.txt"
    assert len(slot.pipeline.connections) == 2
    assert slot.static_tensors == {}
    assert (slot.spec.image_height, slot.spec.image_width) == (39 * 16, 60 * 16)


def test_vlm_generation_runs_end_to_end_on_the_ai_hub_layout(monkeypatch, tmp_path):
    """Exercises start_vlm_generation itself, not just VLMSlot construction —
    this is what caught the *_io fields moving to vlm_layout constants
    without vlm.py's three call sites being updated to match (the AttributeError
    only showed up at request time, board-side, never in a construction-only
    test)."""
    _patch_genie_node(monkeypatch)
    import threading
    from genie_server import vlm

    slot = vlm.VLMSlot(name="vlm0", device_id=None, model_root=FIXTURES / "ai_hub",
                       spec_name=None, htp_ext_cache_dir=tmp_path / "htpcache")

    class Params:
        temperature = top_p = top_k = seed = None

    class Generation:
        request_id = "chatcmpl-test"
        completion_tokens = 0
        finish_reason = None
        error = None

        def __init__(self):
            self.done = threading.Event()
            self.aborted = threading.Event()
            self.put_calls = []

        def put_threadsafe(self, item):
            self.put_calls.append(item)

    system_text, parts, sources = "", [("text", "describe"), ("image", 0)], []
    segments = slot.spec.build_prompt_segments(system_text, parts, {}, slot.spec)
    images = [_1x1_image()]

    generation = Generation()
    vlm.start_vlm_generation(None, slot, segments, images, Params(), generation)
    assert generation.done.wait(timeout=5)

    assert generation.error is None, generation.error
    assert slot.pipeline.executed == 1
    # The text-generator's callback IO and the text-encoder/image-encoder
    # input IOs are the vlm_layout constants, not a removed VLMSpec field.
    assert slot.text_generator.text_callback[0] == vlm.vlm_layout.TEXT_GENERATOR_TEXT_OUTPUT_IO
    sent_texts = [io for io, _ in slot.text_encoder.texts]
    assert sent_texts and all(io == vlm.vlm_layout.TEXT_ENCODER_TEXT_INPUT_IO for io in sent_texts)
    assert vlm.vlm_layout.IMAGE_ENCODER_IMAGE_INPUT_IO in slot.image_encoder.buffers



def test_a_vlm_request_abandoned_while_waiting_never_runs(monkeypatch, tmp_path):
    """The composable pipeline has no abort, so a request the caller gave up
    on (a 504, or a client that left) while it queued for the slot must not
    start once it gets the lock: nobody is waiting for its answer."""
    _patch_genie_node(monkeypatch)
    import threading
    import types
    from genie_server import vlm

    slot = vlm.VLMSlot(name="vlm0", device_id=None, model_root=FIXTURES / "ai_hub",
                       spec_name=None, htp_ext_cache_dir=tmp_path / "htpcache")
    generation = types.SimpleNamespace(
        request_id="chatcmpl-abandoned", completion_tokens=0, finish_reason=None,
        error=None, done=threading.Event(), aborted=threading.Event(),
        put_threadsafe=lambda item: None)
    params = types.SimpleNamespace(temperature=None, top_p=None, top_k=None, seed=None)
    segments = slot.spec.build_prompt_segments(
        "", [("text", "describe"), ("image", 0)], {}, slot.spec)

    slot.lock.acquire()                  # another request holds the slot
    vlm.start_vlm_generation(None, slot, segments, [_1x1_image()], params, generation)
    generation.aborted.set()             # the caller gives up while it waits
    slot.lock.release()

    assert generation.done.wait(timeout=5)
    assert slot.pipeline.executed == 0
    assert generation.error is None


def _1x1_image():
    from PIL import Image
    return Image.new("RGB", (1, 1))


def test_vlm_slot_builds_from_the_deepstack_layout(monkeypatch, tmp_path):
    """DeepStack's WILDCARD connection with no static tensors alongside it —
    the shape a positional-encoding image-encoder actually ships."""
    _patch_genie_node(monkeypatch)
    from genie_server import vlm

    slot = vlm.VLMSlot(name="vlm0", device_id=None, model_root=FIXTURES / "deepstack",
                       spec_name="qwen3_vl_deepstack", htp_ext_cache_dir=tmp_path / "htpcache")

    assert slot.spec.name == "qwen3_vl"
    assert len(slot.pipeline.connections) == 3
    assert slot.static_tensors == {}
    assert (slot.spec.image_width, slot.spec.image_height) == (34 * 16, 34 * 16)
