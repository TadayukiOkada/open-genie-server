"""Reads a VLM bundle's own layout (which node configs to load, how they
connect, which static tensors to feed) instead of hard-coding it per model.

Every pipeline bundle ships a genie-app script, and it says which configs to
load, how to connect them, and which static tensors to feed (`node set
embedding`). The role of each node (image-encoder / text-encoder /
text-generator) is fixed by the config's own top-level key, not by the
node's name in the script — so this module reads that key rather than
trusting a filename or an alias.

Priority order (the first one that produces a layout wins):

  1. An explicit override from VLM_SLOTS[] (`node_configs`, optionally with
     `static_tensors`; or `pipeline_script` naming which script to parse).
     The escape hatch for a layout this module cannot yet read on its own.
  2. The genie-app script:
       a. `metadata.json`'s `genie.pipeline` block, with `genie.sample_inputs`
          as the static tensors (filtered to the ones that are not
          request-dependent — see _REQUEST_DEPENDENT_IO).
       b. A script file by name: `genie-app-script.txt`, `VLMScript*`,
          `LMMScript*`, `genie_app_image.txt`.
       c. Failing those, any small text file directly under the bundle whose
          first line is "version" and that contains "pipeline create".
  3. Legacy fallback: the fixed filenames this server hard-coded before this
     module existed (`img-enc-htp.json` / `image-encoder.json` families),
     for a bundle that ships neither a script nor a metadata.json pipeline.

`node set image` / `node set textFile` lines (and the metadata equivalent,
IMAGE_ENCODER_IMAGE_INPUT / TEXT_ENCODER_TEXT_INPUT) are never read as static
tensors — those are request-dependent, filled in per request by vlm.py.
"""
import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

from .slots import resolve_and_verify

logger = logging.getLogger(__name__)

# GenieNode_IOName_t constants used directly by vlm.py (genie_node.NODE_IO
# has the full set; these three are the ones a VLMSlot writes to regardless
# of family or layout, so they do not need to live on a per-bundle spec).
TEXT_ENCODER_TEXT_INPUT_IO = "TEXT_ENCODER_TEXT_INPUT"
IMAGE_ENCODER_IMAGE_INPUT_IO = "IMAGE_ENCODER_IMAGE_INPUT"
TEXT_GENERATOR_TEXT_OUTPUT_IO = "TEXT_GENERATOR_TEXT_OUTPUT"

# A node config's top-level key decides its role — see the module docstring.
_ROLE_FROM_TOP_KEY = {
    "image-encoder": "image_encoder",
    "text-encoder": "text_encoder",
    "text-generator": "text_generator",
}
_TOP_KEY_FROM_ROLE = {v: k for k, v in _ROLE_FROM_TOP_KEY.items()}

# IO names that are filled in per request (image/text content), never as a
# static tensor read once at slot startup.
_REQUEST_DEPENDENT_IO = {"TEXT_ENCODER_TEXT_INPUT", "IMAGE_ENCODER_IMAGE_INPUT"}

# The image-encoder's auxiliary inputs a positional-encoding config computes
# on the device instead (see resolve_static_tensors).
_IMAGE_AUX_IO = {
    "IMAGE_ENCODER_IMAGE_POS_COS", "IMAGE_ENCODER_IMAGE_POS_SIN",
    "IMAGE_ENCODER_IMAGE_FULL_ATTN_MASK", "IMAGE_ENCODER_IMAGE_WINDOW_ATTN_MASK",
}

# The standard 3-node topology every known bundle uses. WILDCARD is needed
# only when the image-encoder carries extra same-named tensors besides the
# main embedding (e.g. DeepStack's deepstack_visual_embeds_*) — see
# .claude/rules/genie-c-api.md §2. Harmless to include when there is nothing
# extra to match, so this is also the legacy/explicit default connection set.
STANDARD_CONNECTIONS = [
    ("image_encoder", "IMAGE_ENCODER_EMBEDDING_OUTPUT",
     "text_generator", "TEXT_GENERATOR_EMBEDDING_INPUT"),
    ("text_encoder", "TEXT_ENCODER_EMBEDDING_OUTPUT",
     "text_generator", "TEXT_GENERATOR_EMBEDDING_INPUT"),
    ("image_encoder", "WILDCARD", "text_generator", "WILDCARD"),
]

# Known genie-app script filenames/patterns, in the order named bundles use
# them. Exact names first, then the glob patterns qairt_convert's recipes
# use (VLMScript_base, LMMScript — no file extension).
_SCRIPT_EXACT_NAMES = ("genie-app-script.txt", "genie_app_image.txt")
_SCRIPT_GLOB_PATTERNS = ("VLMScript*", "LMMScript*")

# Legacy fallback filenames this server hard-coded before this module
# existed. True = pair with WILDCARD + the AI Hub sample_inputs static
# tensors (the shape that historically needed them); False = Gemma 4's shape
# (no WILDCARD, no static tensors).
_LEGACY_IMAGE_ENCODER_NAMES = {
    "img-enc-htp.json": True,
    "image-encoder.json": False,
}
_LEGACY_STATIC_TENSORS = {
    "IMAGE_ENCODER_IMAGE_POS_COS": "sample_inputs/position_ids_cos.raw",
    "IMAGE_ENCODER_IMAGE_POS_SIN": "sample_inputs/position_ids_sin.raw",
    "IMAGE_ENCODER_IMAGE_FULL_ATTN_MASK": "sample_inputs/full_attention_mask.raw",
    "IMAGE_ENCODER_IMAGE_WINDOW_ATTN_MASK": "sample_inputs/window_attention_mask.raw",
}


@dataclass
class BundleLayout:
    """What VLMSlot needs to build the pipeline, read straight from the
    bundle (or from an explicit VLM_SLOTS[] override)."""
    node_config_files: dict            # {role: path relative to model_root}
    connections: list                  # [(producer_role, io, consumer_role, io), ...]
    static_tensor_files: dict          # {IO name: path relative to model_root}
    source: str                        # human-readable, for logs only
    # metadata.json's full parsed content, if the bundle ships one — read
    # once here so a family's bind() can pull whatever else it needs (e.g.
    # genie.vision_preprocessing) without re-opening the file itself.
    metadata: dict = field(default_factory=dict)


class LayoutError(ValueError):
    """The bundle's layout could not be determined, or is inconsistent."""


def _top_level_key(path: Path) -> str | None:
    try:
        with open(path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return next(iter(cfg), None) if isinstance(cfg, dict) else None


def _strip_io(io_name: str) -> str:
    return io_name.removeprefix("GENIE_NODE_")


def _read_metadata_json(model_root: Path) -> dict:
    path = model_root / "metadata.json"
    if not path.is_file():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _resolve_aliases(alias_to_path: dict, model_root: Path) -> dict:
    """alias -> role, by reading each config's own top-level key. Raises if
    two aliases claim the same role; silently drops an alias whose config is
    neither of the three known roles (nothing in a real script names one) —
    _require_all_roles catches an actually-missing role afterwards."""
    role_of = {}
    alias_of_role = {}
    for alias, rel_path in alias_to_path.items():
        abs_path = resolve_and_verify(rel_path, model_root)
        role = _ROLE_FROM_TOP_KEY.get(_top_level_key(Path(abs_path)))
        if role is None:
            continue
        if role in alias_of_role:
            raise LayoutError(
                f"{model_root}: both {alias_of_role[role]!r} and {alias!r} are "
                f"'{role}' node configs — ambiguous layout")
        alias_of_role[role] = alias
        role_of[alias] = role
    return role_of


def _require_all_roles(node_config_files: dict, model_root: Path) -> None:
    missing = sorted(set(_ROLE_FROM_TOP_KEY.values()) - set(node_config_files))
    if missing:
        raise LayoutError(
            f"{model_root}: layout is missing a node config for role(s) {missing}")


def _validate_role_top_keys(node_config_files: dict, model_root: Path) -> None:
    for role, rel_path in node_config_files.items():
        expected = _TOP_KEY_FROM_ROLE.get(role)
        if expected is None:
            raise LayoutError(
                f"{model_root}: unknown VLM node role {role!r}; expected one "
                f"of {sorted(_TOP_KEY_FROM_ROLE)}")
        abs_path = resolve_and_verify(rel_path, model_root)
        top_key = _top_level_key(Path(abs_path))
        if top_key != expected:
            raise LayoutError(
                f"{rel_path}: top-level key {top_key!r} does not match role "
                f"{role!r} (expected {expected!r})")


# ---------------------------------------------------------------- script parsing

def _parse_script(text: str) -> tuple:
    """genie-app script text -> (alias_to_path, connections, embeddings).

    connections: [(producer_alias, io, consumer_alias, io), ...]
    embeddings: {node_alias: {io: path}} — the "node set embedding" lines
    only; "node set image"/"node set textFile" are deliberately not parsed
    (request-dependent, see the module docstring).
    """
    config_path = {}     # configAlias -> path
    node_config = {}      # nodeAlias -> configAlias
    connections = []
    embeddings: dict = {}

    for raw_line in text.splitlines():
        parts = raw_line.split()
        if not parts or parts[0].startswith("#"):
            continue
        if parts[:2] == ["node", "config"] and len(parts) >= 5 and parts[2] == "create":
            config_path[parts[3]] = parts[4]
        elif parts[:2] == ["node", "create"] and len(parts) >= 4:
            node_config[parts[2]] = parts[3]
        elif parts[:2] == ["pipeline", "connect"] and len(parts) >= 7:
            _, _, _pipeline, producer, pio, consumer, cio = parts[:7]
            connections.append((producer, _strip_io(pio), consumer, _strip_io(cio)))
        elif parts[:3] == ["node", "set", "embedding"] and len(parts) >= 6:
            _, _, _, node_alias, io, path = parts[:6]
            embeddings.setdefault(node_alias, {})[_strip_io(io)] = path

    alias_to_path = {
        node_alias: config_path[cfg_alias]
        for node_alias, cfg_alias in node_config.items()
        if cfg_alias in config_path
    }
    return alias_to_path, connections, embeddings


def _find_script_by_name(model_root: Path) -> Path | None:
    for name in _SCRIPT_EXACT_NAMES:
        p = model_root / name
        if p.is_file():
            return p
    for pattern in _SCRIPT_GLOB_PATTERNS:
        matches = sorted(model_root.glob(pattern))
        if matches:
            return matches[0]
    return None


def _find_script_generic(model_root: Path) -> Path | None:
    """A small text file directly under the bundle whose first line is
    "version" and that contains "pipeline create" — the last resort before
    falling back to fixed filenames."""
    for p in sorted(model_root.iterdir()):
        if not p.is_file() or p.stat().st_size > 1_000_000:
            continue
        try:
            text = p.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        lines = text.splitlines()
        if lines and lines[0].strip() == "version" and "pipeline create" in text:
            return p
    return None


def _layout_from_script(model_root: Path, script_path: Path) -> BundleLayout:
    alias_to_path, connections_alias, embeddings_alias = _parse_script(
        script_path.read_text())
    alias_to_role = _resolve_aliases(alias_to_path, model_root)
    node_config_files = {role: alias_to_path[alias] for alias, role in alias_to_role.items()}
    _require_all_roles(node_config_files, model_root)

    connections = [
        (alias_to_role[p], pio, alias_to_role[c], cio)
        for p, pio, c, cio in connections_alias
        if p in alias_to_role and c in alias_to_role
    ]
    static_tensor_files: dict = {}
    for iomap in embeddings_alias.values():
        static_tensor_files.update(iomap)

    return BundleLayout(node_config_files=node_config_files, connections=connections,
                        static_tensor_files=static_tensor_files,
                        source=f"script {script_path.name}")


# ---------------------------------------------------------------- metadata.json parsing

def _layout_from_metadata_pipeline(model_root: Path, metadata: dict) -> BundleLayout | None:
    genie_block = metadata.get("genie")
    if not isinstance(genie_block, dict):
        return None
    pipeline = genie_block.get("pipeline")
    if not isinstance(pipeline, dict) or not pipeline.get("nodes"):
        return None

    alias_to_path = dict(pipeline["nodes"])
    alias_to_role = _resolve_aliases(alias_to_path, model_root)
    node_config_files = {role: alias_to_path[alias] for alias, role in alias_to_role.items()}
    _require_all_roles(node_config_files, model_root)

    connections = []
    for c in pipeline.get("connections") or []:
        producer, consumer = c.get("producer_node"), c.get("consumer_node")
        if producer not in alias_to_role or consumer not in alias_to_role:
            continue
        connections.append((
            alias_to_role[producer], _strip_io(c.get("producer_node_io", "")),
            alias_to_role[consumer], _strip_io(c.get("consumer_node_io", "")),
        ))

    static_tensor_files = {}
    for entry in genie_block.get("sample_inputs") or []:
        io = _strip_io(entry.get("node_io", ""))
        if io and io not in _REQUEST_DEPENDENT_IO and entry.get("file"):
            static_tensor_files[io] = entry["file"]

    return BundleLayout(node_config_files=node_config_files, connections=connections,
                        static_tensor_files=static_tensor_files,
                        source="metadata.json genie.pipeline")


# ---------------------------------------------------------------- legacy / explicit

def _layout_from_legacy(model_root: Path) -> BundleLayout | None:
    for image_name, wildcard in _LEGACY_IMAGE_ENCODER_NAMES.items():
        text_enc, text_gen = "text-encoder.json", "text-generator.json"
        if not all((model_root / n).is_file() for n in (image_name, text_enc, text_gen)):
            continue
        node_config_files = {
            "image_encoder": image_name, "text_encoder": text_enc, "text_generator": text_gen,
        }
        connections = list(STANDARD_CONNECTIONS if wildcard else STANDARD_CONNECTIONS[:2])
        static_tensor_files = {}
        if wildcard:
            static_tensor_files = {
                io: p for io, p in _LEGACY_STATIC_TENSORS.items()
                if (model_root / p).is_file()
            }
        return BundleLayout(node_config_files=node_config_files, connections=connections,
                            static_tensor_files=static_tensor_files,
                            source=f"legacy fallback ({image_name})")
    return None


def _layout_from_explicit_node_configs(model_root: Path, node_configs: dict) -> BundleLayout:
    unknown = sorted(set(node_configs) - set(_TOP_KEY_FROM_ROLE))
    if unknown:
        raise LayoutError(
            f"{model_root}: unknown VLM node role(s) {unknown}; expected a "
            f"subset of {sorted(_TOP_KEY_FROM_ROLE)}")
    _require_all_roles(dict(node_configs), model_root)
    return BundleLayout(node_config_files=dict(node_configs),
                        connections=list(STANDARD_CONNECTIONS),
                        static_tensor_files={},
                        source="explicit VLM_SLOTS[].node_configs")


def _raise_no_layout(model_root: Path) -> None:
    for path in sorted(model_root.glob("*.json")):
        if _top_level_key(path) == "dialog":
            raise LayoutError(
                f"{model_root}: no VLM node configs found, only a dialog "
                f"config ({path.name}). That is the GenieX pipeline format "
                "(one dialog config, no image-encoder/text-encoder/"
                "text-generator node configs), which this server does not "
                "support for VLM bundles — see PLATFORM_NOTES.")
    raise LayoutError(
        f"{model_root}: could not determine a VLM bundle layout — no "
        "genie-app pipeline script, no metadata.json genie.pipeline, and "
        "none of the known node-config filenames. Pass "
        "VLM_SLOTS[].node_configs to set the layout explicitly.")


# ---------------------------------------------------------------- entry point

def read_layout(model_root: Path, *, pipeline_script: str | None = None,
                node_configs: dict | None = None,
                static_tensors: dict | None = None) -> BundleLayout:
    """Reads one VLM bundle's layout. See the module docstring for the
    priority order. Raises LayoutError (a ValueError) on anything this
    server cannot make sense of, including the GenieX dialog-only shape."""
    metadata = _read_metadata_json(model_root)

    if node_configs:
        layout = _layout_from_explicit_node_configs(model_root, node_configs)
    elif pipeline_script:
        script_path = model_root / pipeline_script
        if not script_path.is_file():
            raise LayoutError(f"VLM_SLOTS[].pipeline_script not found: {script_path}")
        layout = _layout_from_script(model_root, script_path)
    else:
        layout = _layout_from_metadata_pipeline(model_root, metadata)
        if layout is None:
            script_path = _find_script_by_name(model_root) or _find_script_generic(model_root)
            if script_path is not None:
                layout = _layout_from_script(model_root, script_path)
        if layout is None:
            layout = _layout_from_legacy(model_root)
        if layout is None:
            _raise_no_layout(model_root)

    if static_tensors:
        layout = replace(layout, static_tensor_files=dict(static_tensors))
    layout = replace(layout, metadata=metadata)
    _validate_role_top_keys(layout.node_config_files, model_root)
    logger.info(f"[{model_root.name}] VLM bundle layout: {layout.source}")
    return layout


def resolve_static_tensors(node_cfgs: dict, static_tensor_files: dict, slot_name: str) -> dict:
    """Drops the image-encoder's auxiliary tensors when its own config also
    carries positional-encoding — nsp-image-model.cpp computes position ids
    and attention masks on the device in that case
    (.claude/rules/vision-preprocessing.md), so feeding stale ones from
    another export would silently be ignored at best. The config wins."""
    image_cfg = node_cfgs.get("image_encoder")
    if not image_cfg:
        return static_tensor_files
    inner = next(iter(image_cfg.values()), {})
    has_pos_enc = bool(inner.get("engine", {}).get("model", {}).get("positional-encoding"))
    overlap = _IMAGE_AUX_IO & set(static_tensor_files)
    if not (has_pos_enc and overlap):
        return static_tensor_files
    logger.warning(
        f"[{slot_name}] image-encoder config has positional-encoding, but the "
        f"layout also names static tensor(s) {sorted(overlap)} — the device "
        "computes these itself, so they are not passed. Drop "
        "positional-encoding from the config to feed them instead.")
    return {io: p for io, p in static_tensor_files.items() if io not in overlap}
