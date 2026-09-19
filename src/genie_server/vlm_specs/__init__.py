"""Per-VLM-model families (preprocessing, node topology defaults, prompt
template) plus the registry that resolves VLM_SLOTS[].spec — or, when it is
not given, auto-detects the family from the bundle itself.

Adding support for a new VLM just means adding a family module in this
package and registering it in FAMILIES below (no changes needed to
vlm_layout.py or vlm.py, which only deal with VLMSpec/VLMFamily's generic
shape). Adding support for a new BUNDLE LAYOUT of an already-supported
model needs no code change at all — see vlm_layout.py's module docstring.
"""
from .base import VLMFamily, VLMSpec, resolve, template
from .gemma4 import GEMMA4_FAMILY
from .qwen3_vl import QWEN3_VL_FAMILY

FAMILIES = {
    "qwen3_vl": QWEN3_VL_FAMILY,
    "gemma4": GEMMA4_FAMILY,
}

# Names VLM_SLOTS[].spec used before bundle layouts were auto-read.
# qwen3_vl_deepstack was its own VLMSpec (identical topology, just a
# different set of node-config filenames and no static tensors) back when
# layouts were hard-coded per name; now that the layout comes from the
# bundle itself, both names just select the Qwen3-VL family.
_ALIASES = {
    "qwen3_vl_deepstack": "qwen3_vl",
}


def get_family(name: str) -> VLMFamily:
    canonical = _ALIASES.get(name, name)
    try:
        return FAMILIES[canonical]
    except KeyError:
        raise KeyError(
            f"Unknown VLM family {name!r}. Registered: {sorted(FAMILIES)} "
            f"(aliases: {sorted(_ALIASES)})"
        ) from None


def detect_family(tokenizer_json: dict, node_cfgs: dict) -> VLMFamily:
    """Auto-detects the family from the bundle's own tokenizer.json and node
    configs, for a VLM_SLOTS[] entry that does not give "spec". Raises
    ValueError when zero or several families match — an ambiguous or
    unrecognized bundle needs an explicit "spec" rather than a guess."""
    matches = [f for f in FAMILIES.values() if f.detect and f.detect(tokenizer_json, node_cfgs)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            "could not auto-detect the VLM family for this bundle; pass "
            f"VLM_SLOTS[].spec explicitly (one of {sorted(FAMILIES)})")
    raise ValueError(
        f"ambiguous VLM family — {sorted(m.name for m in matches)} all "
        f"matched; pass VLM_SLOTS[].spec explicitly")


def get_spec(name: str) -> VLMSpec:
    """A VLMSpec built from nothing but the named family's own defaults — no
    bundle layout, unbound. For a caller that wants a template rather than a
    bundle-derived resolution (tests, mainly); VLMSlot itself always calls
    resolve() with a real bundle's BundleLayout and loaded node configs."""
    return template(get_family(name))
