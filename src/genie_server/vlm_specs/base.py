"""VLMFamily / VLMSpec: the model-specific half of VLM support, paired with
vlm_layout.py (the generic, bundle-read half) and genie_node.py (the generic
plumbing layer) — the same split as GenieX's `core/` vs `models/*.h`
pattern.

A VLMFamily is what a model needs regardless of which bundle ships it:
preprocessing, chat template, and defaults for the parameters a bundle may
or may not state explicitly (patch size, spatial merge, temporal packing,
normalization, the encoder's fixed row budget). A VLMSpec is the resolved
combination of one VLMFamily and one vlm_layout.BundleLayout — what VLMSlot
actually builds nodes and runs preprocessing from.

Adding support for a new VLM model means adding one family module here and
registering it in __init__.py's FAMILIES — no changes needed to vlm_layout.py
or vlm.py. Adding support for a new BUNDLE LAYOUT of an already-supported
model needs no code change at all, per vlm_layout.py's module docstring.
"""
from dataclasses import dataclass, field, replace
from typing import Callable


@dataclass
class VLMFamily:
    """The model-specific layer: chat template, preprocessing, and the
    defaults a bundle's own config may override (see each family module's
    bind() for which fields it actually reads from the bundle)."""
    name: str

    # (system_text, parts, video_meta, spec) -> list[("text", str) | ("step", payload)]
    # See VLMSpec.build_prompt_segments for the full contract — it is the
    # same function, just stored here before a layout is known.
    build_prompt_segments: Callable = field(repr=False)

    # (images, payload, spec) -> pixel_values ndarray for one step. See
    # VLMSpec.preprocess_step.
    preprocess_step: Callable = field(repr=False)

    # (spec, node_cfgs, layout) -> VLMSpec, or None. Called once per slot
    # with every node config loaded (paths already resolved) and the
    # bundle's BundleLayout (including its parsed metadata.json, if any),
    # before any node is created. Adjusts the template spec's preprocessing
    # parameters to what this particular bundle was actually exported with.
    bind: Callable | None = field(default=None, repr=False)

    # (tokenizer_json, node_cfgs) -> bool. Used only when VLM_SLOTS[].spec is
    # not given — see vlm_specs/__init__.py's detect_family.
    detect: Callable | None = field(default=None, repr=False)

    # Defaults, used as-is unless bind() overrides them from the bundle.
    image_width: int = 0
    image_height: int = 0
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 1
    normalize_mean: tuple = (0.0, 0.0, 0.0)
    normalize_std: tuple = (1.0, 1.0, 1.0)
    # Rows pixel_values is zero-padded to, for an encoder exported with a
    # fixed maximum and told at load time how many of them are real
    # (Gemma 4). 0 = the input is exactly the resolution's own patch count.
    max_patches: int = 0


@dataclass
class VLMSpec:
    """One VLMFamily resolved against one bundle's vlm_layout.BundleLayout —
    what VLMSlot actually builds from. See vlm_specs/base.py's VLMFamily for
    what varies by model and vlm_layout.py for what varies by bundle."""
    name: str

    # Paths relative to the model directory root, from the bundle's
    # BundleLayout. Passed straight to genie_node.Node(...).
    node_config_files: dict          # {"image_encoder": ..., "text_encoder": ..., "text_generator": ...}

    # [(producer_role, producer_io, consumer_role, consumer_io), ...], from
    # the BundleLayout. Each role is a key in node_config_files.
    connections: list

    # Auxiliary tensors that don't depend on image content (positional
    # encoding / attention masks), from the BundleLayout. {IO name: path
    # relative to the model directory}. Assumes a fixed resolution — loaded
    # once at startup and reused across every subsequent request.
    static_tensor_files: dict

    # Preprocessing parameters, resolved by the family's bind() from
    # whatever the bundle's own node configs (and metadata.json) state.
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
    # this family's own preprocess_step understands (for Qwen3-VL, a tuple
    # of indices into the images list) — the generic code in vlm.py passes
    # it straight through without interpreting it.
    build_prompt_segments: Callable = field(repr=False)

    # Preprocessing function: (images, payload, spec) -> pixel_values ndarray
    # for one step. Takes the whole images list plus the payload its own
    # build_prompt_segments emitted, so that packing several frames into one
    # step stays entirely inside the family module. Takes spec as an
    # argument (rather than a self-referential closure bound at dataclass
    # construction time).
    preprocess_step: Callable = field(repr=False)

    # Rows pixel_values is zero-padded to; 0 = exactly the resolution's own
    # patch count. See VLMFamily.max_patches.
    max_patches: int = 0

    # True when the bundle's text-encoder config names a bos-token, so
    # libGenie prepends a BOS to every text segment itself. Set by VLMSlot
    # from the loaded config; a template with a BOS of its own leaves it out.
    text_encoder_adds_bos: bool = False

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


def template(family: VLMFamily) -> VLMSpec:
    """A VLMSpec built from nothing but the family's own defaults — no
    bundle layout, unbound. What get_spec() hands back for a caller that
    wants a template to bind itself (tests, mainly); resolve() starts from
    this and then fills in the bundle's layout and runs bind()."""
    return VLMSpec(
        name=family.name,
        node_config_files={},
        connections=[],
        static_tensor_files={},
        image_width=family.image_width,
        image_height=family.image_height,
        patch_size=family.patch_size,
        spatial_merge_size=family.spatial_merge_size,
        temporal_patch_size=family.temporal_patch_size,
        normalize_mean=family.normalize_mean,
        normalize_std=family.normalize_std,
        build_prompt_segments=family.build_prompt_segments,
        preprocess_step=family.preprocess_step,
        max_patches=family.max_patches,
    )


def resolve(family: VLMFamily, layout, node_cfgs: dict) -> VLMSpec:
    """Combines one VLMFamily with one bundle's BundleLayout + loaded node
    configs into the VLMSpec VLMSlot actually runs from."""
    spec = replace(
        template(family),
        node_config_files=dict(layout.node_config_files),
        connections=list(layout.connections),
        static_tensor_files=dict(layout.static_tensor_files),
    )
    if family.bind is not None:
        spec = family.bind(spec, node_cfgs, layout)
    return spec
