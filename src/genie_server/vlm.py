"""VLM (multimodal) support via the GenieNode/GeniePipeline composable API.

A VLMSlot wraps three GenieNode handles (image-encoder, text-encoder,
text-generator) plus one GeniePipeline handle — a completely different object
model from Slot's single GenieDialog handle. VLM slots live in their own
registry and are never mixed into text slots: GenieDialog-only features
(LoRA, prefix cache, grammar, /v1/models/switch, per-request max_tokens/stop
enforcement, abort-on-disconnect) do not apply to them — GenieNode.h/
GeniePipeline.h simply have no equivalent APIs.

A request's own "max_tokens" therefore cannot be honoured. The only limit the
composable-pipeline API exposes is the text-generator node's static
"max-num-tokens", read once when the node is created, which this module fills
in from VLM_SLOTS[].max_tokens. Whichever limit stops the generation — that
cap or the context filling up — the response is reported with
finish_reason="length".

Imports of genie_node/vlm_specs (and their numpy/Pillow dependencies) are
kept optional so a missing numpy/Pillow degrades gracefully to "VLM_SLOTS
unavailable" rather than killing a text-only deployment.
"""

import json
import logging
import threading
from dataclasses import replace
from pathlib import Path

from . import capi
from . import vlm_layout
from .config import ServerConfig
from .slots import (resolve_and_verify, resolve_lut_paths, pin_htp_device,
                    load_tokenizer_file)

logger = logging.getLogger(__name__)

try:
    from . import genie_node
    from . import vlm_specs
    VLM_AVAILABLE = True
except ImportError as e:
    VLM_AVAILABLE = False
    _VLM_IMPORT_ERROR = str(e)

# GenieNode_TextOutput_SentenceCode_t string names (genie_node.SENTENCE_CODE)
_TERMINAL_CODES = {"complete", "end", "abort"}


def _load_vlm_node_config(config_path: Path, device_id: int | None,
                          slot_name: str, node_key: str, htp_ext_cache_dir: Path,
                          max_tokens: int = 0):
    """Reads one node config (img-enc-htp.json / text-encoder.json /
    text-generator.json) and returns a dict for genie_node.Node(...) with
    every relative asset path resolved against the config's own directory —
    the node-config counterpart of slots.load_dialog_config.

    Absolutizing is mandatory, not cosmetic: genie_node.Node passes the
    config to GenieNodeConfig_createFromJson as a JSON *string*, so libGenie
    has no idea which directory it came from and resolves "vision_encoder.bin"
    and friends against the server process's CWD. Without this the node
    configs only load if the server happens to be started from inside the
    model directory ("NSPModel: Can't access model file : vision_encoder.bin").

    If device_id is given, the HTP backend extensions file is also pinned to
    that device, mirroring slots.pin_htp_device. Not every node type has an
    engine.backend.extensions field (text-encoder is a pure CPU-side LUT —
    no HTP device to pin)."""
    with open(config_path) as f:
        node_cfg = json.load(f)
    base = config_path.parent
    top_key = next(iter(node_cfg))
    cfg = node_cfg[top_key]

    # The only generation cap reachable on this path: Dialog reads
    # "max-num-tokens" once, at node-creation time. GenieNode.h exposes no
    # per-request limit and no abort, and the text callback's return value is
    # discarded, so this static cap is what stands between a non-terminating
    # generation and a wedged slot. Only the text-generator has a dialog
    # config to put it in.
    if max_tokens and top_key == "text-generator":
        cfg["max-num-tokens"] = int(max_tokens)

    # tokenizer.json (text-encoder, text-generator)
    tok = cfg.get("tokenizer", {})
    if tok.get("path"):
        tok["path"] = resolve_and_verify(tok["path"], base)

    # LUT embeddings — "lut" for text-encoder, "embedding" for text-generator
    # (same LUT file, two different config shapes). The "perlayer-" pair is
    # Gemma 4's per-layer embedding table, which both nodes name as well; a
    # PCQ table adds quant-param files to each (slots.resolve_lut_paths).
    for lut_key in ("lut", "perlayer-lut", "embedding", "perlayer-embedding"):
        lut = cfg.get(lut_key)
        if isinstance(lut, dict):
            resolve_lut_paths(lut, base)

    engine = cfg.get("engine", {})

    # htp_backend_ext_config.json (image-encoder, text-generator)
    backend = engine.get("backend", {})
    if backend.get("extensions"):
        backend["extensions"] = resolve_and_verify(backend["extensions"], base)
        if device_id is not None:
            backend["extensions"] = pin_htp_device(
                backend["extensions"], device_id,
                f"{slot_name}_{node_key}", htp_ext_cache_dir)
    elif device_id is not None:
        logger.debug(
            f"[{slot_name}] node '{node_key}': no engine.backend.extensions to "
            "patch — HTP pinning skipped (expected for a CPU-side LUT node).")

    # vision_encoder.bin / part*_of_4.bin
    bincfg = engine.get("model", {}).get("binary", {})
    if isinstance(bincfg.get("ctx-bins"), list):
        bincfg["ctx-bins"] = [resolve_and_verify(b, base) for b in bincfg["ctx-bins"]]

    return node_cfg


def _pipeline_tokenizer_path(node_cfgs: dict) -> str | None:
    """The tokenizer.json path this pipeline's text nodes use, or None.

    GenieNode exposes no tokenizer to the host, but every node config that
    tokenizes names the file it does it with, and _load_vlm_node_config has
    already resolved that to an absolute path.

    The text-generator's is preferred: its ids are the ones the generation is
    counted in. Every family so far points both text nodes at one file, so
    the text-encoder is only a fallback for a bundle where the generator has
    none.
    """
    for node_key in ("text_generator", "text_encoder"):
        cfg = node_cfgs.get(node_key)
        if not cfg:
            continue
        path = next(iter(cfg.values())).get("tokenizer", {}).get("path")
        if path:
            return path
    return None


def _load_pipeline_tokenizer(node_cfgs: dict):
    """The tokenizer this pipeline's text nodes use, or None. Loading it is
    what puts a VLM slot's `usage` on the same basis as a text slot's."""
    path = _pipeline_tokenizer_path(node_cfgs)
    return load_tokenizer_file(path) if path else None


def _pipeline_tokenizer_json(node_cfgs: dict) -> dict:
    """The raw parsed tokenizer.json (added_tokens and all), or {} — used
    only for vlm_specs.detect_family, which needs the JSON's own fields
    (added_tokens) rather than the loaded HF Tokenizer object."""
    path = _pipeline_tokenizer_path(node_cfgs)
    if not path:
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _context_size(node_cfgs: dict) -> int:
    """The text-generator's `context.size`, or 0 when the config does not say.

    Same shape as a text bundle's genie_config.json, one level down:
    {"text-generator": {"context": {"size": 4096, ...}, ...}}. Returning 0
    rather than a default keeps plan_segments from enforcing a limit it made
    up — a spec whose config omits the field gets no budget check.
    """
    cfg = node_cfgs.get("text_generator")
    if not cfg:
        return 0
    inner = next(iter(cfg.values()), {})
    try:
        return int(inner.get("context", {}).get("size", 0))
    except (TypeError, ValueError):
        return 0


def _text_encoder_bos(node_cfgs: dict) -> int | None:
    """The bos-token the text-encoder config names, or None."""
    cfg = node_cfgs.get("text_encoder")
    if not cfg:
        return None
    token = next(iter(cfg.values()), {}).get("context", {}).get("bos-token")
    return token if isinstance(token, int) and token >= 0 else None


class VLMSlot:
    """One independent VLM pipeline: image-encoder + text-encoder +
    text-generator GenieNodes wired into a GeniePipeline, per a
    vlm_layout.BundleLayout's topology and a vlm_specs.VLMFamily's
    preprocessing. Optionally pinned to a single HTP device the same way a
    text Slot is."""

    # The bos-token libGenie's LUT text-encoder prepends to every text
    # segment, or None; see __init__.
    text_encoder_bos = None

    def __init__(self, name: str, device_id: int | None, model_root: Path,
                 spec_name: str | None, htp_ext_cache_dir: Path, max_tokens: int = 0,
                 log_handle=None, pipeline_script: str | None = None,
                 node_configs: dict | None = None, static_tensors: dict | None = None):
        self.name = name
        self.device_id = device_id
        self.model_root = model_root
        self.max_tokens = max_tokens
        self.lock = threading.Lock()
        self.active_model_id = model_root.name
        # Filled in below from the text-generator node's tokenizer.json, the
        # same file the node itself tokenizes with — see count_tokens.
        self.tokenizer = None

        # Read straight from the bundle (or an explicit VLM_SLOTS[]
        # override) — see vlm_layout.py's module docstring for the priority
        # order.
        self.layout = vlm_layout.read_layout(
            model_root, pipeline_script=pipeline_script,
            node_configs=node_configs, static_tensors=static_tensors)

        # Create the text-generator FIRST, then everything else. On QAIRT
        # 2.49 the image-encoder's context reserves DSP memory in a way that
        # leaves the text-generator's weight-shared ctx-bins unable to
        # allocate: GenieNode_create(text-generator) dies with
        #   "Could not create context from binary for context index = 2 :
        #    err 1002"  (err 1002 = QNN_COMMON_ERROR_MEM_ALLOC)
        # even with no text slot loaded and no HTP device pinning. Building
        # the big model first and letting the small image encoder fit around
        # it works on both 2.48 and 2.49. Reproduced with the stock
        # genie-app on the SDK's own genie-app-script.txt, so this is a
        # backend-level constraint, not something this server introduces.
        # 2.50.0.260828 does not need this -- genie-app runs that same script
        # in its own order there -- but which layer fixed it was not
        # established, and every 2.49.x still needs it, so it stays.
        # Pipeline add/connect order still follows the layout.
        node_keys = sorted(self.layout.node_config_files,
                           key=lambda k: k != "text_generator")
        node_cfgs = {}
        for node_key in node_keys:
            cfg_path = Path(resolve_and_verify(
                self.layout.node_config_files[node_key], model_root))
            node_cfgs[node_key] = _load_vlm_node_config(
                cfg_path, device_id, name, node_key, htp_ext_cache_dir, max_tokens)

        # Before any node exists, so a family's bind() can both read the
        # bundle's own settings (Gemma 4's patch grid, Qwen3-VL's
        # vision-param) and adjust the configs the nodes are built from.
        # Reading configs allocates nothing on the device, so resolving the
        # family here leaves the creation order below unchanged. spec_name
        # absent means auto-detect from the bundle's tokenizer + node configs
        # (config.py's VLMSlotSpec.spec — None = auto).
        if spec_name:
            family = vlm_specs.get_family(spec_name)
        else:
            family = vlm_specs.detect_family(
                _pipeline_tokenizer_json(node_cfgs), node_cfgs)
            logger.info(f"[{name}] VLM family auto-detected: {family.name} "
                       f"(layout: {self.layout.source})")
        self.spec = vlm_specs.resolve(family, self.layout, node_cfgs)

        # The LUT text-encoder prepends its configured bos-token on every
        # setData, and a prompt reaches it as one segment per stretch of text
        # around each image. That is left as the bundle declares it — the
        # spec's template writes no BOS of its own, usage counts the SDK's,
        # and the repetition is reported here rather than removed.
        self.text_encoder_bos = _text_encoder_bos(node_cfgs)
        if self.text_encoder_bos is not None:
            logger.warning(
                f"[{name}] text-encoder context.bos-token={self.text_encoder_bos}: "
                "libGenie prepends that token to every text segment, and a prompt "
                "is fed as one segment per stretch of text around each image, so "
                "a prompt with N images carries it N+1 times. Left as the bundle "
                "declares it; drop bos-token from the text-encoder config to keep "
                "only the BOS the model's chat format itself has, if any.")
        self.spec = replace(self.spec,
                            text_encoder_adds_bos=self.text_encoder_bos is not None,
                            static_tensor_files=vlm_layout.resolve_static_tensors(
                                node_cfgs, self.spec.static_tensor_files, name))
        built = {}
        for node_key in node_keys:
            built[node_key] = genie_node.Node(node_cfgs[node_key], log_handle=log_handle)
        nodes = {k: built[k] for k in self.spec.node_config_files}
        self.tokenizer = _load_pipeline_tokenizer(node_cfgs)
        # Baked into the context binaries at export time, so the config's
        # number is the real ceiling — plan_segments budgets vision tokens
        # against it. 0 disables that check rather than guessing.
        self.context_size = _context_size(node_cfgs)
        self.image_encoder = nodes["image_encoder"]
        self.text_encoder = nodes["text_encoder"]
        self.text_generator = nodes["text_generator"]

        self.pipeline = genie_node.Pipeline(log_handle=log_handle)
        for node in nodes.values():
            self.pipeline.add(node)
        for producer_key, io, consumer_key, io2 in self.spec.connections:
            self.pipeline.connect(nodes[producer_key], io, nodes[consumer_key], io2)

        # Content-independent tensors (position encodings, attention masks)
        # for the spec's fixed resolution — read once, reused every request.
        self.static_tensors = {}
        for io_name, rel_path in self.spec.static_tensor_files.items():
            with open(resolve_and_verify(rel_path, model_root), "rb") as f:
                self.static_tensors[io_name] = f.read()

    def count_tokens(self, text: str) -> int:
        """Exact token count via the pipeline's own tokenizer.json; whitespace
        fallback when 'tokenizers' is not installed or the file is unreadable.
        Mirrors Slot.count_tokens so a VLM slot's usage numbers are on the
        same basis as a text slot's.

        Text only: the image path never becomes text on the host (the
        image-encoder node emits embeddings straight into the pipeline), so
        there is nothing here to tokenize. The visual half of `usage` comes
        from count_vision_tokens instead, and the caller adds the two."""
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text).ids)
        return len(text.split())


def create_vlm_slots(config: ServerConfig, genie_cdll,
                     log_handle=None) -> list[VLMSlot]:
    """Builds every configured VLM slot. Raises on failure (startup-fatal).
    Reuses the already-loaded libGenie CDLL for GenieNode_*/GeniePipeline_*
    symbols instead of loading the shared library a second time.

    Must run BEFORE any text slot's GenieDialog is created (bootstrap.py
    orders it that way). libGenie's dialog-config validator keeps
    "pos-id-dim seen" / "rope-theta seen" in process-global state that is
    only cleared when a dialog config is created through the public
    GenieDialogConfig_createFromJson entry point. The GenieNode
    text-generator path validates its config without going through that
    entry point, so a text model configured with pos-id-dim/rope-theta
    (e.g. qwen3_0_6b) leaves the flags set and the VLM text-generator's
    "positional-encoding" block is then rejected with
    "Specify one config from pos-id-dim and positional-encoding".
    Creating the VLM slots first sidesteps it: text dialog creation resets
    the flags on entry, so the reverse order is harmless."""
    if not config.vlm_slots:
        return []
    if not VLM_AVAILABLE:
        logger.warning("env_config.json has VLM_SLOTS but VLM support failed to "
                       f"import ({_VLM_IMPORT_ERROR}) — skipping. "
                       "Install with: pip install numpy pillow")
        return []

    genie_node.attach(genie_cdll)
    htp_ext_cache_dir = Path(config.prefix_cache_dir) / ".htp_ext_cache"
    out = []
    for spec in config.vlm_slots:
        vslot = VLMSlot(name=spec.name, device_id=spec.device_id,
                        model_root=spec.model_root, spec_name=spec.spec,
                        htp_ext_cache_dir=htp_ext_cache_dir,
                        max_tokens=spec.max_tokens, log_handle=log_handle,
                        pipeline_script=spec.pipeline_script,
                        node_configs=spec.node_configs,
                        static_tensors=spec.static_tensors)
        out.append(vslot)
        logger.info(
            f"VLM slot '{vslot.name}' ready: model={vslot.active_model_id} "
            f"device_id={vslot.device_id if vslot.device_id is not None else '(unpinned)'} "
            f"family={vslot.spec.name} (layout: {vslot.layout.source}) "
            f"max-num-tokens={vslot.max_tokens or '(uncapped)'}")
    return out


# ---------------------------------------------------------------- request parsing

_MULTIMODAL_PART_TYPES = ("image_url", "video_url")


def is_vlm_request(messages: list) -> bool:
    """True if any message's `content` is a parts array containing an
    image_url or video_url part — the only signal used to route a chat
    request to a VLM slot instead of the (unmodified) GenieDialog text
    path."""
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if (isinstance(part, dict)
                        and part.get("type") in _MULTIMODAL_PART_TYPES):
                    return True
    return False


def extract_video_meta(body: dict) -> dict:
    """The request's video timeline metadata, or {}.

    Follows vLLM's `media_io_kwargs.video` for client-side frame extraction
    (`fps`, `frames_indices`, ...). OpenAI clients put it there by passing
    `extra_body={"media_io_kwargs": {...}}`, which lands at the top level of
    the request body.

    This is a vLLM convention, not part of the OpenAI API, so a server that
    does not know it simply ignores the key. That is why the `<t seconds>`
    markers it feeds are best-effort: without an fps the spec emits none
    rather than inventing a timeline."""
    kwargs = body.get("media_io_kwargs")
    if not isinstance(kwargs, dict):
        return {}
    video = kwargs.get("video")
    return video if isinstance(video, dict) else {}


def _decode_base64_image(b64data: str, what: str):
    """One base64 payload (an image_url's, or one frame out of a video_url's
    comma-joined list) -> a loaded PIL image."""
    import base64
    import io
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64data)))
        img.load()
    except Exception as e:
        raise ValueError(f"failed to decode {what} data: {e}") from e
    return img


def decode_media_sources(sources: list) -> list:
    """The base64 payloads extract_multimodal_parts collected -> PIL images,
    in the same order, so a part's indices keep pointing at the right frame.

    Split out of the parsing deliberately: the whole plan (how many encoder
    steps, and whether they fit the context) is known from the *counts*
    alone, so a request the budget guard is going to refuse never pays for
    decoding its frames. That matters at the sizes this path invites — a
    500-frame request is 500 JPEG decodes and their bitmaps resident at once,
    on a board whose memory is the reason the guard exists.
    """
    return [_decode_base64_image(b64, what) for b64, what in sources]


def extract_multimodal_parts(messages: list) -> tuple:
    """Parses OpenAI-style multimodal `messages` into (system_text, parts,
    sources) for vlm_specs.VLMSpec.build_prompt_segments:
      - system_text: the system message's content (string), or "".
      - parts: ordered [("text", str) | ("image", index) |
        ("video", [index, ...])] from the LAST non-system message's content
        list (V1 is single-turn — only one user turn with media is
        supported).
      - sources: ordered list of (base64 payload, description) pairs, still
        undecoded. Both an "image" part's index and a "video" part's index
        list point into it — a video's frames are just images that the spec
        knows to pack several-per-step. decode_media_sources turns them into
        the PIL images start_vlm_generation feeds, once the request is known
        to be one worth decoding.
    Only `data:` (base64) URLs are supported (V1 does not fetch remote
    http(s) URLs). Raises ValueError with a client-safe message."""
    system_text = ""
    user_messages = []
    for m in messages:
        role = m.get("role", "user")
        if role == "system":
            c = m.get("content", "")
            system_text = c if isinstance(c, str) else "".join(
                p.get("text", "") for p in c
                if isinstance(p, dict) and p.get("type") == "text")
        else:
            user_messages.append(m)

    if not user_messages:
        raise ValueError("no user message with content")
    content = user_messages[-1].get("content")
    if not isinstance(content, list):
        raise ValueError("expected a multimodal 'content' array on the last message")

    parts, sources = [], []
    for part in content:
        ptype = part.get("type")
        if ptype == "text":
            parts.append(("text", part.get("text", "")))
        elif ptype == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            if not url.startswith("data:"):
                raise ValueError(
                    "only data: (base64) image URLs are supported; remote "
                    "http(s) URLs are not fetched by this server")
            sources.append((url.split(",", 1)[1] if "," in url else "",
                            "image_url"))
            parts.append(("image", len(sources) - 1))
        elif ptype == "video_url":
            # vLLM's client-side frame-extraction form: the frames are already
            # JPEGs, comma-joined inside one data URL, and the media type says
            # so ("video/jpeg") to stop a server from re-decoding a container.
            # A real container (video/mp4 and friends) would need a demuxer
            # this server does not carry, so it is refused rather than
            # half-supported.
            url = (part.get("video_url") or {}).get("url", "")
            if not url.startswith("data:"):
                raise ValueError(
                    "only data: (base64) video URLs are supported; remote "
                    "http(s) URLs are not fetched by this server")
            header = url.split(",", 1)[0]
            if "video/jpeg" not in header:
                raise ValueError(
                    "video_url must carry pre-extracted frames as "
                    "'data:video/jpeg;base64,<frame>,<frame>,...'; this "
                    f"server does not decode video containers ({header!r})")
            payload = url.split(",", 1)[1] if "," in url else ""
            frames_b64 = [f for f in payload.split(",") if f]
            if not frames_b64:
                raise ValueError("video_url contained no frames")
            frame_indices = []
            for n, frame_b64 in enumerate(frames_b64):
                sources.append((frame_b64, f"video_url frame {n}"))
                frame_indices.append(len(sources) - 1)
            parts.append(("video", frame_indices))
        else:
            raise ValueError(f"unsupported content part type: {ptype!r}")

    return system_text, parts, sources


# ---------------------------------------------------------------- planning

# What to keep free for generation when a slot is uncapped (max_tokens=0).
# Only a guess — an uncapped slot can still run into the context; the cap is
# the only thing that actually bounds it.
UNCAPPED_GENERATION_RESERVE = 256


def count_vision_tokens(spec, segments: list) -> int:
    """How much of the text-generator's context this request's visual input
    occupies.

    Derived, not reported: the image path never becomes text on the host (the
    image-encoder emits embeddings straight into the pipeline) and neither
    GenieNode.h nor GeniePipeline.h has a call that hands the count back, so
    steps x the spec's per-step cost is the only way to know it.

    The derivation is confirmed by where the context actually runs out: the
    4096-context Qwen3-VL 4B bundle takes 15 steps and fails on the 16th,
    which is exactly 16 x 256.
    """
    steps = sum(1 for kind, _ in segments if kind == "step")
    return steps * spec.vision_tokens_per_step


def count_text_encoder_bos(vslot, segments: list) -> int:
    """The BOS tokens libGenie adds to this request on its own: one per text
    segment when the text-encoder config names a bos-token
    (VLMSlot.text_encoder_bos). The host tokenizer never sees them, so usage
    and the budget have to add them."""
    if vslot.text_encoder_bos is None:
        return 0
    return sum(1 for kind, _ in segments if kind == "text")


def count_prompt_tokens(vslot, segments: list) -> int:
    """What the text-generator is prefilled with for this request, on the
    same basis as a text slot's Slot.count_prompt_tokens: every text segment
    as the spec rendered it — chat-template markers included, each segment
    tokenized on its own because that is how the text-encoder receives it —
    plus the vision tokens and the BOS the text-encoder adds per segment."""
    text = sum(vslot.count_tokens(v) for kind, v in segments if kind == "text")
    return (text + count_vision_tokens(vslot.spec, segments)
            + count_text_encoder_bos(vslot, segments))


def plan_segments(vslot: VLMSlot, system_text: str, parts: list,
                  video_meta: dict, guard: bool = False) -> list:
    """Builds the spec's text/step segment list, and — when `guard` is on —
    refuses it up front if its vision tokens cannot fit the text-generator's
    context.

    **The guard is off by default** (VLM_VISION_BUDGET_GUARD), because it
    conceals a defect this server exists to expose. What it conceals is worth
    stating plainly. Measured on SA8255P / QAIRT 2.49, Qwen3-VL 4B, context
    4096, 256 vision tokens per step, sweeping the prompt a token at a time:

      prompt <= 3969        answers normally.
      3970..4096            0 tokens and finish_reason "length". Nothing is
                            damaged: the same slot answers the next request,
                            and 3969/3970 can be alternated indefinitely.
                            3969 is ctx - 127, the margin the AR128 prefill
                            graph keeps; the node config does not state it,
                            so the host cannot compute this line.
      prompt > 4096         0 tokens, and then the slot is wedged: every
                            later request, however small, fails in
                            GenieNode_setData on the image encoder
                            ("status=-1", WINDOW_ATTN_MASK) until the process
                            is restarted. GeniePipeline_reset does not clear
                            it. Reproduced two ways at 15 steps + long text
                            (prompt 4142) and at 16 steps (prompt 4231), so
                            the trigger is the prompt passing the context,
                            not the step count.
      17+ steps             the process dies outright ("free(): invalid next
                            size"), the overrun being large enough.

    **Decoding across the context does not wedge anything.** A prompt of 3940
    with max_tokens 200 generates exactly 156 tokens, stops at 4096 on the
    nose with finish_reason "length", and leaves the slot healthy. The reserve
    below still subtracts the whole generation budget, but for a plainer
    reason than safety: so the answer is not cut off mid-sentence, and so the
    ~127-token prefill margin above is absorbed by something, since the host
    has no way to read it.

    GenieNode.h/GeniePipeline.h expose no way to ask how much room is left,
    so arithmetic on the host is the only guard available at all.

    With the guard off the request is passed through unchanged, but an
    oversized one is logged at WARNING first. Saying what is about to happen
    does not hide it — it is the opposite.
    """
    spec = vslot.spec
    segments = spec.build_prompt_segments(system_text, parts, video_meta, spec)
    steps = sum(1 for kind, _ in segments if kind == "step")
    if not steps or not vslot.context_size:
        return segments

    per_step = spec.vision_tokens_per_step
    # The rendered text segments and the text-encoder's BOS: everything the
    # text-generator is prefilled with besides the vision tokens, on the same
    # basis usage reports.
    text_tokens = count_prompt_tokens(vslot, segments) - steps * per_step
    reserve = vslot.max_tokens or UNCAPPED_GENERATION_RESERVE
    # The ceiling is the whole context, because that is where the wedge is:
    # a prompt of 4096 still answers (with nothing), 4097 poisons the slot.
    # The narrower line at ctx - 127, past which the answer is empty but the
    # slot survives, is deliberately not the ceiling -- the node config does
    # not state the AR length it comes from, so a server that subtracted a
    # guessed one would refuse requests a differently exported bundle can
    # serve. The reserve covers it in practice for any sane max_tokens; a
    # slot configured below ~127 can still be handed a request that comes
    # back empty, which is a wasted request rather than a broken slot.
    budget = vslot.context_size - text_tokens - reserve
    max_steps = max(budget // per_step, 0)

    if steps > max_steps:
        frames_per_step = spec.temporal_patch_size
        per_video_step = ("frame" if frames_per_step == 1
                          else f"{frames_per_step} frames")
        detail = (
            f"{steps} encoder steps ({steps * per_step} vision tokens) but "
            f"only {max_steps} fit (context {vslot.context_size} - "
            f"{text_tokens} prompt text - {reserve} reserved for generation, "
            f"at {per_step} tokens per step)")
        if not guard:
            logger.warning(
                "VLM request exceeds the context and VLM_VISION_BUDGET_GUARD "
                "is off, so it is being sent as-is: %s. Expect the slot to "
                "wedge, or the process to die. Set VLM_VISION_BUDGET_GUARD "
                "to refuse it with a 400 instead.", detail)
            return segments
        raise ValueError(
            f"too much visual input for this slot: {detail}. One still image "
            f"is one step; a video is one step per {per_video_step}. "
            f"Send fewer frames, or lower VLM_SLOTS[].max_tokens to free up "
            f"context.")
    return segments


# ---------------------------------------------------------------- generation

def start_vlm_generation(lib, vslot: VLMSlot, segments: list,
                         images: list, params, generation) -> None:
    """Kicks off one VLM pipeline execution on a worker thread, feeding
    generation.queue like the text engine does. There is no GenieNode/
    GeniePipeline abort or signal API: a disconnected client's pipeline keeps
    running server-side (holding vslot.lock) until it finishes naturally."""

    def on_text(text: str, code: str) -> None:
        try:
            if text and code != "abort":
                generation.completion_tokens += 1
                generation.put_threadsafe(text)
        except Exception as e:
            logger.error(f"Exception in VLM callback [{generation.request_id}]: {e}")

    def worker() -> None:
        try:
            with vslot.lock:
                if generation.aborted.is_set():
                    # The caller gave up (a 504, or a client that left) while
                    # this waited for the slot. There is no abort once the
                    # pipeline runs, but it need not start at all.
                    logger.info(f"[{vslot.name}] Request abandoned while waiting "
                                f"for the slot; not running it [{generation.request_id}]")
                    return
                vslot.text_generator.set_text_callback(
                    vlm_layout.TEXT_GENERATOR_TEXT_OUTPUT_IO, on_text)
                sampler_params = capi.make_sampler_params(
                    {}, params.temperature, params.top_p, params.top_k, params.seed)
                if sampler_params:
                    try:
                        lib.apply_sampler_params_to_handle(
                            vslot.text_generator.get_sampler(), sampler_params)
                    except Exception as e:
                        logger.warning(f"VLM sampling params not applied: {e}")

                vslot.pipeline.reset()
                spec = vslot.spec
                for kind, value in segments:
                    if kind == "text":
                        vslot.text_encoder.set_text(
                            vlm_layout.TEXT_ENCODER_TEXT_INPUT_IO, value)
                    else:  # "step" — one image-encoder execution
                        pixel_values = spec.preprocess_step(images, value, spec)
                        vslot.image_encoder.set_buffer(
                            vlm_layout.IMAGE_ENCODER_IMAGE_INPUT_IO, pixel_values)
                        for io_name, static_bytes in vslot.static_tensors.items():
                            vslot.image_encoder.set_buffer(io_name, static_bytes)
                vslot.pipeline.execute()

            # Reaching here means the SDK returned SUCCESS. It does that both
            # for a natural EOS stop and for hitting the node's
            # "max-num-tokens" cap, so the token count is the only way to tell
            # them apart (there is no per-request limit to compare against —
            # see the module docstring).
            if vslot.max_tokens and generation.completion_tokens >= vslot.max_tokens:
                generation.finish_reason = "length"
        except genie_node.GenieStatusError as e:
            if e.status == capi.WARNING_CONTEXT_EXCEEDED:
                # The generation ran until the context filled up. Whatever was
                # produced before that is valid output, so report it as a
                # length stop rather than a server error.
                logger.warning(
                    f"VLM generation hit the context limit [{generation.request_id}] "
                    f"after {generation.completion_tokens} tokens; returning a "
                    "truncated response (finish_reason=length). Set "
                    "VLM_SLOTS[].max_tokens to stop earlier.")
                generation.finish_reason = "length"
            else:
                logger.error(
                    f"VLM pipeline execute failed [{generation.request_id}]: {e}")
                generation.error = str(e)
        except Exception as e:
            logger.error(f"VLM pipeline execute failed [{generation.request_id}]: {e}")
            generation.error = str(e)
        finally:
            generation.put_threadsafe(None)
            generation.done.set()

    threading.Thread(target=worker, daemon=True).start()
