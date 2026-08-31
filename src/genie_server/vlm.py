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
from pathlib import Path

from . import capi
from .config import ServerConfig
from .slots import resolve_and_verify, pin_htp_device, load_tokenizer_file

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

    # embedding_weights.raw — "lut" for text-encoder, "embedding" for
    # text-generator (same LUT file, two different config shapes).
    for lut_key in ("lut", "embedding"):
        lut = cfg.get(lut_key, {})
        if lut.get("lut-path"):
            lut["lut-path"] = resolve_and_verify(lut["lut-path"], base)

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


def _load_pipeline_tokenizer(node_cfgs: dict):
    """The tokenizer.json this pipeline's text nodes use, or None.

    GenieNode exposes no tokenizer to the host, but every node config that
    tokenizes names the file it does it with, and _load_vlm_node_config has
    already resolved that to an absolute path. Loading the same file here is
    what puts a VLM slot's `usage` on the same basis as a text slot's.

    The text-generator's is preferred: its ids are the ones the generation is
    counted in. Every spec so far points both text nodes at one file, so the
    text-encoder is only a fallback for a spec where the generator has none.
    """
    for node_key in ("text_generator", "text_encoder"):
        cfg = node_cfgs.get(node_key)
        if not cfg:
            continue
        path = next(iter(cfg.values())).get("tokenizer", {}).get("path")
        if path:
            return load_tokenizer_file(path)
    return None


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


class VLMSlot:
    """One independent VLM pipeline: image-encoder + text-encoder +
    text-generator GenieNodes wired into a GeniePipeline, per a
    vlm_specs.VLMSpec's topology. Optionally pinned to a single HTP device
    the same way a text Slot is."""

    def __init__(self, name: str, device_id: int | None, model_root: Path,
                 spec_name: str, htp_ext_cache_dir: Path, max_tokens: int = 0):
        self.name = name
        self.device_id = device_id
        self.model_root = model_root
        self.max_tokens = max_tokens
        self.spec = vlm_specs.get_spec(spec_name)
        self.lock = threading.Lock()
        self.active_model_id = model_root.name
        # Filled in below from the text-generator node's tokenizer.json, the
        # same file the node itself tokenizes with — see count_tokens.
        self.tokenizer = None

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
        # Pipeline add/connect order still follows the spec.
        node_keys = sorted(self.spec.node_config_files,
                           key=lambda k: k != "text_generator")
        built = {}
        node_cfgs = {}
        for node_key in node_keys:
            cfg_path = Path(resolve_and_verify(
                self.spec.node_config_files[node_key], model_root))
            cfg = _load_vlm_node_config(cfg_path, device_id, name, node_key,
                                        htp_ext_cache_dir, max_tokens)
            node_cfgs[node_key] = cfg
            built[node_key] = genie_node.Node(cfg)
        nodes = {k: built[k] for k in self.spec.node_config_files}
        self.tokenizer = _load_pipeline_tokenizer(node_cfgs)
        # Baked into the context binaries at export time, so the config's
        # number is the real ceiling — plan_segments budgets vision tokens
        # against it. 0 disables that check rather than guessing.
        self.context_size = _context_size(node_cfgs)
        self.image_encoder = nodes["image_encoder"]
        self.text_encoder = nodes["text_encoder"]
        self.text_generator = nodes["text_generator"]

        self.pipeline = genie_node.Pipeline()
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


def create_vlm_slots(config: ServerConfig, genie_cdll) -> list[VLMSlot]:
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
                        max_tokens=spec.max_tokens)
        out.append(vslot)
        logger.info(
            f"VLM slot '{vslot.name}' ready: model={vslot.active_model_id} "
            f"device_id={vslot.device_id if vslot.device_id is not None else '(unpinned)'} "
            f"spec={vslot.spec.name} "
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


def _decode_data_url(url: str, what: str):
    """One `data:<media type>;base64,<payload>` URL -> a loaded PIL image."""
    import base64
    import io
    from PIL import Image

    try:
        _, b64data = url.split(",", 1)
        img = Image.open(io.BytesIO(base64.b64decode(b64data)))
        img.load()
    except Exception as e:
        raise ValueError(f"failed to decode {what} data: {e}") from e
    return img


def extract_multimodal_parts(messages: list) -> tuple:
    """Parses OpenAI-style multimodal `messages` into (system_text, parts,
    images) for vlm_specs.VLMSpec.build_prompt_segments:
      - system_text: the system message's content (string), or "".
      - parts: ordered [("text", str) | ("image", index) |
        ("video", [index, ...])] from the LAST non-system message's content
        list (V1 is single-turn — only one user turn with media is
        supported).
      - images: ordered list of PIL.Image.Image. Both an "image" part's index
        and a "video" part's index list point into it — a video's frames are
        just images that the spec knows to pack several-per-step.
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

    parts, images = [], []
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
            images.append(_decode_data_url(url, "image_url"))
            parts.append(("image", len(images) - 1))
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
                images.append(_decode_data_url(
                    "data:image/jpeg;base64," + frame_b64,
                    f"video_url frame {n}"))
                frame_indices.append(len(images) - 1)
            parts.append(("video", frame_indices))
        else:
            raise ValueError(f"unsupported content part type: {ptype!r}")

    return system_text, parts, images


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


def plan_segments(vslot: VLMSlot, system_text: str, parts: list,
                  video_meta: dict, guard: bool = False) -> list:
    """Builds the spec's text/step segment list, and — when `guard` is on —
    refuses it up front if its vision tokens cannot fit the text-generator's
    context.

    **The guard is off by default** (VLM_VISION_BUDGET_GUARD), because it
    conceals a defect this server exists to expose. What it conceals is worth
    stating plainly, though: on this path an overrun does not fail cleanly.
    Measured on SA8255P / QAIRT 2.49, Qwen3-VL 4B, context 4096, 256 vision
    tokens per step:

      15 steps  ->  answers normally
      16 steps  ->  "Context Size was exceeded", 0 tokens, and the slot is
                    then wedged: every later request, however small, fails in
                    GenieNode_setData on the image encoder until the process
                    is restarted. GeniePipeline_reset does not clear it.
      17 steps  ->  the process dies outright ("free(): invalid next size").

    GenieNode.h/GeniePipeline.h expose no way to ask how much room is left,
    so arithmetic on the host is the only guard available, and it has to be
    conservative: the reserve subtracts the whole generation budget, because
    a request that fits its prompt can still cross the line while decoding.

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
    # The template-inclusive text, which is what actually reaches the
    # tokenizer — more than the raw prompt text `usage` counts as text.
    text_tokens = vslot.count_tokens("".join(v for k, v in segments if k == "text"))
    reserve = vslot.max_tokens or UNCAPPED_GENERATION_RESERVE
    budget = vslot.context_size - text_tokens - reserve
    max_steps = max(budget // per_step, 0)

    if steps > max_steps:
        frames_per_step = spec.temporal_patch_size
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
            f"is one step; a video is one step per {frames_per_step} frames. "
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
                vslot.text_generator.set_text_callback(
                    vslot.spec.text_generator_text_output_io, on_text)
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
                            spec.text_encoder_text_input_io, value)
                    else:  # "step" — one image-encoder execution
                        pixel_values = spec.preprocess_step(images, value, spec)
                        vslot.image_encoder.set_buffer(
                            spec.image_encoder_image_input_io, pixel_values)
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
