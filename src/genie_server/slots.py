"""Slot management: model loading, request routing, and hot-swapping.

A Slot is one independent GenieDialog instance, optionally pinned to a single
HTP device (device_id). On a dual-NSP SoC like SA8255P (cdsp0/
cdsp1), two Slots let two different models (or two LoRA variants of one base
model) run fully independently: separate locks, separate KV state, separate
everything. Requests routed to different slots run truly concurrently instead
of queueing behind one lock.
"""

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

from . import templates, tool_formats
from .config import DEFAULT_DIALOG_CONFIG, ServerConfig

logger = logging.getLogger(__name__)

try:
    from tokenizers import Tokenizer as HFTokenizer
    HF_TOKENIZER_AVAILABLE = True
except ImportError:
    HF_TOKENIZER_AVAILABLE = False
    logger.warning("'tokenizers' not installed — token counts will be approximated.")


class UnknownSlotError(ValueError):
    """Raised when a request names a slot that does not exist (HTTP 404)."""


class SlotNotLoadedError(RuntimeError):
    """Raised when a slot currently has no model loaded (HTTP 503)."""


def resolve_and_verify(rel: str, base: Path) -> str:
    p = (base / rel).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Required asset not found: {p}")
    return str(p)


@dataclass
class ModelAssets:
    """Everything load_model() produces for one model, before any Slot adopts
    it. Loading and adopting are separate steps so a hot-swap can load the
    new model FIRST and only free the old handle after success — a bad
    model_dir then never leaves a slot without a working model."""
    handle: object
    config_json: bytes
    dialog_cfg: dict
    tokenizer: object | None
    template: str
    tool_format: object
    model_dir: Path
    sampler_defaults: dict = field(default_factory=dict)

    @property
    def context_size(self) -> int | None:
        size = self.dialog_cfg.get("context", {}).get("size")
        return size if isinstance(size, int) and size > 0 else None


class Slot:
    def __init__(self, name: str, device_id: int | None, model_root: Path,
                 poll: bool | None = None,
                 config_file: str = DEFAULT_DIALOG_CONFIG):
        self.name = name
        self.device_id = device_id
        self.model_root = model_root
        # Both apply to whatever model this slot loads, including after a
        # /v1/models/switch — they belong to the slot, not the model.
        self.poll = poll
        self.config_file = config_file or DEFAULT_DIALOG_CONFIG
        # GenieProfile handle (when GENIE_PROFILE is on). Bound to every
        # dialog config this slot creates, so it survives hot-swaps; freed
        # only after the dialog it is bound to (the SDK refuses otherwise).
        self.profile = None
        self.lock = threading.Lock()
        self.handle = None
        self.tokenizer = None
        self.dialog_cfg: dict = {}
        self.chat_template = "chatml"
        # Which tool dialect this slot's model speaks — derived from the chat
        # template unless TOOL_FORMAT forces one. See tool_formats.
        self.tool_format = tool_formats.HermesToolFormat
        self.sampler_defaults: dict = {}
        self.active_model_id = model_root.name
        self.active_lora_adapter = ""
        # Logprobs (custom sampler) support — see engine.ensure_logprobs_sampler.
        # The callback registration is per-slot (names are process-global in
        # the SDK) and survives model hot-swaps; the collector is per-request,
        # swapped under self.lock.
        self.logprobs_registered = False
        self.active_collector = None

    @property
    def sampler_callback_name(self) -> str:
        return f"genie-server-logprobs-{self.name}"

    def adopt(self, assets: ModelAssets) -> None:
        """Adopts a load_model() result. The caller is responsible for
        freeing any handle this slot previously held — this only rebinds."""
        self.handle = assets.handle
        self.dialog_cfg = assets.dialog_cfg
        self.tokenizer = assets.tokenizer
        self.chat_template = assets.template
        self.tool_format = assets.tool_format
        self.sampler_defaults = assets.sampler_defaults
        self.model_root = assets.model_dir
        self.active_model_id = assets.model_dir.name
        self.active_lora_adapter = ""

    @property
    def context_size(self) -> int | None:
        size = self.dialog_cfg.get("context", {}).get("size")
        return size if isinstance(size, int) and size > 0 else None

    @property
    def cache_namespace(self) -> str:
        """Current (slot, model, LoRA) identity — see PrefixCache.key."""
        return f"{self.name}|{self.active_model_id}|{self.active_lora_adapter}"

    def count_tokens(self, text: str) -> int:
        """Exact token count via the model tokenizer; whitespace fallback."""
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text).ids)
        return len(text.split())


# ---------------------------------------------------------------- model loading

def load_tokenizer_file(tok_path: str):
    """Best-effort HF tokenizer load from a tokenizer.json path; None on any
    failure. Shared with vlm.VLMSlot, whose tokenizer.json comes out of a
    GenieNode config rather than a dialog config."""
    if not HF_TOKENIZER_AVAILABLE or not tok_path:
        return None
    try:
        tok = HFTokenizer.from_file(tok_path)
        logger.info(f"HF tokenizer loaded: {tok_path}")
        return tok
    except Exception as e:
        logger.warning(f"HF tokenizer load failed (approximation fallback): {e}")
        return None


def _load_tokenizer(dialog_cfg: dict):
    """Best-effort HF tokenizer load from a dialog config; None on any failure."""
    return load_tokenizer_file(dialog_cfg.get("tokenizer", {}).get("path", ""))


def pin_htp_device(extensions_path: str, device_id: int, slot_name: str,
                   cache_dir: Path) -> str:
    """Returns a path to a copy of the HTP backend extension config
    (dialog.engine.backend.extensions) with devices[0].device_id overridden.
    This is what actually selects cDSP0 vs cDSP1 on a dual-NSP SoC (QNN's
    htp_backend_ext_config schema). Patching a per-slot COPY means the same
    model directory can be assigned to multiple slots/devices without editing
    its config, and two slots loading the same model never fight over one
    file."""
    with open(extensions_path) as f:
        ext = json.load(f)
    devices = ext.get("devices") or [{}]
    devices[0]["device_id"] = device_id
    ext["devices"] = devices

    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{slot_name}_{Path(extensions_path).name}"
    with open(out_path, "w") as f:
        json.dump(ext, f)
    return str(out_path)


def load_dialog_config(model_dir: Path, device_id: int | None, slot_name: str,
                       htp_ext_cache_dir: Path,
                       poll: bool | None = None,
                       config_file: str = DEFAULT_DIALOG_CONFIG) -> tuple[bytes, dict]:
    """Reads the dialog config under model_dir and resolves all relative
    asset paths (tokenizer, backend extensions, ctx-bins, grammar file)
    against model_dir. If device_id is given, also pins the HTP backend to
    that device; if poll is given, it overrides QnnHtp.poll. Raises on any
    problem — never exits the process, so /v1/models/switch can reuse this
    safely at runtime."""
    genie_config_path = model_dir / (config_file or DEFAULT_DIALOG_CONFIG)
    with open(genie_config_path) as f:
        genie_config_data = json.load(f)

    dcfg = genie_config_data["dialog"]
    if "tokenizer" in dcfg and "path" in dcfg["tokenizer"]:
        dcfg["tokenizer"]["path"] = resolve_and_verify(dcfg["tokenizer"]["path"], model_dir)

    # Grammar-constrained decoding (dialog.context.grammar): backend must be
    # "xgrammar" (the SDK's only implementation), type is "json-schema"
    # (default) | "regex" | "ebnf", and "file" holds the grammar definition
    # itself. This is baked into the Dialog at GenieDialog_create() time: it
    # applies to every query on this model/slot; there is no per-request
    # override.
    grammar_cfg = dcfg.get("context", {}).get("grammar")
    if grammar_cfg and grammar_cfg.get("file"):
        grammar_cfg["file"] = resolve_and_verify(grammar_cfg["file"], model_dir)
        if grammar_cfg.get("backend") != "xgrammar":
            logger.warning(
                f"[{slot_name}] {genie_config_path}: dialog.context.grammar.backend="
                f"'{grammar_cfg.get('backend')}' — the SDK only implements \"xgrammar\"; "
                "GenieDialog_create will likely fail.")

    bcfg = dcfg.get("engine", {}).get("backend", {})

    # QnnHtp.poll: busy-wait for the DSP instead of blocking. The bundles we
    # have ship true, which costs ~260% CPU on SA8255P for latency that is
    # indistinguishable from false, so it is worth being able to turn off
    # without editing someone else's model directory.
    if poll is not None:
        htp = bcfg.setdefault("QnnHtp", {})
        if htp.get("poll") != poll:
            logger.info(f"[{slot_name}] QnnHtp.poll {htp.get('poll')!r} -> {poll!r} "
                        "(overridden by config)")
        htp["poll"] = poll

    if bcfg.get("extensions"):
        bcfg["extensions"] = resolve_and_verify(bcfg["extensions"], model_dir)
        if device_id is not None:
            bcfg["extensions"] = pin_htp_device(
                bcfg["extensions"], device_id, slot_name, htp_ext_cache_dir)
    elif device_id is not None:
        logger.warning(
            f"[{slot_name}] device_id={device_id} requested but {genie_config_path} has no "
            "dialog.engine.backend.extensions file to patch — NSP pinning skipped.")

    bincfg = dcfg.get("engine", {}).get("model", {}).get("binary", {})
    if isinstance(bincfg.get("ctx-bins"), list):
        bincfg["ctx-bins"] = [resolve_and_verify(b, model_dir) for b in bincfg["ctx-bins"]]

    # LUT embeddings (dialog.embedding / dialog.perlayer-embedding): the SDK
    # opens lut-path with a plain ifstream, so a relative one resolves against
    # the server's working directory rather than the bundle -- "Embedding File
    # not present." from LUT.cpp with the file sitting right there in the model
    # directory. Gemma-class bundles carry these; the Qwen3 exports do not,
    # which is why nothing needed it until now.
    for key in ("embedding", "perlayer-embedding"):
        emb = dcfg.get(key)
        if isinstance(emb, dict) and emb.get("lut-path"):
            emb["lut-path"] = resolve_and_verify(emb["lut-path"], model_dir)

    # LoRA adapter weights (dialog.engine.model.binary.lora.adapters[].
    # bin-sections[]): same story as the LUTs and the ctx-bins -- a relative
    # path resolves against the server's working directory, and the SDK
    # rejects the whole config at create time with "Error in parsing params -
    # LoRA: Can't access adapter file", naming a path that exists in the
    # bundle. One section per ctx-bin, per adapter.
    lora_cfg = bincfg.get("lora")
    if isinstance(lora_cfg, dict):
        for adapter in lora_cfg.get("adapters") or []:
            if isinstance(adapter, dict) and isinstance(adapter.get("bin-sections"), list):
                adapter["bin-sections"] = [
                    resolve_and_verify(b, model_dir) for b in adapter["bin-sections"]]

    # Speculative decoding (dialog.type "ssd-q1") restores a prefill KV cache
    # from the directory named by forecast-prefix-name, and the SDK hands that
    # string straight to the engine's restore() -- relative means the server's
    # working directory. Getting it wrong is not a missing-file error but
    # "SSD : Loaded 0 KV$ from <name> but expected 16 KV$", which reads like a
    # corrupt cache rather than a path resolved somewhere else. The SDK's own
    # sample configs write "./ssd_forecast_prefix", i.e. they assume you run
    # from inside the bundle; this server does not.
    ssd = dcfg.get("ssd-q1")
    if isinstance(ssd, dict) and ssd.get("forecast-prefix-name"):
        ssd["forecast-prefix-name"] = resolve_and_verify(
            ssd["forecast-prefix-name"], model_dir)

    return json.dumps(genie_config_data).encode("utf-8"), dcfg


_SAMPLER_DEFAULT_KEYS = ("temp", "top-k", "top-p")


class SlotManager:
    """Owns every text Slot (and the VLM slot list), routes requests to
    them, and performs model hot-swaps."""

    def __init__(self, config: ServerConfig, lib):
        self.config = config
        self.lib = lib
        self.slots: list[Slot] = []
        self.vlm_slots: list = []  # populated by vlm.create_vlm_slots()
        self._by_name: dict[str, Slot] = {}
        self._by_model_id: dict[str, Slot] = {}
        # Per-slot processing phase for /v1/server/status (GIL-atomic dict
        # key assignments; VLM slots don't report phases in V1).
        self.status: dict[str, dict] = {}
        self._htp_ext_cache_dir = Path(config.prefix_cache_dir) / ".htp_ext_cache"

    # ------------------------------------------------------------ loading

    def load_model(self, model_dir: Path, device_id: int | None,
                   slot_name: str, poll: bool | None = None,
                   profile=None,
                   config_file: str = DEFAULT_DIALOG_CONFIG) -> ModelAssets:
        """Fully loads a Genie model in isolation: config + dialog handle +
        tokenizer. Does NOT touch any Slot's state — the caller decides
        whether/when to adopt the result via Slot.adopt()."""
        config_json, dcfg = load_dialog_config(
            model_dir, device_id, slot_name, self._htp_ext_cache_dir, poll,
            config_file)
        handle = self.lib.create_dialog(config_json, profile)
        sampler_cfg = dcfg.get("sampler", {}) or {}
        template = templates.detect_template(
            self.config.chat_template_override or model_dir.name)
        return ModelAssets(
            handle=handle,
            config_json=config_json,
            dialog_cfg=dcfg,
            tokenizer=_load_tokenizer(dcfg),
            template=template,
            tool_format=tool_formats.get(
                self.config.tool_format_override or tool_formats.detect(template)),
            model_dir=model_dir,
            sampler_defaults={k: sampler_cfg[k] for k in _SAMPLER_DEFAULT_KEYS
                              if k in sampler_cfg},
        )

    def load_all(self) -> None:
        """Startup initialization — one GenieDialog per configured slot.
        Raises on the first failure (startup is the only fatal path)."""
        for spec in self.config.text_slots:
            slot = Slot(name=spec.name, device_id=spec.device_id,
                        model_root=spec.model_root, poll=spec.poll,
                        config_file=spec.config_file)
            if self.config.genie_profile:
                # Must exist before the dialog config it binds to.
                slot.profile = self.lib.create_profile()
                logger.info(f"[{slot.name}] GenieProfile bound (GENIE_PROFILE=true)")
            assets = self.load_model(spec.model_root, spec.device_id, spec.name,
                                     spec.poll, slot.profile, spec.config_file)
            slot.adopt(assets)
            self.slots.append(slot)
            self.status[slot.name] = {"phase": "idle", "detail": ""}
            logger.info(
                f"Slot '{slot.name}' ready: model={slot.active_model_id} "
                f"device_id={slot.device_id if slot.device_id is not None else '(unpinned)'} "
                f"template={slot.chat_template}")
        self._by_name = {s.name: s for s in self.slots}
        self.reindex()

    def reindex(self) -> None:
        """Rebuilds the active_model_id -> Slot lookup. Call after any slot's
        active_model_id changes. If two slots hold models with the same id,
        the later one in `slots` order wins — route by slot name (the
        request 'slot' field) if that ambiguity matters."""
        self._by_model_id = {s.active_model_id: s for s in self.slots}

    def free_all(self) -> None:
        for s in self.slots:
            self.lib.free_dialog(s.handle)

    # ------------------------------------------------------------ routing

    def select(self, model_name: str) -> Slot:
        """Routes by the request's 'model' field; falls back to the primary
        slot for lm_eval's fixed placeholder ("genie-local") or any name
        that doesn't match a loaded model — a single-slot deployment needs
        no client changes at all."""
        self._require_text_slots()
        return self._by_model_id.get(model_name, self.slots[0])

    def _require_text_slots(self) -> None:
        """A VLM-only deployment ("TEXT_SLOTS": []) has no text slot to fall
        back to — say so instead of raising IndexError on self.slots[0]."""
        if not self.slots:
            raise UnknownSlotError(
                "This server has no text slots configured (VLM_SLOTS only). "
                "Send a chat request with an image_url content part to reach "
                f"a VLM slot: {sorted(s.name for s in self.vlm_slots)}")

    def select_by_name(self, name: str) -> Slot:
        """Routes by slot name ('chat', 'tool_call', ... whatever the config
        calls them) — the slot's logical address rather than a loaded model
        identity. Empty name = primary."""
        if not name:
            self._require_text_slots()
            return self.slots[0]
        slot = self._by_name.get(name)
        if slot is None:
            raise UnknownSlotError(
                f"Unknown slot '{name}'. Known slots: {sorted(self._by_name)}")
        return slot

    def select_for_request(self, body: dict, model_name: str) -> Slot:
        """An explicit body['slot'] (hardware slot name) takes priority over
        'model' matching — needed when two slots load the same model
        directory, so 'model'-based routing can't tell them apart."""
        slot_name = body.get("slot", "")
        if slot_name:
            return self.select_by_name(slot_name)
        return self.select(model_name)

    @staticmethod
    def require_loaded(slot: Slot) -> None:
        """The only way a slot has no handle post-startup is a failed
        unload_first model switch — reject with a clear 503 instead of
        letting a null handle crash into a ctypes call."""
        if slot.handle is None:
            raise SlotNotLoadedError(
                f"Slot '{slot.name}' has no model loaded (a previous unload_first "
                f"model switch failed after freeing the old model). Retry "
                f"POST /v1/models/switch for this slot.")

    # ------------------------------------------------------------ VLM routing

    def select_vlm(self, model_name: str):
        """Same convention as select(), over the VLM registry. Returns None
        when no VLM slots are configured (a "not a VLM deployment" case the
        caller distinguishes from "named VLM slot not found")."""
        if not self.vlm_slots:
            return None
        by_id = {s.active_model_id: s for s in self.vlm_slots}
        return by_id.get(model_name, self.vlm_slots[0])

    def select_vlm_for_request(self, body: dict, model_name: str):
        if not self.vlm_slots:
            return None
        slot_name = body.get("slot", "")
        if slot_name:
            by_name = {s.name: s for s in self.vlm_slots}
            vslot = by_name.get(slot_name)
            if vslot is None:
                raise UnknownSlotError(
                    f"Unknown VLM slot '{slot_name}'. Known VLM slots: {sorted(by_name)}")
            return vslot
        return self.select_vlm(model_name)

    # ------------------------------------------------------------ hot-swap

    def switch_model(self, slot: Slot, model_dir: Path, unload_first: bool) -> None:
        """Hot-swaps the model loaded into one slot. Caller must hold
        slot.lock.

        Default (unload_first=True): the old handle is freed first, so the new
        model loads with the device to itself. This is the reliable order, and
        the cost is that a failed load leaves the slot with NO model until a
        later switch succeeds.

        unload_first=False loads the new model BEFORE freeing the old handle,
        so a bad model_dir leaves the slot running its previous model. It needs
        room for both on that HTP device at once, and on the SA8255P board that
        overlap is not dependable: one swap that succeeded 6 times out of 6 in
        one run failed every time in another, decided by device state the host
        cannot see. Use it only where the device has memory to spare and the
        swaps you actually perform have been tested."""
        if unload_first:
            old_handle, slot.handle = slot.handle, None
            self.lib.free_dialog(old_handle)
            logger.info(f"[{slot.name}] Freed previous model before loading "
                        "(unload_first=true)")

        self.status[slot.name] = {"phase": "loading model", "detail": str(model_dir)}
        try:
            assets = self.load_model(model_dir, slot.device_id, slot.name, slot.poll,
                                     slot.profile, slot.config_file)
        except Exception:
            if unload_first:
                logger.error(f"[{slot.name}] Slot has NO model loaded (unload_first "
                             "freed the old one before this failure) — retry the switch.")
            raise

        old_handle_to_free = None if unload_first else slot.handle
        slot.adopt(assets)
        self.reindex()
        self.lib.free_dialog(old_handle_to_free)
        logger.info(f"[{slot.name}] Model switched: model={slot.active_model_id} "
                    f"template={slot.chat_template}")
