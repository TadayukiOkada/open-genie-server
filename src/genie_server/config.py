"""env_config.json parsing and process-environment setup.

All keys are read once at startup into an immutable ServerConfig. Unknown keys
are ignored so a config written for a newer server version still loads.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "env_config.json"
# The dialog config inside a model directory. The SDK's examples use this
# name, but genie-app takes the path on its command line, so an export is free
# to call it anything — hence the per-slot "config_file" override.
DEFAULT_DIALOG_CONFIG = "genie_config.json"

# Default generated-token cap for a VLM slot (VLM_SLOTS[].max_tokens).
# Well under a typical 4096-token context so that the prompt plus a
# runaway generation cannot exhaust it — see VLMSlotSpec.max_tokens.
DEFAULT_VLM_MAX_TOKENS = 1024

# Upper bound for TEXT_SLOTS/VLM_SLOTS device_id. Deliberately generous: the
# only target measured here exposes two HTP devices (SA8255P, cDSP0/cDSP1),
# nothing reports the count to the host before a dialog is created, and parts
# with more are plausible. So this rejects values that are certainly wrong
# without pretending to know the real limit — see _parse_device_id.
MAX_DEVICE_ID = 7

# Target platforms this server knows how to set up. The differences are not
# cosmetic: each needs a different QAIRT ABI directory and a different DSP
# skel-library search path, and Android additionally needs the DSP-side C++
# runtime staged next to the skels (see _adsp_library_path).
PLATFORM_LINUX_OE = "linux-oe"
PLATFORM_ANDROID = "android"
KNOWN_PLATFORMS = (PLATFORM_LINUX_OE, PLATFORM_ANDROID)

# QAIRT ships one lib/ directory per ABI. Android is bionic, OE Linux is glibc;
# a library built for one will not load on the other.
QAIRT_ABI_DIR = {
    PLATFORM_LINUX_OE: "aarch64-oe-linux-gcc11.2",
    PLATFORM_ANDROID: "aarch64-android",
}


def detect_platform() -> str:
    """Guesses the target platform for TARGET_PLATFORM: "auto".

    Android reports itself as Linux to platform.system(), so that is no help.
    CPython built for Android defines sys.getandroidapilevel(); the /system
    check catches interpreters that do not (a glibc build running under a
    compatibility layer, for instance).
    """
    import sys
    if hasattr(sys, "getandroidapilevel"):
        return PLATFORM_ANDROID
    if os.path.isdir("/system/bin") and os.path.exists("/system/build.prop"):
        return PLATFORM_ANDROID
    return PLATFORM_LINUX_OE


def resolve_model_path(value, base: Path | None) -> Path:
    """Resolves a configured model location against MODELS_BASE_DIR.

    An absolute path is taken as given. A relative one is resolved against
    `base` when MODELS_BASE_DIR is set, and against the process working
    directory when it is not (the behaviour from before this key existed).
    The same rule covers TEXT_SLOTS/VLM_SLOTS `model_root` and the
    `model_dir` of POST /v1/models/switch, so one config can name every
    model by bare directory name.

    This is a base for relative paths, NOT a sandbox: an absolute path
    still resolves outside the tree, which is what keeps a one-off model
    elsewhere on the device loadable. Confining which directories a
    running server will open is a separate concern.
    """
    p = Path(value)
    if p.is_absolute() or base is None:
        return p.resolve()
    return (base / p).resolve()


@dataclass(frozen=True)
class SlotSpec:
    """One GenieDialog instance to create at startup (a "text slot")."""
    name: str
    device_id: int | None
    model_root: Path
    # None = leave the model bundle's QnnHtp.poll alone. See _parse_text_slots.
    poll: bool | None = None
    # Which file in model_root holds the Genie dialog config. Bundles from the
    # export tooling do not always call it genie_config.json.
    config_file: str = DEFAULT_DIALOG_CONFIG


@dataclass(frozen=True)
class VLMSlotSpec:
    """One GenieNode/GeniePipeline-based multimodal slot to create at startup."""
    name: str
    device_id: int | None
    model_root: Path
    spec: str
    # Hard cap on generated tokens, baked into the text-generator node config
    # as "max-num-tokens". The composable-pipeline API exposes no per-request
    # token limit and no abort, so without a cap a generation that never emits
    # EOS runs until the context is exhausted, which permanently wedges the
    # slot (see vlm.VLMSlot). 0 disables the cap (the SDK default, UINT32_MAX).
    max_tokens: int = DEFAULT_VLM_MAX_TOKENS


@dataclass(frozen=True)
class ServerConfig:
    sdk_root: str
    hexagon_version: str = "v73"
    prefix_cache_dir: str = "./prefix_cache"
    # Optional base directory every relative model path resolves against:
    # TEXT_SLOTS/VLM_SLOTS "model_root" at startup and the "model_dir" of
    # POST /v1/models/switch. None => relative paths are CWD-relative.
    # Absolute paths ignore it either way. See resolve_model_path.
    models_base_dir: Path | None = None
    # Force the chat-template family ("llama3" | "llama2" | "chatml") instead
    # of auto-detecting it from the model directory name.
    chat_template_override: str = ""
    # "" = derive each slot's tool dialect from its chat template. A name from
    # tool_formats.FORMATS forces it. See tool_formats.detect.
    tool_format_override: str = ""
    # Optional extra cap applied to unspecified max_tokens, on top of the
    # model's own context window. 0 = no extra cap (bound only by remaining
    # context space, matching Qualcomm's qai-appbuilder reference server).
    default_max_tokens_cap: int = 0
    # Watchdog limit for one GenieDialog_query call (seconds). Long
    # generations on slow targets may need this raised.
    inference_timeout_s: float = 120.0
    # Which target the process is running on: "auto", "linux-oe" or "android".
    # Selects the QAIRT ABI directory and the DSP search-path policy. "auto"
    # is right unless you are pointing at an SDK laid out for another target.
    target_platform: str = "auto"
    # Explicit path to libGenie.so; empty = <sdk_root>/lib/<abi>/libGenie.so
    # for the resolved platform.
    genie_lib_path: str = ""
    # Prompt scoring (echo+logprobs teacher forcing for lm_eval loglikelihood)
    # enabled at startup. Also toggleable at runtime via
    # POST /v1/server/prompt_logprobs. Default off: one scoring request
    # occupies its slot for len(prompt)/decode-rate seconds.
    prompt_logprobs: bool = False
    # Bind a GenieProfile to every text slot (SDK-side TTFT / prefill /
    # decode KPIs, read back via GET /v1/server/profile). Costs a model
    # reload to change, since the profiler binds to the dialog config.
    genie_profile: bool = False
    # Reject prompt-scoring requests longer than this many tokens.
    prompt_logprobs_max_tokens: int = 4096
    # Recover a tool call whose <tool_call> marker the model mangled or
    # omitted, by matching the JSON's "name" against the request's own tool
    # names. qwen3_4b_instruct_2507 w4a16 emits Cyrillic in place of the
    # <tool_call> token on half of its calls, and qwen3_0_6b drops the tags
    # entirely -- in both cases the JSON itself is correct and the caller
    # would otherwise get prose where their code reads message.tool_calls.
    #
    # OFF by default, because it conceals exactly the kind of defect this
    # server exists to expose: with it on, a bundle that cannot emit its own
    # marker measures as though it could, and measures differently on
    # /v1/chat/completions than on /v1/completions, which never sees the
    # recovery. Turn it on when you want the application to work despite the
    # bundle -- and read the score you took with it on as the application's,
    # not the model's.
    tool_call_recovery: bool = False
    host: str = "0.0.0.0"
    port: int = 8080
    text_slots: tuple[SlotSpec, ...] = field(default_factory=tuple)
    vlm_slots: tuple[VLMSlotSpec, ...] = field(default_factory=tuple)
    # Which kind of slot is created first at startup. Only matters when both
    # TEXT_SLOTS and VLM_SLOTS are non-empty; QAIRT 2.49 makes the choice
    # consequential in both directions:
    #
    #   "vlm-first" (default) sidesteps libGenie's process-global
    #     positional-encoding validator flags without any extra call.
    #   "text-first" needs GenieLib.reset_dialog_validator_flags() in between
    #     (bootstrap does this), but leaves the smaller model holding DSP
    #     resources first, which is what lets two models co-exist at all on a
    #     dual-NSP target -- loading a large model first makes the second
    #     GenieDialog_create fail with err 1002 regardless of device_id.
    #
    # Neither order fits a ~4B VLM alongside a text model on SA8255P; the
    # option exists for smaller VLMs where the total does fit.
    slot_load_order: str = "vlm-first"

    # Derived timeouts (kept relative to inference_timeout_s).
    @property
    def abort_drain_timeout_s(self) -> float:
        return 5.0

    @property
    def prefix_warmup_timeout_s(self) -> float:
        return max(60.0, self.inference_timeout_s / 2)

    @property
    def warmup_join_timeout_s(self) -> float:
        return 600.0

    @property
    def platform(self) -> str:
        """The resolved target platform — never "auto"."""
        if self.target_platform == "auto":
            return detect_platform()
        return self.target_platform

    @property
    def qairt_abi_dir(self) -> str:
        """The lib/<abi> directory of the SDK that matches this target."""
        return QAIRT_ABI_DIR[self.platform]

    def resolved_genie_lib_path(self) -> str:
        if self.genie_lib_path:
            return self.genie_lib_path
        return os.path.join(self.sdk_root, "lib", self.qairt_abi_dir, "libGenie.so")

    def _skel_dir(self) -> str:
        """The SDK's DSP-side skel directory. Named "unsigned" after the
        protection domain the skels run in, not after their signing state."""
        return f"{self.sdk_root}/lib/hexagon-{self.hexagon_version}/unsigned"

    def _adsp_library_path(self) -> str:
        """Builds ADSP_LIBRARY_PATH for the resolved platform.

        This is a DSP-side skel-library SEARCH path, not itself a device
        selector — device_id in the HTP backend extension config is what
        actually pins execution to a core (slots.pin_htp_device).

        The two platforms need genuinely different values:

        linux-oe: the SDK skels, the rootfs skel directory, and every cdspN
        image directory in use. The cdspN entries are built from the slots'
        device_ids rather than hardcoding cdsp0.

        android: the vendor skel directory FIRST, then the SDK skels. Order
        matters here. This is the layout Qualcomm documents for Android
        Automotive, and it is also where the DSP-side C++ runtime has to be:
        libc++.so.1 and libc++abi.so.1 must be copied into the SDK skel
        directory or the HTP backend fails to bring up a device at all
        ("Failed to create device: 14001"). On this platform they live in the
        QNX primary VM, or in the Linux guest under /dsp/image/dsp/cdsp0/.
        There are no cdspN entries: the guest reaches the DSP through virtio
        fastrpc and has no /dsp mount of its own.
        """
        if self.platform == PLATFORM_ANDROID:
            return f"/vendor/lib/rfsa/adsp;{self._skel_dir()};"
        device_ids = sorted({
            s.device_id for s in (*self.text_slots, *self.vlm_slots)
            if s.device_id is not None
        })
        cdsp_paths = ";".join(f"/dsp/image/dsp/cdsp{d}" for d in device_ids) \
            or "/dsp/image/dsp/cdsp0"
        return f"{self._skel_dir()};/usr/lib/rfsa/adsp;{cdsp_paths}"

    def apply_process_env(self) -> None:
        """Sets QAIRT/QNN/ADSP environment variables for libGenie and the DSP."""
        os.environ["QAIRT_SDK_ROOT"] = self.sdk_root
        os.environ["QNN_SDK_ROOT"] = self.sdk_root
        os.environ["ADSP_LIBRARY_PATH"] = self._adsp_library_path()
        if self.platform == PLATFORM_ANDROID:
            # libGenie pulls in vendor libraries (libcdsprpc.so and friends)
            # that are not in the default bionic search path for a binary run
            # out of /data. Setting this here is too late for the current
            # process's own loader, so the launcher must already have it —
            # this keeps child processes and dlopen() consistent with it.
            parts = [f"{self.sdk_root}/lib/{self.qairt_abi_dir}", "/vendor/lib64"]
            existing = os.environ.get("LD_LIBRARY_PATH", "")
            if existing:
                parts.append(existing)
            os.environ["LD_LIBRARY_PATH"] = ":".join(parts)


def _parse_target_platform(raw) -> str:
    value = raw.get("TARGET_PLATFORM", "auto")
    if value != "auto" and value not in KNOWN_PLATFORMS:
        raise ValueError(
            f"TARGET_PLATFORM must be \"auto\" or one of {list(KNOWN_PLATFORMS)}, "
            f"got {value!r}")
    return value


def _parse_device_id(value, slot_name: str) -> int | None:
    """Validates a slot's "device_id" (which HTP device the slot loads on).

    None (absent) leaves the slot unpinned — the bundle's own
    devices[0].device_id applies. Anything else must be a non-negative
    integer within MAX_DEVICE_ID, which is a typo guard, not a hardware
    probe: how many HTP devices this SoC actually has is only known to QNN
    at dialog-create time, so device_id 1 on a single-NSP target still fails
    there (GenieDialog_create, err 1002). Catching the shape here keeps the
    obvious mistakes (a string, a float, a negative, "device_id": 10) from
    reaching slots.pin_htp_device, which writes the value into the HTP
    backend extension config unchecked.
    """
    if value is None:
        return None
    # bool is an int subclass; "device_id": true is a mistake, not device 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"slot {slot_name!r}: device_id must be an integer, got {value!r}")
    if not 0 <= value <= MAX_DEVICE_ID:
        raise ValueError(
            f"slot {slot_name!r}: device_id must be between 0 and "
            f"{MAX_DEVICE_ID}, got {value}")
    return value


def _parse_poll(value) -> bool | None:
    """None (absent) means "do not touch the bundle's setting"."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise ValueError(f"poll must be true or false, got {value!r}")


def _parse_text_slots(raw_cfg: dict,
                      models_base_dir: Path | None = None) -> tuple[SlotSpec, ...]:
    """TEXT_SLOTS — one entry per text model to keep resident.

    Multi-NSP, e.g. on SA8255P (two HTP devices, cDSP0/cDSP1). "name" is a
    logical address clients route to; "device_id" picks the core:
        "TEXT_SLOTS": [
          {"name": "tool_call", "device_id": 0, "model_root": "/models/tool-model"},
          {"name": "chat", "device_id": 1, "model_root": "/models/general"}
        ]
    Each slot gets its own GenieDialog handle, lock, and KV state, and — if
    device_id is set — is pinned to that HTP device by patching a per-slot
    copy of its model's HTP backend extension config (slots.pin_htp_device).

    Only "model_root" is required. A single-slot deployment is therefore
        "TEXT_SLOTS": [{"model_root": "/models/general"}]
    which names the slot "slot0" and leaves it unpinned. A relative
    model_root resolves under MODELS_BASE_DIR — see resolve_model_path.

    Each slot may also set "config_file" to name the dialog config inside
    model_root. It defaults to "genie_config.json", which is what the SDK's
    own examples use, but an export can name it after the model
    ("acme-7b-htp.json") — genie-app takes the path on the command
    line, so nothing forces the name. Point the slot at the file rather than
    copying it.

    Each slot may also set "poll": true/false to override
    dialog.engine.backend.QnnHtp.poll in that model's genie_config.json, or
    the top-level "POLL" key can set the default for every slot. Omitted
    (the default) leaves the bundle's own value untouched. poll:true makes
    the HTP backend busy-wait for the DSP: measured on SA8255P it costs
    ~260% CPU for latency indistinguishable from poll:false.

    Absent or empty means no text slots, which is a valid VLM-only
    deployment — the only shape that fits a target which cannot hold a VLM
    and a text model at once. load_config rejects a config with neither
    TEXT_SLOTS nor VLM_SLOTS.
    """
    default_poll = raw_cfg.get("POLL")
    raw = raw_cfg.get("TEXT_SLOTS") or []
    return tuple(
        SlotSpec(
            name=s.get("name", f"slot{i}"),
            device_id=_parse_device_id(s.get("device_id"), s.get("name", f"slot{i}")),
            model_root=resolve_model_path(s["model_root"], models_base_dir),
            poll=_parse_poll(s.get("poll", default_poll)),
            config_file=s.get("config_file") or DEFAULT_DIALOG_CONFIG,
        )
        for i, s in enumerate(raw)
    )


def _parse_vlm_slots(raw_cfg: dict,
                     models_base_dir: Path | None = None) -> tuple[VLMSlotSpec, ...]:
    """VLM_SLOTS — a separate, parallel key to TEXT_SLOTS (VLM slots use a
    completely different handle type and are never mixed into text slots):
        "VLM_SLOTS": [
          {"name": "vision", "device_id": 0, "model_root": "/models/qwen3-vl",
           "spec": "qwen3_vl"}
        ]
    model_root follows the same MODELS_BASE_DIR rule as TEXT_SLOTS.
    """
    raw = raw_cfg.get("VLM_SLOTS") or []
    return tuple(
        VLMSlotSpec(
            name=s.get("name", f"vlm{i}"),
            device_id=_parse_device_id(s.get("device_id"), s.get("name", f"vlm{i}")),
            model_root=resolve_model_path(s["model_root"], models_base_dir),
            spec=s.get("spec", "qwen3_vl"),
            max_tokens=int(s.get("max_tokens", DEFAULT_VLM_MAX_TOKENS)),
        )
        for i, s in enumerate(raw)
    )


SLOT_LOAD_ORDERS = ("vlm-first", "text-first")


def _parse_slot_load_order(raw: dict) -> str:
    """SLOT_LOAD_ORDER: "vlm-first" (default) | "text-first". See
    ServerConfig.slot_load_order for what the choice costs either way."""
    order = str(raw.get("SLOT_LOAD_ORDER", "vlm-first")).strip().lower()
    if order not in SLOT_LOAD_ORDERS:
        raise ValueError(
            f"SLOT_LOAD_ORDER must be one of {SLOT_LOAD_ORDERS}, got {order!r}")
    return order


def load_config(path: str = DEFAULT_CONFIG_PATH) -> ServerConfig:
    """Loads and validates env_config.json. Raises FileNotFoundError /
    json.JSONDecodeError / KeyError on a broken config file."""
    with open(path) as f:
        raw = json.load(f)

    models_base = raw.get("MODELS_BASE_DIR", "")
    models_base_dir = Path(models_base).resolve() if models_base else None
    text_slots = _parse_text_slots(raw, models_base_dir)
    vlm_slots = _parse_vlm_slots(raw, models_base_dir)
    if not text_slots and not vlm_slots:
        raise ValueError(
            f"{path}: no models configured. Set TEXT_SLOTS, VLM_SLOTS, or both. "
            'A single text model is TEXT_SLOTS: [{"model_root": "/path/to/model"}].')
    return ServerConfig(
        sdk_root=raw.get("QAIRT_SDK_ROOT", ""),
        hexagon_version=raw.get("HEXAGON_VERSION", "v73"),
        prefix_cache_dir=raw.get("PREFIX_CACHE_DIR", "./prefix_cache"),
        models_base_dir=models_base_dir,
        chat_template_override=raw.get("CHAT_TEMPLATE", ""),
        tool_format_override=raw.get("TOOL_FORMAT", ""),
        default_max_tokens_cap=int(raw.get("DEFAULT_MAX_TOKENS", 0)),
        inference_timeout_s=float(raw.get("INFERENCE_TIMEOUT", 120)),
        target_platform=_parse_target_platform(raw),
        genie_lib_path=raw.get("GENIE_LIB_PATH", ""),
        prompt_logprobs=bool(raw.get("PROMPT_LOGPROBS", False)),
        genie_profile=bool(raw.get("GENIE_PROFILE", False)),
        prompt_logprobs_max_tokens=int(raw.get("PROMPT_LOGPROBS_MAX_TOKENS", 4096)),
        tool_call_recovery=bool(raw.get("TOOL_CALL_RECOVERY", False)),
        host=raw.get("HOST", "0.0.0.0"),
        port=int(raw.get("PORT", 8080)),
        text_slots=text_slots,
        vlm_slots=vlm_slots,
        slot_load_order=_parse_slot_load_order(raw),
    )
