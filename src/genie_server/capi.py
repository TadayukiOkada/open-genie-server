"""ctypes bindings for the GenieDialog C API, plus SDK constants.

Everything SDK-facing goes through the GenieLib class so the rest of the
server never touches ctypes directly — and so tests can substitute a fake
implementation with the same method surface (see tests/fake_genie.py).
"""

import codecs
import ctypes
import json
import logging
import re
import secrets
import threading

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- constants

# GenieDialog_SentenceCode_t
SENTENCE_COMPLETE = 0  # Entire prompt/response in one call (input & output)
SENTENCE_BEGIN = 1     # First segment — prefill without generating
SENTENCE_CONTINUE = 2  # Intermediate token
SENTENCE_END = 3       # Final segment / final token
SENTENCE_ABORT = 4     # Aborted by GenieDialog_signal
SENTENCE_REWIND = 5    # KV cache rewind for prefix match
SENTENCE_RESUME = 6    # Resumed after pause

TERMINAL_SENTENCE_CODES = frozenset({SENTENCE_COMPLETE, SENTENCE_END, SENTENCE_ABORT})


def utf8_stream():
    """A decoder for one stream of token callbacks: feed(data, final) -> str.

    A callback's bytes are not guaranteed to end on a character boundary.
    The SDK's own detokenizer holds back an incomplete sequence in the cases
    we could read, but a split that reached us used to be dropped by
    errors="ignore" (the dialog path) or turned into U+FFFD at the split
    (the VLM path). Here the bytes are joined across callbacks; what is
    still incomplete at the end of the stream (final=True, then the decoder
    starts over), or plain invalid, shows up as U+FFFD instead of vanishing.
    A callback whose bytes are all held back yields "".
    """
    decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def feed(data: bytes | None, final: bool) -> str:
        text = decoder.decode(data or b"", final)
        if final:
            decoder.reset()
        return text

    return feed

# GenieDialog_Action_t
ACTION_ABORT = 0x01
ACTION_PAUSE = 0x02

# Genie_Status_t: 0 = SUCCESS, >0 = warning, <0 = error
STATUS_SUCCESS = 0

# Warning codes (positive Genie_Status_t returns from GenieDialog_query).
# Only WARNING_CONTEXT_EXCEEDED actually means "hit a length limit" — the
# others must NOT be reported as finish_reason="length".
WARNING_ABORTED = 1
WARNING_BOUND_HANDLE = 2
WARNING_PAUSED = 3
WARNING_CONTEXT_EXCEEDED = 4

# The SDK's "unlimited" value for GenieDialog_setMaxNumTokens. IMPORTANT: 0
# does NOT mean unlimited on this SDK — the generation loop does
# `keepGoing = ++genTokenCount < m_tokenLimit`, so with 0 generation stops
# after exactly one token. The SDK's own default is UINT32_MAX.
MAX_NUM_TOKENS_UNLIMITED = 0xFFFFFFFF

# GenieLog_Level_t (GenieLog.h). A logger must be created and bound to a
# dialog config for libGenie to emit anything at all: every __INFO/__ERROR in
# the SDK expands to _LOG(_env->logger(), ...), which is a no-op while the
# logger is null. Without this the server sees none of the SDK's own
# diagnostics -- not even its errors.
LOG_LEVELS = {
    "error":   1,
    "warn":    2,
    "info":    3,
    "verbose": 4,
}
LOG_LEVEL_NAMES = {v: k for k, v in LOG_LEVELS.items()}

# Genie_PerformancePolicy_t
PERFORMANCE_POLICIES = {
    "burst":                      10,
    "sustained_high_performance": 20,
    "high_performance":           30,
    "balanced":                   40,
    "low_balanced":               50,
    "high_power_saver":           60,
    "power_saver":                70,
    "low_power_saver":            80,
    "extreme_power_saver":        90,
}
PERFORMANCE_POLICY_NAMES = {v: k for k, v in PERFORMANCE_POLICIES.items()}

# Genie_DataType_t
DATATYPE_UINT_32 = 12
DATATYPE_STRING = 30

# GenieCommon.h; in every QAIRT release this server supports, but looked up
# rather than assumed, like GenieDialog_setGrammar.
API_VERSION_FUNCTIONS = ("Genie_getApiMajorVersion", "Genie_getApiMinorVersion",
                         "Genie_getApiPatchVersion")

# The Genie C API that QAIRT 2.52.0 ships (GENIE_API_VERSION 1.22.0). A library
# older than this still works, but every path that exists only to make up for
# what it lacks is deprecated (see GenieLib.legacy_runtime).
CURRENT_API = (1, 22)

# Added in QAIRT 2.52.0 (API 1.22.0). Each is looked up, not assumed, so a
# 2.51 or older libGenie still loads; the GenieLib.has_* properties say which.
NATIVE_LOGPROBS_FUNCTIONS = ("GenieSampler_registerLogprobsCallback",)
PREFIX_CACHE_FLUSH_FUNCTIONS = ("GenieDialog_flushPrefixCache",)
PROFILE_CLEAR_FUNCTIONS = ("GenieProfile_clear",)
STANDALONE_TOKENIZER_FUNCTIONS = (
    "GenieTokenizerConfig_createFromJson", "GenieTokenizerConfig_free",
    "GenieTokenizer_create", "GenieTokenizer_free", "GenieTokenizer_encode",
    "GenieTokenizer_decode", "GenieTokenizer_decodeToSubWords")

# GenieDialog_Param_t (keys for GenieDialog_getValue)
PARAM_CONTEXT_OCCUPANCY = 0     # -> DATATYPE_UINT_32
PARAM_APPLIED_LORA_ADAPTER = 1  # -> DATATYPE_STRING


def status_to_finish_reason(ret: int, default: str = "stop") -> str:
    """Maps a GenieDialog_query return code to an OpenAI finish_reason.
    Only CONTEXT_EXCEEDED means "length"; other warnings keep the default."""
    if ret == WARNING_CONTEXT_EXCEEDED:
        return "length"
    return default


# What the SDK's basic sampler uses for a key that neither the model config nor
# a request sets. Sent explicitly for every key so nothing is inherited from an
# earlier request. seed is not here: see make_sampler_params.
SDK_SAMPLER_DEFAULTS = {"temp": 0.1, "top-k": 0, "top-p": 0.8}


def make_sampler_params(defaults: dict, temperature=None, top_p=None, top_k=None,
                        seed=None) -> dict[str, str]:
    """Builds the {sdk_key: value_string} dict for GenieSamplerConfig_setParam.

    The SDK's setParam whitelist is "temp"/"top-k"/"top-p"/"seed" (NOT the
    OpenAI names "temperature"/"top_p"/"top_k" — passing those throws
    GENIE_STATUS_ERROR_JSON_SCHEMA inside the SDK, so none of them ever took
    effect; see Sampler.cpp SamplerConfig::setParam).

    Sampler state persists across requests on a GenieDialog handle
    (GenieSampler_applyConfig is a partial merge), so every request must
    re-apply a COMPLETE parameter set: request value if given, else the
    model's own genie_config.json default (`defaults`, SDK-key form), else
    the SDK's own default (SDK_SAMPLER_DEFAULTS). A key the config omits is
    still sent: otherwise the previous request's value (greedy's top-k=1, a
    seed) would survive into a request that never asked for it.

    seed is the exception to "else the model's default". It is the request's
    seed if it gave one, and otherwise a fresh random one, never the config's:
    applying a seed re-seeds the sampler, so re-sending a config's "seed": 42
    on every request would give every unseeded request the same random
    stream. Repeated sampling (pass@k, self-consistency) would then return
    one answer n times, silently. A config's seed seeds the dialog once,
    when it is created; a caller who wants reproducible output sends seed.
    The random seed is drawn here rather than left to the SDK's own "unset"
    value, so what an unseeded request gets does not depend on how a given
    SDK version reads that value.

    "type": "basic" is always included so that a preceding logprobs request
    (which switches the dialog's sampler to "custom" — see
    GenieLib.register_custom_sampler) can never leak custom mode into a
    normal request.

    temperature<=0 means greedy decoding (required for reproducible lm_eval
    scores). The runtime applyConfig path never recomputes the sampler's
    internal greedy flag, and temp=0 would blow up softmax(temp) — so greedy
    is implemented as top-k=1 with a safe temp instead.
    """
    temp, k, p = resolve_sampler_values(defaults, temperature, top_p, top_k)
    return {"type": "basic", "temp": str(temp), "top-k": str(k),
            "top-p": str(p),
            "seed": str(secrets.randbits(31) if seed is None else int(seed))}


def resolve_sampler_values(defaults: dict, temperature=None, top_p=None,
                           top_k=None) -> tuple[float, int, float]:
    """(temp, top-k, top-p) a request actually samples with: its own value,
    else the model's genie_config.json default, else the SDK's; greedy
    (temperature <= 0) as top-k 1. make_sampler_params sends these to the
    SDK, and the logprobs path, which samples on the host, uses the same
    ones -- otherwise asking for logprobs would change what gets sampled."""
    base = {**SDK_SAMPLER_DEFAULTS,
            **{k: v for k, v in defaults.items() if k in SDK_SAMPLER_DEFAULTS}}
    p = float(base["top-p"] if top_p is None else top_p)
    if temperature is not None and float(temperature) <= 0.0:
        return 1.0, 1, p
    return (float(base["temp"] if temperature is None else temperature),
            int(base["top-k"] if top_k is None else top_k), p)


def sampler_defaults_from(sampler_cfg: dict | None) -> dict:
    """The keys of a config's "sampler" section that make_sampler_params
    treats as the model's defaults: temp, top-k and top-p. Not seed, which
    applies once, at dialog creation (see make_sampler_params)."""
    return {k: sampler_cfg[k] for k in SDK_SAMPLER_DEFAULTS
            if sampler_cfg and k in sampler_cfg}


# ---------------------------------------------------------------- ctypes types

class DialogConfigHandle(ctypes.c_void_p):
    pass


class ProfileHandle(ctypes.c_void_p):
    """GenieProfile_Handle_t — collects per-query KPIs from the SDK itself."""


class DialogHandle(ctypes.c_void_p):
    pass


class LogHandle(ctypes.c_void_p):
    """GenieLog_Handle_t — the sink every SDK log message goes through."""



class GenieValue(ctypes.Union):
    """Mirrors the Genie_Value_t union (GenieCommon.h)."""
    _fields_ = [
        ("int32Value", ctypes.c_int32),
        ("uint32Value", ctypes.c_uint32),
        ("uint64Value", ctypes.c_uint64),
        ("floatValue", ctypes.c_float),
        ("stringValue", ctypes.c_char_p),
    ]


QUERY_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p)

# Genie_AllocCallback_t: void (*)(const size_t size, const char** allocatedData).
# The SDK asks us for `size` bytes and expects the pointer written back through
# allocatedData[0]; it then fills that buffer with its result — the profile JSON
# for GenieProfile_getJsonData, the string value for GenieDialog_getValue (see
# GenieLib.get_profile_json / get_value_string). The buffer must stay alive
# until the SDK call itself returns.
ALLOC_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_size_t,
                                  ctypes.POINTER(ctypes.c_char_p))
# GenieSampler_UserDataCallback_t: (logitsSizeBytes, logits, numTokens,
# tokens[out], userData). The SDK dequantizes logits to float32 before the
# call (qualla custom_process applies scale/offset), so `logits` is always a
# float32 array of logitsSizeBytes/4 elements.
SAMPLER_CALLBACK = ctypes.CFUNCTYPE(
    None, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_int32), ctypes.c_void_p)
# GenieSampler_LogprobsCallback_t (API 1.22): (tokenId, logprob, numTopLogprobs,
# topTokenIds, topLogprobs, userData), once per accepted token of a basic
# dialog query. logprob is the raw model distribution's (log-softmax at
# temperature 1, before penalties), the same quantity logprobs.py computes from
# the custom sampler's logits. The arrays are valid only during the call.
LOGPROBS_CALLBACK = ctypes.CFUNCTYPE(
    None, ctypes.c_uint32, ctypes.c_float, ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_float), ctypes.c_void_p)


class GenieLib:
    """Thin, typed wrapper around libGenie.so's GenieDialog_* C API.

    All methods take/return plain Python types; ctypes stays inside this
    class. Methods that mutate per-dialog state must be called with the
    owning slot's lock held (the engine does this).
    """

    def __init__(self, cdll: ctypes.CDLL):
        self._lib = cdll
        # Custom-sampler callback registrations are process-global and
        # permanent in the SDK (a static name->callback map) — the ctypes
        # trampolines must stay alive for the process lifetime.
        self._sampler_callback_refs: dict[str, object] = {}
        # Native logprobs callbacks, per sampler handle (register_logprobs_callback).
        self._logprobs_callback_refs: dict[int, object] = {}
        self._bind_signatures()

    @classmethod
    def load(cls, so_path: str) -> "GenieLib":
        return cls(ctypes.CDLL(so_path))

    @property
    def cdll(self) -> ctypes.CDLL:
        """The raw CDLL — shared with genie_node.attach() for VLM support."""
        return self._lib

    def _bind_signatures(self) -> None:
        lib = self._lib

        def bind(fn, argtypes, restype):
            fn.argtypes = argtypes
            fn.restype = restype

        # Core dialog lifecycle
        bind(lib.GenieDialogConfig_createFromJson,
             [ctypes.c_char_p, ctypes.POINTER(DialogConfigHandle)], ctypes.c_int)
        bind(lib.GenieDialog_create,
             [DialogConfigHandle, ctypes.POINTER(DialogHandle)], ctypes.c_int)
        bind(lib.GenieDialogConfig_free, [DialogConfigHandle], None)
        bind(lib.GenieDialog_free, [DialogHandle], None)

        # Inference
        bind(lib.GenieDialog_query,
             [DialogHandle, ctypes.c_char_p, ctypes.c_int, QUERY_CALLBACK, ctypes.c_void_p],
             ctypes.c_int)
        bind(lib.GenieDialog_reset, [DialogHandle], ctypes.c_int)
        bind(lib.GenieDialog_signal, [DialogHandle, ctypes.c_int], ctypes.c_int)

        # Profiling (SDK-side KPIs: TTFT, prefill rate, decode rate)
        bind(lib.GenieProfile_create,
             [ctypes.c_void_p, ctypes.POINTER(ProfileHandle)], ctypes.c_int)
        bind(lib.GenieProfile_getJsonData,
             [ProfileHandle, ALLOC_CALLBACK, ctypes.POINTER(ctypes.c_char_p)],
             ctypes.c_int)
        bind(lib.GenieProfile_free, [ProfileHandle], ctypes.c_int)
        bind(lib.GenieDialogConfig_bindProfiler,
             [DialogConfigHandle, ProfileHandle], ctypes.c_int)

        # Logging. GenieLog_create's callback parameter is a
        # GenieLog_Callback_t taking a va_list, which ctypes cannot construct
        # or consume portably, so we always pass NULL and take the SDK's own
        # sink: printf to stdout on Linux, logcat on Android (Logger.cpp
        # picks it when the callback is null). Messages arrive as
        # "Genie:  <ms> [ LEVEL ] <text>".
        bind(lib.GenieLog_create,
             [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
              ctypes.POINTER(LogHandle)], ctypes.c_int)
        bind(lib.GenieLog_free, [LogHandle], ctypes.c_int)
        bind(lib.GenieDialogConfig_bindLogger,
             [DialogConfigHandle, LogHandle], ctypes.c_int)

        # Prefix caching (KV-cache snapshots)
        bind(lib.GenieDialog_save, [DialogHandle, ctypes.c_char_p], ctypes.c_int)
        bind(lib.GenieDialog_restore, [DialogHandle, ctypes.c_char_p], ctypes.c_int)

        # Per-request parameter control
        bind(lib.GenieDialog_setMaxNumTokens, [DialogHandle, ctypes.c_uint32], ctypes.c_int)
        bind(lib.GenieDialog_setStopSequence, [DialogHandle, ctypes.c_char_p], ctypes.c_int)

        # Sampler API for temperature / top_p / top_k control
        bind(lib.GenieDialog_getSampler,
             [DialogHandle, ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int)
        bind(lib.GenieSamplerConfig_createFromJson,
             [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int)
        bind(lib.GenieSamplerConfig_setParam,
             [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p], ctypes.c_int)
        bind(lib.GenieSampler_applyConfig, [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int)
        bind(lib.GenieSamplerConfig_free, [ctypes.c_void_p], None)

        # Custom sampler registration (logits access for logprobs)
        bind(lib.GenieSampler_registerUserDataCallback,
             [ctypes.c_char_p, SAMPLER_CALLBACK, ctypes.c_void_p], ctypes.c_int)

        # LoRA control
        bind(lib.GenieDialog_applyLora,
             [DialogHandle, ctypes.c_char_p, ctypes.c_char_p], ctypes.c_int)
        bind(lib.GenieDialog_setLoraStrength,
             [DialogHandle, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_float], ctypes.c_int)
        bind(lib.GenieDialog_releaseLoraMemory,
             [DialogHandle, ctypes.c_char_p, ctypes.c_char_p], ctypes.c_int)

        # State introspection
        bind(lib.GenieDialog_getValue,
             [DialogHandle, ctypes.c_int, ALLOC_CALLBACK,
              ctypes.POINTER(ctypes.c_int), ctypes.POINTER(GenieValue)], ctypes.c_int)

        # Performance policy
        bind(lib.GenieDialog_setPerformancePolicy, [DialogHandle, ctypes.c_int], ctypes.c_int)
        bind(lib.GenieDialog_getPerformancePolicy,
             [DialogHandle, ctypes.POINTER(ctypes.c_int)], ctypes.c_int)

        # Added in QAIRT 2.51.0 (API 1.21.0). Bound only when present, so an
        # older libGenie still loads; has_set_grammar says which it is.
        self._has_set_grammar = hasattr(lib, "GenieDialog_setGrammar")
        if self._has_set_grammar:
            bind(lib.GenieDialog_setGrammar,
                 [DialogHandle, ctypes.c_char_p, ctypes.c_char_p], ctypes.c_int)
        self._has_api_version = all(hasattr(lib, n) for n in API_VERSION_FUNCTIONS)
        if self._has_api_version:
            bind(lib.Genie_getApiMajorVersion, [], ctypes.c_uint32)
            bind(lib.Genie_getApiMinorVersion, [], ctypes.c_uint32)
            bind(lib.Genie_getApiPatchVersion, [], ctypes.c_uint32)

        # Added in QAIRT 2.52.0 (API 1.22.0); see the *_FUNCTIONS tuples.
        self._has_native_logprobs = all(hasattr(lib, n) for n in NATIVE_LOGPROBS_FUNCTIONS)
        if self._has_native_logprobs:
            bind(lib.GenieSampler_registerLogprobsCallback,
                 [ctypes.c_void_p, LOGPROBS_CALLBACK, ctypes.c_void_p], ctypes.c_int)
        self._has_prefix_cache_flush = all(
            hasattr(lib, n) for n in PREFIX_CACHE_FLUSH_FUNCTIONS)
        if self._has_prefix_cache_flush:
            bind(lib.GenieDialog_flushPrefixCache, [DialogHandle], ctypes.c_int)
        self._has_profile_clear = all(hasattr(lib, n) for n in PROFILE_CLEAR_FUNCTIONS)
        if self._has_profile_clear:
            bind(lib.GenieProfile_clear, [ProfileHandle], ctypes.c_int)
        self._has_standalone_tokenizer = all(
            hasattr(lib, n) for n in STANDALONE_TOKENIZER_FUNCTIONS)
        if self._has_standalone_tokenizer:
            bind(lib.GenieTokenizerConfig_createFromJson,
                 [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int)
            bind(lib.GenieTokenizerConfig_free, [ctypes.c_void_p], ctypes.c_int)
            bind(lib.GenieTokenizer_create,
                 [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int)
            bind(lib.GenieTokenizer_free, [ctypes.c_void_p], ctypes.c_int)
            bind(lib.GenieTokenizer_encode,
                 [ctypes.c_void_p, ctypes.c_char_p, ALLOC_CALLBACK,
                  ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),
                  ctypes.POINTER(ctypes.c_uint32)], ctypes.c_int)
            bind(lib.GenieTokenizer_decode,
                 [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32,
                  ALLOC_CALLBACK, ctypes.POINTER(ctypes.c_char_p)], ctypes.c_int)
            bind(lib.GenieTokenizer_decodeToSubWords,
                 [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32,
                  ALLOC_CALLBACK, ctypes.POINTER(ctypes.POINTER(ctypes.c_char_p))],
                 ctypes.c_int)

    # ------------------------------------------------------------ lifecycle

    def create_dialog(self, config_json: bytes,
                      profile: ProfileHandle | None = None,
                      log_handle: "LogHandle | None" = None) -> DialogHandle:
        """GenieDialogConfig_createFromJson + GenieDialog_create. Raises
        RuntimeError on failure; never exits the process.

        A profile handle can only be attached to the *config*, before the
        dialog exists (GenieDialog.h:216) — which is why enabling profiling
        needs a model reload rather than a runtime toggle. The logger binds
        the same way and has the same consequence."""
        cfg = DialogConfigHandle()
        ret = self._lib.GenieDialogConfig_createFromJson(config_json, ctypes.byref(cfg))
        if ret != STATUS_SUCCESS or not cfg.value:
            raise RuntimeError(f"GenieDialogConfig_createFromJson failed: {ret}")
        if log_handle is not None:
            ret = self._lib.GenieDialogConfig_bindLogger(cfg, log_handle)
            if ret != STATUS_SUCCESS:
                self._lib.GenieDialogConfig_free(cfg)
                raise RuntimeError(f"GenieDialogConfig_bindLogger failed: {ret}")
        if profile is not None:
            ret = self._lib.GenieDialogConfig_bindProfiler(cfg, profile)
            if ret != STATUS_SUCCESS:
                self._lib.GenieDialogConfig_free(cfg)
                raise RuntimeError(f"GenieDialogConfig_bindProfiler failed: {ret}")
        handle = DialogHandle()
        ret = self._lib.GenieDialog_create(cfg, ctypes.byref(handle))
        self._lib.GenieDialogConfig_free(cfg)
        if ret != STATUS_SUCCESS or not handle.value:
            raise RuntimeError(f"GenieDialog_create failed: {ret}")
        return handle

    def create_logger(self, level: str) -> LogHandle:
        """GenieLog_create at `level` ("error" | "warn" | "info" | "verbose"),
        using the SDK's own sink (see the binding note: the callback form
        takes a va_list, which ctypes cannot handle portably).

        The handle must be bound to a dialog config BEFORE the dialog is
        created — binding is what makes _env->logger() non-null inside
        libGenie, and everything the SDK logs is gated on that pointer.
        Nothing routes through Python's logging module: the SDK writes the
        lines itself, to this process's stdout."""
        code = LOG_LEVELS.get(level)
        if code is None:
            raise ValueError(
                f"Unknown Genie log level {level!r}; expected one of "
                f"{sorted(LOG_LEVELS)}")
        handle = LogHandle()
        ret = self._lib.GenieLog_create(None, None, code, ctypes.byref(handle))
        if ret != STATUS_SUCCESS or not handle.value:
            raise RuntimeError(f"GenieLog_create({level}) failed: {ret}")
        return handle

    def free_logger(self, log_handle: LogHandle | None) -> None:
        """Frees a log handle. Unlike GenieProfile_free this does not refuse
        while the handle is still bound — GenieLog_free only drops the SDK's
        own reference (Logger::reset), and each dialog holds a shared_ptr of
        its own, so an early free silently leaves those dialogs logging to a
        handle the caller thinks is gone. Free it after the dialogs."""
        if log_handle is not None and log_handle.value:
            self._lib.GenieLog_free(log_handle)

    def create_profile(self) -> ProfileHandle:
        """GenieProfile_create. The config handle is a documented placeholder
        that the SDK does not use yet (GenieProfile.h:24-27), so NULL."""
        handle = ProfileHandle()
        ret = self._lib.GenieProfile_create(None, ctypes.byref(handle))
        if ret != STATUS_SUCCESS or not handle.value:
            raise RuntimeError(f"GenieProfile_create failed: {ret}")
        return handle

    def get_profile_json(self, profile: ProfileHandle) -> str:
        """The SDK's own KPIs for the most recent query on the bound dialog:
        time-to-first-token, prompt-processing-rate, token-generation-rate,
        and the token counts behind them."""
        out = ctypes.c_char_p()
        held_buf = None  # keeps the ctypes buffer alive for the SDK write

        def allocate(size, allocated):
            nonlocal held_buf
            held_buf = ctypes.create_string_buffer(int(size))
            allocated[0] = ctypes.cast(held_buf, ctypes.c_char_p)

        cb = ALLOC_CALLBACK(allocate)
        ret = self._lib.GenieProfile_getJsonData(profile, cb, ctypes.byref(out))
        if ret != STATUS_SUCCESS or not out.value:
            raise RuntimeError(f"GenieProfile_getJsonData failed: {ret}")
        return out.value.decode("utf-8", "replace")

    def free_profile(self, profile: ProfileHandle) -> None:
        """Fails while the handle is still bound to a live dialog, so free the
        dialog first."""
        if profile is not None and profile.value:
            self._lib.GenieProfile_free(profile)

    def reset_dialog_validator_flags(self) -> None:
        """Clears libGenie's process-global positional-encoding validator flags.

        Dialog::Config::createConfigFromJson resets rope_theta_set and
        position_dim_set on its first two lines, BEFORE it parses the string
        (Dialog.cpp:2942-2944), and GenieDialogConfig_createFromJson does
        nothing but construct a Dialog::Config. So a deliberately invalid
        payload clears the flags and then returns an error we discard; no
        handle is produced, so there is nothing to free.

        Why this is needed: creating a GenieDialog for a text model whose
        backend config uses "pos-id-dim"/"rope-theta" leaves those flags set
        for the whole process. pipeline::TextGenerator calls
        Dialog::validateDialogConfig() directly rather than going through
        GenieDialogConfig_createFromJson, so it never clears them, and a
        later VLM text-generator node using "positional-encoding" is
        rejected with "Specify one config from pos-id-dim and
        positional-encoding". Calling this between the two makes text->VLM
        creation order viable.
        """
        cfg = DialogConfigHandle()
        ret = self._lib.GenieDialogConfig_createFromJson(b"{}", ctypes.byref(cfg))
        if ret == STATUS_SUCCESS and cfg.value:
            # Not expected -- an empty object fails schema validation -- but
            # do not leak the handle if a future SDK accepts it.
            self._lib.GenieDialogConfig_free(cfg)

    def free_dialog(self, handle) -> None:
        if handle is not None and handle.value:
            self._lib.GenieDialog_free(handle)

    # ------------------------------------------------------------ inference

    def query(self, handle, text: str, sentence_code: int, on_token) -> int:
        """Blocking GenieDialog_query. on_token(token_str, sentence_code,
        got_bytes=...) is invoked from inside the C call for every generated
        piece of text. Returns the Genie_Status_t (0 success, >0 warning, <0
        error).

        got_bytes says whether the SDK's callback carried any bytes, i.e.
        whether it was a generated token. It is not the same as a non-empty
        token_str: bytes that end mid-character are held back and arrive as
        "". Count tokens by got_bytes, or a generation that hit max_tokens
        with a split character in it reads as having stopped short."""

        feed = utf8_stream()

        def trampoline(token_bytes, code, _user_data):
            on_token(feed(token_bytes, code in TERMINAL_SENTENCE_CODES), code,
                     got_bytes=bool(token_bytes))

        cb = QUERY_CALLBACK(trampoline)  # local ref keeps it alive for the call
        ret = self._lib.GenieDialog_query(
            handle, text.encode("utf-8"), ctypes.c_int(sentence_code), cb, None)
        # A terminal code flushes the decoder. If the query returned without
        # one (an error), bytes still held back would vanish here; show them
        # as U+FFFD instead. Not a token of its own: its bytes were counted
        # when they arrived.
        tail = feed(None, True)
        if tail:
            on_token(tail, SENTENCE_CONTINUE, got_bytes=False)
        return ret

    def reset(self, handle) -> int:
        return self._lib.GenieDialog_reset(handle)

    def signal_abort(self, handle) -> int:
        return self._lib.GenieDialog_signal(handle, ctypes.c_int(ACTION_ABORT))

    def save_state(self, handle, path: str) -> int:
        return self._lib.GenieDialog_save(handle, path.encode())

    def restore_state(self, handle, path: str) -> int:
        return self._lib.GenieDialog_restore(handle, path.encode())

    # ------------------------------------------------------------ parameters

    def set_max_tokens(self, handle, max_tokens: int | None) -> None:
        """None/0 => the SDK's own "unlimited" (UINT32_MAX — never 0, see
        MAX_NUM_TOKENS_UNLIMITED)."""
        val = int(max_tokens) if max_tokens else MAX_NUM_TOKENS_UNLIMITED
        ret = self._lib.GenieDialog_setMaxNumTokens(handle, ctypes.c_uint32(val))
        if ret != STATUS_SUCCESS:
            logger.warning(f"GenieDialog_setMaxNumTokens({val}) failed: {ret}")

    def set_stop_sequences(self, handle, stop: list[str] | None) -> None:
        """The SDK expects a JSON OBJECT '{"stop-sequence": [...]}' — a bare
        JSON array parses but silently sets no stop sequences (Dialog.cpp
        reads only the "stop-sequence" key). "{}" clears. A non-null empty
        string is NOT safe (nlohmann parse_error on empty input)."""
        if not stop:
            payload = b"{}"
        else:
            payload = json.dumps({"stop-sequence": list(stop)}).encode("utf-8")
        ret = self._lib.GenieDialog_setStopSequence(handle, payload)
        if ret != STATUS_SUCCESS:
            logger.warning(f"GenieDialog_setStopSequence failed: {ret}")

    def apply_sampler_params(self, handle, params: dict[str, str]) -> None:
        """Applies SDK-keyed sampler params (see make_sampler_params) to the
        dialog's sampler. Gracefully logs-and-continues on failure."""
        if not params:
            return
        sampler_h = ctypes.c_void_p()
        if self._lib.GenieDialog_getSampler(handle, ctypes.byref(sampler_h)) != STATUS_SUCCESS:
            logger.warning("GenieDialog_getSampler failed — sampling params not applied")
            return
        self.apply_sampler_params_to_handle(sampler_h, params)

    def apply_sampler_params_to_handle(self, sampler_h, params: dict[str, str]) -> None:
        """Shared with the VLM path, which obtains sampler_h from
        GenieNode_getSampler instead of GenieDialog_getSampler."""
        cfg_h = ctypes.c_void_p()
        # SamplerConfig::SamplerConfig / validateSamplerConfig (Sampler.cpp)
        # require a top-level "sampler" key whose object in turn requires its
        # own mandatory "version" field (must be exactly 1) — a bare "{}" or
        # an empty "sampler" object both throw GENIE_STATUS_ERROR_JSON_SCHEMA
        # before any of the setParam calls below ever run.
        if self._lib.GenieSamplerConfig_createFromJson(
                b'{"sampler": {"version": 1}}', ctypes.byref(cfg_h)) != STATUS_SUCCESS:
            logger.warning("GenieSamplerConfig_createFromJson failed")
            return
        try:
            for k, v in params.items():
                ret = self._lib.GenieSamplerConfig_setParam(cfg_h, k.encode(), v.encode())
                if ret != STATUS_SUCCESS:
                    logger.warning(f"GenieSamplerConfig_setParam({k}={v}) failed: {ret}")
            ret = self._lib.GenieSampler_applyConfig(sampler_h, cfg_h)
            if ret != STATUS_SUCCESS:
                logger.warning(f"GenieSampler_applyConfig failed: {ret}")
            else:
                logger.debug(f"Sampling params applied: {params}")
        finally:
            self._lib.GenieSamplerConfig_free(cfg_h)

    def register_custom_sampler(self, name: str, on_logits) -> None:
        """Registers a custom sampler callback under `name` (process-global,
        permanent). A dialog switches to it via apply_sampler_params(handle,
        {"type": "custom", "callback-name": name}) and back with
        {"type": "basic", ...}.

        on_logits(logits_addr: int, n_floats: int, num_tokens: int) ->
        list[int] receives the address of the SDK's dequantized float32
        logits array and must return the token id(s) to emit — i.e. the
        callback both observes the full distribution (logprobs) and decides
        the sampled/forced token. It runs on the SDK's inference thread and
        must never raise.
        """
        if name in self._sampler_callback_refs:
            return

        def trampoline(logits_size_bytes, logits_ptr, num_tokens, tokens_out, _user):
            try:
                ids = on_logits(logits_ptr, logits_size_bytes // 4, num_tokens)
            except Exception as e:
                logger.error(f"custom sampler '{name}' failed: {e}")
                ids = [0]
            for i in range(num_tokens):
                tokens_out[i] = int(ids[i] if i < len(ids) else ids[-1])

        cb = SAMPLER_CALLBACK(trampoline)
        ret = self._lib.GenieSampler_registerUserDataCallback(name.encode(), cb, None)
        if ret != STATUS_SUCCESS:
            raise RuntimeError(f"GenieSampler_registerUserDataCallback({name}) failed: {ret}")
        self._sampler_callback_refs[name] = cb

    # ------------------------------------------------------------ introspection

    def get_value_string(self, handle, key: int) -> str | None:
        """Reads a string-typed dialog param via GenieDialog_getValue. The SDK
        asks us (via the alloc callback) for a buffer of the size it needs,
        then writes the string into it before returning — the buffer must
        stay alive until GenieDialog_getValue itself returns."""
        held_buf = None  # keeps the ctypes buffer alive for the SDK write

        def _alloc(size, out_ptr):
            nonlocal held_buf
            held_buf = ctypes.create_string_buffer(size)
            out_ptr[0] = ctypes.cast(held_buf, ctypes.c_char_p)

        alloc_cb = ALLOC_CALLBACK(_alloc)
        dtype = ctypes.c_int()
        value = GenieValue()
        ret = self._lib.GenieDialog_getValue(
            handle, ctypes.c_int(key), alloc_cb, ctypes.byref(dtype), ctypes.byref(value))
        if ret != STATUS_SUCCESS:
            logger.warning(f"GenieDialog_getValue({key}) failed: {ret}")
            return None
        if dtype.value != DATATYPE_STRING or not value.stringValue:
            return ""
        return value.stringValue.decode("utf-8", errors="replace")

    def get_context_occupancy(self, handle) -> int | None:
        """The dialog's current KV-cache/context occupancy (tokens), or None."""

        def _noop_alloc(size, out_ptr):
            out_ptr[0] = None  # not a string param; callback must exist but is unused

        alloc_cb = ALLOC_CALLBACK(_noop_alloc)
        dtype = ctypes.c_int()
        value = GenieValue()
        ret = self._lib.GenieDialog_getValue(
            handle, ctypes.c_int(PARAM_CONTEXT_OCCUPANCY), alloc_cb,
            ctypes.byref(dtype), ctypes.byref(value))
        if ret != STATUS_SUCCESS or dtype.value != DATATYPE_UINT_32:
            return None
        return int(value.uint32Value)

    def get_applied_lora(self, handle) -> str:
        """Currently-applied LoRA adapter name, "" if none (base model)."""
        return self.get_value_string(handle, PARAM_APPLIED_LORA_ADAPTER) or ""

    # ------------------------------------------------------------ grammar

    @property
    def has_set_grammar(self) -> bool:
        """Whether this libGenie exports GenieDialog_setGrammar. Not whether
        it works: see grammar.probe_support."""
        return self._has_set_grammar

    def set_grammar(self, handle, kind: str | None, path: str | None) -> int:
        """GenieDialog_setGrammar: compiles the grammar in the file at `path`
        (kind "json-schema" | "regex" | "ebnf") and makes it the dialog's
        constraint from the next query on. (None, None) removes the
        constraint. The SDK prints its own reason for a failure to stderr."""
        if not self._has_set_grammar:
            raise RuntimeError("GenieDialog_setGrammar is not in this libGenie")
        return self._lib.GenieDialog_setGrammar(
            handle, kind.encode() if kind else None,
            path.encode() if path else None)

    def api_version(self) -> str | None:
        """Genie_getApi{Major,Minor,Patch}Version as "1.21.0", or None when
        the library does not export them."""
        if not self._has_api_version:
            return None
        return (f"{self._lib.Genie_getApiMajorVersion()}."
                f"{self._lib.Genie_getApiMinorVersion()}."
                f"{self._lib.Genie_getApiPatchVersion()}")

    def api_version_tuple(self) -> tuple[int, int, int] | None:
        """api_version() as (major, minor, patch), or None."""
        version = self.api_version()
        if version is None:
            return None
        major, minor, patch = (int(x) for x in version.split("."))
        return major, minor, patch

    @property
    def legacy_runtime(self) -> bool:
        """True for a libGenie older than CURRENT_API (QAIRT 2.51 and before).

        Such a library is still supported, but deprecated: the server falls
        back to the host-side paths it needed before 2.52 (logprobs through a
        custom sampler, its own on-disk prefix cache as the only one, the host
        tokenizer), and says so once at startup. A library that does not
        report its version at all counts as legacy."""
        version = self.api_version_tuple()
        return version is None or version[:2] < CURRENT_API

    # ------------------------------------------------------------ 2.52 additions

    @property
    def has_native_logprobs(self) -> bool:
        """GenieSampler_registerLogprobsCallback (API 1.22): the SDK samples and
        reports each accepted token's logprob, so a logprobs request no longer
        needs the custom sampler. Prompt scoring (teacher forcing) still does."""
        return self._has_native_logprobs

    @property
    def has_sdk_prefix_cache(self) -> bool:
        """The dialog config's "system_prompt_cache" (API 1.22). Detected by
        GenieDialog_flushPrefixCache, which ships with it: there is no symbol for
        a JSON key, and an older libGenie rejects the key at create time."""
        return self._has_prefix_cache_flush

    @property
    def has_profile_clear(self) -> bool:
        return self._has_profile_clear

    @property
    def has_standalone_tokenizer(self) -> bool:
        return self._has_standalone_tokenizer

    def get_sampler(self, handle):
        """GenieDialog_getSampler: the dialog-owned sampler handle, or None. The
        handle is the dialog's and must not be freed."""
        sampler_h = ctypes.c_void_p()
        if self._lib.GenieDialog_getSampler(handle, ctypes.byref(sampler_h)) != STATUS_SUCCESS:
            return None
        return sampler_h

    def register_logprobs_callback(self, sampler_h, on_logprobs) -> int:
        """GenieSampler_registerLogprobsCallback on a dialog's sampler.

        on_logprobs(token_id, logprob, top) gets the accepted token and a list
        of (token_id, logprob) alternatives, highest first (empty unless the
        sampler's "top-logprobs" is set). It runs on the SDK's inference thread
        and must not raise. None unregisters. The ctypes trampoline is kept
        alive per sampler handle until it is replaced or unregistered: the SDK
        holds only the raw function pointer."""
        if not self._has_native_logprobs:
            raise RuntimeError("GenieSampler_registerLogprobsCallback is not in this libGenie")
        key = sampler_h.value
        if on_logprobs is None:
            ret = self._lib.GenieSampler_registerLogprobsCallback(
                sampler_h, ctypes.cast(None, LOGPROBS_CALLBACK), None)
            if ret == STATUS_SUCCESS:
                self._logprobs_callback_refs.pop(key, None)
            return ret

        def trampoline(token_id, logprob, n_top, top_ids, top_lps, _user):
            try:
                top = [(int(top_ids[i]), float(top_lps[i])) for i in range(n_top)] \
                    if n_top and top_ids and top_lps else []
                on_logprobs(int(token_id), float(logprob), top)
            except Exception as e:  # never let an exception cross into C
                logger.error(f"logprobs callback failed: {e}")

        cb = LOGPROBS_CALLBACK(trampoline)
        ret = self._lib.GenieSampler_registerLogprobsCallback(sampler_h, cb, None)
        if ret == STATUS_SUCCESS:
            self._logprobs_callback_refs[key] = cb
        return ret

    def flush_prefix_cache(self, handle) -> int:
        """GenieDialog_flushPrefixCache: drops every entry of the SDK prefix
        cache partition this dialog belongs to (shared with any dialog of the
        same model geometry). A no-op when the dialog has none."""
        if not self._has_prefix_cache_flush:
            raise RuntimeError("GenieDialog_flushPrefixCache is not in this libGenie")
        return self._lib.GenieDialog_flushPrefixCache(handle)

    def clear_profile(self, profile: ProfileHandle) -> int:
        """GenieProfile_clear: empties what a profile handle has collected."""
        if not self._has_profile_clear:
            raise RuntimeError("GenieProfile_clear is not in this libGenie")
        return self._lib.GenieProfile_clear(profile)

    def create_tokenizer(self, tokenizer_path: str) -> "GenieTokenizer":
        """A standalone GenieTokenizer for a tokenizer.json (API 1.22). Raises
        RuntimeError on failure."""
        if not self._has_standalone_tokenizer:
            raise RuntimeError("GenieTokenizer_create is not in this libGenie")
        cfg_json = json.dumps({"standalone-tokenizer": {
            "version": 1, "tokenizer": {"version": 1, "path": tokenizer_path}}})
        cfg = ctypes.c_void_p()
        ret = self._lib.GenieTokenizerConfig_createFromJson(cfg_json.encode(), ctypes.byref(cfg))
        if ret != STATUS_SUCCESS or not cfg.value:
            raise RuntimeError(f"GenieTokenizerConfig_createFromJson failed: {ret}")
        try:
            handle = ctypes.c_void_p()
            ret = self._lib.GenieTokenizer_create(cfg, ctypes.byref(handle))
            if ret != STATUS_SUCCESS or not handle.value:
                raise RuntimeError(f"GenieTokenizer_create failed: {ret}")
        finally:
            self._lib.GenieTokenizerConfig_free(cfg)
        return GenieTokenizer(self._lib, handle, tokenizer_path)

    # ------------------------------------------------------------ LoRA

    def apply_lora(self, handle, engine: str, adapter: str) -> int:
        """Per SDK docs, a GenieDialog_reset must follow a LoRA switch."""
        ret = self._lib.GenieDialog_applyLora(handle, engine.encode(), adapter.encode())
        if ret == STATUS_SUCCESS:
            self._lib.GenieDialog_reset(handle)
        return ret

    def set_lora_strength(self, handle, engine: str, tensor: str, alpha: float) -> int:
        ret = self._lib.GenieDialog_setLoraStrength(
            handle, engine.encode(), tensor.encode(), ctypes.c_float(alpha))
        if ret == STATUS_SUCCESS:
            self._lib.GenieDialog_reset(handle)
        return ret

    def release_lora_memory(self, handle, engine: str, adapter: str) -> int:
        ret = self._lib.GenieDialog_releaseLoraMemory(
            handle, engine.encode(), adapter.encode())
        if ret == STATUS_SUCCESS:
            self._lib.GenieDialog_reset(handle)
        return ret

    # ------------------------------------------------------------ performance

    def set_performance_policy(self, handle, policy_value: int) -> int:
        return self._lib.GenieDialog_setPerformancePolicy(handle, ctypes.c_int(policy_value))

    def get_performance_policy(self, handle) -> int | None:
        val = ctypes.c_int()
        ret = self._lib.GenieDialog_getPerformancePolicy(handle, ctypes.byref(val))
        if ret != STATUS_SUCCESS:
            return None
        return val.value


# ---------------------------------------------------------------- tokenizer

def _gpt2_unicode_to_byte() -> dict[str, int]:
    """Inverse of GPT-2's bytes_to_unicode: the printable stand-in each byte
    gets in a byte-level BPE vocabulary ("Ġ" for a space, ...)."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_FALLBACK = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


class GenieTokenizer:
    """A standalone GenieTokenizer (QAIRT 2.52+) behind the slice of the HF
    `tokenizers.Tokenizer` interface this server uses: encode(text).ids and
    decode(ids, skip_special_tokens=False).

    encode is libGenie's own: it never adds special tokens (qualla adds BOS
    itself, see Slot.sdk_bos_token), which is exactly what the SDK prefills.
    HF's Tokenizer.encode adds them by default, so the two can differ by a
    BOS on tokenizers whose post-processor inserts one.

    decode does NOT use GenieTokenizer_decode: that call is stateful (it holds
    back a token whose bytes end mid-character and returns "" until the next
    call completes it), which would corrupt every later decode on a shared
    handle. It maps GenieTokenizer_decodeToSubWords' vocabulary spellings back
    to bytes instead -- byte-level BPE (GPT-2 stand-ins) or SentencePiece
    ("▁", <0xNN> byte fallback), chosen from tokenizer.json's decoder -- and
    decodes them as UTF-8 with replacement. A token that is only part of a
    character therefore decodes to U+FFFD, as with HF."""

    def __init__(self, cdll, handle, tokenizer_path: str):
        self._lib = cdll
        self._handle = handle
        self._lock = threading.Lock()  # libGenie's tokenizer is not reentrant across calls
        self._scheme = _subword_scheme(tokenizer_path)
        self._byte_map = _gpt2_unicode_to_byte() if self._scheme == "byte_level" else None

    class _Encoding:
        __slots__ = ("ids",)

        def __init__(self, ids):
            self.ids = ids

    def encode(self, text: str) -> "GenieTokenizer._Encoding":
        held = []

        def allocate(size, allocated):
            buf = ctypes.create_string_buffer(max(int(size), 1))
            held.append(buf)
            allocated[0] = ctypes.cast(buf, ctypes.c_char_p)

        cb = ALLOC_CALLBACK(allocate)
        ids_ptr = ctypes.POINTER(ctypes.c_int32)()
        n = ctypes.c_uint32()
        with self._lock:
            ret = self._lib.GenieTokenizer_encode(
                self._handle, text.encode("utf-8"), cb, ctypes.byref(ids_ptr), ctypes.byref(n))
        if ret != STATUS_SUCCESS:
            raise RuntimeError(f"GenieTokenizer_encode failed: {ret}")
        return self._Encoding([int(ids_ptr[i]) for i in range(n.value)])

    def subwords(self, ids) -> list[str]:
        """GenieTokenizer_decodeToSubWords: each id's vocabulary spelling."""
        ids = list(ids)
        if not ids:
            return []
        held = []

        def allocate(size, allocated):
            # One block for the pointer array and the strings: pointer-aligned.
            buf = (ctypes.c_void_p * ((int(size) + 7) // 8 + 1))()
            held.append(buf)
            allocated[0] = ctypes.cast(buf, ctypes.c_char_p)

        cb = ALLOC_CALLBACK(allocate)
        arr = (ctypes.c_int32 * len(ids))(*ids)
        out = ctypes.POINTER(ctypes.c_char_p)()
        with self._lock:
            ret = self._lib.GenieTokenizer_decodeToSubWords(
                self._handle, arr, len(ids), cb, ctypes.byref(out))
        if ret != STATUS_SUCCESS or not out:
            raise RuntimeError(f"GenieTokenizer_decodeToSubWords failed: {ret}")
        return [(out[i] or b"").decode("utf-8", "replace") for i in range(len(ids))]

    def token_bytes(self, token_id: int) -> bytes:
        """The bytes one token stands for (exact, even mid-character)."""
        return self._subword_bytes(self.subwords([token_id])[0])

    def _subword_bytes(self, word: str) -> bytes:
        if self._byte_map is not None:
            if all(ch in self._byte_map for ch in word):
                return bytes(self._byte_map[ch] for ch in word)
            return word.encode("utf-8")  # an added/special token, spelled as text
        if self._scheme == "sentencepiece":
            m = _BYTE_FALLBACK.match(word)
            if m:
                return bytes([int(m.group(1), 16)])
            return word.replace("▁", " ").encode("utf-8")
        return word.encode("utf-8")

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        # skip_special_tokens is accepted for interface parity; this server
        # always passes False.
        data = b"".join(self._subword_bytes(w) for w in self.subwords(ids))
        return data.decode("utf-8", "replace")

    def free(self) -> None:
        if self._handle is not None and self._handle.value:
            self._lib.GenieTokenizer_free(self._handle)
            self._handle = None


def _subword_scheme(tokenizer_path: str) -> str:
    """"byte_level" | "sentencepiece" | "plain", from tokenizer.json's decoder."""
    try:
        with open(tokenizer_path, encoding="utf-8") as f:
            decoder = json.load(f).get("decoder") or {}
    except (OSError, ValueError):
        return "plain"
    kinds = set()

    def walk(d):
        if isinstance(d, dict):
            kinds.add(d.get("type"))
            for sub in d.get("decoders") or []:
                walk(sub)

    walk(decoder)
    if "ByteLevel" in kinds:
        return "byte_level"
    if kinds & {"Metaspace", "ByteFallback", "Replace"}:
        return "sentencepiece"
    return "plain"
