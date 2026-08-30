"""ctypes bindings for the GenieDialog C API, plus SDK constants.

Everything SDK-facing goes through the GenieLib class so the rest of the
server never touches ctypes directly — and so tests can substitute a fake
implementation with the same method surface (see tests/fake_genie.py).
"""

import ctypes
import json
import logging

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

# GenieDialog_Param_t (keys for GenieDialog_getValue)
PARAM_CONTEXT_OCCUPANCY = 0     # -> DATATYPE_UINT_32
PARAM_APPLIED_LORA_ADAPTER = 1  # -> DATATYPE_STRING


def status_to_finish_reason(ret: int, default: str = "stop") -> str:
    """Maps a GenieDialog_query return code to an OpenAI finish_reason.
    Only CONTEXT_EXCEEDED means "length"; other warnings keep the default."""
    if ret == WARNING_CONTEXT_EXCEEDED:
        return "length"
    return default


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
    model's own genie_config.json default (`defaults`, SDK-key form). A
    request that omits temperature therefore gets the model default back
    instead of inheriting whatever the previous request set.

    "type": "basic" is always included so that a preceding logprobs request
    (which switches the dialog's sampler to "custom" — see
    GenieLib.register_custom_sampler) can never leak custom mode into a
    normal request.

    temperature<=0 means greedy decoding (required for reproducible lm_eval
    scores). The runtime applyConfig path never recomputes the sampler's
    internal greedy flag, and temp=0 would blow up softmax(temp) — so greedy
    is implemented as top-k=1 with a safe temp instead.
    """
    params: dict[str, str] = {"type": "basic"}

    if temperature is not None and float(temperature) <= 0.0:
        params["temp"] = "1.0"
        params["top-k"] = "1"
    else:
        if temperature is not None:
            params["temp"] = str(float(temperature))
        elif "temp" in defaults:
            params["temp"] = str(float(defaults["temp"]))
        if top_k is not None:
            params["top-k"] = str(int(top_k))
        elif "top-k" in defaults:
            params["top-k"] = str(int(defaults["top-k"]))

    if top_p is not None:
        params["top-p"] = str(float(top_p))
    elif "top-p" in defaults:
        params["top-p"] = str(float(defaults["top-p"]))

    if seed is not None:
        params["seed"] = str(int(seed))
    return params


# ---------------------------------------------------------------- ctypes types

class DialogConfigHandle(ctypes.c_void_p):
    pass


class ProfileHandle(ctypes.c_void_p):
    """GenieProfile_Handle_t — collects per-query KPIs from the SDK itself."""


class DialogHandle(ctypes.c_void_p):
    pass


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

    # ------------------------------------------------------------ lifecycle

    def create_dialog(self, config_json: bytes,
                      profile: ProfileHandle | None = None) -> DialogHandle:
        """GenieDialogConfig_createFromJson + GenieDialog_create. Raises
        RuntimeError on failure; never exits the process.

        A profile handle can only be attached to the *config*, before the
        dialog exists (GenieDialog.h:216) — which is why enabling profiling
        needs a model reload rather than a runtime toggle."""
        cfg = DialogConfigHandle()
        ret = self._lib.GenieDialogConfig_createFromJson(config_json, ctypes.byref(cfg))
        if ret != STATUS_SUCCESS or not cfg.value:
            raise RuntimeError(f"GenieDialogConfig_createFromJson failed: {ret}")
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
        """Blocking GenieDialog_query. on_token(token_str, sentence_code) is
        invoked from inside the C call for every generated piece of text.
        Returns the Genie_Status_t (0 success, >0 warning, <0 error)."""

        def trampoline(token_bytes, code, _user_data):
            on_token(token_bytes.decode("utf-8", errors="ignore") if token_bytes else "",
                     code)

        cb = QUERY_CALLBACK(trampoline)  # local ref keeps it alive for the call
        return self._lib.GenieDialog_query(
            handle, text.encode("utf-8"), ctypes.c_int(sentence_code), cb, None)

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
        return value.stringValue.decode("utf-8", errors="ignore")

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
