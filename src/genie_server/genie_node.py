"""ctypes bindings for the libGenie composable pipeline API (GenieNode/GeniePipeline).

Matches QAIRT 2.48/2.49's include/Genie/{GenieNode,GeniePipeline,GenieCommon}.h.
GenieNode_setData takes void*/size_t, so numpy arrays can be passed directly
without going through a file.

attach() takes the ctypes.CDLL handle genie-server.py has already loaded for
GenieDialog and adds the GenieNode_*/GeniePipeline_* signatures to that same
in-process library (never loads the CDLL a second time). For standalone use,
load() can load it fresh instead.
"""
import ctypes as C
import json
import os

# ---------------------------------------------------------------- constants

GENIE_STATUS_SUCCESS = 0

# GenieNode_IOName_t (GenieNode.h:56-104)
NODE_IO = {
    "TEXT_GENERATOR_TEXT_INPUT": 0,
    "TEXT_GENERATOR_EMBEDDING_INPUT": 1,
    "TEXT_GENERATOR_TEXT_OUTPUT": 2,
    "TEXT_GENERATOR_TOKEN_INPUT": 3,
    "TEXT_GENERATOR_TOKEN_OUTPUT": 4,
    "TEXT_ENCODER_TEXT_INPUT": 100,
    "TEXT_ENCODER_EMBEDDING_OUTPUT": 101,
    "TEXT_ENCODER_POOLED_OUTPUT": 102,
    "IMAGE_ENCODER_IMAGE_INPUT": 200,
    "IMAGE_ENCODER_EMBEDDING_OUTPUT": 201,
    "IMAGE_ENCODER_IMAGE_POS_SIN": 202,
    "IMAGE_ENCODER_IMAGE_POS_COS": 203,
    "IMAGE_ENCODER_IMAGE_FULL_ATTN_MASK": 204,
    "IMAGE_ENCODER_IMAGE_WINDOW_ATTN_MASK": 205,
    "WILDCARD": 1000,
}

# GenieNode_TextOutput_SentenceCode_t (GenieNode.h) — no REWIND/RESUME here,
# unlike GenieDialog_SentenceCode_t; the composable pipeline only has these 5.
SENTENCE_CODE = {0: "complete", 1: "begin", 2: "continue", 3: "end", 4: "abort"}

# GenieNode_TextOutput_Callback_t
TEXT_CALLBACK = C.CFUNCTYPE(C.c_int32, C.c_char_p, C.c_int, C.c_void_p)

_lib = None


def _install_signatures(lib) -> None:
    """Sets argtypes/restype for GenieNode_*/GeniePipeline_* on an already-
    loaded CDLL. Idempotent (re-setting the same signature is harmless), so
    it's safe to call this on a CDLL shared with the GenieDialog bindings."""
    Handle = C.c_void_p
    Status = C.c_int32

    sigs = [
        # config
        ("GenieNodeConfig_createFromJson", [C.c_char_p, C.POINTER(Handle)]),
        ("GenieNodeConfig_free", [Handle]),
        ("GeniePipelineConfig_createFromJson", [C.c_char_p, C.POINTER(Handle)]),
        ("GeniePipelineConfig_free", [Handle]),
        # node
        ("GenieNode_create", [Handle, C.POINTER(Handle)]),
        ("GenieNode_free", [Handle]),
        ("GenieNode_setData", [Handle, C.c_int, C.c_void_p, C.c_size_t, C.c_char_p]),
        ("GenieNode_getSampler", [Handle, C.POINTER(Handle)]),
        ("GenieNode_reset", [Handle]),
        # pipeline
        ("GeniePipeline_create", [Handle, C.POINTER(Handle)]),
        ("GeniePipeline_addNode", [Handle, Handle]),
        ("GeniePipeline_connect", [Handle, Handle, C.c_int, Handle, C.c_int]),
        ("GeniePipeline_execute", [Handle, C.c_void_p]),
        ("GeniePipeline_reset", [Handle]),
        ("GeniePipeline_free", [Handle]),
    ]
    for name, argtypes in sigs:
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = Status

    # setTextCallback needs the callback ctypes type as an argtype.
    lib.GenieNode_setTextCallback.argtypes = [Handle, C.c_int, TEXT_CALLBACK]
    lib.GenieNode_setTextCallback.restype = Status


def attach(existing_cdll) -> None:
    """Reuses an already-loaded libGenie.so CDLL (e.g. genie-server.py's
    `genie_lib`) instead of loading the shared library a second time."""
    global _lib
    _install_signatures(existing_cdll)
    _lib = existing_cdll


def load(path="libGenie.so"):
    """Loads libGenie standalone (for scripts not already holding a CDLL,
    e.g. vlm_stream.py). Prefer attach() inside genie-server.py."""
    global _lib
    if _lib is not None:
        return _lib
    lib = C.CDLL(path, mode=C.RTLD_GLOBAL)
    _install_signatures(lib)
    _lib = lib
    return _lib


def _get_lib():
    if _lib is None:
        raise RuntimeError("genie_node: call attach(existing_cdll) or load() first")
    return _lib


class GenieStatusError(RuntimeError):
    """A non-success Genie_Status_t from a GenieNode_*/GeniePipeline_* call.

    Carries the raw status so callers can tell a warning apart from a real
    failure — GENIE_STATUS_WARNING_CONTEXT_EXCEEDED (4) in particular means
    "the generation stopped because the context filled up", which is a
    finish_reason, not an error. Stays a RuntimeError so existing
    `except RuntimeError` handlers keep working.
    """

    def __init__(self, status, what):
        super().__init__(f"{what} failed: status={status}")
        self.status = status
        self.what = what


def _check(status, what):
    if status != GENIE_STATUS_SUCCESS:
        raise GenieStatusError(status, what)


# ---------------------------------------------------------------- wrappers


class Node:
    def __init__(self, config):
        """config: a path to e.g. img-enc-htp.json (str/PathLike, read and
        passed as-is), or an already-loaded/modified dict (e.g. after
        rewriting the extensions path for HTP pinning, to pass it without
        going through a file)."""
        lib = _get_lib()
        if isinstance(config, dict):
            cfg_str = json.dumps(config)
            # Node config JSON is always {"<node-name>": {...}} — use that
            # single top-level key as a human-readable name for error messages.
            self._name = next(iter(config), "<dict-config>")
        else:
            with open(config) as f:
                cfg_str = f.read()
            self._name = os.path.basename(str(config))

        cfg = C.c_void_p()
        _check(lib.GenieNodeConfig_createFromJson(cfg_str.encode(), C.byref(cfg)),
               f"GenieNodeConfig_createFromJson({self._name})")

        self._handle = C.c_void_p()
        try:
            _check(lib.GenieNode_create(cfg, C.byref(self._handle)),
                   f"GenieNode_create({self._name})")
        finally:
            lib.GenieNodeConfig_free(cfg)

        self._cb_ref = None      # keeps the callback alive (GC)
        self._keep = None        # keeps the last set_buffer() argument alive (GC)

    @property
    def handle(self):
        return self._handle

    def set_text(self, io_name, text):
        b = text.encode("utf-8")
        _check(_get_lib().GenieNode_setData(self._handle, NODE_IO[io_name], b, len(b), None),
               f"setData(text, {io_name})")

    def set_buffer(self, io_name, buf):
        """buf: a numpy array or bytes. Must be contiguous memory."""
        if hasattr(buf, "__array_interface__"):
            import numpy as np
            arr = np.ascontiguousarray(buf)
            ptr = arr.ctypes.data_as(C.c_void_p)
            size = arr.nbytes
            self._keep = arr                 # keep alive for the duration of the call (GC)
        else:
            ptr = C.cast(C.c_char_p(buf), C.c_void_p)
            size = len(buf)
            self._keep = buf
        _check(_get_lib().GenieNode_setData(self._handle, NODE_IO[io_name], ptr, size, None),
               f"setData(buffer, {io_name})")

    def set_text_callback(self, io_name, fn):
        """fn(text: str, code: str) -> None"""
        def _trampoline(response, code, user_data):
            try:
                fn(response.decode("utf-8", "replace") if response else "",
                   SENTENCE_CODE.get(code, str(code)))
            except Exception as e:                     # never let an exception cross into C
                print(f"[genie_node callback error] {e}")
            return GENIE_STATUS_SUCCESS

        self._cb_ref = TEXT_CALLBACK(_trampoline)      # keep a reference alive
        _check(_get_lib().GenieNode_setTextCallback(self._handle, NODE_IO[io_name], self._cb_ref),
               f"setTextCallback({io_name})")

    def get_sampler(self):
        """Returns the raw GenieNode_getSampler handle (c_void_p). Callers
        use it with GenieSamplerConfig_* / GenieSampler_applyConfig."""
        sampler_h = C.c_void_p()
        _check(_get_lib().GenieNode_getSampler(self._handle, C.byref(sampler_h)),
               "GenieNode_getSampler")
        return sampler_h

    def reset(self):
        _check(_get_lib().GenieNode_reset(self._handle), f"GenieNode_reset({self._name})")

    def free(self):
        if self._handle:
            _get_lib().GenieNode_free(self._handle)
            self._handle = None


class Pipeline:
    def __init__(self, config_json=None):
        """config_json: dict or JSON string. genie-app's
        'pipeline config create pipelineConfig' takes no argument, so the
        default is empty."""
        lib = _get_lib()
        if config_json is None:
            cfg_str = "{}"
        elif isinstance(config_json, dict):
            cfg_str = json.dumps(config_json)
        else:
            cfg_str = config_json

        cfg = C.c_void_p()
        _check(lib.GeniePipelineConfig_createFromJson(cfg_str.encode(), C.byref(cfg)),
               "GeniePipelineConfig_createFromJson")

        self._handle = C.c_void_p()
        try:
            _check(lib.GeniePipeline_create(cfg, C.byref(self._handle)),
                   "GeniePipeline_create")
        finally:
            lib.GeniePipelineConfig_free(cfg)

    def add(self, node):
        _check(_get_lib().GeniePipeline_addNode(self._handle, node.handle),
               f"addNode({node._name})")

    def connect(self, producer, producer_io, consumer, consumer_io):
        _check(_get_lib().GeniePipeline_connect(
            self._handle, producer.handle, NODE_IO[producer_io],
            consumer.handle, NODE_IO[consumer_io]),
            f"connect({producer_io} -> {consumer_io})")

    def execute(self):
        _check(_get_lib().GeniePipeline_execute(self._handle, None), "execute")

    def reset(self):
        _check(_get_lib().GeniePipeline_reset(self._handle), "reset")

    def free(self):
        if self._handle:
            _get_lib().GeniePipeline_free(self._handle)
            self._handle = None
