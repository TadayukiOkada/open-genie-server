"""The ctypes layer (capi.GenieLib, genie_node) against a stub libGenie.

The rest of the suite replaces GenieLib wholesale (fake_genie.FakeGenieLib),
so the code that actually crosses into C -- argtypes, out-parameters,
callbacks the SDK calls back into, buffers that must outlive a call -- was
the least tested part of the server. Here GenieLib is real and only the C
functions are stubs: each one is a genuine CFUNCTYPE pointer built from the
prototype GenieLib itself binds, so every call goes through ctypes' own
argument conversion, exactly as it does against libGenie.so.
"""
import ctypes
import os
import re
import types
from pathlib import Path

import pytest

from genie_server import capi, genie_node

SRC = Path(capi.__file__).parent


class StubCDLL:
    """Stands in for ctypes.CDLL. Until a function is install()ed it is a
    placeholder that bind()/_install_signatures can set argtypes on; install
    swaps in a real function pointer with that prototype, calling `impl`."""

    def __init__(self):
        self.calls: list[tuple] = []
        self._keep: list = []

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        placeholder = types.SimpleNamespace(argtypes=None, restype=ctypes.c_int,
                                            name=name)
        object.__setattr__(self, name, placeholder)
        return placeholder

    def install(self, name, impl):
        ph = getattr(self, name)
        assert ph.argtypes is not None, f"{name} has no argtypes bound"

        def recording(*args):
            self.calls.append((name, args))
            result = impl(*args)
            return 0 if result is None and ph.restype is not None else result

        fp = ctypes.CFUNCTYPE(ph.restype, *ph.argtypes)(recording)
        fp.argtypes, fp.restype = ph.argtypes, ph.restype
        self._keep.append(fp)
        object.__setattr__(self, name, fp)
        return fp

    def called(self, name):
        return [args for n, args in self.calls if n == name]


def _value(x):
    """A c_void_p subclass arrives in a callback as the instance, a plain
    pointer as an int; either way, the address."""
    return getattr(x, "value", x)


@pytest.fixture
def stub():
    return StubCDLL()


@pytest.fixture
def lib(stub):
    return capi.GenieLib(stub)


# ---------------------------------------------------------------- every call has a signature

def _calls_in(source: str, pattern: str) -> set[str]:
    return set(re.findall(pattern, source))


def test_every_function_genielib_calls_has_argtypes():
    """An unbound function gets ctypes' defaults: every argument an int, the
    result a C int. On a 64-bit target that silently truncates handles and
    pointers, so a missing bind() is a crash waiting for the first pointer
    above 4 GiB."""
    src = (SRC / "capi.py").read_text()
    used = _calls_in(src, r"self\._lib\.(\w+)\(")
    bound = _calls_in(src, r"bind\(lib\.(\w+)")
    assert used and used <= bound, f"called without argtypes: {sorted(used - bound)}"


def test_every_function_genie_node_calls_has_argtypes():
    src = (SRC / "genie_node.py").read_text()
    used = _calls_in(src, r"_get_lib\(\)\.(\w+)\(")
    # The signature table, plus any function given argtypes on its own
    # (GenieNode_setTextCallback, whose argtype is the callback type).
    bound = (_calls_in(src, r'\("(\w+)", \[')
             | _calls_in(src, r"lib\.(\w+)\.argtypes\s*="))
    assert used and used <= bound, f"called without argtypes: {sorted(used - bound)}"


def _header_arities(include_dir: Path) -> dict[str, int]:
    arity = {}
    for header in include_dir.glob("Genie*.h"):
        text = re.sub(r"/\*.*?\*/|//[^\n]*", "", header.read_text(), flags=re.DOTALL)
        for name, params in re.findall(
                r"Genie_Status_t\s+(Genie\w+)\s*\(([^;{]*?)\)\s*;", text):
            params = params.strip()
            arity[name] = 0 if params in ("", "void") else params.count(",") + 1
    return arity


def _sdk_include_dir() -> Path | None:
    """QAIRT_SDK_ROOT (an unpacked SDK, as env_config.json points at), else
    the newest /opt/qcom/aistack/qairt/* install."""
    env_root = os.environ.get("QAIRT_SDK_ROOT")
    if env_root and (Path(env_root) / "include" / "Genie").is_dir():
        return Path(env_root) / "include" / "Genie"
    for root in sorted(Path("/opt/qcom/aistack/qairt").glob("*"), reverse=True):
        inc = root / "include" / "Genie"
        if inc.is_dir():
            return inc
    return None


def test_bound_arities_match_the_sdk_headers(stub):
    """Where a QAIRT SDK is installed, every bound prototype takes as many
    arguments as the public header declares. Skipped elsewhere (CI)."""
    inc = _sdk_include_dir()
    if inc is None:
        pytest.skip("no QAIRT SDK headers on this machine")
    arity = _header_arities(inc)
    capi.GenieLib(stub)
    genie_node._install_signatures(stub)
    checked = 0
    for name, ph in vars(stub).items():
        if name in arity and getattr(ph, "argtypes", None) is not None:
            assert len(ph.argtypes) == arity[name], name
            checked += 1
    assert checked >= 30


# ---------------------------------------------------------------- out-parameters and failure paths

def _write_handle(out, value):
    out[0] = value


def test_create_dialog_frees_its_config_once_and_returns_the_handle(stub, lib):
    stub.install("GenieDialogConfig_createFromJson",
                 lambda json_, out: _write_handle(out, 0x1111))
    stub.install("GenieDialog_create",
                 lambda cfg, out: _write_handle(out, 0x2222))
    stub.install("GenieDialogConfig_free", lambda cfg: 0)
    handle = lib.create_dialog(b'{"dialog": {}}')
    assert handle.value == 0x2222
    (json_arg, _), = stub.called("GenieDialogConfig_createFromJson")
    assert json_arg == b'{"dialog": {}}'
    assert [_value(a[0]) for a in stub.called("GenieDialog_create")] == [0x1111]
    assert [_value(a[0]) for a in stub.called("GenieDialogConfig_free")] == [0x1111]


def test_a_failed_create_still_frees_the_config(stub, lib):
    stub.install("GenieDialogConfig_createFromJson",
                 lambda json_, out: _write_handle(out, 0x1111))
    stub.install("GenieDialog_create", lambda cfg, out: -1)
    stub.install("GenieDialogConfig_free", lambda cfg: 0)
    with pytest.raises(RuntimeError, match="GenieDialog_create failed: -1"):
        lib.create_dialog(b"{}")
    assert len(stub.called("GenieDialogConfig_free")) == 1


def test_a_failed_logger_bind_frees_the_config_before_raising(stub, lib):
    stub.install("GenieDialogConfig_createFromJson",
                 lambda json_, out: _write_handle(out, 0x1111))
    stub.install("GenieDialogConfig_bindLogger", lambda cfg, log: -3)
    stub.install("GenieDialogConfig_free", lambda cfg: 0)
    with pytest.raises(RuntimeError, match="bindLogger failed: -3"):
        lib.create_dialog(b"{}", log_handle=capi.LogHandle(0x9))
    assert len(stub.called("GenieDialogConfig_free")) == 1


@pytest.mark.parametrize("method, fn", [
    ("free_dialog", "GenieDialog_free"),
    ("free_logger", "GenieLog_free"),
    ("free_profile", "GenieProfile_free"),
])
@pytest.mark.parametrize("handle", [None, ctypes.c_void_p(0)])
def test_freeing_nothing_never_reaches_the_sdk(stub, lib, method, fn, handle):
    stub.install(fn, lambda h: 0)
    getattr(lib, method)(handle)
    assert stub.called(fn) == []


# ---------------------------------------------------------------- callbacks the SDK calls into

def test_query_hands_the_sdk_utf8_and_joins_what_comes_back(stub, lib):
    def query(handle, text, code, cb, user_data):
        assert text == "日本".encode()
        assert code == capi.SENTENCE_COMPLETE
        cb("日".encode()[:2], capi.SENTENCE_CONTINUE, None)
        cb("日".encode()[2:], capi.SENTENCE_END, None)
        return 0

    stub.install("GenieDialog_query", query)
    got = []
    assert lib.query(capi.DialogHandle(0x1), "日本", capi.SENTENCE_COMPLETE,
                     lambda t, c, **kw: got.append((t, c))) == 0
    assert got == [("", capi.SENTENCE_CONTINUE), ("日", capi.SENTENCE_END)]


def _sdk_writes_string(alloc_cb, text: bytes):
    """What GenieDialog_getValue / GenieProfile_getJsonData do: ask the
    caller for a buffer, then write into it."""
    out = ctypes.c_char_p()
    alloc_cb(len(text) + 1, ctypes.byref(out))
    ptr = ctypes.cast(out, ctypes.c_void_p).value
    assert ptr, "the alloc callback returned no buffer"
    ctypes.memmove(ptr, text + b"\0", len(text) + 1)
    return ptr


def test_get_value_string_reads_what_the_sdk_wrote_into_our_buffer(stub, lib):
    def get_value(handle, key, alloc_cb, dtype, value):
        assert key == capi.PARAM_APPLIED_LORA_ADAPTER
        ptr = _sdk_writes_string(alloc_cb, "finetuned-ü".encode())
        dtype[0] = capi.DATATYPE_STRING
        value[0].stringValue = ctypes.cast(ptr, ctypes.c_char_p).value
        return 0

    stub.install("GenieDialog_getValue", get_value)
    assert lib.get_applied_lora(capi.DialogHandle(0x1)) == "finetuned-ü"


def test_get_context_occupancy_reads_the_uint32(stub, lib):
    def get_value(handle, key, alloc_cb, dtype, value):
        dtype[0] = capi.DATATYPE_UINT_32
        value[0].uint32Value = 3072
        return 0

    stub.install("GenieDialog_getValue", get_value)
    assert lib.get_context_occupancy(capi.DialogHandle(0x1)) == 3072


def test_get_profile_json_reads_the_sdk_written_buffer(stub, lib):
    def get_json(profile, alloc_cb, out):
        ptr = _sdk_writes_string(alloc_cb, b'{"components": []}')
        out[0] = ctypes.cast(ptr, ctypes.c_char_p).value
        return 0

    stub.install("GenieProfile_getJsonData", get_json)
    assert lib.get_profile_json(capi.ProfileHandle(0x1)) == '{"components": []}'


def test_a_custom_sampler_sees_float32_logits_and_writes_the_token(stub, lib,
                                                                  caplog):
    registered = {}
    stub.install("GenieSampler_registerUserDataCallback",
                 lambda name, cb, user: registered.update(name=name, cb=cb))
    seen = []

    def on_logits(addr, n, num_tokens):
        arr = ctypes.cast(addr, ctypes.POINTER(ctypes.c_float))
        seen.append([arr[i] for i in range(n)])
        return [2]

    lib.register_custom_sampler("s", on_logits)
    assert registered["name"] == b"s"
    logits = (ctypes.c_float * 4)(0.5, 1.5, 2.5, -1.0)
    out = (ctypes.c_int32 * 1)()
    registered["cb"](ctypes.sizeof(logits), ctypes.addressof(logits), 1, out, None)
    assert seen == [[0.5, 1.5, 2.5, -1.0]] and out[0] == 2

    # A raising callback must not cross into C: it logs and emits token 0.
    lib.register_custom_sampler("t", lambda *a: 1 / 0)
    with caplog.at_level("ERROR", logger="genie_server.capi"):
        registered["cb"](ctypes.sizeof(logits), ctypes.addressof(logits), 1, out,
                         None)
    assert out[0] == 0 and "custom sampler 't' failed" in caplog.text


# ---------------------------------------------------------------- scalar arguments

def test_lora_strength_goes_out_as_a_float_and_resets_only_on_success(stub,
                                                                     lib):
    stub.install("GenieDialog_setLoraStrength",
                 lambda h, engine, tensor, alpha: 0 if alpha == 0.5 else 7)
    stub.install("GenieDialog_reset", lambda h: 0)
    assert lib.set_lora_strength(capi.DialogHandle(1), "primary", "alpha0", 0.5) == 0
    (_, engine, tensor, alpha), = stub.called("GenieDialog_setLoraStrength")
    assert (engine, tensor, alpha) == (b"primary", b"alpha0", 0.5)
    assert len(stub.called("GenieDialog_reset")) == 1
    assert lib.set_lora_strength(capi.DialogHandle(1), "primary", "alpha0", 0.25) == 7
    assert len(stub.called("GenieDialog_reset")) == 1


@pytest.mark.parametrize("requested, sent", [(None, 0xFFFFFFFF),
                                             (0, 0xFFFFFFFF), (128, 128)])
def test_max_tokens_unset_is_the_sdks_unlimited_not_zero(stub, lib, requested,
                                                         sent):
    stub.install("GenieDialog_setMaxNumTokens", lambda h, n: 0)
    lib.set_max_tokens(capi.DialogHandle(1), requested)
    (_, n), = stub.called("GenieDialog_setMaxNumTokens")
    assert n == sent


def test_sampler_params_go_through_a_config_that_is_always_freed(stub, lib):
    stub.install("GenieDialog_getSampler",
                 lambda h, out: _write_handle(out, 0x5))
    stub.install("GenieSamplerConfig_createFromJson",
                 lambda j, out: _write_handle(out, 0x6))
    stub.install("GenieSamplerConfig_setParam", lambda cfg, k, v: 0)
    stub.install("GenieSampler_applyConfig", lambda s, cfg: -1)
    stub.install("GenieSamplerConfig_free", lambda cfg: 0)
    lib.apply_sampler_params(capi.DialogHandle(1), {"temp": "0.7", "top-k": "5"})
    assert [a[1:] for a in stub.called("GenieSamplerConfig_setParam")] == [
        (b"temp", b"0.7"), (b"top-k", b"5")]
    assert [_value(a[0]) for a in stub.called("GenieSamplerConfig_free")] == [0x6]


# ---------------------------------------------------------------- genie_node

@pytest.fixture
def node_stub(monkeypatch):
    stub = StubCDLL()
    genie_node._install_signatures(stub)
    monkeypatch.setattr(genie_node, "_lib", stub)
    stub.install("GenieNodeConfig_createFromJson",
                 lambda json_, out: _write_handle(out, 0x10))
    stub.install("GenieNodeConfig_free", lambda cfg: 0)
    return stub


def test_a_node_is_created_from_its_config_which_is_then_freed(node_stub):
    node_stub.install("GenieNode_create", lambda cfg, out: _write_handle(out, 0x20))
    node = genie_node.Node({"image-encoder": {"version": 1}})
    assert node.handle.value == 0x20
    (json_arg, _), = node_stub.called("GenieNodeConfig_createFromJson")
    assert json_arg == b'{"image-encoder": {"version": 1}}'
    assert [_value(a[0]) for a in node_stub.called("GenieNodeConfig_free")] == [0x10]


def test_a_failed_node_create_frees_the_config_and_names_the_node(node_stub):
    node_stub.install("GenieNode_create", lambda cfg, out: -5)
    with pytest.raises(genie_node.GenieStatusError, match="image-encoder"):
        genie_node.Node({"image-encoder": {}})
    assert len(node_stub.called("GenieNodeConfig_free")) == 1


def test_set_buffer_hands_the_sdk_contiguous_bytes_that_stay_alive(node_stub):
    """A non-contiguous view is copied; the copy is what the SDK reads, and
    the node keeps it referenced after the call."""
    np = pytest.importorskip("numpy")
    node_stub.install("GenieNode_create", lambda cfg, out: _write_handle(out, 0x20))
    received = []

    def set_data(handle, io, ptr, size, user):
        received.append(ctypes.string_at(ptr, size))
        return 0

    node_stub.install("GenieNode_setData", set_data)
    node = genie_node.Node({"n": {}})
    a = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]   # strided view
    node.set_buffer(next(iter(genie_node.NODE_IO)), a)
    assert received == [np.ascontiguousarray(a).tobytes()]
    assert node._keep is not None and node._keep.flags["C_CONTIGUOUS"]


def test_set_text_sends_utf8_with_its_byte_length(node_stub):
    node_stub.install("GenieNode_create", lambda cfg, out: _write_handle(out, 0x20))
    received = []
    node_stub.install("GenieNode_setData",
                      lambda h, io, ptr, size, user:
                      received.append(ctypes.string_at(ptr, size)))
    genie_node.Node({"n": {}}).set_text(next(iter(genie_node.NODE_IO)), "東京")
    assert received == ["東京".encode()]


# ---------------------------------------------------------------- the rest of GenieLib

def test_create_logger_passes_the_level_code_and_returns_the_handle(stub, lib):
    stub.install("GenieLog_create",
                 lambda cfg, cb, level, out: _write_handle(out, 0x31))
    handle = lib.create_logger("warn")
    assert handle.value == 0x31
    (cfg, cb, level, _), = stub.called("GenieLog_create")
    assert (cfg, cb, level) == (None, None, capi.LOG_LEVELS["warn"])
    with pytest.raises(ValueError, match="Unknown Genie log level"):
        lib.create_logger("loud")
    assert len(stub.called("GenieLog_create")) == 1   # refused before C


def test_create_logger_and_profile_refuse_a_null_handle(stub, lib):
    stub.install("GenieLog_create", lambda cfg, cb, level, out: 0)
    stub.install("GenieProfile_create", lambda cfg, out: 0)
    with pytest.raises(RuntimeError, match="GenieLog_create"):
        lib.create_logger("error")
    with pytest.raises(RuntimeError, match="GenieProfile_create"):
        lib.create_profile()


def test_create_profile_returns_the_handle(stub, lib):
    stub.install("GenieProfile_create", lambda cfg, out: _write_handle(out, 0x41))
    assert lib.create_profile().value == 0x41
    (cfg, _), = stub.called("GenieProfile_create")
    assert cfg is None


@pytest.mark.parametrize("stop, payload", [
    (None, b"{}"), ([], b"{}"),
    (["</s>", "東"], b'{"stop-sequence": ["</s>", "\\u6771"]}'),
])
def test_stop_sequences_go_out_as_a_json_object(stub, lib, stop, payload):
    """A bare JSON array parses but sets nothing, and an empty string is a
    parse error in the SDK; "{}" is what clears."""
    stub.install("GenieDialog_setStopSequence", lambda h, p: 0)
    lib.set_stop_sequences(capi.DialogHandle(1), stop)
    (_, sent), = stub.called("GenieDialog_setStopSequence")
    assert sent == payload


@pytest.mark.parametrize("method, fn", [
    ("apply_lora", "GenieDialog_applyLora"),
    ("release_lora_memory", "GenieDialog_releaseLoraMemory"),
])
def test_lora_calls_send_bytes_and_reset_only_on_success(stub, lib, method, fn):
    stub.install(fn, lambda h, engine, adapter: 0 if adapter == b"good" else 5)
    stub.install("GenieDialog_reset", lambda h: 0)
    call = getattr(lib, method)
    assert call(capi.DialogHandle(1), "primary", "good") == 0
    (_, engine, adapter), = stub.called(fn)
    assert (engine, adapter) == (b"primary", b"good")
    assert len(stub.called("GenieDialog_reset")) == 1
    assert call(capi.DialogHandle(1), "primary", "bad") == 5
    assert len(stub.called("GenieDialog_reset")) == 1


def test_performance_policy_round_trips_an_int(stub, lib):
    stub.install("GenieDialog_setPerformancePolicy", lambda h, v: 0)
    stored = {}

    def get(h, out):
        if "fail" in stored:
            return 9
        out[0] = 40
        return 0

    stub.install("GenieDialog_getPerformancePolicy", get)
    assert lib.set_performance_policy(capi.DialogHandle(1), 30) == 0
    (_, value), = stub.called("GenieDialog_setPerformancePolicy")
    assert value == 30
    assert lib.get_performance_policy(capi.DialogHandle(1)) == 40
    stored["fail"] = True
    assert lib.get_performance_policy(capi.DialogHandle(1)) is None


def test_reset_abort_save_and_restore_pass_their_arguments(stub, lib):
    for fn in ("GenieDialog_reset", "GenieDialog_signal", "GenieDialog_save",
               "GenieDialog_restore"):
        stub.install(fn, lambda *a: 0)
    h = capi.DialogHandle(0x7)
    lib.reset(h)
    lib.signal_abort(h)
    lib.save_state(h, "/tmp/kv/prefix_é.geniestate")
    lib.restore_state(h, "/tmp/kv/prefix_é.geniestate")
    assert [_value(a[0]) for a in stub.called("GenieDialog_reset")] == [0x7]
    assert [a[1] for a in stub.called("GenieDialog_signal")] == [capi.ACTION_ABORT]
    path = "/tmp/kv/prefix_é.geniestate".encode()
    assert [a[1] for a in stub.called("GenieDialog_save")] == [path]
    assert [a[1] for a in stub.called("GenieDialog_restore")] == [path]


# ---------------------------------------------------------------- the rest of genie_node

def test_node_sampler_reset_and_free(node_stub):
    node_stub.install("GenieNode_create", lambda cfg, out: _write_handle(out, 0x20))
    node_stub.install("GenieNode_getSampler", lambda h, out: _write_handle(out, 0x55))
    node_stub.install("GenieNode_reset", lambda h: 0)
    node_stub.install("GenieNode_free", lambda h: 0)
    node = genie_node.Node({"n": {}})
    assert node.get_sampler().value == 0x55
    node.reset()
    node.free()
    node.free()                                     # idempotent
    assert [_value(a[0]) for a in node_stub.called("GenieNode_free")] == [0x20]
    assert node.handle is None


def test_node_get_sampler_failure_is_raised(node_stub):
    node_stub.install("GenieNode_create", lambda cfg, out: _write_handle(out, 0x20))
    node_stub.install("GenieNode_getSampler", lambda h, out: -2)
    with pytest.raises(genie_node.GenieStatusError, match="GenieNode_getSampler"):
        genie_node.Node({"n": {}}).get_sampler()


@pytest.fixture
def pipeline_stub(node_stub):
    node_stub.install("GeniePipelineConfig_createFromJson",
                      lambda json_, out: _write_handle(out, 0x60))
    node_stub.install("GeniePipelineConfig_free", lambda cfg: 0)
    return node_stub


def test_a_pipeline_is_created_wired_run_and_freed(pipeline_stub):
    """The VLM path's own C calls: nodes added by handle, connections by IO
    enum (WILDCARD included), execute with a NULL user pointer."""
    handles = iter([0x21, 0x22])
    pipeline_stub.install("GenieNode_create",
                          lambda cfg, out: _write_handle(out, next(handles)))
    pipeline_stub.install("GeniePipelineConfig_bindLogger", lambda cfg, log: 0)
    pipeline_stub.install("GeniePipeline_create",
                          lambda cfg, out: _write_handle(out, 0x70))
    for fn in ("GeniePipeline_addNode", "GeniePipeline_connect",
               "GeniePipeline_execute", "GeniePipeline_reset",
               "GeniePipeline_free"):
        pipeline_stub.install(fn, lambda *a: 0)
    enc, gen = genie_node.Node({"enc": {}}), genie_node.Node({"gen": {}})

    p = genie_node.Pipeline(log_handle=capi.LogHandle(0x9))
    (json_arg, _), = pipeline_stub.called("GeniePipelineConfig_createFromJson")
    assert json_arg == b"{}"
    (cfg, log), = pipeline_stub.called("GeniePipelineConfig_bindLogger")
    assert (_value(cfg), _value(log)) == (0x60, 0x9)
    assert [_value(a[0]) for a in pipeline_stub.called("GeniePipelineConfig_free")] == [0x60]

    p.add(enc)
    p.add(gen)
    assert [(_value(a[0]), _value(a[1]))
            for a in pipeline_stub.called("GeniePipeline_addNode")] == [
        (0x70, 0x21), (0x70, 0x22)]
    p.connect(enc, "IMAGE_ENCODER_EMBEDDING_OUTPUT",
              gen, "TEXT_GENERATOR_EMBEDDING_INPUT")
    p.connect(enc, "WILDCARD", gen, "WILDCARD")
    assert [tuple(_value(x) for x in a)
            for a in pipeline_stub.called("GeniePipeline_connect")] == [
        (0x70, 0x21, 201, 0x22, 1), (0x70, 0x21, 1000, 0x22, 1000)]
    p.execute()
    (handle, user), = pipeline_stub.called("GeniePipeline_execute")
    assert (_value(handle), user) == (0x70, None)
    p.reset()
    p.free()
    p.free()                                        # idempotent
    assert len(pipeline_stub.called("GeniePipeline_free")) == 1


def test_a_failed_pipeline_create_still_frees_its_config(pipeline_stub):
    pipeline_stub.install("GeniePipeline_create", lambda cfg, out: -4)
    with pytest.raises(genie_node.GenieStatusError, match="GeniePipeline_create"):
        genie_node.Pipeline({"pipeline": {}})
    (json_arg, _), = pipeline_stub.called("GeniePipelineConfig_createFromJson")
    assert json_arg == b'{"pipeline": {}}'
    assert len(pipeline_stub.called("GeniePipelineConfig_free")) == 1


def test_a_failed_pipeline_step_names_what_failed(pipeline_stub):
    pipeline_stub.install("GeniePipeline_create",
                          lambda cfg, out: _write_handle(out, 0x70))
    pipeline_stub.install("GeniePipeline_execute", lambda h, user: -1)
    with pytest.raises(genie_node.GenieStatusError, match="execute"):
        genie_node.Pipeline().execute()
