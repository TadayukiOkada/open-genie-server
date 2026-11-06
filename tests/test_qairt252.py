"""QAIRT 2.52.0 (Genie API 1.22) additions, and the deprecated pre-2.52 paths
that stay behind them: native logprobs, the SDK prefix cache, GenieProfile_clear,
the standalone tokenizer, and the legacy-runtime warning.

FakeGenieLib reports API 1.21 (QAIRT 2.51) unless a test sets api_minor = 22.
"""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from genie_server import bootstrap, capi, engine, slots, templates
from genie_server.app import create_app
from genie_server.config import ServerConfig, load_config

from conftest import build_state

CHAT = {"messages": [{"role": "user", "content": "hi"}], "logprobs": True,
        "top_logprobs": 2}


@pytest.fixture
def state252(tmp_path):
    state = build_state(tmp_path)
    state.lib.api_minor = 22
    return state


# ---------------------------------------------------------------- logprobs

def test_native_logprobs_chat_uses_sdk_sampler(state252):
    client = TestClient(create_app(state252))
    lib = state252.lib
    response = client.post("/v1/chat/completions", json=CHAT)
    assert response.status_code == 200, response.text
    content = response.json()["choices"][0]["logprobs"]["content"]
    assert content and all(entry["logprob"] == -0.25 for entry in content)
    # The SDK's alternatives come out of a heap; the response orders them.
    tops = [[alt["logprob"] for alt in entry["top_logprobs"]] for entry in content]
    assert all(t == [-1.0, -2.0] for t in tops)
    # The SDK sampled (no custom sampler) and the callback is gone again.
    assert lib.custom_samplers == {}
    handle = state252.manager.slots[0].handle.value
    assert lib.logprobs_callbacks == {}
    final = lib.sampler_params[handle]
    assert final["type"] == "basic" and final["logprobs"] == "false"


def test_native_logprobs_completions(state252):
    client = TestClient(create_app(state252))
    response = client.post("/v1/completions", json={"prompt": "hi", "logprobs": 1})
    assert response.status_code == 200, response.text
    lp = response.json()["choices"][0]["logprobs"]
    assert lp["token_logprobs"] and set(lp["token_logprobs"]) == {-0.25}
    assert all(len(top) == 1 for top in lp["top_logprobs"])


def test_native_logprobs_keep_request_sampler_values(state252):
    client = TestClient(create_app(state252))
    lib = state252.lib
    seen = []
    original = lib._query_native_logprobs

    def spy(handle, text, params, on_token):
        seen.append(dict(params))
        return original(handle, text, params, on_token)

    lib._query_native_logprobs = spy
    body = {**CHAT, "temperature": 0.7, "top_p": 0.5, "seed": 7}
    assert client.post("/v1/chat/completions", json=body).status_code == 200
    params = seen[0]
    assert params["temp"] == "0.7" and params["top-p"] == "0.5" and params["seed"] == "7"
    assert params["logprobs"] == "true" and params["top-logprobs"] == "2"


def test_native_logprobs_need_no_numpy(state252, monkeypatch):
    from genie_server import logprobs as logprobs_mod
    monkeypatch.setattr(logprobs_mod, "LOGPROBS_AVAILABLE", False)
    client = TestClient(create_app(state252))
    assert client.post("/v1/chat/completions", json=CHAT).status_code == 200


def test_prompt_scoring_still_uses_custom_sampler_on_252(state252):
    client = TestClient(create_app(state252))
    client.post("/v1/server/prompt_logprobs", json={"enabled": True})
    response = client.post("/v1/completions", json={
        "prompt": "a b c", "echo": True, "logprobs": 1, "max_tokens": 0})
    assert response.status_code == 200, response.text
    assert state252.lib.custom_samplers  # teacher forcing needs it
    assert state252.lib.logprobs_callbacks == {}


def test_legacy_logprobs_warn_once(state, caplog):
    engine.legacy_logprobs_warned.clear()
    client = TestClient(create_app(state))
    with caplog.at_level(logging.WARNING, logger="genie_server.engine"):
        for _ in range(2):
            assert client.post("/v1/chat/completions", json=CHAT).status_code == 200
    warnings = [r for r in caplog.records if "deprecated" in r.getMessage()]
    assert len(warnings) == 1
    assert state.lib.custom_samplers  # pre-2.52: the custom sampler
    assert state.lib.logprobs_callbacks == {}


def test_native_logprobs_unsupported_runtime_detected(state252):
    """A runtime that takes the logprobs settings but never calls back."""
    from genie_server import capi as capi_mod
    client = TestClient(create_app(state252))

    def silent(handle, text, params, on_token):
        on_token("Hello", capi_mod.SENTENCE_CONTINUE)
        on_token("", capi_mod.SENTENCE_END)
        return 0

    state252.lib._query_native_logprobs = silent
    response = client.post("/v1/chat/completions", json=CHAT)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "logprobs_not_supported"


def test_collector_native_rejects_forced_tokens():
    from genie_server.logprobs import LogprobsCollector
    with pytest.raises(ValueError):
        LogprobsCollector(native=True, forced_tokens=[1, 2])


def test_native_logprobs_params():
    assert engine.native_logprobs_params(None) == {"logprobs": "false", "top-logprobs": "0"}
    assert engine.native_logprobs_params(0) == {"logprobs": "true", "top-logprobs": "0"}
    assert engine.native_logprobs_params(5) == {"logprobs": "true", "top-logprobs": "5"}


# ---------------------------------------------------------------- prefix cache

def _cfg(**kw) -> ServerConfig:
    return ServerConfig(sdk_root="/x", **kw)


def _lib(api_minor: int = 22):
    return SimpleNamespace(has_sdk_prefix_cache=api_minor >= 22)


SWA_DCFG = {"engine": {"cache-groups": [
    {"prefix": "past_"},
    {"prefix": "swa_", "longcontext": {"type": "sliding-window",
                                       "sliding-window": {"window-size": 512}}}]}}


@pytest.mark.parametrize("template,boundary", [
    ("chatml", "<|im_end|>\n"), ("qwen3_5", "<|im_end|>\n"),
    ("llama3", "<|eot_id|>"), ("gemma4", "<turn|>\n")])
def test_sdk_prefix_cache_plan_boundaries(template, boundary):
    plan = slots.sdk_prefix_cache_plan(_lib(), _cfg(), {}, template, "s")
    assert plan == {"enabled": True, "prefix_boundary": boundary, "max_entries": 8,
                    "min_prefix_tokens": 128, "global_max_total_mb": 512}


@pytest.mark.parametrize("template", ["llama2", "gemma"])
def test_sdk_prefix_cache_off_without_system_turn(template):
    assert slots.sdk_prefix_cache_plan(_lib(), _cfg(), {}, template, "s") is None


def test_sdk_prefix_cache_off_on_legacy_or_host():
    assert slots.sdk_prefix_cache_plan(_lib(21), _cfg(), {}, "chatml", "s") is None
    assert slots.sdk_prefix_cache_plan(
        _lib(), _cfg(prefix_cache_backend="host"), {}, "chatml", "s") is None


def test_sdk_prefix_cache_auto_skips_swa_and_lora_but_sdk_forces():
    lora = {"engine": {"model": {"binary": {"lora": {"adapters": []}}}}}
    for dcfg in (SWA_DCFG, lora):
        assert slots.sdk_prefix_cache_plan(_lib(), _cfg(), dcfg, "chatml", "s") is None
        forced = slots.sdk_prefix_cache_plan(
            _lib(), _cfg(prefix_cache_backend="sdk"), dcfg, "chatml", "s")
        assert forced and forced["enabled"]


def test_sdk_prefix_cache_never_with_linear_attention_or_ssd():
    lin = {"engine": {"model": {"linear-attention": {"conv": 4}}}}
    for dcfg in (lin, {"type": "ssd-q1"}):
        for backend in ("auto", "sdk"):
            assert slots.sdk_prefix_cache_plan(
                _lib(), _cfg(prefix_cache_backend=backend), dcfg, "chatml", "s") is None


def test_sdk_prefix_cache_limits_from_config():
    plan = slots.sdk_prefix_cache_plan(_lib(), _cfg(
        sdk_prefix_cache_max_entries=3, sdk_prefix_cache_global_mb=64,
        sdk_prefix_cache_partition_mb=32, sdk_prefix_cache_min_tokens=0),
        {}, "chatml", "s")
    assert plan["max_entries"] == 3 and plan["global_max_total_mb"] == 64
    assert plan["partition_max_total_mb"] == 32 and plan["min_prefix_tokens"] == 0


def _bundle(tmp_path: Path, name: str, dialog: dict) -> Path:
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "tokenizer.json").write_text("{}")
    (model_dir / "genie_config.json").write_text(json.dumps({"dialog": {
        "version": 1, "type": "basic", "context": {"size": 1024},
        "tokenizer": {"version": 1, "path": "tokenizer.json"}, **dialog}}))
    return model_dir


def test_load_model_injects_system_prompt_cache(state252, tmp_path):
    manager = state252.manager
    model_dir = _bundle(tmp_path, "qwen3-1.7b", {})
    assets = manager.load_model(model_dir, None, "default")
    sent = state252.lib.dialog_configs[-1]["dialog"]["system_prompt_cache"]
    assert sent["prefix_boundary"] == "<|im_end|>\n" and sent["enabled"] is True
    assert assets.sdk_prefix_cache == sent


def test_load_model_leaves_legacy_config_alone(state, tmp_path):
    model_dir = _bundle(tmp_path, "qwen3-1.7b", {})
    assets = state.manager.load_model(model_dir, None, "default")
    assert "system_prompt_cache" not in state.lib.dialog_configs[-1]["dialog"]
    assert assets.sdk_prefix_cache is None


def test_switch_model_flushes_old_partition(state252, tmp_path):
    manager = state252.manager
    slot = manager.slots[0]
    slot.sdk_prefix_cache = {"prefix_boundary": "<|im_end|>\n", "max_entries": 8}
    old = slot.handle.value
    with slot.lock:
        manager.switch_model(slot, _bundle(tmp_path, "qwen3-4b", {}), unload_first=True)
    assert old in state252.lib.flushed_prefix_caches
    assert slot.sdk_prefix_cache  # the new dialog is on it too


def test_lora_change_flushes_sdk_prefix_cache(state252):
    slot = state252.manager.slots[0]
    slot.sdk_prefix_cache = {"prefix_boundary": "<|im_end|>\n", "max_entries": 8}
    client = TestClient(create_app(state252))
    response = client.post("/v1/lora/apply", json={"lora_adapter_name": "a1"})
    assert response.status_code == 200, response.text
    assert state252.lib.flushed_prefix_caches == [slot.handle.value]


def test_prefix_cache_endpoints_report_and_flush_sdk(state252):
    slot = state252.manager.slots[0]
    slot.sdk_prefix_cache = {"prefix_boundary": "<|im_end|>\n", "max_entries": 8}
    client = TestClient(create_app(state252))
    listed = client.get("/v1/prefix/cache").json()
    assert listed["sdk"] == [{"slot": "default", "prefix_boundary": "<|im_end|>\n",
                              "max_entries": 8}]
    pruned = client.delete("/v1/prefix/cache?scope=all").json()
    assert pruned["sdk_flushed"] == ["default"]
    assert state252.lib.flushed_prefix_caches == [slot.handle.value]


def test_warmup_is_deprecated_on_252(state252):
    client = TestClient(create_app(state252))
    response = client.post("/v1/prefix/warmup", json={"system_prompt": "You are terse."})
    assert response.status_code == 200, response.text
    assert "deprecated" in response.json()


def test_warmup_not_deprecated_on_legacy(state):
    client = TestClient(create_app(state))
    response = client.post("/v1/prefix/warmup", json={"system_prompt": "You are terse."})
    assert response.status_code == 200, response.text
    assert "deprecated" not in response.json()


# ---------------------------------------------------------------- profile

def test_profile_cleared_before_each_query_on_252(state252):
    slot = state252.manager.slots[0]
    slot.profile = state252.lib.create_profile()
    client = TestClient(create_app(state252))
    for _ in range(2):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        assert client.post("/v1/chat/completions", json=body).status_code == 200
    assert state252.lib.cleared_profiles == [slot.profile, slot.profile]


def test_profile_not_cleared_on_legacy(state):
    slot = state.manager.slots[0]
    slot.profile = state.lib.create_profile()
    client = TestClient(create_app(state))
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert client.post("/v1/chat/completions", json=body).status_code == 200
    assert state.lib.cleared_profiles == []


# ---------------------------------------------------------------- tokenizer

def test_host_tokenizer_selection(monkeypatch, tmp_path):
    hf = object()
    monkeypatch.setattr(slots, "load_tokenizer_file", lambda p: hf)
    genie = SimpleNamespace(has_standalone_tokenizer=True,
                            create_tokenizer=lambda p: "genie-tok")
    path = str(tmp_path / "tokenizer.json")
    assert slots.load_host_tokenizer(path, genie, "auto") is hf
    assert slots.load_host_tokenizer(path, genie, "hf") is hf
    assert slots.load_host_tokenizer(path, genie, "genie") == "genie-tok"
    monkeypatch.setattr(slots, "load_tokenizer_file", lambda p: None)
    assert slots.load_host_tokenizer(path, genie, "auto") == "genie-tok"
    legacy = SimpleNamespace(has_standalone_tokenizer=False)
    assert slots.load_host_tokenizer(path, legacy, "auto") is None
    assert slots.load_host_tokenizer(path, legacy, "genie") is None


def _tokenizer_json(tmp_path, decoder) -> str:
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps({"decoder": decoder}))
    return str(path)


def test_genie_tokenizer_byte_level_subwords(tmp_path):
    path = _tokenizer_json(tmp_path, {"type": "ByteLevel"})
    tok = capi.GenieTokenizer(None, None, path)
    words = {1: "Hello", 2: "Ġworld", 3: "æĹ", 4: "¥"}  # "Ġworld", "日" split
    tok.subwords = lambda ids: [words[i] for i in ids]
    assert tok.decode([1, 2]) == "Hello world"
    assert tok.token_bytes(3) == b"\xe6\x97"
    assert tok.decode([3, 4]) == "日"
    assert tok.decode([3]) == "�"
    assert tok.decode([]) == ""


def test_genie_tokenizer_sentencepiece_subwords(tmp_path):
    path = _tokenizer_json(tmp_path, {"type": "Sequence", "decoders": [
        {"type": "Replace"}, {"type": "ByteFallback"}, {"type": "Fuse"}]})
    tok = capi.GenieTokenizer(None, None, path)
    words = {1: "▁Hello", 2: "<0xE6>", 3: "<0x97>", 4: "<0xA5>"}
    tok.subwords = lambda ids: [words[i] for i in ids]
    assert tok.decode([1]) == " Hello"
    assert tok.token_bytes(2) == b"\xe6"
    assert tok.decode([2, 3, 4]) == "日"


def test_token_bytes_used_for_chat_logprobs():
    from genie_server.logprobs import chat_logprobs
    tok = SimpleNamespace(decode=lambda ids, skip_special_tokens=False: "�",
                          token_bytes=lambda i: b"\xe6\x97")
    out = chat_logprobs(tok, [(3, -0.5, [(3, -0.5)])])
    entry = out["content"][0]
    assert entry["bytes"] == [0xE6, 0x97]
    assert entry["top_logprobs"][0]["bytes"] == [0xE6, 0x97]


# ---------------------------------------------------------------- startup / config

def test_legacy_runtime_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="genie_server.bootstrap"):
        assert bootstrap.warn_if_legacy_runtime(SimpleNamespace(
            legacy_runtime=True, api_version=lambda: "1.21.0"))
        assert not bootstrap.warn_if_legacy_runtime(SimpleNamespace(legacy_runtime=False))
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 1 and "Deprecated QAIRT runtime" in messages[0]


@pytest.mark.parametrize("version,legacy", [
    (None, True), ("1.21.0", True), ("1.22.0", False), ("1.23.1", False), ("2.0.0", False)])
def test_genielib_legacy_runtime(version, legacy):
    lib = capi.GenieLib.__new__(capi.GenieLib)
    lib.api_version = lambda: version
    assert lib.legacy_runtime is legacy


def test_config_new_keys(tmp_path):
    path = tmp_path / "env_config.json"
    base = {"TEXT_SLOTS": [{"model_root": str(tmp_path)}]}
    path.write_text(json.dumps(base))
    cfg = load_config(str(path))
    assert cfg.prefix_cache_backend == "auto" and cfg.tokenizer_backend == "auto"
    assert cfg.sdk_prefix_cache_global_mb == 512
    path.write_text(json.dumps({**base, "PREFIX_CACHE_BACKEND": "SDK",
                                "TOKENIZER_BACKEND": "genie",
                                "SDK_PREFIX_CACHE_GLOBAL_MB": 64}))
    cfg = load_config(str(path))
    assert (cfg.prefix_cache_backend, cfg.tokenizer_backend,
            cfg.sdk_prefix_cache_global_mb) == ("sdk", "genie", 64)
    for bad in ({"PREFIX_CACHE_BACKEND": "disk"}, {"SDK_PREFIX_CACHE_GLOBAL_MB": 0},
                {"SDK_PREFIX_CACHE_MAX_ENTRIES": True}):
        path.write_text(json.dumps({**base, **bad}))
        with pytest.raises(ValueError):
            load_config(str(path))


def test_sdk_prefix_boundaries_match_the_split():
    """The SDK's boundary is where split_prompt_for_prefix_cache puts the split."""
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    for template, boundary in templates.SDK_PREFIX_BOUNDARIES.items():
        prefix, rest, ok = templates.split_prompt_for_prefix_cache(messages, template, bos=False)
        assert ok and prefix.endswith(boundary), template
        assert boundary not in prefix[:-len(boundary)], template


# ---------------------------------------------------------------- sampler updates

class _RecordingCdll:
    """Just enough of libGenie for GenieLib.apply_sampler_params_to_handle."""

    def __init__(self, minor: int):
        self.minor = minor
        self.set_param_calls = []

    def Genie_getApiMajorVersion(self):
        return 1

    def Genie_getApiMinorVersion(self):
        return self.minor

    def Genie_getApiPatchVersion(self):
        return 0

    def GenieSamplerConfig_createFromJson(self, payload, out):
        return 0

    def GenieSamplerConfig_setParam(self, cfg, key, value):
        self.set_param_calls.append((key, value))
        return 0

    def GenieSampler_applyConfig(self, sampler, cfg):
        return 0

    def GenieSamplerConfig_free(self, cfg):
        return None


def _lib_over(cdll):
    lib = capi.GenieLib.__new__(capi.GenieLib)
    lib._lib = cdll
    lib._has_api_version = True
    return lib


def test_sampler_update_is_one_json_on_252():
    """QAIRT 2.52 validates after every single-key setParam, so a switch to the
    custom sampler key by key is refused both ways round (measured on x86)."""
    cdll = _RecordingCdll(22)
    _lib_over(cdll).apply_sampler_params_to_handle(
        None, {"type": "custom", "callback-name": "cb"})
    assert len(cdll.set_param_calls) == 1
    key, value = cdll.set_param_calls[0]
    assert key == b""
    assert json.loads(value) == {"sampler": {"version": 1, "type": "custom",
                                             "callback-name": "cb"}}


def test_sampler_update_is_key_by_key_on_legacy():
    """2.51's JSON form needs every numeric field present (measured: rc -13)."""
    cdll = _RecordingCdll(21)
    _lib_over(cdll).apply_sampler_params_to_handle(None, {"type": "basic", "temp": "0.5"})
    assert cdll.set_param_calls == [(b"type", b"basic"), (b"temp", b"0.5")]


def test_sampler_json_values_types():
    values = capi.sampler_json_values({
        **capi.make_sampler_params({}, 0.7, 0.9, 40, 7), **engine.native_logprobs_params(3)})
    assert values == {"type": "basic", "temp": 0.7, "top-k": 40, "top-p": 0.9, "seed": 7,
                      "logprobs": True, "top-logprobs": 3}
    assert capi.sampler_json_values(engine.native_logprobs_params(None)) == {
        "logprobs": False, "top-logprobs": 0}
