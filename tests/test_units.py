"""Unit tests for the pure-Python modules (templates, tools, capi helpers)."""

import json
import types
from pathlib import Path

import pytest

from genie_server import templates, tools
from genie_server.capi import make_sampler_params


# ---------------------------------------------------------------- templates

def test_detect_template():
    assert templates.detect_template("Llama3.2-3B") == "llama3"
    assert templates.detect_template("llama-3-8b") == "llama3"
    assert templates.detect_template("mistral-7b") == "llama2"
    assert templates.detect_template("gemma-2-9b-it") == "gemma"
    assert templates.detect_template("gemma_3_4b_it") == "gemma"
    # gemma4 is its own family: same turn structure, different turn tokens.
    assert templates.detect_template("gemma4-e2b-it") == "gemma4"
    assert templates.detect_template("gemma-4-9b-it") == "gemma4"
    assert templates.detect_template("qwen3_4b") == "chatml"


def test_content_to_text_flattens_parts():
    assert templates.content_to_text("plain") == "plain"
    assert templates.content_to_text(None) == ""
    assert templates.content_to_text(
        [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"


def test_chatml_render():
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"}]
    out = templates.render_chat_prompt(msgs, "chatml")
    assert out == ("<|im_start|>system\nsys<|im_end|>\n"
                   "<|im_start|>user\nhi<|im_end|>\n"
                   "<|im_start|>assistant\n")


def test_llama3_render():
    msgs = [{"role": "user", "content": "hi"}]
    out = templates.render_chat_prompt(msgs, "llama3")
    assert out.startswith("<|begin_of_text|>")
    assert out.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


def test_split_prefix_concat_equals_full_render():
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"}]
    for template in ("chatml", "llama3"):
        prefix, remaining, cacheable = \
            templates.split_prompt_for_prefix_cache(msgs, template)
        assert cacheable
        assert prefix + remaining == templates.render_chat_prompt(msgs, template)


def test_split_prefix_llama2_not_cacheable():
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"}]
    for template in ("llama2", "gemma"):
        _, _, cacheable = templates.split_prompt_for_prefix_cache(msgs, template)
        assert not cacheable


def test_gemma_render():
    msgs = [{"role": "system", "content": "Be brief."},
            {"role": "user", "content": "hi"}]
    out = templates.render_chat_prompt(msgs, "gemma")
    assert out == ("<bos><start_of_turn>user\nBe brief.\n\nhi<end_of_turn>\n"
                   "<start_of_turn>model\n")


def test_gemma4_render_uses_its_own_turn_tokens_and_a_system_turn():
    """gemma4 marks turns with <|turn> / <turn|> (ids 105/106) — the Gemma 2/3
    spelling is absent from its vocabulary — and keeps system as its OWN turn
    rather than folding it into the first user turn the way Gemma 2/3 does."""
    msgs = [{"role": "system", "content": "Be brief."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
            {"role": "user", "content": "bye"}]
    out = templates.render_chat_prompt(msgs, "gemma4")
    assert out == ("<bos><|turn>system\nBe brief.<turn|>\n"
                   "<|turn>user\nhi<turn|>\n"
                   "<|turn>model\nyo<turn|>\n"
                   "<|turn>user\nbye<turn|>\n"
                   "<|turn>model\n")
    # the Gemma 2/3 markers must not appear anywhere
    assert "start_of_turn" not in out and "end_of_turn" not in out


def test_gemma4_differs_from_gemma_on_the_system_turn():
    """Spelling is not the only difference: Gemma 2/3 folds system into the
    first user turn, gemma4 gives it a turn of its own."""
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": "u"}]
    assert "<|turn>system\ns<turn|>" in templates.render_chat_prompt(msgs, "gemma4")
    assert "<start_of_turn>user\ns\n\nu" in templates.render_chat_prompt(msgs, "gemma")


def test_gemma4_prefix_cache_splits_on_the_system_turn():
    """Because system is its own turn, gemma4 can split a cacheable prefix —
    unlike Gemma 2/3. The two halves must rejoin to the full prompt exactly."""
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"}]
    prefix, remaining, cacheable = \
        templates.split_prompt_for_prefix_cache(msgs, "gemma4")
    assert cacheable
    assert prefix == "<bos><|turn>system\nsys<turn|>\n"
    assert prefix + remaining == templates.render_chat_prompt(msgs, "gemma4")


def test_no_think_directive():
    msgs = [{"role": "user", "content": "hi"}]
    out = templates.prepare_messages(msgs, enable_thinking=False)
    assert out[0]["role"] == "system"
    assert "/no_think" in out[0]["content"]
    # original list untouched
    assert msgs[0]["role"] == "user"

    msgs2 = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    out2 = templates.prepare_messages(msgs2, enable_thinking=False)
    assert out2[0]["content"] == "sys\n\n/no_think"
    assert msgs2[0]["content"] == "sys"


def test_no_think_is_separated_from_the_tools_block():
    """The tools block ends with the "</tool_call>" of its format example.  A
    glued "/no_think" becomes part of that example, and the model copies it into
    its output -- qwen3_4b_instruct_2507 emitted a trailing "/no_think" line
    after every tool call on real hardware."""
    tool = {"type": "function", "function": {"name": "f", "parameters": {}}}
    out = templates.prepare_messages(
        [{"role": "user", "content": "hi"}], enable_thinking=False, tools=[tool])
    system = out[0]["content"]
    assert "</tool_call>/no_think" not in system
    assert system.endswith("</tool_call>\n\n/no_think")


def test_no_think_alone_has_no_leading_blank_line():
    """With no system message and no tools the directive IS the system turn, so
    it must not start with the separator."""
    out = templates.prepare_messages(
        [{"role": "user", "content": "hi"}], enable_thinking=False)
    assert out[0]["content"] == "/no_think"


def test_tools_injected_into_system():
    tool = {"type": "function", "function": {"name": "f", "parameters": {}}}
    out = templates.prepare_messages(
        [{"role": "user", "content": "hi"}], tools=[tool])
    assert out[0]["role"] == "system"
    assert "<tools>" in out[0]["content"]
    assert '"name": "f"' in out[0]["content"]


def test_tool_history_rendering():
    msgs = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "get_weather",
                         "arguments": '{"city": "Tokyo"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]
    out = templates.render_chat_prompt(templates.prepare_messages(msgs), "chatml")
    assert '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n</tool_call>' in out
    assert "<tool_response>\nsunny\n</tool_response>" in out


# ---------------------------------------------------------------- tools parsing

def test_parse_tool_calls():
    text = ('before <tool_call>\n{"name": "f", "arguments": {"x": 1}}\n'
            '</tool_call> after')
    content, calls = tools.parse_tool_calls(text)
    assert content == "before  after".replace("  ", " ") or "before" in content
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "f"
    assert json.loads(calls[0]["function"]["arguments"]) == {"x": 1}


def test_parse_tool_calls_bad_json_left_in_content():
    text = "<tool_call>not json</tool_call>"
    content, calls = tools.parse_tool_calls(text)
    assert calls == []
    assert "not json" in content


def test_parse_tool_calls_leaves_an_unterminated_block_as_text():
    """Observed on qwen3_0_6b w4a16: the model emits EOS straight after the
    JSON without the closing tag. It is recoverable — generation has stopped
    and the JSON is complete — but a model that will not close its own call
    has a defect, so repairing it is behind TOOL_CALL_RECOVERY rather than
    on by default. See the recovery test below."""
    text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Osaka"}}'
    content, calls = tools.parse_tool_calls(text)
    assert calls == []
    assert content == text.strip()


def test_parse_tool_calls_recovers_unterminated_block():
    """With recovery on, the same reply becomes a call."""
    text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Osaka"}}'
    content, calls = tools.parse_tool_calls(text, {"get_weather"})
    assert len(calls) == 1, content
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Osaka"}
    assert "<tool_call>" not in content
    assert content == ""


def test_parse_tool_calls_unterminated_keeps_leading_prose():
    text = 'Let me check. <tool_call>{"name": "f", "arguments": {}}'
    content, calls = tools.parse_tool_calls(text, {"f"})
    assert len(calls) == 1
    assert content == "Let me check."


def test_parse_tool_calls_unterminated_incomplete_json_stays_text():
    """A generation cut off mid-JSON by max_tokens must not be guessed at."""
    text = '<tool_call>\n{"name": "f", "arguments": {"city": "Osa'
    content, calls = tools.parse_tool_calls(text, {"f"})
    assert calls == []
    assert "<tool_call>" in content


def test_parse_tool_calls_unterminated_without_name_stays_text():
    text = '<tool_call>{"arguments": {"x": 1}}'
    content, calls = tools.parse_tool_calls(text)
    assert calls == []
    assert "<tool_call>" in content


def test_parse_tool_calls_terminated_then_unterminated():
    """A well-formed block followed by an unterminated one yields both."""
    text = ('<tool_call>{"name": "a", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "b", "arguments": {"k": 2}}')
    content, calls = tools.parse_tool_calls(text, {"a", "b"})
    assert [c["function"]["name"] for c in calls] == ["a", "b"]
    assert content == ""


def test_parse_tool_calls_plain_json_is_not_a_tool_call():
    """Recovery keys off the opening tag; bare JSON stays prose."""
    text = '{"name": "f", "arguments": {}}'
    content, calls = tools.parse_tool_calls(text)
    assert calls == []
    assert content == text


def test_stream_filter_recovers_unterminated_block():
    f = tools.ToolCallStreamFilter({"f"})
    chunks = ["Sure. ", "<tool", '_call>\n{"name": "f", ', '"arguments": {}}']
    out = "".join(f.feed(c) for c in chunks)
    leftover, calls = f.finalize()
    assert "<tool_call>" not in out + leftover
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "f"


def test_stream_filter_passthrough():
    f = tools.ToolCallStreamFilter()
    out = "".join(f.feed(t) for t in ["Hello ", "world", "!"])
    leftover, calls = f.finalize()
    assert out + leftover == "Hello world!"
    assert calls == []


def test_stream_filter_holds_back_tool_call():
    f = tools.ToolCallStreamFilter()
    chunks = ["Sure. ", "<tool", '_call>\n{"name": "f", ',
              '"arguments": {}}\n</tool_call>', " done"]
    out = "".join(f.feed(c) for c in chunks)
    assert "<tool_call>" not in out
    assert out.startswith("Sure. ")
    leftover, calls = f.finalize()
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "f"
    assert "<tool_call>" not in leftover


def test_stream_filter_false_alarm_lt():
    """A lone '<' that never becomes '<tool_call>' must still be emitted."""
    f = tools.ToolCallStreamFilter()
    out = f.feed("a < b")
    out += f.feed(" and c")
    leftover, calls = f.finalize()
    assert out + leftover == "a < b and c"
    assert calls == []


# ---------------------------------------------------------------- sampler params

def test_sampler_greedy_mapping():
    params = make_sampler_params({}, temperature=0.0)
    assert params == {"type": "basic", "temp": "1.0", "top-k": "1"}


def test_sampler_defaults_reset():
    defaults = {"temp": 0.8, "top-k": 40, "top-p": 0.95}
    # Request omits everything -> model defaults are re-applied (no leak
    # from a previous request's settings). "type": "basic" always included
    # so a preceding logprobs request's custom sampler can't leak either.
    params = make_sampler_params(defaults)
    assert params == {"type": "basic", "temp": "0.8", "top-k": "40", "top-p": "0.95"}
    # Request overrides only temperature.
    params = make_sampler_params(defaults, temperature=0.2)
    assert params["temp"] == "0.2"
    assert params["top-k"] == "40"


def test_sampler_seed():
    assert make_sampler_params({}, seed=42)["seed"] == "42"


# ---------------------------------------------------------------- logprobs

def _fake_logits(values):
    import ctypes
    arr = (ctypes.c_float * len(values))(*values)
    return ctypes.addressof(arr), len(values), arr  # keep arr alive


def test_collector_greedy_records_logsoftmax():
    import math
    from genie_server.logprobs import LogprobsCollector

    c = LogprobsCollector(top_n=2, temperature=0.0)
    addr, n, _keep = _fake_logits([0.0, 3.0, 1.0, 0.0])
    ids = c.on_logits(addr, n, 1)
    assert ids == [1]  # argmax
    token, lp, top = c.results[0]
    assert token == 1
    z = math.log(sum(math.exp(v) for v in [0.0, 3.0, 1.0, 0.0]))
    assert abs(lp - (3.0 - z)) < 1e-5
    assert [t for t, _ in top] == [1, 2]  # top-2, descending


def test_collector_force_mode():
    from genie_server.logprobs import LogprobsCollector

    c = LogprobsCollector(top_n=1, forced_tokens=[3, 0])
    addr, n, _keep = _fake_logits([0.0, 9.0, 0.0, 0.0])
    assert c.on_logits(addr, n, 1) == [3]   # forced, not argmax
    addr, n, _keep = _fake_logits([0.0, 9.0, 0.0, 0.0])
    assert c.on_logits(addr, n, 1) == [0]
    (tok0, lp0, top0), (tok1, lp1, top1) = c.results
    assert tok0 == 3 and lp0 < top0[0][1]   # forced token below the argmax
    assert top0[0][0] == 1


# ---------------------------------------------------------------- config

def test_text_slots_empty_list_is_a_vlm_only_deployment(tmp_path):
    """No text slots plus VLM slots is a valid deployment, and the only shape
    that fits a target which cannot hold a VLM and a text model at once."""
    from genie_server.config import load_config

    path = tmp_path / "env_config.json"
    path.write_text(json.dumps({
        "QAIRT_SDK_ROOT": "/opt/qairt",
        "TEXT_SLOTS": [],
        "VLM_SLOTS": [{"name": "vlm0", "device_id": 0,
                       "model_root": str(tmp_path), "spec": "qwen3_vl"}],
    }))
    cfg = load_config(str(path))
    assert cfg.text_slots == ()
    assert [s.name for s in cfg.vlm_slots] == ["vlm0"]


def test_a_config_with_no_models_is_rejected(tmp_path):
    """A server with no models starts and then fails every request, so the
    config is refused at load instead."""
    from genie_server.config import load_config

    path = tmp_path / "env_config.json"
    path.write_text(json.dumps({"QAIRT_SDK_ROOT": "/opt/qairt"}))
    with pytest.raises(ValueError, match="no models configured"):
        load_config(str(path))


def test_a_single_text_slot_needs_only_model_root(tmp_path):
    """The minimal single-model config: name and device_id both default."""
    from genie_server.config import load_config

    path = tmp_path / "env_config.json"
    path.write_text(json.dumps({"QAIRT_SDK_ROOT": "/opt/qairt",
                                "TEXT_SLOTS": [{"model_root": str(tmp_path)}]}))
    cfg = load_config(str(path))
    assert [(s.name, s.device_id) for s in cfg.text_slots] == [("slot0", None)]


def test_slot_config_file_defaults_to_genie_config_json(tmp_path):
    """The SDK's own examples use that name, so a config that says nothing
    must keep pointing at it."""
    from genie_server.config import load_config

    path = tmp_path / "env_config.json"
    path.write_text(json.dumps({"QAIRT_SDK_ROOT": "/opt/qairt",
                                "TEXT_SLOTS": [{"model_root": str(tmp_path)}]}))
    assert load_config(str(path)).text_slots[0].config_file == "genie_config.json"


def test_a_slot_can_name_its_dialog_config(tmp_path):
    """An export names the config after the model as often as not
    ("acme-7b-htp.json"), because genie-app takes the path on its
    command line. Point the slot at the file instead of copying it."""
    from genie_server.config import load_config

    path = tmp_path / "env_config.json"
    path.write_text(json.dumps({
        "QAIRT_SDK_ROOT": "/opt/qairt",
        "TEXT_SLOTS": [{"model_root": str(tmp_path),
                        "config_file": "some-model-htp.json"}]}))
    assert load_config(str(path)).text_slots[0].config_file == "some-model-htp.json"


def test_load_dialog_config_reads_the_named_file(tmp_path):
    """The name has to reach the loader, not just the config object."""
    from genie_server.slots import load_dialog_config

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "some-model-htp.json").write_text(json.dumps(
        {"dialog": {"context": {"size": 2048}}}))

    dcfg = load_dialog_config(model_dir, None, "chat", tmp_path / "htpcache",
                              config_file="some-model-htp.json")[1]
    assert dcfg["context"]["size"] == 2048

    # ...and the default still points at genie_config.json.
    with pytest.raises(FileNotFoundError):
        load_dialog_config(model_dir, None, "chat", tmp_path / "htpcache")


def test_a_slot_keeps_its_config_file_across_a_model_switch(tmp_path):
    """Like poll, the name belongs to the slot rather than to the model it
    happens to be holding, so a hot-swap must not drop it."""
    from genie_server.slots import Slot

    slot = Slot(name="chat", device_id=0, model_root=tmp_path,
                config_file="some-model-htp.json")
    assert slot.config_file == "some-model-htp.json"
    assert Slot(name="chat", device_id=0,
                model_root=tmp_path).config_file == "genie_config.json"


def test_tool_call_recovery_defaults_to_off(tmp_path):
    """Opt-in, because it hides a defect in the bundle being measured.

    A config that does not mention it must not get the recovery: the whole
    point of the default is that a plain install reports what the model
    emitted. Setting it explicitly still works.
    """
    from genie_server.config import load_config

    base = {"QAIRT_SDK_ROOT": "/opt/qairt",
            "TEXT_SLOTS": [{"model_root": str(tmp_path)}]}
    path = tmp_path / "env_config.json"

    path.write_text(json.dumps(base))
    assert load_config(str(path)).tool_call_recovery is False

    path.write_text(json.dumps({**base, "TOOL_CALL_RECOVERY": True}))
    assert load_config(str(path)).tool_call_recovery is True


def test_slot_load_order_defaults_to_vlm_first(tmp_path):
    from genie_server.config import load_config

    path = tmp_path / "env_config.json"
    path.write_text(json.dumps({"QAIRT_SDK_ROOT": "/opt/qairt",
                                "TEXT_SLOTS": [{"model_root": str(tmp_path)}]}))
    assert load_config(str(path)).slot_load_order == "vlm-first"


def test_slot_load_order_accepts_text_first(tmp_path):
    from genie_server.config import load_config

    path = tmp_path / "env_config.json"
    path.write_text(json.dumps({"QAIRT_SDK_ROOT": "/opt/qairt",
                                "TEXT_SLOTS": [{"model_root": str(tmp_path)}],
                                "SLOT_LOAD_ORDER": "Text-First"}))
    assert load_config(str(path)).slot_load_order == "text-first"


def test_slot_load_order_rejects_unknown_value(tmp_path):
    from genie_server.config import load_config

    path = tmp_path / "env_config.json"
    path.write_text(json.dumps({"QAIRT_SDK_ROOT": "/opt/qairt",
                                "TEXT_SLOTS": [{"model_root": str(tmp_path)}],
                                "SLOT_LOAD_ORDER": "whatever"}))
    with pytest.raises(ValueError, match="SLOT_LOAD_ORDER"):
        load_config(str(path))


# --------------------------------------------------- device_id validation

def _cfg_with_device_id(tmp_path, device_id, key="TEXT_SLOTS"):
    from genie_server.config import load_config

    slot = {"name": "chat", "model_root": str(tmp_path), "device_id": device_id}
    if key == "VLM_SLOTS":
        slot["spec"] = "qwen3_vl"
    path = tmp_path / "env_config.json"
    path.write_text(json.dumps({"QAIRT_SDK_ROOT": "/opt/qairt", key: [slot]}))
    return load_config(str(path))


def test_device_id_is_kept_when_it_is_a_plausible_core_index(tmp_path):
    assert _cfg_with_device_id(tmp_path, 1).text_slots[0].device_id == 1
    assert _cfg_with_device_id(tmp_path, 0).text_slots[0].device_id == 0


def test_a_slot_without_device_id_stays_unpinned(tmp_path):
    """None is not an error: the bundle's own devices[0].device_id applies."""
    assert _cfg_with_device_id(tmp_path, None).text_slots[0].device_id is None


def test_negative_and_out_of_range_device_ids_are_rejected(tmp_path):
    """slots.pin_htp_device writes device_id into the HTP backend extension
    config unchecked, so a typo has to be caught here or it reaches QNN."""
    for bad in (-1, 99):
        with pytest.raises(ValueError, match="device_id must be between"):
            _cfg_with_device_id(tmp_path, bad)


def test_non_integer_device_ids_are_rejected(tmp_path):
    for bad in ("0", 1.0, [0]):
        with pytest.raises(ValueError, match="device_id must be an integer"):
            _cfg_with_device_id(tmp_path, bad)


def test_device_id_true_is_a_mistake_not_device_one(tmp_path):
    """bool is an int subclass, so this needs its own guard."""
    with pytest.raises(ValueError, match="device_id must be an integer"):
        _cfg_with_device_id(tmp_path, True)


def test_vlm_slots_validate_device_id_the_same_way(tmp_path):
    assert _cfg_with_device_id(tmp_path, 1, "VLM_SLOTS").vlm_slots[0].device_id == 1
    with pytest.raises(ValueError, match="device_id must be between"):
        _cfg_with_device_id(tmp_path, 99, "VLM_SLOTS")


def test_the_device_id_error_names_the_slot(tmp_path):
    """A multi-slot config has to say which entry is wrong."""
    with pytest.raises(ValueError, match="slot 'chat'"):
        _cfg_with_device_id(tmp_path, 99)


# ------------------------------------------------------ slot creation order

class _OrderRecorder:
    """Stands in for SlotManager + vlm.create_vlm_slots to record which of the
    two ran first and whether the validator-flag reset landed between them."""

    def __init__(self, lib):
        self.lib = lib
        self.events: list[str] = []

    def load_all(self):
        self.events.append("text")

    def create_vlm_slots(self, config, cdll):
        self.events.append("vlm")
        return ()


def _run_build_order(monkeypatch, order):
    # build_state imports GenieLib/vlm/SlotManager lazily inside the function,
    # so they must be patched on their defining modules, not on bootstrap.
    from genie_server import bootstrap, capi, slots, vlm
    from tests.fake_genie import FakeGenieLib

    lib = FakeGenieLib()
    rec = _OrderRecorder(lib)

    class Cfg:
        slot_load_order = order
        prefix_cache_dir = "."

        def apply_process_env(self):
            pass

        def resolved_genie_lib_path(self):
            return "libGenie.so"

    cfg = Cfg()
    monkeypatch.setattr(bootstrap, "load_config", lambda path: cfg)
    monkeypatch.setattr(capi.GenieLib, "load", staticmethod(lambda p: lib))
    # build_state hands lib.cdll to create_vlm_slots; the fake refuses to
    # produce one because it cannot emulate the VLM node API.
    monkeypatch.setattr(FakeGenieLib, "cdll",
                        property(lambda self: "fake-cdll"))
    monkeypatch.setattr(slots, "SlotManager", lambda c, l: rec)
    monkeypatch.setattr(vlm, "create_vlm_slots", rec.create_vlm_slots)
    monkeypatch.setattr(bootstrap, "PrefixCache", lambda d: None)
    monkeypatch.setattr(bootstrap, "ServerState",
                        lambda **kw: types.SimpleNamespace(**kw))
    bootstrap.build_state("env_config.json")
    return rec.events, lib.validator_flag_resets


def test_vlm_first_order_creates_vlm_slots_before_text(monkeypatch):
    events, resets = _run_build_order(monkeypatch, "vlm-first")
    assert events == ["vlm", "text"]
    # No text dialog has run yet, so nothing has set the validator flags.
    assert resets == 0


def test_text_first_order_resets_validator_flags_between(monkeypatch):
    """Without the reset, libGenie rejects every VLM text-generator node
    config with "Specify one config from pos-id-dim and positional-encoding"
    once a text dialog with pos-id-dim has been created in this process."""
    events, resets = _run_build_order(monkeypatch, "text-first")
    assert events == ["text", "vlm"]
    assert resets == 1


# ---------------------------------------------------------------- VLM finish_reason

def _vlm_gen_stub(completion_tokens=0):
    """Minimal stand-in for engine.Generation for the VLM worker's bookkeeping."""
    class G:
        request_id = "chatcmpl-test"
        finish_reason = "stop"
        error = None
    g = G()
    g.completion_tokens = completion_tokens
    return g


def test_genie_status_error_carries_the_status():
    """The VLM worker needs the raw Genie_Status_t to tell a warning
    (context exceeded) apart from a real failure."""
    from genie_server.genie_node import GenieStatusError, _check

    try:
        _check(4, "execute")
    except GenieStatusError as e:
        assert e.status == 4
        assert isinstance(e, RuntimeError)   # existing handlers still catch it
        assert "status=4" in str(e)
    else:
        raise AssertionError("_check did not raise on a non-success status")

    _check(0, "execute")   # success must not raise


def test_vlm_context_exceeded_is_length_not_an_error():
    """A generation that ran until the context filled up produced valid
    output; report finish_reason=length rather than failing the request."""
    from genie_server import capi
    from genie_server.genie_node import GenieStatusError

    gen = _vlm_gen_stub(completion_tokens=3800)
    try:
        raise GenieStatusError(capi.WARNING_CONTEXT_EXCEEDED, "execute")
    except GenieStatusError as e:
        if e.status == capi.WARNING_CONTEXT_EXCEEDED:
            gen.finish_reason = "length"
        else:
            gen.error = str(e)
    assert gen.finish_reason == "length"
    assert gen.error is None


def test_vlm_hitting_the_static_cap_is_length():
    """The SDK returns SUCCESS both for EOS and for the node's
    max-num-tokens cap, so the token count is the only discriminator."""
    slot_cap = 1024
    for produced, expected in [(1024, "length"), (1030, "length"), (7, "stop")]:
        gen = _vlm_gen_stub(completion_tokens=produced)
        if slot_cap and gen.completion_tokens >= slot_cap:
            gen.finish_reason = "length"
        assert gen.finish_reason == expected, produced


def test_vlm_uncapped_slot_never_reports_length_from_the_cap():
    """max_tokens=0 disables the cap; only a context-exceeded warning can
    make such a slot report length."""
    slot_cap = 0
    gen = _vlm_gen_stub(completion_tokens=99999)
    if slot_cap and gen.completion_tokens >= slot_cap:
        gen.finish_reason = "length"
    assert gen.finish_reason == "stop"


# ------------------------------------------------- VLM slot token counting

def _write_tokenizer_json(path):
    """A real tokenizer.json whose count differs from a whitespace split, so a
    test can tell which of the two produced a number."""
    tokenizers = pytest.importorskip("tokenizers")
    tok = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.BertPreTokenizer()
    tok.save(str(path))
    return path


def _node_cfg(tokenizer_path=None, top_key="text-generator"):
    """The shape _load_vlm_node_config returns: {node type: {...}} with every
    path already resolved."""
    cfg = {"version": 1}
    if tokenizer_path is not None:
        cfg["tokenizer"] = {"version": 1, "path": str(tokenizer_path)}
    return {top_key: cfg}


def test_vlm_pipeline_tokenizer_comes_from_the_text_generator_node(tmp_path):
    """GenieNode exposes no tokenizer, but the node config names the file —
    loading it is what puts VLM usage on the same basis as a text slot's."""
    from genie_server.vlm import _load_pipeline_tokenizer

    tok_file = _write_tokenizer_json(tmp_path / "tokenizer.json")
    tok = _load_pipeline_tokenizer({
        "image_encoder": _node_cfg(top_key="image-encoder"),
        "text_encoder": _node_cfg(tok_file, "text-encoder"),
        "text_generator": _node_cfg(tok_file, "text-generator"),
    })
    assert tok is not None
    assert len(tok.encode("a,b c").ids) == 4      # whitespace would say 2


def test_vlm_pipeline_tokenizer_falls_back_to_the_text_encoder(tmp_path):
    from genie_server.vlm import _load_pipeline_tokenizer

    tok_file = _write_tokenizer_json(tmp_path / "tokenizer.json")
    tok = _load_pipeline_tokenizer({
        "text_encoder": _node_cfg(tok_file, "text-encoder"),
        "text_generator": _node_cfg(None, "text-generator"),
    })
    assert tok is not None


def test_vlm_pipeline_tokenizer_is_none_when_no_node_names_one():
    from genie_server.vlm import _load_pipeline_tokenizer

    assert _load_pipeline_tokenizer(
        {"text_generator": _node_cfg(None)}) is None


def test_vlm_pipeline_tokenizer_survives_an_unreadable_file(tmp_path):
    """A bundle that names a tokenizer.json it does not ship must degrade to
    the whitespace count, not fail slot creation."""
    from genie_server.vlm import _load_pipeline_tokenizer

    assert _load_pipeline_tokenizer(
        {"text_generator": _node_cfg(tmp_path / "missing.json")}) is None


def test_vlm_count_tokens_uses_the_tokenizer_then_falls_back(tmp_path):
    from genie_server.vlm import VLMSlot, _load_pipeline_tokenizer

    slot = VLMSlot.__new__(VLMSlot)          # no pipeline needed for counting
    tok_file = _write_tokenizer_json(tmp_path / "tokenizer.json")
    slot.tokenizer = _load_pipeline_tokenizer({"text_generator": _node_cfg(tok_file)})
    assert slot.count_tokens("a,b c") == 4
    slot.tokenizer = None
    assert slot.count_tokens("a,b c") == 2   # whitespace fallback


# ------------------------------------------------------------- QnnHtp.poll

def _model_dir_with_poll(tmp_path, poll_value):
    """A minimal model directory whose bundle sets QnnHtp.poll."""
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "htp.json").write_text(json.dumps({"devices": [{}]}))
    config = {"dialog": {"engine": {"backend": {
        "type": "QnnHtp",
        "QnnHtp": {"poll": poll_value, "cpu-mask": "0xe0"},
        "extensions": "htp.json"}}}}
    (model_dir / "genie_config.json").write_text(json.dumps(config))
    return model_dir


def _load(model_dir, tmp_path, poll):
    from genie_server.slots import load_dialog_config
    return load_dialog_config(model_dir, None, "cdsp0", tmp_path / "htpcache",
                              poll)[1]["engine"]["backend"]["QnnHtp"]


def test_poll_override_turns_the_bundles_busy_wait_off(tmp_path):
    """poll:true costs ~260% CPU on SA8255P for no measurable latency gain,
    and it lives in the model bundle — the override is how a deployment turns
    it off without editing someone else's model directory."""
    model_dir = _model_dir_with_poll(tmp_path, True)

    assert _load(model_dir, tmp_path, False)["poll"] is False


def test_poll_override_can_also_turn_it_on(tmp_path):
    model_dir = _model_dir_with_poll(tmp_path, False)

    assert _load(model_dir, tmp_path, True)["poll"] is True


def test_poll_none_leaves_the_bundle_alone(tmp_path):
    model_dir = _model_dir_with_poll(tmp_path, True)

    htp = _load(model_dir, tmp_path, None)

    assert htp["poll"] is True
    assert htp["cpu-mask"] == "0xe0"   # nothing else in the block is touched


def test_poll_override_on_a_bundle_without_the_key(tmp_path):
    model_dir = tmp_path / "bare"
    model_dir.mkdir()
    (model_dir / "genie_config.json").write_text(
        json.dumps({"dialog": {"engine": {"backend": {"type": "QnnHtp"}}}}))

    assert _load(model_dir, tmp_path, False)["poll"] is False


# --------------------------------------------------------- POLL in config

def _slots(raw, base=None):
    from genie_server.config import _parse_text_slots
    return _parse_text_slots(raw, base)


def test_text_slots_poll_defaults_to_none():
    slots = _slots({"TEXT_SLOTS": [{"name": "cdsp0", "model_root": "/models/a"}]})

    assert slots[0].poll is None


def test_text_slots_poll_per_slot():
    slots = _slots({"TEXT_SLOTS": [
        {"name": "cdsp0", "model_root": "/models/a", "poll": False},
        {"name": "cdsp1", "model_root": "/models/b", "poll": True}]})

    assert [s.poll for s in slots] == [False, True]


def test_top_level_poll_is_the_default_and_a_slot_can_override_it():
    slots = _slots({"POLL": False, "TEXT_SLOTS": [
        {"name": "cdsp0", "model_root": "/models/a"},
        {"name": "cdsp1", "model_root": "/models/b", "poll": True}]})

    assert [s.poll for s in slots] == [False, True]


def test_top_level_poll_applies_to_a_slot_that_does_not_set_it():
    slots = _slots({"POLL": False,
                    "TEXT_SLOTS": [{"model_root": "/models/only"}]})

    assert slots[0].poll is False


def test_poll_must_be_a_boolean():
    with pytest.raises(ValueError, match="poll must be true or false"):
        _slots({"TEXT_SLOTS": [{"name": "x", "model_root": "/m", "poll": "yes"}]})



# --------------------------------------- MODELS_BASE_DIR resolves model paths

def _vlm_slots(raw, base=None):
    from genie_server.config import _parse_vlm_slots
    return _parse_vlm_slots(raw, base)


def test_relative_text_model_root_resolves_against_models_base_dir():
    slots = _slots({"TEXT_SLOTS": [{"model_root": "qwen3-4b"}]},
                   Path("/models"))

    assert slots[0].model_root == Path("/models/qwen3-4b")


def test_relative_vlm_model_root_resolves_against_models_base_dir():
    slots = _vlm_slots({"VLM_SLOTS": [{"model_root": "qwen3-vl"}]},
                       Path("/models"))

    assert slots[0].model_root == Path("/models/qwen3-vl")


def test_an_absolute_model_root_ignores_models_base_dir():
    """An absolute path stays where it points — that is what keeps a one-off
    model outside the tree loadable."""
    slots = _slots({"TEXT_SLOTS": [{"model_root": "/elsewhere/qwen3-4b"}]},
                   Path("/models"))

    assert slots[0].model_root == Path("/elsewhere/qwen3-4b")


def test_without_models_base_dir_a_relative_model_root_is_cwd_relative():
    slots = _slots({"TEXT_SLOTS": [{"model_root": "qwen3-4b"}]}, None)

    assert slots[0].model_root == (Path.cwd() / "qwen3-4b").resolve()


def test_a_relative_model_root_is_normalised_under_the_base():
    slots = _slots({"TEXT_SLOTS": [{"model_root": "./sub/../qwen3-4b"}]},
                   Path("/models"))

    assert slots[0].model_root == Path("/models/qwen3-4b")


def test_load_config_applies_models_base_dir_to_both_slot_kinds(tmp_path):
    from genie_server.config import load_config

    (tmp_path / "env.json").write_text(json.dumps({
        "QAIRT_SDK_ROOT": "/sdk",
        "MODELS_BASE_DIR": "/models",
        "TEXT_SLOTS": [{"name": "cdsp0", "model_root": "text-model"}],
        "VLM_SLOTS": [{"name": "vlm0", "model_root": "vlm-model"}],
    }))

    cfg = load_config(str(tmp_path / "env.json"))

    assert cfg.models_base_dir == Path("/models")
    assert cfg.text_slots[0].model_root == Path("/models/text-model")
    assert cfg.vlm_slots[0].model_root == Path("/models/vlm-model")


def test_switch_model_dir_uses_the_same_rule_as_model_root():
    """POST /v1/models/switch and startup must agree, or a bare directory
    name would mean two different places."""
    from genie_server.config import resolve_model_path

    assert resolve_model_path("qwen3-4b", Path("/models")) \
        == Path("/models/qwen3-4b")
    assert resolve_model_path("/elsewhere/qwen3-4b", Path("/models")) \
        == Path("/elsewhere/qwen3-4b")


# --------------------- CHAT_TEMPLATE / DEFAULT_MAX_TOKENS / INFERENCE_TIMEOUT

def _minimal_bundle(tmp_path, dir_name):
    """The smallest model directory load_dialog_config accepts."""
    model_dir = tmp_path / dir_name
    model_dir.mkdir()
    (model_dir / "htp.json").write_text(json.dumps({"devices": [{}]}))
    (model_dir / "genie_config.json").write_text(json.dumps(
        {"dialog": {"engine": {"backend": {"type": "QnnHtp",
                                           "extensions": "htp.json"}}}}))
    return model_dir


def _assets_for(tmp_path, model_dir, chat_template=""):
    from fake_genie import FakeGenieLib
    from genie_server.config import ServerConfig, SlotSpec
    from genie_server.slots import SlotManager

    cfg = ServerConfig(
        sdk_root="/nonexistent",
        prefix_cache_dir=str(tmp_path / "prefix_cache"),
        chat_template_override=chat_template,
        text_slots=(SlotSpec(name="chat", device_id=None,
                             model_root=model_dir),))
    return SlotManager(cfg, FakeGenieLib()).load_model(model_dir, None, "chat")


def test_without_chat_template_the_family_comes_from_the_directory_name(tmp_path):
    assert _assets_for(tmp_path, _minimal_bundle(tmp_path, "Llama3.2-3B")).template         == "llama3"


def test_chat_template_overrides_the_directory_name(tmp_path):
    """The escape hatch for a bundle whose directory name says nothing about
    its prompt format."""
    model_dir = _minimal_bundle(tmp_path, "internal-build-42")
    assert _assets_for(tmp_path, model_dir).template == "chatml"      # the default
    assert _assets_for(tmp_path, model_dir, chat_template="gemma4").template         == "gemma4"


def test_chat_template_is_detected_from_the_override_not_taken_literally(tmp_path):
    """The override feeds the same detector, so a full model name works too."""
    model_dir = _minimal_bundle(tmp_path, "internal-build-42")
    assert _assets_for(tmp_path, model_dir,
                       chat_template="Llama3.2-3B").template == "llama3"


def _slot_stub(context_size, tokens_per_text=None):
    class S:
        pass
    s = S()
    s.context_size = context_size
    s.count_tokens = tokens_per_text or (lambda text: len(text.split()))
    return s


def test_default_max_tokens_is_the_remaining_context(tmp_path):
    """No client max_tokens: bound generation by what is left of the window."""
    from genie_server.engine import default_max_tokens

    assert default_max_tokens(_slot_stub(4096), "a b c", None, 0) == 4093


def test_default_max_tokens_cap_lowers_the_remaining_context(tmp_path):
    """DEFAULT_MAX_TOKENS is an additional ceiling, never a raise."""
    from genie_server.engine import default_max_tokens

    assert default_max_tokens(_slot_stub(4096), "a b c", None, 256) == 256
    assert default_max_tokens(_slot_stub(4096), "a b c", None, 99999) == 4093


def test_an_explicit_max_tokens_ignores_the_cap():
    """DEFAULT_MAX_TOKENS only fills in a value the client did not send —
    otherwise it would silently truncate an explicit request."""
    from genie_server.engine import default_max_tokens

    assert default_max_tokens(_slot_stub(4096), "a b c", 2000, 256) == 2000
    assert default_max_tokens(_slot_stub(4096), "a b c", 0, 256) == 0


def test_default_max_tokens_without_a_known_context_falls_back_to_the_cap():
    """A bundle with no context size in its config leaves the cap as the only
    bound; with no cap either, generation stays unbounded (None)."""
    from genie_server.engine import default_max_tokens

    assert default_max_tokens(_slot_stub(0), "a b c", None, 256) == 256
    assert default_max_tokens(_slot_stub(0), "a b c", None, 0) is None


def test_a_prompt_filling_the_context_still_leaves_one_token():
    """max(..., 1) — a zero would mean "one token" to the SDK (F5) and a
    negative would be nonsense."""
    from genie_server.engine import default_max_tokens

    assert default_max_tokens(_slot_stub(4), "a b c d e f", None, 0) == 1


def test_inference_timeout_defaults_and_parses(tmp_path):
    from genie_server.config import load_config

    def _load(raw):
        path = tmp_path / "env_config.json"
        path.write_text(json.dumps(
            {"QAIRT_SDK_ROOT": "/opt/qairt",
             "TEXT_SLOTS": [{"model_root": str(tmp_path)}], **raw}))
        return load_config(str(path))

    assert _load({}).inference_timeout_s == 120.0
    assert _load({"INFERENCE_TIMEOUT": 600}).inference_timeout_s == 600.0
    assert _load({"INFERENCE_TIMEOUT": 0.5}).inference_timeout_s == 0.5


def test_prefix_warmup_timeout_tracks_inference_timeout_with_a_floor():
    """The derived timeouts are why INFERENCE_TIMEOUT is worth raising on a
    slow target: a warmup must not be cut short by the default."""
    from genie_server.config import ServerConfig

    def _cfg(t):
        return ServerConfig(sdk_root="/s", inference_timeout_s=t)

    assert _cfg(600).prefix_warmup_timeout_s == 300.0
    assert _cfg(10).prefix_warmup_timeout_s == 60.0      # floor


# ------------------------------- measurement scripts pick slots from status

def _order_slot_names(status, want=1):
    """Imports the helper by path: tests/integration is not on sys.path for
    the offline run (it needs a live server), but this part is pure."""
    import importlib.util

    path = Path(__file__).resolve().parent / "integration" / "slot_names.py"
    spec = importlib.util.spec_from_file_location("slot_names", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.order_slot_names(status, want)


def _status(*slots):
    return {"slots": [{"name": n, "device_id": d, "loaded": loaded}
                      for n, d, loaded in slots]}


def test_measurement_slots_are_ordered_by_device_id():
    """Whatever the slots are called, the first one measured must be the same
    on every run — otherwise two runs are not comparable."""
    assert _order_slot_names(
        _status(("chat", 1, True), ("tool_call", 0, True)), want=2) \
        == ["tool_call", "chat"]


def test_measurement_slots_ignore_a_slot_that_failed_to_load():
    """A slot left empty by a failed model switch would answer 503."""
    with pytest.raises(SystemExit, match="needs 2"):
        _order_slot_names(
            _status(("chat", 0, True), ("tool_call", 1, False)), want=2)


def test_measurement_slots_put_unpinned_slots_last():
    """device_id: null sorts after every pinned slot instead of raising."""
    assert _order_slot_names(
        _status(("free", None, True), ("chat", 0, True)), want=2) \
        == ["chat", "free"]


def test_measurement_slot_error_names_what_it_found():
    with pytest.raises(SystemExit, match="0 loaded text slot"):
        _order_slot_names({"slots": []}, want=1)


# ------------------------------------------- unmarked tool-call recovery (F25)

_KNOWN = {"get_weather", "get_current_time"}
_MANGLED = ("ФРАГМЕНТ\n"
            '{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n'
            "ФРАГМЕНТ")


def test_recovers_a_call_whose_marker_was_mangled():
    """qwen3_4b_instruct_2507 w4a16 emits Cyrillic in place of <tool_call> on
    half its calls; the body is correct, so the name identifies it."""
    content, calls = tools.parse_tool_calls(_MANGLED, _KNOWN)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Tokyo"}
    assert content == ""          # the marker debris goes with the call


def test_recovers_several_mangled_calls_in_one_reply():
    text = (_MANGLED + "\n"
            '{"name": "get_weather", "arguments": {"city": "Osaka"}}\n'
            "ФРАГМЕНТ")
    content, calls = tools.parse_tool_calls(text, _KNOWN)
    assert [json.loads(c["function"]["arguments"])["city"] for c in calls] \
        == ["Tokyo", "Osaka"]
    assert content == ""


def test_recovers_a_call_with_no_marker_at_all():
    """qwen3_0_6b drops the tags entirely and leaves bare JSON after </think>."""
    text = '<think>\n</think>\n\n{"name": "get_current_time", "arguments": {}}'
    content, calls = tools.parse_tool_calls(text, _KNOWN)
    assert len(calls) == 1 and calls[0]["function"]["name"] == "get_current_time"
    assert content == "<think>\n</think>"


def test_recovery_needs_a_name_the_caller_declared():
    """The declared-name match is what makes bare JSON unambiguous. Without it
    a model answering in JSON would be misread as calling a function."""
    text = '{"name": "not_a_tool", "arguments": {}}'
    content, calls = tools.parse_tool_calls(text, _KNOWN)
    assert calls == [] and content == text


def test_recovery_leaves_a_json_answer_alone():
    text = 'Here is the record you asked for:\n{"name": "Alice", "age": 30}'
    content, calls = tools.parse_tool_calls(text, _KNOWN)
    assert calls == [] and content == text


def test_recovery_keeps_prose_next_to_the_call():
    text = ('I will check that for you.\n'
            '{"name": "get_weather", "arguments": {"city": "Tokyo"}}')
    content, calls = tools.parse_tool_calls(text, _KNOWN)
    assert len(calls) == 1
    assert content == "I will check that for you."


def test_recovery_handles_braces_inside_arguments():
    text = ('Ф\n{"name": "get_weather", "arguments": '
            '{"city": "a}b", "opts": {"units": "c"}}}')
    content, calls = tools.parse_tool_calls(text, _KNOWN)
    assert len(calls) == 1
    assert json.loads(calls[0]["function"]["arguments"])["city"] == "a}b"


def test_recovery_off_is_the_old_strict_parse():
    """Without known names nothing bare is reinterpreted -- the behaviour
    TOOL_CALL_RECOVERY=false restores."""
    for text in (_MANGLED, '{"name": "get_current_time", "arguments": {}}'):
        content, calls = tools.parse_tool_calls(text)
        assert calls == [] and content == text.strip()


def _stream(text: str, known, chunk: int = 3):
    f = tools.ToolCallStreamFilter(known)
    emitted = "".join(f.feed(text[i:i + chunk])
                      for i in range(0, len(text), chunk))
    leftover, calls = f.finalize()
    return emitted, leftover, calls


def test_stream_filter_recovers_a_mangled_call():
    """The mangled marker is not a tag, so without screening the whole call
    would already have gone out as content deltas before finalize saw it."""
    emitted, leftover, calls = _stream(_MANGLED, _KNOWN)
    assert len(calls) == 1 and calls[0]["function"]["name"] == "get_weather"
    assert emitted.strip() == "" and leftover == ""


def test_stream_filter_keeps_prose_in_order():
    """Screening must not reorder a line it has already started emitting."""
    text = "Sure, let me check.\nIt is sunny in Tokyo."
    emitted, leftover, calls = _stream(text, _KNOWN)
    assert calls == []
    assert emitted + leftover == text


def test_stream_filter_streams_prose_before_a_later_call():
    text = "I will check that.\n" + _MANGLED
    emitted, leftover, calls = _stream(text, _KNOWN)
    assert len(calls) == 1
    assert emitted.strip() == "I will check that."


def test_stream_filter_off_matches_the_old_behaviour():
    emitted, leftover, calls = _stream(_MANGLED, None)
    assert calls == []
    assert emitted + leftover == _MANGLED


# --------------------------------------------------- LUT embedding lut-path

def _model_dir_with_luts(tmp_path, *, write_files=True,
                         embedding="embedding_int16_lut.bin",
                         perlayer="embed_token_int16_lut.bin"):
    """A Gemma-shaped bundle: LUT embeddings named relative to the bundle."""
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    config = {"dialog": {
        "embedding": {"version": 1, "type": "lut", "lut-path": embedding},
        "perlayer-embedding": {"version": 1, "type": "lut", "lut-path": perlayer},
    }}
    (model_dir / "genie_config.json").write_text(json.dumps(config))
    if write_files:
        (model_dir / embedding).write_bytes(b"\0")
        (model_dir / perlayer).write_bytes(b"\0")
    return model_dir


def _load_dialog(model_dir, tmp_path):
    from genie_server.slots import load_dialog_config
    return load_dialog_config(model_dir, None, "cdsp0", tmp_path / "htpcache")[1]


def test_lut_paths_are_resolved_against_the_model_dir(tmp_path):
    """The SDK opens lut-path with a plain ifstream (LUT.cpp), so a relative
    one resolves against the server's working directory, not the bundle —
    "Embedding File not present." with the file sitting right there. Observed
    loading gemma_4_e4b_it, which is the first bundle to use LUT embeddings."""
    model_dir = _model_dir_with_luts(tmp_path)

    dcfg = _load_dialog(model_dir, tmp_path)

    for key, name in (("embedding", "embedding_int16_lut.bin"),
                      ("perlayer-embedding", "embed_token_int16_lut.bin")):
        resolved = dcfg[key]["lut-path"]
        assert resolved == str(model_dir / name)
        assert Path(resolved).is_absolute()


def test_a_missing_lut_file_is_reported_at_load(tmp_path):
    """Same contract as the other assets: fail here, with the path, rather
    than inside GenieDialog_create with "Embedding File not present."."""
    model_dir = _model_dir_with_luts(tmp_path, write_files=False)
    with pytest.raises(FileNotFoundError) as exc:
        _load_dialog(model_dir, tmp_path)
    assert "embedding_int16_lut.bin" in str(exc.value)


def test_bundles_without_lut_embeddings_are_untouched(tmp_path):
    """The Qwen3 exports carry no embedding block at all."""
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "genie_config.json").write_text(
        json.dumps({"dialog": {"context": {"size": 4096}}}))

    dcfg = _load_dialog(model_dir, tmp_path)

    assert "embedding" not in dcfg and "perlayer-embedding" not in dcfg


# ------------------------------------------------- LoRA adapter bin-sections

def _model_dir_with_lora(tmp_path, *, write_files=True):
    """A bundle shipping LoRA adapters: one bin-section per ctx-bin, per
    adapter, named relative to the bundle."""
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    sections = {"adapter_a": ["a_1_of_2_first.bin", "a_2_of_2_first.bin"],
                "adapter_b": ["a_1_of_2_second.bin", "a_2_of_2_second.bin"]}
    config = {"dialog": {"engine": {"model": {"binary": {
        "lora": {
            "version": 1,
            "alpha-tensor-name": "lora_alpha",
            "adapters": [{"version": 1, "name": name,
                          "alphas": ["alpha0", "alpha1"],
                          "bin-sections": bins}
                         for name, bins in sections.items()],
        }}}}}}
    (model_dir / "genie_config.json").write_text(json.dumps(config))
    if write_files:
        for bins in sections.values():
            for b in bins:
                (model_dir / b).write_bytes(b"\0")
    return model_dir, sections


def test_lora_bin_sections_are_resolved_against_the_model_dir(tmp_path):
    """A relative bin-section resolves against the server's working directory,
    and GenieDialog_create then rejects the whole config with "Error in
    parsing params - LoRA: Can't access adapter file", naming a path that is
    sitting in the bundle. Found on the board, loading the first bundle we
    have that ships adapters."""
    model_dir, sections = _model_dir_with_lora(tmp_path)

    dcfg = _load_dialog(model_dir, tmp_path)

    adapters = dcfg["engine"]["model"]["binary"]["lora"]["adapters"]
    assert {a["name"] for a in adapters} == set(sections)
    for adapter in adapters:
        assert adapter["bin-sections"] == [
            str(model_dir / b) for b in sections[adapter["name"]]]
        assert all(Path(b).is_absolute() for b in adapter["bin-sections"])


def test_a_missing_lora_section_is_reported_at_load(tmp_path):
    """Same contract as every other asset: fail here, with the path."""
    model_dir, _sections = _model_dir_with_lora(tmp_path, write_files=False)
    with pytest.raises(FileNotFoundError) as exc:
        _load_dialog(model_dir, tmp_path)
    assert "a_1_of_2_first.bin" in str(exc.value)


def test_bundles_without_lora_are_untouched(tmp_path):
    """Every Qwen3 and Gemma bundle we have: no lora block at all."""
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "genie_config.json").write_text(json.dumps(
        {"dialog": {"engine": {"model": {"binary": {"ctx-bins": []}}}}}))

    dcfg = _load_dialog(model_dir, tmp_path)

    assert "lora" not in dcfg["engine"]["model"]["binary"]


# ------------------------------------------- speculative decoding prefix KV$

def test_ssd_forecast_prefix_is_resolved_against_the_model_dir(tmp_path):
    """A relative forecast-prefix-name resolves against the working directory,
    and the SDK reports that as "SSD : Loaded 0 KV$ from forecast-prefix but
    expected 16 KV$" — which reads like a corrupt cache, not a path problem.
    The name is a directory holding kv-cache.primary.qnn-htp."""
    model_dir = tmp_path / "model"
    (model_dir / "forecast-prefix").mkdir(parents=True)
    (model_dir / "forecast-prefix" / "kv-cache.primary.qnn-htp").write_bytes(b"\0")
    (model_dir / "genie_config.json").write_text(json.dumps({"dialog": {
        "type": "ssd-q1",
        "ssd-q1": {"version": 1, "forecast-prefix": 16,
                   "forecast-prefix-name": "forecast-prefix"},
    }}))

    dcfg = _load_dialog(model_dir, tmp_path)

    resolved = dcfg["ssd-q1"]["forecast-prefix-name"]
    assert resolved == str(model_dir / "forecast-prefix")
    assert Path(resolved).is_absolute()


def test_a_missing_forecast_prefix_is_reported_at_load(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "genie_config.json").write_text(json.dumps({"dialog": {
        "type": "ssd-q1",
        "ssd-q1": {"version": 1, "forecast-prefix": 16,
                   "forecast-prefix-name": "forecast-prefix"},
    }}))
    with pytest.raises(FileNotFoundError) as exc:
        _load_dialog(model_dir, tmp_path)
    assert "forecast-prefix" in str(exc.value)


# ---------------------------------------------------------------- platform

def _cfg(**kw):
    from genie_server.config import ServerConfig
    return ServerConfig(sdk_root="/opt/qairt/X", text_slots=[], vlm_slots=[], **kw)


def test_linux_oe_paths_are_unchanged():
    """The OE target is what every hardware measurement was taken on, so its
    two paths are pinned here to catch an accidental change."""
    c = _cfg(target_platform="linux-oe")
    assert c.resolved_genie_lib_path() == \
        "/opt/qairt/X/lib/aarch64-oe-linux-gcc11.2/libGenie.so"
    assert c._adsp_library_path() == (
        "/opt/qairt/X/lib/hexagon-v73/unsigned;/usr/lib/rfsa/adsp;"
        "/dsp/image/dsp/cdsp0")


def test_linux_oe_adsp_lists_every_device_id_in_use():
    from genie_server.config import ServerConfig, SlotSpec
    c = ServerConfig(
        sdk_root="/s", target_platform="linux-oe", vlm_slots=[],
        text_slots=[SlotSpec(name="a", device_id=1, model_root="/m"),
                    SlotSpec(name="b", device_id=0, model_root="/m")])
    assert c._adsp_library_path().endswith(
        "/dsp/image/dsp/cdsp0;/dsp/image/dsp/cdsp1")


def test_android_uses_the_bionic_abi_and_vendor_first_adsp_path():
    """Verbatim the layout that brought a dialog up on the Android guest:
    vendor skels first, SDK skels second, and no cdspN entries (the guest
    reaches the DSP through virtio fastrpc and has no /dsp mount)."""
    c = _cfg(target_platform="android")
    assert c.resolved_genie_lib_path() == \
        "/opt/qairt/X/lib/aarch64-android/libGenie.so"
    assert c._adsp_library_path() == \
        "/vendor/lib/rfsa/adsp;/opt/qairt/X/lib/hexagon-v73/unsigned;"


def test_android_adsp_path_ignores_device_ids():
    from genie_server.config import ServerConfig, SlotSpec
    c = ServerConfig(
        sdk_root="/s", target_platform="android", vlm_slots=[],
        text_slots=[SlotSpec(name="a", device_id=1, model_root="/m")])
    assert "/dsp/image" not in c._adsp_library_path()


def test_android_sets_ld_library_path(monkeypatch):
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    _cfg(target_platform="android").apply_process_env()
    import os
    assert os.environ["LD_LIBRARY_PATH"] == \
        "/opt/qairt/X/lib/aarch64-android:/vendor/lib64"


def test_android_ld_library_path_keeps_what_was_already_there(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/preexisting")
    _cfg(target_platform="android").apply_process_env()
    import os
    assert os.environ["LD_LIBRARY_PATH"].endswith(":/preexisting")


def test_linux_oe_does_not_touch_ld_library_path(monkeypatch):
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    _cfg(target_platform="linux-oe").apply_process_env()
    import os
    assert "LD_LIBRARY_PATH" not in os.environ


def test_explicit_genie_lib_path_wins_on_every_platform():
    for plat in ("linux-oe", "android"):
        c = _cfg(target_platform=plat, genie_lib_path="/custom/libGenie.so")
        assert c.resolved_genie_lib_path() == "/custom/libGenie.so"


def test_auto_resolves_to_a_known_platform():
    from genie_server.config import KNOWN_PLATFORMS
    assert _cfg(target_platform="auto").platform in KNOWN_PLATFORMS


def test_unknown_target_platform_is_rejected(tmp_path):
    from genie_server.config import load_config
    p = tmp_path / "env_config.json"
    p.write_text(json.dumps({
        "QAIRT_SDK_ROOT": "/s", "TARGET_PLATFORM": "windows",
        "TEXT_SLOTS": [{"model_root": "/m"}]}))
    with pytest.raises(ValueError, match="TARGET_PLATFORM"):
        load_config(str(p))
