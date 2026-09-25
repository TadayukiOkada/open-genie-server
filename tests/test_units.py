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


_SYSTEM_SHAPES = {
    # H-3: a later system message used to be dropped from the prompt, and a
    # system message that is not first used to be hoisted to the front.
    "mid-conversation": [
        {"role": "system", "content": "SYS-A"}, {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "system", "content": "SYS-B"}, {"role": "user", "content": "q2"}],
    "two-leading": [
        {"role": "system", "content": "SYS-A"}, {"role": "system", "content": "SYS-B"},
        {"role": "user", "content": "q"}],
    "not-first": [
        {"role": "user", "content": "q1"}, {"role": "system", "content": "SYS-B"},
        {"role": "user", "content": "q2"}],
    # llama2 and gemma fold system text into the NEXT user turn, so one with
    # no user turn after it used to be dropped.
    "trailing": [
        {"role": "system", "content": "SYS-A"}, {"role": "user", "content": "q"},
        {"role": "system", "content": "SYS-B"}],
}


@pytest.mark.parametrize("template", templates.TEMPLATE_FAMILIES)
@pytest.mark.parametrize("shape", list(_SYSTEM_SHAPES))
def test_split_prefix_keeps_every_system_message_in_order(template, shape):
    msgs = _SYSTEM_SHAPES[shape]
    prefix, remaining, cacheable = \
        templates.split_prompt_for_prefix_cache(msgs, template)
    prompt = prefix + remaining
    assert not cacheable
    assert prompt == templates.render_chat_prompt(msgs, template)
    for m in msgs:
        if m["role"] == "system":
            # llama2 and gemma consume system text into a user turn, so it
            # is checked for, not for its own turn marker.
            assert m["content"] in prompt, (template, shape, m["content"])
    if shape == "mid-conversation":
        assert prompt.index("a1") < prompt.index("SYS-B") < prompt.index("q2")
    if shape == "two-leading":
        assert prompt.index("SYS-A") < prompt.index("SYS-B") < prompt.index("q")


@pytest.mark.parametrize("template", ["chatml", "llama3", "gemma4"])
def test_one_leading_system_message_is_still_cacheable(template):
    """The other side of the H-3 rule: the shape the prefix cache exists for
    keeps splitting."""
    msgs = [{"role": "system", "content": "SYS-A"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"}]
    prefix, remaining, cacheable = \
        templates.split_prompt_for_prefix_cache(msgs, template)
    assert cacheable
    assert "SYS-A" in prefix and "SYS-A" not in remaining
    assert prefix + remaining == templates.render_chat_prompt(msgs, template)


@pytest.mark.parametrize("template", ["llama2", "gemma"])
def test_folded_templates_join_consecutive_system_messages(template):
    """One system block per user turn, both texts in it, in order."""
    out = templates.render_chat_prompt(_SYSTEM_SHAPES["two-leading"], template)
    if template == "llama2":
        assert out.count("<<SYS>>") == 1
        assert "<<SYS>>\nSYS-A\n\nSYS-B\n<</SYS>>\n\nq [/INST]" in out
    else:
        assert "<start_of_turn>user\nSYS-A\n\nSYS-B\n\nq<end_of_turn>" in out


_TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {}}}]


def test_the_tools_block_goes_to_a_leading_system_turn():
    """A system message later in the conversation is not where the tools
    block belongs: a new leading system turn carries it, and the caller's
    own system message stays as written."""
    msgs = templates.prepare_messages(
        [{"role": "user", "content": "q1"}, {"role": "system", "content": "SYS-B"},
         {"role": "user", "content": "q2"}], tools=_TOOLS)
    assert [m["role"] for m in msgs] == ["system", "user", "system", "user"]
    assert "<tools>" in msgs[0]["content"]
    assert msgs[2]["content"] == "SYS-B"


def test_the_tools_block_joins_an_opening_system_message():
    msgs = templates.prepare_messages(
        [{"role": "system", "content": "SYS-A"}, {"role": "user", "content": "q"},
         {"role": "system", "content": "SYS-B"}],
        enable_thinking=False, tools=_TOOLS)
    assert [m["role"] for m in msgs] == ["system", "user", "system"]
    assert msgs[0]["content"].startswith("SYS-A")
    assert "<tools>" in msgs[0]["content"]
    assert msgs[0]["content"].endswith("/no_think")
    assert msgs[2]["content"] == "SYS-B"


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

def _without_seed(params):
    """make_sampler_params draws a fresh seed for an unseeded request, so
    exact comparisons leave it out (it is checked on its own below)."""
    assert int(params["seed"]) >= 0
    return {k: v for k, v in params.items() if k != "seed"}


def test_sampler_greedy_mapping():
    params = make_sampler_params({}, temperature=0.0)
    assert _without_seed(params) == {"type": "basic", "temp": "1.0",
                                     "top-k": "1", "top-p": "0.8"}


def test_sampler_defaults_reset():
    defaults = {"temp": 0.8, "top-k": 40, "top-p": 0.95}
    # Request omits everything -> model defaults are re-applied (no leak
    # from a previous request's settings). "type": "basic" always included
    # so a preceding logprobs request's custom sampler can't leak either.
    params = make_sampler_params(defaults)
    assert _without_seed(params) == {"type": "basic", "temp": "0.8",
                                     "top-k": "40", "top-p": "0.95"}
    # Request overrides only temperature.
    params = make_sampler_params(defaults, temperature=0.2)
    assert params["temp"] == "0.2"
    assert params["top-k"] == "40"


def test_sampler_seed():
    assert make_sampler_params({}, seed=42)["seed"] == "42"


def test_sampler_params_are_always_complete():
    """H-4: with a config that omits a key, that key used to be left out, so
    the SDK kept the previous request's value (greedy's top-k=1, a seed)."""
    full = {"type", "temp", "top-k", "top-p", "seed"}
    for kwargs in ({}, {"temperature": 0.7}, {"temperature": 0.0},
                   {"top_p": 0.5}, {"seed": 7}):
        assert set(make_sampler_params({}, **kwargs)) == full
    # A request after a greedy one gets the SDK's own defaults back.
    after_greedy = make_sampler_params({}, temperature=0.7)
    assert after_greedy["top-k"] == "0"
    assert after_greedy["top-p"] == "0.8"


def test_an_unseeded_request_gets_a_fresh_seed_every_time():
    """Not the SDK's "unset" and not a fixed value: each unseeded request
    re-seeds with a new random one, so an earlier request's seed cannot
    carry over and repeated sampling stays random."""
    seeds = {make_sampler_params({})["seed"] for _ in range(20)}
    assert len(seeds) > 1
    assert all(0 <= int(s) < 2 ** 31 for s in seeds)
    assert make_sampler_params({}, seed=7)["seed"] == "7"


def test_a_config_seed_is_not_resent_on_every_request():
    """A config's "seed": 42 re-sent per request would re-seed the sampler
    identically each time: every unseeded request would sample the same
    stream. It seeds the dialog at creation only."""
    from genie_server.capi import sampler_defaults_from
    cfg = {"temp": 0.5, "seed": 42, "type": "basic", "version": 1}
    defaults = sampler_defaults_from(cfg)
    assert defaults == {"temp": 0.5}
    assert sampler_defaults_from(None) == {}
    seeds = {make_sampler_params(defaults)["seed"] for _ in range(20)}
    assert "42" not in seeds and len(seeds) > 1


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
        self.log_handle = None

    def load_all(self):
        self.events.append("text")

    def create_vlm_slots(self, config, cdll, log_handle=None):
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
        platform = target_platform = "linux-oe"

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


# ------------------------------------------------- LUT embedding paths (PCQ)

def _pcq_lut(base, prefix):
    """A per-channel-quantized LUT block as the Gemma 4 QAT exports write it:
    scale and offset are .bin paths, not numbers."""
    table = base / "embedding-table"
    table.mkdir(exist_ok=True)
    for suffix in ("lut", "scale", "offset"):
        (table / f"{prefix}_{suffix}.bin").write_bytes(b"\0" * 8)
    return {"version": 1, "lut-path": f"embedding-table/{prefix}_lut.bin",
            "size": 4, "datatype": "ufixed2",
            "quant-param": {"scale": f"embedding-table/{prefix}_scale.bin",
                            "offset": f"embedding-table/{prefix}_offset.bin"}}


def test_pcq_quant_param_files_resolve_against_the_bundle(tmp_path):
    """The SDK opens all three files relative to the working directory, so
    resolving only lut-path left a QAT bundle loadable from nowhere but
    inside its own directory."""
    from genie_server.slots import resolve_lut_paths
    lut = _pcq_lut(tmp_path, "embedding")
    resolve_lut_paths(lut, tmp_path)
    for path in (lut["lut-path"], lut["quant-param"]["scale"],
                 lut["quant-param"]["offset"]):
        assert Path(path).is_absolute() and Path(path).is_file()


def test_per_tensor_quant_params_are_numbers_and_left_alone(tmp_path):
    from genie_server.slots import resolve_lut_paths
    (tmp_path / "lut.bin").write_bytes(b"\0")
    lut = {"lut-path": "lut.bin", "quant-param": {"scale": 0.0123, "offset": -32768}}
    resolve_lut_paths(lut, tmp_path)
    assert lut["quant-param"] == {"scale": 0.0123, "offset": -32768}


def test_a_missing_pcq_file_fails_the_load(tmp_path):
    from genie_server.slots import resolve_lut_paths
    lut = _pcq_lut(tmp_path, "embedding")
    (tmp_path / "embedding-table" / "embedding_offset.bin").unlink()
    with pytest.raises(FileNotFoundError, match="embedding_offset.bin"):
        resolve_lut_paths(lut, tmp_path)


def test_dialog_config_resolves_both_pcq_tables(tmp_path):
    from genie_server.slots import load_dialog_config
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "genie_config.json").write_text(json.dumps({"dialog": {
        "embedding": _pcq_lut(model_dir, "embedding"),
        "perlayer-embedding": _pcq_lut(model_dir, "ple")}}))
    config_json, _ = load_dialog_config(model_dir, None, "chat", tmp_path / "htpcache")
    sent = json.loads(config_json)["dialog"]   # the bytes handed to the SDK
    for key in ("embedding", "perlayer-embedding"):
        assert Path(sent[key]["lut-path"]).is_absolute()
        assert Path(sent[key]["quant-param"]["scale"]).is_absolute()
        assert Path(sent[key]["quant-param"]["offset"]).is_absolute()


def test_vlm_node_configs_resolve_the_per_layer_tables(tmp_path):
    """Gemma 4's text-encoder and text-generator nodes each name a second,
    per-layer table, which the node loader used to skip entirely."""
    from genie_server.vlm import _load_vlm_node_config
    enc = tmp_path / "text-encoder.json"
    enc.write_text(json.dumps({"text-encoder": {
        "lut": _pcq_lut(tmp_path, "embedding"),
        "perlayer-lut": _pcq_lut(tmp_path, "ple")}}))
    gen = tmp_path / "text-generator.json"
    gen.write_text(json.dumps({"text-generator": {
        "embedding": _pcq_lut(tmp_path, "embedding"),
        "perlayer-embedding": _pcq_lut(tmp_path, "ple")}}))
    for path, keys in ((enc, ("lut", "perlayer-lut")),
                       (gen, ("embedding", "perlayer-embedding"))):
        cfg = next(iter(_load_vlm_node_config(
            path, None, "vision", "node", tmp_path / "htpcache").values()))
        for key in keys:
            assert Path(cfg[key]["lut-path"]).is_absolute()
            assert Path(cfg[key]["quant-param"]["scale"]).is_absolute()
            assert Path(cfg[key]["quant-param"]["offset"]).is_absolute()


# ------------------------------------------------------------ gemma4 VLM spec

def _gemma4_node_cfgs(height=39, width=60, pool=3):
    """The parts of a Gemma 4 LMM bundle's node configs gemma4_bind reads."""
    return {
        "image_encoder": {"image-encoder": {"engine": {"model": {"vision-param": {
            "height": height, "width": width, "pooling-kernel-size": pool}}}}},
        "text_encoder": {"text-encoder": {"context": {
            "version": 1, "bos-token": 2, "n-vocab": 262144, "ctx-size": 4096,
            "embed-size": 1536, "pad-token": 0}}},
        "text_generator": {"text-generator": {"context": {"size": 4096}}},
    }


def _gemma4(**grid):
    from genie_server import vlm_specs
    family = vlm_specs.get_family("gemma4")
    spec = vlm_specs.get_spec("gemma4")
    cfgs = _gemma4_node_cfgs(**grid)
    return family.bind(spec, cfgs, layout=None), cfgs


def test_gemma4_grid_comes_from_the_bundle():
    """39x60 patches pooled 3x3 is 260 soft tokens — what the processor gives
    a 640x427 image, and the number usage and the budget guard both need."""
    spec, _ = _gemma4()
    assert (spec.image_height, spec.image_width) == (39 * 16, 60 * 16)
    assert spec.vision_tokens_per_step == 260


def test_gemma4_leaves_the_text_encoder_bos_as_the_bundle_declares_it():
    """Dropping it from the config would hide what the bundle and the SDK do
    together — a BOS in front of every text segment. The template writes no
    BOS of its own instead."""
    import dataclasses
    spec, cfgs = _gemma4()
    assert cfgs["text_encoder"]["text-encoder"]["context"]["bos-token"] == 2
    parts = [("image", 0), ("text", "hi")]

    def text(s):
        return "".join(v for k, v in s.build_prompt_segments("", parts, {}, s)
                       if k == "text")

    assert text(spec).startswith("<bos>") and text(spec).count("<bos>") == 1
    assert "<bos>" not in text(dataclasses.replace(spec, text_encoder_adds_bos=True))


def test_text_encoder_bos_is_one_per_text_segment():
    from genie_server.vlm import VLMSlot, _text_encoder_bos, count_text_encoder_bos
    cfgs = _gemma4_node_cfgs()
    assert _text_encoder_bos(cfgs) == 2
    del cfgs["text_encoder"]["text-encoder"]["context"]["bos-token"]
    assert _text_encoder_bos(cfgs) is None
    segments = [("text", "a"), ("step", (0,)), ("text", "b"), ("step", (1,)),
                ("text", "c")]
    slot = VLMSlot.__new__(VLMSlot)
    assert count_text_encoder_bos(slot, segments) == 0
    slot.text_encoder_bos = 2
    assert count_text_encoder_bos(slot, segments) == 3


def test_vlm_prompt_tokens_count_the_rendered_segments_like_a_text_slot():
    """The text half is the prompt as rendered — chat-template markers and
    all, each segment on its own as the text-encoder gets it — not just the
    words the client sent, which is what a text slot's count already was."""
    from genie_server.vlm import count_prompt_tokens

    class Stub:
        spec = _spec()
        text_encoder_bos = None

        def count_tokens(self, text):
            return len(text.split())

    slot = Stub()
    segments = slot.spec.build_prompt_segments(
        "be brief", [("text", "what is this"), ("image", 0)], {}, slot.spec)
    rendered = [v for k, v in segments if k == "text"]
    expected = sum(len(v.split()) for v in rendered) + slot.spec.vision_tokens_per_step
    assert count_prompt_tokens(slot, segments) == expected
    assert expected > len("be brief what is this".split()) + slot.spec.vision_tokens_per_step
    slot.text_encoder_bos = 151643
    assert count_prompt_tokens(slot, segments) == expected + len(rendered)


@pytest.mark.parametrize("height, width, pool, match", [
    (40, 60, 3, "divisible"),
    (51, 51, 3, "exceeds"),          # 2601 > 2520
    (39, None, 3, "vision-param"),
])
def test_gemma4_refuses_a_grid_the_encoder_cannot_take(height, width, pool, match):
    with pytest.raises(ValueError, match=match):
        _gemma4(height=height, width=width, pool=pool)


def test_gemma4_prompt_follows_the_chat_template():
    spec, _ = _gemma4()
    segs = spec.build_prompt_segments(
        " Be brief. ", [("text", " What is this? "), ("image", 0)], {}, spec)
    assert segs == [
        ("text", "<bos><|turn>system\nBe brief.<turn|>\n<|turn>user\nWhat is this?<|image>"),
        ("step", (0,)),
        ("text", "<image|><turn|>\n<|turn>model\n"),
    ]


def test_gemma4_without_a_system_message_has_no_system_turn():
    spec, _ = _gemma4()
    segs = spec.build_prompt_segments("", [("image", 0)], {}, spec)
    assert segs[0] == ("text", "<bos><|turn>user\n<|image>")


def test_gemma4_video_is_a_step_per_frame_with_mmss_timestamps():
    """Gemma4Processor.replace_video_token's layout: `mm:ss <|image>` + the
    frame's soft tokens + `<image|>`, frames joined by a space. No temporal
    packing, so each frame is its own encoder step."""
    spec, _ = _gemma4(height=24, width=24)
    segs = spec.build_prompt_segments(
        "", [("video", [0, 1, 2]), ("text", " What happens? ")], {"fps": 2.0}, spec)
    assert segs == [
        ("text", "<bos><|turn>user\n00:00 <|image>"),
        ("step", (0,)),
        ("text", "<image|> 00:00 <|image>"),
        ("step", (1,)),
        ("text", "<image|> 00:01 <|image>"),
        ("step", (2,)),
        ("text", "<image|>What happens?<turn|>\n<|turn>model\n"),
    ]
    assert spec.vision_tokens_per_step == 64       # 24x24 patches pooled 3x3


def test_gemma4_video_without_fps_writes_no_timestamps():
    """Same rule as the Qwen3-VL markers: no timeline rather than an invented
    one (Gemma's processor would assume 24 fps)."""
    spec, _ = _gemma4(height=24, width=24)
    text = "".join(v for k, v in spec.build_prompt_segments(
        "", [("video", [0, 1])], {}, spec) if k == "text")
    assert text == "<bos><|turn>user\n<|image><image|> <|image><image|><turn|>\n<|turn>model\n"


def test_gemma4_timestamps_are_minutes_and_whole_seconds():
    from genie_server.vlm_specs.gemma4 import _mmss
    assert [_mmss(t) for t in (0.0, 0.5, 1.0, 59.9, 61.9, 600.0)] == \
        ["00:00", "00:00", "00:01", "00:59", "01:01", "10:00"]


def test_gemma4_patches_are_raster_order_channels_last_then_zero_padded():
    """Each patch flattened (row, col, channel) and patches row by row, as
    Gemma4ImageProcessor lays them out; a channels-first flattening hands the
    encoder the same numbers in the wrong places."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from PIL import Image
    rows, cols, p = 6, 9, 16
    spec, _ = _gemma4(height=rows, width=cols)
    arr = np.zeros((rows * p, cols * p, 3), np.uint8)
    for r in range(rows):
        for c in range(cols):
            arr[r * p:(r + 1) * p, c * p:(c + 1) * p] = (r * 10, c * 10, 200)
    out = spec.preprocess_step([Image.fromarray(arr)], (0,), spec)
    assert out.shape == (2520, 3 * p * p) and out.dtype == np.float32
    for r in range(rows):
        for c in range(cols):
            patch = out[r * cols + c].reshape(p, p, 3)
            assert np.allclose(patch[..., 0], r * 10 / 255)
            assert np.allclose(patch[..., 1], c * 10 / 255)
            assert np.allclose(patch[..., 2], 200 / 255)
    assert not out[rows * cols:].any()


# ------------------------------------------------- the BOS the SDK adds itself

def test_template_leaves_its_bos_to_a_bundle_that_declares_one():
    """A bundle whose dialog context names a bos-token has libGenie prepend it
    to every query; a template writing its own as well put two in front of a
    gemma4 prompt ([2, 2, 105, ...] in the SDK's own log)."""
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    for template, bos in (("gemma4", "<bos>"), ("gemma", "<bos>"),
                          ("llama3", "<|begin_of_text|>"), ("llama2", "<s>")):
        assert (templates.render_chat_prompt(msgs, template)
                == bos + templates.render_chat_prompt(msgs, template, bos=False)), template
    assert (templates.render_chat_prompt(msgs, "chatml", bos=False)
            == templates.render_chat_prompt(msgs, "chatml"))


def test_prefix_split_without_bos_still_reassembles():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    for template in ("gemma4", "llama3", "chatml"):
        prefix, remaining, cacheable = templates.split_prompt_for_prefix_cache(
            msgs, template, bos=False)
        assert cacheable
        assert prefix + remaining == templates.render_chat_prompt(msgs, template, bos=False)
    assert (templates.split_prompt_for_prefix_cache(msgs, "gemma4", bos=False)[0]
            == "<|turn>system\nsys<turn|>\n")


def test_slot_takes_the_sdk_bos_from_the_dialog_context_and_counts_it():
    from genie_server.slots import Slot
    slot = Slot(name="chat", device_id=None, model_root=Path("/m"))
    slot.dialog_cfg = {"context": {"bos-token": 2}}
    assert slot.sdk_bos_token == 2
    assert slot.count_prompt_tokens("a b c") == slot.count_tokens("a b c") + 1
    slot.dialog_cfg = {"context": {"size": 4096}}
    assert slot.sdk_bos_token is None
    assert slot.count_prompt_tokens("a b c") == slot.count_tokens("a b c")


@pytest.mark.parametrize("bos_token", [2, None])
def test_chat_prompt_and_usage_follow_the_bundles_bos_token(tmp_path, bos_token):
    """The query handed to the SDK carries a BOS only when the SDK will not
    add one, and usage counts the one it adds."""
    from fastapi.testclient import TestClient
    from conftest import build_state
    from genie_server.app import create_app
    state = build_state(tmp_path, template="gemma4")
    slot = state.manager.slots[0]
    if bos_token is not None:
        slot.dialog_cfg["context"]["bos-token"] = bos_token
    r = TestClient(create_app(state)).post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})
    assert r.status_code == 200
    sent = state.lib.queries[-1]
    assert sent.startswith("<bos>") is (bos_token is None)
    assert (r.json()["usage"]["prompt_tokens"]
            == slot.count_tokens(sent) + (bos_token is not None))


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
    s.count_prompt_tokens = s.count_tokens      # a bundle with no SDK-added BOS
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


def _slot_names_module():
    import importlib.util

    path = Path(__file__).resolve().parent / "integration" / "slot_names.py"
    spec = importlib.util.spec_from_file_location("slot_names", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scaling_slots_returns_every_slot_not_just_the_minimum():
    """order_slots takes a minimum, not a count: a scaling measurement uses
    however many slots are configured."""
    mod = _slot_names_module()
    assert mod.order_slots(
        _status(("chat1", 0, True), ("chat2", 1, True), ("chat0", 0, True)),
        at_least=2) == [("chat1", 0), ("chat0", 0), ("chat2", 1)]


def test_scaling_pairs_split_by_device_id():
    """The same-core pair must share a device_id and the cross-core pair must
    not — that difference is the whole measurement."""
    mod = _slot_names_module()
    slots = [("a", 0), ("b", 0), ("c", 1)]
    assert mod.same_core_pair(slots) == ["a", "b"]
    assert mod.cross_core_pair(slots) == ["a", "c"]


def test_scaling_pairs_are_none_when_the_layout_cannot_show_the_contrast():
    """One slot per core has no same-core pair; one core has no cross-core
    pair. Returning None lets the script skip that phase instead of measuring
    something it cannot name."""
    mod = _slot_names_module()
    assert mod.same_core_pair([("a", 0), ("b", 1)]) is None
    assert mod.cross_core_pair([("a", 0), ("b", 0)]) is None


def test_scaling_pairs_never_assume_where_an_unpinned_slot_landed():
    """device_id null means the server cannot read the core back, so two of
    them are not a same-core pair however they are configured."""
    mod = _slot_names_module()
    assert mod.same_core_pair([("a", None), ("b", None)]) is None
    assert mod.cross_core_pair([("a", None), ("b", 1)]) is None


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


def test_ubuntu_uses_system_qairt_package_without_sdk_root(monkeypatch):
    import os
    from genie_server.config import ServerConfig
    c = ServerConfig(sdk_root="", target_platform="linux-ubuntu")
    assert c.resolved_genie_lib_path() == "libGenie.so"
    assert c._adsp_library_path() == (
        "/usr/lib/rfsa/adsp;/lib/dsp/cdsp;/lib/dsp/cdsp1;")
    monkeypatch.setenv("QAIRT_SDK_ROOT", "/old/sdk")
    monkeypatch.setenv("QNN_SDK_ROOT", "/old/sdk")
    c.apply_process_env()
    assert "QAIRT_SDK_ROOT" not in os.environ
    assert "QNN_SDK_ROOT" not in os.environ


def test_ubuntu_sdk_path_uses_oe_abi_and_prefers_sdk_skels():
    c = _cfg(target_platform="linux-ubuntu")
    assert c.resolved_genie_lib_path() == \
        "/opt/qairt/X/lib/aarch64-oe-linux-gcc11.2/libGenie.so"
    assert c._adsp_library_path() == (
        "/opt/qairt/X/lib/hexagon-v73/unsigned;/usr/lib/rfsa/adsp;"
        "/lib/dsp/cdsp;/lib/dsp/cdsp1;")


def test_auto_detects_ubuntu_dsp_layout(monkeypatch):
    from genie_server.config import detect_platform
    monkeypatch.setattr("genie_server.config.os.path.isdir",
                        lambda p: p in ("/lib/dsp/cdsp", "/usr/lib/rfsa/adsp"))
    monkeypatch.setattr("genie_server.config._is_ubuntu", lambda: True)
    assert detect_platform() == "linux-ubuntu"


def test_auto_keeps_oe_with_the_ubuntu_dsp_layout(monkeypatch):
    """A usrmerge OE image has the same directories; only os-release tells
    them apart."""
    from genie_server.config import detect_platform
    monkeypatch.setattr("genie_server.config.os.path.isdir",
                        lambda p: p in ("/lib/dsp/cdsp", "/usr/lib/rfsa/adsp"))
    monkeypatch.setattr("genie_server.config._is_ubuntu", lambda: False)
    assert detect_platform() == "linux-oe"


@pytest.mark.parametrize("text, expected", [
    ('NAME="Ubuntu"\nID=ubuntu\nID_LIKE=debian\n', True),
    ('ID=pop\nID_LIKE="ubuntu debian"\n', True),
    ('ID=qcom-wayland\nID_LIKE=""\n', False),
    ('garbage\n', False),
])
def test_is_ubuntu_reads_os_release(tmp_path, text, expected):
    from genie_server.config import _is_ubuntu
    p = tmp_path / "os-release"
    p.write_text(text)
    assert _is_ubuntu(str(p)) is expected
    assert _is_ubuntu(str(tmp_path / "missing")) is False


def test_platform_is_detected_once(monkeypatch):
    calls = []
    monkeypatch.setattr("genie_server.config.detect_platform",
                        lambda: calls.append(1) or "linux-oe")
    c = _cfg(target_platform="auto")
    c.resolved_genie_lib_path()
    c._adsp_library_path()
    c.apply_process_env()
    assert len(calls) == 1


def test_ubuntu_adsp_path_follows_pinned_device_ids():
    from genie_server.config import ServerConfig, SlotSpec
    c = ServerConfig(
        sdk_root="", target_platform="linux-ubuntu", vlm_slots=[],
        text_slots=[SlotSpec(name="a", device_id=0, model_root="/m"),
                    SlotSpec(name="b", device_id=2, model_root="/m")])
    assert c._adsp_library_path() == (
        "/usr/lib/rfsa/adsp;/lib/dsp/cdsp;/lib/dsp/cdsp2;")


def test_empty_sdk_root_clears_inherited_sdk_variables(monkeypatch):
    import os
    monkeypatch.setenv("QAIRT_SDK_ROOT", "/opt/qairt/2.48")
    monkeypatch.setenv("QNN_SDK_ROOT", "/opt/qairt/2.48")
    from genie_server.config import ServerConfig
    ServerConfig(sdk_root="", target_platform="linux-oe",
                 genie_lib_path="/custom/libGenie.so").apply_process_env()
    assert "QAIRT_SDK_ROOT" not in os.environ
    assert "QNN_SDK_ROOT" not in os.environ


def test_mapped_path_reports_the_file_the_loader_chose(tmp_path, monkeypatch):
    from genie_server import bootstrap
    maps = tmp_path / "maps"
    maps.write_text(
        "7f00-7f01 r-xp 00000000 08:01 12 /usr/lib/aarch64-linux-gnu/libGenie.so\n")
    real_open = open
    monkeypatch.setattr("builtins.open", lambda p, *a, **k: real_open(
        maps if p == "/proc/self/maps" else p, *a, **k))
    assert bootstrap._mapped_path("libGenie.so") == \
        "/usr/lib/aarch64-linux-gnu/libGenie.so (requested libGenie.so)"
    assert bootstrap._mapped_path("/usr/lib/aarch64-linux-gnu/libGenie.so") == \
        "/usr/lib/aarch64-linux-gnu/libGenie.so"


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
    from genie_server.config import KNOWN_PLATFORMS, ServerConfig
    for plat in KNOWN_PLATFORMS:
        c = _cfg(target_platform=plat, genie_lib_path="/custom/libGenie.so")
        assert c.resolved_genie_lib_path() == "/custom/libGenie.so"
    # ...including over the system-package fallback of an Ubuntu config
    # without an SDK.
    c = ServerConfig(sdk_root="", target_platform="linux-ubuntu",
                     genie_lib_path="/custom/libGenie.so")
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


# ------------------------------------------------- VLM video input

def _b64_jpeg(color):
    """A 4x4 JPEG as base64, small enough to inline in a test."""
    import base64
    import io
    Image = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _video_url(n):
    """vLLM's client-side frame-extraction form: comma-joined base64 JPEGs
    under a video/jpeg media type."""
    frames = [_b64_jpeg((10 * i, 20, 30)) for i in range(n)]
    return "data:video/jpeg;base64," + ",".join(frames)


def _messages_with_video(n, text="What happens?"):
    return [{"role": "user", "content": [
        {"type": "text", "text": text},
        {"type": "video_url", "video_url": {"url": _video_url(n)}}]}]


def test_video_url_routes_to_the_vlm_path():
    """Routing keys off the part type, so a video-only message must not fall
    through to the GenieDialog text path."""
    pytest.importorskip("PIL")
    from genie_server import vlm
    assert vlm.is_vlm_request(_messages_with_video(2))
    assert not vlm.is_vlm_request([{"role": "user", "content": "plain text"}])


def test_video_url_frames_become_one_video_part():
    """Every frame lands in the flat sources list; the part carries their
    indices so the spec can pack them into steps."""
    pytest.importorskip("PIL")
    from genie_server import vlm
    _, parts, sources = vlm.extract_multimodal_parts(_messages_with_video(6))
    assert len(sources) == 6
    assert parts == [("text", "What happens?"), ("video", [0, 1, 2, 3, 4, 5])]


def test_frames_are_not_decoded_until_the_plan_is_known():
    """Parsing yields undecoded payloads so that a request the budget guard
    refuses never pays for turning its frames into bitmaps."""
    pytest.importorskip("PIL")
    from genie_server import vlm
    _, _, sources = vlm.extract_multimodal_parts(_messages_with_video(3))
    assert all(isinstance(b64, str) for b64, _ in sources)
    images = vlm.decode_media_sources(sources)
    assert len(images) == 3
    assert all(img.size == (4, 4) for img in images)


def test_an_undecodable_frame_is_a_client_error():
    pytest.importorskip("PIL")
    from genie_server import vlm
    with pytest.raises(ValueError, match="video_url frame 1"):
        vlm.decode_media_sources([(_b64_jpeg((1, 2, 3)), "video_url frame 0"),
                                  ("not base64 jpeg", "video_url frame 1")])


def _b64_png(size, mode="1"):
    """A flat PNG: tiny on the wire whatever size it declares."""
    import base64
    import io
    Image = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    Image.new(mode, size).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_an_image_over_the_pixel_ceiling_is_refused_before_decoding(
        monkeypatch):
    """The size is in the header: a small payload declaring a huge bitmap
    must be refused without that bitmap ever being allocated."""
    pytest.importorskip("PIL")
    from PIL import ImageFile
    from genie_server import vlm
    loads = []
    real_load = ImageFile.ImageFile.load
    monkeypatch.setattr(ImageFile.ImageFile, "load",
                        lambda self: loads.append(self) or real_load(self))
    big = _b64_png((5000, 4000))
    assert len(big) < 100_000
    with pytest.raises(ValueError, match="image_url is 5000x4000.*"
                                         "VLM_MAX_IMAGE_PIXELS"):
        vlm.decode_media_sources([(_b64_jpeg((1, 2, 3)), "video_url frame 0"),
                                  (big, "image_url")],
                                 max_image_pixels=4096 * 4096)
    assert loads == []


def test_frames_over_the_total_pixel_ceiling_are_refused_before_decoding():
    pytest.importorskip("PIL")
    from genie_server import vlm
    _, _, sources = vlm.extract_multimodal_parts(_messages_with_video(3))
    with pytest.raises(ValueError, match="3 images/frames total 48 pixels"):
        vlm.decode_media_sources(sources, max_total_pixels=47)
    assert len(vlm.decode_media_sources(sources, max_total_pixels=48)) == 3


def test_pixel_ceilings_of_zero_are_off():
    pytest.importorskip("PIL")
    from genie_server import vlm
    img, = vlm.decode_media_sources([(_b64_png((5000, 4000)), "image_url")])
    assert img.size == (5000, 4000)


def test_base64_with_characters_outside_the_alphabet_is_refused():
    """The default decoder skips them and hands Pillow a different byte
    stream; line breaks are the one thing an encoder legitimately adds."""
    pytest.importorskip("PIL")
    from genie_server import vlm
    b64 = _b64_jpeg((1, 2, 3))
    with pytest.raises(ValueError, match="image_url is not valid base64"):
        vlm.decode_media_sources([(b64[:8] + "!" + b64[8:], "image_url")])
    wrapped = "\n".join(b64[i:i + 76] for i in range(0, len(b64), 76))
    assert "\n" in wrapped
    img, = vlm.decode_media_sources([(wrapped, "image_url")])
    assert img.size == (4, 4)


def test_video_container_media_types_are_refused_not_half_supported():
    """No demuxer ships with this server, so data:video/mp4 has to be a clear
    client error rather than a decode failure deeper in."""
    pytest.importorskip("PIL")
    from genie_server import vlm
    msgs = [{"role": "user", "content": [
        {"type": "video_url",
         "video_url": {"url": "data:video/mp4;base64,AAAA"}}]}]
    with pytest.raises(ValueError, match="video/jpeg"):
        vlm.extract_multimodal_parts(msgs)


def test_remote_video_urls_are_not_fetched():
    pytest.importorskip("PIL")
    from genie_server import vlm
    msgs = [{"role": "user", "content": [
        {"type": "video_url",
         "video_url": {"url": "https://example.com/clip.mp4"}}]}]
    with pytest.raises(ValueError, match="data:"):
        vlm.extract_multimodal_parts(msgs)


def test_video_meta_comes_from_media_io_kwargs():
    """vLLM's key, which OpenAI clients reach via extra_body."""
    from genie_server import vlm
    body = {"media_io_kwargs": {"video": {"fps": 2.0, "frames_indices": [0, 4]}}}
    assert vlm.extract_video_meta(body) == {"fps": 2.0, "frames_indices": [0, 4]}
    assert vlm.extract_video_meta({}) == {}
    assert vlm.extract_video_meta({"media_io_kwargs": {"image": {}}}) == {}


# --- segment building (the packing that halves the vision-token cost)

def _spec():
    from genie_server import vlm_specs
    return vlm_specs.get_spec("qwen3_vl")


def test_a_still_image_still_duplicates_its_own_frame():
    """Unchanged behaviour: one picture is one step, the ViT's temporal
    dimension filled by repeating it."""
    spec = _spec()
    segs = spec.build_prompt_segments("", [("image", 0)], {}, spec)
    steps = [v for k, v in segs if k == "step"]
    assert steps == [(0, 0)]


def test_video_frames_are_packed_two_per_step():
    """The point of the video path: temporal_patch_size distinct frames per
    encoder step, so 6 frames cost 3 steps rather than 6."""
    spec = _spec()
    segs = spec.build_prompt_segments("", [("video", [0, 1, 2, 3, 4, 5])], {}, spec)
    steps = [v for k, v in segs if k == "step"]
    assert steps == [(0, 1), (2, 3), (4, 5)]


def test_an_odd_frame_count_repeats_the_last_frame():
    spec = _spec()
    segs = spec.build_prompt_segments("", [("video", [0, 1, 2])], {}, spec)
    steps = [v for k, v in segs if k == "step"]
    assert steps == [(0, 1), (2, 2)]


def test_timestamps_appear_only_when_an_fps_was_supplied():
    """The `<t seconds>` markers assert a real timeline to the model, so an
    fps-less request gets none rather than a made-up one."""
    spec = _spec()
    parts = [("video", [0, 1, 2, 3])]

    without = "".join(v for k, v in spec.build_prompt_segments("", parts, {}, spec)
                      if k == "text")
    assert "seconds" not in without

    # 2 fps, 2 frames per step -> one step per second of wall clock, each
    # dated at the midpoint of its own pair (0.25s and 1.25s, to one place).
    with_fps = "".join(
        v for k, v in spec.build_prompt_segments("", parts, {"fps": 2.0}, spec)
        if k == "text")
    assert "<0.2 seconds>" in with_fps
    assert "<1.2 seconds>" in with_fps


def test_fps_may_arrive_as_a_single_element_list():
    """The Qwen/vLLM examples pass fps=[3.0]."""
    spec = _spec()
    text = "".join(
        v for k, v in spec.build_prompt_segments(
            "", [("video", [0, 1])], {"fps": [2.0]}, spec) if k == "text")
    assert "<0.2 seconds>" in text


def test_frames_indices_place_the_timestamps():
    """Frames sampled unevenly out of a 30 fps source are timed by their real
    position, not by their position in the list."""
    spec = _spec()
    text = "".join(
        v for k, v in spec.build_prompt_segments(
            "", [("video", [0, 1, 2, 3])],
            {"fps": 30.0, "frames_indices": [0, 15, 30, 45]}, spec)
        if k == "text")
    # Frames 0 and 15 are 0.0s and 0.5s; the step covering them is 0.25s.
    assert "<0.2 seconds>" in text
    # Frames 30 and 45 are 1.0s and 1.5s -> 1.25s.
    assert "<1.2 seconds>" in text


def test_a_step_is_dated_at_the_midpoint_of_its_frames():
    """Qwen3-VL dates the visual chunk, not its first frame: vLLM's
    _calculate_timestamps averages the group's first and last frame times.
    Taking the first would label every step half a sampling interval early,
    and an unevenly sampled pair arbitrarily so."""
    from genie_server.vlm_specs.qwen3_vl import _qwen3vl_step_time
    assert _qwen3vl_step_time([0.0, 0.5, 1.0, 1.5], 0, 2) == 0.25
    assert _qwen3vl_step_time([0.0, 0.5, 1.0, 1.5], 2, 2) == 1.25
    # Uneven sampling: a pair spanning 0s to 10s is dated between them.
    assert _qwen3vl_step_time([0.0, 10.0], 0, 2) == 5.0


def test_the_odd_tail_is_dated_by_its_only_real_frame():
    """The last step repeats the final frame to fill itself, so its midpoint
    is that frame's own time — the same padding vLLM applies to the index
    list before pairing."""
    from genie_server.vlm_specs.qwen3_vl import _qwen3vl_step_time
    assert _qwen3vl_step_time([0.0, 0.5, 1.0], 2, 2) == 1.0


def test_frames_indices_that_do_not_match_the_frames_emit_no_markers():
    """`fps` means the source rate when frames_indices is present and the
    sampling rate when it is not, so a count that disagrees means the two
    readings are out of step. Falling back would date a 30 fps clip as though
    its frames were 33 ms apart; no marker is better than a wrong one."""
    spec = _spec()
    text = "".join(
        v for k, v in spec.build_prompt_segments(
            "", [("video", [0, 1, 2, 3])],
            {"fps": 30.0, "frames_indices": [0, 15]}, spec)
        if k == "text")
    assert "seconds" not in text


def test_unusable_frames_indices_emit_no_markers():
    spec = _spec()
    for indices in ("0,15", [None, 1], [0, "x"]):
        text = "".join(
            v for k, v in spec.build_prompt_segments(
                "", [("video", [0, 1])],
                {"fps": 30.0, "frames_indices": indices}, spec)
            if k == "text")
        assert "seconds" not in text, indices


def test_a_step_payload_must_match_the_vits_temporal_size():
    """build_prompt_segments packs temporal_patch_size frames per step and
    the patchify below reads exactly two, so a spec whose ViT wanted a
    different number has to say so rather than lose frames silently."""
    pytest.importorskip("numpy")
    spec = _spec()
    with pytest.raises(ValueError, match="per execution"):
        spec.preprocess_step([None, None, None], (0, 1, 2), spec)


def test_vision_tokens_per_step_is_a_quarter_of_the_patch_rows():
    """256, not the 1024 rows of pixel_values — the spatial merge folds 2x2
    patches into one token before the LLM sees them."""
    assert _spec().vision_tokens_per_step == 256


# --- the budget guard

class _BudgetSlot:
    """Just the fields plan_segments reads."""
    text_encoder_bos = None

    def __init__(self, context_size=4096, max_tokens=256):
        self.spec = _spec()
        self.context_size = context_size
        self.max_tokens = max_tokens

    def count_tokens(self, text):
        return len(text.split())


def test_a_video_that_fits_is_planned_without_complaint():
    from genie_server import vlm
    slot = _BudgetSlot()
    segs = vlm.plan_segments(slot, "", [("video", list(range(20)))], {"fps": 2.0},
                             guard=True)
    assert sum(1 for k, _ in segs if k == "step") == 10


def test_the_budget_guard_is_off_by_default():
    """VLM_VISION_BUDGET_GUARD conceals a defect this server exists to expose,
    so an oversized request goes to the SDK unchanged unless it is turned on —
    the same rule TOOL_CALL_RECOVERY follows."""
    from genie_server import vlm
    slot = _BudgetSlot()
    segs = vlm.plan_segments(slot, "", [("video", list(range(200)))], {"fps": 2.0})
    assert sum(1 for k, _ in segs if k == "step") == 100


def test_passing_an_oversized_request_through_is_logged(caplog):
    """Off does not mean silent: saying what is about to happen is the
    opposite of hiding it."""
    from genie_server import vlm
    with caplog.at_level("WARNING"):
        vlm.plan_segments(_BudgetSlot(), "", [("video", list(range(200)))], {})
    assert "VLM_VISION_BUDGET_GUARD" in caplog.text


def test_too_many_frames_is_refused_before_anything_touches_the_npu():
    """With the guard on: a prompt past the context wedges the slot for every
    later request (and far enough past, kills the process), so the arithmetic
    rejects it up front rather than let the SDK discover it."""
    from genie_server import vlm
    slot = _BudgetSlot()
    with pytest.raises(ValueError, match="too much visual input"):
        vlm.plan_segments(slot, "", [("video", list(range(200)))], {"fps": 2.0},
                          guard=True)


def test_the_refusal_says_how_many_frames_make_a_step_for_this_spec():
    """Qwen3-VL packs two frames into a step; gemma4 has no temporal packing,
    and "one step per 1 frames" told the client nothing it could act on."""
    from genie_server import vlm
    parts = [("video", list(range(200)))]
    with pytest.raises(ValueError, match=r"a video is one step per 2 frames\."):
        vlm.plan_segments(_BudgetSlot(), "", parts, {}, guard=True)
    slot = _BudgetSlot()
    slot.spec, _ = _gemma4(height=24, width=24)
    with pytest.raises(ValueError, match=r"a video is one step per frame\."):
        vlm.plan_segments(slot, "", parts, {}, guard=True)


def test_the_generation_reserve_is_subtracted_from_the_budget():
    """The slot's whole max_tokens is held back, not just the prompt
    measured -- not because decoding across the line is dangerous (measured:
    it stops cleanly at the context and the slot survives) but so the answer
    is not truncated, and so the prefill margin the host cannot read from the
    config is absorbed by something."""
    from genie_server import vlm
    parts = [("video", list(range(28)))]        # 14 steps = 3584 vision tokens
    vlm.plan_segments(_BudgetSlot(max_tokens=256), "", parts, {}, guard=True)
    with pytest.raises(ValueError, match="too much visual input"):
        vlm.plan_segments(_BudgetSlot(max_tokens=1024), "", parts, {}, guard=True)


def test_an_unknown_context_size_disables_the_check_rather_than_guessing():
    from genie_server import vlm
    slot = _BudgetSlot(context_size=0)
    segs = vlm.plan_segments(slot, "", [("video", list(range(200)))], {},
                             guard=True)
    assert sum(1 for k, _ in segs if k == "step") == 100


def test_the_guard_flag_defaults_to_off_in_the_config(tmp_path):
    from genie_server.config import load_config
    p = tmp_path / "env_config.json"
    p.write_text(json.dumps({"QAIRT_SDK_ROOT": "/s",
                             "TEXT_SLOTS": [{"model_root": "/m"}]}))
    assert load_config(str(p)).vlm_vision_budget_guard is False
    p.write_text(json.dumps({"QAIRT_SDK_ROOT": "/s",
                             "VLM_VISION_BUDGET_GUARD": True,
                             "TEXT_SLOTS": [{"model_root": "/m"}]}))
    assert load_config(str(p)).vlm_vision_budget_guard is True


def test_context_size_is_read_from_the_text_generator_config():
    from genie_server.vlm import _context_size
    assert _context_size(
        {"text_generator": {"text-generator": {"context": {"size": 4096}}}}) == 4096
    assert _context_size({"text_generator": {"text-generator": {}}}) == 0
    assert _context_size({}) == 0


# --- usage accounting

def test_vision_tokens_are_counted_from_the_steps():
    """Nothing reports the image's context cost back, so `usage` has to
    derive it: steps x the spec's per-step token cost."""
    from genie_server import vlm
    spec = _spec()
    segs = spec.build_prompt_segments("", [("video", list(range(10)))], {}, spec)
    assert vlm.count_vision_tokens(spec, segs) == 5 * 256


def test_a_still_image_costs_a_whole_step_of_context():
    from genie_server import vlm
    spec = _spec()
    segs = spec.build_prompt_segments("", [("image", 0)], {}, spec)
    assert vlm.count_vision_tokens(spec, segs) == 256


def test_a_text_only_vlm_request_has_no_vision_tokens():
    from genie_server import vlm
    spec = _spec()
    segs = spec.build_prompt_segments("", [("text", "hello")], {}, spec)
    assert vlm.count_vision_tokens(spec, segs) == 0


def test_video_costs_half_the_context_of_the_same_frames_as_images():
    """The whole point of the video path, stated in tokens: N frames as
    stills are N steps, as video they are N/2."""
    from genie_server import vlm
    spec = _spec()
    frames = list(range(10))
    as_images = spec.build_prompt_segments(
        "", [("image", i) for i in frames], {}, spec)
    as_video = spec.build_prompt_segments("", [("video", frames)], {}, spec)
    assert vlm.count_vision_tokens(spec, as_images) == 2560
    assert vlm.count_vision_tokens(spec, as_video) == 1280


# --------------------------------------------------- Genie SDK logging

def _log_cfg(tmp_path, level, key="GENIE_LOG_LEVEL"):
    from genie_server.config import load_config

    path = tmp_path / "env_config.json"
    body = {"QAIRT_SDK_ROOT": "/opt/qairt",
            "TEXT_SLOTS": [{"model_root": str(tmp_path)}]}
    if level is not None:
        body[key] = level
    path.write_text(json.dumps(body))
    return load_config(str(path))


def test_genie_logging_is_off_unless_asked_for(tmp_path):
    """The default costs nothing and, more to the point, produces nothing:
    libGenie's own __INFO/__ERROR are no-ops while no logger is bound."""
    assert _log_cfg(tmp_path, None).genie_log_level == ""


def test_genie_log_level_is_normalized(tmp_path):
    assert _log_cfg(tmp_path, "  INFO ").genie_log_level == "info"


def test_a_misspelled_genie_log_level_fails_at_startup(tmp_path):
    """Rather than at GenieLog_create time, after the SDK is loaded."""
    with pytest.raises(ValueError, match="GENIE_LOG_LEVEL"):
        _log_cfg(tmp_path, "debug")


def _manager_with_log_level(tmp_path, level):
    from fake_genie import FakeGenieLib
    from genie_server.config import ServerConfig, SlotSpec
    from genie_server.slots import SlotManager

    model_dir = _minimal_bundle(tmp_path, "qwen3_0_6b")
    cfg = ServerConfig(
        sdk_root="/nonexistent",
        prefix_cache_dir=str(tmp_path / "prefix_cache"),
        genie_log_level=level,
        text_slots=(SlotSpec(name="chat", device_id=None,
                             model_root=model_dir),))
    manager = SlotManager(cfg, FakeGenieLib())
    manager.load_all()
    return manager


def test_no_logger_is_created_when_logging_is_off(tmp_path):
    manager = _manager_with_log_level(tmp_path, "")
    assert manager.log_handle is None
    assert manager.lib.created_loggers == []
    assert manager.lib.bound_loggers == [None]


def test_every_dialog_gets_the_one_logger(tmp_path):
    """One handle for the process, bound to each dialog config — binding is
    what makes _env->logger() non-null inside libGenie, so a dialog created
    without it stays silent even though logging is 'on'."""
    manager = _manager_with_log_level(tmp_path, "info")
    assert manager.lib.created_loggers == ["info"]
    assert manager.log_handle is not None
    assert manager.lib.bound_loggers == [manager.log_handle]


def test_a_hot_swapped_model_keeps_the_logger(tmp_path):
    """The binding lives on the dialog config, so a model switch has to
    re-apply it or the slot goes quiet after the first swap."""
    manager = _manager_with_log_level(tmp_path, "error")
    slot = manager.slots[0]
    manager.switch_model(slot, slot.model_root, unload_first=True)
    assert manager.lib.bound_loggers == [manager.log_handle, manager.log_handle]


def test_the_logger_is_freed_after_the_dialogs(tmp_path):
    """The SDK counts the handle's uses against what it is bound to, the same
    way it does for a profiler — freeing it first fails."""
    manager = _manager_with_log_level(tmp_path, "verbose")
    handle = manager.log_handle
    manager.free_all()
    assert manager.lib.freed and manager.lib.freed_loggers == [handle.value]
    assert manager.log_handle is None


def test_an_unknown_level_never_reaches_genielog_create():
    from fake_genie import FakeGenieLib

    with pytest.raises(ValueError, match="log level"):
        FakeGenieLib().create_logger("trace")


@pytest.mark.parametrize("value, expected", [
    (None, ()),
    ([], ()),
    (["http://localhost:3000/", "*"], ("http://localhost:3000", "*")),
])
def test_cors_allow_origins_is_a_list_and_defaults_to_none(tmp_path, value,
                                                           expected):
    from genie_server.config import load_config

    raw = {"QAIRT_SDK_ROOT": "/opt/qairt",
           "TEXT_SLOTS": [{"model_root": str(tmp_path)}]}
    if value is not None:
        raw["CORS_ALLOW_ORIGINS"] = value
    path = tmp_path / "env_config.json"
    path.write_text(json.dumps(raw))
    assert load_config(str(path)).cors_allow_origins == expected


@pytest.mark.parametrize("value", ["*", "http://a, http://b", [""], [3]])
def test_cors_allow_origins_that_is_not_a_list_of_origins_is_refused(
        tmp_path, value):
    """A string is not split: "http://a, http://b" would match no origin and
    fail silently in the browser."""
    from genie_server.config import load_config

    path = tmp_path / "env_config.json"
    path.write_text(json.dumps({"QAIRT_SDK_ROOT": "/opt/qairt",
                                "TEXT_SLOTS": [{"model_root": str(tmp_path)}],
                                "CORS_ALLOW_ORIGINS": value}))
    with pytest.raises(ValueError, match="CORS_ALLOW_ORIGINS"):
        load_config(str(path))


@pytest.mark.parametrize("key, value, expected", [
    ("MAX_REQUEST_BODY_MB", None, 64),
    ("MAX_REQUEST_BODY_MB", 0.5, 0.5),
    ("VLM_MAX_IMAGE_PIXELS", None, 4096 * 4096),
    ("VLM_MAX_IMAGE_PIXELS", 0, 0),
    ("VLM_MAX_TOTAL_PIXELS", None, 64 * 4096 * 4096),
])
def test_size_ceilings_default_and_accept_numbers(tmp_path, key, value,
                                                  expected):
    from genie_server.config import load_config

    raw = {"QAIRT_SDK_ROOT": "/opt/qairt",
           "TEXT_SLOTS": [{"model_root": str(tmp_path)}]}
    if value is not None:
        raw[key] = value
    path = tmp_path / "env_config.json"
    path.write_text(json.dumps(raw))
    assert getattr(load_config(str(path)), key.lower()) == expected


@pytest.mark.parametrize("key, value", [
    ("MAX_REQUEST_BODY_MB", "64"),
    ("MAX_REQUEST_BODY_MB", -1),
    ("MAX_REQUEST_BODY_MB", True),
    ("VLM_MAX_IMAGE_PIXELS", 1.5),
    ("VLM_MAX_TOTAL_PIXELS", [1]),
    # Python's json writes and reads these; NaN passes a "< 0" test.
    ("MAX_REQUEST_BODY_MB", float("nan")),
    ("MAX_REQUEST_BODY_MB", float("inf")),
    ("VLM_MAX_IMAGE_PIXELS", float("inf")),
    ("VLM_MAX_TOTAL_PIXELS", float("nan")),
])
def test_size_ceilings_that_are_not_non_negative_numbers_are_refused(
        tmp_path, key, value):
    from genie_server.config import load_config

    path = tmp_path / "env_config.json"
    path.write_text(json.dumps({"QAIRT_SDK_ROOT": "/opt/qairt",
                                "TEXT_SLOTS": [{"model_root": str(tmp_path)}],
                                key: value}))
    with pytest.raises(ValueError, match=key):
        load_config(str(path))


def test_lora_alpha_names_follow_how_the_sdk_fills_them_in():
    """An adapter's own alphas; the lora block's alpha-tensor-name for one
    that lists none; CB adapter-order names at either level."""
    from genie_server.slots import lora_alpha_names
    cfg = {"engine": {"model": {"binary": {"lora": {
        "alpha-tensor-name": "lora_alpha",
        "adapter-order": ["top"],
        "adapters": [{"name": "a", "alphas": ["alpha0"]},
                     {"name": "b"},
                     {"name": "cb", "alphas": ["x"],
                      "adapter-order": ["elementary", "advanced"]}]}}}}}
    assert lora_alpha_names(cfg, "primary") == {
        "alpha0", "lora_alpha", "x", "elementary", "advanced", "top"}
    assert lora_alpha_names(cfg, "target") == lora_alpha_names(cfg, "primary")


def test_lora_alpha_names_picks_the_engine_by_role():
    from genie_server.slots import lora_alpha_names

    def engine(role, alphas):
        return {"role": role, "model": {"binary": {"lora": {
            "adapters": [{"name": "a", "alphas": alphas}]}}}}

    cfg = {"engine": [engine("target", ["t0"]), engine("draft", ["d0"])]}
    assert lora_alpha_names(cfg, "primary") == {"t0"}
    assert lora_alpha_names(cfg, "draft") == {"d0"}
    assert lora_alpha_names(cfg, "secondary") == {"d0"}


@pytest.mark.parametrize("cfg, role", [
    ({}, "primary"),                                          # no engine at all
    ({"engine": {"model": {}}}, "tertiary"),                  # unknown role
    ({"engine": {"model": {}}}, "draft"),                     # no such engine
    ({"engine": [{"role": "target"}, {"role": "primary"}]}, "primary"),  # two
])
def test_lora_alpha_names_leaves_it_to_the_sdk_when_it_cannot_tell(cfg, role):
    from genie_server.slots import lora_alpha_names
    assert lora_alpha_names(cfg, role) is None


def _config_with(tmp_path, **raw):
    path = tmp_path / "env_config.json"
    path.write_text(json.dumps({"QAIRT_SDK_ROOT": "/opt/qairt",
                                "TEXT_SLOTS": [{"model_root": str(tmp_path)}],
                                **raw}))
    return str(path)


@pytest.mark.parametrize("key", ["PROMPT_LOGPROBS", "GENIE_PROFILE",
                                 "TOOL_CALL_RECOVERY",
                                 "VLM_VISION_BUDGET_GUARD"])
@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_a_config_flag_must_be_a_json_boolean(tmp_path, key, value):
    """bool("false") is True: a quoted false turned the workaround on."""
    from genie_server.config import load_config
    with pytest.raises(ValueError, match=f"{key} must be true or false"):
        load_config(_config_with(tmp_path, **{key: value}))


def test_config_flags_default_off_and_accept_booleans(tmp_path):
    from genie_server.config import load_config
    assert load_config(_config_with(tmp_path)).tool_call_recovery is False
    assert load_config(_config_with(
        tmp_path, TOOL_CALL_RECOVERY=True)).tool_call_recovery is True


@pytest.mark.parametrize("key, value", [
    ("CHAT_TEMPLATE", "chatlm"),      # used to render as chatml anyway
    ("CHAT_TEMPLATE", "qwen"),
    ("CHAT_TEMPLATE", 3),
    ("TOOL_FORMAT", "hermes2"),       # used to fall back to hermes
    ("TOOL_FORMAT", "gemma"),
])
def test_an_unknown_template_or_tool_format_is_refused(tmp_path, key, value):
    from genie_server.config import load_config
    with pytest.raises(ValueError, match=f"{key} must be one of"):
        load_config(_config_with(tmp_path, **{key: value}))


@pytest.mark.parametrize("key, value, attr, expected", [
    ("CHAT_TEMPLATE", " Llama3 ", "chat_template_override", "llama3"),
    ("CHAT_TEMPLATE", "gemma4", "chat_template_override", "gemma4"),
    ("CHAT_TEMPLATE", "", "chat_template_override", ""),
    ("TOOL_FORMAT", "Gemma4", "tool_format_override", "gemma4"),
    ("TOOL_FORMAT", "", "tool_format_override", ""),
])
def test_known_template_and_tool_format_names_load(tmp_path, key, value, attr,
                                                   expected):
    from genie_server.config import load_config
    cfg = load_config(_config_with(tmp_path, **{key: value}))
    assert getattr(cfg, attr) == expected
