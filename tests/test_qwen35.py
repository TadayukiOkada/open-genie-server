"""Qwen3.5: chat template, tool dialect and the linear-attention state.

The golden prompts in data/qwen3_5_chat_template_golden.json were rendered by
transformers' apply_chat_template from Qwen/Qwen3.5-2B's own chat_template.jinja
(revision 15852e8c); the renderer here has to reproduce them byte for byte.
"""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from genie_server import templates, tool_formats
from genie_server.app import create_app

DATA = json.loads((Path(__file__).parent / "data" / "qwen3_5_chat_template_golden.json").read_text())
GOLDEN, TOOLS = DATA["golden"], DATA["tools"]
FMT = tool_formats.get("qwen3_xml")


def _render(messages, thinking=None, tools=None):
    prepared = templates.prepare_messages(messages, True if thinking is None else thinking,
                                          tools, FMT, template="qwen3_5")
    return templates.render_chat_prompt(prepared, "qwen3_5", FMT, enable_thinking=thinking)


# ------------------------------------------------------------------ detection

@pytest.mark.parametrize("name", ["qwen3_5-2b-genie-lpbq4w8a16-52_v73_cl8192", "Qwen3.5-2B", "qwen35_2b"])
def test_qwen3_5_bundles_get_their_own_template_and_dialect(name):
    template = templates.detect_template(name)
    assert template == "qwen3_5"
    assert tool_formats.detect(template) == "qwen3_xml"


def test_plain_qwen3_stays_chatml():
    assert templates.detect_template("qwen3_4b_instruct_2507") == "chatml"


def test_thinking_defaults_follow_each_template():
    assert templates.default_thinking("qwen3_5") is False
    assert templates.default_thinking("chatml") is True
    assert templates.generation_prefix("qwen3_5", True) == "<think>\n"
    assert templates.generation_prefix("qwen3_5", False) == ""
    assert templates.generation_prefix("chatml", True) == ""


# ------------------------------------------------------------------ rendering

def test_default_is_the_empty_think_block():
    assert _render([{"role": "user", "content": "Hi"}]) == GOLDEN["plain"]


def test_thinking_opens_the_reasoning():
    assert _render([{"role": "user", "content": "Hi"}], thinking=True) == GOLDEN["thinking"]


def test_no_think_soft_switch_is_not_used():
    prompt = _render([{"role": "system", "content": "S"}, {"role": "user", "content": "Hi"}], thinking=False)
    assert "/no_think" not in prompt


def test_tools_block_comes_before_the_system_text():
    messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "weather?"}]
    assert _render(messages, tools=TOOLS) == GOLDEN["tools_system"]


def test_tool_round_trip_groups_results_in_one_user_turn():
    call = {"id": "c1", "type": "function",
            "function": {"name": "get_weather", "arguments": json.dumps({"city": "Tokyo", "days": 3})}}
    messages = [{"role": "user", "content": "weather?"},
                {"role": "assistant", "content": "Let me check.", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
                {"role": "tool", "tool_call_id": "c1", "content": "rain"}]
    assert _render(messages, thinking=True) == GOLDEN["roundtrip"]


def test_reasoning_before_the_last_query_is_dropped():
    messages = [{"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "<think>\nr1\n</think>\n\nA1"},
                {"role": "user", "content": "Q2"}]
    assert _render(messages) == GOLDEN["history"]


@pytest.mark.parametrize("messages", [
    [{"role": "user", "content": "q"}, {"role": "system", "content": "late"}, {"role": "user", "content": "q2"}],
    [{"role": "system", "content": "a"}, {"role": "system", "content": "b"}, {"role": "user", "content": "q"}],
])
def test_a_system_message_after_the_first_is_refused(messages):
    with pytest.raises(templates.UnrenderableMessageError):
        _render(messages)


def test_split_keeps_the_system_turn_as_prefix():
    messages = templates.prepare_messages(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "Hi"}], False, None, FMT,
        template="qwen3_5")
    prefix, rest, cacheable = templates.split_prompt_for_prefix_cache(
        messages, "qwen3_5", FMT, enable_thinking=False)
    assert cacheable and prefix == "<|im_start|>system\nS<|im_end|>\n"
    assert prefix + rest == templates.render_chat_prompt(messages, "qwen3_5", FMT, enable_thinking=False)


# ------------------------------------------------------------------ tool dialect

_CALL = ("Sure.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\n007\n</parameter>\n"
         "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>")


def test_calls_are_typed_by_the_request_schema():
    content, calls = FMT.bind(TOOLS).parse_tool_calls(_CALL)
    assert content == "Sure."
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "007", "days": 3}


def test_without_schemas_values_stay_strings():
    _, calls = FMT.parse_tool_calls(_CALL)
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "007", "days": "3"}


def test_an_unclosed_call_stays_in_the_content():
    text = "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n"
    content, calls = FMT.parse_tool_calls(text)
    assert calls == [] and content == text.strip()


def test_object_arguments_round_trip_as_json():
    call = {"function": {"name": "f", "arguments": json.dumps({"opts": {"a": 1}, "flag": True})}}
    text = FMT.format_tool_call_for_prompt(call)
    assert '<parameter=opts>\n{"a": 1}\n</parameter>' in text and "<parameter=flag>\nTrue\n</parameter>" in text


# ------------------------------------------------------------------ server

def _qwen35_client(state, linear_attention=True):
    slot = state.manager.slots[0]
    slot.chat_template = "qwen3_5"
    slot.tool_format = FMT
    if linear_attention:
        slot.dialog_cfg = {"context": {"size": 4096},
                           "engine": {"model": {"linear-attention": True}}}
    return TestClient(create_app(state))


def test_linear_attention_turns_the_prefix_cache_off(state):
    client = _qwen35_client(state)
    slot = state.manager.slots[0]
    assert slot.saves_state is False
    for _ in range(2):
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "Hi"}]})
        assert r.status_code == 200
    assert state.lib.saved_states == {}
    assert all(q.startswith("<|im_start|>system\nS") for q in state.lib.queries)


def test_warmup_is_refused_on_a_linear_attention_slot(state):
    client = _qwen35_client(state)
    r = client.post("/v1/prefix/warmup", json={"system_prompt": "S"})
    assert r.status_code == 422 and "linear-attention" in r.text


def test_a_bundle_without_linear_attention_keeps_saving(state):
    _qwen35_client(state, linear_attention=False)
    assert state.manager.slots[0].saves_state is True


def test_thinking_reply_carries_the_opening_tag(state):
    client = _qwen35_client(state)
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Hi"}], "enable_thinking": True})
    assert r.json()["choices"][0]["message"]["content"].startswith("<think>\n")
    assert state.lib.queries[-1].endswith("<|im_start|>assistant\n<think>\n")


def test_default_reply_is_plain(state):
    client = _qwen35_client(state)
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hi"}]})
    assert not r.json()["choices"][0]["message"]["content"].startswith("<think>")
    assert state.lib.queries[-1].endswith("<think>\n\n</think>\n\n")


def test_streamed_thinking_reply_starts_with_the_tag(state):
    client = _qwen35_client(state)
    with client.stream("POST", "/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hi"}], "stream": True,
            "chat_template_kwargs": {"enable_thinking": True}}) as r:
        chunks = [json.loads(line[6:]) for line in r.iter_lines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c.get("choices"))
    assert text.startswith("<think>\n")


def test_tool_calls_come_back_typed(state):
    client = _qwen35_client(state)
    state.lib.canned_response = _CALL
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "weather?"}], "tools": TOOLS})
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {"city": "007", "days": 3}
    prompt = state.lib.queries[-1]
    assert prompt.startswith("<|im_start|>system\n# Tools\n\nYou have access to the following functions:")
