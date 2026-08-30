"""The tool dialects: gemma4's own tokens, and that Hermes is untouched.

gemma4 does not speak Hermes. It declares tools with <|tool> ... <tool|>,
calls them with <|tool_call>call:NAME{...}<tool_call|>, and writes values in
its own notation rather than JSON — strings wrapped in <|"|>, keys in dictsort
order. A model answering in its own format was previously parsed by the Hermes
reader, which sees no tags and hands the caller prose.
"""

import json

import pytest

from genie_server import tool_formats
from genie_server.tool_formats import Gemma4ToolFormat as G, HermesToolFormat as H

WEATHER = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get the weather",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string",
                                           "description": "City name"},
                                  "days": {"type": "integer"}},
                   "required": ["city"]}}}]


# ------------------------------------------------------------------ selection

def test_the_dialect_follows_the_chat_template():
    """Derived rather than matched again, so the two cannot disagree: a
    bundle rendering gemma4 turns also speaks gemma4 tool tokens."""
    assert tool_formats.detect("gemma4") == "gemma4"
    for other in ("chatml", "gemma", "llama3", "llama2"):
        assert tool_formats.detect(other) == "hermes"


def test_an_unknown_dialect_name_falls_back_to_hermes():
    assert tool_formats.get("no-such-format") is H
    assert tool_formats.get("") is H
    assert tool_formats.get("gemma4") is G


# ------------------------------------------------------------- gemma4 output

def test_gemma4_renders_declarations_in_its_own_notation():
    block = G.render_tools_block(WEATHER)
    assert block.startswith("\n<|tool>declaration:get_weather{")
    assert block.rstrip().endswith("<tool|>")
    # Strings are delimited, types are Gemini-flavoured, keys are sorted.
    assert 'description:<|"|>Get the weather<|"|>' in block
    assert 'type:<|"|>STRING<|"|>' in block and 'type:<|"|>OBJECT<|"|>' in block
    assert block.index("city:") < block.index("days:")
    assert "{" in block and '"name"' not in block   # not JSON


def test_gemma4_declares_nothing_for_an_empty_tool_list():
    assert G.render_tools_block([]) == ""


def test_gemma4_parses_a_call_into_the_openai_shape():
    text = ('Sure.<|tool_call>call:get_weather'
            '{city:<|"|>Tokyo<|"|>,days:3}<tool_call|>')
    content, calls = G.parse_tool_calls(text)

    assert content == "Sure."
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Tokyo", "days": 3}


def test_gemma4_parses_nested_and_typed_values():
    text = ('<|tool_call>call:f{s:<|"|>a,b{c}<|"|>,n:1.5,t:true,z:null,'
            'l:[1,2],d:{k:<|"|>v<|"|>}}<tool_call|>')
    _content, calls = G.parse_tool_calls(text)
    args = json.loads(calls[0]["function"]["arguments"])
    # The comma and braces inside the delimited string must not split it.
    assert args == {"s": "a,b{c}", "n": 1.5, "t": True, "z": None,
                    "l": [1, 2], "d": {"k": "v"}}


def test_an_unterminated_call_stays_text_by_default():
    """Observed on the board: gemma4-e2b writes the call and emits EOS without
    <tool_call|>. It is recoverable — the body is complete — but a model that
    will not close its own call has a defect of the same kind as one that
    mangles the opening marker, and BFCL counts both as failures. Repairing it
    by default would make the bundle score better here than through
    /v1/completions, which has no recovery to apply."""
    text = '<|tool_call>call:get_weather{city:<|"|>Tokyo<|"|>}'
    content, calls = G.parse_tool_calls(text)
    assert calls == []
    assert content == text


def test_an_unterminated_call_is_recovered_with_recovery_on():
    """TOOL_CALL_RECOVERY is the switch for exactly this: repair the call so
    the application works despite the bundle, knowing what it conceals."""
    text = '<|tool_call>call:get_weather{city:<|"|>Tokyo<|"|>}'
    content, calls = G.parse_tool_calls(text, {"get_weather"})
    assert content == ""
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Tokyo"}


def test_a_terminated_call_needs_no_recovery():
    """The well-formed case is not gated — nothing is being repaired."""
    text = '<|tool_call>call:get_weather{city:<|"|>Tokyo<|"|>}<tool_call|>'
    _content, calls = G.parse_tool_calls(text)
    assert calls[0]["function"]["name"] == "get_weather"


def test_gemma4_leaves_a_call_cut_off_mid_body_as_text():
    """The other half of that rule: a body that never closes is a generation
    truncated by max_tokens, and guessing at it would invent arguments."""
    text = 'ok <|tool_call>call:get_weather{city:<|"|>Tok'
    content, calls = G.parse_tool_calls(text)
    assert calls == [] and content == text.strip()


def test_gemma4_parses_two_calls_in_one_reply():
    text = ('<|tool_call>call:a{x:1}<tool_call|> and '
            '<|tool_call>call:b{y:<|"|>z<|"|>}<tool_call|>')
    content, calls = G.parse_tool_calls(text)
    assert [c["function"]["name"] for c in calls] == ["a", "b"]
    assert content == "and"


def test_gemma4_leaves_a_reply_with_no_call_alone():
    content, calls = G.parse_tool_calls("The weather in Tokyo is fine.")
    assert calls == [] and content == "The weather in Tokyo is fine."


def test_gemma4_round_trips_a_call_back_into_the_prompt():
    """A client sends the assistant turn back for the follow-up; it has to go
    out the way the model wrote it, not as Hermes JSON."""
    _content, calls = G.parse_tool_calls(
        '<|tool_call>call:get_weather{city:<|"|>Tokyo<|"|>}<tool_call|>')
    rendered = G.format_tool_call_for_prompt(calls[0])
    assert rendered == '<|tool_call>call:get_weather{city:<|"|>Tokyo<|"|>}<tool_call|>'
    # ...and reparsing it gives the same arguments back.
    assert json.loads(G.parse_tool_calls(rendered)[1][0]["function"]["arguments"]) \
        == {"city": "Tokyo"}


# ---------------------------------------------------------------- streaming

def test_gemma4_streaming_holds_everything_until_the_end():
    """The buffering filter: no text mid-stream, correct content and calls at
    finalize(). Worse than the Hermes filter, and the same result."""
    f = G.stream_filter()
    text = 'Sure.<|tool_call>call:get_weather{city:<|"|>Tokyo<|"|>}<tool_call|>'
    emitted = "".join(f.feed(text[i:i + 3]) for i in range(0, len(text), 3))

    assert emitted == ""
    content, calls = f.finalize()
    assert content == "Sure."
    assert calls[0]["function"]["name"] == "get_weather"


def test_gemma4_streaming_never_leaks_the_markers():
    """The failure this replaces: raw tokens arriving as assistant content."""
    f = G.stream_filter()
    text = 'x<|tool_call>call:f{a:1}<tool_call|>'
    emitted = "".join(f.feed(c) for c in text)
    content, _calls = f.finalize()
    assert "<|tool_call>" not in emitted + content


# ------------------------------------------------------------------- hermes

def test_hermes_still_goes_through_tools_py():
    """The seam must not have changed the dialect every other test covers."""
    text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n</tool_call>'
    content, calls = H.parse_tool_calls(text)
    assert content == "" and calls[0]["function"]["name"] == "get_weather"
    assert "<tools>" in H.render_tools_block(WEATHER)
    assert H.stream_filter(None).__class__.__name__ == "ToolCallStreamFilter"
