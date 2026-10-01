"""Per-request grammar: response_format / structured_outputs ->
GenieDialog_setGrammar (QAIRT 2.51.0+).

Three layers:

  * grammar.parse_request_grammar -- what each request spelling becomes, and
    what is refused rather than dropped.
  * grammar.probe_support -- the per-model decision whether setGrammar can
    work, which the exported symbol alone does not settle.
  * the HTTP path against FakeGenieLib -- which grammar the dialog holds
    for each request, that it is not recompiled needlessly, and that a
    request without one gets the bundle's (or none) back.
"""

import json

import pytest

from genie_server import capi, grammar
from genie_server.grammar import (GrammarRequestError, GrammarSupport,
                                  RequestGrammar, parse_request_grammar)

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}},
          "required": ["a"]}
MSGS = [{"role": "user", "content": "hi"}]


# ------------------------------------------------------------------ parsing

def test_no_constraint():
    assert parse_request_grammar({}) is None
    assert parse_request_grammar({"response_format": {"type": "text"}}) is None
    assert parse_request_grammar({"structured_outputs": {}}) is None


def test_response_format_json_schema():
    g = parse_request_grammar({"response_format": {
        "type": "json_schema",
        "json_schema": {"name": "x", "schema": SCHEMA, "strict": True}}})
    assert g.kind == "json-schema"
    assert json.loads(g.definition) == SCHEMA
    assert g.param == "response_format"


def test_response_format_json_object_is_an_object_schema():
    g = parse_request_grammar({"response_format": {"type": "json_object"}})
    assert (g.kind, json.loads(g.definition)) == ("json-schema", {"type": "object"})


@pytest.mark.parametrize("rf", [
    {"type": "json_schema"},
    {"type": "json_schema", "json_schema": {"name": "x"}},
    {"type": "json_schema", "json_schema": {"schema": [1, 2]}},
    {"type": "structural_tag"},
    {"type": "yaml"},
    "json_object",
])
def test_response_format_refused(rf):
    with pytest.raises(GrammarRequestError):
        parse_request_grammar({"response_format": rf})


@pytest.mark.parametrize("so, kind, definition", [
    ({"json": SCHEMA}, "json-schema", json.dumps(SCHEMA, sort_keys=True)),
    ({"json": json.dumps(SCHEMA)}, "json-schema", json.dumps(SCHEMA, sort_keys=True)),
    ({"regex": r"\d{3}"}, "regex", r"\d{3}"),
    ({"grammar": 'root ::= "yes" | "no"'}, "ebnf", 'root ::= "yes" | "no"'),
    ({"json_object": True}, "json-schema", '{"type": "object"}'),
    ({"choice": ["yes", "no"], "disable_fallback": True,
      "disable_any_whitespace": False}, "ebnf", 'root ::= "yes" | "no"\n'),
])
def test_structured_outputs(so, kind, definition):
    g = parse_request_grammar({"structured_outputs": so})
    assert (g.kind, g.definition) == (kind, definition)
    assert g.param.startswith("structured_outputs.")


@pytest.mark.parametrize("so", [
    {"json": SCHEMA, "regex": "a"},          # two constraints
    {"structural_tag": "{}"},
    {"regex": ""},
    {"choice": []},
    {"choice": ["a", 1]},
    {"json": "{not json"},
    {"json_object": "yes"},
    {"disable_any_whitespace": True, "json": SCHEMA},
    {"whitespace_pattern": " ", "json": SCHEMA},
    {"jsn": SCHEMA},                          # a typo is not ignored
])
def test_structured_outputs_refused(so):
    with pytest.raises(GrammarRequestError):
        parse_request_grammar({"structured_outputs": so})


def test_both_spellings_at_once_refused():
    with pytest.raises(GrammarRequestError):
        parse_request_grammar({"response_format": {"type": "json_object"},
                               "structured_outputs": {"regex": "a"}})


def test_vllm_removed_guided_fields_refused():
    with pytest.raises(GrammarRequestError, match="v0.12.0"):
        parse_request_grammar({"guided_json": SCHEMA})


def test_choice_escaping():
    """Quotes, backslashes, newlines and non-ASCII survive as EBNF literals
    in the escapes XGrammar reads."""
    ebnf = grammar.choice_to_ebnf(['say "hi"', "a\\b", "x\ny", "東京"])
    assert ebnf == 'root ::= "say \\"hi\\"" | "a\\\\b" | "x\\ny" | "東京"\n'


def test_key_tells_grammars_apart():
    a = RequestGrammar("regex", "a", "p")
    assert a.key == RequestGrammar("regex", "a", "q").key
    assert a.key != RequestGrammar("ebnf", "a", "p").key
    assert a.key != RequestGrammar("regex", "b", "p").key


# ------------------------------------------------------------------ probing

class _Handle:
    value = 7


def test_probe_without_the_symbol(fake_lib):
    fake_lib.has_set_grammar = False
    s = grammar.probe_support(fake_lib, _Handle(), {}, "s")
    assert not s.supported and "2.51.0" in s.reason
    assert fake_lib.grammar_calls == []


def test_probe_asks_the_dialog(fake_lib):
    assert grammar.probe_support(fake_lib, _Handle(), {}, "s").supported
    assert fake_lib.grammar_calls == [(7, None, None)]


def test_probe_refused_by_a_library_without_the_backend(fake_lib):
    fake_lib.set_grammar_status = -1
    s = grammar.probe_support(fake_lib, _Handle(), {"type": "ssd-q1"}, "s")
    assert not s.supported and "'ssd-q1'" in s.reason


def test_probe_leaves_a_bundle_grammar_alone(fake_lib):
    """The disabling call would throw the bundle's grammar away."""
    cfg = {"context": {"grammar": {"backend": "xgrammar", "file": "/m/s.json"}}}
    s = grammar.probe_support(fake_lib, _Handle(), cfg, "s")
    assert s.supported and s.bundle == ("json-schema", "/m/s.json")
    assert fake_lib.grammar_calls == []


@pytest.fixture
def fake_lib():
    from fake_genie import FakeGenieLib
    return FakeGenieLib()


# ------------------------------------------------------------------ HTTP

@pytest.fixture
def slot(state):
    s = state.manager.slots[0]
    s.grammar_support = GrammarSupport(True)
    s.active_grammar = None
    return s


def chat(client, **extra):
    return client.post("/v1/chat/completions",
                       json={"model": "genie-local", "messages": MSGS, **extra})


def test_request_grammar_reaches_the_dialog(client, state, slot):
    r = chat(client, response_format={
        "type": "json_schema", "json_schema": {"schema": SCHEMA}})
    assert r.status_code == 200, r.text
    (handle, kind, content), = state.lib.grammar_calls
    assert (handle, kind) == (slot.handle.value, "json-schema")
    assert json.loads(content) == SCHEMA


def test_same_grammar_is_not_recompiled(client, state, slot):
    for _ in range(3):
        assert chat(client, structured_outputs={"regex": "[a-z]+"}).status_code == 200
    assert len(state.lib.grammar_calls) == 1


def test_next_request_without_one_is_unconstrained(client, state, slot):
    chat(client, structured_outputs={"choice": ["a", "b"]})
    chat(client)
    chat(client)
    kinds = [k for _, k, _ in state.lib.grammar_calls]
    assert kinds == ["ebnf", None]           # cleared once, then left alone
    assert slot.handle.value not in state.lib.grammar


def test_bundle_grammar_comes_back(client, state, slot):
    slot.grammar_support = GrammarSupport(True, "", ("regex", "/m/bundle.txt"))
    slot.active_grammar = grammar.BUNDLE
    state.lib.set_grammar = _recording_set_grammar(state.lib)
    chat(client)                             # bundle already in force
    chat(client, structured_outputs={"regex": "x"})
    chat(client)
    assert state.lib.calls == [("regex", "<temp>"), ("regex", "/m/bundle.txt")]


def _recording_set_grammar(lib):
    """Records paths instead of reading them (the bundle's does not exist)."""
    lib.calls = []

    def set_grammar(handle, kind, path):
        lib.calls.append((kind, path if path == "/m/bundle.txt" else "<temp>"))
        return 0
    return set_grammar


def test_temp_file_is_removed(client, state, slot, tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", None)
    chat(client, structured_outputs={"regex": "x"})
    assert not list(tmp_path.glob("ogs-grammar-*"))


def test_unsupported_slot_refuses(client, state):
    slot = state.manager.slots[0]
    slot.grammar_support = GrammarSupport(False, "no backend here")
    r = chat(client, response_format={"type": "json_object"})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "grammar_not_supported"
    assert err["param"] == "response_format"
    assert "no backend here" in err["message"]
    assert state.lib.queries == []


def test_unsupported_slot_still_serves_plain_requests(client, state):
    state.manager.slots[0].grammar_support = GrammarSupport(False, "x")
    assert chat(client).status_code == 200
    assert state.lib.grammar_calls == []


def test_sdk_refusal_is_a_400_and_the_query_does_not_run(client, state, slot):
    state.lib.set_grammar_status = -1
    r = chat(client, structured_outputs={"grammar": "root ::= oops"})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "structured_outputs.grammar"
    assert state.lib.queries == []
    assert slot.active_grammar == grammar.UNKNOWN


def test_after_a_refusal_the_dialog_is_set_again(client, state, slot):
    state.lib.set_grammar_status = -1
    chat(client, structured_outputs={"regex": "x"})
    state.lib.set_grammar_status = 0
    assert chat(client).status_code == 200
    assert state.lib.grammar_calls[-1][1:] == (None, None)


def test_sdk_refusal_mid_stream(client, state, slot):
    state.lib.set_grammar_status = -1
    with client.stream("POST", "/v1/chat/completions", json={
            "model": "genie-local", "messages": MSGS, "stream": True,
            "structured_outputs": {"regex": "x"}}) as r:
        body = r.read().decode()
    events = [json.loads(line[6:]) for line in body.splitlines()
              if line.startswith("data: {")]
    assert events[-1]["error"]["type"] == "invalid_request_error"
    assert events[-1]["error"]["param"] == "structured_outputs.regex"


def test_malformed_constraint_never_reaches_the_slot(client, state, slot):
    r = chat(client, structured_outputs={"structural_tag": "{}"})
    assert r.status_code == 400
    assert state.lib.grammar_calls == [] and state.lib.queries == []


def test_vlm_request_with_grammar_refused(client, state, slot):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}],
        "response_format": {"type": "json_object"}})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "grammar_not_supported"


def test_completions_take_a_grammar(client, state, slot):
    r = client.post("/v1/completions", json={
        "prompt": "x", "max_tokens": 4,
        "structured_outputs": {"choice": ["a"]}})
    assert r.status_code == 200, r.text
    assert state.lib.grammar_calls[0][1:] == ("ebnf", 'root ::= "a"\n')


def test_prompt_scoring_with_grammar_refused(client, state, slot):
    state.prompt_logprobs_enabled = True
    r = client.post("/v1/completions", json={
        "prompt": "x", "max_tokens": 0, "echo": True, "logprobs": 1,
        "structured_outputs": {"regex": "a"}})
    assert r.status_code == 400


def test_status_reports_grammar_support(client, state, slot):
    slot.grammar_support = GrammarSupport(True, "", ("ebnf", "/m/g.ebnf"))
    rep = client.get("/v1/server/status").json()["slots"][0]["grammar"]
    assert rep == {"per_request": True, "reason": None, "bundle": "ebnf"}


# ------------------------------------------------------------------ ctypes

def test_ctypes_binding_passes_null_to_disable():
    import ctypes
    from test_ctypes_layer import StubCDLL
    stub = StubCDLL()
    lib = capi.GenieLib(stub)
    seen = []
    stub.install("GenieDialog_setGrammar",
                 lambda h, k, f: seen.append((_addr(h), k, f)) or 0)
    h = capi.DialogHandle(1234)
    assert lib.set_grammar(h, "regex", "/tmp/g.txt") == 0
    assert lib.set_grammar(h, None, None) == 0
    assert seen == [(1234, b"regex", b"/tmp/g.txt"), (1234, None, None)]
    assert isinstance(stub.GenieDialog_setGrammar, ctypes._CFuncPtr)


def _addr(h):
    return getattr(h, "value", h)


def test_ctypes_library_without_the_symbol():
    from test_ctypes_layer import StubCDLL

    class Older(StubCDLL):
        def __getattr__(self, name):
            if name.startswith("GenieDialog_setGrammar") or \
                    name.startswith("Genie_getApi"):
                raise AttributeError(name)
            return super().__getattr__(name)

    lib = capi.GenieLib(Older())
    assert not lib.has_set_grammar
    assert lib.api_version() is None
    with pytest.raises(RuntimeError):
        lib.set_grammar(capi.DialogHandle(1), None, None)
