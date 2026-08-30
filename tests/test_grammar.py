"""Grammar-constrained decoding (dialog.context.grammar / XGrammar backend).

Two layers are covered here:

  * `slots.load_dialog_config` — the only server-side code that touches the
    grammar block: it resolves `file` against the model directory and warns
    about a backend the SDK cannot honour.
  * the JSON the server hands to `GenieDialog_create`, checked against a
    transcription of the SDK's own validator (`Genie/src/Context.cpp:69-105`).
    The SDK rejects the whole config on an unknown key or a bad enum value, so
    an offline check here catches config mistakes that would otherwise only
    surface as a failed model load on the board.

Per-request constraints do not exist: grammar is read once, inside
GenieDialog_create (docs/MANUAL.md, "Grammar-Constrained Decoding"). The
OpenAI-facing counterpart (`response_format`) is therefore *not* implemented,
which the last test pins down.
"""

import json
import logging
from pathlib import Path

import pytest

from genie_server.slots import load_dialog_config


# ------------------------------------------------------------------ helpers

SCHEMA_TEXT = json.dumps({
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
})


def write_model(tmp_path: Path, grammar: dict | None, *,
                schema_name: str = "grammar_schema.txt",
                write_schema: bool = True) -> Path:
    """A minimal model directory: genie_config.json plus (optionally) the
    grammar definition file it points at."""
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    context: dict = {"version": 1, "size": 4096, "n-vocab": 151936}
    if grammar is not None:
        context["grammar"] = grammar
    config = {"dialog": {"version": 1, "type": "basic", "context": context}}
    (model_dir / "genie_config.json").write_text(json.dumps(config))
    if write_schema:
        (model_dir / schema_name).write_text(SCHEMA_TEXT)
    return model_dir


def load(model_dir: Path, tmp_path: Path):
    """load_dialog_config with the arguments the tests never vary."""
    return load_dialog_config(model_dir, None, "cdsp0", tmp_path / "htpcache")


# The SDK's grammar validator, transcribed from
# examples/Genie/Genie/src/Context.cpp:69-105. Every rejection there is an
# exception out of GenieDialogConfig_createFromJson / GenieDialog_create.
GRAMMAR_KEYS = {"backend", "type", "file", "tokenizer-json"}
GRAMMAR_TYPES = {"json-schema", "regex", "ebnf"}


def sdk_reject_reason(config_json: bytes) -> str | None:
    """None if the SDK would accept this config's grammar block."""
    grammar = json.loads(config_json)["dialog"].get("context", {}).get("grammar")
    if grammar is None:
        return None
    if not isinstance(grammar, dict):
        return "context.grammar must be an object"
    for key, value in grammar.items():
        if key not in GRAMMAR_KEYS:
            return f"Unknown context.grammar config key: {key}"
        if not isinstance(value, str):
            return f"context.grammar.{key} must be a string"
    backend = grammar.get("backend", "")
    if backend not in ("xgrammar", ""):
        return f"context.grammar.backend: unsupported value '{backend}'"
    gtype = grammar.get("type", "json-schema")
    if gtype not in GRAMMAR_TYPES:
        return f"context.grammar.type: unsupported value '{gtype}'"
    return None


# ------------------------------------------------- load_dialog_config: paths

def test_grammar_file_resolved_against_model_dir(tmp_path):
    model_dir = write_model(tmp_path, {
        "backend": "xgrammar", "type": "json-schema",
        "file": "grammar_schema.txt"})

    config_json, dcfg = load(model_dir, tmp_path)

    resolved = dcfg["context"]["grammar"]["file"]
    assert resolved == str(model_dir / "grammar_schema.txt")
    assert Path(resolved).is_absolute()
    # and it survives into the bytes actually handed to the SDK
    assert json.loads(config_json)["dialog"]["context"]["grammar"]["file"] == resolved


def test_absolute_grammar_file_passes_through(tmp_path):
    outside = tmp_path / "shared_schema.txt"
    outside.write_text(SCHEMA_TEXT)
    model_dir = write_model(tmp_path, {
        "backend": "xgrammar", "file": str(outside)}, write_schema=False)

    _, dcfg = load(model_dir, tmp_path)

    assert dcfg["context"]["grammar"]["file"] == str(outside)


def test_missing_grammar_file_raises(tmp_path):
    model_dir = write_model(tmp_path, {
        "backend": "xgrammar", "file": "does_not_exist.txt"}, write_schema=False)

    # A missing asset must fail the load, not reach GenieDialog_create (where
    # it becomes "Failed to open grammar file", qualla/dialog.cpp:361).
    with pytest.raises(FileNotFoundError, match="does_not_exist.txt"):
        load(model_dir, tmp_path)


def test_no_grammar_block_is_untouched(tmp_path):
    model_dir = write_model(tmp_path, None)

    config_json, dcfg = load(model_dir, tmp_path)

    assert "grammar" not in dcfg["context"]
    assert sdk_reject_reason(config_json) is None


def test_grammar_block_without_file_is_left_alone(tmp_path):
    # The SDK turns this into "Grammar backend configured but grammar.file is
    # empty" (qualla/dialog.cpp:354-355). The server must not crash on it.
    model_dir = write_model(tmp_path, {"backend": "xgrammar"})

    _, dcfg = load(model_dir, tmp_path)

    assert dcfg["context"]["grammar"] == {"backend": "xgrammar"}


# --------------------------------------------- load_dialog_config: backend

def test_unsupported_backend_warns_but_still_loads(tmp_path, caplog):
    model_dir = write_model(tmp_path, {
        "backend": "llguidance", "file": "grammar_schema.txt"})

    with caplog.at_level(logging.WARNING, logger="genie_server.slots"):
        config_json, dcfg = load(model_dir, tmp_path)

    assert "llguidance" in caplog.text and "xgrammar" in caplog.text
    # The server only warns; the SDK is what actually refuses the config
    # (Context.cpp:79, GENIE_STATUS_ERROR_JSON_VALUE), so the load itself
    # succeeds and the model fails later, at create time.
    assert dcfg["context"]["grammar"]["backend"] == "llguidance"
    assert sdk_reject_reason(config_json) == \
        "context.grammar.backend: unsupported value 'llguidance'"


def test_xgrammar_backend_does_not_warn(tmp_path, caplog):
    model_dir = write_model(tmp_path, {
        "backend": "xgrammar", "file": "grammar_schema.txt"})

    with caplog.at_level(logging.WARNING, logger="genie_server.slots"):
        load(model_dir, tmp_path)

    assert "grammar" not in caplog.text


# ------------------------------------------------------ SDK schema contract

@pytest.mark.parametrize("gtype", sorted(GRAMMAR_TYPES))
def test_every_grammar_type_is_emitted_verbatim(tmp_path, gtype):
    model_dir = write_model(tmp_path, {
        "backend": "xgrammar", "type": gtype, "file": "grammar_schema.txt"})

    config_json, dcfg = load(model_dir, tmp_path)

    assert dcfg["context"]["grammar"]["type"] == gtype
    assert sdk_reject_reason(config_json) is None


@pytest.mark.parametrize("grammar,expected", [
    ({"backend": "xgrammar", "type": "json_schema", "file": "grammar_schema.txt"},
     "context.grammar.type: unsupported value 'json_schema'"),
    ({"backend": "xgrammar", "schema": "x", "file": "grammar_schema.txt"},
     "Unknown context.grammar config key: schema"),
])
def test_config_mistakes_the_sdk_would_reject(tmp_path, grammar, expected):
    # The server passes these through untouched, so the board would fail the
    # model load. Named here so the failure mode is recognisable.
    model_dir = write_model(tmp_path, grammar)

    config_json, _ = load(model_dir, tmp_path)

    assert sdk_reject_reason(config_json) == expected


def test_shipped_example_config_is_sdk_valid():
    """examples/grammar/ must stay loadable as-is."""
    example = Path(__file__).resolve().parent.parent / "examples" / "grammar"
    config = json.loads((example / "genie_config.json").read_text())
    grammar = config["dialog"]["context"]["grammar"]

    assert sdk_reject_reason(json.dumps(config).encode()) is None
    assert grammar["backend"] == "xgrammar"
    # the file it names must exist and hold a parseable JSON Schema
    schema_file = example / grammar["file"]
    assert schema_file.exists()
    assert json.loads(schema_file.read_text())["type"] == "object"


# --------------------------------------------------------- per-request gate

def test_response_format_is_accepted_and_ignored(client):
    """Grammar is fixed per slot, so OpenAI's per-request `response_format`
    cannot be honoured. The server currently accepts and ignores it — the
    same silent fallthrough that `tool_choice` used to have before it was
    turned into a 400."""
    r = client.post("/v1/chat/completions", json={
        "model": "genie-local",
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {"type": "json_object"},
    })

    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    assert content == "Hello world from Genie!"   # unconstrained, not JSON


# ------------------------------------------- integration checker (offline)

def load_runner():
    """The G0x hardware tests live in the integration runner; its output
    checks are pure functions, so they are unit-tested here rather than only
    on a board."""
    import importlib.util
    path = (Path(__file__).resolve().parent / "integration"
            / "run_integration_tests.py")
    spec = importlib.util.spec_from_file_location("integration_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_json_schema_checker_accepts_conforming_output():
    runner = load_runner()
    obj = runner.check_json_schema_output('{"answer": "yes", "confidence": 0.9}')
    assert obj == {"answer": "yes", "confidence": 0.9}


@pytest.mark.parametrize("text,reason", [
    ("Hello world from Genie!", "not JSON"),
    ('["answer"]', "not a JSON object"),
    ('{"answer": "yes"}', "required"),
    ('{"answer": "yes", "confidence": 0.9, "extra": 1}', "additionalProperties"),
    ('{"answer": 1, "confidence": 0.9}', "not string"),
    ('{"answer": "yes", "confidence": "high"}', "not a number"),
])
def test_json_schema_checker_rejects_unconstrained_output(text, reason):
    # Negative control: an unconstrained model (or a grammar that silently did
    # nothing) must not slip past G01.
    runner = load_runner()
    with pytest.raises(runner.CheckFailure, match=reason):
        runner.check_json_schema_output(text)


def test_default_regex_matches_only_the_intended_shape():
    import re
    runner = load_runner()
    assert re.fullmatch(runner.REGEX_DEFAULT,
                        '{"sentiment": "positive", "score": 0.87}')
    for bad in ('{"sentiment": "happy", "score": 0.87}',
                '{"sentiment": "positive", "score": 1.87}',
                "positive"):
        assert not re.fullmatch(runner.REGEX_DEFAULT, bad)
