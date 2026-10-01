"""Per-request grammar-constrained decoding: request parsing and slot support.

QAIRT 2.51.0 added GenieDialog_setGrammar, which replaces a dialog's grammar
without recreating the dialog. Before it, a grammar could only come from the
bundle's genie_config.json (dialog.context.grammar), read once at
GenieDialog_create. This module turns a request's constraint into what that
call takes, and decides once per loaded model whether the call can work.

Two request spellings are accepted, on both /v1/chat/completions and
/v1/completions:

  * OpenAI's `response_format`: {"type": "json_schema", "json_schema":
    {"schema": ...}} or {"type": "json_object"}. "text" means no constraint.
  * vLLM's `structured_outputs` (an extra_body field): exactly one of
    "json", "regex", "choice", "grammar" or "json_object". vLLM's grammar is
    XGrammar's EBNF, the same dialect the SDK compiles, so it passes through.

Nothing is approximated. A form the SDK has no kind for (`structural_tag`),
an option it cannot honour (`disable_any_whitespace`), vLLM's removed
`guided_*` fields, and a slot whose libGenie cannot set a grammar are all
refused with a 400 rather than answered without the constraint.
"""

import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass

from . import capi

logger = logging.getLogger(__name__)

# The kinds GenieDialog_setGrammar accepts (qualla/dialog.cpp setGrammar).
KINDS = ("json-schema", "regex", "ebnf")

# OpenAI promises a JSON *object* for json_object; XGrammar's "{}" schema
# would also admit a bare string or number.
JSON_OBJECT_SCHEMA = {"type": "object"}

# vLLM's StructuredOutputsParams: the constraint keys (exactly one may be
# set) and the options next to them.
_SO_SUPPORTED = ("json", "regex", "choice", "grammar", "json_object")
_SO_CONSTRAINTS = (*_SO_SUPPORTED, "structural_tag")
# Options the SDK gives no way to pass: refused when set to anything but
# their default, so a client relying on one notices.
_SO_UNSUPPORTED_OPTIONS = ("disable_any_whitespace",
                           "disable_additional_properties",
                           "whitespace_pattern")
# Options that change nothing here: vLLM's fallback between its own backends.
_SO_IGNORED_OPTIONS = ("disable_fallback",)

# vLLM removed these in v0.12.0 (structured_outputs replaced them).
# Accepting them silently would answer without the constraint.
_LEGACY_GUIDED = ("guided_json", "guided_regex", "guided_choice",
                  "guided_grammar", "guided_decoding_backend",
                  "guided_whitespace_pattern")


class GrammarRequestError(ValueError):
    """A request's constraint is malformed or unsupported (HTTP 400)."""

    def __init__(self, message: str, param: str | None,
                 code: str | None = None):
        super().__init__(message)
        self.param = param
        self.code = code


@dataclass(frozen=True)
class RequestGrammar:
    """One request's constraint, in the form GenieDialog_setGrammar takes."""
    kind: str          # one of KINDS
    definition: str    # the file's content: a schema, a regex or an EBNF
    param: str         # the request field it came from, for error messages

    @property
    def key(self) -> str:
        """Identifies the compiled grammar, so a slot can skip recompiling
        the one it already holds."""
        digest = hashlib.sha256(
            f"{self.kind}\0{self.definition}".encode("utf-8")).hexdigest()
        return f"{self.kind}:{digest[:16]}"


def _schema_text(schema, param: str) -> str:
    """A JSON Schema given as an object, or (vLLM) as a JSON string."""
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except json.JSONDecodeError as e:
            raise GrammarRequestError(
                f"{param} is not valid JSON: {e}", param) from None
    if not isinstance(schema, dict):
        raise GrammarRequestError(
            f"{param} must be a JSON Schema object, got {type(schema).__name__}",
            param)
    return json.dumps(schema, ensure_ascii=False, sort_keys=True)


def choice_to_ebnf(choices: list[str]) -> str:
    """An EBNF whose root matches exactly one of the strings.

    json.dumps writes each one as a double-quoted literal whose escapes
    (\\" \\\\ \\n \\t \\uXXXX ...) are all ones XGrammar's EBNF parser reads
    (support/encoding.h ParseNextEscaped); non-ASCII stays as UTF-8, which
    the parser also takes."""
    alternatives = " | ".join(json.dumps(c, ensure_ascii=False) for c in choices)
    return f"root ::= {alternatives}\n"


def _from_response_format(rf) -> RequestGrammar | None:
    param = "response_format"
    if not isinstance(rf, dict):
        raise GrammarRequestError(f"{param} must be an object", param)
    kind = rf.get("type")
    if kind == "text":
        return None
    if kind == "json_object":
        return RequestGrammar("json-schema", json.dumps(JSON_OBJECT_SCHEMA), param)
    if kind == "json_schema":
        spec = rf.get("json_schema")
        if not isinstance(spec, dict) or "schema" not in spec:
            raise GrammarRequestError(
                f'{param}.json_schema must be an object with a "schema"', param)
        # "strict" is not read: the schema is enforced whatever it says, as
        # vLLM does.
        return RequestGrammar(
            "json-schema", _schema_text(spec["schema"], f"{param}.json_schema.schema"),
            param)
    if kind == "structural_tag":
        raise GrammarRequestError(
            f'{param} type "structural_tag" is not supported: the Genie SDK '
            "compiles only JSON Schema, regex and EBNF grammars.", param)
    raise GrammarRequestError(
        f'{param}.type must be "text", "json_object" or "json_schema", '
        f"got {kind!r}", param)


def _from_structured_outputs(so) -> RequestGrammar | None:
    param = "structured_outputs"
    if not isinstance(so, dict):
        raise GrammarRequestError(f"{param} must be an object", param)
    unknown = sorted(set(so) - set(_SO_CONSTRAINTS) - set(_SO_UNSUPPORTED_OPTIONS)
                     - set(_SO_IGNORED_OPTIONS))
    if unknown:
        raise GrammarRequestError(
            f"{param} has unknown field(s) {unknown}; supported: "
            f"{list(_SO_SUPPORTED)}", param)
    for option in _SO_UNSUPPORTED_OPTIONS:
        if so.get(option) not in (None, False):
            raise GrammarRequestError(
                f"{param}.{option} is not supported: the Genie SDK has no way "
                "to pass grammar compiler options.", f"{param}.{option}")
    given = [k for k in _SO_CONSTRAINTS
             if so.get(k) is not None and so.get(k) is not False]
    if not given:
        return None
    if len(given) > 1:
        raise GrammarRequestError(
            f"{param} must set exactly one of {list(_SO_SUPPORTED)}, "
            f"got {given}", param)
    key = given[0]
    value = so[key]
    where = f"{param}.{key}"
    if key == "json":
        return RequestGrammar("json-schema", _schema_text(value, where), where)
    if key == "json_object":
        if value is not True:
            raise GrammarRequestError(f"{where} must be true or false", where)
        return RequestGrammar("json-schema", json.dumps(JSON_OBJECT_SCHEMA), where)
    if key in ("regex", "grammar"):
        if not isinstance(value, str) or not value:
            raise GrammarRequestError(f"{where} must be a non-empty string", where)
        return RequestGrammar("regex" if key == "regex" else "ebnf", value, where)
    if key == "choice":
        if (not isinstance(value, list) or not value
                or not all(isinstance(c, str) for c in value)):
            raise GrammarRequestError(
                f"{where} must be a non-empty array of strings", where)
        return RequestGrammar("ebnf", choice_to_ebnf(value), where)
    raise GrammarRequestError(
        f'{where} is not supported: the Genie SDK compiles only JSON Schema, '
        "regex and EBNF grammars.", where)


def parse_request_grammar(body: dict) -> RequestGrammar | None:
    """The request's constraint, or None when it asks for none. Raises
    GrammarRequestError for anything malformed or not honourable."""
    for legacy in _LEGACY_GUIDED:
        if legacy in body:
            raise GrammarRequestError(
                f"{legacy} is not supported (vLLM removed the guided_* fields "
                "in v0.12.0); use structured_outputs or response_format.",
                legacy)
    rf = body.get("response_format")
    so = body.get("structured_outputs")
    from_rf = _from_response_format(rf) if rf is not None else None
    from_so = _from_structured_outputs(so) if so is not None else None
    if from_rf is not None and from_so is not None:
        raise GrammarRequestError(
            "response_format and structured_outputs both set a constraint; "
            "send one of them.", "structured_outputs")
    return from_rf or from_so


# ---------------------------------------------------------------- slot support

# Slot.active_grammar values besides a RequestGrammar.key.
BUNDLE = "bundle"   # the grammar from the bundle's genie_config.json
UNKNOWN = "?"       # a failed setGrammar left the dialog in a state we do not know


@dataclass(frozen=True)
class GrammarSupport:
    """Whether a loaded model's dialog takes a per-request grammar."""
    supported: bool
    reason: str = ""                 # why not, when not supported
    # The bundle's own grammar (kind, absolute file path), which a request
    # without a constraint gets back. None: the bundle has none.
    bundle: tuple[str, str] | None = None

    @property
    def baseline(self) -> str | None:
        """Slot.active_grammar for a request that sets no constraint."""
        return BUNDLE if self.bundle else None


def bundle_grammar(dialog_cfg: dict) -> tuple[str, str] | None:
    """(kind, file) of the dialog config's grammar block, or None. The type
    defaults to json-schema, as in the SDK (Context.cpp). The file is already
    absolute: slots.load_dialog_config resolves it."""
    g = dialog_cfg.get("context", {}).get("grammar")
    if not isinstance(g, dict) or not g.get("file"):
        return None
    return (g.get("type") or "json-schema", g["file"])


def probe_support(lib, handle, dialog_cfg: dict, slot_name: str) -> GrammarSupport:
    """Decides, right after GenieDialog_create, whether this dialog can take
    GenieDialog_setGrammar.

    The symbol alone does not say: a libGenie built from the SDK's sources
    without the grammar backend exports it too, and then every call fails
    with GENIE_STATUS_ERROR_GENERAL, as it does on a dialog type that does
    not implement grammar ("ssd-q1", ...). So the dialog is asked once, with
    the call that disables grammar -- a no-op on a dialog that has none.

    A bundle that ships a grammar is not asked: the dialog was created with
    one, which proves the backend is there, and the disabling call would
    throw that grammar away."""
    bundle = bundle_grammar(dialog_cfg)
    if not lib.has_set_grammar:
        return GrammarSupport(
            False, "this libGenie has no GenieDialog_setGrammar (QAIRT 2.51.0 "
                   "or later is required)", bundle)
    if bundle is not None:
        return GrammarSupport(True, "", bundle)
    ret = lib.set_grammar(handle, None, None)
    if ret != capi.STATUS_SUCCESS:
        dialog_type = dialog_cfg.get("type", "basic")
        logger.info(f"[{slot_name}] per-request grammar unavailable: "
                    f"GenieDialog_setGrammar(NULL, NULL) returned {ret} "
                    f"(dialog type {dialog_type!r})")
        return GrammarSupport(
            False, f"GenieDialog_setGrammar returned {ret} on this dialog: the "
                   "libGenie was built without the grammar backend, or dialog "
                   f"type {dialog_type!r} does not implement grammar (the SDK "
                   'supports "basic" and "eaglet")')
    return GrammarSupport(True)


def apply_to_slot(lib, slot, grammar: RequestGrammar | None,
                  request_id: str) -> str | None:
    """Puts the grammar this request needs on the slot's dialog: its own, or,
    without one, the bundle's (or none). Skipped when the dialog already
    holds it. Call with slot.lock held, after GenieDialog_reset and before
    any prefix-cache restore. Returns an error message, or None.

    setGrammar takes only a file path (the SDK reads the file into a string
    and compiles it), so a request's grammar goes through a private temporary
    file that is removed as soon as the call returns."""
    support: GrammarSupport = slot.grammar_support
    target = grammar.key if grammar is not None else support.baseline
    if slot.active_grammar == target:
        return None
    started = time.perf_counter()
    if grammar is not None:
        fd, path = tempfile.mkstemp(prefix="ogs-grammar-", suffix=".txt")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(grammar.definition)
            ret = lib.set_grammar(slot.handle, grammar.kind, path)
        finally:
            os.unlink(path)
        what = f"{grammar.kind} {target}"
    elif support.bundle is not None:
        ret = lib.set_grammar(slot.handle, *support.bundle)
        what = "the bundle's grammar"
    else:
        ret = lib.set_grammar(slot.handle, None, None)
        what = "no grammar"
    ms = (time.perf_counter() - started) * 1000
    if ret != capi.STATUS_SUCCESS:
        slot.active_grammar = UNKNOWN
        logger.error(f"[{slot.name}] GenieDialog_setGrammar ({what}) failed: "
                     f"{ret} [{request_id}]")
        if grammar is None:
            return (f"Could not restore {what} on slot '{slot.name}' "
                    f"(GenieDialog_setGrammar returned {ret}).")
        return (f"The grammar from {grammar.param} was rejected by the SDK "
                f"(GenieDialog_setGrammar returned {ret}); the SDK's message "
                "is in the server log.")
    slot.active_grammar = target
    logger.info(f"[{slot.name}] grammar set: {what} in {ms:.1f}ms [{request_id}]")
    return None
