"""Tool-call dialects: how a model family declares and emits function calls.

OpenAI's `tools` are a wire format, not a prompt format. Every model family
has been trained on its own way of receiving declarations and emitting calls,
and they do not resemble each other:

    Hermes (Qwen3, ...)   <tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>
    gemma4                <|tool_call>call:f{a:1}<tool_call|>

The second is not JSON: strings are wrapped in a `<|"|>` delimiter, keys come
in dictsort order, and type names are Gemini-flavoured (OBJECT, NUMBER). A
configuration file can pick between dialects; it cannot describe one, which is
why this is a registry of implementations rather than a schema. Adding a family
means writing a class here — the same shape `vlm_specs.py` uses for VLMs.

Each dialect supplies four things:

    render_tools_block(tools)          declarations, appended to the system turn
    format_tool_call_for_prompt(call)  one call rendered back for a follow-up turn
    parse_tool_calls(text, known)      (content, calls) out of a finished reply
    stream_filter(known)               feed()/finalize() over a live stream

`parse_tool_calls` and the filter are separate because streaming has to decide
what is safe to emit before the reply is complete.
"""

import json
import re

from . import tools as hermes

# --------------------------------------------------------------------- hermes


class HermesToolFormat:
    """Qwen3 and everything else we have seen: JSON inside <tool_call> tags.

    The implementation lives in tools.py, which predates this module and is
    exercised by most of the tool tests; this class is the seam, not a rewrite.
    """

    name = "hermes"

    render_tools_block = staticmethod(hermes.render_tools_block)
    format_tool_call_for_prompt = staticmethod(hermes.format_tool_call_for_prompt)
    parse_tool_calls = staticmethod(hermes.parse_tool_calls)

    @staticmethod
    def stream_filter(known_tool_names=None):
        return hermes.ToolCallStreamFilter(known_tool_names)


# --------------------------------------------------------------------- gemma4

Q = '<|"|>'   # gemma4's string delimiter (token 52)

# Keys that describe a schema node rather than a property of it, so
# _fmt_params must not treat them as parameters.
_SCHEMA_KEYS = ("description", "type", "properties", "required", "nullable")

_CALL_OPEN = re.compile(r"<\|tool_call>call:([A-Za-z_][\w.]*)\{")
_CALL_CLOSE = "<tool_call|>"


def _scan_call(text: str, brace_at: int):
    """End of the `{...}` body that starts at brace_at, quote- and nest-aware,
    or -1 if it never closes. Same scanning rule as _split_top, which has to
    ignore braces and commas inside a <|"|> string."""
    depth, i = 0, brace_at
    while i < len(text):
        if text.startswith(Q, i):
            end = text.find(Q, i + len(Q))
            if end < 0:
                return -1
            i = end + len(Q)
            continue
        c = text[i]
        if c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _fmt_arg(value, escape_keys=True) -> str:
    """Google's format_argument(): a value in gemma4's own notation."""
    if isinstance(value, str):
        return f"{Q}{value}{Q}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, dict):
        parts = []
        for k, v in sorted(value.items()):
            key = f"{Q}{k}{Q}" if escape_keys else str(k)
            parts.append(f"{key}:{_fmt_arg(v, escape_keys)}")
        return "{" + ",".join(parts) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_fmt_arg(v, escape_keys) for v in value) + "]"
    if value is None:
        return "null"
    return str(value)


def _fmt_params(properties: dict, required) -> str:
    """Google's format_parameters(). One entry per property, dictsort order."""
    out = []
    for key, value in sorted((properties or {}).items()):
        if key in _SCHEMA_KEYS or not isinstance(value, dict):
            continue
        body = []
        if value.get("description"):
            body.append(f"description:{Q}{value['description']}{Q}")
        if value.get("nullable"):
            body.append("nullable:true")

        vtype = str(value.get("type", "")).upper()
        if vtype == "STRING" and value.get("enum"):
            body.append(f"enum:{_fmt_arg(value['enum'])}")
        elif vtype == "OBJECT":
            inner = value.get("properties")
            nested = _fmt_params(inner, value.get("required", [])) \
                if isinstance(inner, dict) else ""
            body.append("properties:{" + nested + "}")
            if value.get("required"):
                req = ",".join(f"{Q}{i}{Q}" for i in value["required"])
                body.append(f"required:[{req}]")
        elif vtype == "ARRAY" and isinstance(value.get("items"), dict):
            items = []
            for ik, iv in sorted(value["items"].items()):
                if iv is None:
                    continue
                if ik == "properties" and isinstance(iv, dict):
                    items.append("properties:{" + _fmt_params(
                        iv, value["items"].get("required", [])) + "}")
                elif ik == "required":
                    items.append("required:[" + ",".join(
                        f"{Q}{r}{Q}" for r in iv) + "]")
                elif ik == "type":
                    items.append("type:" + (
                        _fmt_arg(iv.upper()) if isinstance(iv, str)
                        else _fmt_arg([str(x).upper() for x in iv])))
                else:
                    items.append(f"{ik}:{_fmt_arg(iv)}")
            body.append("items:{" + ",".join(items) + "}")

        body.append(f"type:{Q}{vtype}{Q}")
        out.append(f"{key}:{{" + ",".join(body) + "}")
    return ",".join(out)


def _fmt_declaration(tool: dict) -> str:
    """Google's format_function_declaration().

    Type names are uppercased rather than mapped: an OpenAI client sends JSON
    Schema types (object, string, number, integer, boolean, array), and
    uppercasing those gives exactly the names gemma4 was trained on. A
    non-OpenAI schema with its own type vocabulary would need mapping first —
    examples/bfcl does that, since Gorilla calls an object a "dict".
    """
    fn = tool.get("function", tool)
    out = (f"declaration:{fn.get('name', '')}"
           f"{{description:{Q}{fn.get('description', '')}{Q}")

    params = fn.get("parameters") or {}
    if params:
        inner = []
        if params.get("properties"):
            inner.append("properties:{" + _fmt_params(
                params["properties"], params.get("required")) + "}")
        if params.get("required"):
            inner.append("required:[" + ",".join(
                f"{Q}{i}{Q}" for i in params["required"]) + "]")
        if params.get("type"):
            inner.append(f"type:{Q}{str(params['type']).upper()}{Q}")
        out += ",parameters:{" + ",".join(inner) + "}"

    return out + "}"


def _split_top(body: str):
    """Split `a:1,b:{c:2}` on top-level commas, brace/bracket/quote aware."""
    parts, depth, i, start = [], 0, 0, 0
    while i < len(body):
        if body.startswith(Q, i):                 # skip a quoted string whole
            end = body.find(Q, i + len(Q))
            i = len(body) if end < 0 else end + len(Q)
            continue
        c = body[i]
        if c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(body[start:i])
            start = i + 1
        i += 1
    if start < len(body):
        parts.append(body[start:])
    return [p for p in parts if p.strip()]


def _parse_value(text: str):
    t = text.strip()
    if t.startswith(Q) and t.endswith(Q) and len(t) >= 2 * len(Q):
        return t[len(Q):-len(Q)]
    if t in ("true", "false"):
        return t == "true"
    if t == "null":
        return None
    if t.startswith("{") and t.endswith("}"):
        return {k: v for k, v in (_parse_pair(p) for p in _split_top(t[1:-1]))}
    if t.startswith("[") and t.endswith("]"):
        return [_parse_value(p) for p in _split_top(t[1:-1])]
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        return t


def _parse_pair(part: str):
    key, _, value = part.partition(":")
    key = key.strip()
    if key.startswith(Q) and key.endswith(Q):
        key = key[len(Q):-len(Q)]
    return key, _parse_value(value)


class Gemma4ToolFormat:
    """gemma4's native tool tokens.

    Ported from examples/bfcl/gemma4_fc_handler.py, which mirrors Google's own
    template (llama.cpp's google-gemma-4-31B-it-interleaved.jinja, macros
    format_function_declaration / format_parameters / format_argument) and was
    validated against a leaderboard before this existed.
    """

    name = "gemma4"

    @staticmethod
    def render_tools_block(tools: list) -> str:
        """Declarations for the system turn. gemma4 keeps `system` as its own
        turn (see templates.py), which is where these belong."""
        out = ""
        for tool in tools or []:
            if not isinstance(tool, dict) or tool.get("type", "function") != "function":
                continue
            out += "<|tool>" + _fmt_declaration(tool) + "<tool|>"
        return ("\n" + out) if out else ""

    @staticmethod
    def format_tool_call_for_prompt(tool_call: dict) -> str:
        """One assistant tool_call rendered back the way the model emits it,
        for the follow-up turn of a round trip."""
        fn = tool_call.get("function", {})
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        body = ",".join(f"{k}:{_fmt_arg(v)}" for k, v in sorted(args.items()))
        return f"<|tool_call>call:{fn.get('name', '')}{{{body}}}<tool_call|>"

    @staticmethod
    def parse_tool_calls(text: str, known_tool_names=None):
        """(content, calls).

        known_tool_names is the TOOL_CALL_RECOVERY signal (app.py passes it
        only when that is on). Here it gates one repair: accepting a call
        whose closing `<tool_call|>` never arrived. That is a real thing
        gemma4-e2b does — often — but a model that will not close its own
        call is a model with a defect, the same kind as one that mangles the
        opening marker, and a leaderboard counts both as failures (see the
        BFCL notes in examples/bfcl). Recovering it by default would make the
        bundle measure better here than through /v1/completions, which has no
        recovery to apply: the endpoint-dependent score TOOL_CALL_RECOVERY
        was turned off to stop producing.
        """
        text = text or ""
        calls, spans, pos = [], [], 0
        while True:
            match = _CALL_OPEN.search(text, pos)
            if not match:
                break
            body_end = _scan_call(text, match.end() - 1)
            if body_end < 0:
                # Ran off the end mid-body: a generation cut short by
                # max_tokens, not a call. Leave it in the content.
                break
            end = body_end + 1
            if text.startswith(_CALL_CLOSE, end):
                end += len(_CALL_CLOSE)
            elif not known_tool_names:
                # Unterminated, and recovery is off: leave it as text so the
                # caller sees what the model actually emitted.
                pos = end
                continue
            args = {}
            for part in _split_top(text[match.end():body_end]):
                k, v = _parse_pair(part)
                if k:
                    args[k] = v
            calls.append(hermes.build_call(match.group(1), args))
            spans.append((match.start(), end))
            pos = end

        content = text
        for start, end in reversed(spans):
            content = content[:start] + content[end:]
        return content.strip(), calls

    @staticmethod
    def stream_filter(known_tool_names=None):
        return BufferedStreamFilter(Gemma4ToolFormat, known_tool_names)


# ------------------------------------------------------------------ streaming


class BufferedStreamFilter:
    """Streaming for a dialect with no incremental filter of its own.

    Holds the whole reply and resolves it in finalize(), so the client gets
    correct content and tool_calls but no text until generation ends. That is
    a worse experience than the Hermes filter, which releases prose as soon as
    it can prove it is not part of a call — but it is the same result, and it
    is what makes a new dialect usable the day it is written instead of
    waiting for a bespoke state machine.

    Same interface as tools.ToolCallStreamFilter.
    """

    def __init__(self, fmt, known_tool_names=None):
        self._fmt = fmt
        self._known = known_tool_names
        self._buf = ""

    def feed(self, text: str) -> str:
        self._buf += text
        return ""

    def finalize(self):
        return self._fmt.parse_tool_calls(self._buf, self._known)


# ------------------------------------------------------------------- registry

FORMATS = {f.name: f for f in (HermesToolFormat, Gemma4ToolFormat)}
DEFAULT_FORMAT = HermesToolFormat.name


def detect(chat_template: str) -> str:
    """Which dialect a slot should use, derived from the chat template it
    already detected.

    Deriving it rather than matching model names again keeps the two in step:
    a bundle that renders as gemma4 turns also speaks gemma4 tool tokens, and
    there is no way to configure a combination that cannot exist.
    """
    return "gemma4" if chat_template == "gemma4" else DEFAULT_FORMAT


def get(name: str):
    """Dialect by name, falling back to Hermes for an unknown one — the same
    permissive rule chat templates follow."""
    return FORMATS.get(name or "", HermesToolFormat)
