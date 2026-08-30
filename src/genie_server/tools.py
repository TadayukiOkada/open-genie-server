"""OpenAI `tools` (function calling) support via the Hermes prompt format.

Qwen3's chat template natively uses the Hermes convention: function
signatures are presented inside <tools></tools> XML tags in the system
prompt, and the model emits calls as <tool_call>{"name": ..., "arguments":
...}</tool_call> blocks. This module renders the prompt side and parses the
output side; genie_server.app maps the results onto OpenAI's
message.tool_calls / finish_reason="tool_calls" wire format.

(Same approach as Qualcomm's qai-appbuilder GenieAPIService reference, minus
its Qwen1.5-era ✿FUNCTION✿ dialect — Hermes is what Qwen3/Llama3-class
models are actually trained on.)
"""

import json
import logging
import re
import uuid

logger = logging.getLogger(__name__)

_TOOLS_BLOCK_HEADER = """

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
"""

_TOOLS_BLOCK_FOOTER = """</tools>

For each function call, return a json object with function name and arguments \
within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_OPEN_TAG = "<tool_call>"
_CLOSE_TAG = "</tool_call>"


def render_tools_block(tools: list) -> str:
    """The Hermes tools block to append to the system prompt (leading
    newlines included; templates.prepare_messages strips them when the block
    becomes the whole system message)."""
    lines = ""
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type", "function") != "function":
            continue
        lines += json.dumps(tool, ensure_ascii=False) + "\n"
    if not lines:
        return ""
    return _TOOLS_BLOCK_HEADER + lines + _TOOLS_BLOCK_FOOTER


def format_tool_call_for_prompt(tool_call: dict) -> str:
    """Renders one OpenAI assistant-message tool_call back into the prompt
    form the model originally emitted (used when a client sends the tool
    round-trip history back for the follow-up turn)."""
    fn = tool_call.get("function", {})
    args = fn.get("arguments", "{}")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass  # keep the raw string — better than dropping the call
    payload = json.dumps({"name": fn.get("name", ""), "arguments": args},
                         ensure_ascii=False)
    return f"<tool_call>\n{payload}\n</tool_call>"


def build_call(name: str, arguments) -> dict:
    """One name + arguments -> the OpenAI wire shape. Shared with the other
    tool dialects in tool_formats.py, which parse differently but have to
    produce the same thing."""
    return _build_call({"name": name, "arguments": arguments})


def _build_call(obj: dict) -> dict:
    """One parsed {"name", "arguments"} object -> the OpenAI wire shape."""
    args = obj.get("arguments", {})
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": obj["name"],
            "arguments": args if isinstance(args, str)
            else json.dumps(args, ensure_ascii=False),
        },
    }


def _recover_unterminated(content: str, tool_calls: list) -> str:
    """Recovers a <tool_call> the model opened but never closed.
    TOOL_CALL_RECOVERY gates this; the caller only calls it when that is on.

    Small models drop the closing tag and emit EOS straight after the JSON --
    observed on qwen3_0_6b w4a16, where the same prompt emits the closing tag
    or not depending on wording. Once generation has stopped there is nothing
    ambiguous left: an opening tag with no closer, followed by a complete JSON
    object carrying a "name", running to the end of the text, is a tool call.

    Recoverable is not the same as harmless, which is why it is behind the
    flag rather than always on. A model that will not terminate its own call
    has a defect of the same kind as one that mangles the opening marker, and
    repairing it here would make the bundle score better on
    /v1/chat/completions than on /v1/completions, which has no recovery to
    apply -- the endpoint-dependent score the flag defaults off to avoid.

    Anything that does not parse is left in the content untouched, so a
    generation cut off mid-JSON by max_tokens still surfaces as text.
    """
    open_at = content.rfind(_OPEN_TAG)
    if open_at == -1 or _CLOSE_TAG in content[open_at:]:
        return content
    payload = content[open_at + len(_OPEN_TAG):].strip()
    try:
        obj = json.loads(payload)
        obj["name"]
    except (json.JSONDecodeError, TypeError, KeyError):
        logger.warning("Unterminated <tool_call> block left in content: %.120s",
                       payload)
        return content
    tool_calls.append(_build_call(obj))
    return content[:open_at]


def _json_object_spans(text: str):
    """Yields (start, end) of every balanced {...} run in `text`.

    Brace counting is string- and escape-aware, so a "}" inside an argument
    value does not end the object early. Runs that never balance are skipped.
    """
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth, j, in_str, esc = 0, i, False, False
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    yield i, j + 1
                    break
            j += 1
        if j >= n:          # never balanced — nothing further can either
            return
        i = j + 1


def _marker_debris_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Widens a recovered call's span over the mangled marker beside it.

    Whatever token the model emitted in place of "<tool_call>" sits on its own
    line next to the JSON, so a neighbouring line with no whitespace in it is
    taken as that marker and removed with the call. Prose does not look like
    that -- it has spaces -- but a genuine one-word line next to a tool call
    would be swallowed, which is the price of not showing the user "ФРАГМЕНТ".
    """
    lo = text.rfind("\n", 0, start) + 1          # start of the JSON's own line
    prev_end = lo - 1
    if prev_end > 0:
        prev_lo = text.rfind("\n", 0, prev_end) + 1
        line = text[prev_lo:prev_end]
        if line.strip() and not any(ch.isspace() for ch in line.strip()):
            lo = prev_lo
    hi = text.find("\n", end)
    hi = len(text) if hi == -1 else hi + 1
    nxt_end = text.find("\n", hi)
    nxt_end = len(text) if nxt_end == -1 else nxt_end
    line = text[hi:nxt_end]
    if line.strip() and not any(ch.isspace() for ch in line.strip()):
        hi = nxt_end + 1 if nxt_end < len(text) else len(text)
    return lo, hi


def _recover_unmarked(content: str, tool_calls: list,
                      known_tool_names: set) -> str:
    """Recovers a call whose <tool_call> marker never made it into the text.

    Two models on the SA8255P board produce a correct call body with a broken
    marker: qwen3_4b_instruct_2507 w4a16 replaces the <tool_call> token with
    Cyrillic ("ФРАГМЕНТ", "Флагорное", a different string per request) on half
    of its calls, and qwen3_0_6b omits the tags altogether and drops the JSON
    after its think block. Both are silent -- the caller gets prose and
    finish_reason "stop" where their code reads message.tool_calls.

    The discriminator is the tool name: only a JSON object whose "name" is one
    the caller actually declared in this request is taken. Bare JSON was
    deliberately left as text when only the closing tag was missing, because a
    model answering a question in JSON would be misread as calling a function;
    requiring the name to match a declared tool removes that.
    """
    spans = []
    for start, end in _json_object_spans(content):
        try:
            obj = json.loads(content[start:end])
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or obj.get("name") not in known_tool_names:
            continue
        spans.append((_marker_debris_span(content, start, end), obj))

    if not spans:
        return content
    for (lo, hi), obj in spans:
        tool_calls.append(_build_call(obj))
        logger.info("Recovered an unmarked tool call for %r", obj["name"])
    out, prev = [], 0
    for (lo, hi), _ in spans:
        out.append(content[prev:lo])
        prev = hi
    out.append(content[prev:])
    return "".join(out)


def parse_tool_calls(text: str,
                     known_tool_names: set | None = None) -> tuple[str, list]:
    """Extracts <tool_call> blocks from generated text.

    Returns (content_without_tool_calls, tool_calls) where tool_calls is a
    list of OpenAI-shaped {"id", "type", "function": {"name", "arguments"}}
    dicts. Blocks whose JSON cannot be parsed are left in the content
    untouched rather than silently dropped. A final block that was opened but
    never closed is recovered -- see _recover_unterminated.

    `known_tool_names` arms the last resort: with it, a bare JSON object naming
    one of those tools is taken as a call even though its marker is missing or
    mangled (_recover_unmarked). Callers pass it only when the request carried
    `tools` and TOOL_CALL_RECOVERY is on; without it this function behaves
    exactly as it did before, and no reply can be reinterpreted as a call.
    """
    tool_calls = []

    def _replace(match: re.Match) -> str:
        try:
            obj = json.loads(match.group(1))
            obj["name"]
        except (json.JSONDecodeError, TypeError, KeyError):
            logger.warning("Unparseable <tool_call> block left in content: %.120s",
                           match.group(1))
            return match.group(0)
        tool_calls.append(_build_call(obj))
        return ""

    content = _TOOL_CALL_RE.sub(_replace, text)
    if known_tool_names:
        content = _recover_unterminated(content, tool_calls)
    if known_tool_names:
        content = _recover_unmarked(content, tool_calls, known_tool_names)
    return content.strip(), tool_calls


class ToolCallStreamFilter:
    """Streaming filter that holds back <tool_call>...</tool_call> blocks.

    feed(token) returns text that is safe to emit to the client now (i.e.
    provably not part of a tool-call block); everything held back is
    resolved by finalize(), which returns (remaining_text, tool_calls).

    The filter is conservative: any buffer suffix that could still grow into
    "<tool_call>" is withheld until disambiguated, so clients never see a
    partial opening tag.
    """

    _OPEN = "<tool_call>"
    _CLOSE = "</tool_call>"

    def __init__(self, known_tool_names: set | None = None) -> None:
        self._buf = ""          # text not yet classified (emit vs. block)
        self._captured = ""     # completed tool-call blocks (+ open block prefix)
        self._in_block = False  # currently between _OPEN and its _CLOSE
        # Recovery mode (see _recover_unmarked): a mangled marker is not a tag,
        # so the state machine above would stream the call out as content and
        # only finalize() would notice -- too late to take it back. With names
        # to match against, text leaving the tag scan is additionally held one
        # line at a time until it is provably not a call body or its marker.
        self._known = known_tool_names or None
        self._pending = ""      # recovery mode: text held pending a verdict
        self._json_hold = False # recovery mode: inside a candidate JSON object
        self._line_open = False # recovery mode: current line already emitted

    def _open_pos(self) -> int:
        """Index in _buf where a (possibly partial) _OPEN begins, else -1."""
        pos = self._buf.find(self._OPEN)
        if pos != -1:
            return pos
        # longest _buf suffix that is a prefix of _OPEN
        max_len = min(len(self._buf), len(self._OPEN) - 1)
        for n in range(max_len, 0, -1):
            if self._OPEN.startswith(self._buf[-n:]):
                return len(self._buf) - n
        return -1

    def _screen(self, text: str) -> str:
        """Recovery mode only: hold back anything that could be a call body or
        the mangled marker beside it, and release the rest as it completes.

        A line is held when it is a single whitespace-free token (marker
        shaped) or opens a JSON object; a JSON object is held until its braces
        balance. Everything held is resolved by finalize(). Prose streams as
        usual apart from its first word, which waits for the space that proves
        the line is not a marker.
        """
        self._pending += text
        out = ""
        while self._pending:
            if self._line_open:
                # The verdict for this line was already taken and part of it
                # has gone to the client; the rest of it goes too, or the
                # client sees the line reordered around the held text.
                nl = self._pending.find("\n")
                if nl == -1:
                    out += self._pending
                    self._pending = ""
                    return out
                out += self._pending[:nl + 1]
                self._pending = self._pending[nl + 1:]
                self._line_open = False
                continue
            if self._json_hold:
                brace = self._pending.find("{")
                end = next((e for st, e in _json_object_spans(self._pending)
                            if st == brace), None)
                if end is None:
                    return out              # object still incomplete
                self._captured += self._pending[:end]
                self._pending = self._pending[end:]
                self._json_hold = False
                continue
            nl = self._pending.find("\n")
            head = self._pending if nl == -1 else self._pending[:nl + 1]
            stripped = head.strip()
            if stripped.startswith("{"):
                self._json_hold = True
                continue
            if nl == -1:
                # Incomplete line: emit it only once it cannot become either
                # a marker (needs whitespace) or a JSON object (needs "{").
                if stripped and any(ch.isspace() for ch in stripped):
                    out += self._pending
                    self._pending = ""
                    self._line_open = True
                return out
            if stripped and not any(ch.isspace() for ch in stripped):
                self._captured += head       # marker-shaped line
            else:
                out += head
            self._pending = self._pending[nl + 1:]
        return out

    def feed(self, token: str) -> str:
        if self._known:
            return self._screen(self._feed_tags(token))
        return self._feed_tags(token)

    def _feed_tags(self, token: str) -> str:
        self._buf += token
        emit = ""
        while True:
            if self._in_block:
                close = self._buf.find(self._CLOSE)
                if close == -1:
                    return emit  # inside a block, waiting for its close tag
                close_end = close + len(self._CLOSE)
                self._captured += self._buf[:close_end]
                self._buf = self._buf[close_end:]
                self._in_block = False
                continue
            pos = self._open_pos()
            if pos == -1:
                emit += self._buf
                self._buf = ""
                return emit
            emit += self._buf[:pos]
            rest = self._buf[pos:]
            if not rest.startswith(self._OPEN):
                self._buf = rest
                return emit  # partial open tag — wait for more tokens
            self._buf = rest[len(self._OPEN):]
            self._captured += self._OPEN
            self._in_block = True

    def finalize(self) -> tuple[str, list]:
        """Flushes everything still held back. A block left unterminated is
        recovered by parse_tool_calls only when TOOL_CALL_RECOVERY is on
        (which is also what self._known signals); otherwise it comes back as
        text, which is what the model emitted."""
        if self._known:
            self._captured += self._screen(self._feed_tags(""))
            self._captured += self._pending
            self._pending, self._json_hold = "", False
            self._line_open = False
        text = self._captured + self._buf
        self._buf, self._captured, self._in_block = "", "", False
        return parse_tool_calls(text, self._known)
