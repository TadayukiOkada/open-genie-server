"""Chat-template rendering (chatml / llama3 / llama2) and prompt splitting.

Template selection runs off the loaded model's own directory name (or the
CHAT_TEMPLATE override) — never the client-supplied request 'model' field,
which lm_eval always sets to a fixed placeholder and therefore can't be used
to distinguish models.

All functions here are pure (no SDK, no I/O) and operate on messages that
have been normalized by prepare_messages().
"""

import logging

from . import tool_formats

logger = logging.getLogger(__name__)

TEMPLATE_FAMILIES = ("chatml", "llama3", "llama2", "gemma", "gemma4")


class UnrenderableMessageError(ValueError):
    """A message the slot's chat template has no form for. Raised instead of
    dropping it: a turn that silently vanishes from the prompt is a wrong
    answer nobody can trace back to the request."""


def _refuse_tool_history(messages: list, template: str) -> None:
    """llama2 and Gemma 2/3 have no tool-call or tool-result form in their
    own chat templates. llama2 used to drop tool-role messages outright, and
    both dropped an assistant turn's tool_calls, so a tool round trip reached
    the model with its calls and results missing."""
    for i, m in enumerate(messages):
        if m.get("role") == "tool" or m.get("tool_calls"):
            what = "a tool result" if m.get("role") == "tool" else "tool_calls"
            raise UnrenderableMessageError(
                f"messages[{i}] carries {what}, which the {template!r} chat "
                "template has no form for; send the round trip to a slot "
                "whose template does (chatml, llama3, gemma4)")


def detect_template(hint: str) -> str:
    """Guess a chat-template family from a free-text hint (the model
    directory name or an explicit CHAT_TEMPLATE override)."""
    h = (hint or "").lower()
    if "llama3" in h or "llama-3" in h:
        return "llama3"
    if "llama2" in h or "mistral" in h:
        return "llama2"
    # Checked before plain "gemma": gemma4 turns are marked with a different
    # pair of tokens, and the Gemma 2/3 spelling is not in its vocabulary at
    # all (see render_chat_prompt).
    if "gemma4" in h or "gemma-4" in h:
        return "gemma4"
    if "gemma" in h:
        return "gemma"
    return "chatml"


def content_to_text(content) -> str:
    """Flattens an OpenAI message `content` to plain text.

    Clients like Open WebUI may send `content` as a parts array
    ([{"type": "text", "text": ...}, ...]) even for text-only requests;
    embedding the raw list repr into the prompt would corrupt it.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return str(content)


def prepare_messages(messages: list, enable_thinking: bool = True,
                     tools: list | None = None, tool_format=None) -> list:
    """Normalizes an OpenAI messages array for rendering:

    - copies every message (the caller's list is never mutated),
    - flattens parts-array content to plain text,
    - injects the Hermes tools block into the system message that opens the
      conversation when `tools` are given, synthesizing one at the front if
      the conversation does not open with a system message — matching
      Qwen3's own chat template, which renders the tools block in a leading
      system turn even without a system prompt. A system message later in
      the conversation is left as the caller wrote it,
    - applies Qwen3's "/no_think" soft switch when enable_thinking=False,
      separated from whatever precedes it by a blank line.

    "/no_think" is Qwen3's own documented mechanism for disabling reasoning:
    a literal command appended to the system prompt text. (Injecting an empty
    <think></think> block into the prompt structure instead — HuggingFace's
    chat-template approach — was found by Qualcomm's qai-appbuilder reference
    service, via real-device testing, to make Qwen3 models degenerate on
    short prompts, so it is deliberately NOT done here.)
    """
    out = [dict(m, content=content_to_text(m.get("content"))) for m in messages]

    def _append_to_system(text: str) -> None:
        # Only the system message that OPENS the conversation: appending to
        # the first system message wherever it sits would put the tools block
        # (or /no_think) in the middle of the conversation.
        if out and out[0].get("role") == "system":
            m = out[0]
            m["content"] = (m["content"] + text) if m["content"] else text.lstrip("\n")
            return
        out.insert(0, {"role": "system", "content": text.lstrip("\n")})

    fmt = tool_format or tool_formats.HermesToolFormat
    if tools:
        _append_to_system(fmt.render_tools_block(tools))
    if not enable_thinking:
        # The leading blank line is load-bearing.  render_tools_block() ends
        # with the literal "</tool_call>" of its format example, so appending a
        # bare "/no_think" glues the directive onto that example and the model
        # reproduces it as part of the tool-call format -- observed on
        # qwen3_4b_instruct_2507, which emitted a trailing "/no_think" line
        # after its tool call.  render_tools_block() carries its own leading
        # newlines for the same reason; this one has to supply its own.
        _append_to_system("\n\n/no_think")
    return out


# ---------------------------------------------------------------- rendering

def _render_chatml_message(m: dict, tool_format=None) -> str:
    """One ChatML turn. Assistant tool_calls and tool-role results use
    Qwen3's own template forms (<tool_call> / <tool_response>)."""
    role = m.get("role", "user")
    content = m.get("content", "")

    if role == "assistant" and m.get("tool_calls"):
        body = content
        for tc in m["tool_calls"]:
            body += ("\n" if body else "") + (
                tool_format or tool_formats.HermesToolFormat
            ).format_tool_call_for_prompt(tc)
        return f"<|im_start|>assistant\n{body}<|im_end|>\n"

    if role == "tool":
        # Qwen3 renders tool results inside a user turn as <tool_response>.
        return (f"<|im_start|>user\n<tool_response>\n{content}\n"
                f"</tool_response><|im_end|>\n")

    return f"<|im_start|>{role}\n{content}<|im_end|>\n"


def render_chat_prompt(messages: list, template: str, tool_format=None,
                       bos: bool = True) -> str:
    """Formats a prepared messages array into a prompt string ending with the
    assistant generation header.

    tool_format decides how an assistant turn's tool_calls are rendered back
    into the prompt; it defaults to Hermes, which is what every template here
    except gemma4 expects.

    bos=False leaves out the BOS the template would otherwise open with
    (`<bos>`, `<|begin_of_text|>`, llama2's first `<s>`). Pass it when the
    bundle's dialog context names a bos-token: libGenie then prepends that
    token to every query itself, and writing it here as well puts two in
    front of the prompt (measured on gemma4, where the SDK prefilled
    [2, 2, 105, ...]). Either way the bundle's own setting is left as it is."""
    if template == "llama3":
        out = "<|begin_of_text|>" if bos else ""
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "tool":
                role = "ipython"  # Llama 3.x's tool-result role name
            elif role == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    content += ("\n" if content else "") + (
                        tool_format or tool_formats.HermesToolFormat
                    ).format_tool_call_for_prompt(tc)
            out += (f"<|start_header_id|>{role}<|end_header_id|>"
                    f"\n\n{content}<|eot_id|>")
        return out + "<|start_header_id|>assistant<|end_header_id|>\n\n"

    if template == "llama2":
        _refuse_tool_history(messages, template)
        # No system turn: system text rides in the next [INST], inside
        # <<SYS>>. Consecutive system messages share that one block rather
        # than overwriting each other, and system text with no user turn
        # after it gets an [INST] of its own instead of being dropped.
        out = ""
        pending: list[str] = []

        def sys_block() -> str:
            if not pending:
                return ""
            body = "\n\n".join(pending)
            pending.clear()
            return f"<<SYS>>\n{body}\n<</SYS>>\n\n"

        for m in messages:
            r, c = m.get("role", "user"), m.get("content", "")
            if r == "system":
                pending.append(c)
            elif r == "user":
                out += f"<s>[INST] {sys_block()}{c} [/INST]"
            elif r == "assistant":
                out += f" {c} </s>"
            else:
                raise UnrenderableMessageError(
                    f"role {r!r} has no form in the 'llama2' chat template "
                    "(system, user and assistant only)")
        if pending:
            out += f"<s>[INST] {sys_block()} [/INST]"
        return out if bos else out.removeprefix("<s>")

    if template == "gemma":
        _refuse_tool_history(messages, template)
        # Gemma 2/3 family. There is no system role: per Google's own chat
        # template, system text is prepended to the next user turn. The
        # assistant role is named "model". As with llama2, consecutive system
        # messages are joined rather than overwritten, and system text with no
        # user turn after it becomes a user turn of its own.
        out = "<bos>" if bos else ""
        pending = []

        def sys_text() -> str:
            body = "\n\n".join(pending)
            pending.clear()
            return body

        for m in messages:
            r, c = m.get("role", "user"), m.get("content", "")
            if r == "system":
                pending.append(c)
            elif r == "assistant":
                out += f"<start_of_turn>model\n{c}<end_of_turn>\n"
            else:
                prefix = f"{sys_text()}\n\n" if pending else ""
                out += f"<start_of_turn>user\n{prefix}{c}<end_of_turn>\n"
        if pending:
            out += f"<start_of_turn>user\n{sys_text()}<end_of_turn>\n"
        return out + "<start_of_turn>model\n"

    if template == "gemma4":
        # gemma4 differs from Gemma 2/3 in two ways, both load-bearing:
        #
        # 1. Turns are marked with <|turn> ... <turn|> (ids 105/106), not
        #    <start_of_turn> ... <end_of_turn>. The Gemma 2/3 spelling does not
        #    exist in the gemma4 vocabulary at all, so writing it splits into
        #    ~9 ordinary tokens per marker and puts the model off its trained
        #    format. Measured on gemma4-e2b-it: 39 prompt tokens for a question
        #    that costs 22 here, with worse answers.
        # 2. system is its OWN turn — it is not folded into the first user
        #    turn. Google's template opens <|turn>system, and tool
        #    declarations live inside that same turn.
        fmt = tool_format or tool_formats.HermesToolFormat
        out = "<bos>" if bos else ""
        # tool_call_id -> function name, from the assistant turns so far:
        # an OpenAI tool message carries only the id, and gemma4 writes the
        # function's name into response:NAME{...}.
        call_names: dict = {}
        for m in messages:
            r, c = m.get("role", "user"), m.get("content", "")
            if r == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    if isinstance(tc, dict) and isinstance(tc.get("id"), str):
                        fn = tc.get("function")
                        call_names[tc["id"]] = (fn.get("name", "")
                                                if isinstance(fn, dict) else "")
                    c += ("\n" if c else "") + fmt.format_tool_call_for_prompt(tc)
            elif r == "tool":
                # A tool result comes back in a user turn, marked with
                # gemma4's own response tokens.
                name, tid = m.get("name"), m.get("tool_call_id")
                if not isinstance(name, str) or not name:
                    name = call_names.get(tid, "") if isinstance(tid, str) else ""
                if not isinstance(name, str) or not name:
                    raise UnrenderableMessageError(
                        "a tool message needs a 'name', or a 'tool_call_id' "
                        "matching an earlier assistant tool_call, for gemma4 "
                        "to write response:NAME{...}")
                out += (f"<|turn>user\n<|tool_response>response:{name}"
                        f"{{{c}}}<tool_response|><turn|>\n")
                continue
            role = "model" if r == "assistant" else \
                   "system" if r == "system" else "user"
            out += f"<|turn>{role}\n{c}<turn|>\n"
        return out + "<|turn>model\n"

    # "chatml" — ChatML / Qwen
    return "".join(_render_chatml_message(m, tool_format) for m in messages) \
        + "<|im_start|>assistant\n"


def split_prompt_for_prefix_cache(messages: list, template: str,
                                  tool_format=None,
                                  bos: bool = True) -> tuple[str, str, bool]:
    """Returns (prefix_prompt, remaining_prompt, cacheable) for the prefix KV
    cache: the system turn is the cacheable prefix, everything after it the
    per-request remainder. Llama2/Mistral fuses system into [INST] and Gemma
    2/3 prepends it to the first user turn — neither is splittable. gemma4
    keeps system as its own turn, so it splits like chatml does."""
    if template in ("llama2", "gemma"):
        return "", render_chat_prompt(messages, template, tool_format, bos), False

    # Only a conversation that OPENS with its one and only system message
    # splits into prefix + remainder. Anything else -- no system message, a
    # system message that is not first, or several -- would lose or reorder
    # the extra ones if they were pulled out, so it is rendered whole and
    # left uncached (the message order is exactly what the caller sent).
    n_sys = sum(1 for m in messages if m.get("role") == "system")
    if n_sys != 1 or messages[0].get("role") != "system":
        return "", render_chat_prompt(messages, template, tool_format, bos), False

    sc = messages[0].get("content", "")
    non_sys = messages[1:]
    if template == "gemma4":
        # gemma4 keeps system as its own turn, so unlike Gemma 2/3 it splits.
        prefix = ("<bos>" if bos else "") + f"<|turn>system\n{sc}<turn|>\n"
        remaining = render_chat_prompt(non_sys, template, tool_format, bos=False)
        return prefix, remaining, True
    if template == "llama3":
        prefix = (("<|begin_of_text|>" if bos else "")
                  + f"<|start_header_id|>system<|end_header_id|>\n\n{sc}<|eot_id|>")
        remaining = render_chat_prompt(non_sys, template, tool_format, bos=False)
    else:  # chatml
        prefix = f"<|im_start|>system\n{sc}<|im_end|>\n"
        remaining = render_chat_prompt(non_sys, template, tool_format)
    return prefix, remaining, True
