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
from . import tools as tools_mod

logger = logging.getLogger(__name__)

TEMPLATE_FAMILIES = ("chatml", "llama3", "llama2", "gemma", "gemma4")


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
    - injects the Hermes tools block into the system message when `tools`
      are given (synthesizing a system message if there is none — matching
      Qwen3's own chat template, which renders the tools block even without
      a system prompt),
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
        for m in out:
            if m.get("role") == "system":
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


def render_chat_prompt(messages: list, template: str, tool_format=None) -> str:
    """Formats a prepared messages array into a prompt string ending with the
    assistant generation header.

    tool_format decides how an assistant turn's tool_calls are rendered back
    into the prompt; it defaults to Hermes, which is what every template here
    except gemma4 expects."""
    if template == "llama3":
        out = "<|begin_of_text|>"
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "tool":
                role = "ipython"  # Llama 3.x's tool-result role name
            elif role == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    content += ("\n" if content else "") + \
                        tools_mod.format_tool_call_for_prompt(tc)
            out += (f"<|start_header_id|>{role}<|end_header_id|>"
                    f"\n\n{content}<|eot_id|>")
        return out + "<|start_header_id|>assistant<|end_header_id|>\n\n"

    if template == "llama2":
        out, sys_c = "", ""
        for m in messages:
            r, c = m.get("role", "user"), m.get("content", "")
            if r == "system":
                sys_c = f"<<SYS>>\n{c}\n<</SYS>>\n\n"
            elif r == "user":
                out += f"<s>[INST] {sys_c}{c} [/INST]"
                sys_c = ""
            elif r == "assistant":
                out += f" {c} </s>"
        return out

    if template == "gemma":
        # Gemma 2/3 family. There is no system role: per Google's own chat
        # template, system text is prepended to the first user turn. The
        # assistant role is named "model".
        out, sys_c = "<bos>", ""
        for m in messages:
            r, c = m.get("role", "user"), m.get("content", "")
            if r == "system":
                sys_c = f"{c}\n\n"
            elif r == "assistant":
                out += f"<start_of_turn>model\n{c}<end_of_turn>\n"
            else:
                out += f"<start_of_turn>user\n{sys_c}{c}<end_of_turn>\n"
                sys_c = ""
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
        out = "<bos>"
        for m in messages:
            r, c = m.get("role", "user"), m.get("content", "")
            if r == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    c += ("\n" if c else "") + fmt.format_tool_call_for_prompt(tc)
            elif r == "tool":
                # A tool result comes back in a user turn, marked with
                # gemma4's own response tokens.
                name = m.get("name", "")
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
                                  tool_format=None) -> tuple[str, str, bool]:
    """Returns (prefix_prompt, remaining_prompt, cacheable) for the prefix KV
    cache: the system turn is the cacheable prefix, everything after it the
    per-request remainder. Llama2/Mistral fuses system into [INST] and Gemma
    2/3 prepends it to the first user turn — neither is splittable. gemma4
    keeps system as its own turn, so it splits like chatml does."""
    if template in ("llama2", "gemma"):
        return "", render_chat_prompt(messages, template, tool_format), False

    sys_msgs = [m for m in messages if m.get("role") == "system"]
    non_sys = [m for m in messages if m.get("role") != "system"]
    if not sys_msgs:
        return "", render_chat_prompt(messages, template, tool_format), False

    sc = sys_msgs[0].get("content", "")
    if template == "gemma4":
        # gemma4 keeps system as its own turn, so unlike Gemma 2/3 it splits.
        prefix = f"<bos><|turn>system\n{sc}<turn|>\n"
        remaining = render_chat_prompt(non_sys, template, tool_format)[len("<bos>"):]
        return prefix, remaining, True
    if template == "llama3":
        prefix = (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
                  f"\n\n{sc}<|eot_id|>")
        remaining = render_chat_prompt(non_sys, template, tool_format)[len("<|begin_of_text|>"):]
    else:  # chatml
        prefix = f"<|im_start|>system\n{sc}<|im_end|>\n"
        remaining = render_chat_prompt(non_sys, template, tool_format)
    return prefix, remaining, True
