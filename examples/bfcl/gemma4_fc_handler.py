"""Register a native function-calling (FC) model id for gemma4 with BFCL.

gemma4 declares and calls tools with its own tokens, not with the Hermes
`<tool_call>` JSON that BFCL's other OSS handlers use:

    declaration  <|tool>declaration:NAME{description:<|"|>...<|"|>,
                 parameters:{properties:{...},required:[...],type:<|"|>OBJECT<|"|>}}<tool|>
    call         <|tool_call>call:NAME{arg:value,arg2:value2}<tool_call|>
    string arg   key:<|"|>value<|"|>
    reasoning    <|channel>thought...<channel|>

Declarations live inside the system turn. Argument keys are emitted in
`dictsort` order.

Rendering follows Google's own template, mirrored from llama.cpp's
`models/templates/google-gemma-4-31B-it-interleaved.jinja` (macros
`format_function_declaration`, `format_parameters`, `format_argument`).

Without this, gemma4 can only be evaluated through BFCL's *prompting* mode,
which asks for `[func(param=value)]` and scores a model that answers in its own
native tokens as a syntax error on every entry.

Install into a bfcl-eval venv with `install_gemma4_handler.sh`, which registers
both this and the prompting-mode handler in `gemma4_handler.py`.
"""
import re
from typing import Any

from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import convert_to_function_call
from overrides import override

Q = '<|"|>'                         # gemma4's string delimiter (id 52)
STANDARD_KEYS = ("description", "type", "properties", "required", "nullable")


# ----------------------------------------------------------------- rendering

def _fmt_arg(value, escape_keys=True) -> str:
    """format_argument()."""
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
    return str(value)


def _fmt_params(properties: dict, required) -> str:
    """format_parameters(). One entry per property, dictsort order."""
    out = []
    for key, value in sorted((properties or {}).items()):
        if key in STANDARD_KEYS:
            continue
        if not isinstance(value, dict):
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
                    items.append("required:[" + ",".join(f"{Q}{r}{Q}" for r in iv) + "]")
                elif ik == "type":
                    items.append("type:" + (_fmt_arg(iv.upper()) if isinstance(iv, str)
                                            else _fmt_arg([str(x).upper() for x in iv])))
                else:
                    items.append(f"{ik}:{_fmt_arg(iv)}")
            body.append("items:{" + ",".join(items) + "}")

        body.append(f"type:{Q}{vtype}{Q}")
        out.append(f"{key}:{{" + ",".join(body) + "}")
    return ",".join(out)


def _normalise_types(node):
    """BFCL's schemas use Gorilla's type names ('dict', 'float', 'tuple'...).
    gemma4's declarations are Gemini-flavoured, where the object type is
    OBJECT and the number type is NUMBER, so map them before uppercasing.
    Without this a parameter object goes out as type:<|"|>DICT<|"|>, which is
    not a type the model was trained on."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "type" and isinstance(v, str):
                out[k] = GORILLA_TO_OPENAPI.get(v, v)
            else:
                out[k] = _normalise_types(v)
        return out
    if isinstance(node, list):
        return [_normalise_types(v) for v in node]
    return node


def _fmt_declaration(tool: dict) -> str:
    """format_function_declaration(). Accepts a bare function dict too."""
    fn = _normalise_types(tool.get("function", tool))
    out = f"declaration:{fn.get('name', '')}{{description:{Q}{fn.get('description', '')}{Q}"

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

    if "response" in fn:
        resp = fn["response"] or {}
        inner = []
        if resp.get("description"):
            inner.append(f"description:{Q}{resp['description']}{Q}")
        if str(resp.get("type", "")).upper() == "OBJECT":
            inner.append(f"type:{Q}OBJECT{Q}")
        out += ",response:{" + ",".join(inner) + "}"

    return out + "}"


# ------------------------------------------------------------------ parsing

CALL = re.compile(r"<\|tool_call>call:([A-Za-z_][\w.]*)\{(.*?)\}?<tool_call\|>", re.DOTALL)


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
            parts.append(body[start:i]); start = i + 1
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


def extract_tool_calls(text: str) -> list[dict]:
    """[{"name": ..., "arguments": {...}}] out of a gemma4 reply."""
    calls = []
    for name, body in CALL.findall(text or ""):
        args = {}
        for part in _split_top(body):
            k, v = _parse_pair(part)
            if k:
                args[k] = v
        calls.append({"name": name, "arguments": args})
    return calls


def strip_reasoning(text: str) -> tuple[str, str]:
    """(clean, reasoning) — pulls out <|channel>thought...<channel|>."""
    m = re.search(r"<\|channel>(.*?)<channel\|>", text or "", re.DOTALL)
    if not m:
        return text, ""
    return (text[:m.start()] + text[m.end():]), m.group(1).strip()


# ------------------------------------------------------------------ handler

class Gemma4FCHandler(OSSHandler):
    """gemma4 with its native tool tokens."""

    def __init__(self, model_name, temperature, registry_name, is_fc_model,
                 dtype="bfloat16", **kwargs) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)

    @override
    def _format_prompt(self, messages, function):
        system = ""
        rest = list(messages)
        if rest and rest[0].get("role") in ("system", "developer"):
            system = str(rest[0].get("content", "")).strip()
            rest = rest[1:]

        out = "<bos>"
        if system or function:
            out += "<|turn>system\n" + system
            for tool in (function or []):
                out += "<|tool>" + _fmt_declaration(tool).strip() + "<tool|>"
            out += "<turn|>\n"

        for m in rest:
            role = "model" if m.get("role") == "assistant" else m.get("role", "user")
            out += f"<|turn>{role}\n{str(m.get('content', '')).strip()}<turn|>\n"

        return out + "<|turn>model\n"

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        calls = extract_tool_calls(result)
        if not calls:
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return [{c["name"]: c["arguments"]} for c in calls]

    @override
    def decode_execute(self, result, has_tool_call_tag):
        calls = extract_tool_calls(result)
        if not calls:
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return convert_to_function_call([{c["name"]: c["arguments"]} for c in calls])

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        # An FC model brings its own declarations; no system prompt is injected.
        return {"message": [], "function": test_entry["function"]}

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        raw = api_response.choices[0].text
        cleaned, reasoning = strip_reasoning(raw)
        calls = extract_tool_calls(raw)
        if calls:
            history = {"role": "assistant", "content": "",
                       "tool_calls": calls}
        else:
            history = {"role": "assistant", "content": cleaned}
        history["reasoning_content"] = reasoning
        return {
            "model_responses": raw,
            "reasoning_content": reasoning,
            "model_responses_message_for_chat_history": history,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def _add_assistant_message_prompting(self, inference_data: dict,
                                         model_response_data: dict) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"])
        return inference_data
