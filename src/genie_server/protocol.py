"""OpenAI wire-format builders and error shapes.

Every error this server returns — including raised HTTPExceptions and
malformed request bodies — is rendered in OpenAI's error envelope
{"error": {"message", "type", "param", "code"}} so OpenAI SDKs, lm_eval and
LiteLLM can always parse failures (install_error_handlers)."""

import json
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .slots import SlotNotLoadedError, UnknownSlotError


def openai_error(status_code: int, message: str,
                 error_type: str = "server_error",
                 param: str | None = None,
                 code: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type,
                           "param": param, "code": code}},
    )


class InvalidRequestError(Exception):
    """Client-side request problem, rendered as a 400 invalid_request_error."""

    def __init__(self, message: str, param: str | None = None,
                 status_code: int = 400, code: str | None = None):
        super().__init__(message)
        self.param = param
        self.status_code = status_code
        self.code = code   # OpenAI's machine-readable code, e.g. context_length_exceeded


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(InvalidRequestError)
    async def _invalid_request(request: Request, exc: InvalidRequestError):
        return openai_error(exc.status_code, str(exc), "invalid_request_error",
                            exc.param, exc.code)

    @app.exception_handler(UnknownSlotError)
    async def _unknown_slot(request: Request, exc: UnknownSlotError):
        return openai_error(404, str(exc), "invalid_request_error", "slot")

    @app.exception_handler(SlotNotLoadedError)
    async def _slot_not_loaded(request: Request, exc: SlotNotLoadedError):
        return openai_error(503, str(exc), "server_error", code="model_not_loaded")

    @app.exception_handler(HTTPException)
    async def _http_exception(request: Request, exc: HTTPException):
        error_type = "invalid_request_error" if exc.status_code < 500 else "server_error"
        return openai_error(exc.status_code, str(exc.detail), error_type)

    @app.exception_handler(json.JSONDecodeError)
    async def _bad_json(request: Request, exc: json.JSONDecodeError):
        return openai_error(400, f"Request body is not valid JSON: {exc}",
                            "invalid_request_error")

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        return openai_error(400, str(exc), "invalid_request_error")


async def read_json_body(request: Request) -> dict:
    """Parses the request body, mapping malformed JSON to a clean 400."""
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        raise InvalidRequestError(f"Request body is not valid JSON: {e}") from e
    if not isinstance(body, dict):
        raise InvalidRequestError("Request body must be a JSON object")
    return body


# ---------------------------------------------------------------- SSE / chunks

def sse(data: dict | str) -> str:
    if isinstance(data, str):
        return f"data: {data}\n\n"
    return f"data: {json.dumps(data)}\n\n"


def usage_dict(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def model_object(model_id: str) -> dict:
    # `root`, `parent` and `permission` are legacy OpenAI model fields. They
    # look like dead weight, but Open WebUI 0.11.0 will not let you send a
    # message to a model whose entry lacks them: the model appears in the
    # picker and the frontend then silently refuses to POST anything. A
    # textbook OpenAI stub carrying these fields works with the same build,
    # and 0.6.18 works either way — see docs/MANUAL.md ("Open WebUI").
    now = int(time.time())
    return {"id": model_id, "object": "model", "created": now,
            "owned_by": "local", "root": model_id, "parent": None,
            "permission": [{
                "id": f"modelperm-{model_id}", "object": "model_permission",
                "created": now, "allow_create_engine": False,
                "allow_sampling": True, "allow_logprobs": True,
                "allow_search_indices": False, "allow_view": True,
                "allow_fine_tuning": False, "organization": "*",
                "group": None, "is_blocking": False}]}


# -- chat.completion --

def chat_chunk(request_id: str, model: str, delta: dict,
               finish_reason: str | None = None) -> dict:
    return {
        "id": request_id, "object": "chat.completion.chunk",
        "created": int(time.time()), "model": model,
        # Some clients read system_fingerprint off the chunk rather than the
        # final response; OpenAI sends it on both, so we do too.
        "system_fingerprint": None,
        "choices": [{"delta": delta, "index": 0, "finish_reason": finish_reason}],
    }


def chat_role_chunk(request_id: str, model: str) -> dict:
    """OpenAI streaming spec: the FIRST chunk carries delta.role="assistant"
    with empty content, before any text deltas. Many client SDKs rely on
    this to initialize the message object."""
    return chat_chunk(request_id, model, {"role": "assistant", "content": ""})


def chat_usage_chunk(request_id: str, model: str, prompt_tokens: int,
                     completion_tokens: int,
                     object_type: str = "chat.completion.chunk") -> dict:
    """stream_options.include_usage: a usage-only chunk just before [DONE]."""
    return {
        "id": request_id, "object": object_type,
        "created": int(time.time()), "model": model,
        "system_fingerprint": None,
        "choices": [],
        "usage": usage_dict(prompt_tokens, completion_tokens),
    }


def chat_response(request_id: str, model: str, content: str, finish_reason: str,
                  prompt_tokens: int, completion_tokens: int,
                  tool_calls: list | None = None,
                  logprobs: dict | None = None) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
        if not content:
            message["content"] = None
    return {
        "id": request_id, "object": "chat.completion",
        "created": int(time.time()), "model": model,
        "system_fingerprint": None,
        "choices": [{
            "message": message,
            "index": 0, "logprobs": logprobs,
            "finish_reason": finish_reason,
        }],
        "usage": usage_dict(prompt_tokens, completion_tokens),
    }


def tool_calls_chunk(request_id: str, model: str, tool_calls: list) -> dict:
    """Streaming delta carrying complete tool calls (emitted once, after the
    held-back <tool_call> blocks have been parsed)."""
    deltas = [{
        "index": i,
        "id": tc["id"],
        "type": "function",
        "function": {"name": tc["function"]["name"],
                     "arguments": tc["function"]["arguments"]},
    } for i, tc in enumerate(tool_calls)]
    return chat_chunk(request_id, model, {"tool_calls": deltas})


# -- text_completion --

def completion_chunk(request_id: str, model: str, text: str,
                     finish_reason: str | None = None, index: int = 0) -> dict:
    return {
        "id": request_id, "object": "text_completion",
        "created": int(time.time()), "model": model,
        "choices": [{"text": text, "index": index, "logprobs": None,
                     "finish_reason": finish_reason}],
    }


def completion_response(request_id: str, model: str, choices: list,
                        prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "id": request_id, "object": "text_completion",
        "created": int(time.time()), "model": model,
        "choices": choices,
        "usage": usage_dict(prompt_tokens, completion_tokens),
    }


# ---------------------------------------------------------------- profiling

# GenieProfile reports each KPI as {"value": ..., "unit": ...} nested under a
# per-component, per-stat tree (Profile.cpp getProfilingStat). These are the
# few numbers a caller actually wants, flattened and in familiar units.
_PROFILE_FIELDS = {
    "time-to-first-token": ("ttft_ms", 1e-3),          # microseconds
    "prompt-processing-rate": ("prefill_tokens_per_s", 1.0),
    "token-generation-rate": ("decode_tokens_per_s", 1.0),
    "token-generation-time": ("generation_ms", 1e-3),  # microseconds
    "num-prompt-tokens": ("prompt_tokens", 1.0),
    "num-generated-tokens": ("generated_tokens", 1.0),
    # NOTE: this is the shared-engine state path (qualla/dialog.cpp:2653),
    # NOT GenieDialog_restore — the prefix cache's cost is in host_measured.
    "apply-engine-state-time": ("apply_engine_state_ms", 1e-3),
    "lora-adapter-switch-time": ("lora_switch_ms", 1e-3),
}


def profile_summary(raw) -> dict:
    """Pull the dialog KPIs out of GenieProfile's JSON, whatever depth the
    SDK nests them at. Unknown or missing fields are simply absent."""
    out: dict = {}

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                mapped = _PROFILE_FIELDS.get(key)
                if mapped and isinstance(value, dict) and "value" in value:
                    name, scale = mapped
                    try:
                        out[name] = round(float(value["value"]) * scale, 4)
                    except (TypeError, ValueError):
                        pass
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(raw)
    return out
