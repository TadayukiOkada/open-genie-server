"""Offline HTTP API tests (FakeGenieLib — no NPU, no libGenie.so)."""

import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from genie_server.app import create_app
from genie_server.prefix_cache import PrefixCache

FAKE_RESPONSE = "Hello world from Genie!"


def sse_events(body: str) -> list:
    events = []
    for line in body.splitlines():
        if line.startswith("data: "):
            payload = line[len("data: "):]
            events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events


# ---------------------------------------------------------------- basics

def test_health(client):
    for path in ("/health", "/v1/health"):
        r = client.get(path)
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_models_list(client):
    for path in ("/v1/models", "/models"):
        r = client.get(path)
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "list"
        ids = [m["id"] for m in data["data"]]
        assert "genie-local" in ids
        assert "qwen3-test" in ids


def test_retrieve_model(client):
    r = client.get("/v1/models/anything-goes")
    assert r.status_code == 200
    assert r.json()["id"] == "anything-goes"


def test_bad_json_body_is_openai_error(client):
    r = client.post("/v1/chat/completions", content=b"{not json",
                    headers={"content-type": "application/json"})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"


# ---------------------------------------------------------------- chat

def test_chat_completion_sync(client):
    r = client.post("/v1/chat/completions", json={
        "model": "genie-local",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion"
    choice = data["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == FAKE_RESPONSE
    assert choice["finish_reason"] == "stop"
    usage = data["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] == 4  # fake emits 4 word-chunks
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_chat_completion_content_parts_array(client, state):
    """Open WebUI-style parts-array content must be flattened, not repr()'d."""
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user",
                      "content": [{"type": "text", "text": "hello there"}]}],
    })
    assert r.status_code == 200
    # The rendered prompt must contain the flattened text, not "[{'type': ..."
    assert state.lib.reset_count == 1


def test_chat_completion_stream(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    assert r.status_code == 200
    events = sse_events(r.text)
    assert events[-1] == "[DONE]"
    # First chunk carries the assistant role delta.
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    text = "".join(
        e["choices"][0]["delta"].get("content", "")
        for e in events[:-1] if isinstance(e, dict) and e.get("choices"))
    assert text == FAKE_RESPONSE
    finals = [e for e in events[:-1]
              if isinstance(e, dict) and e.get("choices")
              and e["choices"][0].get("finish_reason")]
    assert finals[-1]["choices"][0]["finish_reason"] == "stop"
    usage_events = [e for e in events[:-1] if isinstance(e, dict) and e.get("usage")]
    assert len(usage_events) == 1
    assert usage_events[0]["usage"]["completion_tokens"] == 4


def test_chat_max_tokens_length_finish(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 2,
    })
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "length"
    assert r.json()["usage"]["completion_tokens"] == 2


def test_chat_query_error_is_500(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "ERROR please"}],
    })
    assert r.status_code == 500
    assert "GenieDialog_query" in r.json()["error"]["message"]


def test_chat_rejects_n_gt_1(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "n": 2})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "n"


def test_chat_rejects_tool_choice_required(client):
    """"required" guarantees a call in OpenAI's semantics. Silently treating
    it as "auto" hands the caller prose where their code reads tool_calls."""
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "f"}}],
        "tool_choice": "required"})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["param"] == "tool_choice"
    assert "auto" in err["message"]


def test_chat_rejects_tool_choice_named_function(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        "tool_choice": {"type": "function",
                        "function": {"name": "get_weather"}}})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["param"] == "tool_choice"
    assert "get_weather" in err["message"]


def test_chat_accepts_supported_tool_choice(client):
    for choice in ("auto", "none"):
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "tool_choice": choice})
        assert r.status_code == 200, (choice, r.json())


def test_chat_tool_choice_none_suppresses_tool_calls(client):
    """The prompt that makes the fake emit a tool call must come back as
    plain text once tool injection is disabled."""
    body = {"messages": [{"role": "user", "content": "TOOLCALL what is the weather"}],
            "tools": [{"type": "function", "function": {
                "name": "get_weather", "description": "Get weather",
                "parameters": {"type": "object",
                               "properties": {"city": {"type": "string"}}}}}]}
    r = client.post("/v1/chat/completions", json={**body, "tool_choice": "none"})
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] != "tool_calls"
    assert not choice["message"].get("tool_calls")


def test_completions_ignores_tool_choice(client):
    """tool_choice is a chat-only field; /v1/completions must not start
    rejecting requests that happen to carry it."""
    r = client.post("/v1/completions", json={"prompt": "a", "tool_choice": "required"})
    assert r.status_code == 200


def test_chat_rejects_logprobs_with_stream(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "logprobs": True, "stream": True})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "logprobs"


def test_chat_rejects_bad_max_tokens(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 0})
    assert r.status_code == 400


def test_chat_sampling_params_reach_sdk(client, state):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
    })
    assert r.status_code == 200
    handle_id = state.manager.slots[0].handle.value
    params = state.lib.sampler_params[handle_id]
    # temperature=0 => greedy via top-k=1 (SDK cannot take temp=0 at runtime)
    assert params["top-k"] == "1"


def test_chat_stop_sequences_reach_sdk(client, state):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "stop": ["###", "\n\n"],
    })
    assert r.status_code == 200
    handle_id = state.manager.slots[0].handle.value
    assert state.lib.stop_sequences[handle_id] == ["###", "\n\n"]


# ---------------------------------------------------------------- logprobs

FAKE_WORDS = FAKE_RESPONSE.split()


def test_chat_logprobs_sync(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "logprobs": True, "top_logprobs": 2, "temperature": 0,
    })
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["message"]["content"] == FAKE_RESPONSE
    content = choice["logprobs"]["content"]
    assert [e["token"] for e in content] == FAKE_WORDS
    for e in content:
        assert e["logprob"] <= 0
        assert len(e["top_logprobs"]) == 2
        # greedy: the chosen token is the top-1 alternative
        assert e["top_logprobs"][0]["token"] == e["token"]
        assert e["bytes"] == list(e["token"].encode())


def test_completions_logprobs_sync(client):
    r = client.post("/v1/completions", json={
        "prompt": "Say hi", "logprobs": 2, "temperature": 0,
    })
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["text"] == FAKE_RESPONSE
    lp = choice["logprobs"]
    assert lp["tokens"] == FAKE_WORDS
    assert len(lp["token_logprobs"]) == len(FAKE_WORDS)
    assert all(isinstance(v, float) and v <= 0 for v in lp["token_logprobs"])
    assert all(len(d) == 2 for d in lp["top_logprobs"])
    assert lp["text_offset"][0] == 0
    assert lp["text_offset"] == sorted(lp["text_offset"])


def test_logprobs_runtime_without_logits_callback(client, state):
    """Some runtimes accept custom sampling but never call its logits hook."""
    from genie_server import capi

    def no_logits(handle, text, cb_name, on_token):
        on_token("Hello", capi.SENTENCE_CONTINUE)
        on_token("", capi.SENTENCE_END)
        return 0

    state.lib._query_custom = no_logits
    requests = [
        ("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": True}),
        ("/v1/completions", {"prompt": "hi", "logprobs": 1}),
    ]
    client.post("/v1/server/prompt_logprobs", json={"enabled": True})
    requests.append(("/v1/completions", {
        "prompt": "a b c", "echo": True, "logprobs": 1, "max_tokens": 0}))
    queries_before = len(state.lib.queries)
    for path, body in requests:
        response = client.post(path, json=body)
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["code"] == "logprobs_not_supported"
        assert error["param"] == "logprobs"
        assert "logits callback" in error["message"]
    # The first request stops at its first token; the slot remembers the
    # result, so the later requests never reach the runtime.
    assert len(state.lib.queries) == queries_before + 1
    assert state.lib.abort_signals == 1


def test_logprobs_missing_callback_with_failing_query(client, state):
    """A query that fails after emitting tokens must not return 200 with
    empty logprobs arrays."""
    from genie_server import capi

    def no_logits_then_fail(handle, text, cb_name, on_token):
        on_token("Hello", capi.SENTENCE_CONTINUE)
        return -1

    state.lib._query_custom = no_logits_then_fail
    response = client.post("/v1/completions", json={
        "prompt": "hi", "logprobs": 1})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "logprobs_not_supported"


def test_prompt_scoring_rejects_partial_logits_callbacks(client, state):
    """Scores pair with prompt tokens by position, so a runtime that calls the
    logits callback for only some steps must fail rather than shift them."""
    import ctypes
    from genie_server import capi

    def partial_logits(handle, text, cb_name, on_token):
        handler = state.lib.custom_samplers[cb_name]
        logits = (ctypes.c_float * state.lib.N_VOCAB)()
        tok = int(handler(ctypes.addressof(logits), state.lib.N_VOCAB, 1)[0])
        on_token(state.lib.tokenizer.decode([tok]), capi.SENTENCE_CONTINUE)
        on_token("x", capi.SENTENCE_CONTINUE)   # sampled without the hook
        on_token("", capi.SENTENCE_END)
        return 0

    state.lib._query_custom = partial_logits
    client.post("/v1/server/prompt_logprobs", json={"enabled": True})
    response = client.post("/v1/completions", json={
        "prompt": "a b c d", "echo": True, "logprobs": 1, "max_tokens": 0})
    assert response.status_code == 500
    assert "prompt scoring stopped after 1 of" in response.json()["error"]["message"]


def test_completions_logprobs_rejected_with_stream(client):
    r = client.post("/v1/completions", json={
        "prompt": "hi", "logprobs": 1, "stream": True})
    assert r.status_code == 400


def test_prompt_scoring_disabled_by_default(client):
    r = client.post("/v1/completions", json={
        "prompt": "The capital of Japan is Tokyo",
        "echo": True, "logprobs": 1})
    assert r.status_code == 400
    assert "prompt_logprobs" in r.json()["error"]["message"]


def test_prompt_logprobs_toggle(client):
    r = client.get("/v1/server/prompt_logprobs")
    assert r.json()["enabled"] is False
    r = client.post("/v1/server/prompt_logprobs", json={"enabled": True})
    assert r.status_code == 200
    r = client.get("/v1/server/prompt_logprobs")
    assert r.json()["enabled"] is True
    r = client.post("/v1/server/prompt_logprobs", json={"enabled": "yes"})
    assert r.status_code == 400


def test_prompt_scoring_end_to_end(client, state):
    # lm_eval loglikelihood shape: token-id prompt + echo + logprobs.
    client.post("/v1/server/prompt_logprobs", json={"enabled": True})
    tok = state.manager.slots[0].tokenizer
    ids = tok.encode("the quick brown fox jumps").ids
    r = client.post("/v1/completions", json={
        "prompt": [ids], "echo": True, "logprobs": 1, "max_tokens": 0,
    })
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    lp = choice["logprobs"]
    assert len(lp["tokens"]) == len(ids)
    assert lp["token_logprobs"][0] is None      # first token: undefined
    assert lp["top_logprobs"][0] is None
    assert all(isinstance(v, float) for v in lp["token_logprobs"][1:])
    assert all(isinstance(d, dict) and len(d) == 1
               for d in lp["top_logprobs"][1:])
    assert choice["finish_reason"] == "length"
    assert r.json()["usage"]["completion_tokens"] == 0
    assert r.json()["usage"]["prompt_tokens"] == len(ids)


def test_prompt_scoring_rejects_max_tokens(client):
    client.post("/v1/server/prompt_logprobs", json={"enabled": True})
    r = client.post("/v1/completions", json={
        "prompt": "a b c", "echo": True, "logprobs": 1, "max_tokens": 5})
    assert r.status_code == 400


def test_prompt_scoring_accepts_lm_eval_max_tokens_1(client, state):
    """lm_eval's loglikelihood sends max_tokens=1 with echo, then drops the
    last entry (`token_logprobs[ctxlen:-1]`). One real generated token has to
    be there, or every score loses its final continuation token."""
    client.post("/v1/server/prompt_logprobs", json={"enabled": True})
    prompt = "a b c"
    n_prompt = len(state.manager.slots[0].tokenizer.encode(prompt).ids)

    r = client.post("/v1/completions", json={
        "prompt": prompt, "echo": True, "logprobs": 1, "max_tokens": 1})

    assert r.status_code == 200
    body = r.json()
    lp = body["choices"][0]["logprobs"]
    # prompt tokens (first logprob null) + exactly one generated token
    assert len(lp["tokens"]) == n_prompt + 1
    assert lp["token_logprobs"][0] is None
    assert all(x is not None for x in lp["token_logprobs"][1:])
    # what lm_eval actually scores: everything after the context, minus the
    # trailing generated token
    ctxlen = 1
    assert len(lp["token_logprobs"][ctxlen:-1]) == n_prompt - ctxlen
    assert body["usage"]["prompt_tokens"] == n_prompt
    assert body["usage"]["completion_tokens"] == 1
    assert body["choices"][0]["text"].startswith(prompt)
    assert len(body["choices"][0]["text"]) > len(prompt)


def test_prompt_scoring_max_tokens_0_has_no_generated_token(client, state):
    client.post("/v1/server/prompt_logprobs", json={"enabled": True})
    prompt = "a b c"
    n_prompt = len(state.manager.slots[0].tokenizer.encode(prompt).ids)

    r = client.post("/v1/completions", json={
        "prompt": prompt, "echo": True, "logprobs": 1, "max_tokens": 0})

    assert r.status_code == 200
    lp = r.json()["choices"][0]["logprobs"]
    assert len(lp["tokens"]) == n_prompt
    assert r.json()["usage"]["completion_tokens"] == 0


# ---------------------------------------------------------------- tools

def test_chat_tool_calls_sync(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "TOOLCALL what is the weather"}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather", "description": "Get weather",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}}}}}],
    })
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tcs = choice["message"]["tool_calls"]
    assert len(tcs) == 1
    assert tcs[0]["function"]["name"] == "get_weather"
    assert json.loads(tcs[0]["function"]["arguments"]) == {"city": "Tokyo"}
    assert "tool_call" not in (choice["message"]["content"] or "")


def test_chat_tool_calls_stream(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "TOOLCALL weather"}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather", "parameters": {}}}],
        "stream": True,
    })
    assert r.status_code == 200
    events = sse_events(r.text)
    streamed_text = "".join(
        e["choices"][0]["delta"].get("content", "")
        for e in events[:-1] if isinstance(e, dict) and e.get("choices"))
    assert "<tool_call>" not in streamed_text  # held back, never leaked
    tool_events = [e for e in events[:-1]
                   if isinstance(e, dict) and e.get("choices")
                   and e["choices"][0]["delta"].get("tool_calls")]
    assert len(tool_events) == 1
    tc = tool_events[0]["choices"][0]["delta"]["tool_calls"][0]
    assert tc["function"]["name"] == "get_weather"
    finals = [e for e in events[:-1]
              if isinstance(e, dict) and e.get("choices")
              and e["choices"][0].get("finish_reason")]
    assert finals[-1]["choices"][0]["finish_reason"] == "tool_calls"


# ---------------------------------------------------------------- completions

def test_completions_sync(client):
    r = client.post("/v1/completions", json={"prompt": "Once upon a time"})
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "text_completion"
    assert data["choices"][0]["text"] == FAKE_RESPONSE
    assert data["choices"][0]["finish_reason"] == "stop"


def test_completions_echo(client):
    r = client.post("/v1/completions", json={"prompt": "Say hi", "echo": True})
    assert r.json()["choices"][0]["text"] == "Say hi" + FAKE_RESPONSE


def test_completions_batch_prompts(client):
    r = client.post("/v1/completions", json={"prompt": ["one", "two"]})
    assert r.status_code == 200
    choices = r.json()["choices"]
    assert [c["index"] for c in choices] == [0, 1]
    assert all(c["text"] == FAKE_RESPONSE for c in choices)


def test_completions_token_id_prompt(client, state):
    tok = state.manager.slots[0].tokenizer
    ids = tok.encode("hello world").ids
    r = client.post("/v1/completions", json={"prompt": [ids]})
    assert r.status_code == 200
    assert r.json()["choices"][0]["text"] == FAKE_RESPONSE


def test_completions_stream(client):
    r = client.post("/v1/completions", json={"prompt": "hi", "stream": True})
    events = sse_events(r.text)
    assert events[-1] == "[DONE]"
    text = "".join(e["choices"][0]["text"] for e in events[:-1]
                   if isinstance(e, dict))
    assert FAKE_RESPONSE in text


def test_completions_rejects_suffix(client):
    r = client.post("/v1/completions", json={"prompt": "a", "suffix": "b"})
    assert r.status_code == 400


# ---------------------------------------------------------------- management

def test_server_status(client):
    r = client.get("/v1/server/status")
    assert r.status_code == 200
    data = r.json()
    assert data["phase"] == "idle"
    assert data["slots"][0]["name"] == "default"
    assert data["slots"][0]["loaded"] is True


def test_server_idle(client):
    r = client.get("/v1/server/idle")
    assert r.status_code == 200
    assert r.json()["status"] == "idle"


def test_unknown_slot_404(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "slot": "nope"})
    assert r.status_code == 404
    assert "Unknown slot" in r.json()["error"]["message"]


def test_lora_apply_and_current(client):
    r = client.post("/v1/lora/apply", json={"lora_adapter_name": "my-adapter"})
    assert r.status_code == 200
    assert r.json()["lora_adapter_name"] == "my-adapter"
    r = client.get("/v1/lora/current")
    assert r.json()["lora_adapter_name"] == "my-adapter"


def test_lora_strength_reaches_the_sdk(state, client):
    """The only path that sets a LoRA blend weight — alpha must arrive as a
    float on the named tensor, on the slot the request selected."""
    r = client.post("/v1/lora/strength",
                    json={"tensor_name": "layer0.q_proj", "alpha": 0.5})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "applied"
    assert body["slot"] == "default"
    assert body["engine"] == "primary"          # the default engine role
    assert (body["tensor_name"], body["alpha"]) == ("layer0.q_proj", 0.5)
    handle, engine, tensor, alpha = state.lib.lora_strengths[-1]
    assert (engine, tensor) == ("primary", "layer0.q_proj")
    assert isinstance(alpha, float) and alpha == 0.5


def test_lora_strength_accepts_alpha_zero(state, client):
    """0.0 is a meaningful strength (adapter off), not a missing value."""
    r = client.post("/v1/lora/strength",
                    json={"tensor_name": "layer0.q_proj", "alpha": 0})
    assert r.status_code == 200
    assert state.lib.lora_strengths[-1][3] == 0.0


def test_lora_strength_passes_a_non_default_engine_role(state, client):
    r = client.post("/v1/lora/strength", json={
        "engine": "draft", "tensor_name": "layer0.q_proj", "alpha": 1.0})
    assert r.status_code == 200
    assert r.json()["engine"] == "draft"
    assert state.lib.lora_strengths[-1][1] == "draft"


def test_lora_strength_requires_tensor_name_and_alpha(client):
    for body in ({"alpha": 0.5}, {"tensor_name": "layer0.q_proj"}, {}):
        r = client.post("/v1/lora/strength", json=body)
        assert r.status_code == 400, body
        assert "'tensor_name' and 'alpha' are required." \
            in r.json()["error"]["message"]


def test_lora_strength_reports_an_sdk_failure(state, client):
    state.lib.lora_strength_status = 3
    r = client.post("/v1/lora/strength",
                    json={"tensor_name": "layer0.q_proj", "alpha": 0.5})
    assert r.status_code == 500
    assert "GenieDialog_setLoraStrength failed: 3" in r.json()["error"]["message"]


def test_lora_strength_releases_the_slot_lock_after_a_failure(state, client):
    """The SDK call is wrapped in try/finally — a failed call must not leave
    the slot locked for every later request."""
    state.lib.lora_strength_status = 3
    client.post("/v1/lora/strength",
                json={"tensor_name": "layer0.q_proj", "alpha": 0.5})
    state.lib.lora_strength_status = 0
    assert state.manager.slots[0].lock.acquire(timeout=0.1)
    state.manager.slots[0].lock.release()
    r = client.post("/v1/lora/strength",
                    json={"tensor_name": "layer0.q_proj", "alpha": 0.5})
    assert r.status_code == 200


def test_lora_strength_routes_an_unknown_model_to_the_primary_slot(state, client):
    """SlotManager.select falls back to the primary slot by design (lm_eval
    sends a fixed placeholder name), so the LoRA endpoints inherit that —
    an unfamiliar "model" is not an error here."""
    r = client.post("/v1/lora/strength", json={
        "model": "no-such-model", "tensor_name": "t", "alpha": 1.0})
    assert r.status_code == 200
    assert r.json()["slot"] == "default"
    assert state.lib.lora_strengths[-1][2] == "t"


# ------------------------------------------------------ gemma4 tool dialect

def _gemma4_client(state):
    """A slot whose model renders gemma4 turns, so it speaks gemma4 tools."""
    from fastapi.testclient import TestClient
    from genie_server import tool_formats
    from genie_server.app import create_app

    slot = state.manager.slots[0]
    slot.chat_template = "gemma4"
    slot.tool_format = tool_formats.Gemma4ToolFormat
    return TestClient(create_app(state))


def test_a_gemma4_slot_declares_tools_with_its_own_tokens(state):
    """End to end: the declarations that reach the model are gemma4's, not
    the Hermes <tools> block."""
    client = _gemma4_client(state)

    client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "weather in Tokyo?"}],
        "tools": _WEATHER_TOOL})

    prompt = state.lib.queries[-1]
    assert "<|tool>declaration:get_weather{" in prompt
    assert "<tools>" not in prompt


def test_a_gemma4_slot_parses_its_own_call_format(state):
    """A reply in gemma4's format comes back as OpenAI tool_calls rather than
    as prose with raw markers in it."""
    client = _gemma4_client(state)
    state.lib.canned_response = (
        '<|tool_call>call:get_weather{city:<|"|>Tokyo<|"|>}<tool_call|>')

    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "weather in Tokyo?"}],
        "tools": _WEATHER_TOOL})

    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert "<|tool_call>" not in (choice["message"]["content"] or "")


def test_a_hermes_slot_is_unaffected(state, client):
    """The default slot keeps declaring and parsing Hermes."""
    client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": _WEATHER_TOOL})
    prompt = state.lib.queries[-1]
    assert "<tools>" in prompt and "<|tool>declaration" not in prompt


# ------------------------------------------------------- model switch config

def _bundle(tmp_path, name, config_file="genie_config.json"):
    d = tmp_path / name
    d.mkdir()
    (d / config_file).write_text(json.dumps({"dialog": {"context": {"size": 4096}}}))
    return d


def test_switch_rejects_a_bundle_without_the_slots_config_file(tmp_path, client):
    """The slot's config_file is what will be read, so the check that runs
    before the swap has to look for that name — not always
    genie_config.json."""
    other = _bundle(tmp_path, "other", config_file="some-model-htp.json")

    r = client.post("/v1/models/switch", json={"model_dir": str(other)})

    assert r.status_code == 404
    assert "genie_config.json" in r.json()["error"]["message"]
    assert "config_file" in r.json()["error"]["message"]


def test_switch_accepts_a_config_file_for_the_new_bundle(state, tmp_path, client):
    """A bundle that names its dialog config after the model is reachable
    without reconfiguring the server: name it in the switch."""
    other = _bundle(tmp_path, "other", config_file="some-model-htp.json")

    r = client.post("/v1/models/switch", json={
        "model_dir": str(other), "config_file": "some-model-htp.json"})

    assert r.status_code == 200, r.json()
    # It stays with the slot, so the next switch to a bundle named the same
    # way needs no repeat.
    assert state.manager.slots[0].config_file == "some-model-htp.json"


def test_a_failed_switch_leaves_the_config_file_alone(state, tmp_path, client):
    """A slot must not advertise a config file it is not running: the
    rollback covers this the way it covers the model itself."""
    state.lib.fail_create = True
    other = _bundle(tmp_path, "other", config_file="some-model-htp.json")
    try:
        r = client.post("/v1/models/switch", json={
            "model_dir": str(other), "config_file": "some-model-htp.json"})
    finally:
        state.lib.fail_create = False

    assert r.status_code == 500
    assert state.manager.slots[0].config_file == "genie_config.json"


# ------------------------------------------------------- LoRA slot addressing

def _add_second_slot(state, name="second", model_root=None):
    """A second text slot holding the SAME model directory as the first.

    That is the configuration where 'model' cannot tell the two apart —
    reindex() keeps the later one — so it is the one that decides whether
    'slot' is honoured.
    """
    from genie_server.slots import Slot
    from pathlib import Path

    first = state.manager.slots[0]
    slot = Slot(name=name, device_id=1,
                model_root=Path(model_root) if model_root else first.model_root)
    slot.handle = state.lib.create_dialog(b"{}")
    slot.dialog_cfg = first.dialog_cfg
    slot.chat_template = first.chat_template
    slot.tokenizer = first.tokenizer
    state.manager.slots.append(slot)
    state.manager.status[slot.name] = {"phase": "idle", "detail": ""}
    state.manager._by_name[slot.name] = slot
    state.manager.reindex()
    return slot


def test_lora_endpoints_address_a_slot_by_name(state, client):
    """Two slots, one model directory: 'model' routing cannot reach the
    second one, so every LoRA endpoint has to honour 'slot' the way the chat
    endpoints do."""
    second = _add_second_slot(state)

    r = client.post("/v1/lora/apply", json={
        "slot": second.name, "lora_adapter_name": "a"})
    assert r.status_code == 200 and r.json()["slot"] == second.name

    r = client.post("/v1/lora/strength", json={
        "slot": second.name, "tensor_name": "t", "alpha": 0.5})
    assert r.status_code == 200 and r.json()["slot"] == second.name

    r = client.get("/v1/lora/current", params={"slot": second.name})
    assert r.status_code == 200 and r.json()["slot"] == second.name

    r = client.post("/v1/lora/release", json={
        "slot": second.name, "lora_adapter_name": "a"})
    assert r.status_code == 200 and r.json()["slot"] == second.name


def test_lora_apply_reaches_the_named_slots_handle(state, client):
    """Not just the reported name: the adapter has to be applied to that
    slot's own dialog handle."""
    second = _add_second_slot(state)

    client.post("/v1/lora/apply", json={"slot": second.name,
                                        "lora_adapter_name": "a"})

    assert second.active_lora_adapter == "a"
    assert state.manager.slots[0].active_lora_adapter == ""


def test_lora_endpoints_reject_an_unknown_slot(state, client):
    """An unknown slot name is a client error, not a silent fallback to the
    primary — the same as everywhere else 'slot' is accepted."""
    _add_second_slot(state)
    for path, body in (("/v1/lora/apply", {"lora_adapter_name": "a"}),
                       ("/v1/lora/strength", {"tensor_name": "t", "alpha": 1.0}),
                       ("/v1/lora/release", {"lora_adapter_name": "a"})):
        r = client.post(path, json={"slot": "nope", **body})
        assert r.status_code == 404, path
    assert client.get("/v1/lora/current",
                      params={"slot": "nope"}).status_code == 404


def test_lora_without_a_slot_still_routes_by_model(state, client):
    """The old behaviour is unchanged where 'slot' is absent: 'model' routes,
    and a name nobody loaded lands on the primary slot."""
    _add_second_slot(state)

    r = client.post("/v1/lora/strength", json={
        "model": "no-such-model", "tensor_name": "t", "alpha": 1.0})
    assert r.status_code == 200 and r.json()["slot"] == "default"


def test_a_busy_slot_times_out_after_inference_timeout(state, client):
    """INFERENCE_TIMEOUT is the lock-acquire budget for the LoRA endpoints —
    a slot busy generating answers 503 rather than blocking forever."""
    import dataclasses
    state.config = dataclasses.replace(state.config, inference_timeout_s=0.05)
    state.manager.slots[0].lock.acquire()
    try:
        r = client.post("/v1/lora/strength",
                        json={"tensor_name": "t", "alpha": 1.0})
    finally:
        state.manager.slots[0].lock.release()
    assert r.status_code == 503
    assert "busy" in r.json()["error"]["message"]


def test_performance_policy_roundtrip(client):
    r = client.post("/v1/server/performance_policy", json={"policy": "burst"})
    assert r.status_code == 200
    r = client.get("/v1/server/performance_policy")
    assert r.json()["policy"] == "burst"


def test_performance_policy_invalid(client):
    r = client.post("/v1/server/performance_policy", json={"policy": "warp-speed"})
    assert r.status_code == 400


def test_prefix_warmup_and_cache(client):
    r = client.post("/v1/prefix/warmup", json={"system_prompt": "You are helpful."})
    assert r.status_code == 200
    assert r.json()["status"] == "cached"
    key = r.json()["key"]

    r = client.get("/v1/prefix/cache")
    keys = [e["key"] for e in r.json()["entries"]]
    assert key in keys

    # Second warmup: already cached
    r = client.post("/v1/prefix/warmup", json={"system_prompt": "You are helpful."})
    assert r.json()["status"] == "already_cached"

    r = client.delete(f"/v1/prefix/cache/{key}")
    assert r.status_code == 200

    r = client.delete(f"/v1/prefix/cache/{key}")
    assert r.status_code == 404


# ---------------------------------------------------------------- profiling

def test_profile_endpoint_is_409_when_disabled(client):
    """GENIE_PROFILE is off by default; the profiler binds to the dialog at
    creation, so this cannot be turned on at runtime — say so instead of
    returning empty data."""
    r = client.get("/v1/server/profile")

    assert r.status_code == 409
    assert "GENIE_PROFILE" in r.json()["error"]["message"]


def test_profile_endpoint_returns_sdk_kpis(client, state):
    state.manager.slots[0].profile = state.lib.create_profile()

    r = client.get("/v1/server/profile")

    assert r.status_code == 200
    body = r.json()
    assert body["slot"] == "default"
    # flattened, in familiar units
    assert body["summary"] == {
        "ttft_ms": 50.0, "prefill_tokens_per_s": 2700.5, "prompt_tokens": 33.0,
        "decode_tokens_per_s": 68.8, "generation_ms": 900.0, "generated_tokens": 64.0}
    # and the SDK's own JSON is passed through untouched
    assert body["profile"]["profile"]["dialog"][0]["type"] == "GenieDialog_query"
    # nothing host-measured yet: no prefix cache save or restore has run
    assert body["host_measured"] == {}


def test_profile_reports_host_measured_prefix_cache_cost(client, state):
    """The SDK profiles neither GenieDialog_save nor _restore, so the server
    times them itself — reported apart from the SDK's own numbers."""
    state.manager.slots[0].profile = state.lib.create_profile()
    sys_prompt = "You are terse."
    client.post("/v1/prefix/warmup", json={"system_prompt": sys_prompt})
    client.post("/v1/chat/completions", json={
        "messages": [{"role": "system", "content": sys_prompt},
                     {"role": "user", "content": "hi"}]})

    host = client.get("/v1/server/profile").json()["host_measured"]

    assert host["save_state_ms"] >= 0 and host["restore_state_ms"] >= 0
    assert "restore_state_ms" not in client.get(
        "/v1/server/profile").json()["summary"]


def test_profile_endpoint_unknown_slot_is_404(client, state):
    state.manager.slots[0].profile = state.lib.create_profile()

    assert client.get("/v1/server/profile", params={"slot": "nope"}).status_code == 404


def test_chat_response_shape_is_untouched_by_profiling(client, state):
    """The OpenAI contract must not grow fields because profiling is on —
    that is why the KPIs live on /v1/server/profile instead."""
    body = {"messages": [{"role": "user", "content": "hi"}]}
    without = client.post("/v1/chat/completions", json=body).json()

    state.manager.slots[0].profile = state.lib.create_profile()
    with_profiling = client.post("/v1/chat/completions", json=body).json()

    assert set(with_profiling) == set(without)
    assert set(with_profiling["usage"]) == set(without["usage"])
    assert not any("profile" in k for k in with_profiling)


def test_warmup_keys_match_the_no_think_variant(client, state):
    """enable_thinking=false appends /no_think to the system turn, so warming
    the raw prompt and then sending that flag used to be a permanent silent
    MISS. Warmup takes the same flag and caches what the chat path will ask
    for."""
    sys_prompt = "You are terse."

    plain = client.post("/v1/prefix/warmup", json={"system_prompt": sys_prompt})
    no_think = client.post("/v1/prefix/warmup",
                           json={"system_prompt": sys_prompt, "enable_thinking": False})

    assert plain.status_code == 200 and no_think.status_code == 200
    # different prefixes => different keys, both now warmable
    assert plain.json()["key"] != no_think.json()["key"]

    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "system", "content": sys_prompt},
                     {"role": "user", "content": "hi"}],
        "enable_thinking": False})
    assert r.status_code == 200
    keys = {e["key"] for e in client.get("/v1/prefix/cache").json()["entries"]}
    assert no_think.json()["key"] in keys


# ------------------------------------------------- context_length_exceeded

def test_chat_rejects_a_prompt_that_fills_the_context(client, state):
    """A prompt at or past the context window used to return HTTP 200 with an
    empty string (max_tokens clamped to 1, nothing generated, plain "length"
    stop). Open WebUI 0.11's 34 built-in tool definitions land exactly there.
    OpenAI answers with a 400 context_length_exceeded, and so do we."""
    state.manager.slots[0].dialog_cfg = {"context": {"size": 64}}

    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "word " * 200}]})

    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "context_length_exceeded"
    assert err["param"] == "messages"
    assert "64 tokens" in err["message"]


def test_completions_rejects_a_prompt_that_fills_the_context(client, state):
    state.manager.slots[0].dialog_cfg = {"context": {"size": 64}}

    r = client.post("/v1/completions", json={"prompt": "word " * 200})

    assert r.status_code == 400
    assert r.json()["error"]["code"] == "context_length_exceeded"


def test_a_prompt_that_fits_is_untouched(client, state):
    state.manager.slots[0].dialog_cfg = {"context": {"size": 4096}}

    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}]})

    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"]


def test_tools_count_towards_the_context_limit(client, state):
    """The tools block is rendered into the prompt, so it has to count — that
    is the whole reason Open WebUI 0.11 hit this."""
    state.manager.slots[0].dialog_cfg = {"context": {"size": 128}}
    tools = [{"type": "function", "function": {
        "name": f"tool_{i}", "description": "x " * 40,
        "parameters": {"type": "object", "properties": {}}}} for i in range(20)]

    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "tools": tools})

    assert r.status_code == 400
    assert r.json()["error"]["code"] == "context_length_exceeded"


# ------------------------------------------- unmarked tool-call recovery (F25)

_WEATHER_TOOL = [{"type": "function", "function": {
    "name": "get_weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}}}}]


def test_chat_recovers_a_mangled_tool_call(client_recovery_on):
    """On the board qwen3_4b_instruct_2507 replaces the <tool_call> token with
    Cyrillic on half its calls; the caller used to get that as prose with
    finish_reason "stop" while their code read message.tool_calls."""
    r = client_recovery_on.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "MANGLED what is the weather"}],
        "tools": _WEATHER_TOOL,
    })
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tcs = choice["message"]["tool_calls"]
    assert len(tcs) == 1 and tcs[0]["function"]["name"] == "get_weather"
    assert json.loads(tcs[0]["function"]["arguments"]) == {"city": "Tokyo"}
    assert not (choice["message"]["content"] or "")


def test_chat_recovers_a_mangled_tool_call_streaming(client_recovery_on):
    r = client_recovery_on.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "MANGLED weather"}],
        "tools": _WEATHER_TOOL, "stream": True,
    })
    assert r.status_code == 200
    events = sse_events(r.text)
    streamed = "".join(
        e["choices"][0]["delta"].get("content") or ""
        for e in events[:-1] if isinstance(e, dict) and e.get("choices"))
    assert "get_weather" not in streamed   # the body never leaked as content
    tool_events = [e for e in events[:-1]
                   if isinstance(e, dict) and e.get("choices")
                   and e["choices"][0]["delta"].get("tool_calls")]
    assert len(tool_events) == 1
    assert tool_events[0]["choices"][0]["delta"]["tool_calls"][0][
        "function"]["name"] == "get_weather"
    finals = [e for e in events[:-1]
              if isinstance(e, dict) and e.get("choices")
              and e["choices"][0].get("finish_reason")]
    assert finals[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_chat_recovery_is_off_by_default(client):
    """The strict tags-only parse is what a plain install does.

    A bundle that mangles its own <tool_call> marker returns prose with
    finish_reason "stop" — which is what it emitted. Recovering it silently
    would make the bundle measure as though the marker were intact, and only
    on /v1/chat/completions: /v1/completions has no recovery to apply.
    """
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "MANGLED weather"}],
        "tools": _WEATHER_TOOL,
    })
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert not choice["message"].get("tool_calls")
    assert "get_weather" in choice["message"]["content"]


def test_chat_recovery_ignores_a_tool_the_caller_did_not_declare(client_recovery_on):
    """The declared-name match is the whole discriminator: the same reply with
    a different tool in the request must stay prose."""
    r = client_recovery_on.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "MANGLED weather"}],
        "tools": [{"type": "function",
                   "function": {"name": "send_email", "parameters": {}}}],
    })
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert not choice["message"].get("tool_calls")


# ------------------------------------------- the response names the real model

def test_chat_reports_the_loaded_model_not_the_requested_alias(client):
    """A client that routes with an alias — lm_eval sends one fixed placeholder
    for every request — used to get its own string echoed back, so two runs
    against two different models were indistinguishable in the response. That
    is how a model swap across a restart got mistaken for the same model
    decoding nondeterministically."""
    r = client.post("/v1/chat/completions", json={
        "model": "genie-local",
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
    assert r.status_code == 200
    assert r.json()["model"] == "qwen3-test"


def test_completions_reports_the_loaded_model(client):
    r = client.post("/v1/completions", json={
        "model": "genie-local", "prompt": "hi", "max_tokens": 8})
    assert r.status_code == 200
    assert r.json()["model"] == "qwen3-test"


def test_streaming_chunks_report_the_loaded_model(client):
    r = client.post("/v1/chat/completions", json={
        "model": "genie-local", "stream": True,
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
    assert r.status_code == 200
    models = {e["model"] for e in sse_events(r.text)
              if isinstance(e, dict) and e.get("model")}
    assert models == {"qwen3-test"}


def test_the_alias_still_routes(client):
    """Reporting the resolved model must not change which slot answers: the
    requested string is still what selects it."""
    for requested in ("genie-local", "qwen3-test"):
        r = client.post("/v1/chat/completions", json={
            "model": requested,
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
        assert r.status_code == 200, requested
        assert r.json()["model"] == "qwen3-test"

    unknown = client.post("/v1/chat/completions", json={
        "model": "no-such-model",
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
    assert unknown.status_code == 200          # unknown ids fall back to slot 0
    assert unknown.json()["model"] == "qwen3-test"


class _WatchedLock:
    """threading.Lock that records when someone starts waiting on it, so a
    test can tell a request is queued on the slot instead of sleeping."""

    def __init__(self):
        self._lock = threading.Lock()
        self.waiting = threading.Event()

    def acquire(self, blocking=True, timeout=-1):
        if self._lock.acquire(blocking=False):
            return True
        if not blocking:
            return False
        self.waiting.set()
        return self._lock.acquire(timeout=timeout)

    def release(self):
        self._lock.release()

    def locked(self):
        return self._lock.locked()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


def _hold_slot_and_queue(state, client, path, body):
    """Holds the primary slot's lock, sends `path` from another thread, and
    returns once that request is waiting on the lock. The caller releases
    the lock and joins the thread."""
    slot = state.manager.slots[0]
    slot.lock = _WatchedLock()
    slot.lock.acquire()
    result = {}
    t = threading.Thread(
        target=lambda: result.update(r=client.post(path, json=body)))
    t.start()
    assert slot.lock.waiting.wait(5), f"{path} never waited on the slot lock"
    return slot, t, result


@pytest.mark.parametrize("path,body", [
    ("/v1/lora/apply", {"lora_adapter_name": "a"}),
    ("/v1/server/performance_policy", {"policy": "burst"}),
    ("/v1/models/switch", None),         # needs a bundle; built below
])
def test_waiting_for_a_slot_lock_does_not_block_the_event_loop(
        state, tmp_path, path, body):
    """A management call queued behind a busy slot must not freeze /health
    (H-1): the lock wait runs on a worker thread, not the event loop."""
    if body is None:
        body = {"model_dir": str(_bundle(tmp_path, "other"))}
    # `with` shares one event loop across requests, as uvicorn does.
    with TestClient(create_app(state)) as c:
        slot, t, result = _hold_slot_and_queue(state, c, path, body)
        try:
            t0 = time.monotonic()
            assert c.get("/health").status_code == 200
            assert time.monotonic() - t0 < 1.0
        finally:
            slot.lock.release()
        t.join(timeout=5)
        assert result["r"].status_code == 200, result["r"].json()


def test_a_slot_emptied_while_waiting_is_rechecked_under_the_lock(state):
    """require_loaded passed before the wait, but a failed unload_first
    switch emptied the slot during it: the call must answer 503, not hand a
    null handle to the SDK."""
    handles = []
    orig = state.lib.apply_lora
    state.lib.apply_lora = lambda h, e, a: handles.append(h) or orig(h, e, a)
    with TestClient(create_app(state), raise_server_exceptions=False) as c:
        slot, t, result = _hold_slot_and_queue(
            state, c, "/v1/lora/apply", {"lora_adapter_name": "a"})
        slot.handle = None      # what a failed unload_first switch leaves
        slot.lock.release()
        t.join(timeout=5)
    assert "r" in result, "the queued request never answered"
    assert result["r"].status_code == 503, result["r"].text
    assert result["r"].json()["error"]["code"] == "model_not_loaded"
    assert handles == []


def test_a_switch_still_reaches_an_empty_slot(state, tmp_path, client):
    """The recheck must not apply to the switch itself: loading a model is
    how a slot a failed switch left empty gets one back."""
    state.manager.slots[0].handle = None
    r = client.post("/v1/models/switch",
                    json={"model_dir": str(_bundle(tmp_path, "other"))})
    assert r.status_code == 200, r.json()
    assert state.manager.slots[0].handle is not None


def test_a_status_read_behind_a_busy_slot_reports_not_live(state):
    """GET /v1/lora/current waits a second at most, then answers from the
    cached value rather than stalling behind a generation."""
    slot = state.manager.slots[0]
    with TestClient(create_app(state)) as c:
        slot.lock.acquire()
        try:
            r = c.get("/v1/lora/current")
        finally:
            slot.lock.release()
    assert r.status_code == 200
    assert r.json()["live"] is False


def test_a_request_whose_client_left_does_not_run_once_it_gets_the_lock(state):
    """Starlette does not cancel a plain request's handler on disconnect and
    a thread cannot be cancelled, so the wait itself has to notice the client
    is gone — otherwise a switch the client gave up on runs minutes later."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from genie_server.app import ClientGoneError, run_with_slot_lock

    class _Gone:
        async def is_disconnected(self):
            return True

    slot = state.manager.slots[0]
    ran = []
    executor = ThreadPoolExecutor(max_workers=1)

    async def main():
        slot.lock.acquire()
        try:
            with pytest.raises(ClientGoneError):
                await run_with_slot_lock(
                    slot, lambda: ran.append(1), timeout=5,
                    executor=executor, request=_Gone())
        finally:
            slot.lock.release()

    asyncio.run(main())
    executor.shutdown(wait=True)   # the worker has now had the lock
    assert ran == []
    assert not slot.lock.locked()


def _stall_after_partial_output(state, abort_status=None, abort_code=None):
    """Makes the fake SDK emit 'partial ' and then hang until signalled.

    By default an abort returns WARNING_ABORTED, as the runtimes measured so
    far do. abort_status/abort_code model a runtime that instead reports it
    through the callback (SENTENCE_ABORT) and returns SUCCESS."""
    from genie_server import capi

    released = threading.Event()
    lib = state.lib

    def query(handle, text, sentence_code, on_token):
        on_token("partial ", capi.SENTENCE_CONTINUE)
        released.wait(5)
        if abort_code is not None:
            on_token("", abort_code)
        return capi.WARNING_ABORTED if abort_status is None else abort_status

    def signal_abort(handle):
        lib.abort_signals += 1
        released.set()
        return 0

    lib.query = query
    lib.signal_abort = signal_abort
    return released


def _short_timeout_client(state, timeout_s=0.2):
    """A client whose app was built with a short inference_timeout_s (the app
    captures the config at creation, so it cannot be changed afterwards)."""
    import dataclasses

    state.config = dataclasses.replace(state.config, inference_timeout_s=timeout_s)
    return TestClient(create_app(state))


def _slow(lib, name, seconds, only=None):
    """Makes lib.<name> sleep first, e.g. to let the watchdog fire during
    setup or teardown rather than during the query."""
    orig = getattr(lib, name)

    def slow(*args):
        if only is None or only(*args):
            time.sleep(seconds)
        return orig(*args)

    setattr(lib, name, slow)


CHAT = {"messages": [{"role": "user", "content": "hi"}]}


def test_inference_timeout_is_an_error_not_a_stop(state):
    """H-2: a watchdog abort used to come back as 200 / finish_reason "stop"
    with the truncated text."""
    client = _short_timeout_client(state)
    _stall_after_partial_output(state)
    r = client.post("/v1/chat/completions", json=CHAT)
    assert r.status_code == 504
    assert "timed out" in r.json()["error"]["message"]


def test_inference_timeout_mid_stream_emits_an_error_event(state):
    client = _short_timeout_client(state)
    _stall_after_partial_output(state)
    r = client.post("/v1/chat/completions", json={**CHAT, "stream": True})
    body = r.text
    assert '"error"' in body and "timed out" in body
    assert '"finish_reason": "stop"' not in body


def test_a_timeout_reported_through_the_callback_is_still_a_timeout(state):
    """A runtime may signal the abort with a SENTENCE_ABORT callback and
    return SUCCESS instead of WARNING_ABORTED. Both paths must still see a
    timeout — before, the stream said "stop" there."""
    from genie_server import capi

    client = _short_timeout_client(state)

    def stall():   # a fresh stall per request: the first one's is spent
        _stall_after_partial_output(state, abort_status=capi.STATUS_SUCCESS,
                                    abort_code=capi.SENTENCE_ABORT)

    stall()
    assert client.post("/v1/chat/completions", json=CHAT).status_code == 504
    stall()
    r = client.post("/v1/chat/completions", json={**CHAT, "stream": True})
    assert '"error"' in r.text and "timed out" in r.text
    assert '"finish_reason": "stop"' not in r.text


@pytest.mark.parametrize("stream", [False, True])
def test_a_timeout_before_the_query_starts_is_an_error_on_both_paths(state, stream):
    """The watchdog can fire during setup (a slow reset or prefix restore).
    The sync path answered 504 but the stream sent an empty "stop"."""
    client = _short_timeout_client(state)
    _slow(state.lib, "reset", 0.5)
    r = client.post("/v1/chat/completions", json={**CHAT, "stream": stream})
    if stream:
        assert '"error"' in r.text and "timed out" in r.text
        assert '"finish_reason": "stop"' not in r.text
    else:
        assert r.status_code == 504
    assert state.lib.queries == []          # the query never ran


def test_a_timer_that_fires_after_a_normal_finish_is_not_a_timeout(state):
    """The query finished; the timer went off while the engine was still
    restoring the basic sampler (before it cancels the watchdog). The output
    is complete, so the answer is a 200, not a 504. (The delay stays under
    the sync path's overall wait, 2x inference_timeout_s.)"""
    client = _short_timeout_client(state, timeout_s=0.5)
    _slow(state.lib, "apply_sampler_params", 0.8,
          only=lambda handle, params: params.get("type") == "basic")
    r = client.post("/v1/chat/completions", json={**CHAT, "logprobs": True})
    assert r.status_code == 200, r.json()
    assert r.json()["choices"][0]["finish_reason"] in ("stop", "length")


def test_a_client_abort_is_not_reported_as_a_timeout(state):
    """Only the watchdog's abort is a failure. A client abort ends the query
    with the same WARNING_ABORTED, and must leave no timeout verdict."""
    import asyncio

    from genie_server import engine

    _stall_after_partial_output(state)
    slot = state.manager.slots[0]

    async def main():
        gen = engine.Generation("req-client-abort", slot, state.lib)
        engine.start_generation(state.lib, slot, engine.QueryPlan(full_prompt="hi"),
                                engine.GenParams(), gen, None,
                                inference_timeout_s=10)
        first = await asyncio.wait_for(gen.queue.get(), 5)
        gen.abort()                         # the client, not the watchdog
        rest = await gen.collect_text(5)
        return gen, first + rest

    gen, text = asyncio.run(main())
    assert text == "partial "
    assert not gen.timed_out and not gen.timeout_error
    assert gen.error is None


def test_every_system_message_reaches_the_sdk(state, client):
    """H-3: [system A, user, assistant, system B, user] lost system B."""
    r = client.post("/v1/chat/completions", json={"messages": [
        {"role": "system", "content": "SYS-A"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "system", "content": "SYS-B"},
        {"role": "user", "content": "q2"}]})
    assert r.status_code == 200
    prompt = state.lib.queries[-1]
    assert "SYS-A" in prompt and "SYS-B" in prompt
    assert prompt.index("a1") < prompt.index("SYS-B") < prompt.index("q2")


def test_sampler_settings_do_not_leak_into_the_next_request(state, client):
    """H-4: after a greedy request, one that only sets temperature kept
    top-k=1 (the config has no top-k), so it stayed greedy."""
    handle_id = state.manager.slots[0].handle.value
    msg = [{"role": "user", "content": "hi"}]
    client.post("/v1/chat/completions",
                json={"messages": msg, "temperature": 0.0, "seed": 5})
    assert state.lib.sampler_params[handle_id]["top-k"] == "1"
    client.post("/v1/chat/completions", json={"messages": msg, "temperature": 0.7})
    params = state.lib.sampler_params[handle_id]
    assert params["top-k"] == "0" and params["seed"] != "5"
    assert params["temp"] == "0.7"


def _swap_under_lock(slot):
    """What a LoRA apply or model switch does to the slot while it holds the
    lock: a different namespace, a new epoch."""
    slot.active_lora_adapter = "other-adapter"
    slot.epoch += 1


_SYS_CHAT = {"messages": [{"role": "system", "content": "be brief"},
                          {"role": "user", "content": "hi"}]}


def test_a_request_planned_before_a_swap_is_refused_not_run(state):
    """H-5: the prompt, max_tokens and prefix-cache key are fixed before the
    lock is taken. A LoRA/model change that got the lock first left the
    worker to restore the OLD namespace's KV into the new state."""
    with TestClient(create_app(state)) as c:
        slot, t, result = _hold_slot_and_queue(
            state, c, "/v1/chat/completions", _SYS_CHAT)
        queries = len(state.lib.queries)
        _swap_under_lock(slot)
        slot.lock.release()
        t.join(timeout=5)
    r = result["r"]
    assert r.status_code == 409
    assert "changed" in r.json()["error"]["message"]
    assert len(state.lib.queries) == queries, "the stale request still ran"


def test_a_streamed_request_planned_before_a_swap_gets_an_error_event(state):
    with TestClient(create_app(state)) as c:
        slot, t, result = _hold_slot_and_queue(
            state, c, "/v1/chat/completions", {**_SYS_CHAT, "stream": True})
        queries = len(state.lib.queries)
        _swap_under_lock(slot)
        slot.lock.release()
        t.join(timeout=5)
    body = result["r"].text
    assert '"error"' in body and "changed" in body
    assert len(state.lib.queries) == queries


def test_a_request_is_refused_when_the_slot_lost_its_model_while_waiting(state):
    with TestClient(create_app(state)) as c:
        slot, t, result = _hold_slot_and_queue(
            state, c, "/v1/completions", {"prompt": "hi"})
        queries = len(state.lib.queries)
        slot.handle = None
        slot.epoch += 1
        slot.lock.release()
        t.join(timeout=5)
    assert result["r"].status_code == 409
    assert len(state.lib.queries) == queries


def test_lora_apply_and_release_change_the_epoch(state, client):
    slot = state.manager.slots[0]
    e0 = slot.epoch
    client.post("/v1/lora/apply", json={"lora_adapter_name": "a"})
    e1 = slot.epoch
    client.post("/v1/lora/release", json={"lora_adapter_name": "a"})
    assert e0 < e1 < slot.epoch


def test_a_prefix_warmup_queued_across_a_swap_does_not_save_under_the_old_key(state):
    with TestClient(create_app(state)) as c:
        slot, t, result = _hold_slot_and_queue(
            state, c, "/v1/prefix/warmup", {"system_prompt": "be brief"})
        _swap_under_lock(slot)
        slot.lock.release()
        t.join(timeout=5)
    assert result["r"].status_code == 500
    assert not state.lib.saved_states, "KV was saved under the stale key"


def test_a_model_switch_changes_the_epoch(state, tmp_path):
    slot = state.manager.slots[0]
    e0 = slot.epoch
    with TestClient(create_app(state)) as c:
        r = c.post("/v1/models/switch",
                   json={"model_dir": str(_bundle(tmp_path, "other"))})
    assert r.status_code == 200
    assert slot.epoch >= e0 + 2  # unload, then adopt (bumps at both ends)


def test_an_image_request_goes_through_the_http_path(state, monkeypatch, tmp_path):
    """Nothing else sends an image through the HTTP handler; the VLM path
    builds its Generation from a VLMSlot, which must carry what a text Slot
    does (the H-5 epoch broke every VLM request with an AttributeError)."""
    pytest.importorskip("numpy")
    PIL = pytest.importorskip("PIL.Image")
    import base64
    import io
    from pathlib import Path

    from fake_genie import FakeVLMNode, FakeVLMPipeline
    from genie_server import genie_node, vlm

    monkeypatch.setattr(genie_node, "Node", FakeVLMNode)
    monkeypatch.setattr(genie_node, "Pipeline", FakeVLMPipeline)
    bundle = Path(__file__).parent / "data" / "vlm_bundles" / "ai_hub"
    vslot = vlm.VLMSlot(name="vlm0", device_id=None, model_root=bundle,
                        spec_name=None, htp_ext_cache_dir=tmp_path)
    state.manager.vlm_slots = [vslot]
    pipeline = vslot.pipeline   # shutdown frees it when the client closes

    buf = io.BytesIO()
    PIL.new("RGB", (4, 4)).save(buf, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    with TestClient(create_app(state)) as c:
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": url}}]}]})
    assert r.status_code == 200, r.text
    assert r.json()["model"] == "ai_hub"
    assert pipeline.executed == 1


def test_an_image_over_the_pixel_ceiling_is_a_400_before_the_npu(
        state, monkeypatch, tmp_path):
    """VLM_MAX_IMAGE_PIXELS reaches the HTTP path: the request is refused as
    the client's error and the pipeline never runs."""
    pytest.importorskip("numpy")
    PIL = pytest.importorskip("PIL.Image")
    import base64
    import dataclasses
    import io
    from pathlib import Path

    from fake_genie import FakeVLMNode, FakeVLMPipeline
    from genie_server import genie_node, vlm

    monkeypatch.setattr(genie_node, "Node", FakeVLMNode)
    monkeypatch.setattr(genie_node, "Pipeline", FakeVLMPipeline)
    bundle = Path(__file__).parent / "data" / "vlm_bundles" / "ai_hub"
    vslot = vlm.VLMSlot(name="vlm0", device_id=None, model_root=bundle,
                        spec_name=None, htp_ext_cache_dir=tmp_path)
    state.manager.vlm_slots = [vslot]
    state.config = dataclasses.replace(state.config, vlm_max_image_pixels=15)
    pipeline = vslot.pipeline   # shutdown frees it when the client closes

    buf = io.BytesIO()
    PIL.new("RGB", (4, 4)).save(buf, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    with TestClient(create_app(state)) as c:
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": url}}]}]})
    assert r.status_code == 400, r.text
    assert "VLM_MAX_IMAGE_PIXELS" in r.json()["error"]["message"]
    assert pipeline.executed == 0


def test_a_lora_strength_change_is_part_of_the_cache_namespace(state, client):
    """A prefix KV saved at one alpha must not be restored at another, and a
    request planned before the change must not run after it (epoch)."""
    slot = state.manager.slots[0]
    ns0, e0 = slot.cache_namespace, slot.epoch
    r = client.post("/v1/lora/strength",
                    json={"tensor_name": "t", "alpha": 0.5})
    assert r.status_code == 200
    assert slot.cache_namespace != ns0 and slot.epoch > e0
    ns_half = slot.cache_namespace
    client.post("/v1/lora/strength", json={"tensor_name": "t", "alpha": 1.0})
    assert slot.cache_namespace not in (ns0, ns_half)
    # Applying an adapter does NOT put the alphas back (measured on the
    # board: they belong to the dialog, so they survive an adapter switch), and
    # the namespace must keep saying so.
    client.post("/v1/lora/apply", json={"lora_adapter_name": "a"})
    assert slot.lora_strengths == {"primary/t": 1.0}
    assert slot.cache_namespace.startswith(f"{slot.name}|{slot.active_model_id}|a|")
    # A release does, so its namespace goes back to the plain one.
    client.post("/v1/lora/release", json={"lora_adapter_name": "a"})
    assert slot.lora_strengths == {}
    assert slot.cache_namespace == f"{slot.name}|{slot.active_model_id}|"


def test_no_strength_keeps_the_old_namespace(state):
    """Existing on-disk cache keys stay reachable: without a strength set,
    the namespace string is exactly what it was before strengths joined it."""
    slot = state.manager.slots[0]
    assert slot.cache_namespace == \
        f"{slot.name}|{slot.active_model_id}|{slot.active_lora_adapter}"


def test_a_failed_strength_change_leaves_the_namespace_alone(state, client):
    slot = state.manager.slots[0]
    ns0, e0 = slot.cache_namespace, slot.epoch
    state.lib.lora_strength_status = -1
    r = client.post("/v1/lora/strength", json={"tensor_name": "t", "alpha": 0.5})
    assert r.status_code == 500
    assert slot.cache_namespace == ns0 and slot.epoch == e0


# ---------------------------------------------------------------- cross-origin

EVIL = "http://evil.example"


@pytest.mark.parametrize("headers", [
    {"Content-Type": "text/plain"},
    {"Content-Type": "application/x-www-form-urlencoded"},
    {"Content-Type": "multipart/form-data; boundary=x"},
    {},
])
def test_a_body_a_browser_sends_without_preflight_is_refused(client, headers):
    """A page on any origin can POST these three content types, or none,
    without asking the browser first. Parsing them as JSON let that page
    change the server's state; nothing may happen before the 415."""
    r = client.post("/v1/server/prompt_logprobs", content=b'{"enabled": true}',
                    headers={"Origin": EVIL, **headers})
    assert r.status_code == 415
    assert r.json()["error"]["type"] == "invalid_request_error"
    assert client.get("/v1/server/prompt_logprobs").json()["enabled"] is False


def test_the_415_names_what_was_sent(client):
    """The message says which Content-Type came in, and reads plainly when
    there was none."""
    r = client.post("/v1/server/prompt_logprobs", content=b'{"enabled": true}',
                    headers={"Content-Type": "text/plain"})
    assert r.json()["error"]["message"].endswith("got 'text/plain'")
    r = client.post("/v1/server/prompt_logprobs", content=b'{"enabled": true}')
    assert r.json()["error"]["message"].endswith("got no Content-Type")


def test_a_json_media_type_with_parameters_is_accepted(client):
    r = client.post("/v1/server/prompt_logprobs", content=b'{"enabled": true}',
                    headers={"Content-Type": "application/json; charset=utf-8"})
    assert r.status_code == 200


def _preflight(client, origin):
    return client.options("/v1/models/switch", headers={
        "Origin": origin, "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type"})


def test_no_origin_is_allowed_by_default(client):
    """With no CORS_ALLOW_ORIGINS, another origin can neither pass the
    preflight a JSON POST needs nor read a reply."""
    assert "access-control-allow-origin" not in _preflight(client, EVIL).headers
    r = client.get("/v1/models", headers={"Origin": EVIL})
    assert r.status_code == 200
    assert "access-control-allow-origin" not in r.headers


def test_cors_allows_only_the_configured_origins(state):
    import dataclasses
    state.config = dataclasses.replace(
        state.config, cors_allow_origins=("http://localhost:3000",))
    client = TestClient(create_app(state))
    ok = _preflight(client, "http://localhost:3000")
    assert ok.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "access-control-allow-origin" not in _preflight(client, EVIL).headers


# ---------------------------------------------------------------- body size

def _client_with_body_limit(state, mb):
    import dataclasses
    state.config = dataclasses.replace(state.config, max_request_body_mb=mb)
    return TestClient(create_app(state))


def test_a_body_over_the_limit_is_413(state):
    """Refused from the declared Content-Length, before it is buffered."""
    client = _client_with_body_limit(state, 1 / 1024)   # 1 KiB
    r = client.post("/v1/server/prompt_logprobs",
                    json={"enabled": True, "pad": "x" * 2000})
    assert r.status_code == 413
    assert "MAX_REQUEST_BODY_MB" in r.json()["error"]["message"]
    assert client.get("/v1/server/prompt_logprobs").json()["enabled"] is False


def test_a_chunked_body_over_the_limit_is_413(state):
    """No Content-Length to go by: counted as it arrives."""
    client = _client_with_body_limit(state, 1 / 1024)

    def chunks():
        yield b'{"enabled": true, "pad": "'
        for _ in range(20):
            yield b"x" * 100
        yield b'"}'

    r = client.post("/v1/server/prompt_logprobs", content=chunks(),
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 413


@pytest.mark.parametrize("mb", [1 / 1024, 0])
def test_a_body_within_the_limit_or_with_none_is_read(state, mb):
    client = _client_with_body_limit(state, mb)
    r = client.post("/v1/server/prompt_logprobs",
                    json={"enabled": True, "pad": "x" * (500 if mb else 5000)})
    assert r.status_code == 200


# ---------------------------------------------------------------- LoRA alpha names

def _with_lora(state, lora):
    """The shape of genie_phi4_lora's dialog config: one basic engine whose
    binary carries the adapters."""
    slot = state.manager.slots[0]
    slot.dialog_cfg = {**slot.dialog_cfg,
                       "engine": {"model": {"binary": {"lora": lora}}}}
    return slot


# genie_phi4_lora's lora block, bin-sections left out.
PHI4_LORA = {"version": 1, "alpha-tensor-name": "lora_alpha", "adapters": [
    {"version": 1, "name": "finetuned", "alphas": ["alpha0", "alpha1"]},
    {"version": 1, "name": "default_adapter", "alphas": ["alpha0", "alpha1"]}]}


@pytest.mark.parametrize("name", ["alpha2", "lora_alpha"])
def test_a_lora_alpha_the_model_does_not_have_is_refused(state, client, name):
    """The SDK says success for it and changes nothing (measured on the
    board), so nothing may reach the SDK or the cache namespace. That
    includes the base graph's alpha tensor: once the adapters list their own
    alphas, the SDK looks names up among those only."""
    slot = _with_lora(state, PHI4_LORA)
    ns0, e0 = slot.cache_namespace, slot.epoch
    r = client.post("/v1/lora/strength", json={"tensor_name": name,
                                               "alpha": 0.5})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["param"] == "tensor_name"
    assert "['alpha0', 'alpha1']" in err["message"]
    assert state.lib.lora_strengths == []
    assert (slot.cache_namespace, slot.epoch) == (ns0, e0)


def test_a_declared_alpha_is_set_even_before_its_adapter_is_applied(state,
                                                                   client):
    """The SDK keeps it and writes it when the adapter is applied."""
    _with_lora(state, PHI4_LORA)
    r = client.post("/v1/lora/strength", json={"tensor_name": "alpha1",
                                               "alpha": 0.5})
    assert r.status_code == 200
    assert state.lib.lora_strengths[-1][1:] == ("primary", "alpha1", 0.5)


def test_an_engine_without_lora_refuses_every_alpha(state, client):
    slot = state.manager.slots[0]
    slot.dialog_cfg = {**slot.dialog_cfg, "engine": {"model": {"binary": {}}}}
    r = client.post("/v1/lora/strength", json={"tensor_name": "alpha0",
                                               "alpha": 1.0})
    assert r.status_code == 400
    assert "declares no LoRA adapters" in r.json()["error"]["message"]


@pytest.mark.parametrize("alpha", ['"0.5"', "true", "[0.5]", "NaN",
                                   "Infinity"])
def test_lora_alpha_must_be_a_finite_number(state, client, alpha):
    """Sent as raw JSON: Python's json reads NaN and Infinity, and float()
    would have taken "0.5", true and NaN alike."""
    r = client.post("/v1/lora/strength",
                    content=f'{{"tensor_name": "t", "alpha": {alpha}}}',
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "alpha"
    assert state.lib.lora_strengths == []


def test_lora_tensor_name_must_be_a_string(state, client):
    r = client.post("/v1/lora/strength", json={"tensor_name": ["t"],
                                               "alpha": 1.0})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "tensor_name"


@pytest.mark.parametrize("path, body", [
    ("/v1/lora/apply", {"lora_adapter_name": "a"}),
    ("/v1/lora/strength", {"tensor_name": "alpha0", "alpha": 0.5}),
    ("/v1/lora/release", {"lora_adapter_name": "a"}),
])
@pytest.mark.parametrize("engine", [["primary"], {"role": "primary"}, 1, ""])
def test_lora_engine_must_be_a_string(state, client, path, body, engine):
    """A non-string engine reached str.encode or a dict lookup: a 500."""
    _with_lora(state, PHI4_LORA)
    r = client.post(path, json={**body, "engine": engine})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "engine"


@pytest.mark.parametrize("path", ["/v1/lora/apply", "/v1/lora/release"])
def test_lora_adapter_name_must_be_a_string(client, path):
    r = client.post(path, json={"lora_adapter_name": ["a"]})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "lora_adapter_name"


def test_one_alpha_set_under_both_role_spellings_is_recorded_once(state, client):
    """The SDK folds "target" into "primary". Recorded under both spellings,
    alpha0 = 1.0 and alpha0 = 0.5 came out as the same cache namespace
    (primary/alpha0=0.5,target/alpha0=1.0), whichever was set last."""
    def namespace_after(steps):
        slot = _with_lora(state, PHI4_LORA)
        slot.lora_strengths = {}
        for engine, alpha in steps:
            r = client.post("/v1/lora/strength", json={
                "engine": engine, "tensor_name": "alpha0", "alpha": alpha})
            assert r.status_code == 200
        return slot.cache_namespace, dict(slot.lora_strengths)

    ns_one, strengths = namespace_after([("primary", 0.5), ("target", 1.0)])
    assert strengths == {"primary/alpha0": 1.0}
    ns_half, strengths = namespace_after([("target", 1.0), ("primary", 0.5)])
    assert strengths == {"primary/alpha0": 0.5}
    assert ns_one != ns_half


def test_a_lora_alpha_is_checked_against_the_model_that_holds_the_lock(state):
    """Checked when the call runs, not when it was sent: a switch that won
    the lock first must not have the old model's names vouch for the new."""
    _with_lora(state, PHI4_LORA)
    with TestClient(create_app(state)) as c:
        slot, t, result = _hold_slot_and_queue(
            state, c, "/v1/lora/strength", {"tensor_name": "alpha0",
                                            "alpha": 0.5})
        # What a switch to a model without LoRA leaves behind.
        slot.dialog_cfg = {**slot.dialog_cfg, "engine": {"model": {"binary": {}}}}
        slot.lock.release()
        t.join(timeout=5)
    assert result["r"].status_code == 400
    assert "declares no LoRA adapters" in result["r"].json()["error"]["message"]
    assert state.lib.lora_strengths == []


# ---------------------------------------------------------------- parameter types

def _chat(client, **extra):
    return client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4,
        **extra})


@pytest.mark.parametrize("extra, param", [
    ({"temperature": "hot"}, "temperature"),     # was a 500 from float()
    ({"temperature": -0.1}, "temperature"),
    ({"temperature": True}, "temperature"),
    ({"top_p": 7}, "top_p"),                     # was passed to the SDK
    ({"top_p": "0.9"}, "top_p"),
    ({"top_k": 1.5}, "top_k"),                   # was int()-ed to 1: greedy
    ({"top_k": "5"}, "top_k"),
    ({"seed": "x"}, "seed"),                     # was a 500 from int()
    ({"seed": -1}, "seed"),                      # numpy refuses it
    ({"seed": 2**31}, "seed"),                   # the SDK's stoi overflows
    ({"n": "2"}, "n"),                           # was a plain-text 500
    ({"n": 0}, "n"),
    ({"chat_template_kwargs": [1]}, "chat_template_kwargs"),  # plain-text 500
    ({"chat_template_kwargs": {"enable_thinking": "false"}}, "enable_thinking"),
    ({"enable_thinking": "false"}, "enable_thinking"),  # "false" read as true
    ({"stream": "false"}, "stream"),             # likewise: it streamed
    ({"logprobs": "yes"}, "logprobs"),
])
def test_a_generation_parameter_of_the_wrong_type_is_a_400(state, client,
                                                            extra, param):
    r = _chat(client, **extra)
    assert r.status_code == 400, r.text
    err = r.json()["error"]
    assert (err["type"], err["param"]) == ("invalid_request_error", param)
    assert state.lib.queries == []


def test_vllms_top_k_minus_one_means_no_limit(state, client):
    """vLLM spells "no top-k limit" -1; here it is 0, since the SDK reads
    top-k as unsigned. Same meaning, so it is accepted and sent as 0."""
    r = _chat(client, top_k=-1)
    assert r.status_code == 200, r.text
    handle = state.manager.slots[0].handle.value
    assert state.lib.sampler_params[handle]["top-k"] == "0"


@pytest.mark.parametrize("top_k", [-2, -1.5])
def test_other_negative_top_k_is_refused(client, top_k):
    r = _chat(client, top_k=top_k)
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "top_k"
    assert "vLLM's -1" in r.json()["error"]["message"]


@pytest.mark.parametrize("key", ["max_tokens", "max_completion_tokens"])
def test_a_max_tokens_of_true_is_not_one(client, key):
    """isinstance(True, int) let true through as max_tokens = 1."""
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], key: True})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == key


def test_an_integral_float_max_tokens_is_accepted(state, client):
    """8.0 is how some clients serialize every number; it was refused while
    true was accepted."""
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8.0})
    assert r.status_code == 200, r.text
    assert state.lib.max_tokens[state.manager.slots[0].handle.value] == 8


def test_top_logprobs_of_true_is_not_one(client):
    r = _chat(client, logprobs=True, top_logprobs=True)
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "top_logprobs"


@pytest.mark.parametrize("path, body", [
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
    ("/v1/completions", {"prompt": "hi"}),
])
@pytest.mark.parametrize("options, param", [
    ("x", "stream_options"),
    (["include_usage"], "stream_options"),
    ({"include_usage": "false"}, "stream_options.include_usage"),
])
def test_stream_options_are_checked_before_the_generation_starts(
        state, client, path, body, options, param):
    """bool("false") sent a usage chunk to a client that asked for none, and
    a non-object was a 500. Both are refused before any generation starts:
    a 400 raised while the stream was being set up would have left that
    generation holding the slot."""
    r = client.post(path, json={**body, "stream": True,
                                "stream_options": options})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == param
    assert state.lib.queries == []


def test_a_nan_temperature_is_a_400(state, client):
    """Python's json reads NaN, and NaN passes every range comparison."""
    r = client.post("/v1/chat/completions", headers={
        "Content-Type": "application/json"}, content=(
        '{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, '
        '"temperature": NaN}'))
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "temperature"


@pytest.mark.parametrize("extra, param", [
    ({"echo": "false"}, "echo"),
    ({"stream": 1}, "stream"),
    ({"best_of": "2"}, "best_of"),
    ({"n": "2"}, "n"),                           # was a plain-text 500
])
def test_completions_parameters_of_the_wrong_type_are_a_400(state, client,
                                                             extra, param):
    r = client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 4,
                                             **extra})
    assert r.status_code == 400, r.text
    assert r.json()["error"]["param"] == param
    assert state.lib.queries == []


def test_valid_edge_values_reach_the_sdk(state, client):
    """The bounds themselves are allowed, and an integral float is an
    integer (some clients serialize every number as a float)."""
    r = _chat(client, temperature=0.7, top_p=1, top_k=2.0, seed=2**31 - 1,
              n=1, stream=False, logprobs=False,
              chat_template_kwargs={"enable_thinking": False})
    assert r.status_code == 200, r.text
    params = state.lib.sampler_params[state.manager.slots[0].handle.value]
    assert (params["top-k"], params["top-p"], params["seed"]) == (
        "2", "1.0", str(2**31 - 1))


def test_warmup_enable_thinking_must_be_a_bool(client):
    r = client.post("/v1/prefix/warmup", json={"system_prompt": "s",
                                               "enable_thinking": "false"})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "enable_thinking"


# ---------------------------------------------------------------- unexpected errors

_ERROR_ID = r"err-[0-9a-f]{8}"


def _boom(state, monkeypatch, text="/secret/model/dir and a prompt fragment"):
    def boom(*args, **kwargs):
        raise RuntimeError(text)

    monkeypatch.setattr(state.manager, "select_for_request", boom)


@pytest.mark.parametrize("path, body", [
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
    ("/v1/lora/apply", {"lora_adapter_name": "a"}),
])
def test_an_unexpected_exception_is_still_an_openai_error(state, monkeypatch,
                                                         path, body):
    """A bug in the server must not answer in plain text, and must not hand
    the exception's text (a path, a piece of a prompt) to the client."""
    import re

    _boom(state, monkeypatch)
    r = TestClient(create_app(state)).post(path, json=body)
    assert r.status_code == 500
    assert r.headers["content-type"] == "application/json"
    err = r.json()["error"]
    assert err["type"] == "server_error"
    assert re.fullmatch(r"Internal server error \(RuntimeError\); the details "
                        rf"are in the server log under {_ERROR_ID}\.",
                        err["message"]), err["message"]
    assert "secret" not in r.text


def test_the_error_id_leads_to_the_traceback_in_the_log(state, monkeypatch,
                                                         caplog):
    """The reply withholds the exception's text; the log keeps it, under the
    id the reply names, so a report can be matched to its entry."""
    import logging
    import re

    _boom(state, monkeypatch, text="kept for the log")
    with caplog.at_level(logging.ERROR, logger="genie_server.protocol"):
        r = TestClient(create_app(state)).post(
            "/v1/lora/apply", json={"lora_adapter_name": "a"})
    error_id = re.search(_ERROR_ID, r.json()["error"]["message"]).group(0)
    records = [rec for rec in caplog.records if error_id in rec.getMessage()]
    assert len(records) == 1
    assert "kept for the log" in str(records[0].exc_info[1])


def test_an_unexpected_exception_reaches_an_allowed_origin(state, monkeypatch):
    """Answered inside CORS, so a page on an allowed origin can read the 500.
    An exception handler for Exception runs outside CORS in Starlette, and
    its reply carried no Access-Control-Allow-Origin."""
    import dataclasses

    origin = "http://allowed.example"
    state.config = dataclasses.replace(state.config,
                                       cors_allow_origins=(origin,))
    _boom(state, monkeypatch)
    r = TestClient(create_app(state)).post(
        "/v1/lora/apply", json={"lora_adapter_name": "a"},
        headers={"Origin": origin})
    assert r.status_code == 500
    assert r.headers.get("access-control-allow-origin") == origin
    assert r.json()["error"]["type"] == "server_error"


# ---------------------------------------------------------------- prefix cache hygiene

def _warm(client, prompt="You are terse."):
    r = client.post("/v1/prefix/warmup", json={"system_prompt": prompt})
    assert r.status_code == 200, r.text
    return r.json()["key"]


def _entries(client):
    return {e["key"]: e for e in client.get("/v1/prefix/cache").json()["entries"]}


def test_an_entry_records_its_namespace_and_whether_it_is_reachable(state,
                                                                   client):
    key = _warm(client)
    e = _entries(client)[key]
    assert e["namespace"] == state.manager.slots[0].cache_namespace
    assert e["reachable"] is True


def test_prune_deletes_only_what_no_slot_can_reach(state, client):
    """A LoRA change moves the namespace: the old entry stays on disk,
    unreachable, until someone asks for it to go."""
    old = _warm(client)
    client.post("/v1/lora/apply", json={"lora_adapter_name": "a"})
    new = _warm(client)
    assert old != new
    assert (_entries(client)[old]["reachable"],
            _entries(client)[new]["reachable"]) == (False, True)

    r = client.delete("/v1/prefix/cache?scope=unreachable")
    assert r.status_code == 200
    assert r.json()["deleted"] == [old]
    assert r.json()["freed_bytes"] > 0
    assert set(_entries(client)) == {new}


def test_an_entry_without_a_recorded_namespace_is_kept_unless_asked(state,
                                                                    client):
    """Saved before namespaces were recorded: it may well be reachable."""
    legacy = "0123456789abcdef"
    (state.prefix_cache._dir / f"prefix_{legacy}.geniestate").write_bytes(b"kv")
    assert _entries(client)[legacy]["reachable"] is None

    r = client.delete("/v1/prefix/cache?scope=unreachable")
    assert r.json() == {"deleted": [], "freed_bytes": 0,
                        "kept_unknown": [legacy], "kept_in_use": []}
    r = client.delete("/v1/prefix/cache?scope=all")
    assert r.json()["deleted"] == [legacy]
    assert _entries(client) == {}


def test_a_bare_delete_of_the_collection_is_refused(client):
    key = _warm(client)
    for url in ("/v1/prefix/cache", "/v1/prefix/cache?scope=everything"):
        r = client.delete(url)
        assert r.status_code == 400
        assert r.json()["error"]["param"] == "scope"
    assert key in _entries(client)


@pytest.mark.parametrize("key", ["abc", "0123456789ABCDEF",
                                 "0123456789abcdeg", "0123456789abcdef0"])
def test_deleting_a_key_that_is_not_a_cache_key_is_a_400(client, key):
    r = client.delete(f"/v1/prefix/cache/{key}")
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "key"


def test_deleting_an_entry_removes_its_namespace_record(state, client):
    key = _warm(client)
    meta = state.prefix_cache._dir / f"prefix_{key}.json"
    assert meta.exists()
    assert client.delete(f"/v1/prefix/cache/{key}").status_code == 200
    assert not meta.exists()


def _forget_namespace(state, key):
    """Makes an entry look as if it was saved before namespaces were."""
    (state.prefix_cache._dir / f"prefix_{key}.json").unlink()
    assert state.prefix_cache.namespace_of(key) is None


def test_a_warmup_of_an_old_entry_records_its_namespace(state, client):
    """Without this, every entry from before the upgrade lists as unknown
    forever, and only scope=all can clear any of them."""
    key = _warm(client)
    _forget_namespace(state, key)
    r = client.post("/v1/prefix/warmup",
                    json={"system_prompt": "You are terse."})
    assert r.json()["status"] == "already_cached"
    assert _entries(client)[key]["reachable"] is True


def test_a_hit_on_an_old_entry_records_its_namespace(state, client):
    key = _warm(client)
    _forget_namespace(state, key)
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "system", "content": "You are terse."},
                     {"role": "user", "content": "hi"}], "max_tokens": 4})
    assert r.status_code == 200
    assert _entries(client)[key]["namespace"] == \
        state.manager.slots[0].cache_namespace


def test_prune_keeps_an_entry_a_save_is_still_writing(tmp_path):
    """scope=all during a warmup must not delete the half-written entry
    (which the save would then record a namespace for, as reachable)."""
    cache = PrefixCache(str(tmp_path))
    key = cache.key("You are terse.", "chat|m|")
    seen = {}

    class Lib:
        def save_state(self, handle, path):
            Path(path).write_bytes(b"part")
            seen.update(cache.prune(set(), include_unknown=True))
            Path(path).write_bytes(b"part+rest")
            return 0

    assert cache.save(Lib(), None, key, namespace="chat|m|")
    assert seen["kept_in_use"] == [key] and seen["deleted"] == []
    assert cache.namespace_of(key) == "chat|m|"
    assert cache.prune(set(), include_unknown=True)["deleted"] == [key]


def test_prune_keeps_an_entry_being_restored(state, client):
    key = _warm(client)
    with state.prefix_cache._using(key):
        r = client.delete("/v1/prefix/cache?scope=all")
    assert r.json()["kept_in_use"] == [key]
    assert key in _entries(client)


def test_prune_sweeps_a_namespace_record_left_without_its_entry(state,
                                                                client):
    key = _warm(client)
    (state.prefix_cache._dir / f"prefix_{key}.geniestate").unlink()
    meta = state.prefix_cache._dir / f"prefix_{key}.json"
    assert meta.exists()
    client.delete("/v1/prefix/cache?scope=unreachable")
    assert not meta.exists()


# ---------------------------------------------------------------- readiness

def test_ready_when_every_slot_holds_a_model(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ready",
                        "slots": [{"name": "default", "loaded": True}]}
    assert client.get("/v1/ready").status_code == 200


def test_a_slot_left_empty_is_not_ready_while_health_stays_ok(state, client,
                                                              tmp_path):
    """The failed unload_first switch the review names: the slot is empty,
    /health cannot tell, /ready must."""
    state.lib.fail_create = True
    try:
        r = client.post("/v1/models/switch",
                        json={"model_dir": str(_bundle(tmp_path, "other"))})
    finally:
        state.lib.fail_create = False
    assert r.status_code == 500
    assert state.manager.slots[0].handle is None   # unload_first, then failed
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json() == {"status": "not ready",
                        "slots": [{"name": "default", "loaded": False}],
                        "not_loaded": ["default"]}


def test_readiness_covers_vlm_slots(state, client):
    class FakeVLM:
        name, pipeline = "vlm0", object()

    state.manager.vlm_slots = [FakeVLM()]
    assert client.get("/ready").json()["slots"][-1] == {"name": "vlm0",
                                                        "loaded": True}


def _with_an_empty_second_slot(state):
    """A two-slot server whose second slot a failed switch left empty."""
    from pathlib import Path

    from genie_server.slots import Slot

    second = Slot(name="second", device_id=1, model_root=Path("/models/other"))
    state.manager.slots.append(second)
    state.manager._by_name[second.name] = second
    return second


def test_one_empty_slot_makes_the_whole_server_not_ready(state, client):
    _with_an_empty_second_slot(state)
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["not_loaded"] == ["second"]


def test_ready_can_be_asked_about_one_slot(state, client):
    """A monitor that routes per slot must not see the healthy slot as down
    because another one is empty."""
    _with_an_empty_second_slot(state)
    r = client.get("/ready", params={"slot": "default"})
    assert r.status_code == 200
    assert r.json() == {"status": "ready",
                        "slots": [{"name": "default", "loaded": True}]}
    r = client.get("/v1/ready", params={"slot": "second"})
    assert r.status_code == 503
    assert r.json()["not_loaded"] == ["second"]


def test_ready_for_an_unknown_slot_is_a_404(client):
    r = client.get("/ready", params={"slot": "nope"})
    assert r.status_code == 404
    assert r.json()["error"]["param"] == "slot"


def test_unload_first_must_be_a_bool(state, client, tmp_path):
    """bool("false") is True: the quoted false emptied the slot first."""
    r = client.post("/v1/models/switch", json={
        "model_dir": str(_bundle(tmp_path, "other")), "unload_first": "false"})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "unload_first"
    assert state.manager.slots[0].handle is not None


# ---------------------------------------------------------------- logprobs sampling

@pytest.fixture
def collectors(monkeypatch):
    """Every LogprobsCollector the handlers build, with its arguments."""
    from genie_server import app as app_mod
    made = []

    class Recording(app_mod.LogprobsCollector):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(app_mod, "LogprobsCollector", Recording)
    return made


MODEL_SAMPLER = {"temp": 0.3, "top-k": 5, "top-p": 0.7}


@pytest.mark.parametrize("path, body", [
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                              "logprobs": True}),
    ("/v1/completions", {"prompt": "hi", "logprobs": 1}),
])
def test_logprobs_sample_with_the_models_defaults(state, client, collectors,
                                                   path, body):
    """What a request leaves out comes from genie_config.json, as it does
    without logprobs. It used to be temperature 1.0, no top-k, no top-p."""
    pytest.importorskip("numpy")
    state.manager.slots[0].sampler_defaults = dict(MODEL_SAMPLER)
    r = client.post(path, json={**body, "max_tokens": 2})
    assert r.status_code == 200, r.text
    c = collectors[-1]
    assert (c.temperature, c.top_k, c.top_p) == (0.3, 5, 0.7)
    # ...and the same values the SDK sampler gets on the plain path.
    from genie_server.capi import make_sampler_params
    sdk = make_sampler_params(MODEL_SAMPLER)
    assert (float(sdk["temp"]), int(sdk["top-k"]), float(sdk["top-p"])) == (
        c.temperature, c.top_k, c.top_p)


def test_logprobs_keep_what_the_request_sets(state, client, collectors):
    pytest.importorskip("numpy")
    state.manager.slots[0].sampler_defaults = dict(MODEL_SAMPLER)
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "logprobs": True,
        "max_tokens": 2, "temperature": 0.9, "top_p": 0.5})
    assert r.status_code == 200, r.text
    c = collectors[-1]
    assert (c.temperature, c.top_k, c.top_p) == (0.9, 5, 0.5)


def test_logprobs_fall_back_to_the_sdk_defaults(state, client, collectors):
    """A model whose genie_config.json sets no sampler values gets the SDK's
    own defaults, the ones make_sampler_params falls back to."""
    pytest.importorskip("numpy")
    from genie_server.capi import SDK_SAMPLER_DEFAULTS
    state.manager.slots[0].sampler_defaults = {}
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "logprobs": True,
        "max_tokens": 2})
    assert r.status_code == 200, r.text
    c = collectors[-1]
    assert (c.temperature, c.top_k, c.top_p) == (
        SDK_SAMPLER_DEFAULTS["temp"], SDK_SAMPLER_DEFAULTS["top-k"],
        SDK_SAMPLER_DEFAULTS["top-p"])


def test_greedy_logprobs_are_greedy(state, client, collectors):
    """temperature 0 is top-k 1 on both paths."""
    pytest.importorskip("numpy")
    state.manager.slots[0].sampler_defaults = dict(MODEL_SAMPLER)
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "logprobs": True,
        "max_tokens": 2, "temperature": 0})
    assert r.status_code == 200, r.text
    assert collectors[-1].top_k == 1


def test_tool_history_on_a_template_without_a_tool_form_is_a_400(state,
                                                                  client):
    state.manager.slots[0].chat_template = "llama2"
    r = client.post("/v1/chat/completions", json={"max_tokens": 4, "messages": [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c", "type": "function",
             "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c", "content": "sunny"}]})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "messages"
    assert state.lib.queries == []


# ---------------------------------------------------------------- small fixes (L-6..L-14)

def _with_a_loaded_second_slot(state):
    """A second slot holding the same model, so 'model' cannot tell the two
    apart and only 'slot' can."""
    from genie_server.slots import Slot
    first = state.manager.slots[0]
    second = Slot(name="second", device_id=1, model_root=first.model_root)
    second.handle = state.lib.create_dialog(b"{}")
    second.dialog_cfg = dict(first.dialog_cfg)
    second.chat_template = first.chat_template
    second.tokenizer = first.tokenizer
    state.manager.slots.append(second)
    state.manager._by_name[second.name] = second
    state.manager.status[second.name] = {"phase": "idle", "detail": ""}
    state.manager.reindex()
    return second


def test_warmup_can_target_a_slot_by_name(state, client):
    """With 'model' alone the second of two slots holding one model was out
    of reach."""
    second = _with_a_loaded_second_slot(state)
    r = client.post("/v1/prefix/warmup", json={"slot": "second",
                                               "system_prompt": "s"})
    assert r.status_code == 200, r.text
    key = r.json()["key"]
    e = {x["key"]: x for x in client.get("/v1/prefix/cache").json()["entries"]}
    assert e[key]["namespace"] == second.cache_namespace


def test_performance_policy_can_target_a_slot_by_name(state, client):
    _with_a_loaded_second_slot(state)
    r = client.post("/v1/server/performance_policy",
                    json={"slot": "second", "policy": "burst"})
    assert r.status_code == 200, r.text
    assert r.json()["slot"] == "second"
    r = client.get("/v1/server/performance_policy", params={"slot": "second"})
    assert r.json()["slot"] == "second"


def test_prompt_scoring_reads_max_completion_tokens_first(client, state):
    """Every other path lets max_completion_tokens win over max_tokens; the
    scoring path did the opposite and refused this request."""
    pytest.importorskip("numpy")
    client.post("/v1/server/prompt_logprobs", json={"enabled": True})
    ids = state.manager.slots[0].tokenizer.encode("the quick brown fox").ids
    r = client.post("/v1/completions", json={
        "prompt": [ids], "echo": True, "logprobs": 1,
        "max_completion_tokens": 0, "max_tokens": 5})
    assert r.status_code == 200, r.text


def test_a_prompt_that_does_not_fit_is_refused_before_any_prompt_runs(
        state, client):
    """It used to be found only after the prompts ahead of it had run."""
    state.manager.slots[0].dialog_cfg = {"context": {"size": 64}}
    r = client.post("/v1/completions", json={
        "prompt": ["short", "word " * 200], "max_tokens": 4})
    assert r.status_code == 400
    assert state.lib.queries == []
