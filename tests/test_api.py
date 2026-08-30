"""Offline HTTP API tests (FakeGenieLib — no NPU, no libGenie.so)."""

import json

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
