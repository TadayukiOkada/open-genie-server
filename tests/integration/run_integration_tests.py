#!/usr/bin/env python3
"""Host-side integration test runner for a genie-server instance on a device.

Runs from the host PC against a live genie-server (Hexagon NPU device) over
its REST API, exercises every feature area, and writes a Markdown + JSON
report at the end. If the server dies mid-run, the incident (which test,
which request, what the server looked like before) is recorded in the report
and the remaining tests are marked ABORTED.

Usage:
    python3 run_integration_tests.py --config test_config.json \
        [--base-url http://<device-ip>:8080] [--only C01,CH02] [--list]

Everything is driven by the config file (see test_config.sample.json);
--base-url overrides the config's base_url. Feature areas without config
(model switch, LoRA, VLM) are SKIPPED, not failed.

Dependencies on the host: python3 + `requests` (pip install requests).
"""

import argparse
import base64
import json
import re
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("This runner needs the 'requests' package: pip install requests")
    sys.exit(2)

# 64x64 red PNG (used for the VLM test when no image_path is configured).
RED_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAS0lEQVR42u3PQQkAAAgAsetf"
    "WiP4FgYrsKZeS0BAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBA"
    "QEDgsqnc8OJg6Ln3AAAAAElFTkSuQmCC"
)


# ---------------------------------------------------------------- outcomes

class CheckFailure(Exception):
    """A test assertion failed (server responded, but wrongly)."""

    def __init__(self, message, detail=""):
        super().__init__(message)
        self.detail = detail


class SkipTest(Exception):
    """Feature not configured/available — recorded as SKIP."""


class ServerGone(Exception):
    """Connection-level failure talking to the server."""


class Result:
    def __init__(self, test_id, name, status, duration_s=0.0, note="", detail=""):
        self.test_id = test_id
        self.name = name
        self.status = status  # PASS / FAIL / SKIP / ABORT
        self.duration_s = duration_s
        self.note = note
        self.detail = detail


# ---------------------------------------------------------------- context

class Context:
    def __init__(self, cfg, base_url):
        self.cfg = cfg
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.timeout = float(cfg.get("request_timeout", 180))
        self.model = cfg.get("model", "genie-local")
        self.results: list[Result] = []
        self.server_down = False
        self.down_incident: dict | None = None
        self.notes: dict = {}          # free-form facts for the report
        self.status_snapshots: dict = {}

    # ------------------------------------------------------------ HTTP

    def call(self, method, path, body=None, stream=False, timeout=None,
             params=None):
        """One HTTP request. Raises ServerGone on connection-level failure."""
        url = f"{self.base_url}{path}"
        try:
            return self.session.request(
                method, url, json=body, params=params, stream=stream,
                timeout=timeout or self.timeout)
        except (requests.ConnectionError, requests.Timeout) as e:
            raise ServerGone(f"{method} {path}: {type(e).__name__}: {e}") from e

    def get_json(self, path, params=None, expect=200):
        r = self.call("GET", path, params=params)
        if r.status_code != expect:
            raise CheckFailure(f"GET {path} -> HTTP {r.status_code} (expected {expect})",
                               detail=r.text[:1000])
        return r.json()

    def post_json(self, path, body, expect=200):
        r = self.call("POST", path, body=body)
        if r.status_code != expect:
            raise CheckFailure(f"POST {path} -> HTTP {r.status_code} (expected {expect})",
                               detail=f"request: {json.dumps(body, ensure_ascii=False)[:500]}\n"
                                      f"response: {r.text[:1000]}")
        return r.json()

    def sse_events(self, path, body, timeout=None):
        """POST + parse an SSE stream into a list of dict events (the final
        '[DONE]' marker is returned as the string '[DONE]')."""
        r = self.call("POST", path, body=body, stream=True, timeout=timeout)
        if r.status_code != 200:
            raise CheckFailure(f"POST {path} (stream) -> HTTP {r.status_code}",
                               detail=r.text[:1000])
        events = []
        try:
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                events.append("[DONE]" if payload == "[DONE]" else json.loads(payload))
        except (requests.ConnectionError, requests.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            raise ServerGone(f"stream {path} broke mid-read: {e}") from e
        return events

    def probe_alive(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def snapshot_status(self, label):
        try:
            self.status_snapshots[label] = self.get_json("/v1/server/status")
        except (ServerGone, CheckFailure, Exception):
            self.status_snapshots[label] = {"error": "unavailable"}


# ---------------------------------------------------------------- helpers

def chat_body(ctx, content, **extra):
    body = {"model": ctx.model,
            "messages": [{"role": "user", "content": content}]}
    body.update(extra)
    return body


def stream_text(events, kind="chat"):
    """Concatenated text from SSE events."""
    out = ""
    for e in events:
        if not isinstance(e, dict) or not e.get("choices"):
            continue
        c = e["choices"][0]
        out += (c.get("delta", {}).get("content") or "") if kind == "chat" \
            else (c.get("text") or "")
    return out


def finish_reasons(events):
    return [e["choices"][0].get("finish_reason")
            for e in events if isinstance(e, dict) and e.get("choices")
            and e["choices"][0].get("finish_reason")]


# ---------------------------------------------------------------- test cases
# Each function: takes ctx, returns a PASS note (str). Raises CheckFailure /
# SkipTest / ServerGone otherwise.

def t_health(ctx):
    for path in ("/health", "/v1/health"):
        data = ctx.get_json(path)
        if data.get("status") != "ok":
            raise CheckFailure(f"{path} returned {data}")
    return "both /health and /v1/health ok"


def t_models(ctx):
    data = ctx.get_json("/v1/models")
    ids = [m["id"] for m in data.get("data", [])]
    if "genie-local" not in ids:
        raise CheckFailure(f"'genie-local' missing from /v1/models: {ids}")
    ctx.notes["models"] = ids
    alias = ctx.get_json("/models")
    if [m["id"] for m in alias.get("data", [])] != ids:
        raise CheckFailure("/models alias differs from /v1/models")
    one = ctx.get_json(f"/v1/models/{ids[-1]}")
    if one.get("id") != ids[-1]:
        raise CheckFailure(f"GET /v1/models/{{id}} echoed {one}")
    return f"models: {ids}"


def t_server_status(ctx):
    data = ctx.get_json("/v1/server/status")
    slots = data.get("slots")
    if not isinstance(slots, list) or not slots:
        raise CheckFailure(f"no slots in status: {data}")
    for s in slots:
        for key in ("name", "active_model", "loaded", "phase"):
            if key not in s:
                raise CheckFailure(f"slot report missing '{key}': {s}")
        if not s["loaded"]:
            raise CheckFailure(f"slot '{s['name']}' reports loaded=false")
    ctx.notes["slots"] = [(s["name"], s["active_model"]) for s in slots]
    return f"slots: {ctx.notes['slots']}"


def t_server_idle(ctx):
    data = ctx.get_json("/v1/server/idle")
    if data.get("status") != "idle":
        raise CheckFailure(f"unexpected: {data}")
    return f"slot '{data.get('slot')}' idle"


def t_completion_sync(ctx):
    body = {"model": ctx.model, "prompt": "The capital of Japan is",
            "max_tokens": 16, "temperature": 0}
    data = ctx.post_json("/v1/completions", body)
    if data.get("object") != "text_completion":
        raise CheckFailure(f"object={data.get('object')}")
    choice = data["choices"][0]
    if not choice.get("text", "").strip():
        raise CheckFailure("empty completion text", detail=json.dumps(data)[:800])
    usage = data.get("usage", {})
    if usage.get("prompt_tokens", 0) <= 0 or usage.get("completion_tokens", 0) <= 0:
        raise CheckFailure(f"suspicious usage: {usage}")
    ctx.notes["completion_sample"] = choice["text"][:120]
    return (f"text={choice['text'][:60]!r} finish={choice['finish_reason']} "
            f"usage={usage['prompt_tokens']}/{usage['completion_tokens']}")


def t_completion_echo(ctx):
    prompt = "Q: What is 2+2?\nA:"
    data = ctx.post_json("/v1/completions", {
        "model": ctx.model, "prompt": prompt, "max_tokens": 8,
        "temperature": 0, "echo": True})
    text = data["choices"][0]["text"]
    if not text.startswith(prompt):
        raise CheckFailure("echo=true did not prepend the prompt",
                           detail=text[:300])
    return f"echoed + {len(text) - len(prompt)} chars generated"


def t_completion_batch(ctx):
    data = ctx.post_json("/v1/completions", {
        "model": ctx.model, "prompt": ["One plus one is", "The sky color is"],
        "max_tokens": 8, "temperature": 0})
    choices = data["choices"]
    if [c["index"] for c in choices] != [0, 1]:
        raise CheckFailure(f"expected 2 indexed choices, got {choices}")
    if any(not c["text"].strip() for c in choices):
        raise CheckFailure("a batch choice came back empty")
    return "2 prompts -> 2 choices"


def t_completion_stream(ctx):
    events = ctx.sse_events("/v1/completions", {
        "model": ctx.model, "prompt": "Write one short sentence about the sea.",
        "max_tokens": 48, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True}})
    if events[-1] != "[DONE]":
        raise CheckFailure("stream did not end with [DONE]")
    text = stream_text(events, "completions")
    if not text.strip():
        raise CheckFailure("no streamed text")
    usage = [e for e in events if isinstance(e, dict) and e.get("usage")]
    if len(usage) != 1:
        raise CheckFailure(f"expected exactly 1 usage chunk, got {len(usage)}")
    return f"{len(events)} events, {usage[0]['usage']['completion_tokens']} tokens"


def t_completion_stop(ctx):
    data = ctx.post_json("/v1/completions", {
        "model": ctx.model,
        "prompt": "Count upward from 1, separated by commas: 1, 2, 3,",
        "max_tokens": 64, "temperature": 0, "stop": [" 7"]})
    text = data["choices"][0]["text"]
    if " 7" in text:
        raise CheckFailure("stop sequence ' 7' appeared in the output "
                           "(stop sequences not effective / not trimmed)",
                           detail=text[:300])
    return (f"stop respected (finish={data['choices'][0]['finish_reason']}, "
            f"text={text[:60]!r})")


def t_greedy_determinism(ctx):
    body = {"model": ctx.model, "prompt": "List three colors:",
            "max_tokens": 24, "temperature": 0}
    a = ctx.post_json("/v1/completions", body)["choices"][0]["text"]
    b = ctx.post_json("/v1/completions", body)["choices"][0]["text"]
    if a != b:
        raise CheckFailure("temperature=0 not deterministic",
                           detail=f"run1: {a[:200]!r}\nrun2: {b[:200]!r}")
    return "two greedy runs identical"


def t_max_tokens_finish(ctx):
    data = ctx.post_json("/v1/completions", {
        "model": ctx.model, "prompt": "Tell me a very long story about a dragon.",
        "max_tokens": 8, "temperature": 0})
    ct = data["usage"]["completion_tokens"]
    fr = data["choices"][0]["finish_reason"]
    if ct > 8:
        raise CheckFailure(f"completion_tokens {ct} exceeds max_tokens=8")
    if ct == 8 and fr != "length":
        raise CheckFailure(f"hit max_tokens but finish_reason={fr!r} (expected 'length')")
    return f"completion_tokens={ct} finish_reason={fr}"


def t_chat_sync(ctx):
    data = ctx.post_json("/v1/chat/completions",
                         chat_body(ctx, "Say hello in one short sentence.",
                                   max_tokens=48, temperature=0))
    msg = data["choices"][0]["message"]
    if msg["role"] != "assistant" or not (msg["content"] or "").strip():
        raise CheckFailure(f"bad message: {msg}")
    ctx.notes["chat_sample"] = msg["content"][:120]
    return f"content={msg['content'][:60]!r}"


def t_chat_stream(ctx):
    events = ctx.sse_events("/v1/chat/completions",
                            chat_body(ctx, "Count from one to five in words.",
                                      max_tokens=64, temperature=0, stream=True,
                                      stream_options={"include_usage": True}))
    if events[-1] != "[DONE]":
        raise CheckFailure("stream did not end with [DONE]")
    first = events[0]
    if not (isinstance(first, dict)
            and first["choices"][0]["delta"].get("role") == "assistant"):
        raise CheckFailure("first chunk is not the assistant-role delta",
                           detail=json.dumps(first)[:300])
    text = stream_text(events)
    if not text.strip():
        raise CheckFailure("no streamed content")
    if not finish_reasons(events):
        raise CheckFailure("no finish_reason chunk")
    usage = [e for e in events if isinstance(e, dict) and e.get("usage")]
    if len(usage) != 1:
        raise CheckFailure(f"expected 1 usage chunk, got {len(usage)}")
    return f"{len(events)} events, text={text[:50]!r}"


def t_chat_no_think(ctx):
    data = ctx.post_json("/v1/chat/completions",
                         chat_body(ctx, "What is 3*4? Answer briefly.",
                                   max_tokens=64, temperature=0,
                                   chat_template_kwargs={"enable_thinking": False}))
    content = data["choices"][0]["message"]["content"] or ""
    # Qwen3 may still emit an *empty* think block with /no_think; what must
    # not happen is a long reasoning trace.
    think_body = ""
    if "<think>" in content and "</think>" in content:
        think_body = content.split("<think>", 1)[1].split("</think>", 1)[0].strip()
    if len(think_body) > 40:
        raise CheckFailure("enable_thinking=false but a non-trivial <think> "
                           "block was produced", detail=content[:400])
    return f"ok (think block: {len(think_body)} chars)"


def t_prefix_cache(ctx):
    if not ctx.cfg.get("prefix_cache", {}).get("enabled", True):
        raise SkipTest("disabled in config")
    system = ("You are an integration-test assistant. " * 8
              + f"run-id {uuid.uuid4().hex[:8]}.")
    warm = ctx.post_json("/v1/prefix/warmup",
                         {"model": ctx.model, "system_prompt": system})
    if warm.get("status") not in ("cached", "already_cached"):
        raise CheckFailure(f"warmup status={warm}")
    key = warm["key"]
    entries = ctx.get_json("/v1/prefix/cache")["entries"]
    if key not in [e["key"] for e in entries]:
        raise CheckFailure(f"warmed key {key} not listed in /v1/prefix/cache")

    t0 = time.perf_counter()
    data = ctx.post_json("/v1/chat/completions", {
        "model": ctx.model, "max_tokens": 24, "temperature": 0,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": "Say ready."}]})
    hit_s = time.perf_counter() - t0
    if not (data["choices"][0]["message"]["content"] or "").strip():
        raise CheckFailure("empty response with cached prefix")

    r = ctx.call("DELETE", f"/v1/prefix/cache/{key}")
    if r.status_code != 200:
        raise CheckFailure(f"DELETE cache -> HTTP {r.status_code}")
    return f"warmup+HIT ok (request with cached prefix: {hit_s:.2f}s), entry deleted"


def t_tools(ctx):
    tools = [{"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}]
    data = ctx.post_json("/v1/chat/completions", chat_body(
        ctx, "What is the weather in Tokyo right now? Use the tool.",
        max_tokens=256, temperature=0, tools=tools))
    choice = data["choices"][0]
    calls = choice["message"].get("tool_calls") or []
    if choice["finish_reason"] == "tool_calls":
        fn = calls[0]["function"]
        if fn["name"] != "get_weather":
            raise CheckFailure(f"unexpected tool name: {fn}")
        json.loads(fn["arguments"])  # must be valid JSON
        # Round-trip: send the tool result back.
        followup = ctx.post_json("/v1/chat/completions", {
            "model": ctx.model, "max_tokens": 96, "temperature": 0,
            "messages": [
                {"role": "user", "content": "What is the weather in Tokyo right now? Use the tool."},
                {"role": "assistant", "content": "", "tool_calls": calls},
                {"role": "tool", "tool_call_id": calls[0]["id"],
                 "content": '{"weather": "sunny", "temperature_c": 23}'}]})
        answer = followup["choices"][0]["message"]["content"] or ""
        if not answer.strip():
            raise CheckFailure("empty answer after tool result round-trip")
        return f"tool call + round-trip ok (args={fn['arguments'][:60]})"
    # The model choosing not to call is a model property, not a server bug.
    return (f"server ok; model did not emit a tool call "
            f"(finish={choice['finish_reason']}) — model-dependent")


def t_error_shapes(ctx):
    bad = ctx.session.post(f"{ctx.base_url}/v1/chat/completions",
                           data=b"{not json", timeout=ctx.timeout,
                           headers={"Content-Type": "application/json"})
    if bad.status_code != 400 or "error" not in bad.json():
        raise CheckFailure(f"malformed JSON -> {bad.status_code}: {bad.text[:200]}")
    for body, expect, what in [
        (chat_body(ctx, "hi", n=2), 400, "n>1"),
        (chat_body(ctx, "hi", stream=True, logprobs=True), 400, "stream+logprobs"),
        (dict(chat_body(ctx, "hi"), slot="no-such-slot"), 404, "unknown slot"),
    ]:
        resp = ctx.call("POST", "/v1/chat/completions", body=body)
        if resp.status_code != expect:
            raise CheckFailure(f"{what}: HTTP {resp.status_code} (expected {expect})",
                               detail=resp.text[:300])
        err = resp.json().get("error", {})
        if "message" not in err:
            raise CheckFailure(f"{what}: not an OpenAI error envelope: {resp.text[:200]}")
    return "malformed JSON / n>1 / stream+logprobs / unknown slot all clean"


def t_chat_logprobs(ctx):
    data = ctx.post_json("/v1/chat/completions", chat_body(
        ctx, "Say hello.", max_tokens=16, temperature=0,
        logprobs=True, top_logprobs=2))
    lp = data["choices"][0].get("logprobs")
    if not lp or not lp.get("content"):
        raise CheckFailure(f"no logprobs.content in response: {json.dumps(data)[:400]}")
    mismatches = 0
    for e in lp["content"]:
        if not isinstance(e.get("logprob"), (int, float)) or e["logprob"] > 1e-3:
            raise CheckFailure(f"bad logprob entry: {e}")
        if len(e.get("top_logprobs", [])) != 2:
            raise CheckFailure(f"expected 2 top_logprobs: {e}")
        if e["top_logprobs"][0]["token"] != e["token"]:
            mismatches += 1
    if mismatches:
        raise CheckFailure(
            f"{mismatches}/{len(lp['content'])} greedy tokens are not their "
            f"own top-1 (sampler/logits mismatch?)")
    return f"{len(lp['content'])} tokens, greedy==top1 for all"


def t_completions_logprobs(ctx):
    data = ctx.post_json("/v1/completions", {
        "model": ctx.model, "prompt": "The quick brown fox",
        "max_tokens": 12, "temperature": 0, "logprobs": 2})
    lp = data["choices"][0].get("logprobs")
    if not lp:
        raise CheckFailure("no logprobs object")
    n = len(lp["tokens"])
    if not (n == len(lp["token_logprobs"]) == len(lp["top_logprobs"])
            == len(lp["text_offset"])):
        raise CheckFailure(f"array length mismatch: {[len(lp[k]) for k in lp]}")
    if lp["text_offset"] != sorted(lp["text_offset"]):
        raise CheckFailure("text_offset not monotonic")
    return f"{n} tokens with consistent logprobs arrays"


def t_prompt_scoring(ctx):
    pl_cfg = ctx.cfg.get("prompt_logprobs", {})
    if not pl_cfg.get("enabled", True):
        raise SkipTest("disabled in config")

    # While disabled server-side, echo+logprobs must be a clean 400.
    ctx.post_json("/v1/server/prompt_logprobs", {"enabled": False})
    r = ctx.call("POST", "/v1/completions", body={
        "model": ctx.model, "prompt": "a b c", "echo": True, "logprobs": 1})
    if r.status_code != 400:
        raise CheckFailure(f"echo+logprobs while disabled -> HTTP {r.status_code}")

    ctx.post_json("/v1/server/prompt_logprobs", {"enabled": True})
    try:
        if not ctx.get_json("/v1/server/prompt_logprobs").get("enabled"):
            raise CheckFailure("toggle did not report enabled")
        prompt = pl_cfg.get("prompt",
                            "The Eiffel Tower is located in the city of Paris.")
        t0 = time.perf_counter()
        data = ctx.post_json("/v1/completions", {
            "model": ctx.model, "prompt": prompt, "echo": True,
            "logprobs": 1, "max_tokens": 0})
        dur = time.perf_counter() - t0
        lp = data["choices"][0]["logprobs"]
        n = len(lp["tokens"])
        if lp["token_logprobs"][0] is not None:
            raise CheckFailure("first token's logprob should be null")
        rest = lp["token_logprobs"][1:]
        if not rest or not all(isinstance(v, (int, float)) for v in rest):
            raise CheckFailure(f"non-float logprobs after the first: {rest[:5]}")
        total = sum(rest)
        if total > 0:
            raise CheckFailure(f"positive total loglikelihood {total}")
        greedy_hits = sum(
            1 for tok, top in zip(lp["tokens"][1:], lp["top_logprobs"][1:])
            if top and max(top, key=top.get) == tok)
        ctx.notes["prompt_scoring"] = {
            "tokens": n, "loglikelihood": round(total, 3),
            "greedy_matches": f"{greedy_hits}/{n - 1}",
            "seconds": round(dur, 2), "tok_per_s": round((n - 1) / dur, 1) if dur else 0}
        if data["choices"][0]["finish_reason"] != "length":
            raise CheckFailure(
                f"finish_reason={data['choices'][0]['finish_reason']} (expected length)")
        return (f"{n} tokens scored in {dur:.1f}s "
                f"({(n - 1) / dur:.1f} tok/s), loglik={total:.2f}, "
                f"greedy {greedy_hits}/{n - 1}")
    finally:
        try:
            ctx.post_json("/v1/server/prompt_logprobs", {"enabled": False})
        except (ServerGone, CheckFailure):
            pass


def t_performance_policy(ctx):
    pp = ctx.cfg.get("performance_policy", {})
    if not pp.get("enabled", True):
        raise SkipTest("disabled in config")
    before = ctx.get_json("/v1/server/performance_policy",
                          params={"model": ctx.model})
    ctx.post_json("/v1/server/performance_policy",
                  {"model": ctx.model, "policy": "burst"})
    now = ctx.get_json("/v1/server/performance_policy",
                       params={"model": ctx.model})
    if now.get("policy") != "burst":
        raise CheckFailure(f"policy readback after burst: {now}")
    restore = pp.get("restore", "balanced")
    ctx.post_json("/v1/server/performance_policy",
                  {"model": ctx.model, "policy": restore})
    return f"was {before.get('policy')!r}, burst applied, restored to {restore!r}"


def t_performance_policy_all(ctx):
    """Every Genie_PerformancePolicy_t value round-trips.

    P01 only ever set "burst". The other eight are just as much part of the
    API, and a value the SDK rejects would show up here rather than in
    whatever deployment first tried it. Whether a policy changes anything
    measurable is a separate question — tests/integration/measure_perf_policy.py
    times them; it is not asserted here because a speed assertion on shared
    hardware is a flaky test, not a contract.
    """
    pp = ctx.cfg.get("performance_policy", {})
    if not pp.get("enabled", True):
        raise SkipTest("disabled in config")
    policies = ["burst", "sustained_high_performance", "high_performance",
                "balanced", "low_balanced", "high_power_saver", "power_saver",
                "low_power_saver", "extreme_power_saver"]
    before = ctx.get_json("/v1/server/performance_policy",
                          params={"model": ctx.model}).get("policy")
    accepted, rejected = [], []
    try:
        for name in policies:
            r = ctx.call("POST", "/v1/server/performance_policy",
                         body={"model": ctx.model, "policy": name})
            if r.status_code != 200:
                rejected.append(f"{name}: HTTP {r.status_code} {r.text[:80]}")
                continue
            got = ctx.get_json("/v1/server/performance_policy",
                               params={"model": ctx.model}).get("policy")
            if got != name:
                rejected.append(f"{name}: read back as {got!r}")
            else:
                accepted.append(name)
    finally:
        restore = before or pp.get("restore", "balanced")
        ctx.call("POST", "/v1/server/performance_policy",
                 body={"model": ctx.model, "policy": restore})
    if rejected:
        raise CheckFailure(f"{len(rejected)} of {len(policies)} policies did not "
                           f"round-trip", detail="\n".join(rejected))
    # An unknown name must be refused, or "policy" is not being validated at all.
    bad = ctx.call("POST", "/v1/server/performance_policy",
                   body={"model": ctx.model, "policy": "not_a_policy"})
    if bad.status_code == 200:
        raise CheckFailure("an unknown policy name was accepted with HTTP 200")
    return (f"{len(accepted)}/{len(policies)} round-trip, unknown name -> "
            f"{bad.status_code}, restored to {restore!r}")


def t_model_switch(ctx):
    sw = ctx.cfg.get("switch", {})
    if not sw.get("enabled") or not sw.get("model_dir"):
        raise SkipTest("switch.enabled/model_dir not configured")
    slot = sw.get("slot", "")
    body = {"model_dir": sw["model_dir"],
            "unload_first": bool(sw.get("unload_first", True))}
    if slot:
        body["slot"] = slot
    t0 = time.perf_counter()
    data = ctx.post_json("/v1/models/switch", body)
    load_s = time.perf_counter() - t0
    new_id = data.get("model")
    ids = [m["id"] for m in ctx.get_json("/v1/models")["data"]]
    if new_id not in ids:
        raise CheckFailure(f"switched model {new_id!r} not in /v1/models: {ids}")
    smoke = ctx.post_json("/v1/completions", {
        "model": new_id, "prompt": "Hello", "max_tokens": 8, "temperature": 0})
    if not smoke["choices"][0]["text"].strip():
        raise CheckFailure("empty completion from switched model")

    note = f"switched to {new_id} in {load_s:.1f}s, smoke ok"
    restore = sw.get("restore_model_dir")
    if restore:
        rb = dict(body, model_dir=restore)
        ctx.post_json("/v1/models/switch", rb)
        note += f"; restored {Path(restore).name}"
    return note


def t_lora(ctx):
    lora = ctx.cfg.get("lora", {})
    if not lora.get("enabled") or not lora.get("adapter"):
        raise SkipTest("lora.enabled/adapter not configured")
    engine = lora.get("engine", "primary")
    applied = ctx.post_json("/v1/lora/apply", {
        "model": ctx.model, "engine": engine,
        "lora_adapter_name": lora["adapter"]})
    current = ctx.get_json("/v1/lora/current", params={"model": ctx.model})
    if current.get("lora_adapter_name") != applied.get("lora_adapter_name"):
        raise CheckFailure(f"apply/current mismatch: {applied} vs {current}")
    smoke = ctx.post_json("/v1/completions", {
        "model": ctx.model, "prompt": "Hello", "max_tokens": 8, "temperature": 0})
    if not smoke["choices"][0]["text"].strip():
        raise CheckFailure("empty completion with LoRA applied")
    ctx.post_json("/v1/lora/release", {
        "model": ctx.model, "engine": engine,
        "lora_adapter_name": lora["adapter"]})
    return f"applied+released '{lora['adapter']}' on engine '{engine}'"


def vlm_body(ctx, **extra):
    """The VLM chat request every V0x test sends. Skips the whole group when
    no VLM slot is configured."""
    vlm = ctx.cfg.get("vlm", {})
    if not vlm.get("enabled"):
        raise SkipTest("vlm.enabled not configured")
    image_path = vlm.get("image_path", "")
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode() if image_path \
        else RED_PNG_B64
    body = {
        "model": vlm.get("model") or ctx.model,
        "max_tokens": 96, "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": vlm.get(
                "question", "Describe this image in one sentence.")},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}]}
    if vlm.get("slot"):
        body["slot"] = vlm["slot"]
    body.update(extra)
    return body


def t_vlm(ctx):
    body = vlm_body(ctx)
    t0 = time.perf_counter()
    data = ctx.post_json("/v1/chat/completions", body,)
    dur = time.perf_counter() - t0
    content = data["choices"][0]["message"]["content"] or ""
    if not content.strip():
        raise CheckFailure("empty VLM response")
    ctx.notes["vlm_sample"] = content[:160]
    return f"{dur:.1f}s, response={content[:80]!r}"



def t_vlm_stream(ctx):
    """The VLM SSE path. V01 only ever covered the synchronous one."""
    events = ctx.sse_events("/v1/chat/completions", vlm_body(ctx, stream=True))
    if events and events[-1] != "[DONE]":
        raise CheckFailure("stream did not end with [DONE]")
    chunks = [e for e in events if isinstance(e, dict)]
    if not chunks:
        raise CheckFailure("no SSE chunks")
    first = chunks[0]["choices"][0].get("delta", {})
    if first.get("role") != "assistant":
        raise CheckFailure(f"first chunk must carry delta.role=assistant, got {first}")
    text = stream_text(chunks)
    if not text.strip():
        raise CheckFailure("stream produced no text")
    reasons = finish_reasons(chunks)
    if reasons != ["stop"]:
        raise CheckFailure(f"expected exactly one finish_reason 'stop', got {reasons}")
    bad = {e.get("object") for e in chunks} - {"chat.completion.chunk"}
    if bad:
        raise CheckFailure(f"unexpected chunk object types: {bad}")
    return f"{len(chunks)} chunks, {len(text)} chars, response={text[:60]!r}"


def t_vlm_stream_matches_sync(ctx):
    """Same request both ways must give the same text."""
    sync = ctx.post_json("/v1/chat/completions", vlm_body(ctx))
    sync_text = (sync["choices"][0]["message"]["content"] or "").strip()
    events = ctx.sse_events("/v1/chat/completions", vlm_body(ctx, stream=True))
    stream_txt = stream_text([e for e in events if isinstance(e, dict)]).strip()
    if sync_text != stream_txt:
        raise CheckFailure(
            "streamed text differs from the sync response",
            detail=f"sync:   {sync_text[:300]!r}\nstream: {stream_txt[:300]!r}")
    return f"{len(stream_txt)} chars identical"


def t_vlm_stream_usage(ctx):
    """stream_options.include_usage, and the counts must agree with the sync
    answer to the same request — streaming used to report prompt_tokens 0."""
    sync = ctx.post_json("/v1/chat/completions", vlm_body(ctx))["usage"]
    events = ctx.sse_events("/v1/chat/completions", vlm_body(
        ctx, stream=True, stream_options={"include_usage": True}))
    usage = [e["usage"] for e in events if isinstance(e, dict) and e.get("usage")]
    if not usage:
        raise CheckFailure("include_usage was set but no usage chunk arrived")
    u = usage[-1]
    missing = [k for k in ("prompt_tokens", "completion_tokens", "total_tokens")
               if k not in u]
    if missing:
        raise CheckFailure(f"usage chunk missing {missing}")
    if u["prompt_tokens"] != sync["prompt_tokens"]:
        raise CheckFailure(
            f"streaming prompt_tokens={u['prompt_tokens']} but the same request "
            f"answered synchronously reports {sync['prompt_tokens']}")
    if u["completion_tokens"] <= 0:
        raise CheckFailure(f"completion_tokens={u['completion_tokens']}")
    return f"stream={u} sync={sync}"


def t_vlm_disconnect(ctx):
    """GenieNode exposes no abort call, so hanging up cannot stop a VLM
    generation (F14). What must hold is that the slot comes back on its own."""
    r = ctx.call("POST", "/v1/chat/completions",
                 body=vlm_body(ctx, stream=True), stream=True)
    if r.status_code != 200:
        raise CheckFailure(f"stream -> HTTP {r.status_code}")
    seen = 0
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            seen += 1
            if seen >= 2:
                break
    r.close()
    if seen < 2:
        raise CheckFailure("stream ended before we could disconnect")

    t0 = time.perf_counter()
    data = ctx.post_json("/v1/chat/completions", vlm_body(ctx))
    dur = time.perf_counter() - t0
    if not (data["choices"][0]["message"]["content"] or "").strip():
        raise CheckFailure("slot answered but with empty content after a disconnect")
    return f"slot answered {dur:.1f}s after hanging up mid-stream"


def t_disconnect_aborts(ctx):
    """A client that hangs up must stop the generation, not merely stop
    listening to it. This ran to completion on real transports until the
    stream's cancellation was handled: hanging up 0.4s into an 81s completion
    made the next request wait 82s."""
    long_body = chat_body(
        ctx, "Write a long, detailed essay about the history of computing. "
             "Cover at least ten distinct eras in depth.",
        max_tokens=900, temperature=0, stream=True)
    r = ctx.call("POST", "/v1/chat/completions", body=long_body, stream=True)
    if r.status_code != 200:
        raise CheckFailure(f"stream -> HTTP {r.status_code}")
    seen = 0
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            seen += 1
            if seen >= 3:
                break
    r.close()
    if seen < 3:
        raise CheckFailure("stream ended before we could disconnect")

    t0 = time.perf_counter()
    ctx.post_json("/v1/chat/completions",
                  chat_body(ctx, "Say hi.", max_tokens=8, temperature=0))
    dur = time.perf_counter() - t0
    # A 900-token completion takes far longer than this on any target we run
    # on; waiting that out instead would mean the abort never fired.
    if dur > 20.0:
        raise CheckFailure(
            f"the next request waited {dur:.1f}s — the abandoned generation was "
            "still running, so the disconnect did not abort it")
    return f"next request served {dur:.1f}s after the client hung up"

# ------------------------------------------------------ grammar (G0x)
#
# Grammar-constrained decoding is fixed per model/slot (docs/MANUAL.md), so
# each grammar kind is its own model directory, built by
# setup_grammar_models.py on the target. Every test switches the slot to the
# directory it needs (and skips the switch if it is already loaded), so
# --only G05 works on its own. G08 puts the base model back.

# Mirrors REGEX in setup_grammar_models.py; override via grammar.regex.
REGEX_DEFAULT = r'\{"sentiment": "(positive|negative|neutral)", "score": 0\.[0-9][0-9]\}'


def grammar_cfg(ctx, key=None):
    g = ctx.cfg.get("grammar", {})
    if not g.get("enabled"):
        raise SkipTest("grammar.enabled not configured")
    if key and not g.get(key):
        raise SkipTest(f"grammar.{key} not configured")
    return g


def load_grammar_model(ctx, model_dir, expect=200):
    """Switch the slot to `model_dir` unless it is already loaded. Returns the
    model id (None when `expect` is not 200)."""
    g = ctx.cfg.get("grammar", {})
    want = Path(model_dir).name
    if expect == 200:
        for slot in ctx.get_json("/v1/server/status").get("slots", []):
            if slot.get("active_model") == want:
                return want
    body = {"model_dir": model_dir,
            "unload_first": bool(g.get("unload_first", True))}
    if g.get("slot"):
        body["slot"] = g["slot"]
    r = ctx.call("POST", "/v1/models/switch", body=body)
    if r.status_code != expect:
        raise CheckFailure(
            f"/v1/models/switch({want}) -> HTTP {r.status_code} (expected {expect})",
            detail=r.text[:1000])
    return r.json().get("model") if expect == 200 else None


# Chat-template markers that must never reach a response body. Under a
# grammar the SDK hands the *terminating* token to the token callback before
# the EOS check runs (qualla/dialogs/basic.cpp:157-164), so EOS arrives as
# text; the server has no special-token filter of its own.
EOS_MARKERS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<end_of_turn>")


def split_leaked_eos(text):
    """(text without a trailing EOS marker, the marker or '')."""
    for marker in EOS_MARKERS:
        if marker in text:
            return text.replace(marker, "").strip(), marker
    return text, ""


def constrained_chat(ctx, model, prompt, max_tokens=64):
    """One constrained completion. Grammar violations surface as an aborted
    query (HTTP 500), so a non-200 here is a genuine finding, not plumbing.

    Returns (clean_text, data, leaked_marker) — the constraint is checked
    against clean_text so a leaked EOS marker is reported as its own defect
    instead of masquerading as a grammar failure."""
    r = ctx.call("POST", "/v1/chat/completions", body={
        "model": model, "max_tokens": max_tokens, "temperature": 0,
        "messages": [{"role": "user", "content": prompt}]})
    if r.status_code != 200:
        raise CheckFailure(
            f"constrained chat -> HTTP {r.status_code} "
            "(grammar masking failed, or the query was aborted)",
            detail=r.text[:1000])
    data = r.json()
    raw = (data["choices"][0]["message"]["content"] or "").strip()
    text, leaked = split_leaked_eos(raw)
    return text, data, leaked


def fail_on_leaked_eos(leaked, text, what):
    if leaked:
        raise CheckFailure(
            f"{what} is correct, but {leaked!r} leaked into the response "
            "content (SDK emits the grammar-terminating token as text)",
            detail=f"cleaned output: {text[:300]}")


def check_json_schema_output(text):
    """The schema in setup_grammar_models.py: exactly {answer: str,
    confidence: number}."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise CheckFailure(f"output is not JSON: {e}", detail=text[:500]) from e
    if not isinstance(obj, dict):
        raise CheckFailure(f"output is not a JSON object: {type(obj).__name__}",
                           detail=text[:500])
    if set(obj) != {"answer", "confidence"}:
        raise CheckFailure(f"keys {sorted(obj)} != ['answer', 'confidence'] "
                           "(additionalProperties/required not enforced)",
                           detail=text[:500])
    if not isinstance(obj["answer"], str):
        raise CheckFailure(f"'answer' is {type(obj['answer']).__name__}, not string",
                           detail=text[:500])
    if isinstance(obj["confidence"], bool) or not isinstance(obj["confidence"], (int, float)):
        raise CheckFailure(f"'confidence' is {type(obj['confidence']).__name__}, "
                           "not a number", detail=text[:500])
    return obj


def t_grammar_json_schema(ctx):
    g = grammar_cfg(ctx, "json_schema_model_dir")
    model = load_grammar_model(ctx, g["json_schema_model_dir"])
    text, data, leaked = constrained_chat(
        ctx, model, g.get("prompt", "Is the Earth round? Answer briefly."))
    obj = check_json_schema_output(text)
    ctx.notes["grammar_json_schema_sample"] = text[:200]
    fr = data["choices"][0].get("finish_reason")
    fail_on_leaked_eos(leaked, text, "schema-constrained output")
    return f"valid JSON object {json.dumps(obj)[:80]}, finish_reason={fr}"


def t_grammar_repeat_reset(ctx):
    """The grammar FSM is reset per query (qualla/dialog.cpp:1175-1178). If it
    were not, the second request would start from a terminated state and
    either abort or emit nothing."""
    g = grammar_cfg(ctx, "json_schema_model_dir")
    model = load_grammar_model(ctx, g["json_schema_model_dir"])
    outs, leaks = [], ""
    for prompt in ("Name a primary colour.", "Name a European capital."):
        text, _, leaked = constrained_chat(ctx, model, prompt)
        check_json_schema_output(text)
        outs.append(text)
        leaks = leaks or leaked
    fail_on_leaked_eos(leaks, outs[-1], "both constrained outputs")
    return f"2 consecutive constrained queries both valid: {outs[0][:40]!r}, {outs[1][:40]!r}"


def t_grammar_streaming(ctx):
    g = grammar_cfg(ctx, "json_schema_model_dir")
    model = load_grammar_model(ctx, g["json_schema_model_dir"])
    events = ctx.sse_events("/v1/chat/completions", {
        "model": model, "max_tokens": 64, "temperature": 0, "stream": True,
        "messages": [{"role": "user", "content": "Is water wet? Answer briefly."}]})
    if "[DONE]" not in events:
        raise CheckFailure("stream did not terminate with [DONE]")
    text, leaked = split_leaked_eos(stream_text(events).strip())
    obj = check_json_schema_output(text)
    fail_on_leaked_eos(leaked, text, "streamed schema-constrained output")
    return f"streamed {len(events)} events, reassembled JSON valid: {json.dumps(obj)[:60]}"


def t_grammar_logprobs(ctx):
    """Masked logits reach the custom sampler (grammar masks *before*
    sampler.process, qualla/dialogs/basic.cpp:148-153), so every logprob the
    server reports must still be a finite JSON number."""
    g = grammar_cfg(ctx, "json_schema_model_dir")
    model = load_grammar_model(ctx, g["json_schema_model_dir"])
    r = ctx.call("POST", "/v1/chat/completions", body={
        "model": model, "max_tokens": 48, "temperature": 0,
        "logprobs": True, "top_logprobs": 2,
        "messages": [{"role": "user", "content": "Is fire hot? Answer briefly."}]})
    if r.status_code != 200:
        raise CheckFailure(f"grammar+logprobs -> HTTP {r.status_code}",
                           detail=r.text[:1000])
    if re.search(r"\b(-?Infinity|NaN)\b", r.text):
        raise CheckFailure(
            "response contains Infinity/NaN — masked -inf logits leaked into "
            "the JSON body (invalid JSON for strict clients)",
            detail=r.text[:500])
    entries = r.json()["choices"][0].get("logprobs", {}).get("content") or []
    if not entries:
        raise CheckFailure("no logprobs.content with a grammar attached",
                           detail=r.text[:500])
    return f"{len(entries)} constrained tokens, all logprobs finite"


def t_grammar_regex(ctx):
    g = grammar_cfg(ctx, "regex_model_dir")
    model = load_grammar_model(ctx, g["regex_model_dir"])
    pattern = g.get("regex", REGEX_DEFAULT)
    text, _, leaked = constrained_chat(
        ctx, model, g.get("regex_prompt", "Rate this review: 'I loved it.'"))
    if not re.fullmatch(pattern, text):
        raise CheckFailure(f"output does not match the configured regex: {text!r}",
                           detail=f"pattern: {pattern}")
    ctx.notes["grammar_regex_sample"] = text[:200]
    fail_on_leaked_eos(leaked, text, "regex-constrained output")
    return f"output matches the regex: {text!r}"


def t_grammar_ebnf(ctx):
    g = grammar_cfg(ctx, "ebnf_model_dir")
    model = load_grammar_model(ctx, g["ebnf_model_dir"])
    allowed = g.get("ebnf_alternatives", ["APPROVE", "REJECT", "ESCALATE"])
    text, _, leaked = constrained_chat(
        ctx, model,
        g.get("ebnf_prompt", "The applicant meets every requirement. Decide."))
    if text not in allowed:
        raise CheckFailure(f"output {text!r} is not one of the grammar's "
                           f"alternatives {allowed}")
    ctx.notes["grammar_ebnf_sample"] = text
    fail_on_leaked_eos(leaked, text, "EBNF-constrained output")
    return f"output is a grammar alternative: {text!r}"


def t_grammar_bad_backend(ctx):
    """An unsupported backend is refused by the SDK, not by the server
    (Context.cpp:79). The switch must fail cleanly — and leave the slot
    serving whatever it had."""
    g = grammar_cfg(ctx, "bad_backend_model_dir")
    before = ctx.get_json("/v1/server/status")
    # unload_first=false on purpose (it is no longer the default): that is the
    # mode whose contract is "a bad model_dir never leaves a slot without a
    # working model". Under the default unload_first=true the slot is
    # *expected* to end up empty, so this contract can only be tested here.
    body = {"model_dir": g["bad_backend_model_dir"], "unload_first": False}
    if g.get("slot"):
        body["slot"] = g["slot"]
    r = ctx.call("POST", "/v1/models/switch", body=body)
    if r.status_code != 500:
        raise CheckFailure(
            f"switch to an unsupported grammar backend -> HTTP {r.status_code} "
            "(expected 500)", detail=r.text[:1000])
    after = ctx.get_json("/v1/server/status")
    if not any(s.get("loaded") for s in after.get("slots", [])):
        raise CheckFailure("slot left with no model after a rejected grammar config",
                           detail=json.dumps(after)[:500])
    smoke = ctx.post_json("/v1/completions", {
        "model": ctx.model, "prompt": "Hello", "max_tokens": 8, "temperature": 0})
    if not smoke["choices"][0]["text"].strip():
        raise CheckFailure("slot no longer generates after a rejected grammar config")
    kept = [s.get("active_model") for s in after.get("slots", [])]
    was = [s.get("active_model") for s in before.get("slots", [])]
    return f"rejected with HTTP 500; slot still serving {kept} (was {was})"


def t_grammar_restore(ctx):
    g = grammar_cfg(ctx, "restore_model_dir")
    model = load_grammar_model(ctx, g["restore_model_dir"])
    text, _, _ = constrained_chat(ctx, model, "Say hello in one short sentence.")
    if not text:
        raise CheckFailure("empty completion from the restored model")
    if text.startswith("{") and text.endswith("}"):
        raise CheckFailure(
            "restored model still looks grammar-constrained — the grammar "
            "outlived its Dialog?", detail=text[:300])
    return f"restored {model}, unconstrained output: {text[:60]!r}"


def t_final_alive(ctx):
    data = ctx.get_json("/v1/server/status")
    busy = [s["name"] for s in data.get("slots", []) if s.get("phase") != "idle"]
    if busy:
        return f"alive; note: non-idle slots after run: {busy}"
    return "server alive and idle after the full run"


TESTS = [
    ("S01", "health endpoints",                 t_health),
    ("S02", "model listing + aliases",          t_models),
    ("S03", "server status (slots)",            t_server_status),
    ("S04", "server idle",                      t_server_idle),
    ("C01", "completions: sync",                t_completion_sync),
    ("C02", "completions: echo",                t_completion_echo),
    ("C03", "completions: batch prompts",       t_completion_batch),
    ("C04", "completions: streaming SSE",       t_completion_stream),
    ("C05", "completions: stop sequences",      t_completion_stop),
    ("C06", "greedy determinism (temp=0)",      t_greedy_determinism),
    ("C07", "max_tokens / finish_reason",       t_max_tokens_finish),
    ("CH01", "chat: sync",                      t_chat_sync),
    ("CH02", "chat: streaming SSE + usage",     t_chat_stream),
    ("CH03", "chat: enable_thinking=false",     t_chat_no_think),
    ("CH04", "prefix KV cache warmup/hit",      t_prefix_cache),
    ("CH05", "function calling (tools)",        t_tools),
    ("D01", "client disconnect aborts generation", t_disconnect_aborts),
    ("E01", "error envelope & rejections",      t_error_shapes),
    ("L01", "chat logprobs (top_logprobs)",     t_chat_logprobs),
    ("L02", "completions logprobs",             t_completions_logprobs),
    ("L03", "prompt scoring (echo+logprobs)",   t_prompt_scoring),
    ("P01", "performance policy roundtrip",     t_performance_policy),
    ("P02", "every performance policy round-trips", t_performance_policy_all),
    ("M01", "model hot-swap",                   t_model_switch),
    ("M02", "LoRA apply/release",               t_lora),
    ("V01", "VLM image chat",                   t_vlm),
    ("V02", "VLM chat: streaming SSE",           t_vlm_stream),
    ("V03", "VLM stream matches sync",           t_vlm_stream_matches_sync),
    ("V04", "VLM stream usage accounting",       t_vlm_stream_usage),
    ("V05", "VLM slot recovers after disconnect", t_vlm_disconnect),
    ("G01", "grammar: JSON Schema",             t_grammar_json_schema),
    ("G02", "grammar: FSM reset between queries", t_grammar_repeat_reset),
    ("G03", "grammar: streaming SSE",           t_grammar_streaming),
    ("G04", "grammar: logprobs under masking",  t_grammar_logprobs),
    ("G05", "grammar: regex",                   t_grammar_regex),
    ("G06", "grammar: EBNF",                    t_grammar_ebnf),
    ("G07", "grammar: unsupported backend rejected", t_grammar_bad_backend),
    ("G08", "grammar: restore base model",      t_grammar_restore),
    ("Z01", "final server health",              t_final_alive),
]


# ---------------------------------------------------------------- runner

def run(ctx, only=None):
    ctx.snapshot_status("before")
    aborted = False
    for test_id, name, fn in TESTS:
        if only and test_id not in only:
            continue
        if aborted:
            ctx.results.append(Result(test_id, name, "ABORT",
                                      note="server down earlier in the run"))
            continue
        t0 = time.perf_counter()
        try:
            note = fn(ctx)
            ctx.results.append(Result(test_id, name, "PASS",
                                      time.perf_counter() - t0, note or ""))
            print(f"  PASS  {test_id} {name}  ({time.perf_counter() - t0:.1f}s)")
        except SkipTest as e:
            ctx.results.append(Result(test_id, name, "SKIP",
                                      time.perf_counter() - t0, str(e)))
            print(f"  SKIP  {test_id} {name}: {e}")
        except CheckFailure as e:
            ctx.results.append(Result(test_id, name, "FAIL",
                                      time.perf_counter() - t0, str(e),
                                      detail=e.detail))
            print(f"  FAIL  {test_id} {name}: {e}")
        except ServerGone as e:
            duration = time.perf_counter() - t0
            alive = ctx.probe_alive()
            if alive:
                # transient (e.g. one request timed out but server is up)
                ctx.results.append(Result(
                    test_id, name, "FAIL", duration,
                    f"request failed but server still responds: {e}"))
                print(f"  FAIL  {test_id} {name}: {e} (server still up)")
            else:
                ctx.server_down = True
                ctx.down_incident = {
                    "during_test": f"{test_id} {name}",
                    "error": str(e),
                    "elapsed_in_test_s": round(duration, 1),
                    "last_passed": next(
                        (r.test_id + " " + r.name
                         for r in reversed(ctx.results) if r.status == "PASS"),
                        "(none)"),
                }
                ctx.results.append(Result(test_id, name, "FAIL", duration,
                                          f"SERVER DOWN during this test: {e}"))
                print(f"  FAIL  {test_id} {name}: SERVER APPEARS DOWN ({e})")
                aborted = True
        except Exception as e:  # harness bug — report, keep going
            ctx.results.append(Result(test_id, name, "FAIL",
                                      time.perf_counter() - t0,
                                      f"harness exception: {e}",
                                      detail=traceback.format_exc()[-1500:]))
            print(f"  FAIL  {test_id} {name}: harness exception: {e}")
    if not ctx.server_down:
        ctx.snapshot_status("after")


# ---------------------------------------------------------------- report

def write_report(ctx, report_dir: Path):
    ts = datetime.now(timezone.utc).astimezone()
    stamp = ts.strftime("%Y%m%d-%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=True)
    md_path = report_dir / f"integration_report_{stamp}.md"
    json_path = report_dir / f"integration_report_{stamp}.json"

    counts = {}
    for r in ctx.results:
        counts[r.status] = counts.get(r.status, 0) + 1
    verdict = "SERVER WENT DOWN" if ctx.server_down else (
        "FAILURES" if counts.get("FAIL") else "ALL GREEN")

    lines = [
        "# genie-server integration report",
        "",
        f"- Date: {ts.isoformat(timespec='seconds')}",
        f"- Target: `{ctx.base_url}`",
        f"- Result: **{verdict}** — "
        + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())),
        f"- Models on server: {ctx.notes.get('models', 'n/a')}",
        f"- Slots: {ctx.notes.get('slots', 'n/a')}",
        "",
    ]

    if ctx.server_down and ctx.down_incident:
        inc = ctx.down_incident
        lines += [
            "## ⚠ Server-down incident",
            "",
            f"- The server stopped responding **during {inc['during_test']}** "
            f"(after {inc['elapsed_in_test_s']}s in that test).",
            f"- Error: `{inc['error']}`",
            f"- Last passing test before the incident: {inc['last_passed']}",
            "- Remaining tests were ABORTED.",
            "- Next steps: check the device-side server log (stdout of "
            "genie-server.py) around this time for GenieDialog_query errors, "
            "SDK aborts, or OOM; check `dmesg` for DSP/ION errors; restart "
            "the server and re-run with `--only <failed-id>` to reproduce.",
            "",
        ]

    lines += ["## Results", "",
              "| ID | Test | Status | Time | Notes |",
              "|---|---|---|---|---|"]
    for r in ctx.results:
        icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭", "ABORT": "🚫"}[r.status]
        note = (r.note or "").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {r.test_id} | {r.name} | {icon} {r.status} "
                     f"| {r.duration_s:.1f}s | {note[:200]} |")
    lines.append("")

    fails = [r for r in ctx.results if r.status == "FAIL" and r.detail]
    if fails:
        lines += ["## Failure details", ""]
        for r in fails:
            lines += [f"### {r.test_id} {r.name}", "", f"{r.note}", "",
                      "```", r.detail[:2000], "```", ""]

    interesting = {k: v for k, v in ctx.notes.items()
                   if k in ("prompt_scoring", "vlm_sample", "chat_sample")}
    if interesting:
        lines += ["## Measurements & samples", "",
                  "```json",
                  json.dumps(interesting, ensure_ascii=False, indent=2),
                  "```", ""]

    lines += ["## Server status snapshots", "", "```json",
              json.dumps(ctx.status_snapshots, ensure_ascii=False, indent=2)[:3000],
              "```", ""]

    md_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps({
        "date": ts.isoformat(timespec="seconds"),
        "base_url": ctx.base_url,
        "verdict": verdict,
        "counts": counts,
        "server_down": ctx.server_down,
        "down_incident": ctx.down_incident,
        "results": [vars(r) for r in ctx.results],
        "notes": ctx.notes,
        "status_snapshots": ctx.status_snapshots,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return md_path, json_path, verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="test_config.json",
                    help="test config file (see test_config.sample.json)")
    ap.add_argument("--base-url", default=None,
                    help="override the config's base_url, e.g. http://192.168.1.2:8080")
    ap.add_argument("--only", default=None,
                    help="comma-separated test ids to run (e.g. C01,CH02,L03)")
    ap.add_argument("--report-dir", default="reports",
                    help="directory for the generated reports (default: ./reports)")
    ap.add_argument("--list", action="store_true", help="list test ids and exit")
    args = ap.parse_args()

    if args.list:
        for test_id, name, _ in TESTS:
            print(f"{test_id:5s} {name}")
        return

    cfg_path = Path(args.config)
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
    else:
        print(f"NOTE: {cfg_path} not found — using defaults "
              "(copy test_config.sample.json to customize).")
        cfg = {}
    base_url = args.base_url or cfg.get("base_url", "http://192.168.1.2:8080")

    ctx = Context(cfg, base_url)
    print(f"Target: {ctx.base_url}")
    if not ctx.probe_alive():
        print("ERROR: server is not reachable at "
              f"{ctx.base_url}/health — is genie-server running on the device?")
        sys.exit(2)

    only = set(s.strip() for s in args.only.split(",")) if args.only else None
    started = time.perf_counter()
    run(ctx, only)
    total = time.perf_counter() - started

    md_path, json_path, verdict = write_report(ctx, Path(args.report_dir))
    print(f"\n{verdict} — {total:.0f}s total")
    print(f"Report: {md_path}\n        {json_path}")
    sys.exit(0 if verdict == "ALL GREEN" else 1)


if __name__ == "__main__":
    main()
