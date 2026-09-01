# API Reference

*English | [日本語](./API.ja.md)*

Every endpoint open-genie-server serves, grouped by what it is for. Concepts and
configuration live in [MANUAL.md](./MANUAL.md); this file is the contract.

`base_url = http://<host>:8080` throughout.

| Group | Endpoints |
|---|---|
| [OpenAI-compatible](#openai-compatible-endpoints) | `GET /v1/models` · `GET /v1/models/{id}` · `POST /v1/completions` · `POST /v1/chat/completions` |
| [Server status and control](#server-status-and-control) | `GET /v1/server/status` · `GET /v1/server/idle` · `GET\|POST /v1/server/performance_policy` · `GET\|POST /v1/server/prompt_logprobs` · `GET /v1/server/profile` |
| [Prefix KV cache](#prefix-kv-cache) | `GET /v1/prefix/cache` · `DELETE /v1/prefix/cache/{key}` · `POST /v1/prefix/warmup` |
| [Models and LoRA](#models-and-lora) | `POST /v1/models/switch` · `POST /v1/lora/apply` · `POST /v1/lora/strength` · `POST /v1/lora/release` · `GET /v1/lora/current` |
| [Errors](#error-format) | the envelope every failure uses |

Two conventions apply everywhere:

- The OpenAI-compatible endpoints are **also registered without the `/v1`
  prefix** (`/models`, `/completions`, `/chat/completions`), for clients with
  a misconfigured base URL.
- `GET /health` and `GET /v1/health` return `{"status": "ok"}` (liveness
  probe, vLLM-compatible shape).

Everything outside the first group is this server's own; an OpenAI client
never sees it.

## OpenAI-compatible endpoints

### GET /v1/models

Lists available models: the fixed `genie-local` (an `lm_eval`-compatible placeholder) plus the model ID actually loaded (`active_model_id`) in **every slot**.

```bash
curl $base_url/v1/models
```

```json
{"object": "list", "data": [{"id": "genie-local", ...}, {"id": "tool-model", ...}, {"id": "general", ...}]}
```

### GET /v1/models/{model_id}

Accepts any `model_id` and returns it as-is wrapped in a model object (there's no fixed allow-list for validation).

### POST /v1/completions

Raw text completion (for `lm_eval`'s `local-completions` backend). No chat template or prefix cache is applied. `model` selects the slot.

| Field | Type | Description |
|---|---|---|
| `prompt` | string \| string[] \| int[] \| int[][] | Required. An array of strings runs each prompt **sequentially** and returns one choice per prompt (`index` 0..n-1; non-streaming only). Token-id arrays (what `lm_eval`'s `local-completions` sends with `tokenizer_backend=huggingface`) are decoded with the slot's own tokenizer. |
| `model` | string | Selects which slot to route to (`SlotManager.select`). Falls back to the primary slot if there's no match. **The response does not echo it** — every response and streaming chunk reports the id of the model that actually answered, which is what the selected slot currently holds. The two differ when you route with an alias (`genie-local`, or lm_eval's fixed placeholder) and after a hot-swap, when a slot holds something other than what `env_config.json` names. |
| `slot` | string | Optional. Explicit slot **name** override (e.g. `"chat"`) — takes priority over `model`. Needed when two slots load the same model directory, since `model` alone can't tell them apart (see [Limitations](./MANUAL.md#limitations)). Unknown name → `404`. |
| `stream` | bool | Default `false`. |
| `max_completion_tokens` / `max_tokens` | int | The former takes priority (following OpenAI's deprecation of the latter). If neither is given, defaults to `dialog.context.size` minus the prompt's token count (i.e. remaining context space) — matching Qualcomm's own qai-appbuilder reference server. If `DEFAULT_MAX_TOKENS` (`env_config.json`) is set to a positive value, the smaller of that and remaining context space is used instead — see [Troubleshooting](./MANUAL.md#troubleshooting). |
| `stop` | string \| string[] | Stop sequence(s). Matching runs inside the SDK: partially-matching text is held back during streaming and the matched stop sequence is trimmed from the output (OpenAI semantics). |
| `temperature` / `top_p` / `top_k` | number | Sampling parameters, re-applied per request: a request that omits one gets the **model's own `genie_config.json` default** back (no leakage from the previous request's settings). `temperature=0` means greedy decoding (implemented as `top-k=1` — the SDK's runtime sampler-config path cannot take a literal temp of 0). |
| `seed` | int | Best-effort sampling seed, forwarded to the SDK sampler. |
| `logprobs` | int (0-20) | Returns per-token logprobs for the **generated** tokens, plus that many top alternatives per position (see [Logprobs](./MANUAL.md#logprobs)). Non-streaming only. Combined with `echo`, switches to **prompt scoring** (lm_eval loglikelihood) — gated behind `POST /v1/server/prompt_logprobs`. |
| `suffix` / `best_of` | — | Not supported → `400`. |
| `echo` | bool | If true, prepends the prompt to the response (sent as the first chunk when streaming). |
| `n` | int | Only `1` is supported. `>1` returns `400`. |
| `stream_options.include_usage` | bool | If true, sends a `text_completion` chunk containing `usage` right before `[DONE]`. |

```bash
curl $base_url/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of Japan is", "max_tokens": 16, "temperature": 0}'

# Explicitly targeting the second slot (the model loaded on the chat slot)
curl $base_url/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "general", "prompt": "...", "max_tokens": 16}'
```

`finish_reason` semantics (both here and on `/v1/chat/completions`): `"stop"` for a natural end or a stop-sequence match; `"length"` when the generation hit `max_tokens` or the model's context window; `"tool_calls"` when the response is a function call (chat only, see below). `max_tokens: 0` is only valid together with `echo`+`logprobs` (prompt scoring).

### POST /v1/chat/completions

Chat completion (for `lm_eval`'s `local-chat-completions` backend, Open WebUI, and OpenAI SDKs). Applies the chat template and the prefix cache. `model` selects the slot. Streaming shares its internal implementation (`engine.py` + `_sse_stream`) with `/v1/completions` — only the chunk shape differs (chat vs. text_completion).

Message `content` may be either a plain string or an OpenAI parts array (`[{"type": "text", ...}]`, as sent by Open WebUI); text parts are flattened automatically, and an `image_url` or `video_url` part routes the request to a VLM slot (see [VLM (Multimodal) Support](./MANUAL.md#vlm-multimodal-support)).

A `video_url` carries frames the client already extracted — base64 JPEGs joined by commas under a `video/jpeg` media type, the form vLLM uses for client-side preprocessing — and `media_io_kwargs.video` (top level of the body; `extra_body` from an OpenAI client) carries the `fps`/`frames_indices` the `<t seconds>` markers come from. Four request errors are specific to this path, all `400`:

- **A media type that is not `video/jpeg`.** No demuxer ships with this server, so a container (`data:video/mp4;base64,...`) is refused rather than half-supported.
- **A remote `http(s)` URL**, for video as for images: this server does not fetch them.
- **A `video_url` with no frames in it.**
- **More visual input than the slot's context holds** — only when `VLM_VISION_BUDGET_GUARD` is on, which it is not by default. The message names how many encoder steps fit and why. See [Limiting visual input](./MANUAL.md#limiting-visual-input) for what happens with it off, which is the default and is not graceful.

In addition to `/v1/completions`'s fields:

| Field | Type | Description |
|---|---|---|
| `messages` | array | Required, non-empty. An array of `{"role": "...", "content": "..."}`. |
| `enable_thinking` / `chat_template_kwargs.enable_thinking` | bool | Default `true`. Not an OpenAI-standard field — a Qwen3-specific extension. Accepted both as a flat top-level `enable_thinking` and nested under `chat_template_kwargs` (vLLM/SGLang's convention — they forward that dict into HF's `apply_chat_template()`, and `enable_thinking` is the literal kwarg Qwen3's own chat template reads; `chat_template_kwargs` wins if both are given). `false` appends the literal text `/no_think` to the system prompt (synthesizing an empty one if none was given) — Qwen3's own documented soft-switch for its chat template, which makes the model skip its own reasoning and answer directly. **Not** implemented by pre-seeding an empty `<think>\n\n</think>\n\n` block (HuggingFace's chat-template mechanism) — Qualcomm's own reference server (`qai-appbuilder/samples/genie/c++/Service`) found via real-device testing that doing so causes Qwen3 to degenerate on short prompts (verbatim-repeats the prior turn, then stops), so this server follows their validated `/no_think` approach instead. No effect on templates/models that aren't Qwen3-family — it's pure prompt text, there's no SDK-level reasoning toggle. |
| `tools` | array | OpenAI function-calling tool definitions — see [Function calling](#function-calling-tools) below. |
| `tool_choice` | string | Only `"auto"` (default) and `"none"` (disables tool injection). `"required"` and the `{"type":"function", ...}` named form are **rejected with a `400`** — both guarantee a call in OpenAI's semantics, which needs constrained decoding this server does not implement. Use `"auto"` and check whether the reply actually carries `tool_calls`. |
| `logprobs` / `top_logprobs` | bool / int (0-20) | OpenAI chat logprobs for the generated tokens (`choices[0].logprobs.content[...]`). Non-streaming only. See [Logprobs](./MANUAL.md#logprobs). |

```bash
curl $base_url/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "genie-local",
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Hello"}
        ],
        "stream": true,
        "max_tokens": 128
      }'
```

The streaming response is SSE (`text/event-stream`); per the OpenAI spec, it starts with an empty chunk carrying `delta.role="assistant"`.

#### Function calling (`tools`)

OpenAI's `tools` are a wire format, not a prompt format, so the server renders them into the system prompt in **the dialect the slot's model was trained on**. Two are implemented, and a slot picks one from its chat template (`TOOL_FORMAT` overrides — see [MANUAL](./MANUAL.md#configuration-env_configjson)):

| Dialect | Slots | Declaration | Call |
|---|---|---|---|
| `hermes` (default) | every template but `gemma4` | `<tools>` … `</tools>` | `<tool_call>{"name": …, "arguments": …}</tool_call>` |
| `gemma4` | the `gemma4` template | `<\|tool>declaration:NAME{...}<tool\|>` | `<\|tool_call>call:NAME{...}<tool_call\|>` |

Hermes is the convention Qwen3-class models are actually trained on, and the same approach as Qualcomm's own `qai-appbuilder` GenieAPIService reference. gemma4's is Google's own: not JSON — strings are wrapped in a `<\|"\|>` delimiter and keys come in dictsort order. **The wire shape you send and receive is OpenAI's either way**; only the prompt and the parsing differ. The rest of this section describes Hermes and notes where gemma4 differs.

- Function signatures go inside `<tools></tools>` XML tags appended to the system prompt (one is synthesized if the request has none). This block is part of the cacheable system prefix, so it benefits from the prefix KV cache. gemma4 declares into the system turn the same way, which it can because `system` is its own turn there.
- `<tool_call>{"name": ..., "arguments": ...}</tool_call>` blocks in the model output are parsed into OpenAI `message.tool_calls` (with generated `call_...` ids) and `finish_reason` becomes `"tool_calls"`. Multiple blocks become multiple (parallel) tool calls. Unparseable blocks are left in `content` rather than dropped.
- **A block the model opened but never closed stays text**, unless `TOOL_CALL_RECOVERY` is on. Small models drop the `</tool_call>` tag and emit EOS straight after the JSON (observed on `qwen3_0_6b` w4a16, and on gemma4 with its own markers). It is recoverable — generation has stopped and the JSON is complete — but a model that will not terminate its own call has a defect of the same kind as one that mangles the opening marker, and repairing it by default would make the bundle score better here than on `/v1/completions`, which has no recovery to apply. A generation cut off mid-JSON by `max_tokens` stays in `content` either way: guessing at it would invent arguments.
- **A call whose marker was mangled or omitted is recovered too, if its `name` is one this request declared** (`TOOL_CALL_RECOVERY`, **off by default** — it hides a defect in the bundle you are measuring, and it reaches only this endpoint, never `/v1/completions`). Two models on the SA8255P need it. `qwen3_4b_instruct_2507` w4a16 substitutes Cyrillic for the `<tool_call>` token — `ФРАГМЕНТ`, `Флагорное`, a different string per request — on roughly **half** of its calls, measured over 20 prompts at `temperature: 0`; `qwen3_0_6b` omits the tags entirely and drops bare JSON after its think block, losing about a quarter. The call body is correct in both cases, so without this the caller gets prose with `finish_reason: "stop"` while their code reads `message.tool_calls`. `qwen3_1_7b` loses none and does not need this.
- **The declared-name match is the whole discriminator.** Bare JSON was deliberately left as text when only the closing tag was missing, because a model legitimately answering in JSON would be misread as calling a function; requiring `name` to be a tool the caller actually sent removes that. JSON naming anything else stays in `content`, and so does everything when `TOOL_CALL_RECOVERY` is `false`, the default — which is also the right setting for a model that marks its calls reliably. The mangled marker itself is dropped along with the call: a neighbouring line with no whitespace in it is taken as the marker, which means a genuine one-word line beside a tool call is lost too.
- The tool round-trip history is understood on the way back in: assistant messages with `tool_calls` are re-rendered as `<tool_call>` blocks, and `role: "tool"` result messages are rendered as `<tool_response>` blocks (chatml) / `ipython` turns (llama3), matching the models' own chat templates. A gemma4 slot re-renders both in its own dialect — the call as `<\|tool_call>call:NAME{...}<tool_call\|>`, the result as a user turn holding `<\|tool_response>response:NAME{...}<tool_response\|>`.
- **Streaming (Hermes)**: `<tool_call>` blocks are held back (never leaked to the client as text); after generation completes, one `delta.tool_calls` chunk with the complete calls is emitted, followed by `finish_reason: "tool_calls"`. With `TOOL_CALL_RECOVERY` turned on, a mangled marker is not a tag and so would stream out as content before anything noticed, which is too late to take back — so text is additionally held one line at a time until it is provably neither a call body nor the marker beside one. Prose streams as usual apart from its first word, which waits for the space that proves the line is not a marker.
- **Streaming (gemma4)**: the reply is **buffered whole** and resolved at the end, so a client gets its content and `tool_calls` in the final chunks and **no text before then** — the chunk sequence is a valid SSE stream, but nothing arrives incrementally. Hermes has a hand-written incremental filter and gemma4 does not; the buffered path is what lets a dialect work the day it is written, and swapping in an incremental one later changes nothing a client can see except when the text arrives. Use non-streaming requests for gemma4 unless you specifically want the chunk shape.

```bash
curl $base_url/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
  "tools": [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]
}'
```

Function calling works best with Qwen3-family (chatml) models; for llama3-family and llama2-family the same Hermes block is injected as a best effort, since no dialect of their own is implemented. Whether the model actually emits well-formed calls is a property of the model, not the server — and it varies a lot: the `gemma4-e2b` bundle we measured speaks its own dialect correctly but calls the wrong function often (see [examples/bfcl](../examples/bfcl/README.md#gemma4)).

## Server status and control

### GET /v1/server/status

A non-blocking snapshot of current state. Since `bench_ttft.py` and similar tools assume the top-level `phase`/`detail` exist, those keep returning **the primary slot's (`slots[0]`) values**. A `slots` array with the per-slot breakdown is included as well.

```json
{
  "phase": "idle",
  "detail": "",
  "active_model": "tool-model",
  "active_lora": "",
  "context_occupancy": 128,
  "slots": [
    {"name": "tool_call", "device_id": 0, "active_model": "tool-model", "active_lora": "",
     "loaded": true, "phase": "idle", "detail": "", "context_occupancy": 128},
    {"name": "chat", "device_id": 1, "active_model": "general", "active_lora": "finetune-v2",
     "loaded": true, "phase": "idle", "detail": "", "context_occupancy": 0}
  ],
  "vlm_slots": [
    {"name": "vision", "device_id": 1, "active_model": "qwen3-vl", "spec": "qwen3_vl"}
  ]
}
```

- Each slot's `context_occupancy` is fetched via `GenieDialog_getValue(GENIE_DIALOG_PARAM_CONTEXT_OCCUPANCY)` only if that slot's lock can be acquired immediately (`null` if it can't). This endpoint itself never blocks, even during inference.
- **`loaded` is how you tell a slot that has no model from one that is merely idle.** A `/v1/models/switch` that frees the old model and then fails to load the new one leaves that slot `"loaded": false`, and every endpoint touching it answers `503` until a later switch succeeds.
- `vlm_slots` is always present, empty when none are configured. VLM slots report no `phase` or `context_occupancy`: the composable pipeline exposes neither.

### GET /v1/server/idle

Blocks until the target slot's lock is released (times out after 120s by default with a `503`). Used to synchronize between benchmark phases. `?slot=<name>` selects the target (defaults to the primary slot).

```bash
curl $base_url/v1/server/idle
curl "$base_url/v1/server/idle?slot=chat"
```

### GET /v1/server/performance_policy

Gets the current `Genie_PerformancePolicy_t` (`?model=` selects the slot; non-blocking best-effort, same as `/v1/lora/current`, which also takes `?slot=`).

```json
{"slot": "chat", "policy": "balanced", "raw_value": 40, "live": true}
```

This reports what the SDK was last told, not what the hardware is doing — the getter returns a stored copy rather than querying the device. A successful round trip means the value was accepted, not that anything changed. See [Performance policy](./MANUAL.md#performance-policy) before drawing conclusions from it.

### POST /v1/server/performance_policy

Sets the performance policy. `model` selects the slot. Intended usage: pin to `burst` before running a benchmark, then restore afterward.

**Whether it changes anything is target-dependent, and on at least one target it changes nothing at all** — see [Performance policy](./MANUAL.md#performance-policy).

```bash
curl -X POST $base_url/v1/server/performance_policy \
  -H "Content-Type: application/json" \
  -d '{"policy": "burst"}'
```

Valid values for `policy`:

| Value | `Genie_PerformancePolicy_t` |
|---|---|
| `burst` | `GENIE_PERFORMANCE_BURST` (10) |
| `sustained_high_performance` | 20 |
| `high_performance` | 30 |
| `balanced` | 40 |
| `low_balanced` | 50 |
| `high_power_saver` | 60 |
| `power_saver` | 70 |
| `low_power_saver` | 80 |
| `extreme_power_saver` | 90 |

### GET /v1/server/prompt_logprobs

Returns `{"enabled": bool, "max_tokens": int}` — whether prompt scoring is currently enabled.

### POST /v1/server/prompt_logprobs

Body: `{"enabled": true|false}`. Enable before an lm_eval loglikelihood run, disable after (see [Logprobs](./MANUAL.md#logprobs)).

### GET /v1/server/profile

The SDK's own KPIs for the most recent query on a slot: `time-to-first-token`,
`prompt-processing-rate`, `token-generation-rate` and the token counts behind
them. `?slot=<name>` selects the slot (defaults to the primary one).

Requires `GENIE_PROFILE: true` in env_config.json; returns `409` while it is
off, because the profiler binds to the dialog at creation time and cannot be
switched on at runtime. Not part of the OpenAI surface by design — chat and
completion responses are unchanged whether profiling is on or off. See
[Profiling](./MANUAL.md#profiling-sdk-side-kpis) for the details and the
measured numbers.

```json
{"slot": "chat", "model": "...",
 "summary": {"ttft_ms": 184.6, "prefill_tokens_per_s": 135.4,
             "decode_tokens_per_s": 11.01, "prompt_tokens": 25,
             "generated_tokens": 32, "generation_ms": 2815.6},
 "profile": {"header": {}, "components": []}}
```

`host_measured` carries what the SDK does not profile: `restore_state_ms` and
`save_state_ms` for the prefix cache, timed by the server around the blocking
`GenieDialog_restore`/`_save` calls. Empty until one of them has run. See
[What it costs, and when it pays](./MANUAL.md#what-it-costs-and-when-it-pays).

## Prefix KV cache

### GET /v1/prefix/cache

Lists saved prefix KV cache entries (a directory shared by every slot, logically separated by key).

```json
{"entries": [{"key": "...", "path": "...", "kind": "file", "size_bytes": 12345, "mtime": 1700000000}]}
```

### DELETE /v1/prefix/cache/{key}

Deletes the cache entry for the given key. `404` if it doesn't exist.

### POST /v1/prefix/warmup

Pre-generates the prefix KV cache for a given system prompt. `model` selects the slot.

```bash
curl -X POST $base_url/v1/prefix/warmup \
  -H "Content-Type: application/json" \
  -d '{"system_prompt": "You are a helpful assistant."}'
```

- Warms up against the selected slot's currently-active model/template.
- Returns `422` if the template is llama2/mistral (not splittable).
- Returns `{"status": "already_cached", "slot": "...", ...}` immediately if already cached.
- If warmup takes longer than `WARMUP_JOIN_TIMEOUT` (600s by default), returns `202` and continues in the background (poll `/v1/prefix/cache`). Other slots are never blocked.

> [!NOTE]
> **Verified on hardware** against a bundle carrying LoRA adapters: applying
> one changes generation and is read back live from the SDK, a strength change
> moves it again, releasing reverts it, and an unknown adapter name fails with
> `GenieDialog_applyLora failed: -1`.
>
> Two things are worth knowing before you reach for LoRA. A bundle may ship an
> adapter that is effectively the identity, so selecting it looks like nothing
> happened because nothing did — compare against the released state rather than
> against another adapter. And on a stock **2.49.x or 2.50.x** library, **LoRA cannot be
> used at all on a bundle whose `dialog.type` is `ssd-q1`** — the SDK requires
> a reset after switching adapters, and that reset is what corrupts such a
> dialog there. A patched library, or 2.48.40.260702, has neither problem:
> [D5](./QAIRT_VERSIONS.md#d5--reset-corrupts-a-speculative-decoding-dialog).

> [!IMPORTANT]
> `enable_thinking: false` appends Qwen3's `/no_think` to the system turn, so
> it caches under a **different key**. Warm the variant you will actually send:
> pass the same `"enable_thinking": false` to this endpoint. Warming the raw
> prompt and then sending the flag is a permanent, silent MISS — the request
> works, it just never hits the cache. Measured on the 4B with a 234-token
> system prompt: 361 ms TTFT on a miss against 184 ms on a hit.

## Selecting a slot

Every endpoint that acts on one slot follows the same two-step rule, so it is
written here once rather than at each of them.

1. **An explicit `slot`** — the slot's own name from `TEXT_SLOTS[].name`
   (`"chat"`, `"tool_call"`, …). In a request body for POSTs, `?slot=` for
   GETs. An unknown name is a `404`.
2. **Otherwise `model`** — matched against the model each slot has loaded. A
   name nobody has loaded falls back to the primary slot rather than failing,
   because `lm_eval` sends one fixed placeholder for every request.

`slot` exists because `model` cannot always answer: **two slots holding the
same model directory are indistinguishable by model name**, and the second one
is reachable only by slot. If you never load the same model twice, `model`
alone is enough.

Two GETs predate the rule and accept only one of the two: `/v1/server/idle`
takes `?slot=` and not `?model=`, and `GET /v1/server/performance_policy`
takes `?model=` and not `?slot=`. Everything else accepts both.

## Models and LoRA

### POST /v1/models/switch

Hot-swaps the model loaded in **one hardware slot**.

```bash
curl -X POST $base_url/v1/models/switch \
  -H "Content-Type: application/json" \
  -d '{"slot": "chat", "model_dir": "llama3-8b-htp"}'
```

- `slot`: the target slot name (`env_config.json`'s `TEXT_SLOTS[].name`; `"default"` for a single-slot configuration). Defaults to the primary slot (`slots[0]`) if omitted.
- `model_dir`: a relative path resolves against `MODELS_BASE_DIR` when that is set, and against the server's working directory when it is not; an absolute path is used as given. This is the same rule `TEXT_SLOTS`/`VLM_SLOTS` `model_root` follows at startup, so a bare directory name means the same model in both places. **There is no allow-list and no authentication** — this endpoint will open any path the server process can read, by design (see [Where model paths resolve](./MANUAL.md#where-model-paths-resolve)).
- `config_file`: the dialog config inside `model_dir`. Defaults to the slot's own (`TEXT_SLOTS[].config_file`, itself defaulting to `genie_config.json`) — pass it when the bundle you are switching to names its config differently. It stays with the slot afterwards, so a later switch back needs it again only if that bundle differs too.
- `unload_first` (bool, default `true`): see below.
- By default (`"unload_first": true`), the old `GenieDialog` handle for that slot is freed **before** the new model is loaded, so the new model loads with the HTP device to itself. This is the order that switches reliably. Its cost is that if the new load then fails, the slot ends up with **no model loaded at all** (`GenieDialog_query`/etc. on it return `503` — see [Limitations](./MANUAL.md#limitations)) until a later switch succeeds.
- `"unload_first": false` loads the new model **first** and frees the old handle only after it succeeds, so a failed load leaves the slot still serving its previous model. That is a real advantage, but it needs room on the HTP device for **both** models at once (~2x the smaller model's working set), and **on the SA8255P board that overlap is not dependable**: measured over 36 swaps, the outcome did not follow from which models were involved — one pair succeeded 6 times out of 6 in one run and failed 8 out of 8 in another, and a single run of six flipped from failing to succeeding halfway through. What decides it is device state the host cannot observe. It can also fail outright with SDK-level buffer-registration errors (e.g. `memRegister ERROR(8003)`) on an otherwise-healthy device, when there is simply never room for two instances at once.
- **Use `false` only where the device has memory to spare and you have tested the specific swaps your deployment performs**, repeatedly and from a cold start. If they are not reliable there, keep the default and handle the empty-slot window instead.
- The new model **inherits the same slot's `device_id`** (a slot's hardware assignment never changes when its model is swapped).
- Only acquires **that target slot's own lock**, so it waits for any in-flight inference on that slot to finish before switching (default timeout: 600s), but **never blocks inference on other slots**.
- On success, that slot's prefix KV cache namespace switches automatically too (no explicit cache clear needed).
- On success, that slot's currently-applied LoRA is automatically reset (`active_lora_adapter = ""`).

Response:

```json
{"status": "switched", "slot": "chat", "model": "llama3-8b-htp", "template": "llama3"}
```

On failure: `400` (`model_dir` not given) / `404` (unknown slot name, or the directory has no `genie_config.json`) / `500` (SDK load failure — with the default `unload_first`, the slot is now unloaded; with `"unload_first": false` it keeps serving the old model, see above) / `503` (timed out acquiring the target slot's lock).

### POST /v1/lora/apply

Applies a LoRA adapter. Automatically calls `GenieDialog_reset` afterward. `slot` selects the slot, `model` routes by loaded model name (see [Selecting a slot](#selecting-a-slot)).

```bash
curl -X POST $base_url/v1/lora/apply \
  -H "Content-Type: application/json" \
  -d '{"model": "genie-local", "engine": "primary", "lora_adapter_name": "my-lora"}'
```

- `engine` is the target engine role name (an engine config name from `genie_config.json`; usually `"primary"` in a single-engine configuration). Defaults to `"primary"`.
- The applied result is read back from the SDK (`GenieDialog_getValue`) and included in the response.

### POST /v1/lora/strength

Changes a LoRA's alpha strength (calls `GenieDialog_reset` afterward). `slot` selects the slot, `model` routes by loaded model name (see [Selecting a slot](#selecting-a-slot)).

```bash
curl -X POST $base_url/v1/lora/strength \
  -H "Content-Type: application/json" \
  -d '{"engine": "primary", "tensor_name": "lora_alpha_0", "alpha": 0.8}'
```

### POST /v1/lora/release

Frees a LoRA adapter's memory (calls `GenieDialog_reset` afterward). Re-applying it later requires reloading from disk, which is slow. `slot` selects the slot, `model` routes by loaded model name (see [Selecting a slot](#selecting-a-slot)).

```bash
curl -X POST $base_url/v1/lora/release \
  -H "Content-Type: application/json" \
  -d '{"engine": "primary", "lora_adapter_name": "my-lora"}'
```

### GET /v1/lora/current

Gets the name of the currently-applied LoRA adapter, read live from the SDK (`""` means the base model). `?slot=` selects the slot and `?model=` is the fallback, as in [Selecting a slot](#selecting-a-slot). Tries to acquire the target slot's lock for up to 1 second; if it can't, returns the cached value with `"live": false` (so this never blocks inference).

```json
{"slot": "chat", "lora_adapter_name": "my-lora", "live": true}
```

## Error Format

**Every** error — parameter validation, unknown slots, SDK failures, even malformed JSON bodies and internal `HTTPException`s — is returned in the OpenAI error envelope, so OpenAI SDKs / lm_eval / LiteLLM can always parse failures:

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": "model_dir", "code": null}}
```

Main status codes:

| Code | Meaning |
|---|---|
| `400` | Invalid request parameters (a required field is missing, `n>1`, etc.) |
| `404` | A resource doesn't exist (prefix cache key, model directory, unknown slot name) |
| `422` | Semantically impossible to process (e.g. a prefix warmup request on the llama2 template) |
| `500` | An SDK call failed, or a model load failed |
| `503` | Timed out acquiring the target slot's lock (that slot is busy) |
| `504` | Inference timed out |

A failure in the middle of an SSE stream (after tokens have already been sent) is delivered as a final `data: {"error": {...}}` event before `data: [DONE]` (vLLM-style).

