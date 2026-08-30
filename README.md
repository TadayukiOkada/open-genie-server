# open-genie-server

*English | [日本語](https://github.com/TadayukiOkada/open-genie-server/blob/master/README.ja.md)*

A single-process FastAPI server that exposes the Qualcomm Genie C API (`libGenie.so`) as an OpenAI-compatible REST API. It lets you drive LLMs running on a [Hexagon NPU](https://github.com/TadayukiOkada/open-genie-server/blob/master/docs/MANUAL.md#glossary) — Qualcomm's name for the accelerator this server talks to through the QNN HTP backend — from ordinary OpenAI-compatible HTTP clients — `lm_eval`, `curl`, the OpenAI SDK, [Open WebUI](https://github.com/open-webui/open-webui), and so on. The implementation lives in the `genie_server` package (`src/genie_server/`); `genie-server.py` is the launcher.

> [!IMPORTANT]
> This repository contains only the open-genie-server source itself. To run it you also need the **QAIRT SDK** from Qualcomm (a toolchain distributed under Qualcomm's proprietary license, containing `libGenie.so`) and a model compiled for a Hexagon NPU (a model directory containing `genie_config.json`). Neither the SDK nor any models are included in this repository — obtain them separately from [Qualcomm AI Hub](https://aihub.qualcomm.com/) or similar.

See [MANUAL.md](https://github.com/TadayukiOkada/open-genie-server/blob/master/docs/MANUAL.md) for configuration and behaviour, [API.md](https://github.com/TadayukiOkada/open-genie-server/blob/master/docs/API.md) for the endpoint reference, and [Platform Notes](https://github.com/TadayukiOkada/open-genie-server/blob/master/docs/PLATFORM_NOTES.md) for what every measured number here assumes about the device it was measured on.

## What this is for

**A bench instrument for the Genie C API and for quantized model bundles — not a production inference server.** Everything else follows from that, in this order:

1. **Reach as much of the Genie C API as the API allows.** Not just chat: SDK-side profiling counters, performance policies, prompt scoring through the custom-sampler hook, LoRA, prefix-cache snapshots, and the composable `GenieNode`/`GeniePipeline` path behind VLM slots are all reachable over HTTP, because the point is to exercise them.
2. **Make standard benchmarks easy to point at a device.** `lm_eval` works against an unmodified install, generation and loglikelihood tasks alike, and [examples/bfcl](https://github.com/TadayukiOkada/open-genie-server/tree/master/examples/bfcl) drives the Berkeley Function Calling Leaderboard the same way. A benchmark that needs the server patched is a benchmark you will not run.
3. **Do not hide the SDK's or the model's problems by default.** A defect you cannot see is one you will ship. So the stock-library slot wedge is neither detected nor papered over; grammar's leaked terminal token is reported rather than stripped; a response names the model actually loaded instead of echoing the string the client sent; and the prefix cache fills only on an explicit warmup, so it cannot quietly improve a TTFT measurement. Workarounds exist, but they are switches you turn on knowing what they conceal — `TOOL_CALL_RECOVERY`, which reassembles a tool call whose marker the model mangled, is off until you ask for it.
4. **Stay compatible with the OpenAI API, and with what other inference servers do — where that does not conflict with 3.** Where the two disagree, this server reports what happened. The `model` field above is the worked example: OpenAI and vLLM echo the request, and this server does not, because a benchmark that ran against a hot-swapped model should say so.

**Backward compatibility is not one of these.** Nothing gets broken for the sake of it, and a change to an existing response shape or default is called out under Breaking in the [CHANGELOG](https://github.com/TadayukiOkada/open-genie-server/blob/master/docs/CHANGELOG.md). But this server is a view onto the Genie C API, and when that API moves, this follows — keeping an old shape alive to spare an existing caller would make the instrument lie about what the SDK now does. The same goes for our own defaults: a measurement showing that one of them hides something is reason enough to change it. If you need a surface that holds still, pin a version.

**What it is not.** There is no authentication, no rate limiting and no multi-process scaling, `POST /v1/models/switch` will open any path the process can read, and one text slot serializes its requests behind a single `GenieDialog` handle. Run it on a bench network you control ([SECURITY.md](https://github.com/TadayukiOkada/open-genie-server/blob/master/SECURITY.md) says what that means, and what is worth reporting). If you need a production serving stack on Hexagon, this is the wrong starting point — but it will tell you, in detail, what your bundle and your SDK actually do.

## Features

- `/v1/completions` / `/v1/chat/completions` — OpenAI-compatible text/chat completion (streaming supported; also registered without the `/v1` prefix)
- **Function calling (`tools`)** — Hermes-format tool calling for Qwen3-class models: `<tool_call>` output is parsed into OpenAI `message.tool_calls` / `finish_reason: "tool_calls"`, held back correctly during streaming
- Works out of the box with `lm_eval` (`local-completions` / `local-chat-completions`; token-id prompts are decoded server-side)
- **Logprobs** via the SDK's custom-sampler hook: per-token `logprobs`/`top_logprobs` for generated tokens (a few ms/token overhead, zero when unused), and **prompt scoring** (`echo`+`logprobs` teacher forcing) that makes lm_eval loglikelihood tasks (hellaswag, arc, mmlu, ...) work — gated behind `POST /v1/server/prompt_logprobs` since it runs at decode speed
- Open WebUI-friendly: parts-array `content` flattening, `GET /health`, CORS, streaming `usage` chunks (`stream_options.include_usage`)
- Prefix KV cache for system prompts (namespaced per model/LoRA)
- LoRA adapter hot-swapping via `GenieDialog_applyLora` etc. — apply, strength, release and read-back, verified on hardware
- Model hot-swapping via `/v1/models/switch` (by default the old model is freed before the new one loads, which is the order that switches reliably; a failed load then leaves the slot empty. `"unload_first": false` keeps the old model as a fallback by holding both at once — see the caveat below before using it)
- `Genie_PerformancePolicy_t` switching (e.g. pin to `burst` for benchmarking)
- Non-blocking status monitoring, including context occupancy (KV cache usage)
- **SDK-side profiling** — `GENIE_PROFILE` exposes Genie's own TTFT / prefill / decode KPIs on `GET /v1/server/profile`, without touching the OpenAI response shapes
- **Multi text-slot support** — `TEXT_SLOTS` lets you assign an independent `GenieDialog` handle (its own lock, optionally its own model) to each Hexagon NSP core you can use (cdsp0/cdsp1; see the [Glossary](https://github.com/TadayukiOkada/open-genie-server/blob/master/docs/MANUAL.md#glossary) if HTP/NSP/cDSP/NPU are unfamiliar). Requests to different slots do overlap, but the measured gain on our bench was **~1.3×, not 2×** (see [Multi Text Slots](https://github.com/TadayukiOkada/open-genie-server/blob/master/docs/MANUAL.md#multi-text-slots)). **How many cores you may use is licensed per SKU, not implied by the part number** — see [Platform Notes](https://github.com/TadayukiOkada/open-genie-server/blob/master/docs/PLATFORM_NOTES.md)
- **Grammar-constrained decoding** — constrain output with JSON Schema/regex/EBNF (XGrammar backend, fixed per model/slot)
- **VLM (multimodal) support** — image-input models such as Qwen3-VL that use the `GenieNode`/`GeniePipeline` composable pipeline API can be added via `VLM_SLOTS`, entirely in parallel with `TEXT_SLOTS` (see [examples/vlm](https://github.com/TadayukiOkada/open-genie-server/tree/master/examples/vlm))
- Offline test suite (`tests/`, pytest + a fake SDK) — the whole HTTP/engine/template stack runs without an NPU

## Requirements

- Python 3.10+ (uses `int | None`-style type syntax)
- The QAIRT SDK (`libGenie.so` and its dependencies) and a model that runs on the Hexagon NPU (see the note above)

```bash
pip install .[logprobs,vlm]     # everything
pip install .                   # server only: fastapi, uvicorn, tokenizers
```

The distribution is named `open-genie-server`; the package you import is
`genie_server`. `pip install -r requirements.txt` still works and is the same
as the first line above.

| | in | why |
|---|---|---|
| `fastapi`, `uvicorn` | core | the server |
| `tokenizers` | core | accurate token counts. Without it, counts come from `text.split()`, so a 55-token Japanese paragraph counts as 1 — and that feeds the context check and the default `max_tokens`, not just the reported usage. See [Token counting](https://github.com/TadayukiOkada/open-genie-server/blob/master/docs/MANUAL.md#token-counting) |
| `numpy` | `[logprobs]`, `[vlm]` | **logprobs and prompt scoring**, and VLM. Without it those requests are rejected with HTTP 400 |
| `pillow` | `[vlm]` | image input |
| `pytest`, `httpx`, `requests`, `jsonschema` | `[test]` | the offline suite |

Installing the package also gives you a `genie-server` command; the
repository-root `genie-server.py` launcher does the same thing and needs no
install. On Android three of these publish no wheel — see
[Running on Android](https://github.com/TadayukiOkada/open-genie-server/blob/master/docs/MANUAL.md#running-on-android).

> [!WARNING]
> **Check which QAIRT version you are pointing at before deploying.** Which SDK
> defects you inherit depends on that version, and **every 2.49.x we have tested
> carries three of them in one place: what `GenieDialog_reset()` fails to put
> back.** A server resets between requests to keep them independent, so that is
> the path under every request you serve.
>
> You would see one oversized request wedging a slot for good; a long request
> failing on an empty context because the one before it was shorter; or, on a
> bundle built for speculative decoding, every answer after the first reset
> coming back fluent and wrong. **All three report success.**
>
> **[QAIRT Version Issues](https://github.com/TadayukiOkada/open-genie-server/blob/master/docs/QAIRT_VERSIONS.md)** has the per-version
> matrix, [a check you can run against your own SDK](https://github.com/TadayukiOkada/open-genie-server/blob/master/docs/QAIRT_VERSIONS.md#checking-your-own-sdk),
> and how to choose a library. A version not listed there is one we have not
> tested, not one we know to be clean.

## Quick start

> [!NOTE]
> The examples in this section reach the device at `192.168.1.2:8080`. That
> address is not special to this server — it is the default the SA8255P's LV GVM
> comes up with, and it is what our own bench uses, so it appears throughout the
> docs and in `tests/integration/test_config.sample.json`. Replace it with your
> device's address (or `localhost` if you are running the server and the client
> on the same machine).

1. Place an `env_config.json` in the server's startup (current) directory.

   ```json
   {
     "QAIRT_SDK_ROOT": "/path/to/qairt-dir",
     "HEXAGON_VERSION": "v73",
     "MODELS_BASE_DIR": "/path/to/models",
     "PREFIX_CACHE_DIR": "/path/to/prefix_cache",
     "TEXT_SLOTS": [{"model_root": "model-dir"}]
   }
   ```

   `model_root` points at the directory containing `genie_config.json`. That is the only key a slot needs; `name` and `device_id` default.

   A relative `model_root` resolves under `MODELS_BASE_DIR`, so the example above loads `/path/to/models/model-dir`. Absolute paths also work — see [Where model paths resolve](https://github.com/TadayukiOkada/open-genie-server/blob/master/docs/MANUAL.md#where-model-paths-resolve).

   On a SoC with multiple NSP cores, add an entry per core to keep a model resident on each (see [MANUAL.md](https://github.com/TadayukiOkada/open-genie-server/blob/master/docs/MANUAL.md#multi-text-slots) for details, including the ordering rule — a second slot does not always fit):

   ```json
   {
     "QAIRT_SDK_ROOT": "/path/to/qairt-dir",
     "HEXAGON_VERSION": "v73",
     "MODELS_BASE_DIR": "/path/to/models",
     "PREFIX_CACHE_DIR": "/path/to/prefix_cache",
     "TEXT_SLOTS": [
       {"name": "tool_call", "device_id": 0, "model_root": "model-fast"},
       {"name": "chat", "device_id": 1, "model_root": "model-general"}
     ]
   }
   ```

2. Start the server.

   ```bash
   python3 genie-server.py            # flags: --config/--host/--port
   # or, if the package is installed
   genie-server                       # same flags
   # or under uvicorn directly
   uvicorn genie_server.asgi:app --host 0.0.0.0 --port 8080 --workers 1
   ```

3. Sanity-check it:

   ```bash
   curl http://192.168.1.2:8080/v1/models

   curl http://192.168.1.2:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"genie-local","messages":[{"role":"user","content":"Hello"}]}'
   ```

See [examples/grammar](https://github.com/TadayukiOkada/open-genie-server/tree/master/examples/grammar) for a grammar-constrained decoding config example, [examples/vlm](https://github.com/TadayukiOkada/open-genie-server/tree/master/examples/vlm) for VLM (image input) setup and testing steps, and [examples/lm_eval](https://github.com/TadayukiOkada/open-genie-server/tree/master/examples/lm_eval) for running `lm_eval` against a board (including how to compare the result with the unquantized model).

## Using it with lm_eval

```bash
lm_eval --model local-chat-completions \
  --model_args model=genie-local,base_url=http://192.168.1.2:8080/v1,\
tokenizer_backend=huggingface,tokenizer=<hf_model>,max_tokens=512,num_concurrent=1 \
  --tasks mmlu_generative --apply_chat_template --batch_size 1
```

See [API.md](https://github.com/TadayukiOkada/open-genie-server/blob/master/docs/API.md) for the endpoint reference and [MANUAL.md](https://github.com/TadayukiOkada/open-genie-server/blob/master/docs/MANUAL.md) for configuration.

## Directory layout

```
genie-server.py       — launcher (CLI flags: --config/--host/--port)
src/genie_server/     — the server implementation, one module per concern
pyproject.toml        — packaging: dependencies, extras, the genie-server command
requirements.txt      — kept as a pointer to pyproject.toml's dependencies
SECURITY.md           — what is deliberately absent, and what to report (.ja.md)
LICENSE

tests/                — offline test suite (pytest; fake SDK, no NPU needed)
tests/integration/    — host-side runner against a live device (MD/JSON reports)

docs/MANUAL.md        — configuration, behaviour, and the why behind it (.ja.md)
docs/API.md           — every endpoint, grouped by purpose (.ja.md)
docs/PLATFORM_NOTES.md— what the measured numbers assume about a device (.ja.md)
docs/QAIRT_VERSIONS.md— the SDK defects, per QAIRT version (.ja.md)
docs/CHANGELOG.md     — release notes

examples/config/      — env_config.json samples (single-slot / dual-NSP / VLM)
examples/grammar/     — grammar-constrained decoding
examples/vlm/         — VLM (multimodal) setup and testing steps
examples/lm_eval/     — lm_eval, and comparing against the unquantized model
examples/bfcl/        — the Berkeley Function Calling Leaderboard
```

The module-by-module breakdown of `src/genie_server/` is in
[MANUAL.md § Architecture Overview](https://github.com/TadayukiOkada/open-genie-server/blob/master/docs/MANUAL.md#architecture-overview) rather than repeated
here.

Run the offline tests with:

```bash
pip install -e .[logprobs,vlm,test]
python3 -m pytest tests/
```

The `[test]` dependencies are not optional for a green run: without `requests`
and `jsonschema` eight grammar tests fail with `ModuleNotFoundError`, and
without `numpy` eight logprobs tests do. The same suite runs on every push and
pull request against Python 3.10 and 3.12
(`.github/workflows/offline-tests.yml`).

To exercise a real device end-to-end from the host PC (with a Markdown/JSON report and server-death detection), see [tests/integration/](https://github.com/TadayukiOkada/open-genie-server/tree/master/tests/integration).

## Known limitations

- **This server does not guard against the stock-library reset defects** described at the top of this page — it neither detects nor recovers from them. Avoiding them is a deployment choice: see [QAIRT Version Issues](https://github.com/TadayukiOkada/open-genie-server/blob/master/docs/QAIRT_VERSIONS.md).
- **A bundle built for speculative decoding (`"dialog": {"type": "ssd-q1"}`) needs a patched library**, and cannot use LoRA without one. See [D5](https://github.com/TadayukiOkada/open-genie-server/blob/master/docs/QAIRT_VERSIONS.md#d5--reset-corrupts-a-speculative-decoding-dialog) for why, and for the one-line change to the bundle that avoids it.
- One text slot = one `GenieDialog` handle; requests to a slot are serialized by that slot's own lock (with `TEXT_SLOTS` unset there's only a single slot, so every request is serialized, same as before).
- `n > 1` (multiple completions per request) is not supported; it is rejected with a `400`.
- The Llama2/Mistral template folds the system prompt into `[INST]`, so it's not eligible for prefix KV caching.
- `POST /v1/models/switch` frees the old model before loading the new one by default, so a failed load leaves that slot with no model until a later switch succeeds; every endpoint that touches it returns `503` in the meantime.
- `"unload_first": false` avoids that by holding both models on the slot's HTP device while the new one loads — but **on the SA8255P board that overlap is not dependable**. Over 36 measured swaps the outcome did not follow from which models were involved: one pair succeeded 6/6 in one run and failed 8/8 in another, and a run of six flipped from failing to succeeding halfway through. What decides it is device state the host cannot observe. Use it only where the device has memory to spare and the swaps your deployment actually performs have been tested there, repeatedly and from a cold start.
- VLM slots are single-turn only (no conversation history), and don't support LoRA, prefix KV cache, grammar constraints, or hot-swapping. See [MANUAL.md](https://github.com/TadayukiOkada/open-genie-server/blob/master/docs/MANUAL.md#vlm-multimodal-support) for details.

See [MANUAL.md's Limitations section](https://github.com/TadayukiOkada/open-genie-server/blob/master/docs/MANUAL.md#limitations) for the rest of the known limitations.

## Acknowledgements

Built with [Claude Code](https://claude.com/claude-code). The first commit here
is dated 2026-08-19, and in the eleven days since, this went from a single
2,700-line script to a packaged server with 311 offline tests, a hardware
integration suite, and manuals in two languages.

The code was never the slow part. What took the time was reading the QAIRT
SDK's reference sources closely enough to tell an SDK defect from a bug of our
own, reproducing each one on the board until it was certain which it was, and
then writing down what had been measured rather than what had been assumed —
several of the findings in these documents reverse an earlier conclusion that
looked obvious at the time. Doing that at this pace, as one engineer, would not
have been possible without it.

## License

[MIT](https://github.com/TadayukiOkada/open-genie-server/blob/master/LICENSE)

This repository's license applies only to the open-genie-server source code itself. The Qualcomm QAIRT SDK, `libGenie.so`, and any Hexagon NPU models are not covered — they remain subject to Qualcomm's and each model's own distributor's license terms.

**MIT was a choice, not a default.** A bench instrument is the kind of project
people expect to find under a copyleft licence, so it is worth saying why this
one is not. Two reasons, both specific to what this is. It loads a proprietary
`libGenie.so` through ctypes at run time, and copyleft would put a
combined-work question in front of anyone shipping a board image that carries
both — a normal way to ship on this hardware, and not a question worth handing
to someone's legal team before they can measure a model. And the useful thing
to do with this code is take it apart: lift `logprobs.py` into your own
evaluation harness, fork it for the one endpoint your board needs. A licence
that taxes that is working against the point.
