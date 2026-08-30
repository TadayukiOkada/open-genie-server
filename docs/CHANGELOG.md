# Changelog

## 1.0.0 — first public release

open-genie-server exposes the Qualcomm Genie C API (`libGenie.so`) as an
OpenAI-compatible REST API, so that a model running on a Hexagon NPU can be
driven from `lm_eval`, `curl`, the OpenAI SDK, Open WebUI and anything else
that speaks that protocol. It is a bench instrument for the SDK and for
quantized bundles rather than a production serving stack — see
[What this is for](../README.md#what-this-is-for), which also says what it
deliberately does not do.

Developed and measured against an SA8255P board over August 2026. Everything
below has run on that hardware unless it says otherwise.

### What it does

- **OpenAI endpoints.** `/v1/completions` and `/v1/chat/completions`, streaming
  and not, with `tools`, `logprobs`, `stop`, and prompt scoring
  (`echo` + `logprobs`) for `lm_eval`'s loglikelihood tasks. Registered with
  and without the `/v1` prefix.
- **Chat templates** per loaded model: chatml, llama3, llama2, gemma, gemma4 —
  chosen from the model in the slot, never from the request's `model` field,
  which `lm_eval` sets to one fixed placeholder.
- **Tool-call dialects.** Hermes (`<tool_call>` JSON) and gemma4's own tokens.
  A slot's dialect follows its chat template; `TOOL_FORMAT` overrides.
- **Multiple text slots.** One `GenieDialog`, lock and KV state per Hexagon NSP
  core, so requests to different slots overlap. Measured at ~1.31× on two
  cores, not 2× — see [Multi Text Slots](MANUAL.md#multi-text-slots).
- **VLM slots** for image input through the `GenieNode`/`GeniePipeline`
  composable API, configured separately from text slots.
- **Model and LoRA hot-swapping**, prefix KV cache for system prompts,
  grammar-constrained decoding, SDK-side profiling, and performance policies.
- **An offline test suite** that runs the whole HTTP/engine/template stack
  against a fake SDK, with no NPU and no `libGenie.so`, plus a host-side
  integration runner for a real device.

### What you need to know before deploying

- **Check which QAIRT version you are pointing at.** Every 2.49.x we have
  tested carries three defects in one place — what `GenieDialog_reset()` fails
  to put back — and a server resets between requests. One wedges a slot
  permanently, one fails a long request after a shorter one, and one corrupts
  every reply on a speculative-decoding bundle. All three report success.
  [QAIRT Version Issues](QAIRT_VERSIONS.md) has the per-version matrix, a
  check you can run against your own SDK, and patches that fix all three at
  the cost of grammar-constrained decoding.
- **This server does not hide model or SDK defects by default.** Repairs
  exist — `TOOL_CALL_RECOVERY` reassembles a call whose marker the model
  mangled or never closed — and they are off until you turn them on, because a
  bundle that measures better here than on `/v1/completions` is a bundle you
  are measuring wrong.
- **There is no authentication and no rate limiting**, and
  `POST /v1/models/switch` will open any path the server process can read. Run
  it on a network you control. [SECURITY.md](../SECURITY.md) says what is
  worth reporting.
- **Backward compatibility is not a goal.** This follows the Genie C API; when
  that moves, this moves. Pin a version if you need a surface that holds still.

### Known limitations

The full list is in [MANUAL.md](MANUAL.md#limitations) and the
[README](../README.md#known-limitations). The ones that surprise people:

- `n > 1` is rejected; one text slot serializes its requests behind one
  dialog handle; VLM slots are single-turn and support neither LoRA, prefix
  caching, grammar, nor hot-swapping.
- A bundle whose `dialog.type` is `ssd-q1` needs a patched library **on
  2.49.x** — a 2.49 regression, not a property of such bundles.
- Holding two models on one HTP device at once is not dependable, and whether
  a second model fits at all depends on what loaded first rather than on the
  total size.
