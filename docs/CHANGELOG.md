# Changelog

## 1.1.0 — video as frames, and a prompt count that includes them

A VLM request could already carry several images, but each one spent a whole
encoder step: the preprocessor fills Qwen3-VL's temporal dimension by
duplicating the same frame, so footage sent that way paid twice the context
for the same seconds and no step ever held two different frames for the
encoder to see motion in. This release adds the shape that does, and makes the
cost of it visible in `usage`.

Docs and one comment aside, everything here is on the VLM path. The text-only
`Slot`/`GenieDialog` endpoints are untouched.

### Added

- **`video_url` content parts.** Frames the client already extracted, in the
  form vLLM uses for client-side preprocessing: base64 JPEGs joined by commas
  under a `video/jpeg` media type. Consecutive frames are packed
  `temporal_patch_size` at a time into one encoder step, so the same footage
  enters in half the vision tokens and the encoder sees real frame pairs.
  `media_io_kwargs.video` (top level of the body, `extra_body` from an OpenAI
  client) carries the `fps`/`frames_indices` behind the `<t seconds>` markers.
  A container media type (`video/mp4` and friends) is refused rather than
  half-supported — no demuxer ships here — and remote URLs are not fetched,
  as for images. See [API.md](API.md#post-v1chatcompletions).
- **`VLM_VISION_BUDGET_GUARD`** (default `false`): refuse, as a `400`, a
  request whose vision tokens cannot fit the text generator's context instead
  of letting it reach the SDK. **Off by default for the same reason as
  `TOOL_CALL_RECOVERY`** — it conceals a defect this server exists to expose —
  but what it conceals is severe: past the context the slot wedges for every
  later request until the process restarts, and far past it the process dies.
  Read a run taken with the guard on as the application's behaviour, not the
  SDK's.

### Changed

- **`usage.prompt_tokens` counts visual input on the VLM path.** It previously
  reported tokenized text only, which left out the part that actually fills
  the context: a 10-frame and a 28-frame request both came back as 20 while
  occupying 1280 and 3584 of 4096. Nothing hands this number back — images
  never become text on the host, and neither `GenieNode.h` nor
  `GeniePipeline.h` has a call for it — so it is derived from the step count
  (256 per step for `qwen3_vl`), which is confirmed by where the context
  actually runs out.
- **Frames are decoded only after the plan is accepted.** The step count comes
  from the parts alone, so a 500-frame request no longer pays 500 JPEG decodes
  and their bitmaps before being refused. The ordering matters precisely
  because the guard exists for that board's memory.

### Breaking, if you wrote your own `VLMSpec`

The two preprocessing hooks changed shape, because packing frames crosses the
boundary they used to sit on. `build_prompt_segments` now takes the spec and
the video metadata and returns `("step", payload)` rather than
`("image", index)`; `preprocess_image` is now `preprocess_step`, taking the
whole images list and one step's payload. `vlm.py` therefore never needs to
know how many frames a step holds. A step payload that disagrees with the
ViT's temporal size raises `ValueError`, so a future spec cannot drop frames
quietly.

`qwen3_vl` is still the only registered spec, and no endpoint or config key
changed with it — which is why this is a minor release. See **Backward
compatibility is not a goal** under 1.0.0.

### Documentation

- **QAIRT 2.50.0.260828** was put through the same reproducers as every build
  before it, with a stock 2.49.40.260810 run back to back as a control. D1
  through D5 are exactly where they were; the matrix gains a column and the
  version-scoped claims now say that 2.50.x behaves the same.
  [QAIRT_VERSIONS.md](QAIRT_VERSIONS.md). One thing this server works around
  *is* fixed there — creating the image-encoder node before the text generator
  no longer starves the text generator's context — and the workaround stays
  in, because it costs nothing, every 2.49.x still needs it, and which layer
  fixed it was never established.
- The `ssd-q1` limitation below is a 2.49/2.50 regression, not a property of
  such bundles: 2.48.40.260702 has neither that nor the LoRA consequence.
- Corrections that postdate 1.0.0's tag: the gemma4 tool-call dialect and its
  buffered streaming, `loaded` in `/v1/server/status`, the two GETs that
  accept only one of `?slot=`/`?model=`, and a retraction — a stock library
  *refuses* an unsupported grammar rather than silently ignoring it.

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
