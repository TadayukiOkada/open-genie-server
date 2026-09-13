# Changelog

## Unreleased

### Documentation

- **What the `err 1002` budgets count, corrected.** The reference bench's
  page-table pool is not charged 1,024 KB per QNN context: it follows the
  address space a slot maps, and 1,024 KB per context held for one bundle only.
  A second limit is documented — each DSP protection domain's address space,
  which fragments — and which domain a context lands on is decided by a byte
  budget both cores share, not by counting contexts. `Allocated total size`
  covers libGenie's I/O buffers rather than one mapping. The corrections are
  marked in place in [MANUAL.md](MANUAL.md) and
  [PLATFORM_NOTES.md](PLATFORM_NOTES.md) and their Japanese versions.

## 1.2.0 — Gemma 4 QAT bundles with image and video input, and the SDK's own log

Gemma 4 E2B bundles exported from Google's QAT checkpoint now load and serve
on both kinds of slot. Getting there took three things: resolving the paths
of their per-channel-quantized embedding tables, a VLM spec for Gemma 4's
image encoder, and no longer writing a BOS the SDK already adds. The server
can also bind the SDK's own log now (`GENIE_LOG_LEVEL`), which is what traced
`err 1002` to the budget it actually exhausts.

Everything here is an addition or a fix. `gemma4` is a new spec and
`GENIE_LOG_LEVEL` a new key that defaults to off. Two numbers change under
existing names, both because they reported less than the SDK prefills:
`usage.prompt_tokens` now includes the BOS the SDK adds on a bundle whose
dialog names `bos-token`, and a VLM slot's count includes its chat-template
markers.

### Added

- **The `"gemma4"` VLM spec.** A Gemma 4 LMM bundle (image encoder, LUT text
  encoder, text generator) as a `VLM_SLOTS` entry. The encoder's position ids
  and pooling index are made on the device from `vision-param`, which the
  node reads once at creation, so **the patch grid is fixed per slot rather
  than chosen per image** as Gemma 4's own processor does: every image is
  resized to that grid. A `video_url` is one encoder step per frame, each
  after Gemma 4's `mm:ss` timestamp; the processor's smaller video budget (70
  soft tokens a frame) means a video slot wants a smaller grid. A `bos-token` in the
  bundle's text-encoder config is left as declared: libGenie then prepends it
  to every text segment, which is logged at startup and counted in `usage`,
  and the template writes no BOS of its own. See
  [MANUAL.md](MANUAL.md#configuration-1).
- **`GENIE_LOG_LEVEL`** (default `""`, off; `error`, `warn`, `info` or
  `verbose`). libGenie logs nothing unless a logger is bound to the config a
  handle was created from, so a failure inside the SDK used to leave nothing
  but the status code. One `GenieLog` is now bound to every dialog, node and
  pipeline config the server creates, and re-applied after a hot swap. The SDK
  writes the lines itself — stdout on Linux, logcat on Android — so they do
  not pass through Python logging. `info` is loud: a four-slot startup wrote
  1,836 lines. The first run with it on also settled a question for
  Limitations: the SDK's continuous batching cannot be switched on from a
  bundle config.

### Fixed

- **Per-channel-quantized embedding tables load from outside their
  directory.** A PCQ table — what the Gemma 4 QAT exports ship — names
  `quant-param.scale` and `quant-param.offset` as `.bin` paths, and the SDK
  opens them relative to the working directory. Only `lut-path` was
  resolved, so such a bundle failed to load unless the server ran from
  inside it. Now resolved for text slots (`dialog.embedding`,
  `dialog.perlayer-embedding`) and VLM node configs alike, where the
  per-layer tables (`perlayer-lut`, `perlayer-embedding`) had not been
  resolved at all.
- **A chat template no longer writes a BOS the SDK already adds.** When a
  bundle's dialog context names `bos-token`, libGenie prepends that token to
  every query, and the `gemma4`, `gemma`, `llama3` and `llama2` templates
  wrote their own on top: a gemma4 prompt reached the model as
  `[2, 2, 105, …]` (the SDK's verbose log). The template's BOS is now left out
  for such a slot; the bundle itself is not changed. `usage.prompt_tokens` —
  and the context check and default `max_tokens` built on the same count —
  now includes the BOS the SDK adds, which it was one short of. On a
  LUT-embedding bundle the SDK also puts a BOS in front of a prefix-cache
  hit's remainder; that is documented under Prefix KV Cache and left as it is.
- **A VLM slot's `usage.prompt_tokens` counts the prompt as rendered.** Its
  text half used to be the request's own words, chat-template markers left
  out — while a comment claimed that matched the text path, which has always
  counted its rendered prompt. It now counts each text segment the spec
  renders, markers included, plus the vision tokens and the text-encoder's
  BOS, so both kinds of slot report what the SDK prefills. The budget guard
  counts the same way.
- **The device integration test V06 counts frames for any spec.** It asserted
  Qwen3-VL's 512 prompt tokens for four extra frames, so it failed against a
  gemma4 slot that was doing what its spec says (1,076: one 260-token step per
  frame, plus markers). It now checks that 2, 4 and 6 frames add the same
  prompt tokens each time, at least 64 per two frames.

### Documentation

- **What more slots are worth, and what each one costs.** A sweep of
  `qwen3_0_6b` slots shows that a core does not interleave two dialogs — two
  slots on one core double the latency and move the wall clock 1.06× — so
  slots beyond the number of cores buy nothing. Three over-stated claims in the
  co-residency material are corrected: two slots may share a `device_id`, the
  allocation that fails can be seen by summing DMA-BUF mappings, and "the
  second model must go on the other NSP" holds only for some bundles. The
  sweep's rows were then re-measured from a power cycle each and matched, so
  its caveat is gone.
- **What `err 1002` is.** With the SDK's log on, it is the host failing to map
  a context's shared weights into a DSP protection domain, against a budget
  kept outside the guest. On the reference bench that budget is a hypervisor
  page-table pool charged 1,024 KB per QNN context. The advice that a failed
  startup is clean and safe to retry is corrected in place: the budget is not
  returned, and repeated failures end in `Failed to create device: 14001` and
  a power cycle. The platform-specific part moved to
  [PLATFORM_NOTES.md](PLATFORM_NOTES.md).
- **Japanese anchors.** The docs test's slugger kept CJK punctuation that
  GitHub strips, which had been hiding three dead fragment links and two links
  that rendered as plain text. All five are fixed, and the slugger now agrees
  with GitHub on every heading in the repository.
- **Gemma 4 on a text slot and on a VLM slot**, side by side: what each reads
  from the bundle, how the prompt is built, which request fields apply, and
  how tokens are counted. See
  [MANUAL.md](MANUAL.md#gemma-4-text-slot-vs-vlm-slot).
- **gemma4 streaming is buffered only for a request with `tools`.** The API
  reference's "Streaming (gemma4)" note read as if every gemma4 stream arrived
  in one piece at the end; the buffering filter is attached only when the
  request declares `tools`. Without them a text slot streams token by token,
  and a VLM slot, which ignores `tools`, always does. The comparison table
  gains a `stream: true` row.
- **lm_eval against a Gemma 4 bundle.** `examples/lm_eval` now says when the
  CPU reference needs `add_bos_token=True` — the SDK adds a BOS on the board
  while the model's HF tokenizer does not, and without the flag the reference
  scores 9 to 52 nats low per continuation — and that a bundle built from a QAT
  checkpoint should also be compared with that checkpoint, which on Gemma 4 E2B
  sits 20 nats from the original fp32 model. Gains the Gemma 4 E2B timing.

### For `VLMSpec` authors

`VLMSpec` gains two optional fields: `max_patches` (the row count
`pixel_values` is zero-padded to) and `bind(spec, node_cfgs)`, called with
every node config loaded but before any node is created, so a spec can read
what the bundle was exported with and adjust the configs. The slot now loads
all node configs before creating the first node; the creation order itself is
unchanged. `text_encoder_adds_bos` is set by the slot from the loaded
text-encoder config, for a template that has a BOS of its own to leave out.

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
