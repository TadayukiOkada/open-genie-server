# Changelog

## 1.5.0 — fixes from a repository-wide code review: safer defaults, strict input, and a readiness probe

Almost everything in this release comes out of one code review
(`docs/reviews/2026-09-25-repository-review.md`), fixed in pull requests
#22-#44. The one exception is #29, the LoRA alpha check, which came from a
board test. The review document records what each PR resolved (#45), and
the docs and examples were brought up to date for this release in #47. The
themes are:
- the server no longer blocks its event loop or runs a request against a
  model it was not planned for;
- a browser on the same network can no longer drive it;
- one request can no longer make it allocate without bound;
- malformed input is a `400` instead of a `500` or a silent
  misinterpretation;
- shutdown releases everything under each slot's lock.

**Five changes are breaking**, listed first below; read them before
upgrading. A config that quoted a boolean, a client that POSTs without
`Content-Type: application/json`, and a browser page calling the server
directly (including Open WebUI's Direct Connections) are the likeliest to
notice.

Checked on an SA8255P with QAIRT 2.49.40 before release. The integration
suite was all green on each configuration:
- a text slot;
- each of four VLM bundle layouts (AI Hub Qwen3-VL-4B, the DeepStack
  export, Qwen3-VL-2B, Gemma 4 E2B LMM);
- a text and a VLM slot resident together.

The same run also checked the new status codes, a UTF-8 round trip, and a
shutdown during a streaming generation.

### Breaking

- **Tool history a slot's chat template cannot write is a `400`, not
  dropped.** Send tool round trips to a chatml, llama3 or gemma4 slot.
  - llama2 dropped `tool` messages, and llama2 and Gemma 2/3 both dropped
    an assistant's `tool_calls`, so the model saw the round trip with
    pieces missing. Both templates now refuse tool history, since neither
    has a form for it.
  - llama2 also refuses any role it has no form for, such as `developer`,
    instead of dropping it.
  - gemma4 refuses a tool result it cannot name (no `name` and no matching
    `tool_call_id`), and a call whose `arguments` are not a JSON object.
- **A config flag must be a JSON boolean, and `CHAT_TEMPLATE` and
  `TOOL_FORMAT` must name something that exists.** The four flags
  (`PROMPT_LOGPROBS`, `GENIE_PROFILE`, `TOOL_CALL_RECOVERY`,
  `VLM_VISION_BUDGET_GUARD`) went through `bool()`, so `"false"` turned
  them on. For the two workaround switches, that concealed exactly what they
  exist to show only on request. A `CHAT_TEMPLATE` typo fell through to
  chatml and a `TOOL_FORMAT` typo to hermes, with no error anywhere. All of
  these now stop the server at startup with a message naming the key and the
  allowed values; names are matched ignoring case and surrounding spaces. A
  config that relied on a quoted boolean needs it unquoted. The switch
  endpoint's `unload_first` is held to the same rule (`400`).
- **One request can no longer make the server allocate without bound.**
  Nothing capped the body, the size of an image, or the number of frames,
  and every frame is decoded at full size and held at once: a flat-colour
  PNG of about 500 KB declaring 13000 x 13000 (just under Pillow's own
  ceiling) is 483 MB decoded, and a 64 MB body holds over a hundred. Three
  ceilings, each `0` to turn off: `MAX_REQUEST_BODY_MB` (default 64; over it
  is `413`), and `VLM_MAX_IMAGE_PIXELS` / `VLM_MAX_TOTAL_PIXELS` (4096 x 4096
  per image, 64 times that per request; over either is
  `400`, checked from the image headers before anything is decoded). A
  request past them used to work, if the board had the memory. Base64 is
  also decoded strictly now: a character outside the alphabet is a `400`
  instead of being skipped, and only whitespace is ignored.
- **A web page on another origin can no longer drive the server through a
  browser.** Every origin was allowed by CORS, and a `POST` body was parsed
  as JSON whatever its `Content-Type`, so any page a user on the same network
  opened could switch or unload a model, apply a LoRA, flip
  `prompt_logprobs`, send a VLM request big enough to wedge the slot or kill
  the process, and read the replies. Two changes close it. A `POST` whose
  body is not declared `application/json` (or `application/*+json`) is now
  `415`, including one with no `Content-Type` at all; `curl -d` without
  `-H 'Content-Type: application/json'` is refused where it used to work.
  And CORS is off unless the new `CORS_ALLOW_ORIGINS` lists the origins
  allowed, as in `["http://localhost:3000"]`; `["*"]` restores the old
  behaviour. `lm_eval`, the OpenAI SDK and Open WebUI need neither change,
  except for Open WebUI's Direct Connections, where the browser calls this
  server itself and Open WebUI's origin has to be listed.
  What is still open (DNS rebinding) is in `SECURITY.md`.
- **An inference timeout is an error, not a normal stop.** When the watchdog
  (`INFERENCE_TIMEOUT`) cut a query short, the SDK reported it as it reports
  any abort, and the truncated text came back as a success: HTTP 200 with
  `finish_reason: "stop"` and no error event on a stream. `lm_eval` scored
  such a sample as finished. A timed-out request is now HTTP `504` on the
  non-streaming path, and an `error` event with no `"stop"` chunk on a
  stream. This includes a timeout that fires before the query starts, for
  example during a slow prefix restore. A timer that goes off just after the
  query already finished normally is not a timeout: that output is complete,
  and it is returned as before.

### Added

- **A way to clear out prefix-cache entries no slot can reach.** The key
  hashes the slot, model and LoRA state, so a switch or a LoRA change leaves
  the old KV snapshots on disk, unreachable, some of them gigabytes, with
  nothing to find or remove them. A warmup now records the namespace beside
  the entry (`prefix_<key>.json`). `GET /v1/prefix/cache` reports each
  entry's `namespace` and whether it is `reachable`.
  `DELETE /v1/prefix/cache?scope=unreachable` deletes the orphans, and
  `scope=all` deletes everything. An entry saved before this change gets
  its namespace the next time a slot reaches it, and an entry a save or
  restore is working on is kept. Still nothing is deleted on its own: the
  cache fills only on an explicit warmup and empties only on an explicit
  call. `DELETE /v1/prefix/cache/{key}` now also refuses a key that is not 16
  hex digits with `400`.

- **`GET /ready` (and `/v1/ready`), a readiness probe.** `/health` answers
  `ok` whatever the slots hold, so a slot left empty by a failed
  `unload_first` switch was invisible to monitoring while every request to it
  failed. `/ready` is `200` when every slot holds a model and `503` with the
  empty ones listed otherwise. `?slot=<name>` checks one slot, for a monitor
  that routes per slot. It does not detect a wedged slot: the stock
  library's wedge is reported, not detected, by design.

### Fixed

- **A `model` name that matches nothing is logged.** It still routes to the
  primary slot, which is what lets `lm_eval`'s fixed placeholder work, but a
  typo used to reach the wrong model without a trace. Any name other than
  `genie-local` that matches no loaded model is now logged once at WARNING,
  together with the models that are loaded. An image request that falls
  back to the first VLM slot is logged the same way. Past 256 distinct names
  it says so once and stops remembering them.
- **Small fixes from the code review's Low list:**
  - `POST /v1/prefix/warmup` and `GET|POST /v1/server/performance_policy`
    take `slot`, as the other slot-addressed endpoints do. With `model`
    alone, the second of two slots holding the same model was out of reach.
  - Prompt scoring (`echo` + `logprobs`) reads `max_completion_tokens` before
    `max_tokens`, like every other path; it used to do the reverse.
  - A multi-prompt completion checks every prompt against the context
    before running any, instead of discovering the one that does not fit
    after the prompts ahead of it had run.
  - `--port 0` binds a free port instead of falling back to the configured
    one, and `--host ""` is passed through.
  - An exception in a VLM text callback is logged instead of printed.
- **A Qwen3-VL bundle whose `temporal_patch_size` is not 2 is refused at
  startup.** The patchify stacks exactly two frames, but `metadata.json`
  could set any value. The slot then loaded and failed a reshape with a
  `500` on every request. No export uses anything but 2. A side that is not
  a whole number of patches (`image_width: 392` at patch 16) failed the same
  way and is refused too, and so is a metadata number that is not an
  integer, which `int()` used to truncate (2.7 became 2).
- **Tool round trips render consistently across chat templates.**
  - llama3 wrote tool calls in Hermes whatever the slot's tool format was.
  - gemma4 took a tool result's function name only from `name`, which OpenAI
    tool messages usually lack, so `response:{...}` went out nameless. It
    now resolves the name from the matching `tool_call_id`.
  - gemma4 replaced unparseable `arguments` with `{}`, which told the model
    it had called with no arguments. A list there was a `500`. Both are now
    a `400` (see Breaking). An empty `arguments` string still renders as a
    call with no arguments.
- **A character split across two token callbacks is joined, not dropped.**
  The dialog path decoded each callback with `errors="ignore"`, so a
  multibyte character (Japanese, an emoji) that arrived in two pieces
  vanished from the output without a trace. The VLM path used `replace`,
  which turned the split into U+FFFD. The SDK's detokenizer holds back an
  incomplete sequence in the paths we read, so this should be rare, but both
  paths now keep an incremental decoder per stream: pieces are joined, and
  bytes that never complete, or are invalid, show as U+FFFD instead of
  disappearing, including when a query returns without a terminal code.
  Tokens are counted by callback, not by visible text, so a token whose
  bytes were held back still counts toward `max_tokens`, and a generation
  that hit the cap reports `finish_reason: "length"`.
- **Shutdown no longer frees a dialog under a running call, and frees what
  it used to leak.** `free_all` called `GenieDialog_free` without the slot's
  lock, so a thread still inside the SDK hit a use-after-free. That could be
  a disconnected client's worker, a warmup answered with `202`, or the worker
  behind a `504`. Each slot's lock is now taken first, all within
  `INFERENCE_TIMEOUT` + 5 s, which is long enough for the watchdog to abort a
  text query. A slot still busy then is left allocated and logged, and so is
  the logger bound to it. A `GenieProfile` is now freed after its dialog, and
  a VLM slot's pipeline and nodes are freed too: they never were, although
  the shutdown log said HTP memory was being released.
- **Two slots with the same name are a startup error.** Nothing checked
  it, across `TEXT_SLOTS` and `VLM_SLOTS` or within one list, and a slot's
  name is a key in several places. Routing and `/v1/server/status` reached
  only one of the pair. The second slot's custom-sampler registration was
  skipped, so its logits went to the first slot's collector and could
  corrupt a logprobs request running there. The per-slot copy of the HTP
  extension config (the `device_id` pin) was overwritten. A name that is
  empty or holds a path separator is refused too, since it names that file,
  and so is one with a control character (the callback name reaches the SDK
  as a C string, so a NUL cut it short onto another slot's), a `|` (the prefix cache
  namespace separator) or surrounding whitespace.
- **Asking for logprobs no longer changes what gets sampled.** With
  `logprobs`, sampling moves into the server. There, a `temperature`,
  `top_k` or `top_p` the request left out fell back to 1.0, no top-k and no
  top-p, where the plain path uses the model's `genie_config.json` defaults.
  On a model configured for temp 0.1, adding `logprobs: true` sampled at 1.0.
  Both paths now fill in the same values. To sample as before, send
  `temperature: 1, top_p: 1, top_k: 0`. A `top_k` larger than the vocabulary
  no longer makes a logprobs request emit token 0 at every step; it means no
  top-k limit.
- **Generation parameters of the wrong type are a `400`.** They reached the
  worker thread unchecked. A string `temperature` or `seed` was a `500`, and
  a string `n` or a list `chat_template_kwargs` was a plain-text `500`.
  Others passed silently: `top_k: 1.5` was cut to 1, turning a sampled
  request greedy, `top_p: 7` went to the SDK, and `stream`, `echo`,
  `enable_thinking` and chat `logprobs` went through `bool()`, so the string
  `"false"` meant true. Now:
  - `temperature` must be ≥ 0, `top_p` from 0 to 1, and `top_k` an integer
    ≥ 0. vLLM's `-1` ("no limit") is accepted and sent as `0`, which means
    the same here.
  - `seed` must be an integer from 0 to 2³¹−1.
  - `n` and `best_of` must be integers ≥ 1.
  - `max_tokens`, `max_completion_tokens` and `top_logprobs` must be
    integers. `true` used to be read as 1.
  - `chat_template_kwargs` and `stream_options` must be objects, and the
    flags must be booleans. This includes `stream_options.include_usage`:
    the string `"false"` used to send the usage chunk anyway.
  - An integral float such as `2.0` counts as an integer.
- **An unexpected exception is an OpenAI error, not plain text.** A bug
  reached by a request (a string `n`, a list for `chat_template_kwargs`)
  answered `500` with the body `Internal Server Error` as `text/plain`, which
  an OpenAI client cannot parse, though the API documents the error envelope
  for every failure. It is now the envelope with `type: "server_error"`, and
  a message naming only the exception's type and an error id such as
  `err-1a2b3c4d`. The traceback is logged under that id. The reply passes
  through CORS, so a page on an origin in `CORS_ALLOW_ORIGINS` can read it.
- **`/v1/lora/strength` refuses an alpha the model does not have.** The SDK
  answers success for any name: on an engine with an inference scheduler, a
  name it does not know is kept as a CB multi-LoRA adapter-order name, and no
  tensor changes. A typo therefore returned `200`, and the server even moved
  the prefix-cache namespace to a strength that was never applied. The name is
  now checked against the alphas `genie_config.json` declares for the engine,
  and anything else is a `400` listing them. The check runs under the slot
  lock, against the model the call reaches. `alpha` must be a finite number:
  `"0.5"`, `true` and `NaN` used to be passed to the SDK as floats, and a list
  was a `500`. On all three LoRA endpoints, `engine` and `lora_adapter_name`
  must be strings; a list there was a `500` too. A strength is recorded under
  the engine the SDK acts on. `"target"` is `"primary"` to the SDK, and one
  alpha set under both spellings was recorded twice, so two different
  strengths could produce the same prefix-cache namespace.
- **A request no longer runs against a model or LoRA adapter it was not
  planned for.** The prompt, `max_tokens` and prefix-cache key are worked out
  before a request takes the slot lock. A model switch or LoRA change that got
  the lock first left the waiting request to run with the old model's
  template and context budget and to restore the old namespace's prefix KV
  into the new state (a swapped LoRA keeps the KV shape, so nothing failed);
  after a failed switch it even reached the SDK with no dialog. The slot now
  carries an epoch that every handle, model or adapter change bumps. A request
  that finds it moved once it holds the lock is refused with `409` (an `error`
  event on a stream) and should be resent. `POST /v1/prefix/warmup` checks the
  same, so it cannot save a KV under a stale key.
- **A LoRA strength change no longer reuses the prefix KV of the old
  strength.** The prefix-cache namespace named the slot, model and adapter
  but not the alphas set through `/v1/lora/strength`. A prefix saved at one
  strength was restored at another, and generation continued from the wrong
  KV. The strengths are now part of the namespace, and a strength change
  bumps the slot's epoch like an adapter change does. With no strength set,
  the namespace, and so every existing cache key, is unchanged. Strengths are
  forgotten on a release and on a model switch, not when another adapter is
  applied: on the board, applying a second adapter kept the alphas set for the
  first, since `alpha0`/`alpha1` belong to the dialog.
- **A request no longer inherits the previous request's sampler settings.**
  The SDK merges partial sampler updates, and a parameter the request and the
  model config both left out was simply not sent, so it kept the last value:
  after a `temperature: 0` request (top-k 1) or one with a `seed`, the next
  request that only set `temperature` stayed greedy or kept the seed. The
  VLM path passed no model defaults at all, so every parameter carried over.
  Every request now sends the full set. `temp`, `top-k` and `top-p` are the
  request's value, else the model config's, else the SDK's own default
  (0.1, 0, 0.8). `seed` is the request's, else a fresh random one for every
  request. The VLM path reads the text-generator config's `sampler` section.
  A `seed` in a model config still seeds the dialog only once, when it is
  created. Re-sending it on every request would give every unseeded request
  the same random stream, so repeated sampling would return one answer n
  times. For reproducible output, send `seed` in the request.
- **Waiting for a slot no longer freezes the server.** `/v1/models/switch`,
  the `/v1/lora/*` calls and `POST /v1/server/performance_policy` waited for
  the slot lock (up to 600 s for a switch) and ran the SDK call on the event
  loop. While one waited, `/health` and token delivery on every other slot
  stalled. They now run on worker threads. Each call re-checks the slot for
  a model once it holds the lock, and a call whose client disconnected while
  it waited is not run.
- **A request that answered `504` stops using its slot.** The generation is
  now aborted, so it no longer waits for the slot lock and then runs to the
  end with nobody listening. A VLM request abandoned before it gets its slot
  never starts. There is still no way to stop a VLM request once it is
  running.
- **Every system message now reaches the model.** With a system message in
  the request, the prefix-cache split kept only the first one and left the
  rest out of the prompt, so a `[system, user, assistant, system, user]`
  conversation (agent clients, some Open WebUI features) silently lost its
  later instructions, and a system message that was not first was moved to
  the front. Only a conversation that starts with exactly one system message
  is split now; any other shape is rendered whole, in the order sent, and is
  not prefix-cached. The llama2 and Gemma 2/3 templates, which have no
  system turn and fold system text into the next user turn, lost system
  messages another way. Of two consecutive system messages, the second
  overwrote the first, and a system message with no user turn after it was
  dropped. Consecutive system messages now share one block, and trailing
  system text gets a turn of its own.
- **The tools block and `/no_think` go to the system message that opens
  the conversation.** They were appended to the first system message
  wherever it was, so with `tools` and a system message in the middle of
  the conversation, the tools declarations landed there. Now, if the
  conversation does not open with a system message, a new one is added at
  the front, and a later system message is left as the caller wrote it.

## 1.4.0 — Ubuntu QAIRT-package targets, and an explicit error when logprobs can't be scored

Adds a third target platform, `linux-ubuntu`, alongside `linux-oe` and
`android`: a QCS9075 EVK running Ubuntu with the distro's `qairt-*` packages
instead of a QAIRT SDK tree. Detected automatically from `/etc/os-release`
and the DSP library layout, or set explicitly with `target_platform`. On
this platform `libGenie.so` loads from the system loader (`qairt-libs`
registers it) unless an SDK root is configured, and the DSP search path
covers both QCS9075 cores since an unpinned slot can land on either one.

The other change is unrelated to Ubuntu specifically: QAIRT 2.46 on the
QCS9075 packages accepts the custom-sampler logprobs setting but never
invokes its callback for some bundles, which previously left generated-token
logprobs and prompt scoring silently empty or wrong. The server now detects
a missing callback at the first generated token, stops that query, and
returns HTTP 400 `error.code: "logprobs_not_supported"` instead — for
chat/completion logprobs and prompt scoring alike. A slot remembers the
result, so later logprobs requests on it fail fast without a wasted
generation. Prompt scoring also now checks it got exactly one score per
prompt token, and returns HTTP 500 if the callback stopped partway through
rather than silently shifting every score by one.

### Added

- **`linux-ubuntu` target platform** (`config.py`): autodetected from
  `/etc/os-release` (`ID`/`ID_LIKE` naming Ubuntu) plus `/lib/dsp/cdsp` and
  `/usr/lib/rfsa/adsp`, or set explicitly via `target_platform`.
  `resolved_genie_lib_path` falls back to the bare `libGenie.so` name (the
  system loader's copy) when no `sdk_root` is configured; `apply_process_env`
  no longer exports `QAIRT_SDK_ROOT`/`QNN_SDK_ROOT` when `sdk_root` is unset,
  so a stray SDK exported by the launching shell cannot leak in. See [Ubuntu
  with QAIRT packages in MANUAL.md](MANUAL.md#ubuntu-with-qairt-packages).
- **Explicit `logprobs_not_supported` error** (`engine.py`, `app.py`): a
  generation that completes without the SDK ever invoking the logits
  callback now returns HTTP 400 instead of empty or misleading logprobs, for
  both generated-token logprobs and prompt scoring. See [Logprobs in
  MANUAL.md](MANUAL.md#logprobs).

Verified on a QCS9075 Ubuntu EVK (QAIRT 2.46 distro packages): 272 offline
API/unit tests, core text integration tests, model hot-swap, and VLM
integration (image, video, streaming, disconnect recovery, vision budget
guard) all passed; grammar was rejected by that runtime, as expected for
2.46. Re-verified after merge on both the EVK and the SA8255P board
(`server-verification/results/20260922_iq9075_ubuntu/README.md`).

## 1.3.0 — VLM bundle layouts are read from the bundle, not hard-coded per model

Every VLM bundle needed its own hard-coded node-config filenames, connections
and static tensors in `vlm_specs.py`, and only the AI Hub Qwen3-VL-4B and
Gemma 4 E2B bundles actually ran as shipped. `vlm_layout.py` now reads all of
that straight from the bundle itself — its own genie-app script, or
`metadata.json` when it has one — so `VLM_SLOTS[].spec` is optional too,
auto-detected from the bundle's tokenizer and node configs. Verified on the
board against four bundles that all differ in shape: the AI Hub export, the
Qwen3-VL-4B DeepStack tutorial bundle, the Qwen3-VL-2B tutorial bundle, and
Gemma 4 E2B LMM.

Everything here is an addition. An existing `VLM_SLOTS` config with an
explicit `spec` keeps working exactly as before. The other two changes since
1.2.0 are documentation corrections with no code behind them.

### Added

- **VLM bundle layouts are read from the bundle, not hard-coded per model.**
  `genie_server/vlm_layout.py` reads which node configs to load, how they
  connect, and which static tensors to feed straight from the bundle's own
  genie-app script (or `metadata.json`'s `genie.pipeline`, when it ships one),
  falling back to the fixed filenames this server used before this module
  existed. Verified against four layouts that all differ in shape: the AI Hub
  Qwen3-VL-4B export, the Qwen3-VL-4B DeepStack tutorial bundle, the Qwen3-VL-2B
  tutorial bundle, and Gemma 4 E2B — none of them need anything in
  `VLM_SLOTS[]` beyond `model_root`. See [Bundle layout auto-read in
  MANUAL.md](MANUAL.md#bundle-layout-auto-read) and [the per-bundle table in
  PLATFORM_NOTES.md](PLATFORM_NOTES.md#vlm-bundle-layouts).
- **`VLM_SLOTS[].spec` is now optional.** Left out, the VLM family is
  auto-detected from the bundle's own `tokenizer.json` and node configs
  (`vlm_specs.detect_family`); an explicit value still wins and still works
  exactly as before. Zero or several families matching is a startup error
  that says what to pass explicitly.
- **`VLM_SLOTS[].pipeline_script` / `.node_configs` / `.static_tensors`**: an
  escape hatch for a bundle layout `vlm_layout.py` cannot yet read on its own
  — see MANUAL.md.
- A bundle that ships only a `dialog` config and no node configs (the GenieX
  pipeline format) is now refused at startup with that reason, instead of
  failing later with a confusing "file not found" once layout auto-read
  cannot find the fixed legacy filenames either. Out of scope for now — see
  [PLATFORM_NOTES.md](PLATFORM_NOTES.md#geniex-vlm-bundles-are-out-of-scope).

### For `VLMFamily`/`VLMSpec` authors

`genie_server/vlm_specs.py` is now the package `genie_server/vlm_specs/`, and
`VLMSpec` is split in two: `VLMFamily` (one module per model — preprocessing,
node-topology *defaults*, chat template, `bind`, `detect`) and the resolved
`VLMSpec` a family + a bundle's `vlm_layout.BundleLayout` combine into
(`vlm_specs.resolve`). `bind` gains a third argument, the `BundleLayout`
(including its parsed `metadata.json`, if any), so it can read whatever the
bundle states beyond its node configs. `VLMSpec`'s six `*_io` fields
(`text_encoder_text_input_io` and friends) are gone — they were GenieNode API
constants, not per-model values, and now live as module constants in
`vlm_layout.py`. Adding a new model is one file under `vlm_specs/` and one
line in `vlm_specs.FAMILIES`, unchanged from before; a new **bundle layout**
of an already-supported model now needs no code change at all.

`qwen3_vl_deepstack` (previously its own hard-coded `VLMSpec` for the
DeepStack tutorial bundle, added on a local branch that this absorbs) is now
a compatibility alias for `qwen3_vl` — the layout comes from the bundle
either way, so the two names select the same family.

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
- **Installing on the device needs a virtualenv, and the docs now say so.** On
  the reference board's Linux guest the root filesystem is read-only and the
  system `python3` has no pip, so the server and its dependencies go into a
  venv under `/home/root` — including when `genie-server.py` runs without the
  package installed. The README and [MANUAL.md](MANUAL.md) say this in a note;
  [PLATFORM_NOTES.md](PLATFORM_NOTES.md#installing-on-the-device) lists what is
  writable there and gives the commands, including an install from downloaded
  wheels for a guest that cannot reach PyPI.

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
