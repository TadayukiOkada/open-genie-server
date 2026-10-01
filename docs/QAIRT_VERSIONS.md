# QAIRT Version Issues

open-genie-server is a thin client of `libGenie.so`. Almost every failure mode
that is not a bug in this server is a property of the **QAIRT SDK build** you run
it against. This page is the single place that records which build carries what.

Other pages link here rather than repeating it, so that they can stay about
using the server. If you are choosing an SDK build or deciding how to export a
model, read this page first.

- [At a glance](#at-a-glance)
- [How to read the evidence column](#how-to-read-the-evidence-column)
- [D1 — a stock library wedges a slot](#d1--a-stock-library-wedges-a-slot)
- [D2 — reset does not rewind the KV position allocator](#d2--reset-does-not-rewind-the-kv-position-allocator)
- [D3 — a library you rebuild has no grammar](#d3--a-library-you-rebuild-has-no-grammar)
- [D4 — grammar leaks the terminal token into the text](#d4--grammar-leaks-the-terminal-token-into-the-text)
- [D5 — reset corrupts a speculative-decoding dialog](#d5--reset-corrupts-a-speculative-decoding-dialog)
- [D6 — save and restore break a sliding-window cache group](#d6--save-and-restore-break-a-sliding-window-cache-group)
- [Checking your own SDK](#checking-your-own-sdk)
- [Choosing a library](#choosing-a-library)
- [What 2.49.1 changed](#what-2491-changed)
- [What 2.50.0 changed](#what-2500-changed)
- [What 2.51.0 changed](#what-2510-changed)
- [What we have and have not tested](#what-we-have-and-have-not-tested)

## At a glance

| | 2.48.40.260702 | 2.49.40.260810 | 2.49.1.260821 | 2.50.0.260828 | 2.51.0.260929 |
|---|---|---|---|---|---|
| **D1** slot wedge — KV occupancy is never rewound | not reproduced | **present** (measured) | **present** (source-identical) | **present** (measured) | fixed (measured) |
| **D2** reset does not rewind the KV position allocator | not reproduced | **present** (measured) | **present** (source-identical) | **present** (measured) | fixed (measured) |
| **D3** a rebuilt library has no grammar backend | **present** | **present** | **present** (measured) | **present** (source-identical) | **present** (measured) |
| **D4** grammar leaks the terminal token as text | not tested | **present** (measured) | not tested | **present** (measured) | **present** (measured) |
| **D5** reset corrupts a speculative-decoding dialog | not present (source argument) | **present** (measured) | **present** (source-identical) | **present** (measured) | fixed (source argument) |
| **D6** save/restore breaks a sliding-window cache group | not tested | **present** (measured) | not tested | **present** (source-identical) | **present** (measured) |

> [!WARNING]
> **`2.49.1.260821` is newer than `2.49.40.260810`, despite the smaller number.**
> Compare the `build_id` in `sdk.yaml`: `260821…` (21 Aug) against `260810…`
> (10 Aug). Sorting these version strings will mislead you.

**2.51.0.260929 fixes the three reset defects (D1, D2, D5)**, which every 2.49.x
and 2.50.x build we tested carries. On 2.51.0 a stock library no longer wedges a
slot, so it is the first build since 2.48 on which a stock library is a
reasonable default. D3, D4 and D6 are still there. D1, D2 and D5 are why this
project rebuilt `libGenie.so` for 2.49.x and 2.50.x; D3 is the cost of doing
so. Each section below says how to check a build we have not seen. See
[What 2.51.0 changed](#what-2510-changed).

> [!NOTE]
> **2.50.0.260828 is a large release that leaves every one of them in place.** Its Genie
> sources differ from 2.49.40.260810 in 78 files, and its public C API headers
> are byte-identical — so it is a drop-in swap that changes none of this. It
> does fix one thing this server used to work around; see
> [What 2.50.0 changed](#what-2500-changed).

## How to read the evidence column

- **measured** — reproduced on hardware, with a standalone reproducer or through
  this server.
- **source-identical** — the SDK ships the complete, buildable reference sources
  for the Genie library under `examples/Genie/Genie/`. Where the files that
  implement a defect are byte-for-byte identical between two builds, the defect
  is present in both. For D1 and D2 those files are `qualla/dialog.cpp` and
  `qualla/engines/qnn-htp/KVCache/kvmanager.{cpp,hpp}`, and for D5 it is
  `qualla/dialogs/ssd-q1.cpp`; all of them are identical between
  2.49.40.260810 and 2.49.1.260821. `ssd-q1.cpp` **differs** from
  2.48.40.260702's, and that difference is exactly what D5 is: the two
  builds' `SelfSpecDecDialog::reset()` are byte-identical, and 2.49's
  `completeInit()` is what grew the slot calls that `reset()` does not
  mirror. This is a source argument, not a hardware run: we have not re-run
  the D1/D2 reproducers on 2.49.1. **2.50.0.260828 is not on this footing**:
  D1, D2, D4 and D5 were re-run on hardware there. Only its D3 rests on
  sources — `qualla/grammar.hpp` is identical and that build still ships no
  file defining the backend it declares. **2.51.0.260929**: D1, D2, D3, D4 and
  D6 were measured on hardware, D1 and D2 on both a multi-context and a
  single-context bundle. D5 rests on sources: no `ssd-q1` bundle was on the
  board.
- **not reproduced** — the reproducer was run and did not trigger. This is
  weaker than "absent": it means that sequence is safe on that build, not that
  no sequence is.
- **fixed** — present in the build before, and the cause is gone: the code
  that produced it was changed. "(measured)" adds that the reproducer no
  longer triggers on hardware; "(source argument)" means only the sources were
  read.
- **not present (source argument)** — the code that produces the defect does
  not exist in that build. Also reasoning from the shipped sources rather than
  a hardware run, and it says nothing about defects we have not looked for.
- **not tested** — no claim either way.

**Where D1, D2 and D5 come from.** 2.49 introduced a continuous-batching
scheduler (`qualla/engines/inference-scheduler.cpp`, a file that does not exist
in 2.48) and rewired the dialog reset path around it. All three defects are in
that rewiring: the reset path stopped going through the call that used to
rewind the cache wholesale, and what replaced it does not rewind everything.
That is why 2.48 does not reproduce D1 and D2, and why D5 cannot arise there —
in 2.48 the slot calls D5 is about do not exist at all, so the constructor and
the reset path do the same thing as each other.

## D1 — a stock library wedges a slot

> [!WARNING]
> On a stock 2.49.x or 2.50.x library, a request whose **prompt + generated** tokens cross
> the cache budget permanently breaks its slot. The crossing request itself
> returns normally; every request after it — including trivial ones — fails with
> `GenieDialog_query failed [...]: -1` and `batch dispatch failed` in the log,
> for as long as that `GenieDialog` exists. `GenieDialog_reset()`, which this
> server already calls before every request, does not recover it. Only reloading
> the model does.

**The budget is `context_length − AR_N`**, for the smallest compiled context
length. `AR_N` is not exposed by any API; it is in the bundle's graph names:

```sh
grep -aoE 'ar[0-9]+_cl[0-9]+' part1_of_N.bin | sort -u
#   ar128_cl4096
#   ar1_cl4096      -> AR_N = 128, CL = 4096, budget = 3968
```

**Multi-context-length bundles hit it early.** These bundles are compiled at
several context lengths at once — the Qwen3 w4a16 bundles list
`context_lengths: [512, 1024, 4096]` — and the SDK moves between those variants
as the token count grows. Switching *up* is fine. Coming back *down* is what
breaks, so the boundary is wherever the first switch happens:
`512 − 128 = 384 tokens`. Nothing in the configuration hints at that number, and
it is not a limit on what the model can do.

Measured on `qwen3_0_6b` (`prompt_tokens + completion_tokens`):

| total | result |
|---|---|
| 384 | ok |
| 385 | slot wedged |

It makes no difference whether the boundary is crossed while processing the
prompt or while generating: 380 prompt + 4 generated is fine, 168 prompt + 256
generated is not.

### A single context length raises the budget but does not remove the defect

> [!WARNING]
> **This was previously documented the other way round, and the error is worth
> explaining because it is easy to repeat.** The earlier measurement bisected on
> *prompt length* with a small `max_tokens`, so it only ever exercised prefill —
> and prefill overflow is the one path that behaves well. The decode path was
> never tried.

A single context length changes **where the budget sits**, not whether it can be
crossed. There is no smaller variant to start on, so the budget is
`context_length − AR_N` for the only variant — 3968 on a `[4096]` / AR-128
bundle instead of 384. Much harder to reach by accident, but reachable.

Measured on a stock library, on one `[4096]` bundle, separating the two paths:

| what crosses the budget | total tokens | result |
|---|---|---|
| nothing (control) | 3576 | fine |
| **generation** | 4048 | **wedged** — the request returned 200, `finish_reason: "length"`, no warning |
| **generation** | 4069 | **wedged** |
| prompt alone | 4083 | fine — `GENIE_STATUS_WARNING_CONTEXT_EXCEEDED`, nothing generated, dialog still usable |

So a prompt that overflows on its own is refused cleanly. A prompt that fits,
with a `max_tokens` that carries the total past the budget, wedges the slot —
and **the request that does it looks completely successful**.

The practical shape of the trap: 3698 prompt tokens with `max_tokens: 350`
against a `[4096]` bundle is an ordinary summarisation request, it passes the
obvious `prompt_tokens <= context.size` check, and it kills the slot.

> [!WARNING]
> **On a stock 2.49.x or 2.50.x library there is no configuration that is safe by
> construction.** Exporting at a single context length still leaves you needing
> to cap `prompt_tokens + max_tokens` at `context_length − AR_N` yourself.
> **open-genie-server does not do this for you**: its context check looks at the
> prompt only.

**Cause.** When the running token count outgrows the small context variant, the
SDK correctly reshapes its KV cache up to a larger one. What it never does is
give those cache entries back: the cache group's "valid KV entries" counter is
only ever increased on the continuous-batching scheduler path, and nothing in
the reset path decrements it. The next request needs the small prefill variant
again, the switch back is refused as over budget, and — because that refusal's
return value is discarded — the engine proceeds with mismatched graph tensors
and fails as the generic `batch dispatch failed`.

Reported to Qualcomm with a standalone reproducer. **Fixed in 2.51.0.260929**:
the reproducers no longer trigger on a stock 2.51.0, on a multi-context bundle
or a single-context one, and the reset path now rewinds the cache group.

## D2 — reset does not rewind the KV position allocator

A second, separate reset defect, and a **2.49 regression**: 2.48 does not
reproduce it.

Symptom: one long generation, then a short one that ends on EOS, then another
long one — and the third fails during prefill with
`GENIE_STATUS_WARNING_CONTEXT_EXCEEDED`, even though the context is 4096 and the
prompt is a few hundred tokens, and even though the dialog is reset between
each. The slot does not recover.

**Cause.** Resetting a slot recycles its KV positions onto the *back* of a free
list, while allocation takes from the *front*, and the high-water cursor is
never rewound. After a long run followed by a short one, the next request is
handed positions that start part-way up the cache rather than at zero, and the
engine rejects the batch. Same-length requests never expose it; you need
**long → short → long**.

Fixing D1 alone does not fix this — it only changes the symptom from D1's `-1`
to this warning.

Reported to Qualcomm together with D1. **Fixed in 2.51.0.260929** (measured):
the long → short → long reproducer no longer triggers.

## D3 — a library you rebuild has no grammar

**This one is not version-specific.** It is the same in every build we have
looked at, 2.48.40.260702 through 2.51.0.260929.

The XGrammar backend is **not part of the SDK's source drop**:
`GrammarBackend::create` is declared in `qualla/grammar.hpp` and defined in no
shipped source file, and there is no `ENABLE_GRAMMAR` option in the shipped build
files. Anything built from `examples/Genie/Genie/` therefore refuses to create a
dialog whose `genie_config.json` carries `dialog.context.grammar`:

```
Grammar backend configured but qualla was built without ENABLE_GRAMMAR; rebuild with -DENABLE_GRAMMAR=ON
```

and `GenieDialog_create` returns `-1`. The flag that message names does not exist
in the shipped build system, so its advice cannot be followed.

Check any library:

```sh
strings libGenie.so | grep -ci xgrammar
#  96  -> stock 2.48.x-2.50.x (grammar available); 97 on stock 2.51.0
#  1-2 -> rebuilt (grammar compiled out)
```

Verified on 2.48.40 through 2.50.0: no grammar implementation file, no
`ENABLE_GRAMMAR` in the build files, and 96 in every shipped `.so`. 2.51.0 is
the same, with 97.

**The per-request grammar API added in 2.51.0 is also missing from a rebuilt
library.** `GenieDialog_setGrammar` is exported, but a rebuilt library fails
every call to it. This server detects that when it loads a model and refuses
`response_format` / `structured_outputs` on that slot with a `400`
`grammar_not_supported` (see
[MANUAL § Grammar-Constrained Decoding](./MANUAL.md#grammar-constrained-decoding)).
Measured on two rebuilt 2.51.0 libraries.

**On 2.49.x and 2.50.x, fixing D1/D2 is therefore a trade-off, not a free
upgrade**: the stock library has grammar and the reset defects; a rebuilt one
fixes the defects and loses grammar. On those builds **there is no SDK-supplied
combination that keeps grammar *and* a slot that cannot wedge.** On 2.51.0 the
stock library has both, so the trade-off is gone unless you rebuild for D6.
Raised with Qualcomm, asking for the backend to be included in the source drop.

**The backend can be supplied yourself.** XGrammar is an open-source project
(Apache-2.0). Implementing the declared `GrammarBackend::create` on top of the
upstream library and building with `ENABLE_GRAMMAR` defined gives a rebuilt
library grammar: both the bundle's own grammar and the per-request API. We
have done this for 2.51.0 and checked it on hardware with this server. Such a
library can also carry the fix for D4. The patch is not shipped here.

## D4 — grammar leaks the terminal token into the text

With grammar-constrained decoding on a stock library, all three grammar kinds
(JSON Schema, regex, EBNF) constrain the output correctly, but **the response
ends with the model's end-of-sequence token as literal text** (`<|im_end|>` for
a ChatML model). The server does not strip it. See
[MANUAL § Grammar-Constrained Decoding](./MANUAL.md#grammar-constrained-decoding).

Measured on 2.49.40.260810, 2.50.0.260828 and 2.51.0.260929 — the same five of
the eight grammar checks fail the same way on all three. On 2.51.0 the
per-request grammar (`response_format`, `structured_outputs`) leaks the same
token. Not tested on the other builds.

The cause is one branch of the SDK's basic dialog: when the grammar says the
output is complete, it hands the terminating token to the token callback as
text. A library rebuilt with that branch changed to pass no tokens stops the
leak; measured on hardware with all three grammar kinds.

## D5 — reset corrupts a speculative-decoding dialog

Applies only to bundles whose `dialog.type` is `ssd-q1` (self-speculative
decoding with a forecast prefix). Those bundles are recognisable by a
`forecast-prefix/` directory beside the config and context binaries named
`..._ar64_ar128_...` rather than `..._ar128_ar1_...`.

Symptom: after `GenieDialog_reset`, **the first generated token is correct and
everything after it drifts into unrelated tokens** — plausible-looking words
with no relation to the prompt. Since a stateless server resets before every
request to keep them independent, that is every response.

**Cause.** `SelfSpecDecDialog::completeInit()` prepares the engine's slot,
restores the forecast prefix, sets the dialog's `_n_past`, and then calls
`initSlotFromRestore()` so the engine's slot agrees. `SelfSpecDecDialog::reset()`
does only the middle two. The `Dialog::reset()` it calls first has just set that
slot's `n_past` to zero, so the dialog believes it holds a prefix the engine
does not know about, and every position after the first is off by its length.

Bisected on hardware: making each pre-query SDK call skippable one at a time
isolates it to the reset. The SDK's own `genie-t2t-run`, which never resets, is
correct on the same bundle and the same config — the same 8 buffers, the same
188 MB allocated.

**It takes LoRA with it.** The SDK requires a reset after switching adapters, so
on an `ssd-q1` bundle applying an adapter either corrupts the dialog (with the
reset) or does nothing observable (without it).

**Fixed by a source patch** — the same rebuild that fixes D1 and D2. `reset()`
becomes a mirror of `completeInit()`: fourteen lines in one function, no ABI
change (the 118 exported symbols and the `NEEDED` list are identical). Verified
on hardware: correct output with `ssd-q1` left in place, independent across
resets, LoRA usable again, and speculative decoding delivering **4.42 tok/s
against 3.57** for the same prompts on the same bundle configured as `basic`.
The non-speculative path does not regress.

**Without a patched library**, rewrite the bundle's `dialog.type` to `basic` and
drop the `ssd-q1` block. The same context binaries load and generate correctly;
you lose the speculative speed-up, not the model.

**Which versions.** This is a **2.49 defect**, in every 2.49 we have: measured
on 2.49.40.260810, and present in 2.49.1.260821 whose `ssd-q1.cpp` is
byte-identical. **2.48.40.260702 cannot have it** — the slot calls this is about
arrived with 2.49's scheduler, so there the constructor and `reset()` already do
the same thing as each other. **2.50.0.260828 does not fix it** — its
`ssd-q1.cpp` is byte-identical to 2.49.40.260810's, and the reproducer still
triggers there. **2.51.0.260929 fixes it in its sources**: `reset()` now
prepares the slot and calls `initSlotFromRestore()`, the same change as our
patch. We have not run it on hardware, for lack of an `ssd-q1` bundle on the
board. The check takes a minute on the sources the SDK ships: open
`examples/Genie/Genie/src/qualla/dialogs/ssd-q1.cpp` and compare
`SelfSpecDecDialog::reset()` against `SelfSpecDecDialog::completeInit()`. If
`completeInit()` prepares the slot and calls `initSlotFromRestore()` around the
prefix restore and `reset()` does not, the defect is there. If both do, or
neither does, it is not.

## D6 — save and restore break a sliding-window cache group

Applies to bundles with a sliding-window cache group, which Gemma 4 has, and to
anything that calls `GenieDialog_save` / `GenieDialog_restore` on them — in this
server, the [prefix KV cache](./MANUAL.md#prefix-kv-cache). A restored prefix
longer than the group's budget (about 640 tokens on the Gemma 4 E2B bundle we
measured) gives wrong output, and a longer one kills the process. Symptom,
measurements and what to do are in
[Platform Notes](./PLATFORM_NOTES.md#prefix-cache-on-a-sliding-window-bundle-gemma-4).

Measured on stock 2.49.40.260810 and stock 2.51.0.260929 with the same result:
a 700-token prefix restores to different output, a 1540-token one segfaults.
2.50.0.260828 has the same code. Reported to Qualcomm with a patch; not fixed in
2.51.0.

## Checking your own SDK

Send one request comfortably over the budget, then a trivial one:

```bash
python3 - <<'PYEOF'
import requests
B = "http://<host>:8080"
def chat(pad, mt):
    msgs = ([{"role": "system", "content": "filler " * pad}] if pad else [])
    msgs.append({"role": "user", "content": "Say hello."})
    r = requests.post(B + "/v1/chat/completions", timeout=300, json={
        "model": "genie-local", "max_tokens": mt, "temperature": 0, "messages": msgs})
    print(r.status_code, r.json().get("usage") or r.json())
chat(900, 4)   # oversized: succeeds on both affected and fixed SDKs
chat(0, 4)     # affected SDK: HTTP 500, "GenieDialog_query failed with status -1"
PYEOF
```

If the second call fails, your SDK has D1. Reload the slot to recover:
`POST /v1/models/switch {"slot": "<name>", "model_dir": "<the same model>"}`.

> [!NOTE]
> **Bisect on the total, not on the prompt.** Probing with a long prompt and a
> small `max_tokens` only exercises prefill, which is the path that behaves
> well — that is exactly the mistake that produced the retracted claim in
> [D1](#a-single-context-length-raises-the-budget-but-does-not-remove-the-defect).
> Reload the slot between probes.

## Choosing a library

### On 2.51.0 — the stock library is a reasonable default

A stock 2.51.0 library has none of the reset defects (D1, D2, D5) and has
grammar, including the per-request API. What it still carries:

- **D4**: constrained output ends with the terminal token as text. Strip it in
  the client, or rebuild with the fix.
- **D6**: on a sliding-window bundle (Gemma 4), keep a warmed system prompt
  under about 600 tokens, or do not warm that model up.
- **A swallowed error** (change 2 below): when the SDK fails to switch graph
  variant it carries on and reports `batch dispatch failed` instead of the real
  reason. Harmless while D1 is gone, but it makes an engine failure hard to
  read.
- **The extra BOS on a prefix-cache hit** on a LUT-embedding bundle — see the
  note in [MANUAL § Prefix KV Cache](./MANUAL.md#prefix-kv-cache).

A rebuilt 2.51.0 library is worth it only for those. Rebuilding drops grammar
unless you also supply the backend ([D3](#d3--a-library-you-rebuild-has-no-grammar)).
If you rebuild, note that 2.51.0's OE makefile misses even more source
directories than 2.50.0's (see [What 2.51.0 changed](#what-2510-changed)).

The rest of this section is for 2.49.x and 2.50.x.

### Option 1 — run a fixed `libGenie.so` (recommended on 2.49.x / 2.50.x, unless you need grammar)

The SDK ships the complete, buildable reference sources for the Genie library
under `examples/Genie/Genie/`, with makefiles for OE/Android/x86 (see the SDK's
own "Genie Sample Code Tutorial"). Three changes remove D1 and D2:

1. When a dialog is reset, return that slot's KV positions to the cache group's
   occupancy counters. The reset path currently only zeroes cache memory.
   (There is also an ordering bug in the reset path that makes even that a
   no-op, which needs fixing first.) — D1
2. Stop discarding the failure return of the internal "switch active graph
   variant" call, so a refused switch fails the batch with its real reason
   instead of continuing with stale tensors. — D1
3. On reset, also rewind the scheduler's KV position allocator and its
   high-water cursor. — D2

Verified on hardware: the rebuilt `libGenie.so` exports exactly the same 118
`Genie*` symbols as the stock one — a drop-in replacement — and after the swap a
request that exceeds the budget fails only *itself*, as
`GENIE_STATUS_WARNING_CONTEXT_EXCEEDED`, leaving the slot healthy. This server's
hardware integration suite goes from 3 failures to all green.

Point the server at it with `GENIE_LIB_PATH` in `env_config.json`, or replace the
file in the SDK's `lib/` directory. If you use `GENIE_LIB_PATH`, note that Genie
resolves `libQnnSystem.so` **relative to its own location**, so the directory you
point at must also contain the rest of the QAIRT runtime libraries (symlinks are
fine).

**Cost: you lose grammar-constrained decoding**
([D3](#d3--a-library-you-rebuild-has-no-grammar)), unless you also supply the
grammar backend.

### Option 2 — stay under the budget

If you cannot replace the library, or you need grammar:

- Keep `prompt_tokens + max_tokens` under `context_length − AR_N` for every
  request. **open-genie-server does not enforce this for you** — `max_tokens`,
  when unspecified, defaults to the remaining *declared* context (4096-class),
  which is far above the real budget. Set `DEFAULT_MAX_TOKENS` in
  `env_config.json` and pass explicit `max_tokens` values sized to your budget.
- Prefer a **single context length** export. It does not remove the defect, but
  it raises the budget by an order of magnitude. It also costs decode speed on
  short requests — see
  [MANUAL § What several context lengths buy you](./MANUAL.md#what-several-context-lengths-buy-you).
- Treat `-1` as fatal for the slot: reload it with `/v1/models/switch` and retry
  once.

## What 2.49.1 changed

For the record, since the version number invites the question. Relative to
2.49.40.260810, the Genie sources differ in **12 files, +607/−192 lines, all in
the engine layer**. The public C API headers are byte-identical and the shipped
`libGenie.so` exports the identical symbol set, so it is a drop-in swap.

The changes are almost entirely a rework of how the continuous-batching
scheduler allocates KV positions for **sliding-window attention (SWA) cache
groups** — models that declare a non-default cache group with
`longcontext.type: sliding-window`. There is also a fix for interpreting tensor
dimensions when the runtime batch size is greater than 1.

**None of it reaches the defects on this page**, and in our testing none of it
changed observable behaviour:

- The full hardware integration suite returned identical verdicts on both
  builds, and generated text was byte-identical across long generations, long
  prompts and a function-calling benchmark subset.
- Every SWA change lives on the continuous-batching scheduler path. A model only
  reaches it if it has an SWA cache group **and** passes the scheduler's entry
  conditions — which exclude models with per-layer embeddings, among others.
- The models we have are excluded for one reason or the other: the Qwen3
  bundles have no SWA cache group, and the Gemma-family bundle that does have
  one also has per-layer embeddings, so it never enters the scheduler.

If you run a **sliding-window model without per-layer embeddings** — the code
paths 2.49.1 reworked — this is the release that matters to you, and we have no
measurements for it. Note one risk if you try: the new prefill-chunk clamp sizes
the chunk to `capacity - (window - 1)`, so a bundle whose SWA buffer is exactly
its window size collapses to one token per prefill fire.

## What 2.50.0 changed

A much bigger release than 2.49.1, and it still carries D1 to D6.

**It is a drop-in swap.** The public C API headers are byte-identical to
2.49.40.260810's, and the shipped `libGenie.so` exports the identical 118
`Genie*` symbols. Pointing the server at it needs nothing but
`QAIRT_SDK_ROOT` and the matching Hexagon skel directory on
`ADSP_LIBRARY_PATH`; nothing in this server needs recompiling or reconfiguring.

> [!CAUTION]
> **A green test run on a stock library is not evidence that D1 and D2 are
> absent.** This project's integration suite passed in full against a **stock**
> 2.50.0 — but it was run on a single-context-length `[8192]` / AR-128 bundle,
> whose budget is 8064 tokens, and no test in it comes near that. The
> reproducers, run on the same board in the same session against a multi-context
> bundle, wedged the slot exactly as they do on 2.49.40. The suite's earlier
> "three failures on stock, all green once patched" result was measured on a
> multi-context bundle, where the budget is 384 and ordinary requests cross it.
> **What a suite catches here depends on the bundle, not on the library.**

**What it does fix, for this server: the node creation order.** Building a
`GenieNode` pipeline on 2.49.x, creating the image encoder before the text
generator left the text generator unable to allocate its weight-shared
context — `Could not create context from binary for context index = N : err
1002` — and the server works around it by creating the text-generator node
first. On 2.50.0 the SDK's own `genie-app`, running a bundle's own script in the
bundle's own order, completes instead of dying there. Measured on the same board
minutes apart with the same free memory: 2.49.40 segfaults after the allocation
failure, 2.50.0 finishes twice.

> [!NOTE]
> **We have not established which layer fixed that.** The test swapped the whole
> 2.50.0 tree — library, QNN backend and Hexagon skels together — and the release
> notes' Genie section does not mention it. **The workaround stays in**: it is
> free, and it is still needed on every 2.49.x.

**What it does not fix.** D1, D2, D4 and D5 were re-run against a stock 2.50.0
and reproduce with the same verdicts, and in D1/D2's case the same wording, as a
stock 2.49.40 run back to back as a control. The sources agree: the reset
ordering in `qualla/dialog.cpp`, `KVManager::clearSlotPositions()`,
`SelfSpecDecDialog::reset()` and the terminal-token branch in
`qualla/dialogs/basic.cpp` are all unchanged.

**Other changes worth knowing about**, none of which we have measured:

- **DirectIO** — an opt-in path that loads a large model's weights straight into
  DMA buffers (`qualla/engines/qnn-api/UDmaBufAllocator.cpp`, new in this build).
- **Multi-tensor per-layer embeddings** — per-layer embedding tables can now be
  split across several tensors (`qualla/embeddings/EmbeddingTable.cpp`, new).
- **Engine-layer fixes** listed in the SDK's release notes: sliding-window cache
  with wide prefill chunks, batch-dimension interpretation, graph ordering with
  ten or more context splits, and encoder-output padding accuracy.
- **The sampler config is no longer translated.** 2.49.x passed a dialog's
  `sampler` block through a whitelist that also supplied defaults — `type:
  "basic"` when absent, and `role: "primary"` always. 2.50.0 hands the block to
  the engine as written. Our bundles are unaffected (the integration suite,
  including greedy determinism, is all green), but **do not rely on the library
  filling those in** if you hand-write a sampler block.

**If you rebuild the library** for D1/D2/D5, the three changes described under
[Choosing a library](#option-1--run-a-fixed-libgenieso-recommended-on-249x--250x-unless-you-need-grammar)
apply to 2.50.0's sources unchanged. One extra step: the shipped OE makefile is
byte-identical to 2.49.40's, so it still misses several source directories, and
it has **no entry at all for the new `src/qualla/embeddings/`** — add one, or
`EmbeddingTable.cpp` will not be linked in. We have not built 2.50.0.

## What 2.51.0 changed

**It fixes the three reset defects.** D1, D2 and D5 are gone from 2.51.0.260929.
For D1 and D2 this is measured: the reproducers do not trigger on a stock
2.51.0, on a multi-context bundle or a single-context one. For D5 it rests on
the sources (see [D5](#d5--reset-corrupts-a-speculative-decoding-dialog)).

**It also fixes the VLM pipeline's context overrun.** On 2.49.x and 2.50.x, a
VLM request whose prompt passes the context wedges the slot, and a request far
past it kills the process (see
[MANUAL § Limiting visual input](./MANUAL.md#limiting-visual-input)). On a stock
2.51.0, a standalone reproducer with the same Qwen3-VL-4B bundle gets status 4
(`GENIE_STATUS_WARNING_CONTEXT_EXCEEDED`) for the overrun, the pipeline answers
normally after a reset, and 17 frames no longer kill the process. We have not
yet measured this through this server.

**It is a drop-in swap for this server.** Genie's C API moves from 1.20.0 to
1.21.0: ten functions are added and none removed, and the stock library's
`Genie*` exports go from 118 to 128. Bundles exported with 2.49.40 load and run
on the 2.51.0 runtime without re-export. This server's integration suite on a
stock 2.51.0 gave the same verdicts as on 2.49.40: all green on a Qwen3-1.7B
bundle, and the usual single known failure on a Gemma 4 E2B text bundle.

**What it adds that this server uses:**

- `GenieDialog_setGrammar` replaces a dialog's grammar without recreating it.
  The server maps `response_format` and `structured_outputs` onto it; see
  [MANUAL § Grammar-Constrained Decoding](./MANUAL.md#grammar-constrained-decoding).
  The server logs the Genie API version at startup.

**What it adds that this server does not use yet:**

- `GenieNode_setValue` with `GENIE_NODE_PARAM_TEXT_GENERATOR_MAX_NUM_TOKENS`
  sets a VLM text generator's token limit per execution. VLM slots still use
  `VLM_SLOTS[].max_tokens` for the whole slot; there is still no abort and no
  stop sequence on that path.
- Per-stage performance policies (prefill / decode), and a count of the tokens
  the last query generated.

**One behaviour change.** A `SENTENCE_BEGIN` or `SENTENCE_CONTINUE` query now
only prefills. Up to 2.50 it sampled one token as well. The prefix-cache warmup
uses `SENTENCE_BEGIN`; on hardware, with a library rebuilt for D6, hits matched
misses at least as often as on 2.49.40.

**What it does not fix:** D3, D4, D6, the extra BOS on a LUT-embedding bundle's
prefix-cache hit, and the swallowed graph-variant error (change 2 of
[Option 1](#option-1--run-a-fixed-libgenieso-recommended-on-249x--250x-unless-you-need-grammar)).
The reset now also clears the whole KV buffer, which costs a little time per
request; we have not measured how much on 2.51.0.

**If you rebuild the library**: the shipped OE makefile misses the four source
directories 2.50.0's did, plus five that are new in 2.51.0 (the config manager,
the JSON schema validator and three engine directories). Add the same set
`make/Android.mk` lists. The library links without them, because the shared
object is not linked with `--no-undefined`, and fails at `dlopen` on the board.

## What we have and have not tested

**Tested:** SA8255P board, `aarch64-oe-linux-gcc11.2`, Qwen3 w4a16 bundles
(`qwen3_0_6b`, `qwen3_1_7b`, `qwen3_4b_instruct_2507`, `qwen3_vl_4b_instruct`)
and one Gemma-family bundle, on 2.49.40.260810 (extensively), 2.49.1.260821
(integration suite, generation probes, a benchmark subset) and 2.50.0.260828
(the D1/D2/D4/D5 reproducers with a stock 2.49.40 run as a control, the
integration suite, and the grammar checks), and 2.51.0.260929 (the D1/D2/D4/D6
reproducers, the VLM overrun reproducer, the integration suite on stock and
rebuilt libraries, the grammar checks, and the per-request grammar through this
server on four libraries).

**Not tested:** every other SoC, ABI and model family. The D1/D2 reproducers on
2.49.1. D4 on 2.48.40.260702 or 2.49.1.260821. D5 on 2.51.0. A rebuilt 2.50.0
library. 2.50.40. Any SDK older than 2.48.40.260702 or newer than
2.51.0.260929.

**Treat an untested SDK build as suspect** and run
[the check](#checking-your-own-sdk) against it.
