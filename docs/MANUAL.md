# open-genie-server Manual

*English | [日本語](./MANUAL.ja.md)*

How the server is put together, how to configure it, and what it does in practice — architecture, `env_config.json`, each feature area, and the operational notes behind them. The endpoint reference is [API.md](./API.md).

## Table of Contents

1. [Glossary](#glossary) — **HTP, NSP, cDSP and "Hexagon NPU" all name the same silicon at different levels; start here if that is new**
2. [Architecture Overview](#architecture-overview)
3. [SDK Compatibility](#sdk-compatibility) — how many context lengths to export at
    - → [**QAIRT Version Issues**](./QAIRT_VERSIONS.md) — **the SDK defects, per version. Read this before deploying**
4. [Multi Text Slots](#multi-text-slots)
    - → [**Platform Notes**](./PLATFORM_NOTES.md) — **how many NSP cores you may use is licensed per SKU. Read this before assuming a second slot exists**
    - [Loading two models at once](#loading-two-models-at-once) — **read this before configuring a second slot**
5. [Configuration (env_config.json)](#configuration-env_configjson)
6. [Starting the Server](#starting-the-server)
7. [Running on Android](#running-on-android) — what a non-OE-Linux target changes
8. [Chat Template Selection Rules](#chat-template-selection-rules)
9. [Prefix KV Cache](#prefix-kv-cache)
10. [Grammar-Constrained Decoding](#grammar-constrained-decoding)
11. [Profiling (SDK-side KPIs)](#profiling-sdk-side-kpis)
12. [VLM (Multimodal) Support](#vlm-multimodal-support)
13. [Token counting](#token-counting) — and what breaks without the `tokenizers` package
14. [Logprobs](#logprobs) — generated-token logprobs and prompt scoring
15. [Open WebUI](#open-webui)
16. [Switching Models and LoRA](#switching-models-and-lora)
17. [Integration Testing](#integration-testing)
18. [lm_eval Integration](#lm_eval-integration)
19. [Performance policy](#performance-policy)
20. [Benchmarking](#benchmarking)
21. [Troubleshooting](#troubleshooting)
22. [Limitations](#limitations)

**The endpoint reference lives in [API.md](./API.md)** — every request and
response shape, the status codes, and the error envelope.

---

## Glossary

Qualcomm's stack names one piece of hardware three ways, and this document uses all three because the SDK, the config files and the device paths do. They are not synonyms for each other at the same level. The "Hexagon NPU" of the README is the outward name for that same silicon; the **NPU** and **Hexagon** rows say how it lines up with the three:

| Term | What it means here |
|---|---|
| **HTP** | Hexagon Tensor Processor — the accelerator, and the name of the QNN **backend** that drives it. Everything named after it is backend machinery: `libQnnHtp.so`, `htp_backend_ext_config.json`, `QnnHtp.poll`. "The HTP" is the *kind* of processor, not a particular one. |
| **NSP** | **One HTP device** — one core. A dual-NSP SoC has two of them. Used when counting cores or talking about which one a model lands on. |
| **cDSP0 / cDSP1** | The same cores under their OS-level names, which is how they appear in device paths (`/dsp/image/dsp/cdsp0`). Qualcomm's own docs write "CDSP/NSP ID" and "CDSP0/NSP0" — **cDSP*n* and NSP*n* are the same core**. Lowercase `cdsp0` in this document is always a literal device path, never a slot name — slots are named for their role (see **slot** below). |
| **`device_id`** | The number that selects which one: `0` = cDSP0/NSP0, `1` = cDSP1/NSP1. It is the QNN HTP backend's `devices[0].device_id`, and it is the only thing that actually decides placement. |
| **NPU** | Neural Processing Unit — the generic industry word for this class of accelerator. Qualcomm's marketing name for the silicon above is the **Hexagon NPU**, and that is the sense in which the README uses it: "runs on a Hexagon NPU" means "runs on the HTP, through the QNN HTP backend, on one of its NSP cores". It is a *product* name, not a level in the stack — no config key, log line or device path is named after it. Prefer HTP / NSP / cDSP whenever you need to be precise about which. |
| **Hexagon** | The Qualcomm DSP architecture the whole family is built on, and the token you see in version strings and paths: `HEXAGON_VERSION: "v73"`, `lib/hexagon-v73/unsigned` in `ADSP_LIBRARY_PATH`. "Hexagon *NPU*" is the accelerator; "Hexagon *v73*" is the instruction-set version of the DSP cores in it. The two are not interchangeable. |
| **slot** | **This server's own concept**, not a Qualcomm term: one resident model plus its `GenieDialog` handle, its lock, and its KV state. A slot's `name` is a logical address clients use (e.g. `chat`, `tool_call`, `vision`); `device_id` pins it to a physical core. |

The rest is Qualcomm's or ours:

| Term | What it means here |
|---|---|
| **QAIRT** | Qualcomm AI Runtime — the SDK this server links against. Supplies QNN and Genie. |
| **Genie** | The QAIRT C API for running LLMs (`libGenie.so`): `GenieDialog_*` for text, `GenieNode_*`/`GeniePipeline_*` for multimodal. |
| **`GenieDialog`** | One Genie conversation handle: a loaded model, its KV cache, and its sampler. One per slot. |
| **qualla** | The layer inside `libGenie.so` between the Genie C API and the QNN backend. Appears in SDK log lines and error messages. |
| **model bundle** | A directory holding a model compiled for the device: `genie_config.json`, the `.bin` context binaries, the tokenizer, and the HTP backend extension config. What `model_root` points at. |
| **`ctx-bin`** | A pre-compiled QNN context binary — the `.bin` files in a bundle. Compiled for one SoC and one context length; not portable between them. |
| **context length (CL)** | The token window a bundle was compiled for. A bundle may carry several compiled variants; the request picks the smallest it fits in. |
| **prefill / decode** | The two phases of a generation: prefill processes the whole prompt at once, decode emits one token at a time. They have very different speed characteristics, so measurements separate them. |
| **TTFT** | Time To First Token — wall time from request to the first emitted token. In practice, prefill. |
| **KV cache** | The attention key/value state a `GenieDialog` accumulates. Its size bounds how much context a slot has left. |
| **prefix cache** | This server's reuse of a KV cache for a repeated prompt prefix (typically a system prompt), so the same prefill is not paid twice. |
| **hot-swap** | Replacing the model in a slot at runtime via `POST /v1/models/switch`, without restarting the server. |
| **co-residency** | Keeping two models loaded at once on the same device. Whether it fits is not predictable from any number you can read off the bundles — it depends on how they were exported, and it must be measured. See [Loading two models at once](#loading-two-models-at-once). |
| **w4a16** | A quantization scheme: 4-bit weights, 16-bit activations. Describes how a bundle was compiled. |
| **LoRA** | A small adapter applied on top of a loaded model to change its behaviour without loading a different one. |
| **VLM** | Vision-Language Model — a model that takes images as well as text. Runs through Genie's composable pipeline rather than `GenieDialog`, which is why VLM slots are configured separately. |
| **GVM** | Guest VM — the target runs as a guest under a hypervisor rather than on bare metal. Relevant because some device-level controls do not reach the hardware from inside one. |

## Architecture Overview

- The implementation is a Python package, `genie_server`, driven by a thin
  launcher (`genie-server.py`) or the `genie-server` command an install
  provides. It lives under `src/` in the repository (`src/genie_server/`);
  the module paths below are written from the package root:

| Module | Role |
|---|---|
| `genie_server/config.py` | `env_config.json` parsing, process-environment setup |
| `genie_server/capi.py` | ctypes bindings for the GenieDialog C API (`GenieLib`) + SDK constants |
| `genie_server/templates.py` | chat-template rendering (chatml/llama3/llama2/gemma/gemma4), prefix splitting, `/no_think` |
| `genie_server/tools.py` | OpenAI `tools` (function calling): Hermes prompt rendering + output parsing |
| `genie_server/slots.py` | `Slot`/`SlotManager`: model loading, request routing, hot-swap |
| `genie_server/prefix_cache.py` | on-disk KV-cache snapshots |
| `genie_server/engine.py` | the generation engine: lock, watchdog, SDK params, prefix cache, `finish_reason` |
| `genie_server/vlm.py` | multimodal requests (`GenieNode`/`GeniePipeline`) |
| `genie_server/genie_node.py` | ctypes bindings for the `GenieNode`/`GeniePipeline` composable pipeline API |
| `genie_server/vlm_specs.py` | per-VLM-model specs (preprocessing, node topology, prompt template) |
| `genie_server/logprobs.py` | token logprobs via the SDK's custom-sampler hook (sample + teacher-forcing modes) |
| `genie_server/protocol.py` | OpenAI wire-format builders and the error envelope |
| `genie_server/app.py` | all FastAPI routes (`create_app`) |
| `genie_server/bootstrap.py` / `asgi.py` | startup wiring / uvicorn entry point |
| `tests/` | offline test suite (`FakeGenieLib` — runs without an NPU or `libGenie.so`) |

- The server is made of one or more **slots** (`Slot`). One slot = one independent `GenieDialog` handle + its own `threading.Lock` + its own tokenizer/template/LoRA state. Without `TEXT_SLOTS` set there's exactly one slot (`"default"`), matching every prior single-model version of this server exactly.
- Each request's `model` field (in the body, or the `?model=` query param) decides **which slot it's routed to** (`SlotManager.select`). If no slot matches, it always falls back to the primary slot, so `lm_eval`'s fixed `"genie-local"` string and single-slot deployments never need any client-side changes.
- Inference, parameter changes, LoRA operations, and model switching within a slot are all serialized by **that slot's own** `threading.Lock`. **Operations on other slots are never blocked** — a 2-NSP configuration can genuinely process two requests at once.
- Every request runs `GenieDialog_reset` on its target slot before starting. In other words, this server **never keeps multi-turn conversation state on the SDK side**. Manage conversation history on the client (the `messages` array).
- Streaming bridges the C callback thread and the ASGI event loop with a `threading.Thread` + `asyncio.Queue`. A client disconnect actively aborts the in-flight request via `GenieDialog_signal(ACTION_ABORT)`.
- A watchdog (`threading.Timer`) automatically aborts inference after `INFERENCE_TIMEOUT` seconds (120s by default; configurable in `env_config.json`) elapse.

## SDK Compatibility

open-genie-server is a thin client of `libGenie.so`. Its behaviour — and its
failure modes — therefore depend on which QAIRT SDK build you run it against.

> [!IMPORTANT]
> **The known SDK defects, which build carries each of them, how to test your own
> SDK, and how to choose between a stock and a rebuilt library are collected in
> [QAIRT Version Issues](./QAIRT_VERSIONS.md).** Read that page before you pick
> an SDK build or decide how to export a model. In short:
>
> - A **stock 2.49.x library wedges a slot** when `prompt_tokens + max_tokens`
>   crosses `context_length − AR_N`. The request that does it returns 200 and
>   looks fine; every request after it on that slot fails until the model is
>   reloaded. **This server does not guard against it.**
> - **Exporting at a single context length does not remove that** — it only
>   raises the budget from 384 to 3968 on a `[4096]` / AR-128 bundle.
> - **Rebuilding the library fixes it but loses grammar-constrained decoding.**
>   There is no build that has both.
> - **2.49.1.260821 is newer than 2.49.40.260810** despite the smaller number,
>   and fixes none of the above.

**Verified against:** QAIRT **2.49.40.260810** and **2.49.1.260821**,
`aarch64-oe-linux-gcc11.2`, on an SA8255P board, with Qwen3 w4a16 context
binaries (`qwen3_0_6b`, `qwen3_4b_instruct_2507`).

The rest of this section is about an export choice that is *not* a defect: how
many context lengths to compile into a bundle.

### What several context lengths buy you

Compiling at several context lengths is not just the thing that exposes the
wedge. **The SDK runs the smallest compiled variant a request fits in, and a
smaller variant decodes faster.** A single-context-length bundle pays the
full-context rate from its first token.

Measured on the board, streaming so that prefill (TTFT) and the per-token decode
rate are separated, 120-token generations, median of three:

`qwen3_0_6b` — `context_lengths: [512, 1024, 4096]`:

| prompt | + generated | active CL | TTFT | decode |
|---:|---:|---|---:|---:|
| 34 | 154 | **512** | 82 ms | **68.91 tok/s** |
| 378 | 498 | **512** | 169 ms | **68.84 tok/s** |
| 548 | 668 | **1024** | 253 ms | **59.66 tok/s** |
| 807 | 927 | **1024** | 344 ms | **59.87 tok/s** |
| 980 | 1100 | 4096 | 424 ms | 35.85 tok/s |
| 1754 | 1874 | 4096 | 831 ms | 29.29 tok/s |
| 3903 | 4023 | 4096 | 2017 ms | 29.31 tok/s |

`qwen3_1_7b` — `context_lengths: [4096]`:

| prompt | TTFT | decode |
|---:|---:|---:|
| 33 | 141 ms | **20.49 tok/s** |
| 377 | 330 ms | **20.44 tok/s** |
| 1754 | 1319 ms | **20.42 tok/s** |
| 3904 | 2853 ms | **20.44 tok/s** |

The multi-context-length bundle holds a plateau per variant and steps down at
each boundary — **2.35x between its best and worst**. The single-context-length
bundle moves 0.3% across a prompt range of 33 to 3904 tokens.

The two models are different sizes, so read the shape, not the absolute rates.
The 0.6B also varies within its 4096 variant (35.9 → 29.3) as `n_past` grows;
the 1.7B does not, because weight traffic dominates its decode and the KV
contribution is buried. That difference is why the step pattern, not the drift,
is the thing to read.

So the trade-off is:

| | several context lengths | one context length |
|---|---|---|
| decode on short requests | **faster** (2.35x here) | pays the full-context rate always |
| memory | one set of buffers per variant | one set |
| slot wedge budget on a stock library | **384 tokens** | **3968** — raised, not removed |
| as the *first* of two resident models | leaves less room | leaves more |

Short conversations on a **fixed** library are the case where several context
lengths is the better export; on a stock library they are also the case that
reaches the wedge in a few hundred tokens. Everything else points the other way —
see [Choosing a library](./QAIRT_VERSIONS.md#choosing-a-library).

## Multi Text Slots

On a target with more than one *usable* Hexagon NSP core, setting `TEXT_SLOTS` in `env_config.json` keeps an independent model resident on each NSP core, so requests to different slots are processed concurrently instead of queueing behind one lock.

> [!IMPORTANT]
> **Two cores is not a property of the SoC's part number.** How many you may
> use is gated by your SKU's licence, so an otherwise identical board may
> expose one — in which case nothing in this section applies to it, and a slot
> at `device_id: 1` has nothing to bind to. The measurements below come from
> one machine whose guest was also given more memory than the image it started
> from. See [Platform Notes](./PLATFORM_NOTES.md) before reading any number
> here as a specification.

**What that is worth, measured** on the bench in [Platform Notes](./PLATFORM_NOTES.md). Two slots, the same `qwen3_0_6b` w4a16 on cdsp0 and cdsp1, 8 identical chat requests (SA8255P, QAIRT 2.49):

| | wall clock | vs. serial |
|---|---|---|
| one at a time, one slot | 8.06 s | — |
| two at a time, **same** slot | 8.09 s | 1.00× — the slot lock serializes them, as designed |
| two at a time, **one per slot** | 6.13 s | **1.31×** |

So the slots really do overlap — the same-slot row is the control, and it shows
no gain at all — but a second NSP buys about a third more throughput, not
double. Per-request latency rises from 1.0 s to 1.5 s while both are busy. The
ratio held at 1.33× with longer generations, at 1.38× with the two slots pinned
to disjoint CPU masks (`0x30` / `0xc0` instead of both inheriting `0xe0`), and
at 1.31× again with `poll: false`, which drops the server's CPU use from ~260%
to ~12% (see [`QnnHtp.poll`](#qnnhtppoll-costs-260-cpu-and-buys-nothing-here)).
Host CPU is therefore not the ceiling; the two NSPs share something below the
API, memory bandwidth being the obvious candidate.

**Where the missing throughput goes.** Splitting each request into prefill
(time to first token) and decode (the rest) shows the two halves behave
completely differently under concurrency — `qwen3_0_6b`, 64 generated tokens,
median of 5:

| prompt tokens | | TTFT | prefill | decode |
|---|---|---|---|---|
| 33 | solo | 0.048 s | 686 tok/s | 68.7 tok/s |
| 33 | both slots busy | 0.058 s | 570 tok/s | 44.1 tok/s |
| 121 | solo | 0.053 s | 2263 tok/s | 68.8 tok/s |
| 121 | both slots busy | 0.065 s | 1863 tok/s | 44.1 tok/s |
| 249 | solo | 0.093 s | 2668 tok/s | 69.1 tok/s |
| 249 | both slots busy | 0.110 s | 2264 tok/s | 43.9 tok/s |

Under concurrency each slot keeps **83-85% of its prefill rate but only 64% of
its decode rate**, at every prompt length. That is the signature of a memory
bandwidth limit: prefill reuses each weight across all prompt positions
(compute-bound), while decode streams the entire weight set for every single
token (bandwidth-bound), so two decoders contend and two prefills mostly do
not. Back of the envelope: ~0.3 GB of 4-bit weights per token at 68.7 tok/s is
~21 GB/s for one slot, and the pair tops out around ~26 GB/s.

It predicts the end-to-end numbers: 2 × 0.64 = 1.28× for a decode-dominated
workload (measured 1.31×) and 2 × 0.83 = 1.66× for a prefill-dominated one.
Testing that prediction directly — a 249-token prompt with `max_tokens: 1` —
gives **1.64×**.

**So the shape of your workload decides what a second slot is worth.** Long
prompts and short answers (classification, extraction, scoring, RAG re-ranking)
scale close to 1.6×. Chat-style short prompts and long answers scale ~1.3×.
`tests/integration/measure_ttft_tps.py` reproduces the table.

Size a deployment on ~1.3×, and treat the second slot primarily as a way to
keep two *different* models resident (or to stop one long request from
blocking every other client) rather than as a throughput doubling.

```json
{
  "QAIRT_SDK_ROOT": "/home/root/qairt/2.49.40.260810",
  "HEXAGON_VERSION": "v73",
  "MODELS_BASE_DIR": "/home/root/models",
  "PREFIX_CACHE_DIR": "/home/root/prefix_cache",
  "TEXT_SLOTS": [
    {"name": "tool_call", "device_id": 0, "model_root": "qwen3_1_7b-genie-w4a16-qualcomm_sa8775p"},
    {"name": "chat", "device_id": 1, "model_root": "qwen3_4b_instruct_2507-genie-w4a16-qualcomm_sa8775p"}
  ]
}
```

- `name`: the slot's logical identifier (used for routing and logs; e.g. `tool_call`, `chat`). Referenced by `POST /v1/models/switch`'s `slot` field and the `?slot=`/`?model=` query params.
- `device_id`: the QNN HTP backend's `devices[0].device_id` (0 = cDSP0, 1 = cDSP1, ...). SA8255P is `dsp_arch: v73`.
- `model_root`: the model directory (containing `genie_config.json`) this slot loads at startup.

> **Two slots do not always fit.** Whether the configuration above starts at all
> depends on which model each slot loads and, crucially, on their **load order** in the
> array — a smaller model (e.g. single-CL 1.7B) must load first and a larger model (e.g. 4B)
> second; loading 4B first causes the second model to fail with `err 1002` at startup.
> Read the next subsection before settling on a two-slot config.

### Loading two models at once

Configuring a second slot — two `TEXT_SLOTS` entries, or `TEXT_SLOTS` plus
`VLM_SLOTS` — can fail at startup with:

```
[ERROR] "Could not create context from binary for context index = N : err 1002"
[ERROR] Startup failed: GenieDialog_create failed: -1
```

`err 1002` is `QNN_COMMON_ERROR_MEM_ALLOC`. The first slot loads fine; the
second one fails partway through creating its contexts. **The server cannot
predict this** — there is no API that reports the budget, so the failure
surfaces only when you try the combination.

Everything below was measured on one board (SA8255P / dual NSP / 12.3 GB,
QAIRT 2.49.40.260810, `libGenie.so` with the `0001`+`0003` patches). Treat the
specific model sizes as illustrative of the shape of the limit, not as portable
numbers.

> **The limit moves with how the bundle was exported, not just with the model.**
> Two of these models were later re-exported at a single context length, and
> pairs that had failed began to load. Both tables are kept below, and which
> bundle each row was measured on is stated. Do not carry a row over to a bundle
> it was not measured on.

#### What is established

**Load order decides it, for two text models.** Nine startup configurations
were run against bundles compiled at **three context lengths each**
(`context_lengths: [512, 1024, 4096]`); the same pair of models succeeds or
fails depending only on which one is created first:

| Loaded first → second | `device_id` | Total params | Result |
|---|---|---|---|
| 0.6B → 0.6B | 0, 1 | 1.2B | OK |
| 0.6B → 0.6B | 0, **0** | 1.2B | **err 1002** |
| 0.6B → 1.7B | 0, 1 | 2.3B | OK |
| **1.7B** → 0.6B | 0, 1 | 2.3B | **err 1002** |
| 0.6B → 1.7B | **1, 0** | 2.3B | OK |
| 1.7B → 1.7B | 0, 1 | 3.4B | **err 1002** |
| **4B** → 0.6B | 1, 0 | 4.6B | **err 1002** |
| 0.6B → **4B** | 0, 1 | 4.6B | OK |
| **1.7B** → 4B | 0, 1 | 5.7B | **err 1002** |

Read together these give four statements that hold across every run:

1. **Total size is not the criterion.** 0.6B + 4B (4.6B) loads; 1.7B + 1.7B
   (3.4B) does not. The larger pair is the one that works.
2. **`device_id` is not the criterion.** Swapping which NSP each model is
   pinned to changes nothing (rows 3 and 5 are the same pair on opposite
   cores, both OK). Pinning selects a core; it does not partition a budget.
3. **What the *first* model is decides the outcome.** Every configuration that
   loaded a 0.6B first succeeded, and every configuration that loaded anything
   larger first failed — whatever the second model was. **On these bundles** the
   threshold lay somewhere between 0.6B and 1.7B; only three model sizes were
   tried, so it was not pinned down further. **The threshold is not a property
   of the model alone — see the next table.**
4. **The second model must go on the other NSP.** The one same-device
   configuration failed. Note this run also loaded the *same* model twice, and
   the generated HTP extension config carries
   `"context": {"weight_sharing_enabled": true}` — so a weight-sharing conflict
   is not excluded as the cause. Two *different* models on one NSP was not
   tested.

**A single-context-length export raises the threshold.** The 1.7B and the 4B
were later re-exported at one context length (`context_lengths: [4096]`, and
with a newer export toolchain: QAIRT 2.45.0 → 2.49.0). The 0.6B was left as it
was. Re-running the same startup configurations:

| Loaded first → second | first bundle | second bundle | Total params | Before | Now |
|---|---|---|---|---|---|
| 0.6B → 0.6B | multi-CL | multi-CL | 1.2B | OK | OK |
| 0.6B → 1.7B | multi-CL | single-CL | 2.3B | OK | OK |
| 0.6B → 4B | multi-CL | single-CL | 4.6B | OK | OK |
| **1.7B → 0.6B** | **single-CL** | multi-CL | 2.3B | **err 1002** | **OK** |
| **1.7B → 1.7B** | **single-CL** | single-CL | 3.4B | **err 1002** | **OK** |
| 1.7B → 4B | single-CL | single-CL | 5.7B | err 1002 | err 1002 |
| 4B → 0.6B | single-CL | multi-CL | 4.6B | err 1002 | err 1002 |

The rule keeps its shape — a bigger first model leaves room for less — but the
threshold moved. Where it used to be "the first model must be the 0.6B", it is
now:

- first model 0.6B → a 4B fits alongside it
- first model 1.7B → up to a 1.7B fits; a 4B does not
- first model 4B → nothing fits, not even a 0.6B

The two rows that changed are exactly the two where the re-exported 1.7B is
loaded first, which is consistent with a single-context-length bundle reserving
less as the first model. **It is not proof.** The re-export changed the toolchain
version as well as the context-length count, the 4B went single-context-length
too and still cannot be first, and `Allocated total size` for the 1.7B went *up*
slightly across the change (264 MB → 276 MB) while the behaviour improved. A
bundle exported at one context length also [raises the slot-wedge
budget](./QAIRT_VERSIONS.md#a-single-context-length-raises-the-budget-but-does-not-remove-the-defect),
so there is reason to prefer one regardless.

**Verified working, not just loading:** with two single-context-length 1.7Bs
resident, both slots answer, routing by `"slot"` is correct, and two concurrent
requests finish in 8.32 s against 5.90 s for one alone — 1.42x, in line with the
[parallelism measurement](#multi-text-slots).

**It is not host memory, and not fragmentation** — at least for two text
models. At the moment of failure:

```
/proc/buddyinfo order-10 = 993 free 4 MB blocks  (~4 GB contiguous)
MemAvailable  12.29 GB / 12.66 GB
CmaFree       95 MB / CmaTotal 100 MB            (CMA untouched)
```

A 276 MB allocation failed with 4 GB of contiguous memory free. This was also
re-tested on a **freshly rebooted** board — where 4 MB blocks are ~18× more
plentiful (2818 vs 156 after a long test session) — and the same pair failed at
the same point. So the exhausted resource is on the cDSP side, and the Linux
guest cannot see it.

**Large models do consume real host memory, though.** Loaded alone:

| Model alone | `MemAvailable` | order-10 blocks |
|---|---|---|
| 0.6B text | 12.27 GB | 179 |
| 1.7B text | 12.29 GB | 993 |
| **4B text** | **7.87 GB** | **0** |
| **~4B VLM** | **6.91 GB** | **15** |

The 4B-class models take 4–5 GB that is *not* reclaimable and drain the
large-block pool, while the small ones cost nothing measurable. None of it
shows up in the server process's RSS (170 MB with the VLM loaded) because the
allocations come from DMA-BUF. So for a 4B-class pair, host exhaustion is a
plausible contributor even though it demonstrably is not the cause for the
small-model pairs.

**A text model and a ~4B VLM do co-exist — if the text bundle is
single-context-length and the two sit on different cores.** This is a reversal:
with multi-context-length text bundles the pair failed in either order, with or
without the positional-encoding validator flags cleared, and that is still true
today. Re-running it with the 0.6B re-exported at one context length:

| Text bundle | `SLOT_LOAD_ORDER` | text `device_id` | VLM `device_id` | Result |
|---|---|---|---|---|
| **single-CL `[4096]`** | `text-first` | 0 | 1 | **OK — both slots ready, both answer** |
| **single-CL `[4096]`** | `vlm-first` | 0 | 1 | **OK** |
| single-CL `[4096]` | `text-first` | 0 | **0** | `err 1002` at context index 1 |
| multi-CL `[512,1024,4096]` | `text-first` | 0 | 1 | `err 1002` at context index 3 |

Load order does not matter here; **a different `device_id` does**, which is the
same same-core rule the two-text-model table shows. Verified working, not just
loading: four interleaved text/VLM rounds and one concurrent pair, all correct.

**This is the controlled comparison the section below used to be missing.** The
0.6B exists in both bundle shapes, so everything except the context-length count
is held constant — same VLM, same cores, same order, same library, same board
session — and the outcome flips. Better still, the two text bundles allocate
*byte-identical* amounts:

```
single-CL 0.6B :  Allocated total size = 275120640 across 3 buffers  -> VLM loads
multi-CL  0.6B :  Allocated total size = 275120640 across 3 buffers  -> err 1002
```

The VLM's own sequence is identical in both runs too (348,520,576 across 8
buffers). So **whatever `err 1002` is counting, it is not allocated bytes** — it
tracks something that scales with the number of context-length variants in the
bundle, i.e. with how many QNN contexts get created (two graphs versus six here).
That also means it is not something you can read off a bundle's size.

> **Measure this only on the first startup after a power cycle.** A failed
> startup does not clean up the device — the log fills with
> `Failed to deregister opPackages ... err 6020` and
> `Failed to teardown transport layer: 1004`, once per pdId — and later attempts
> do worse. The same configuration that succeeded first after a power cycle
> failed at context index 1 once two failed startups had accumulated behind it.
> Every row above was taken as the first startup after a power cycle.

See [Slot creation order](#slot-creation-order-slot_load_order) and
[`examples/config/env_config.text-vlm.sample.json`](../examples/config/env_config.text-vlm.sample.json).

#### What is not established

- **There is no formula.** None of parameter count, on-disk size, total size,
  or `Allocated total size` from the libGenie log predicts the outcome. The log
  line in particular is misleading: it reports 262 MB for the 0.6B and 264 MB
  for the 1.7B — nearly identical — for models that behave completely
  differently as the first load. It counts I/O buffers, not weights. The
  strongest case is the text + VLM pair above, where two bundles of the *same
  model* report the identical figure to the byte and yet one leaves room for a
  VLM and the other does not.
- **The mechanism is unknown.** `err 1002` says a DSP-side allocation failed,
  and nothing in the guest's `/proc` accounts for it. This was not traced
  through the SDK sources.
- **Why a *smaller* first model helps is unexplained.** The behavior is
  consistent with the first `GenieDialog` reserving the bulk of some DSP-side
  budget, but that is inference from the outcomes, not something observed.
- ~~**The context-length count's role is not isolated.**~~ **Now isolated, for
  the text + VLM pair.** The 0.6B was exported both ways and kept both bundles,
  which holds the toolchain version and everything else constant; the
  context-length count alone decides whether the VLM loads beside it (table
  above). For the *two text model* rows this is still indirect — those bundles
  were re-exported with a newer toolchain and no equivalent A/B exists.
- **Whether the exact numbers transfer to another SoC, another memory size, or
  another QAIRT version is unknown.** They were measured on one board.

#### Practical guidance

- **Order your slots smallest-first.** For `TEXT_SLOTS`, put the smallest model
  first in the array. For a text + VLM deployment, set
  `"SLOT_LOAD_ORDER": "text-first"` so the (smaller) text model is created
  first.
- **Prefer bundles exported at a single context length.** Two 1.7Bs co-exist
  that way and did not before, and it is what makes a text model and a VLM fit
  together at all. Re-measure after any re-export: the pairs that fit changed
  when these bundles were rebuilt.
- **For a text + VLM deployment**, use a single-context-length text bundle and
  put the two slots on different `device_id`. Start from
  [`env_config.text-vlm.sample.json`](../examples/config/env_config.text-vlm.sample.json).
- **Judge co-residency on the first startup after a power cycle.** A failed
  startup leaves the device dirty and biases everything you try afterwards.
- **Give each slot a different `device_id`** on a dual-NSP part.
- **Expect to determine the limit empirically.** Try the combination; if the
  server starts and both slots report `ready`, it works. There is no way to
  check in advance.
- **`err 1002` at startup is not a wedged DSP.** It is a clean failure — the
  process exits and nothing is left holding resources. Reordering the slots and
  restarting is safe and is the first thing to try.
- **Measuring the cost of a model:** load it alone and read `MemAvailable` and
  `/proc/buddyinfo`. Do **not** use `MemFree` deltas — model files land in the
  page cache and swamp the signal (a 0.6B can appear to cost more than a 1.7B).
  Neither metric captures the DSP-side allocation that actually fails, so both
  are indicative only.

### Pinning a device (`slots.pin_htp_device`)

Each model's `genie_config.json` points at an HTP backend extension config file via `dialog.engine.backend.extensions` (e.g. `htp_backend_ext_config.json`, following QNN's `"devices":[{"device_id":N,...}]` schema). **Without touching the model itself**, this file is copied per slot, its `device_id` overridden, and the copy saved as `PREFIX_CACHE_DIR/.htp_ext_cache/<slot-name>_<original-filename>` — the startup Genie config is then rewritten to reference that copy. As a result:

- Assigning the same model directory to multiple slots never pollutes the original config file.
- If a model's `genie_config.json` has no `extensions` (e.g. it doesn't use the HTP backend), the `device_id` setting is ignored, a warning is logged, and the server still starts (no NSP pinning happens — the backend uses whatever device it defaults to).
- If no `device_id` is given (`TEXT_SLOTS` not used, or `"device_id": null`), this step is skipped entirely and the model's own config file is used as-is (same as before).
- `device_id` is validated at startup: it must be an integer between 0 and 7, or absent. A string, a float, or a negative value is refused with the slot name in the message rather than written into the HTP config. **This is a typo guard, not a hardware check** — how many HTP devices a part actually has is only known to QNN when the dialog is created, so `"device_id": 1` on a single-NSP target still fails there (`GenieDialog_create ... err 1002`).

### ADSP_LIBRARY_PATH

The DSP-side skel library search path is built dynamically from the set of `device_id`s actually in use (e.g. if both 0 and 1 are in use, it includes `/dsp/image/dsp/cdsp0;/dsp/image/dsp/cdsp1`). Without `TEXT_SLOTS` set, it's just `cdsp0`, as before.

### Routing

| Operation | Slot selection | Fallback if unspecified/no match |
|---|---|---|
| `/v1/completions`, `/v1/chat/completions` | body's `model`, or body's `slot` (slot **name**) to override | primary slot (`slots[0]`) |
| `/v1/prefix/warmup`, `/v1/lora/*` | body's `model` | primary slot |
| `/v1/server/performance_policy` (GET), `/v1/lora/current` | query param `?model=` | primary slot |
| `/v1/server/idle` | query param `?slot=` (slot **name**) | primary slot |
| `POST /v1/models/switch` | body's `slot` (slot **name**, not a model ID) | primary slot |

Routing by `model` is decided by "does it exactly match the `active_model_id` (= model directory name) of the model currently loaded in that slot?" (`SlotManager.select`). `POST /v1/models/switch` is the one exception: since its target is the "hardware slot about to have its model swapped out" itself, it's selected by **slot name**, not model ID.

If two slots load the same model directory (so their `active_model_id` is identical and `model` can't disambiguate them — see [Limitations](#limitations)), pass an explicit `"slot": "<name>"` in the `/v1/completions`/`/v1/chat/completions` body to target one directly; it takes priority over `model` when given.

---

## Configuration (env_config.json)

An `env_config.json` is required in the server's startup (current) directory. The server exits at startup if it's missing.

| Key | Required | Default | Description |
|---|---|---|---|
| `QAIRT_SDK_ROOT` | effectively required | `""` | Root path of the QAIRT SDK. Also used for `QNN_SDK_ROOT`/`ADSP_LIBRARY_PATH`. |
| `HEXAGON_VERSION` | optional | `"v73"` | Hexagon version used in `ADSP_LIBRARY_PATH` (e.g. `hexagon-v73`). |
| `TARGET_PLATFORM` | optional | `"auto"` | `"linux-oe"`, `"android"`, or `"auto"` to detect. Selects the QAIRT ABI directory (`aarch64-oe-linux-gcc11.2` vs `aarch64-android`) and the `ADSP_LIBRARY_PATH` layout — see [Running on Android](#running-on-android). Only set it explicitly if the SDK you are pointing at is laid out for a different target than the one you are running on. |
| `TEXT_SLOTS` | required unless `VLM_SLOTS` is set | (unset) | One entry per text model to keep resident: `[{"model_root", "name", "device_id", "poll", "config_file"}, ...]`. Only `model_root` is required — `name` defaults to `slot<i>` and an unset `device_id` leaves the model on whichever core its own HTP config names, so a single-model server is `[{"model_root": "..."}]`. See [Multi Text Slots](#multi-text-slots), and note that a second slot does not always fit. `poll` overrides that model's `QnnHtp.poll` for this slot — see `POLL` below. `config_file` names the dialog config inside `model_root`, default `genie_config.json`: an export is free to call it after the model (`acme-7b-htp.json`) because genie-app takes the path on its command line, and pointing a slot at that file beats copying it. Both `poll` and `config_file` belong to the slot, so they survive a `/v1/models/switch`. A config with neither `TEXT_SLOTS` nor `VLM_SLOTS` is rejected at startup. |
| `POLL` | optional | (unset) | Default for every slot's `poll`: `true`/`false` overrides `dialog.engine.backend.QnnHtp.poll` in each model bundle, unset leaves each bundle as it is. **`false` is usually what you want** — polling costs ~260% CPU on SA8255P for latency indistinguishable from blocking (see [`QnnHtp.poll`](#qnnhtppoll-costs-260-cpu-and-buys-nothing-here)). A per-slot `poll` wins over this. Text slots only; VLM slots are unaffected. |
| `SLOT_LOAD_ORDER` | optional | `"vlm-first"` | Which slot kind is created first when both `TEXT_SLOTS` and `VLM_SLOTS` are set: `"vlm-first"` or `"text-first"`. See [Slot creation order](#slot-creation-order-slot_load_order) and [Loading Two Models at Once](#loading-two-models-at-once). Any other value is rejected at startup. |
| `VLM_SLOTS` | optional | (unset) | A separate, parallel multimodal (`GenieNode`/`GeniePipeline`) slot configuration alongside `TEXT_SLOTS`. `[{"name","device_id","model_root","spec","max_tokens"}, ...]`. Can be set independently of `TEXT_SLOTS` (see [VLM (Multimodal) Support](#vlm-multimodal-support)). `max_tokens` caps generation for that slot (default `1024`, `0` = uncapped) — see [Limiting generation length](#limiting-generation-length). |
| `PREFIX_CACHE_DIR` | optional | `"./prefix_cache"` | Directory for the prefix KV cache and HTP extension config copies (`.htp_ext_cache/`). A relative value is resolved against the server's working directory, so an absolute path is worth setting if the server may be started from anywhere but its own directory. |
| `MODELS_BASE_DIR` | optional | (unset) | Base directory every **relative** model path resolves against: `TEXT_SLOTS`/`VLM_SLOTS` `model_root` at startup, and `POST /v1/models/switch`'s `model_dir`. An **absolute** path ignores it and is used as given. Unset, a relative path is resolved against the server's working directory. Set this and a config can name each model by bare directory name. Note it is a base, not a sandbox — an absolute `model_dir` still loads from outside the tree. |
| `CHAT_TEMPLATE` | optional | (unset = auto-detect) | Pins the chat template to `"llama3"` / `"llama2"` / `"chatml"` / `"gemma"` / `"gemma4"`. Overrides for every slot. If unset, each slot auto-detects it from its model directory name (see [the relevant section](#chat-template-selection-rules)). |
| `TOOL_FORMAT` | optional | (unset) | Which tool-call dialect the models speak: `hermes` (JSON in `<tool_call>` tags — Qwen3 and everything else we have) or `gemma4` (`<\|tool_call>call:NAME{...}<tool_call\|>`, values in gemma4's own notation rather than JSON). Unset derives it per slot from that slot's chat template, which is what you want unless a bundle's name misleads the template detection. A dialect decides how declarations are rendered into the system turn, how a call is parsed out of the reply, and how an assistant turn's `tool_calls` are rendered back for the follow-up. |
| `DEFAULT_MAX_TOKENS` | optional | `0` (disabled) | Extra cap applied to `/v1/completions`/`/v1/chat/completions` requests that don't specify `max_tokens`/`max_completion_tokens`, on top of the model's own remaining context space (see `max_tokens` under [API Reference](./API.md)). `0` means no extra cap — bound only by context size, matching Qualcomm's own qai-appbuilder reference server. Set a positive value if a specific model/deployment is known to run away and you want a smaller safety margin (see [Troubleshooting](#troubleshooting)). Explicit client-provided `max_tokens` always overrides this. |
| `INFERENCE_TIMEOUT` | optional | `120` | Watchdog limit (seconds) for one `GenieDialog_query` call. Long generations on slow targets may need this raised. Also bounds `GET /v1/server/idle` and the sync path's overall wait (2x this value). |
| `HOST` / `PORT` | optional | `"0.0.0.0"` / `8080` | Listen address / port. The `--host` / `--port` CLI flags override these. |
| `GENIE_LIB_PATH` | optional | (unset) | Explicit path to `libGenie.so`. Default: `<QAIRT_SDK_ROOT>/lib/<abi>/libGenie.so` for the resolved `TARGET_PLATFORM`. |
| `GENIE_PROFILE` | optional | `false` | Binds a `GenieProfile` to every text slot; read the SDK's own TTFT / prefill / decode KPIs from `GET /v1/server/profile` (see [Profiling](#profiling-sdk-side-kpis)). Needs a restart to change. |
| `PROMPT_LOGPROBS` | optional | `false` | Enables prompt scoring (`echo`+`logprobs` teacher forcing, used by lm_eval loglikelihood tasks) at startup. Also toggleable at runtime via `POST /v1/server/prompt_logprobs` — see [Logprobs](#logprobs). |
| `PROMPT_LOGPROBS_MAX_TOKENS` | optional | `4096` | Reject prompt-scoring requests longer than this many tokens (each request runs its whole prompt at decode speed). |
| `TOOL_CALL_RECOVERY` | optional | `false` | Recover a tool call whose `<tool_call>` marker the model mangled or omitted, by matching the JSON's `name` against the tool names the request itself declared. `qwen3_4b_instruct_2507` w4a16 emits Cyrillic in place of the `<tool_call>` token on about half its calls and `qwen3_0_6b` omits the tags altogether, and in both cases the call body is correct — so with this off, the caller gets prose with `finish_reason: "stop"` while their code reads `message.tool_calls`. **Off by default anyway**: it conceals a defect in the bundle you are measuring, and it applies only to `/v1/chat/completions`, so the same model scores differently there than on `/v1/completions`, which has no recovery to apply. Turn it on when you want the application to work in spite of the bundle, and read the result as the application's rather than the model's. `qwen3_1_7b` marks its calls reliably and needs nothing. See [Function calling](API.md#function-calling-tools). |

Ready-made samples (single-slot / dual-NSP / text+VLM, with SA8775P model paths) are in [examples/config/](../examples/config/). Single-slot configuration example:

```json
{
  "QAIRT_SDK_ROOT": "/home/root/qairt/2.49.40.260810",
  "HEXAGON_VERSION": "v73",
  "MODELS_BASE_DIR": "/data/models",
  "TEXT_SLOTS": [{"model_root": "qwen3-4b-htp"}],
  "PREFIX_CACHE_DIR": "/data/prefix_cache"
}
```

See [Multi Text Slots](#multi-text-slots) for a 2-slot (e.g. SA8255P) configuration example.

### Where model paths resolve

A **relative** `model_root` resolves under `MODELS_BASE_DIR`; the example above loads `/data/models/qwen3-4b-htp`. An **absolute** one ignores the base and is used as given, which is how you load a bundle that lives outside it. With `MODELS_BASE_DIR` unset, a relative path resolves against the server's working directory — which depends on where the server was launched from, so set the base if you use relative paths at all.

`POST /v1/models/switch`'s `model_dir` follows the same rule, so `{"model_dir": "llama3-8b-htp"}` and a `model_root` of `"llama3-8b-htp"` mean the same directory. One bare name, one model, whether it is loaded at startup or switched in later.

The base is not a sandbox. An absolute `model_dir` still loads from outside it, deliberately — `MODELS_BASE_DIR` shortens paths, it does not restrict which directories a running server will open.

That was a decision, not an oversight: a bench instrument has to be able to point at a bundle someone just dropped in `/tmp`, and confining it would mean registering every model before it could be measured. The consequence is worth stating plainly, since there is no authentication either — **anyone who can reach the port can make the server open any path the process can read.** Run it on a network you control, as [the README says](../README.md#what-this-is-for).

### Relationship to genie_config.json

Each slot's `model_root`'s `genie_config.json` is passed as-is to the Genie SDK's `GenieDialogConfig_createFromJson` (with `device_id` set, only the HTP extension config copy is swapped in). The server only resolves and verifies the existence of the following paths, relative to the model directory (`slots.load_dialog_config`):

- `dialog.tokenizer.path`
- `dialog.engine.backend.extensions`
- `dialog.engine.model.binary.ctx-bins` (an array)
- `dialog.context.grammar.file` (if set — see [Grammar-Constrained Decoding](#grammar-constrained-decoding))

If any of these files is missing, the startup load calls `sys.exit(1)`; a load via `/v1/models/switch` returns an HTTP 500 instead (the server itself keeps running).

#### `QnnHtp.poll` costs ~260% CPU and buys nothing here

The bundles we have ship `"poll": true` in `dialog.engine.backend.QnnHtp`,
which makes the HTP backend busy-wait for the DSP instead of blocking. Measured
on SA8255P (QAIRT 2.49), 25 identical 64-token chat requests, one slot:

| model | `poll` | latency (median) | throughput | server process CPU |
|---|---|---|---|---|
| `qwen3_0_6b` | `true` | 0.961 s | 66.6 tok/s | **~263%** |
| `qwen3_0_6b` | `false` | 0.951 s | 67.3 tok/s | **~12%** |
| `qwen3_4b_instruct_2507` | `true` | 5.738 s | 10.6 tok/s | **~260%** |
| `qwen3_4b_instruct_2507` | `false` | 5.731 s | 10.6 tok/s | **~3-12%** |

Same latency to within noise, ~20× less CPU. Two slots at once are unaffected
too: the 1.31× figure in [Multi Text Slots](#multi-text-slots) is identical with
`poll: false`, which also rules out host CPU as that ceiling's cause.

**Set polling off unless you can measure a reason not to** — on a shared
device those three saturated cores are taken from everything else on the
system. Either edit the bundle, or let the server do it without touching
someone else's model directory:

```json
{
  "MODELS_BASE_DIR": "/home/root/models",
  "POLL": false,
  "TEXT_SLOTS": [
    {"name": "tool_call", "device_id": 0, "model_root": "qwen3_1_7b-genie-w4a16-qualcomm_sa8775p"},
    {"name": "chat", "device_id": 1, "model_root": "qwen3_4b_instruct_2507-genie-w4a16-qualcomm_sa8775p", "poll": true}
  ]
}
```

`POLL` sets the default for every slot and a slot's own `poll` overrides it;
leaving both unset changes nothing, so an existing deployment behaves exactly
as before. The override is a property of the *slot*, so it survives
`/v1/models/switch` onto a different model. VLM slots do not read it.

One caveat we saw once and could not reproduce: a single request right after
startup took 2× the median with `poll: false`; the following 26 requests were
all within 0.01 s of each other. If your workload is latency-critical at the
tail, measure it rather than trusting this table.


## Starting the Server

Install it first, or don't — both work:

```bash
pip install .[logprobs,vlm]      # or -e . while developing
```

The distribution is `open-genie-server`, the import package is `genie_server`,
and the install adds a `genie-server` command. Without an install, the
repository-root launcher puts `src/` on `sys.path` itself, so a plain checkout
runs as-is; so does a deployment where `genie_server/` was copied next to
`genie-server.py`, which is how the device is set up.

```bash
genie-server                     # if installed
python3 genie-server.py          # from a checkout, or on the device
# listens on 0.0.0.0:8080 by default
python3 genie-server.py --config /path/to/env_config.json --host 0.0.0.0 --port 8080
```

Or under uvicorn directly (the config path comes from `GENIE_SERVER_CONFIG`, default `./env_config.json`):

```bash
GENIE_SERVER_CONFIG=env_config.json uvicorn genie_server.asgi:app --host 0.0.0.0 --port 8080 --workers 1
```

**Always keep `--workers` at 1.** Each slot's `GenieDialog` handle is process-global state; splitting across multiple worker processes would have them fight over separate NPU contexts and break.

## Running on Android

The server runs unchanged on an Android target — the only code that had to
learn about the platform is which QAIRT ABI directory to load `libGenie.so`
from and how to build `ADSP_LIBRARY_PATH`. Set `TARGET_PLATFORM` (or leave it
`"auto"`) and the rest of the configuration is the same.

Verified on an SA8255P Android guest (Android 15, arm64-v8a, root shell) with
stock QAIRT 2.48 `aarch64-android` and a Qwen3-0.6B w4a16 bundle: the hardware
integration suite returns **22 pass / 3 fail / 14 skip**, the three failures
being logprobs requests, which the server rejects up front because numpy is not
installed (see [What is missing](#what-is-missing-on-android)).

### What the platform changes

| | `linux-oe` | `android` |
|---|---|---|
| QAIRT ABI directory | `lib/aarch64-oe-linux-gcc11.2` (glibc) | `lib/aarch64-android` (bionic) |
| `ADSP_LIBRARY_PATH` | SDK skels, `/usr/lib/rfsa/adsp`, one `/dsp/image/dsp/cdspN` per `device_id` in use | `/vendor/lib/rfsa/adsp` **first**, then the SDK skels. No `cdspN` entries |
| `LD_LIBRARY_PATH` | not set by the server | SDK ABI directory + `/vendor/lib64` |

A library built for one ABI will not load on the other, so a `libGenie.so` you
rebuilt for OE Linux cannot be used on Android and vice versa.

> [!IMPORTANT]
> **On an automotive target the DSP-side C++ runtime has to be staged by hand.**
> `libc++.so.1` and `libc++abi.so.1` must be copied into the SDK's
> `lib/hexagon-<ver>/unsigned/` directory. Without them every dialog fails at
> creation with `Failed to create device: 14001` and nothing else in the log
> says why. They live in the QNX primary VM; on a board that also runs a Linux
> guest they can be taken from that guest's `/dsp/image/dsp/cdsp0/`, which is
> the same file. This is Qualcomm's documented requirement for Android
> Automotive, not something this server introduces.

### Getting a Python runtime onto the device

Android ships no Python, and python.org's official Android release is an
**embedding** distribution: it contains `libpython3.x.so` and the standard
library but no interpreter executable. A ~5 kB launcher that calls
`Py_BytesMain()`, cross-compiled with the NDK against that `libpython`,
turns it into an ordinary command-line interpreter.

The guest typically has no network route, so collect wheels on a host and
install with `pip install --no-index --find-links=<dir>`. `ensurepip` works
offline and is enough to bootstrap pip itself.

### What is missing on Android

Three of this server's dependencies publish no Android wheel. `tokenizers` is
a core dependency rather than an extra, so pip will refuse the install
outright; take the core with `--no-deps` and install what is available
alongside it, and skip the `[logprobs]` and `[vlm]` extras entirely:

| | effect |
|---|---|
| `numpy` | **logprobs and prompt scoring return HTTP 400.** The server checks for numpy at startup and rejects those requests with a clear message rather than failing mid-generation. |
| `tokenizers` | Token counts fall back to `len(text.split())`. For a language that does not space its words this is drastically wrong, and it feeds the context check and the default `max_tokens`, not just reported usage — see [Token counting](#token-counting). |
| `pillow` | VLM slots unavailable (`numpy` is required for them too). |

`pydantic-core` also publishes no Android wheel. pip will silently resolve
around it by falling back to pydantic 1.x, which then fails at import because
current FastAPI needs pydantic v2 — so a resolution that pip calls successful
is not enough here. Cross-building `pydantic-core` for
`aarch64-linux-android` with the NDK and maturin produces a working wheel; the
same approach should work for `tokenizers`, which is also Rust.


## Chat Template Selection Rules

The template (`llama3` / `llama2` / `chatml` / `gemma` / `gemma4`) used by `format_chat_prompt` / `split_prompt_for_prefix_cache` is decided **not by the request's `model` field, but by the model actually loaded in the selected slot** (`slot.chat_template`).

Determination order (`_detect_template`, run once per slot at startup/switch time):

1. If `env_config.json`'s `CHAT_TEMPLATE` is set, use it (shared by every slot).
2. Otherwise, lowercase that slot's model directory name:
   - contains `"llama3"` or `"llama-3"` → `llama3`
   - contains `"llama2"` or `"mistral"` → `llama2`
   - contains `"gemma4"` or `"gemma-4"` → `gemma4` (checked first — see below)
   - contains `"gemma"` → `gemma` (Gemma 2/3 family — no system role; system text is prepended to the first user turn, assistant role is `model`)
   - otherwise → `chatml` (Qwen etc. fall here)

> **Note**: earlier versions decided this from the request's `model` field, but since `lm_eval` always sends the fixed string `"genie-local"` regardless of the actual model name, it always fell into the ChatML branch — a bug. It's now decided purely by the selected slot's state. The request's `model` field is used only for (1) selecting which slot to route to, and (2) echoing back in the response; it never selects the template.

Prompt format per template:

| Template | Format | Prefix cache splitting |
|---|---|---|
| `llama3` | `<\|begin_of_text\|><\|start_header_id\|>role<\|end_header_id\|>\n\ncontent<\|eot_id\|>...` | possible (only the system message is prefixed) |
| `llama2` | `<s>[INST] <<SYS>>...<</SYS>>...content [/INST] ... </s>` | **not possible** (system gets folded into `[INST]`) |
| `chatml` | `<\|im_start\|>role\ncontent<\|im_end\|>\n...` | possible |
| `gemma` | `<bos><start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n` | **not possible** (system folded into the first user turn) |
| `gemma4` | `<bos><\|turn>system\n...<turn\|>\n<\|turn>user\n...<turn\|>\n<\|turn>model\n` | possible (system is its own turn) |

> **Why gemma4 is a separate family.** Two differences from Gemma 2/3, both
> load-bearing:
>
> 1. **Turns are marked with `<\|turn>` and `<turn\|>`** (ids 105 and 106), not
>    `<start_of_turn>` / `<end_of_turn>` — and **the Gemma 2/3 spelling is not in
>    its vocabulary at all**, so writing it splits into roughly nine ordinary
>    tokens per marker. Measured on `gemma4-e2b-it`: the same question costs 39
>    prompt tokens under the `gemma` template against 22 under `gemma4`, and the
>    answers get worse — one test prompt degenerated into an unterminated list
>    under the old spelling and was answered normally under the new one.
> 2. **`system` is its own turn**, not folded into the first user turn. That is
>    what Google's own template does, and it is why gemma4 can split a cacheable
>    prefix where Gemma 2/3 cannot.
>
> These bundles also need `106` in `genie_config.json`'s `eos-token` array, or
> generation runs past the end of the model's turn and `<turn\|>` leaks into the
> content. `eos-token` is the bundle's to set, not the server's — if you see
> repeated `<turn\|>` in replies, that array is the place to look.
>
> **gemma4 has more native markers than this server currently uses.** Tool
> calls are `<\|tool_call>call:NAME{...}<tool_call\|>`, declarations
> `<\|tool>declaration:NAME{...}<tool\|>`, responses
> `<\|tool_response>response:NAME{...}<tool_response\|>`, string arguments
> `key:<\|"\|>value<\|"\|>`, and reasoning `<\|channel>thought...<channel\|>`.
> This server still declares and parses tools in the Hermes `<tool_call>` JSON
> form, so **tool calling against a gemma4 model is not yet wired up** — the
> model answers in its own markers and they arrive as content.

## Prefix KV Cache

When `messages` includes a `system` role and the template supports splitting (llama3/chatml/gemma4), the system prompt portion is saved and restored as a separate KV cache entry.

- Cache key: `sha256(f"{slot.name}|{slot.active_model_id}|{slot.active_lora_adapter}\x1f{prefix_prompt}")[:16]`
- **Namespaced by slot/model/LoRA**, so switching state via `/v1/models/switch` or `/v1/lora/apply` never accidentally restores a KV cache saved for a different slot/model/LoRA (the key simply changes, so it naturally misses).
- Storage format: the file (or directory) written by `GenieDialog_save`/`GenieDialog_restore` is managed as `PREFIX_CACHE_DIR/prefix_<key>.geniestate` (a single directory shared by every slot, but the keys never collide since they're namespaced).

Flow:

```
MISS: GenieDialog_reset → query(full_prompt, SENTENCE_COMPLETE, cb)   (nothing is cached automatically)
HIT:  GenieDialog_reset → GenieDialog_restore → query(remaining, SENTENCE_END, cb)
Warm-up (POST /v1/prefix/warmup only): GenieDialog_reset → query(prefix, SENTENCE_BEGIN, noop) → GenieDialog_save
```

A normal `/v1/chat/completions`/`/v1/completions` MISS does **not** populate the cache — it just runs the full prompt as a single `SENTENCE_COMPLETE` query (the shared generation path in `engine.py`), identical to an uncacheable request. Only an explicit `POST /v1/prefix/warmup` call runs the two-step `SENTENCE_BEGIN` + `GenieDialog_save` priming shown above; after that, later requests whose prefix matches will `HIT`.

### What it costs, and when it pays

The SDK does not profile `GenieDialog_save`/`_restore` at all — there is no
such profile event type, and `apply-engine-state-time` covers a different path
(`qualla/dialog.cpp:2653`) — so the server times both calls itself and reports
them under `host_measured` in
[`GET /v1/server/profile`](./API.md#get-v1serverprofile), kept apart from the
SDK's own numbers.

Measured on SA8255P with `qwen3_4b_instruct_2507` (median of 3, `max_tokens: 4`):

| system prompt | TTFT on a miss | TTFT on a hit | saved | restore | save (once, at warmup) |
|---|---|---|---|---|---|
| 36 tokens | 183.5 ms | 181.8 ms | 1.7 ms | 2.7 ms | 20 ms |
| 72 tokens | 185.1 ms | 184.9 ms | 0.2 ms | 3.8 ms | 47 ms |
| 135 tokens | 360.4 ms | 183.7 ms | **176.7 ms** | 5.4 ms | 95 ms |
| 270 tokens | 537.3 ms | 184.1 ms | **353.2 ms** | 9.0 ms | 198 ms |
| 477 tokens | 712.2 ms | 183.5 ms | **528.7 ms** | 14.7 ms | 353 ms |

**Restoring is nearly free** — 3-15 ms, growing slowly with the state size.
What decides whether caching pays is not the restore cost but the AR-128
prefill batch (see [Profiling](#profiling-sdk-side-kpis)): a hit removes whole
batches of prefill, so

> **The prefix cache pays off once the system prompt is long enough to occupy
> a prefill batch of its own — about 128 tokens on these models.** Below that
> the system turn shares a batch with the rest of the prompt, a hit saves
> nothing measurable, and the few ms of restore make it a small net loss.

Note the asymmetry: saving costs 5-25x more than restoring (20-353 ms), but it
happens once per prefix, in warmup, off the request path.

## Grammar-Constrained Decoding

The Genie SDK (qualla) supports grammar-constrained decoding (the XGrammar backend) on `basic` dialogs (the type this server uses). Output can be constrained by JSON Schema, regex, or EBNF, with invalid tokens excluded via logit masking (including jump-forward acceleration).

**Important limitation: this is fixed per model/slot, and cannot be switched per request.** The Genie SDK's public C API (`GenieDialog.h`) has no function to set or change grammar at runtime at all (there's no generic setter like `GenieDialog_setValue` either) — grammar is read from `genie_config.json` exactly once, when `GenieDialog_create()` builds the Dialog internally. Changing it requires rebuilding the Dialog from a `GenieDialogConfig`, which costs the same as `/v1/models/switch` (a full model reload).

This can't be used the way OpenAI's `response_format` works, with a different schema passed per request. Instead, set up a dedicated slot for "a model that always outputs following JSON Schema X."

### Configuration

Add a `grammar` block to `genie_config.json`'s `dialog.context`:

```json
{
  "dialog": {
    "context": {
      "size": 4096,
      "grammar": {
        "backend": "xgrammar",
        "type": "json-schema",
        "file": "grammar_schema.json"
      }
    }
  }
}
```

| Field | Value | Notes |
|---|---|---|
| `backend` | must be `"xgrammar"` | The SDK only implements XGrammar. Any other value makes `GenieDialog_create` fail (the server fails to start at startup time, or `/v1/models/switch` returns an HTTP 500). |
| `type` | `"json-schema"` (default) / `"regex"` / `"ebnf"` | The kind of schema. |
| `file` | path relative to the model directory (or absolute) | **A plain text file containing the schema/pattern/grammar definition itself** (its content is used, not its filename or a header). Resolved and verified like every other asset path, by `slots.load_dialog_config`. |

**Do not set** the `tokenizer-json` field. It's deprecated/ignored by the SDK (the vocabulary is now derived automatically from the already-loaded live tokenizer) — setting it has no effect other than a warning log.

### Your library must have XGrammar compiled in

The **stock** `libGenie.so` under `lib/aarch64-oe-linux-gcc11.2/` has XGrammar statically linked, so the configuration above works as shipped.

> [!IMPORTANT]
> A `libGenie.so` **rebuilt from the SDK's own sources has no grammar backend at all** — including the fixed build that [QAIRT Version Issues](./QAIRT_VERSIONS.md#option-1--run-a-fixed-libgenieso-recommended-unless-you-need-grammar) recommends as the real fix for the slot-wedge defect. With such a library, loading a model whose `genie_config.json` has a `grammar` block fails with `GenieDialog_create failed: -1` and `"Grammar backend configured but qualla was built without ENABLE_GRAMMAR"` in the log — at startup, or as an HTTP 500 from `/v1/models/switch`. [D3](./QAIRT_VERSIONS.md#d3--a-library-you-rebuild-has-no-grammar) explains why the rebuild flag that message suggests does not help, and gives a one-line check for your own library.

### Verified behaviour, and one known defect

On a stock 2.49.40.260810 library (SA8255P, `qwen3_0_6b` w4a16) all three kinds constrain the output correctly, including under `stream: true` and alongside `logprobs`.

One SDK defect affects all of them: **the response ends with the model's end-of-sequence token as literal text** (`<|im_end|>` for a ChatML model). The constrained part itself is correct — the JSON object is complete and schema-valid, the regex match is exact — but that trailing marker means a JSON-Schema-constrained response does not parse with `json.loads()`, and a regex-constrained one does not match its own pattern. **The server does not strip it**, so a client using this feature has to remove the trailing special-token string itself. Reported to Qualcomm with a reproducer and a proposed fix.

## Profiling (SDK-side KPIs)

`GENIE_PROFILE: true` binds a `GenieProfile` handle to every text slot, and
`GET /v1/server/profile` returns what the SDK itself measured for the most
recent query on that slot — no host-side timing involved:

```bash
curl "$base_url/v1/server/profile?slot=chat"
```

```json
{
  "slot": "chat",
  "model": "qwen3_4b_instruct_2507-genie-w4a16-qualcomm_sa8775p",
  "summary": {
    "prompt_tokens": 25.0, "ttft_ms": 184.599, "prefill_tokens_per_s": 135.4,
    "generated_tokens": 32.0, "decode_tokens_per_s": 11.01, "generation_ms": 2815.63
  },
  "profile": { "header": {...}, "components": [ ... ] }
}
```

`summary` is the handful of dialog KPIs flattened into familiar units;
`profile` is the SDK's JSON verbatim (it also carries `GenieDialog_create`
init time, and `apply-engine-state-time` / `lora-adapter-switch-time` when
those paths run).

**This is not part of the OpenAI surface, by design.** Chat and completion
responses are unchanged whether profiling is on or off — clients validate
those shapes, and this server keeps everything non-OpenAI under
`/v1/server/`. A profiling client asks for it explicitly.

| | |
|---|---|
| Enable | `"GENIE_PROFILE": true` in env_config.json |
| Runtime toggle | **not possible** — the profiler binds to the dialog *config*, before `GenieDialog_create` (`GenieDialog.h:216`), so it needs a restart. `GET` returns HTTP 409 while disabled |
| Scope | text slots only; survives `/v1/models/switch` (the handle belongs to the slot) |
| Retention | the SDK keeps the **most recent** query's values (`last_usec`), not a history — poll after each request if you need a series |
| Cost | none measurable: 3.017 s per request with it on against 3.014-3.015 s without |

Measured against host-side timing on the same request (4B, 25-token prompt,
32 generated): the SDK reports TTFT 184.6 ms and 11.01 tok/s decode, and the
client sees 3.017 s total against 0.185 + 2.816 = 3.001 s of SDK time — about
16 ms of HTTP, templating and ctypes overhead per request.

## VLM (Multimodal) Support

Image-input models like Qwen3-VL are supported via `genie_server/genie_node.py` (the `GenieNode`/`GeniePipeline` ctypes bindings — the generic plumbing layer) and `genie_server/vlm_specs.py` (model-specific preprocessing, node topology, and prompt templates). This is **a completely separate subsystem from the text-only `Slot`/`GenieDialog` path**, and has zero effect on any existing text-only endpoint's behavior.

Without `numpy`/`Pillow` installed, this is automatically disabled (a warning is logged at startup, `VLM_SLOTS` is ignored) and the server starts as text-only.

### Configuration

Add a `VLM_SLOTS` key alongside `TEXT_SLOTS` in `env_config.json`:

```json
{
  "VLM_SLOTS": [
    {"name": "vision", "device_id": 0, "model_root": "/models/qwen3-vl", "spec": "qwen3_vl"}
  ]
}
```

Under `model_root`, in addition to the file set `genie-app-script.txt` expects (`img-enc-htp.json`, `text-encoder.json`, `text-generator.json`, `vision_encoder.bin`, each `ctx-bins`, `embedding_weights.raw`, `tokenizer.json`), you also need
`sample_inputs/{position_ids_cos,position_ids_sin,full_attention_mask,window_attention_mask}.raw`
for positional encoding/attention masks (loaded once at startup assuming a fixed resolution, then reused for every subsequent request — see `VLMSpec.static_tensor_files` in `genie_server/vlm_specs.py`). When `device_id` is given, each node's HTP extension config is rewritten for NSP pinning using the same mechanism as `Slot` (`slots.pin_htp_device`).

`spec` is a key into `vlm_specs.VLM_SPECS` (the default, and currently only implemented, value is `"qwen3_vl"`). To support a different model or a different resolution export, register a new `VLMSpec` in `genie_server/vlm_specs.py` (the same idea as GenieX's `core/` vs `models/*.h` split — no other file needs to change).

### Slot creation order (`SLOT_LOAD_ORDER`)

Only relevant when `TEXT_SLOTS` and `VLM_SLOTS` are both non-empty. QAIRT 2.49
makes the order consequential in both directions, so it is a config key rather
than a fixed choice:

| Value | Behavior |
|---|---|
| `"vlm-first"` (default) | VLM slots are created, then text slots. |
| `"text-first"` | Text slots are loaded, then VLM slots, with a validator-flag reset in between. |

Why the reset is needed: creating a `GenieDialog` for a text model whose backend
config uses `pos-id-dim`/`rope-theta` sets two **process-global** flags inside
libGenie. `pipeline::TextGenerator` calls `Dialog::validateDialogConfig()`
directly instead of going through `GenieDialogConfig_createFromJson`, so it never
clears them, and every later VLM text-generator node is rejected with
`Specify one config from pos-id-dim and positional-encoding`. With
`"text-first"` the server calls `GenieLib.reset_dialog_validator_flags()` between
the two phases, which clears the flags via the only API that resets them.

Why you might want `"text-first"`: for **two text models** on a dual-NSP target,
whether they co-exist depends on which one is created **first** — creating a
large model first makes the second `GenieDialog_create` fail with `err 1002`
(`QNN_COMMON_ERROR_MEM_ALLOC`) no matter which `device_id` each one is pinned
to. Loading the smaller text model first is what makes that co-residency
possible at all. How large "too large" is also depends on how the first model was
exported: a single-context-length bundle leaves more room than a
multi-context-length one of the same model. For a **text + VLM** pair the order
turned out not to matter — see the note below — so `"text-first"` there is just
a safe default.

> **Measured on SA8255P (QAIRT 2.49):** the ~4B `qwen3_vl_4b_instruct` does fit
> alongside a text model, **but only when the text bundle is exported at a
> single context length and the two slots are on different `device_id`.** Under
> those conditions *both* orders work, so `SLOT_LOAD_ORDER` is not what decides
> it — the bundle shape is. With a multi-context-length text bundle neither
> order fits, which is what earlier versions of this note reported. Two text
> models also co-exist: 0.6B + 4B, and — with single-context-length bundles —
> 1.7B + 1.7B and 1.7B + 0.6B. Read
> [Loading Two Models at Once](#loading-two-models-at-once) first, which
> documents what is and is not known about the limit.

### Request format

Sending an OpenAI-style `content` array (including an `image_url` part) to `POST /v1/chat/completions` automatically routes into the VLM path (the `model` field selects the target slot using the same rule as text slots — `_select_vlm_slot`; an explicit `"slot": "<name>"` in the body overrides it, same as `/v1/completions`/`/v1/chat/completions`'s text-slot behavior — see [Multi Text Slots](#multi-text-slots) — needed when two `VLM_SLOTS` entries load the same model directory, via `_select_vlm_slot_for_request`; unknown name → `404`):

```json
{
  "model": "vision",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
      {"type": "text", "text": "What is in this image?"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<...>"}}
    ]}
  ],
  "stream": true
}
```

A single message can include multiple `image_url` parts (`vlm_specs.qwen3vl_build_prompt_segments` assembles them into an interleaved text/image segment list, including Qwen3-VL's `<|vision_start|>`/`<|vision_end|>` markers).

### Limiting generation length

A VLM request's own `max_tokens` **cannot be honoured**: `GenieNode.h`/`GeniePipeline.h`
have no per-request token limit and no abort, and the text callback's return value is
discarded, so nothing in the pipeline can be told to stop mid-generation. The one limit
the SDK does expose is the text-generator node's `max-num-tokens`, which Genie reads
**once, when the node is created**. The server fills that in from `VLM_SLOTS[].max_tokens`:

```json
{
  "VLM_SLOTS": [
    {"name": "vision", "device_id": 0, "model_root": "/models/qwen3-vl",
     "spec": "qwen3_vl", "max_tokens": 1024}
  ]
}
```

| Value | Effect |
|---|---|
| omitted | **1024** (the default) |
| a positive integer | generation stops after that many tokens |
| `0` | **no cap** — generation runs until the model emits EOS or the context fills up |

Because the value is baked into the node at creation time, changing it means editing
`env_config.json` and **restarting the server**. There is no endpoint to change it at
runtime, and it applies to every request on that slot. Confirm the value that took
effect in the startup log:

```
VLM slot 'vision' ready: model=qwen3-vl device_id=0 spec=qwen3_vl max-num-tokens=1024
```

An uncapped slot logs `max-num-tokens=(uncapped)` instead.

#### What an uncapped slot actually does

Nothing stops a prompt that the model does not answer briefly ("describe this in as much
detail as you can", and so on) from generating until the context window is exhausted.
Measured on Qwen3-VL 4B (4096-token context, ~14 tok/s), same prompt and image:

| `max_tokens` | wall clock | tokens generated |
|---|---|---|
| `1024` | 72 s | 1024 |
| `0` (uncapped) | **318 s** | ~3800 (context exhausted) |

For those 318 seconds the request holds that VLM slot's lock, and **a client that
disconnects cannot stop it** — every other request to the slot waits. The output is also
cut off mid-sentence when the context runs out. Raising the cap costs proportionally more
time; removing it entirely bounds a single request only by the context window.

Set `max_tokens` to `0` when you deliberately want the longest answer the model can
produce and nothing else is competing for the slot. Otherwise keep a cap.

#### `finish_reason`

Whichever limit stopped the generation, the response reports `finish_reason` as
`"length"`; a natural EOS stop reports `"stop"`:

| How generation ended | `finish_reason` |
|---|---|
| model emitted EOS | `stop` |
| hit `VLM_SLOTS[].max_tokens` | `length` |
| context window filled (uncapped, or a cap larger than the context) | `length` |

A context-exhausted response still returns `200` with the text produced so far — the
truncation is reported through `finish_reason`, not as an error. Note that
`usage.completion_tokens` reflects what was actually generated, which for a VLM request
may exceed the `max_tokens` the client asked for, since that value is ignored.

### Out of scope for V1 (known limitations)

`GenieNode.h`/`GeniePipeline.h` **don't have** the following APIs that `GenieDialog` has, so the VLM path doesn't support:

- **Per-request `max_tokens`/`stop`**: no way to enforce these at the SDK level. They're accepted but ignored (there's no `GenieDialog_setMaxNumTokens`/`setStopSequence` equivalent). Generation length is bounded per slot instead — see [Limiting generation length](#limiting-generation-length).
- **Aborting on client disconnect**: `GeniePipeline_execute()` is a blocking call — once started, it can't be stopped before it finishes naturally (there's no `GenieDialog_signal` equivalent). The server keeps running inference to completion even after a disconnect, and that VLM slot's lock is held until it finishes naturally.
- **Multi-turn conversation**: always single-shot (`pipeline.reset()` is called on every request). No conversation history is kept.
- **Hot-swapping via `/v1/models/switch`, LoRA, prefix KV cache, grammar constraints**: these are either `GenieDialog`-only APIs, or simply not implemented yet in V1.
- **Fetching remote-URL images**: only `data:` (base64) URLs are supported. `http(s)://` etc. are explicitly rejected with a 400 (so an embedded/automotive server never makes unexpected outbound network calls).

## API Reference

Every endpoint, grouped by purpose, lives in **[API.md](./API.md)** — request and response shapes, status codes and the error envelope.

## Token counting

Token counts come from the model's own tokenizer, loaded from the path in `genie_config.json` with the `tokenizers` package. They feed three things: the `usage` numbers in a response, the context-overflow check that rejects an oversized prompt with a 400, and the default `max_tokens` when a request does not specify one (context size minus prompt length).

**Without `tokenizers` installed, all three fall back to `len(text.split())`.** That is a reasonable approximation for text that separates words with spaces and a useless one for text that does not. Measured against the Qwen3 tokenizer:

| Sample | Real tokens | `split()` | Off by |
|---|---:|---:|---:|
| English paragraph | 43 | 33 | 1.3x |
| Japanese paragraph | 55 | **1** | 55x |
| Japanese, four paragraphs | 216 | **1** | 216x |
| Chinese sentence | 24 | **1** | 24x |
| Thai sentence | 35 | 3 | 12x |
| Korean sentence | 30 | 8 | 3.8x |
| JSON with no spaces | 17 | **1** | 17x |

A language that does not put spaces between words counts as roughly **one token per sentence**, and a compact JSON tool-call payload counts as one token however long it is.

The consequences go beyond wrong `usage` numbers:

- **The context-overflow check stops working.** A prompt that genuinely exceeds the context window counts as a handful of tokens, passes the check, and reaches the SDK — which is the failure this check exists to prevent, since the model then has no room to generate and returns an empty reply with `finish_reason: "length"` and no explanation.
- **The default `max_tokens` becomes almost the whole context.** It is computed as context size minus prompt tokens, so undercounting the prompt leaves a budget the prompt has already spent.
- **Anything downstream that meters tokens is wrong**, including `lm_eval` accounting and any per-request cost or quota tracking a caller layers on top.

**VLM slots count the same way text slots do.** The composable pipeline gives the host no tokenizer object, but the text-generator node's config names the same `tokenizer.json` the node itself tokenizes with, so the slot loads that file directly and `usage` is on the same basis as a text slot's — with the same `tokenizers`-not-installed fallback to whitespace. Two differences remain: **image tokens are not counted** (the image never becomes text on the host — the image-encoder node emits embeddings straight into the pipeline — so `prompt_tokens` covers the prompt text only), and VLM slots still do not run the context-overflow check; generation there is bounded by the slot's own `max_tokens` (see [VLM (Multimodal) Support](#vlm-multimodal-support)).

So `tokenizers` is optional only in the sense that the server starts without it. Install it unless you are certain every prompt is spaced Latin text. The server logs a warning at startup when it is missing, and again if the tokenizer file named by `genie_config.json` cannot be loaded — that second case falls back the same way even with the package installed, so check the startup log for `HF tokenizer loaded:` rather than assuming.

## Logprobs

The Genie SDK exposes no logits through `GenieDialog` — logprobs are implemented via the SDK's **custom sampler** hook (`GenieSampler_registerUserDataCallback` + sampler config `{"type": "custom"}`), which hands the server the full dequantized float32 logits vector at every generation step and lets it choose the emitted token (`genie_server/logprobs.py`).

**Generated-token logprobs** (always available): `logprobs` on `/v1/completions` (int) or `logprobs`/`top_logprobs` on `/v1/chat/completions` (bool/int) record each generated token's logprob and top-N alternatives. Sampling moves into the server for these requests (greedy / temperature / top-k / top-p over the same logits — `temperature=0` matches the SDK's greedy exactly); the SDK's basic sampler with the model's defaults is restored afterwards. Overhead is one log-softmax over the vocab per token (~1-2 ms) versus tens of ms of NPU decode — and exactly zero for requests that don't ask for logprobs. Requires `numpy` and the model tokenizer. Not supported with `stream: true`, on VLM slots, or with grammar-constrained models' masked-out semantics in mind (logprobs are post-grammar-mask).

**Prompt scoring** (`echo: true` + `logprobs`, the shape lm_eval's loglikelihood tasks send): the server prefills only the first prompt token, then **teacher-forces** every following prompt token through the decode loop, recording P(token_i | tokens_<i) at each position — exact loglikelihood and `is_greedy`, with `token_logprobs[0] = null` (the first token's probability is undefined, same as OpenAI's `echo`). Token-id prompts (what lm_eval sends with `tokenizer_backend=huggingface`) are scored with exact id alignment. **Each request runs its whole prompt at decode speed** (a 500-token document at 20 tok/s ≈ 25 s), so it is gated:

- disabled by default; enable with `POST /v1/server/prompt_logprobs {"enabled": true}` (or `PROMPT_LOGPROBS: true` in env_config.json), check with `GET /v1/server/prompt_logprobs`;
- prompts longer than `PROMPT_LOGPROBS_MAX_TOKENS` (default 4096) are rejected;
- the watchdog auto-scales with prompt length; `GET /v1/server/status` shows a `scoring prompt` phase with progress;
- `max_tokens` must be 0, 1 or omitted. 0 scores the prompt and nothing else; **1 is
  lm_eval's loglikelihood shape** — it appends one genuinely generated token, which
  lm_eval then slices back off (`token_logprobs[ctxlen:-1]`). Anything larger is a 400.

See [examples/lm_eval](../examples/lm_eval) for a runner script, the install
incantation lm_eval needs, measured timings, and how to compare a board run with
the same model in fp32.

Notes: a forced token that happens to be EOS ends scoring early (rare). Prompt scoring applies to basic dialog configs (not spec-dec/lookahead variants). For what the returned numbers mean against a GPU reference, see [Reading scored logprobs](#reading-scored-logprobs).

### Reading scored logprobs

The scoring path itself was verified against HF transformers on SA8255P
(`qwen3_0_6b`, QAIRT 2.49) — tokenization matches exactly, there is no
off-by-one in the teacher-forced prefix, and the result is bit-identical
across `temperature` 0 / 1 / 2 and `top_p`, i.e. it reports the model's
distribution and not the sampling one.

**The values themselves will not match an fp32 reference, and the gap is not
small.** Measured on that board against `Qwen/Qwen3-0.6B` in fp32:

| | |
|---|---|
| Correlation, per-token logprobs | 0.84 |
| Mean absolute difference | **1.06 nats** (67 token positions) |
| Max absolute difference | 6.76 nats |
| Positions agreeing within 0.5 nats | 51% |
| **Top-1 next-token agreement** | **64%** |

That last row is the one to internalize: the deployed model is a **w4a16**
export, and it does not even pick the same most-likely token as its fp32
original at roughly a third of positions. The logprobs are a faithful readout
of the quantized model — they are not, and cannot be, a readout of the
original one.

**How to use the numbers:**

- **Compare a board against itself.** Quantization/export changes, KV-cache
  settings, prompt formatting — all valid A/B comparisons on one target.
- **Do not compare a board score against a published fp32 leaderboard number.**
  Differences of a few points say nothing about the server.
- **Prefer `acc` over `acc_norm`.** On an 8-item hellaswag sample the summed
  loglikelihood picked the same ending as fp32 HF on **8/8** items, while the
  length-normalized variant diverged on 2/8: dividing by token count amplifies
  per-token deviation on short continuations. Report both if you report either,
  and treat `acc_norm` as the noisier one.
- **Rank-based tasks survive, margins do not.** Multiple-choice ranking held up
  in this sample, but a task whose items hinge on sub-nat differences between
  candidates will be dominated by quantization noise.
- **A quantized model can legitimately score higher.** In the same sample the
  board's `acc_norm` was 2/8 against HF's 1/8. That is noise, not an
  improvement — do not read small wins as real.

What this verification did *not* establish: that the residual gap is entirely
quantization. Proving that needs an fp32 run of the same exported graph, which
the toolchain does not provide; the evidence is circumstantial (the deviation
shows no shift, sign, or scale structure that would point at the server) but
consistent.

## Switching Models and LoRA

A typical operational sequence (2-slot configuration, swapping out only the `chat` side):

```bash
# 1. Check current state (every slot)
curl $base_url/v1/server/status

# 2. Wait for the target slot's inference to finish (optional — /v1/models/switch itself
#    also blocks on that slot, so this isn't strictly required)
curl "$base_url/v1/server/idle?slot=chat"

# 3. Switch the model on the chat slot (tool_call keeps running without interruption)
curl -X POST $base_url/v1/models/switch -d '{"slot": "chat", "model_dir": "llama3-8b-htp"}'

# 4. Apply a LoRA to that slot
curl -X POST $base_url/v1/lora/apply -d '{"model": "llama3-8b-htp", "engine": "primary", "lora_adapter_name": "finetune-v2"}'

# 5. Verify it applied
curl "$base_url/v1/lora/current?model=llama3-8b-htp"
```

Every one of these operations acquires the target slot's own lock, so it's safely serialized against inference requests to that slot while **never affecting other slots' inference at all**. After switching, that slot's `chat_template`, `active_model_id`, and prefix cache namespace all switch over to the new model automatically.

## Open WebUI

Open WebUI works against this server with no server-side configuration, but
**what it sends changes between its releases**, and one of those changes can
make replies come back empty on a small-context model. The versions below are
data points, not a compatibility matrix — a release we have not tried may need
the same step, a different one, or none.

| Version | When | Result |
|---|---|---|
| v0.6.18 | 2026-08-22 | Works as shipped. Model list, chat and streaming all fine; **zero 4xx/5xx** on the board across the session |
| 0.11.0 | 2026-08-22 | Works **after unchecking the built-in tools** (below) |

```bash
docker run -d --name open-webui -p 3001:8080 \
  -e WEBUI_AUTH=False \
  -e WEBUI_SECRET_KEY=<any fixed string> \
  -e OPENAI_API_BASE_URL=http://<board>:8080/v1 \
  -e OPENAI_API_KEY=dummy \
  -e ENABLE_OLLAMA_API=false \
  -e ENABLE_EVALUATION_ARENA_MODELS=false \
  -v open-webui:/app/backend/data \
  ghcr.io/open-webui/open-webui:<the tag you intend to run>
```

### Empty replies: check the built-in tools first

Open WebUI ships built-in tools — knowledge bases, memories, notes, tasks,
automations, calendar — and **some versions send all of them as a `tools`
block on every chat request**. Rendered into the prompt that came to **~5,800
tokens** in 0.11.0, more than the 4,096-token context of the Qwen3 exports, so
the model has no room left to generate and the reply is empty.

**Unchecking the built-in tools in Open WebUI's tools menu fixes it** — that is
what we did on 0.11.0. If a newer release still sends something large by
default, the same check is where to start; if it sends something else that does
not fit, the size is what matters rather than the feature's name. The other
ways out are a model with a larger context, or a version that does not send
them (v0.6.18 does not).

The server does not hide this behind an empty `200`. A prompt that does not fit
gets OpenAI's `400 context_length_exceeded`, naming both numbers, so you can
tell "the client sent too much" from "the model said nothing":

```json
{"error": {"message": "This model's maximum context length is 4096 tokens. However, your messages resulted in 5790 tokens. …",
           "type": "invalid_request_error", "param": "messages", "code": "context_length_exceeded"}}
```

### Two more things a version can change under you

Both of these were 0.11.0 behaviours that v0.6.18 did not have, which is the
point: expect the next release to differ again.

- **The model list is validated before anything is posted.** 0.11.0 refuses to
  send to a model whose `/v1/models` entry lacks the legacy `root` / `parent` /
  `permission` fields — the model appears in the picker and nothing happens. We
  return those fields now. A version that wants some *other* field would look
  identical from the outside, so if the picker shows a model and posting does
  nothing, compare a `GET /v1/models` response against what that version
  expects.
- **Streamed replies may not come back through the HTTP response.** 0.11.0
  renders them through its own websocket channel. That is invisible to this
  server, but it changes where you look when debugging the UI rather than the
  API.

### Settings worth getting right up front

Four Open WebUI settings cost us time, and none of them are version-specific:

| Setting | Why |
|---|---|
| `WEBUI_SECRET_KEY` | Unset means a **new JWT signing key on every container start**, so every existing session breaks. Symptom: a sign-in/sign-out loop, `No token found in localStorage` in the browser console. |
| `OPENAI_API_BASE_URL` | Only **seeds the database on first run**. Changing the env var later does nothing — the stored value wins. Fix it in Admin → Connections, or start from a fresh volume. |
| `ENABLE_EVALUATION_ARENA_MODELS=false` | Otherwise `arena-model` appears in the model list; selecting it with no arena models configured makes the frontend fail silently. |
| Task model | Title/tag/follow-up generation runs on a "task model" that defaults to empty, which fails with `Model '' was not found`. Point it at a real model in Admin → Interface, or turn those features off. |

**Turn the background tasks off for a slow target.** Title, tag and follow-up
generation each fire an extra chat completion, and a slot processes requests
one at a time — on a 4B at ~11 tok/s that is several seconds of extra work per
message.

What the server actually sees, for capacity planning: Open WebUI polls
`GET /v1/models` frequently (66 times against 34 chat completions in our
session — it is a cheap in-memory response) and sends the whole conversation
each turn, so TTFT grows with the history (184 ms → 358 ms → 535 ms over three
turns as the prompt crossed AR-128 batch boundaries). It always includes a
system message, so [prefix caching](#prefix-kv-cache) applies — warm it if
your deployment uses a long system prompt.

## Integration Testing

[tests/integration/](../tests/integration/) contains a host-side runner that exercises every feature area of a live device over the REST API (health, completions/chat/streaming, stop sequences, prefix cache, tools, logprobs, prompt scoring, performance policy, hot-swap, LoRA, VLM) and writes a Markdown + JSON report — including a dedicated incident section if the server dies mid-run. See its README for usage; `--base-url` overrides the config's target address.

## lm_eval Integration

```bash
# Generation-based tasks (gsm8k, humaneval, *_generative variants, ...)
lm_eval --model local-completions \
  --model_args model=genie-local,base_url=http://192.168.1.2:8080/v1,\
tokenizer_backend=huggingface,tokenizer=<hf_model>,max_tokens=512,num_concurrent=1 \
  --tasks gsm8k --batch_size 1

# Chat tasks
lm_eval --model local-chat-completions \
  --model_args model=genie-local,base_url=http://192.168.1.2:8080/v1,\
tokenizer_backend=huggingface,tokenizer=<hf_model>,max_tokens=512,num_concurrent=1 \
  --tasks mmlu_generative --apply_chat_template --batch_size 1
```

- **Loglikelihood-based tasks (hellaswag, arc_easy, plain mmlu, ...) work via prompt scoring**, which must be enabled first (`POST /v1/server/prompt_logprobs {"enabled": true}` — see [Logprobs](#logprobs)). Each document is scored at decode speed (~seconds per request), so use `--limit` or short-context tasks for practical runtimes; generation-based tasks remain the fast path. With prompt scoring disabled, `echo`+`logprobs` requests get a clear `400` naming the enable switch.
- `tokenizer_backend=huggingface` makes `lm_eval` send tokenized (token-id) prompts; the server decodes them with the slot's own tokenizer, so make sure `dialog.tokenizer.path` in `genie_config.json` matches the HF tokenizer you pass to `lm_eval`.

- `num_concurrent=1` is recommended. In a single-slot configuration the whole server processes serially, so raising concurrency doesn't increase throughput. If you want to **benchmark both slots of a 2-slot configuration at the same time**, run two `lm_eval` processes in parallel, one pinned to `model=genie-local` (the primary slot) and the other explicitly pinned to `model=<the second slot's active_model_id>` (`num_concurrent>1` within a single process doesn't load-balance across slots).

## BFCL (function-calling benchmark)

[BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html)'s open-source handlers
speak `/v1/completions`, so it runs against this server unchanged. See
[examples/bfcl](../examples/bfcl) for a runner script, the install notes, and
per-category timings.

**BFCL bypasses this server's chat template and tool parsing.** It builds the
Hermes prompt itself from the HF tokenizer's chat template, sends raw text to
`/v1/completions`, and parses `<tool_call>` blocks out of the reply on its own —
`/v1/chat/completions`, `tools`, `tool_choice` and `TOOL_CALL_RECOVERY` play no
part. That is correct for a leaderboard, but it means **a model whose
`<tool_call>` marker is unreliable scores far below its ability and this server's
recovery cannot help**: a correct call whose marker came out mangled is scored
`Wrong number of functions.` A score from such a model is a lower bound, not a
figure to compare against the public leaderboard. `examples/bfcl/README.md`
covers how to tell the two apart.

## Performance policy

`POST /v1/server/performance_policy` hands the SDK a `Genie_PerformancePolicy_t` — nine values from `burst` down to `extreme_power_saver` — which is meant to reach the HTP backend's power/clock driver and trade speed against power.

**Whether the device honours it is target-dependent, and you should not assume it does.** Measured on an SA8255P running as a hypervisor guest: all nine policies produced the same numbers. Across 9 policies x 4 interleaved rounds (36 streamed runs, 4B model, 378-token prompt), median TTFT spanned 561.8-566.2 ms and decode 10.79-10.83 tok/s — a 0.8% and 0.4% spread, with `burst` and `extreme_power_saver` indistinguishable. The spread *within* a single policy was wider than the spread *between* policies. Baking the profile into the model bundle's HTP backend extension config and restarting made no difference either, so this is the device ignoring the profile rather than the runtime call being dropped.

Two things follow for anyone using this endpoint:

- **The GET is not evidence.** It returns the value the SDK stored, not a reading from the device, so a policy that round-trips perfectly may still be doing nothing. Only a timed measurement tells you whether a policy has an effect.
- **Measure before you rely on it.** On a target where the policy does work, pinning `burst` is worth doing before a benchmark. On one where it does not, pinning it is harmless but buys nothing.

Note the measurement above is of speed only — no power draw was measured. A policy that changes neither latency nor throughput is unlikely to be saving power, but that is an inference, not a result. If you need the power side of the trade, measure it directly.

The API is left in place because it costs nothing and the answer differs per target. If you are bringing this server up on new hardware, run a few timed generations under `burst` and under `extreme_power_saver` before deciding whether the policy is a knob you have.

## Benchmarking

Recommended steps for getting reproducible benchmark numbers:

1. Pin the target slot's performance policy with `POST /v1/server/performance_policy {"model": "...", "policy": "burst"}` — on a target where the policy has an effect this removes a source of variance, and where it has none it costs nothing. See [Performance policy](#performance-policy).
2. Wait for that slot to be free with `GET /v1/server/idle?slot=<name>`, then start measuring.
3. TTFT (Time To First Token) is automatically logged as `TTFT [<request_id>] slot=<name> cache=<HIT|MISS|NONE> <ms>ms` (logged by the shared generation engine for `/v1/completions` and `/v1/chat/completions` alike).
4. `GET /v1/server/status`'s `slots[].context_occupancy` lets you monitor each slot's KV cache usage between phases without blocking inference.
5. For multi-turn/long-context benchmarks, pre-caching the system prompt with `POST /v1/prefix/warmup` lets you measure without the first request's prefill time skewing the result.
6. To benchmark two slots at once, run this procedure independently for each slot (they never block each other).
7. After benchmarking, it's recommended to restore `POST /v1/server/performance_policy {"policy": "balanced"}` (or whatever the target's default is).

## Troubleshooting

| Symptom | What to check |
|---|---|
| Exits immediately at startup with `sys.exit(1)` | Check that `env_config.json` exists, and that each slot's `model_root`'s `genie_config.json` and its referenced files (tokenizer/extensions/ctx-bins) actually exist. Look for `Required asset not found` / `GenieDialogConfig_createFromJson failed` / `Failed to load model for slot '<name>'` in the logs. |
| `libGenie.so` fails to load | Check the `QAIRT_SDK_ROOT`/`ADSP_LIBRARY_PATH` settings and that `lib/aarch64-oe-linux-gcc11.2/libGenie.so` actually exists. Fix the path if your target platform differs. |
| The response doesn't match the expected template | Check the relevant slot's `active_model` via `/v1/server/status` (or the startup log's `Slot '<name>' ready: model=... template=...`). Either set `CHAT_TEMPLATE` explicitly, or include a hint string (`llama3`/`llama2`/`mistral`) in the model directory name. |
| `finish_reason` doesn't match expectations | `"length"` is returned when the generation hit `max_tokens` (counted server-side — the SDK reports it as a plain stop) or on `GENIE_STATUS_WARNING_CONTEXT_EXCEEDED`. An abort from a client disconnect is still returned as `"stop"` (the default) by design. |
| Requests never reach the second slot | Check via `GET /v1/models` or `GET /v1/server/status` whether the request's `model` field exactly matches the target slot's `active_model_id` (= its model directory name). Any non-matching string always falls back to the primary slot (`slots[0]`), by design. |
| LoRA still applied after switching models | On a successful `/v1/models/switch`, that slot's `active_lora_adapter` is automatically reset to `""`, but if the SDK's internal state disagrees, check the real value with `/v1/lora/current?model=...`. |
| Prefix warmup returns `422` | If the target slot's model template is llama2/mistral, the system prompt gets folded into `[INST]` and can't be split/cached — this is by design. |
| `device_id` was set but the NSP isn't pinned | Check the startup log for a `device_id=... requested but ... has no dialog.engine.backend.extensions file to patch — NSP pinning skipped` warning. Pinning is impossible if the model's `genie_config.json` doesn't reference an HTP backend extension config file. |
| A slot suddenly fails every request with `GenieDialog_query failed [...]: -1` / `batch dispatch failed` in the logs | The slot is wedged: a stock QAIRT 2.49.40.260810 `libGenie.so` and a model exported at several context lengths. The request that caused it succeeded and returned normally; the failures start with the one after. `GenieDialog_reset()` does not recover it. **Recovery**: reload that slot with `POST /v1/models/switch {"slot": "<name>", "model_dir": "<same model>"}`. **Prevention and the full explanation**: [QAIRT Version Issues](./QAIRT_VERSIONS.md#d1--a-stock-library-wedges-a-slot). |

## Limitations

- **This server does not guard against the stock-library slot wedge**, and does not detect or recover from it — a wedged slot keeps failing until something reloads its model. Preventing it is a deployment choice, not a server setting: see [QAIRT Version Issues](./QAIRT_VERSIONS.md).
- One slot = one `GenieDialog` handle processed serially. The number of concurrent inferences is capped at the number of slots (just 1 in a single-slot configuration).
- `n > 1` (multiple completions per request) is not supported.
- `logprobs` require `numpy` + the model tokenizer, and are non-streaming only. Prompt scoring (`echo`+`logprobs`, lm_eval loglikelihood) runs the whole prompt at decode speed and is therefore off by default (see [Logprobs](#logprobs)).
- Runtime `presence_penalty`/`frequency_penalty` are accepted but ignored: the SDK's runtime sampler-update path (`GenieSampler_applyConfig`) only applies `seed`/`temp`/`top-k`/`top-p`.
- The Llama2/Mistral template doesn't support the prefix KV cache.
- Never start with `--workers` greater than 1 (it breaks each slot's global NPU handle state).
- `POST /v1/models/switch` frees the old model before loading the new one by default, so a slot can end up with no model loaded at all if the new load fails; every endpoint that touches that slot then returns `503` until a later switch succeeds. `"unload_first": false` keeps the old model as a fallback by holding both on the slot's HTP device at once, but **that overlap is not dependable on the SA8255P board** — over 36 measured swaps the outcome did not follow from which models were involved (the same pair went 6/6 in one run and 0/8 in another), so it is only worth using where the device has memory to spare and your own swaps have been tested there. See the endpoint's own docs.
- If two slots share the same `active_model_id` (model directory name), automatic routing by the `model` field prefers whichever slot appears later in the `slots` array. Pass an explicit `"slot": "<name>"` in the request body to `/v1/completions`/`/v1/chat/completions` to target one directly (it overrides `model`-based routing), or use the other APIs that address slots directly by name (`/v1/models/switch`'s `slot`, `/v1/server/idle`'s `?slot=`).
- **A bundle whose `dialog.type` is `ssd-q1` (speculative decoding) needs a patched library**, because this server resets before every query and a stock 2.49 does not survive that on such a dialog. LoRA is unusable there for the same reason. [D5](./QAIRT_VERSIONS.md#d5--reset-corrupts-a-speculative-decoding-dialog) has the symptom, the cause, and what to change if you cannot patch.
- Grammar constraints ([see the relevant section](#grammar-constrained-decoding)) are fixed per model/slot and don't support per-request switching like `response_format` (the Genie SDK's public API has no runtime way to change it).
- VLM ([see the relevant section](#vlm-multimodal-support)) only supports single-shot requests (no conversation history). A request's own `max_tokens`/`stop` are ignored — generation is bounded by `VLM_SLOTS[].max_tokens` for the whole slot instead — and a client disconnect does not stop the run: there is no `GenieNode`/`GeniePipeline` abort call, so the slot stays busy until the answer finishes and the next request waits. Streaming does work, and returns the same text as the non-streaming call. LoRA, the prefix KV cache, grammar constraints, and `/v1/models/switch` are unsupported on VLM slots.
