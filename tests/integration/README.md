# Host-side integration tests

*English | [日本語](./README.ja.md)*

Runs from a **host PC** against a live open-genie-server on a device (over the
REST API), exercises every feature area, and writes a Markdown + JSON report.
If the server dies mid-run, the report records which test it died in, the
request that was in flight, and the last passing test; the remaining tests
are marked ABORTED.

## Setup

On the host PC:

```bash
pip install requests
cp test_config.sample.json test_config.json   # edit paths / toggles
```

On the device: start open-genie-server with one of the sample configs from
[examples/config/](../../examples/config/) (or your own `env_config.json`).

## Run

```bash
python3 run_integration_tests.py --config test_config.json
# override the target address without editing the config:
python3 run_integration_tests.py --config test_config.json --base-url http://192.168.1.2:8080

python3 run_integration_tests.py --list          # list test ids
python3 run_integration_tests.py --only C01,L03  # run a subset
```

Reports land in `./reports/integration_report_<timestamp>.{md,json}`.
Exit code 0 = all green, 1 = failures or server down, 2 = server unreachable
at startup.

## What is covered

| IDs | Area |
|---|---|
| S01-S04 | health, model listing (+`/v1`-less aliases), per-slot status, idle wait |
| C01-C07 | completions: sync/echo/batch/streaming/stop sequences/greedy determinism/`finish_reason` |
| CH01-CH05 | chat: sync/streaming+usage/`enable_thinking=false`/prefix KV cache warmup+hit/function calling round-trip |
| D01 | a mid-stream client disconnect aborts the generation instead of leaving it to run out |
| E01 | OpenAI error envelope: malformed JSON, `n>1`, `stream`+`logprobs`, unknown slot |
| L01-L03 | logprobs (chat + completions) and prompt scoring (lm_eval loglikelihood shape, incl. the disabled→400 gate) |
| P01, P02 | performance policy: one round-trip, then all nine `Genie_PerformancePolicy_t` values plus rejection of an unknown name. Whether a policy changes anything is measured separately by `measure_perf_policy.py`, which reads the SDK's own KPIs and needs `GENIE_PROFILE=true` |
| M01, M02 | model hot-swap (+restore), LoRA apply/release — **config-gated** |
| V01-V05 | VLM: image chat, streaming SSE, stream-vs-sync parity, usage accounting, slot recovery after a disconnect (no abort API, so the generation finishes) — **config-gated** |
| G01-G08 | grammar-constrained decoding: JSON Schema (sync/streaming/repeat), logprobs under masking, regex, EBNF, unsupported-backend rejection, restore — **config-gated**, see below |
| Z01 | final server health |

Feature areas without config (`switch`/`lora`/`vlm`) are reported as SKIP,
never FAIL. Model-dependent behaviors (whether the model actually emits a
tool call, `<think>` handling) pass with a note rather than failing the
server.

## Local dry-run (no device)

`fake_server.py` serves the same FastAPI app against the fake SDK used by
the unit tests — useful for developing the harness itself:

```bash
python3 fake_server.py --port 18080 &
python3 run_integration_tests.py --base-url http://127.0.0.1:18080
```

## Grammar tests (G01-G08)

Grammar is fixed per model/slot, so each grammar kind needs its own model
directory. Build them on the target (they are a few KB each — every heavy
asset is referenced by absolute path back into the base bundle):

```bash
adb push setup_grammar_models.py /home/root/
adb shell "cd /home/root && .venv/bin/python3 setup_grammar_models.py \
    --base /home/root/models/qwen3_0_6b-genie-w4a16-qualcomm_sa8775p \
    --out  /home/root/grammar_test"
```

Then set `grammar.enabled: true` in `test_config.json`. Each G0x test switches
the slot to the directory it needs, and G08 switches
`grammar.restore_model_dir` back.

**These tests need a `libGenie.so` that was built with `ENABLE_GRAMMAR`.** The
XGrammar backend is only in Qualcomm's prebuilt library: `GrammarBackend::create`
is declared in `qualla/grammar.hpp` but defined nowhere in the shipped sources,
so a library rebuilt from `examples/Genie/` has no grammar support and every
G0x model load fails with `GenieDialog_create failed: -1` plus
`"Grammar backend configured but qualla was built without ENABLE_GRAMMAR"` in
the server log.
