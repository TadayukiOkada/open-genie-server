# Running lm_eval against open-genie-server

*English | [日本語](./README.ja.md)*

`lm_eval`'s `local-completions` backend talks to `/v1/completions`. Multiple-choice
tasks (hellaswag, arc, mmlu, ...) score every answer choice with a
`echo` + `logprobs` request — the server's prompt-scoring mode — while generation
tasks (gsm8k, ...) use ordinary completions.

| File | What it is |
|---|---|
| `run_lm_eval.sh` | wrapper: flips the prompt-scoring switch on (and back), then calls `lm_eval` with the model_args this server needs |
| `compare_runs.py` | compares two `--log_samples` runs item by item (e.g. the board against the same model in fp32) |

## 1. Install lm_eval

The `[api]` extra is required — without it `local-completions` fails on a
missing `tenacity`. A CPU-only torch keeps the install to about 1 GB:

```bash
python3 -m venv /tmp/lmeval
/tmp/lmeval/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
/tmp/lmeval/bin/pip install "lm_eval[api]" transformers
```

Run this on a host that can reach the board over HTTP, not on the board itself.

## 2. Load the model you want to evaluate

```bash
curl -X POST http://192.168.1.2:8080/v1/models/switch \
  -H 'Content-Type: application/json' \
  -d '{"slot": "chat", "model_dir": "/home/root/models/qwen3_4b_instruct_2507-genie-w4a16-qualcomm_sa8775p", "unload_first": true}'
```

## 3. Run

```bash
LM_EVAL=/tmp/lmeval/bin/lm_eval \
  ./run_lm_eval.sh http://192.168.1.2:8080 Qwen/Qwen3-4B-Instruct-2507 hellaswag 100
```

The **tokenizer argument must be the same model** the board has loaded.
`lm_eval` tokenizes locally to work out how much of each request is context,
and slices the returned logprobs at that boundary — a different tokenizer
silently misaligns every score. `model=genie-local` is not a HuggingFace repo
id, so lm_eval cannot infer it.

Results land in `./lm_eval_out` (override with `OUT=`), with
`--log_samples` writing per-item detail next to them.

## Expect it to be slow

Prompt scoring runs the **whole prompt at decode speed** — that is what makes
the loglikelihood exact — and requests are serialized per slot. Measured on
SA8255P with `qwen3_4b_instruct_2507` (w4a16):

| | |
|---|---|
| per request | ~3.4 s (hellaswag-sized prompts) |
| hellaswag `--limit 100` | 400 requests, ~23 min |

`--limit` is not optional in practice. A full hellaswag run is 40,168 requests
(~38 hours). Keep `num_concurrent=1`: a slot processes one request at a time
anyway, and concurrency only adds queueing.

## Comparing against the unquantized model

```bash
# same task, same --limit, the fp32 model on CPU
/tmp/lmeval/bin/lm_eval --model hf --tasks hellaswag --limit 100 --batch_size 4 \
    --device cpu --output_path ./lm_eval_hf --log_samples \
    --model_args pretrained=Qwen/Qwen3-4B-Instruct-2507,dtype=float32

/tmp/lmeval/bin/python compare_runs.py \
    ./lm_eval_out/genie-local/samples_hellaswag_*.jsonl \
    ./lm_eval_hf/Qwen__Qwen3-4B-Instruct-2507/samples_hellaswag_*.jsonl
```

`compare_runs.py` prints per-item argmax for both sides and the agreement
rate. Agreement is the meaningful number: raw loglikelihoods differ under
4-bit weights, but the *ranking* is what a multiple-choice task scores.

## Reading the results

- **Do not compare a board score against a published fp32 leaderboard
  number.** Compare it against your own fp32 run of the same model, same task,
  same `--limit`.
- **Prefer `acc` over `acc_norm`.** Length normalization amplifies per-token
  deviation on short continuations.
- With `--limit 100` the standard error is around ±0.05, so differences below
  ~10 points are not evidence of anything.

See [docs/MANUAL.md](../../docs/MANUAL.md#logprobs) for the scoring mode's
gate, its cost, and what has been verified about its numbers.
