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

A Gemma 4 E2B QAT export (w4a16) is faster: ~2.2 s per request, ~15 min for
hellaswag `--limit 100`.

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

### The BOS has to match

A bundle whose dialog context names `bos-token` — Gemma 4's do — has the SDK
prepend that token to every query, prompt scoring included. The server's
tokenizer adds none, so the logprobs it returns start at the prompt's first
token, which is where lm_eval's slice expects them to start. Nothing to set on
the board side.

The reference has to see that BOS too, and `--model hf` does not always give it
one: Gemma 4's HF tokenizer adds no BOS by default, and lm_eval (0.4.13) does
not force it. Pass `add_bos_token=True`:

```bash
--model_args pretrained=google/gemma-4-E2B-it,dtype=float32,add_bos_token=True
```

Without it the reference scores text that starts with no BOS, and the board
appears to beat its own fp32 model: on Gemma 4 E2B, the four continuations of
the first hellaswag item came out 9 to 52 nats lower. Qwen3's tokenizer has no
BOS token, so its reference needs nothing. The rule for another model: if its
tokenizer has a `bos_token_id` that `tokenizer("hello").input_ids` does not
start with, pass the flag.

### A QAT bundle has two references

A bundle quantized from a quantization-aware-trained checkpoint was built from
that checkpoint, not from the fp32 model, and the two can be far apart in raw
loglikelihood. Measured on Gemma 4 E2B, hellaswag `--limit 100`, both on CPU:
`google/gemma-4-E2B-it-qat-mobile-transformers` scored each continuation 20
nats higher than `google/gemma-4-E2B-it`, with the same pick on 88 of 100
items. Two w4a16 exports of that QAT checkpoint on SA8255P sat within 0.4 nats
of it (correlation 0.998). Compared only with the original, those 20 nats look
like the board's.

Run the QAT checkpoint as a second reference; transformers loads it with its
own quantizer (5.17.0 was used):

```bash
--model_args pretrained=google/gemma-4-E2B-it-qat-mobile-transformers,dtype=float32,add_bos_token=True
```

It is slower on CPU than the fp32 model — about 12 minutes against 4 for the
run above — because the quantized layers are unpacked in Python.

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
