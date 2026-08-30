# Running BFCL against open-genie-server

*English | [日本語](./README.ja.md)*

The [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
scores function calling. Its open-source handlers speak the OpenAI
`/v1/completions` API, which this server implements, so it runs against a board
with no changes on either side.

| File | What it is |
|---|---|
| `run_bfcl.sh` | wrapper: checks the server, points BFCL at it, generates, evaluates, prints the score |
| `subset_ids.py` | writes BFCL's id file so a run can cover the first N entries of a category instead of all of it |
| `bfcl_analyze.py` | splits a failure list into "the marker was mangled" vs "the call was wrong" |
| `bfcl_marker_cost.py` | builds a marker-repaired copy of a run so BFCL's own scorer can price what the marker cost — an analysis, never the score |
| `bfcl_format_cost.py` | the same idea for a model that answers in the wrong *shape* — rewrites recognised JSON call forms into the `[func(param=value)]` syntax prompting mode asks for |
| `gemma4_handler.py`, `gemma4_fc_handler.py`, `install_gemma4_handler.sh` | register gemma4 with BFCL, in prompting and native function-calling form. See [gemma4](#gemma4) |
| `prompt_lengths.py` | token-length distribution of the prompts BFCL would send, per category, against a context size — offline, no board needed |

## Read this before reading any number you get

**BFCL bypasses this server's chat template and tool parsing.** Each handler
builds the prompt itself — Hermes from the HF tokenizer's chat template for the
Qwen3 handlers, gemma4's own markers for the [gemma4](#gemma4) ones below —
sends it to `/v1/completions` as raw text, and parses the calls out of the reply
on its own. `/v1/chat/completions`, `tools`, `tool_choice` and
`TOOL_CALL_RECOVERY` are not involved, and neither is the server's own dialect
registry: a handler and the server can implement the same dialect and still
disagree, so a BFCL score is not a test of the server's tool support.

That is the right design for a leaderboard — it measures the model, comparably
across servers — but it has a consequence here:

**A model whose `<tool_call>` marker is unreliable scores far below its actual
ability, and this server's recovery cannot help.** On our board,
`qwen3_4b_instruct_2507` w4a16 substitutes Cyrillic for the `<tool_call>` token
on roughly half its calls. The call body is correct and
`/v1/chat/completions` recovers it, but BFCL sees this:

```
ФРАГМЕНТ
{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5, "unit": "units"}}
ФРАГМЕНТ
```

and scores it `Wrong number of functions.` — a correct call, marked wrong,
because the marker is not there.

Measured on our board, `simple_python`, all 400 entries, both models under
BFCL's `-FC` handlers, on the current single-CL exports:

| Python Simple AST | as scored by BFCL | markers repaired, re-scored | bf16 reference |
|---|---|---|---|
| `qwen3_4b_instruct_2507` | **48.50%** | **88.75%** | 95.75% |
| `qwen3_1_7b` | **90.25%** | *(nothing to repair)* | 92.00% |

The repaired column is `bfcl_marker_cost.py` plus a re-run of BFCL's own scorer;
the bf16 column is the same two models on a host GPU, not on this board.

Of the 4B's 206 failures, **186 (90.3%) are marker-only** — a complete, correctly
named call that BFCL could not see. Nineteen were genuinely wrong calls and one
produced no call at all. The 1.7B has **no marker failures at all**: every one of
its 39 failures is a real mistake.

The marker costs the 4B **40.25 points**: 48.50% as scored, 88.75% once the tags
are put back. BFCL ranks it 41.75 points below the 1.7B; once its calls can be
read, the two are 1.5 points apart. The ranking is not wrong about which of the
two to serve — it is wrong about how far apart they are, and about why.

Repair does not recover everything either: 88.75% is still 7.00 points under the
same model's bf16 reference, and that residue is not the marker. Quote 88.75%,
**not** the `at most 95.00%` line `bfcl_analyze.py` prints — that is a ceiling
estimate, and the re-scored repaired run is the measurement.

### Two things that will bite you when quoting these

**A score describes one export, not a model.** The earlier multi-CL export of
these same two models scored **38.50%** (4B; 79.00% repaired) and **55.25%**
(1.7B) on this same category. The 1.7B's 55.25% was not quantization damage —
that export had stopped emitting thinking (42 output tokens per entry on average,
against 328 now), so a non-thinking model was being compared against thinking
ones. Re-exporting moved it 35 points. Do not carry a number across an export,
and do not compare a thinking run against a non-thinking one.

**A score describes one BFCL mode.** Everything above is `-FC`. The same 1.7B
export under the prompting handler scores **87.00%** — close, but that is a fact
about this model rather than a rule; under [gemma4](#gemma4) the two modes are 24
points apart. The CSV's `Model` column ends in `(FC)` or `(Prompt)`; check it
before putting two numbers in one table.

### Does `TOOL_CALL_RECOVERY` fix the score? No — and yes, for your application

**It does not change the BFCL number.** Recovery lives in
`/v1/chat/completions`; BFCL uses `/v1/completions` and parses the text itself,
so the switch is not in the path. Turning it on and re-running gives 48.50%
again.

**It does change what a real client sees**, and by exactly the amount above. All
186 of the 4B's marker-only failures name a function the request had declared —
which is precisely the condition recovery keys on — so every one of them is a
call the chat endpoint hands back as `message.tool_calls`. An application
talking to `/v1/chat/completions` with `tools` gets the 88.75% behaviour; BFCL
reports 48.50% because it is measuring something else.

Keep the two apart when quoting either. 48.50% is this model's BFCL score.
88.75% is what its calls are worth once they can be read, and it is the number
that describes serving it through this server.

So: **a BFCL score from this board is a lower bound on the model's function
calling, and on this 4B export it is a substantially loose one.** Do not compare
it to the public leaderboard as if it measured the same thing. Do not "repair"
the markers before scoring either — the corruption is real model output, and
patching it out would misreport the result.

If you want a number that reflects function-calling ability rather than that
defect, run a model whose marker is reliable. `Qwen/Qwen3-1.7B-FC` is registered
in BFCL, and as the table above shows, the current 1.7B export produced **no
marker-only failures at all** across 400 entries of BFCL's own prompts.

## 1. Install

```bash
python3 -m venv /tmp/bfclvenv
/tmp/bfclvenv/bin/pip install bfcl-eval soundfile
```

`soundfile` is not optional: `bfcl_eval` imports `qwen_agent`, which imports it
unconditionally, and the CLI will not start without it.

Run this on a host that can reach the board over HTTP, not on the board.

The handler loads the model's tokenizer and config from Hugging Face the first
time it runs (a few MB, no weights). Set `HF_TOKEN` if you hit rate limits, or
pass `--local-model-path` to `bfcl generate` if the host is offline.

## 2. Load the model you are evaluating

BFCL sends its own model id (`Qwen/Qwen3-4B-Instruct-2507`), which does not match
any slot name, so the request lands on the primary slot whatever it holds. **The
model that answers is whatever is loaded** — `run_bfcl.sh` prints it, and it is
worth checking, because nothing else in the pipeline will tell you.

```bash
curl -sS "$BASE_URL/v1/server/status" | python3 -m json.tool
```

Pick a BFCL model id whose chat template matches the loaded model — that is what
builds the prompt. `bfcl models` lists them; `-FC` uses the native
function-calling template, the same name without it uses the prompting template.

## 3. Run

Start small. A category is a few hundred to a thousand entries and every entry is
a full generation on the NPU:

```bash
BFCL=/tmp/bfclvenv/bin/bfcl ./run_bfcl.sh \
    --base-url http://<board>:8080 \
    --categories simple_python \
    --subset 8 \
    --workdir /tmp/bfcl-smoke
```

Then a whole category:

```bash
BFCL=/tmp/bfclvenv/bin/bfcl ./run_bfcl.sh \
    --base-url http://<board>:8080 \
    --categories simple_python \
    --workdir /tmp/bfcl-simple
```

Per-entry cost is set almost entirely by how many tokens the model emits: decode
runs at 19–20 tok/s on this board for every model measured, so a model that
thinks costs several times more per entry than one that does not.

| Run (single-threaded) | s/entry | 400 entries |
|---|---|---|
| `qwen3_4b_instruct_2507`, `-FC` — does not think | 4.94 | ~33 min |
| `qwen3_1_7b`, prompting — thinks on 92% of entries, median 179 output tokens | 10.2 | ~68 min |
| `gemma4-e2b-it`, `-FC` — does not think | 2.3 | ~15 min |

The 4B figure is from its earlier export; both of its exports are non-thinking,
so it carries. The 1.7B's earlier export ran at 2.53 s/entry — four times faster
than the current one, entirely because it had stopped thinking. Scale by entry
count for other categories.

| Category | Entries |
|---|---|
| `simple_python` | 400 |
| `multiple` | 199 |
| `parallel` | 199 |
| `irrelevance` | 239 |
| `live_simple` | 257 |
| `live_multiple` | 1053 |

`bfcl test-categories` lists all of them, including the collections (`all`,
`ast`, `live`, ...) that expand to many at once.

**Leave `--threads 1`.** A text slot serializes requests anyway, so concurrency
buys nothing and makes a timeout harder to attribute.

## 4. Read the score

`run_bfcl.sh` prints the CSVs at the end. For a partial run **read the
per-category column, not `Overall Acc`** — the overall figure averages in every
category you did not run as a zero. An eight-entry `simple_python` run that
scored 50% shows up as `0.42%` overall and `50.00%` under `Python Simple AST`.

Failures are worth opening. `$WORKDIR/score/**/..._score.json` carries the raw
model output for each one, which is how the marker corruption above was
identified rather than guessed at:

```bash
python3 - <<'EOF'
import glob, json
for f in glob.glob("/tmp/bfcl-simple/score/**/*_score.json", recursive=True):
    for i, line in enumerate(open(f)):
        d = json.loads(line)
        if i == 0:
            print(f, d); continue
        print(d["id"], d.get("error"), repr(str(d.get("model_result_raw"))[:120]))
EOF
```

## Notes on this server specifically

- **Context is 4096 tokens, and it is barely a constraint.** BFCL asks for
  `max_tokens` up to 4096 on top of the prompt, taken from the HF config's
  `max_position_embeddings` (262144 for this model) rather than from the server.
  An explicit `max_tokens` is honoured as given and generation stops when the
  context fills, so that part is harmless. Entries whose *prompt alone* exceeds
  the context come back as `400 context_length_exceeded` — measuring the prompts
  BFCL actually builds, that is **2 entries out of 1053 in `live_multiple`
  (0.2%), and none at all** in `simple_python`, `multiple`, `parallel`,
  `live_simple` or `live_parallel_multiple`:

  | Category | median | p90 | p99 | max | over 4096 |
  |---|---|---|---|---|---|
  | `simple_python` | 240 | 292 | 354 | 394 | 0 |
  | `multiple` | 476 | 714 | 920 | 944 | 0 |
  | `parallel` | 276 | 353 | 424 | 427 | 0 |
  | `live_simple` | 279 | 470 | 672 | 819 | 0 |
  | `live_multiple` | 905 | 1628 | 2335 | 5166 | 2 |
  | `live_parallel_multiple` | 805 | 1599 | 2386 | 2386 | 0 |

  The two that do not fit (`live_multiple_217-93-0` at 4597 tokens and
  `live_multiple_985-216-0` at 5166) were sent to the board and both returned
  `400 context_length_exceeded`, naming the count. They count as failures in the
  score, which at 0.2% is not what moves it.
- **Prompt-scoring mode is irrelevant here.** BFCL never uses `echo`+`logprobs`,
  so `PROMPT_LOGPROBS` can stay off.
- **A run leaves the prefix cache populated.** Harmless, and it does not change
  output — greedy decoding is byte-identical with the cache warm or wiped.

## gemma4

BFCL picks the chat template from the model id, and its `GemmaHandler` hard-codes
Gemma 2/3's `<start_of_turn>`. gemma4 does not have that token in its vocabulary
at all, so a gemma4 model evaluated under `google/gemma-3-*` is being fed markers
that split into roughly nine ordinary tokens each. It also has no registered
model id of its own, and the `google/gemma-3-*` repos are gated on Hugging Face.

`install_gemma4_handler.sh` registers two ids against an installed `bfcl-eval`:

| id | handler | what it does |
|---|---|---|
| `google/gemma4-e2b-it` | `Gemma4Handler` | prompting mode with gemma4's `<\|turn>` markers |
| `google/gemma4-e2b-it-FC` | `Gemma4FCHandler` | **native function calling** — declares tools as `<\|tool>declaration:...<tool\|>` and parses `<\|tool_call>call:NAME{...}<tool_call\|>` |

```bash
./install_gemma4_handler.sh /tmp/bfclvenv

BFCL=/tmp/bfclvenv/bin/bfcl ./run_bfcl.sh \
    --base-url http://<board>:8080 \
    --model google/gemma4-e2b-it-FC \
    --local-model-path /path/to/tokenizer-dir \
    --categories simple_python \
    --workdir /tmp/bfcl-gemma4
```

`--local-model-path` needs a directory holding `config.json` (with
`max_position_embeddings`) and `tokenizer_config.json` alongside the bundle's
`tokenizer.json`. It is what gets you past the gated `google/*` repos, and
gemma4 has no HF repo of its own to fetch.

**Which mode you pick dominates the number.** Measured on our board,
`simple_python`, all 400 entries:

| | Python Simple AST |
|---|---|
| `gemma4-e2b-it` — prompting | **7.50%** |
| the same generations with the call shape rewritten (`bfcl_format_cost.py`) | 28.25% |
| **`gemma4-e2b-it` — native FC** | **31.50%** |

Prompting mode asks for `[func_name(param=value)]` and gemma4 answers in JSON, so
330 of its 370 failures there are `Failed to decode AST` — a syntax verdict on a
model that was never speaking that syntax. **Quote the FC number**; the prompting
one measures the mismatch as much as the model.

Even so, 31.50% is a long way below the Qwen3-1.7B export's 90.25% on the same
category — but **that pair is confounded**. The 1.7B thinks on 369 of the 400
entries (median 179 output tokens); gemma4 thinks on none, because this handler
does not enable it. Some unknown part of the gap is chain-of-thought rather than
function-calling ability. Making it fair needs one of the two sides changed:
enable gemma4's thinking (its template injects `<|think|>` at the top of the
system turn) or turn the 1.7B's off (`/no_think` written into the prompt — BFCL
uses `/v1/completions`, so the server's `enable_thinking` is not in the path).
Neither has been run. Note also that **BFCL moves thinking out of `result` into
`reasoning_content`**, so counting tokens in `result` alone will tell you a
thinking model is not thinking — that mistake was made twice here.

Of the 274 FC failures, 205 are calls the model made and got wrong, 69
are prose with no call at all, and 14 are calls it built correctly but never
terminated — `<|tool_call>call:math.gcd{num1:12,num2:15}` with no closing
`<tool_call|>`. Those 14 are counted as failures here, the same way a mangled
marker is: not closing your own call is model behaviour, and a parser that
forgave it would be scoring something else. Tolerating them would add 3.5 points.
