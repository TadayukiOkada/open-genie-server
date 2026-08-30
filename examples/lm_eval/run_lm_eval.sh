#!/bin/sh
# Run an lm_eval task against a running genie-server.
#
#   ./run_lm_eval.sh <base_url> <hf-tokenizer-id> [task] [limit] [extra lm_eval args...]
#
#   ./run_lm_eval.sh http://192.168.1.2:8080 Qwen/Qwen3-4B-Instruct-2507 hellaswag 100
#
# What it does beyond calling lm_eval: enables the server's prompt-scoring
# switch first and restores it afterwards (loglikelihood tasks are 400s while
# it is off), and passes the model_args that this server needs.
#
# Loglikelihood tasks (hellaswag, arc_*, mmlu, ...) run the whole prompt at
# decode speed, one request per answer choice. Budget accordingly: ~3.4 s per
# request for a 4B w4a16 model on SA8255P, i.e. ~23 min for hellaswag
# --limit 100 (400 requests). Generation tasks (gsm8k, ...) do not need the
# switch, but are equally serial.

set -e

BASE_URL="$1"
TOKENIZER="$2"
TASK="${3:-hellaswag}"
LIMIT="${4:-100}"
[ -n "$BASE_URL" ] && [ -n "$TOKENIZER" ] || {
    echo "usage: $0 <base_url> <hf-tokenizer-id> [task] [limit] [extra args...]" >&2
    echo "  <hf-tokenizer-id> must be the HF repo (or local dir) of the SAME" >&2
    echo "  model the board has loaded — lm_eval tokenizes locally to compute" >&2
    echo "  the context length it slices scores at." >&2
    exit 2
}
[ $# -gt 4 ] && shift 4 || shift $#

LM_EVAL="${LM_EVAL:-lm_eval}"
OUT="${OUT:-./lm_eval_out}"

echo "server : $BASE_URL"
echo "model  : $(curl -sS "$BASE_URL/v1/server/status" | sed 's/.*"active_model":"\([^"]*\)".*/\1/')"
echo "task   : $TASK (limit $LIMIT)"

# Prompt scoring is off by default; remember the state so we can restore it.
WAS=$(curl -sS "$BASE_URL/v1/server/prompt_logprobs" | grep -o '"enabled":[a-z]*' | cut -d: -f2)
restore() {
    [ "$WAS" = "true" ] || curl -sS -X POST "$BASE_URL/v1/server/prompt_logprobs" \
        -H 'Content-Type: application/json' -d '{"enabled": false}' > /dev/null
}
trap restore EXIT INT TERM
curl -sS -X POST "$BASE_URL/v1/server/prompt_logprobs" \
    -H 'Content-Type: application/json' -d '{"enabled": true}' > /dev/null

"$LM_EVAL" --model local-completions \
    --tasks "$TASK" \
    --limit "$LIMIT" \
    --batch_size 1 \
    --output_path "$OUT" \
    --log_samples \
    --model_args "model=genie-local,tokenizer=$TOKENIZER,base_url=$BASE_URL/v1/completions,num_concurrent=1,max_retries=2,tokenized_requests=False,timeout=900" \
    "$@"
