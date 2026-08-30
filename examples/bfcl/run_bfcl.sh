#!/usr/bin/env bash
# Run the Berkeley Function Calling Leaderboard against genie-server.
#
# BFCL's OSS handlers talk to /v1/completions and build the prompt themselves,
# so this measures the MODEL, not this server's chat template or tool parsing.
# See README.md — that distinction changes how the number should be read.
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8080}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507-FC}"
CATEGORIES="${CATEGORIES:-simple_python}"
WORKDIR="${WORKDIR:-$PWD/bfcl-run}"
THREADS="${THREADS:-1}"
SUBSET="${SUBSET:-0}"
BFCL="${BFCL:-bfcl}"
LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-}"

usage() {
    cat <<EOF
usage: $(basename "$0") [options]

  --base-url URL     genie-server root, no /v1 (default: $BASE_URL)
  --model NAME       BFCL model id; see 'bfcl models' (default: $MODEL)
  --categories LIST  comma-separated BFCL categories (default: $CATEGORIES)
  --subset N         only the first N entries per category (0 = all)
  --threads N        concurrent requests (default: $THREADS)
  --workdir DIR      where results and scores are written (default: $WORKDIR)
  --local-model-path DIR
                     load the tokenizer/config from DIR instead of Hugging
                     Face. Needed for a gated repo, for an offline host, or
                     for a model that has no HF repo at all. DIR must hold
                     config.json and tokenizer_config.json.

Environment variables of the same name (BASE_URL, MODEL, ...) work too.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --base-url)   BASE_URL="$2"; shift 2 ;;
        --model)      MODEL="$2"; shift 2 ;;
        --categories) CATEGORIES="$2"; shift 2 ;;
        --subset)     SUBSET="$2"; shift 2 ;;
        --threads)    THREADS="$2"; shift 2 ;;
        --workdir)    WORKDIR="$2"; shift 2 ;;
        --local-model-path) LOCAL_MODEL_PATH="$2"; shift 2 ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

command -v "$BFCL" >/dev/null 2>&1 || {
    echo "'$BFCL' not on PATH. Install it (see README.md) or set BFCL=/path/to/bfcl." >&2
    exit 2
}

# subset_ids.py imports bfcl_eval to read the datasets, so it has to run under
# the same interpreter bfcl itself does — usually the venv's, which is not the
# python3 on PATH.
if [ -z "${PYTHON:-}" ]; then
    _bfcl_bin="$(command -v "$BFCL")"
    if [ -x "$(dirname "$_bfcl_bin")/python3" ]; then
        PYTHON="$(dirname "$_bfcl_bin")/python3"
    else
        PYTHON="python3"
    fi
fi

echo "== server =="
curl -fsS --max-time 10 "$BASE_URL/health" || {
    echo "  $BASE_URL/health did not answer." >&2; exit 2; }
echo
# BFCL sends its own model id, which routes to the primary slot rather than
# selecting anything — so print what will actually answer. A BFCL score is
# meaningless without knowing which model produced it.
echo "  loaded: $(curl -fsS --max-time 10 "$BASE_URL/v1/server/status" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["active_model"])' \
    2>/dev/null || echo '(status unavailable)')"
echo "  bfcl model id: $MODEL"
echo

mkdir -p "$WORKDIR"
export BFCL_PROJECT_ROOT="$WORKDIR"
export REMOTE_OPENAI_BASE_URL="${BASE_URL%/}/v1"
export REMOTE_OPENAI_API_KEY="${REMOTE_OPENAI_API_KEY:-EMPTY}"

LOCAL_PATH_ARG=()
if [ -n "$LOCAL_MODEL_PATH" ]; then
    LOCAL_PATH_ARG=(--local-model-path "$LOCAL_MODEL_PATH")
    echo "  local model path: $LOCAL_MODEL_PATH"
    echo
fi

RUN_IDS=()
EVAL_EXTRA=()
if [ "$SUBSET" != "0" ]; then
    "$PYTHON" "$(dirname "$0")/subset_ids.py" \
        --categories "$CATEGORIES" --n "$SUBSET" \
        --out "$WORKDIR/test_case_ids_to_generate.json"
    # --run-ids REPLACES --test-category; --partial-eval then lets the scorer
    # work on the subset instead of insisting on the whole category.
    RUN_IDS=(--run-ids)
    EVAL_EXTRA=(--partial-eval)
    echo
fi

echo "== generate =="
"$BFCL" generate \
    --model "$MODEL" \
    --test-category "$CATEGORIES" \
    --skip-server-setup \
    --num-threads "$THREADS" \
    "${LOCAL_PATH_ARG[@]}" \
    "${RUN_IDS[@]}"

echo
echo "== evaluate =="
"$BFCL" evaluate \
    --model "$MODEL" \
    --test-category "$CATEGORIES" \
    "${EVAL_EXTRA[@]}"

echo
echo "== scores =="
for f in "$WORKDIR"/score/data_overall.csv "$WORKDIR"/score/data_non_live.csv \
         "$WORKDIR"/score/data_live.csv; do
    [ -f "$f" ] && { echo "--- $(basename "$f")"; cat "$f"; echo; }
done
echo "Per-entry results:  $WORKDIR/result/"
echo "Per-entry scoring:  $WORKDIR/score/   (failed entries carry the raw model output)"
