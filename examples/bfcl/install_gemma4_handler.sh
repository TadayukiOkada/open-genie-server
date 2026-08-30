#!/usr/bin/env bash
# Register the gemma4 model id inside an installed bfcl-eval.
#
# BFCL resolves the chat template from the model id and its GemmaHandler
# hard-codes Gemma 2/3's <start_of_turn>, so gemma4 needs its own handler.
# This copies gemma4_handler.py into the package and appends one entry to
# MODEL_CONFIG_MAPPING. Idempotent; re-run after reinstalling bfcl-eval.
#
#   ./install_gemma4_handler.sh /tmp/bfclvenv
set -euo pipefail

VENV="${1:-/tmp/bfclvenv}"
PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "no interpreter at $PY" >&2; exit 2; }

PKG="$("$PY" -c 'import bfcl_eval, os; print(os.path.dirname(bfcl_eval.__file__))')"
SRC="$(dirname "$0")/gemma4_handler.py"
[ -f "$SRC" ] || { echo "gemma4_handler.py not next to this script" >&2; exit 2; }

cp "$SRC" "$PKG/model_handler/local_inference/gemma4.py"

SRC_FC="$(dirname "$0")/gemma4_fc_handler.py"
[ -f "$SRC_FC" ] || { echo "gemma4_fc_handler.py not next to this script" >&2; exit 2; }
cp "$SRC_FC" "$PKG/model_handler/local_inference/gemma4_fc.py"

CFG="$PKG/constants/model_config.py"

if grep -q 'MODEL_CONFIG_MAPPING\["google/gemma4-e2b-it"\]' "$CFG"; then
    echo "prompting handler already registered"
else
    cat >> "$CFG" <<'PYEOF'

# --- added by genie-server examples/bfcl/install_gemma4_handler.sh ---
# gemma4 keeps Gemma's turn structure but marks turns with <|turn> / <turn|>;
# the Gemma 2/3 spelling is not in its vocabulary, so GemmaHandler's hard-coded
# markers would split into ~9 ordinary tokens each. system is its own turn.
from bfcl_eval.model_handler.local_inference.gemma4 import Gemma4Handler  # noqa: E402

MODEL_CONFIG_MAPPING["google/gemma4-e2b-it"] = ModelConfig(
    model_name="google/gemma4-e2b-it",
    display_name="Gemma4-e2b-it (Prompt)",
    url="https://ai.google.dev/gemma",
    org="Google",
    license="gemma-terms-of-use",
    model_handler=Gemma4Handler,
    input_price=None,
    output_price=None,
    is_fc_model=False,
    underscore_to_dot=False,
)
PYEOF
    echo "registered google/gemma4-e2b-it"
fi

if grep -q 'MODEL_CONFIG_MAPPING\["google/gemma4-e2b-it-FC"\]' "$CFG"; then
    echo "FC handler already registered"
else
    cat >> "$CFG" <<'PYEOF'

# Native function calling: gemma4 declares and calls tools with its own tokens
# (<|tool>declaration:..., <|tool_call>call:...), not the Hermes JSON the other
# OSS handlers use.
from bfcl_eval.model_handler.local_inference.gemma4_fc import Gemma4FCHandler  # noqa: E402

MODEL_CONFIG_MAPPING["google/gemma4-e2b-it-FC"] = ModelConfig(
    model_name="google/gemma4-e2b-it",
    display_name="Gemma4-e2b-it (FC)",
    url="https://ai.google.dev/gemma",
    org="Google",
    license="gemma-terms-of-use",
    model_handler=Gemma4FCHandler,
    input_price=None,
    output_price=None,
    is_fc_model=True,
    underscore_to_dot=False,
)
PYEOF
    echo "registered google/gemma4-e2b-it-FC"
fi

"$PY" - <<'PYEOF'
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
for mid in ("google/gemma4-e2b-it", "google/gemma4-e2b-it-FC"):
    c = MODEL_CONFIG_MAPPING[mid]
    print(f"verify: {mid}  ->  {c.display_name}  ({c.model_handler.__name__})")
PYEOF
