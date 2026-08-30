"""How many BFCL entries build a prompt longer than this server's context?

Uses BFCL's own QwenFCHandler._format_prompt and the model's HF tokenizer, so
the count is the prompt BFCL would actually send, not an approximation. Nothing
here touches the board.

    python3 prompt_lengths.py 4096 simple_python,live_multiple

Defaults: context 4096, category live_multiple. Change HF below if the model
you are serving tokenizes differently.
"""
import json, sys
from pathlib import Path

import bfcl_eval
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from transformers import AutoTokenizer

CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
CATS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["live_multiple"]
HF = "Qwen/Qwen3-4B-Instruct-2507"

h = QwenFCHandler.__new__(QwenFCHandler)      # only _format_prompt is needed
tok = AutoTokenizer.from_pretrained(HF, trust_remote_code=True)
data = Path(bfcl_eval.__file__).parent / "data"

for cat in CATS:
    f = sorted(data.glob(f"BFCL_v*_{cat}.json"))[-1]
    lens, over, worst = [], 0, ("", 0)
    for line in open(f):
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        msgs = e["question"][0] if isinstance(e["question"][0], list) else e["question"]
        try:
            prompt = h._format_prompt(msgs, e["function"])
        except Exception as ex:
            print(f"  {e['id']}: could not format ({type(ex).__name__})")
            continue
        n = len(tok.tokenize(prompt))
        lens.append(n)
        if n >= CTX:
            over += 1
            if n > worst[1]:
                worst = (e["id"], n)
    lens.sort()
    pct = lambda p: lens[min(len(lens) - 1, int(len(lens) * p))]
    print(f"== {cat}: {len(lens)} entries, context {CTX}")
    print(f"   median {pct(0.5)}  p90 {pct(0.9)}  p99 {pct(0.99)}  max {lens[-1]}")
    print(f"   at or over the context: {over} ({100*over/len(lens):.1f}%)"
          + (f"   worst: {worst[0]} at {worst[1]}" if over else ""))
