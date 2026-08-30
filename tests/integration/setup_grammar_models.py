#!/usr/bin/env python3
"""Build the grammar-constrained-decoding model directories the G0x
integration tests need, on the target (run this on the board).

Grammar is fixed per model/slot (docs/MANUAL.md, "Grammar-Constrained
Decoding"), so exercising three grammar kinds means three model directories.
Each one is just a genie_config.json plus a grammar definition file: every
heavy asset (ctx-bins, tokenizer, embeddings, HTP extensions) is referenced by
an ABSOLUTE path back into the base model bundle, so nothing is copied and the
variants cost a few kilobytes each.

    python3 setup_grammar_models.py \
        --base /home/root/models/qwen3_0_6b-genie-w4a16-qualcomm_sa8775p \
        --out  /home/root/grammar_test

Writes <out>/{json_schema,regex,ebnf,bad_backend}/. The last one carries an
unsupported backend on purpose: the SDK must refuse it (Context.cpp:79).
"""

import argparse
import json
from pathlib import Path

# Kept in sync with the checks in run_integration_tests.py (t_grammar_*).
JSON_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["answer", "confidence"],
    "additionalProperties": False,
}, indent=2)

REGEX = r'\{"sentiment": "(positive|negative|neutral)", "score": 0\.[0-9][0-9]\}'

EBNF = 'root ::= "APPROVE" | "REJECT" | "ESCALATE"\n'

VARIANTS = {
    "json_schema": ("json-schema", "xgrammar", JSON_SCHEMA),
    "regex":       ("regex",       "xgrammar", REGEX),
    "ebnf":        ("ebnf",        "xgrammar", EBNF),
    "bad_backend": ("json-schema", "llguidance", JSON_SCHEMA),
}


def absolutize(config: dict, base: Path) -> dict:
    """Rewrite every relative asset path in the config to point at `base`.
    The server resolves paths against the *variant* directory, which holds no
    assets of its own."""
    dcfg = config["dialog"]
    tok = dcfg.get("tokenizer", {})
    if "path" in tok:
        tok["path"] = str(base / tok["path"])
    bcfg = dcfg.get("engine", {}).get("backend", {})
    if bcfg.get("extensions"):
        bcfg["extensions"] = str(base / bcfg["extensions"])
    binary = dcfg.get("engine", {}).get("model", {}).get("binary", {})
    if isinstance(binary.get("ctx-bins"), list):
        binary["ctx-bins"] = [str(base / b) for b in binary["ctx-bins"]]
    emb = dcfg.get("embedding", {})
    if emb.get("lut-path"):
        emb["lut-path"] = str(base / emb["lut-path"])
    return config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="model bundle to borrow assets from")
    ap.add_argument("--out", required=True, help="directory to write the variants into")
    args = ap.parse_args()

    base = Path(args.base).resolve()
    out = Path(args.out).resolve()
    source = json.loads((base / "genie_config.json").read_text())

    for name, (gtype, backend, definition) in VARIANTS.items():
        vdir = out / name
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "grammar_def.txt").write_text(definition)

        config = absolutize(json.loads(json.dumps(source)), base)
        config["dialog"]["context"]["grammar"] = {
            "backend": backend,
            "type": gtype,
            # relative on purpose: exercises the server's own path resolution
            "file": "grammar_def.txt",
        }
        (vdir / "genie_config.json").write_text(json.dumps(config, indent=4))
        print(f"wrote {vdir}  (type={gtype}, backend={backend})")


if __name__ == "__main__":
    main()
