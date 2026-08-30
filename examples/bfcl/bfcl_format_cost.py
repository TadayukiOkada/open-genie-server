#!/usr/bin/env python3
"""Counterfactual: what would these generations score if the model had used the
output format BFCL's prompting mode asks for?

BFCL's prompting mode (`ret_fmt=python`) asks for

    [func_name(param=value, ...)]

and decodes the reply with `ast.parse`. A model that emits the same call as JSON
fails every entry with `Invalid syntax. Failed to decode AST.` even when the
function name and every argument are right. This copies a BFCL result tree,
rewrites the recognised JSON shapes into the required syntax, and leaves
everything else untouched — then BFCL's own evaluator scores the copy.

Wrong calls stay wrong: only the wrapper is rewritten. Argument values are
re-emitted with `repr()`, so a string stays a string and a number stays a
number — a model that quoted a numeric argument keeps that mistake.

This is an ANALYSIS, not a score. The real BFCL score is the one from the
unrepaired run: emitting the wrong format is real model behaviour and reporting
the repaired number as the result would misstate it. What it separates is "the
model cannot call functions" from "the model cannot follow the output format".

Sibling of bfcl_marker_cost.py, which does the same job for a mangled
<tool_call> marker.
"""
import argparse, ast, json, re, shutil, sys
from pathlib import Path

FENCE = re.compile(r"^```(?:json|python)?\s*|\s*```$", re.MULTILINE)
PY_CALL = re.compile(r"^\s*\[\s*[A-Za-z_][\w.]*\s*\(")
BARE_CALL = re.compile(r"^\s*[A-Za-z_][\w.]*\s*\(.*\)\s*$", re.DOTALL)

# keys various shapes use for the function name and for the argument mapping
NAME_KEYS = ("name", "function", "tool_name", "function_name", "type")
ARG_KEYS = ("parameters", "params", "arguments", "args")


def _render(name: str, args: dict) -> str:
    return f"{name}({', '.join(f'{k}={v!r}' for k, v in args.items())})"


def _from_obj(obj):
    """(name, args) out of one dict, across the shapes we have observed."""
    if not isinstance(obj, dict):
        return None

    # {"type": "function", "function": "x", "params": {...}}
    if obj.get("type") == "function":
        name = next((obj[k] for k in ("function", "name") if isinstance(obj.get(k), str)), None)
        args = next((obj[k] for k in ARG_KEYS if isinstance(obj.get(k), dict)), {})
        return (name, args) if name else None

    name = next((obj[k] for k in NAME_KEYS if isinstance(obj.get(k), str)), None)
    if not name or not re.fullmatch(r"[A-Za-z_][\w.]*", name):
        return None

    args = next((obj[k] for k in ARG_KEYS if isinstance(obj.get(k), dict)), None)
    if args is None:
        # {"type": "func", "base": 10, "height": 5} — arguments sit inline
        args = {k: v for k, v in obj.items() if k not in NAME_KEYS + ARG_KEYS}
    return name, args


def repair(raw: str) -> tuple[str, bool]:
    text = FENCE.sub("", raw).strip()
    if not text:
        return raw, False

    if PY_CALL.match(text):
        return raw, False                      # already the required syntax

    if BARE_CALL.match(text):                  # a call, just missing the [ ]
        try:
            ast.parse(f"[{text}]", mode="eval")
        except SyntaxError:
            return raw, False
        return f"[{text}]", True

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return raw, False

    # ["func_name", {args}]
    if (isinstance(obj, list) and len(obj) == 2
            and isinstance(obj[0], str) and isinstance(obj[1], dict)
            and re.fullmatch(r"[A-Za-z_][\w.]*", obj[0])):
        return f"[{_render(obj[0], obj[1])}]", True

    items = obj if isinstance(obj, list) else [obj]
    calls = [c for c in (_from_obj(o) for o in items) if c]
    if not calls:
        return raw, False
    return "[" + ", ".join(_render(n, a) for n, a in calls) + "]", True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="a BFCL project root (holds result/ and score/)")
    ap.add_argument("dst", help="where to write the repaired copy")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not (src / "result").is_dir():
        sys.exit(f"{src}/result not found")
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "result").mkdir(parents=True)
    shutil.copytree(src / "result", dst / "result", dirs_exist_ok=True)

    total = repaired = 0
    for path in sorted((dst / "result").rglob("*_result.json")):
        out = []
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if isinstance(d.get("result"), str):
                total += 1
                d["result"], changed = repair(d["result"])
                repaired += changed
            out.append(json.dumps(d, ensure_ascii=False))
        path.write_text("\n".join(out) + "\n")
    print(f"{repaired}/{total} entries had their output format rewritten -> {dst}")
    print(f"now run:  BFCL_PROJECT_ROOT={dst} bfcl evaluate --model <id> "
          f"--test-category <cat> [--partial-eval]")


if __name__ == "__main__":
    main()
