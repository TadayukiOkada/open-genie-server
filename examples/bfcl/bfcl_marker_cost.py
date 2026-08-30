#!/usr/bin/env python3
"""Counterfactual: what would these same generations score if only the
<tool_call> marker had survived?

Copies a BFCL result tree, re-wraps every bare `{"name": ..., "arguments": ...}`
object in the tags BFCL's regex requires, and leaves everything else untouched —
then BFCL's own evaluator scores the copy. Nothing about the calls themselves is
changed: wrong arguments stay wrong.

This is an ANALYSIS, not a score. The real BFCL score is the one from the
unrepaired run; a corrupted marker is real model output and reporting the
repaired number as the result would misstate it. What this separates is "the
model cannot call functions" from "this export corrupts one token".
"""
import argparse, json, re, shutil, sys
from pathlib import Path

PAIR = re.compile(r"<tool_call>\n(.*?)\n</tool_call>", re.DOTALL)


def balanced_objects(text: str):
    """(start, end) of each balanced {...} run, string- and escape-aware."""
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth, j, in_str, esc = 0, i, False, False
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    yield i, j + 1
                    break
            j += 1
        if j >= n:
            return
        i = j + 1


def repair(raw: str) -> tuple[str, bool]:
    if PAIR.search(raw):
        return raw, False          # tags already intact — leave it alone
    calls = []
    for s, e in balanced_objects(raw):
        try:
            obj = json.loads(raw[s:e])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            calls.append(raw[s:e])
    if not calls:
        return raw, False
    return "\n".join(f"<tool_call>\n{c}\n</tool_call>" for c in calls), True


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
    print(f"{repaired}/{total} entries had their marker restored -> {dst}")
    print(f"now run:  BFCL_PROJECT_ROOT={dst} bfcl evaluate --model <id> "
          f"--test-category <cat> [--partial-eval]")


if __name__ == "__main__":
    main()
