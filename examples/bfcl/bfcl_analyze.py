#!/usr/bin/env python3
"""How much of a BFCL failure list is the <tool_call> marker rather than the
model getting the call wrong?

Classifies each failed entry by looking at the raw model output:
  marker  - a complete JSON object naming a function is present, but the
            <tool_call>...</tool_call> pair BFCL's regex needs is not
  model    - the tags are there; the call itself was judged wrong
  no_call  - no parseable function-call JSON at all
"""
import argparse, glob, json, re, sys

PAIR = re.compile(r"<tool_call>\n(.*?)\n</tool_call>", re.DOTALL)
OBJ = re.compile(r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:', re.DOTALL)


def classify(raw: str):
    if PAIR.search(raw):
        return "model"
    if OBJ.search(raw):
        return "marker"
    return "no_call"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("score_dir")
    args = ap.parse_args()

    files = glob.glob(f"{args.score_dir}/**/*_score.json", recursive=True)
    if not files:
        sys.exit(f"no *_score.json under {args.score_dir}")
    for path in sorted(files):
        counts, samples, total, correct = {}, {}, None, None
        for i, line in enumerate(open(path)):
            d = json.loads(line)
            if i == 0:
                total, correct = d.get("total_count"), d.get("correct_count")
                continue
            raw = str(d.get("model_result_raw") or "")
            kind = classify(raw)
            counts[kind] = counts.get(kind, 0) + 1
            samples.setdefault(kind, raw[:100])
        failed = sum(counts.values())
        print(f"== {path.split('/')[-1]}")
        print(f"   {correct}/{total} correct ({100*correct/total:.2f}%), {failed} failed")
        for kind in ("marker", "model", "no_call"):
            n = counts.get(kind, 0)
            if not n:
                continue
            share = 100 * n / failed
            print(f"   {kind:8s} {n:4d}  ({share:5.1f}% of failures)")
            print(f"            e.g. {samples[kind]!r}")
        if counts.get("marker") and total:
            ceiling = 100 * (correct + counts["marker"]) / total
            print(f"   -> if every marker-only failure had carried its tags: "
                  f"at most {ceiling:.2f}%")


if __name__ == "__main__":
    main()
