#!/usr/bin/env python3
"""Write BFCL's test_case_ids_to_generate.json for the first N entries of each
category, so a run can be smoke-tested (or time-boxed) without doing all of it.

BFCL reads that file only when `bfcl generate` is given --run-ids, and then it
replaces --test-category entirely rather than adding to it.

    python3 subset_ids.py --categories simple_python,multiple --n 20 \
        --out "$BFCL_PROJECT_ROOT/test_case_ids_to_generate.json"
"""
import argparse
import json
import sys
from pathlib import Path


def dataset_dir() -> Path:
    try:
        import bfcl_eval
    except ImportError:
        sys.exit("bfcl_eval is not installed in this interpreter — see README.md")
    d = Path(bfcl_eval.__file__).parent / "data"
    if not d.is_dir():
        sys.exit(f"BFCL data directory not found at {d}")
    return d


def ids_for(category: str, n: int, data: Path) -> list:
    """Entry ids for one category, in file order. The prefix is not always the
    category name (v4 files carry their own ids), so read them rather than
    synthesising them."""
    matches = sorted(data.glob(f"BFCL_v*_{category}.json"))
    if not matches:
        sys.exit(f"No dataset file for category {category!r} under {data}")
    ids = []
    with open(matches[-1]) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.append(json.loads(line)["id"])
            if n and len(ids) >= n:
                break
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", required=True,
                    help="comma-separated BFCL category names")
    ap.add_argument("--n", type=int, default=20,
                    help="entries per category (0 = all)")
    ap.add_argument("--out", required=True, help="path to write the id file to")
    args = ap.parse_args()

    data = dataset_dir()
    out = {c.strip(): ids_for(c.strip(), args.n, data)
           for c in args.categories.split(",") if c.strip()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    total = sum(len(v) for v in out.values())
    print(f"{args.out}: {total} entries across {len(out)} categor"
          f"{'y' if len(out) == 1 else 'ies'}")
    for c, ids in out.items():
        print(f"  {c}: {len(ids)}")


if __name__ == "__main__":
    main()
