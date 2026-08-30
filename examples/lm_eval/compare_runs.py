"""Compare two lm_eval --log_samples runs item by item.

    python3 compare_runs.py <board samples_*.jsonl> <reference samples_*.jsonl>

Both runs must be the same task with the same --limit, so the documents line
up. Prints per-item argmax on each side and, at the end, how often the two
agree — which is the number that says whether the board is reproducing the
reference model's *decisions*, independent of how far apart the raw
loglikelihoods are.
"""
import json, sys


def load(path):
    rows = [json.loads(l) for l in open(path)]
    rows.sort(key=lambda r: r["doc_id"])
    return rows


a = load(sys.argv[1])   # board
b = load(sys.argv[2])   # reference
assert len(a) == len(b), (len(a), len(b))

same_argmax = both_acc = 0
print("%-5s %-4s %-7s %-7s %-6s %s" % ("item", "gold", "board", "hf", "match", "board lps"))
for ra, rb in zip(a, b):
    la = [float(x[0]) for x in ra["filtered_resps"]]
    lb = [float(x[0]) for x in rb["filtered_resps"]]
    pa, pb = la.index(max(la)), lb.index(max(lb))
    gold = int(ra["doc"]["label"])
    same_argmax += pa == pb
    both_acc += (pa == gold) == (pb == gold)
    print("%-5d %-4d %-7d %-7d %-6s %s"
          % (ra["doc_id"], gold, pa, pb, "yes" if pa == pb else "NO",
             [round(x, 1) for x in la]))

n = len(a)
acc_a = sum(r["acc"] for r in a) / n
acc_b = sum(r["acc"] for r in b) / n
accn_a = sum(r["acc_norm"] for r in a) / n
accn_b = sum(r["acc_norm"] for r in b) / n
print()
print("items                : %d" % n)
print("acc      board=%.3f  hf=%.3f" % (acc_a, acc_b))
print("acc_norm board=%.3f  hf=%.3f" % (accn_a, accn_b))
print("same argmax          : %d/%d (%.0f%%)" % (same_argmax, n, 100 * same_argmax / n))
print("same correctness     : %d/%d" % (both_acc, n))
