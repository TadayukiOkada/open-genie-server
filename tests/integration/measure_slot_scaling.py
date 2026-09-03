#!/usr/bin/env python3
"""Measure what more slots are worth, when some of them share an NSP core.

    python3 measure_slot_scaling.py [base_url] [requests_per_phase] [max_tokens]

Needs a server with at least two loaded TEXT_SLOTS; their names and device_ids
are read from /v1/server/status, so any naming works. measure_parallelism.py
answers "do two slots overlap"; this one answers "does a third and fourth slot
add anything", which needs to know which core each slot sits on.

Reported in docs/MANUAL.md ("More slots than cores buys nothing"): on SA8255P
with four 0.6B/CL1024 slots, two per NSP, 1.06x for a same-core pair, 1.34x
for a cross-core pair, 1.41x for all four.

Every phase does the SAME total work -- the same prompt, temperature 0, and a
fixed max_tokens so every request generates the same number of tokens. Only
the concurrency pattern changes, so the wall clocks compare directly:

  A  one at a time, one slot                 -> per-request latency L
  B  two at a time, two slots on ONE core    -> serialized if a core cannot
                                                overlap two dialogs
  C  two at a time, one slot per core        -> the real concurrency
  D  all slots at once                       -> whether slots past the cores
                                                add throughput or just queue

B is skipped unless two slots share a device_id, C unless two device_ids are
present, D unless there are more than two slots.
"""
import json
import statistics
import sys
import threading
import time

import requests

from slot_names import cross_core_pair, same_core_pair, text_slots

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.1.2:8080"
TOTAL = int(sys.argv[2]) if len(sys.argv) > 2 else 8    # requests per phase
MAX_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 64
PROMPT = "Write one short paragraph about the sea."
SLOTS = text_slots(BASE, at_least=2)    # every loaded slot, device_id order


def one(slot, out):
    t0 = time.perf_counter()
    r = requests.post(f"{BASE}/v1/chat/completions", timeout=600, json={
        "model": "genie-local", "slot": slot, "max_tokens": MAX_TOKENS,
        "temperature": 0, "enable_thinking": False,
        "messages": [{"role": "user", "content": PROMPT}]})
    dt = time.perf_counter() - t0
    r.raise_for_status()
    out.append({"slot": slot, "seconds": dt,
                "completion_tokens": r.json()["usage"]["completion_tokens"]})


def phase(label, slots):
    """TOTAL requests, len(slots) at a time, one per slot in each round."""
    recs = []
    t0 = time.perf_counter()
    for _ in range(TOTAL // len(slots)):
        threads = [threading.Thread(target=one, args=(s, recs)) for s in slots]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    wall = time.perf_counter() - t0
    lat = [r["seconds"] for r in recs]
    toks = [r["completion_tokens"] for r in recs]
    print(f"{label:<44} wall {wall:6.2f}s | {len(recs)} req | "
          f"latency med {statistics.median(lat):5.2f}s "
          f"(min {min(lat):.2f} max {max(lat):.2f}) | "
          f"tokens {min(toks)}-{max(toks)}")
    return wall


def main():
    names = ", ".join(f"{n}(dev{d})" for n, d in SLOTS)
    print(f"target {BASE} | {TOTAL} requests/phase | max_tokens={MAX_TOKENS}")
    print(f"slots  {names}")

    same, cross = same_core_pair(SLOTS), cross_core_pair(SLOTS)
    walls = {}
    walls["serial"] = phase(f"A  1 at a time, {SLOTS[0][0]}", [SLOTS[0][0]])
    if same:
        walls["same_core_pair"] = phase(
            f"B  2 at a time, {'+'.join(same)} (one core)", same)
    else:
        print("B  skipped: no two slots share a device_id")
    if cross:
        walls["cross_core_pair"] = phase(
            f"C  2 at a time, {'+'.join(cross)} (two cores)", cross)
    else:
        print("C  skipped: all slots are on one device_id")
    if len(SLOTS) > 2 and TOTAL % len(SLOTS) == 0:
        walls["all_slots"] = phase(
            f"D  {len(SLOTS)} at a time, every slot", [n for n, _ in SLOTS])
    elif len(SLOTS) > 2:
        print(f"D  skipped: {TOTAL} requests do not divide over "
              f"{len(SLOTS)} slots")

    print()
    for key, label in [("same_core_pair", "same-core pair"),
                       ("cross_core_pair", "cross-core pair"),
                       ("all_slots", f"all {len(SLOTS)} slots")]:
        if key in walls:
            print(f"{label:<20} speedup vs serial : "
                  f"{walls['serial'] / walls[key]:.2f}x")
    print()
    print(json.dumps({f"wall_{k}": v for k, v in walls.items()}, indent=2))


if __name__ == "__main__":
    main()
