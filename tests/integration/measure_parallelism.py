#!/usr/bin/env python3
"""Measure whether two text slots really run in parallel.

    python3 measure_parallelism.py [base_url] [pairs] [max_tokens]

Needs a server with two loaded TEXT_SLOTS; their names are read from
/v1/server/status (ordered by device_id), so any naming works. Reported in
docs/MANUAL.md ("Multi Text Slots"): 1.31x on SA8255P with two 0.6B slots.

Three phases, identical work in each request (same prompt, temperature 0,
fixed max_tokens so every request generates the same number of tokens):

  A  one request at a time on slot A       -> per-request latency L
  B  N requests, 2 at a time, SAME slot    -> expect ~N/2 * 2L = N*L (the slot
                                              lock serializes them)
  C  N requests, 2 at a time, one per slot -> expect ~N/2 * L if the two NSPs
                                              are genuinely concurrent
"""
import json
import statistics
import sys
import threading
import time

import requests

from slot_names import text_slot_names

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.1.2:8080"
PAIRS = int(sys.argv[2]) if len(sys.argv) > 2 else 4     # request pairs per phase
MAX_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 64
PROMPT = "Write one short paragraph about the sea."
SLOT_A, SLOT_B = text_slot_names(BASE, want=2)


def one(slot, out=None):
    t0 = time.perf_counter()
    r = requests.post(f"{BASE}/v1/chat/completions", timeout=600, json={
        "model": "genie-local", "slot": slot, "max_tokens": MAX_TOKENS,
        "temperature": 0, "enable_thinking": False,
        "messages": [{"role": "user", "content": PROMPT}]})
    dt = time.perf_counter() - t0
    r.raise_for_status()
    body = r.json()
    rec = {"slot": slot, "seconds": dt,
           "completion_tokens": body["usage"]["completion_tokens"]}
    if out is not None:
        out.append(rec)
    return rec


def phase_serial(n):
    recs = []
    t0 = time.perf_counter()
    for _ in range(n):
        one(SLOT_A, recs)
    return time.perf_counter() - t0, recs


def phase_pairs(n_pairs, slots):
    """n_pairs rounds of two concurrent requests, sent to `slots`."""
    recs = []
    t0 = time.perf_counter()
    for _ in range(n_pairs):
        threads = [threading.Thread(target=one, args=(s, recs)) for s in slots]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    return time.perf_counter() - t0, recs


def summarize(name, wall, recs):
    lat = [r["seconds"] for r in recs]
    toks = [r["completion_tokens"] for r in recs]
    print(f"{name:<34} wall {wall:6.2f}s | {len(recs)} req | "
          f"latency med {statistics.median(lat):5.2f}s "
          f"(min {min(lat):.2f} max {max(lat):.2f}) | "
          f"tokens {min(toks)}-{max(toks)}")
    return wall


print(f"target {BASE} | {PAIRS} pairs/phase | max_tokens={MAX_TOKENS}")
one(SLOT_A)   # warm both slots (first request pays any lazy cost)
one(SLOT_B)

wall_a, recs_a = phase_serial(2 * PAIRS)
summarize(f"A serial, {SLOT_A} only", wall_a, recs_a)

wall_b, recs_b = phase_pairs(PAIRS, [SLOT_A, SLOT_A])
summarize("B 2-at-a-time, same slot", wall_b, recs_b)

wall_c, recs_c = phase_pairs(PAIRS, [SLOT_A, SLOT_B])
summarize("C 2-at-a-time, one per slot", wall_c, recs_c)

print()
print(f"same-slot concurrency speedup : {wall_a / wall_b:.2f}x  (expect ~1.0 — the slot lock serializes)")
print(f"two-slot  concurrency speedup : {wall_a / wall_c:.2f}x  (expect ~2.0 if the NSPs are independent)")
print()
print(json.dumps({"wall_serial": wall_a, "wall_same_slot": wall_b,
                  "wall_two_slot": wall_c,
                  "median_latency_serial": statistics.median([r["seconds"] for r in recs_a]),
                  "median_latency_two_slot": statistics.median([r["seconds"] for r in recs_c])},
                 indent=2))
