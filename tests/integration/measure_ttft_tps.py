#!/usr/bin/env python3
"""Split prefill (TTFT) from decode (TPS), solo vs two slots at once.

    python3 measure_ttft_tps.py [base_url] [reps]

Needs two loaded TEXT_SLOTS; their names are read from /v1/server/status
(ordered by device_id), so any naming works. Results are in docs/MANUAL.md
("Multi Text Slots"): concurrency costs decode ~36% per slot but prefill only
~17%, which is what makes the end-to-end speedup 1.31x on a decode-dominated
workload and 1.64x on a prefill-dominated one.

Prefill is compute-bound, decode streams the whole weight set per token and is
memory-bound. If concurrency costs decode much more than prefill, the ceiling
is bandwidth; if both degrade alike, it is something shared upstream.
"""
import json, statistics, sys, threading, time
import requests

from slot_names import text_slot_names

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.1.2:8080"
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
MAX_TOKENS = 64
PROMPT_WORDS = [8, 96, 224]          # ~= prompt tokens; stays inside the KV budget
SLOT_A, SLOT_B = text_slot_names(BASE, want=2)


def stream_once(slot, words, out=None):
    prompt = ("filler " * words) + "Write one short paragraph about the sea."
    body = {"model": "genie-local", "slot": slot, "max_tokens": MAX_TOKENS,
            "temperature": 0, "enable_thinking": False, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}]}
    t0 = time.perf_counter()
    ttft = None
    usage = None
    with requests.post(f"{BASE}/v1/chat/completions", json=body, stream=True,
                       timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            if ev.get("usage"):
                usage = ev["usage"]
            choices = ev.get("choices") or []
            if ttft is None and choices and (choices[0].get("delta") or {}).get("content"):
                ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    rec = {"slot": slot, "ttft": ttft, "total": total,
           "prompt_tokens": usage["prompt_tokens"] if usage else None,
           "completion_tokens": usage["completion_tokens"] if usage else None}
    rec["decode_tps"] = (rec["completion_tokens"] - 1) / (total - ttft) \
        if ttft and rec["completion_tokens"] and total > ttft else None
    rec["prefill_tps"] = rec["prompt_tokens"] / ttft if ttft and rec["prompt_tokens"] else None
    if out is not None:
        out.append(rec)
    return rec


def med(recs, key):
    vals = [r[key] for r in recs if r.get(key)]
    return statistics.median(vals) if vals else float("nan")


def solo(words, reps):
    recs = []
    for _ in range(reps):
        stream_once(SLOT_A, words, recs)
    return recs


def concurrent(words, reps):
    recs = []
    for _ in range(reps):
        ts = [threading.Thread(target=stream_once, args=(s, words, recs))
              for s in (SLOT_A, SLOT_B)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    return recs


stream_once(SLOT_A, 8)
stream_once(SLOT_B, 8)

print(f"{'prompt':>7} {'mode':<12} {'TTFT':>8} {'prefill':>10} {'decode':>10} {'total':>8}")
print(f"{'tokens':>7} {'':<12} {'(s)':>8} {'(tok/s)':>10} {'(tok/s)':>10} {'(s)':>8}")
rows = []
for words in PROMPT_WORDS:
    s = solo(words, REPS)
    c = concurrent(words, REPS)
    pt = s[0]["prompt_tokens"]
    for name, recs in (("solo", s), ("2 slots", c)):
        print(f"{pt:>7} {name:<12} {med(recs,'ttft'):>8.3f} {med(recs,'prefill_tps'):>10.1f} "
              f"{med(recs,'decode_tps'):>10.1f} {med(recs,'total'):>8.3f}")
    rows.append({"prompt_tokens": pt,
                 "ttft_solo": med(s, "ttft"), "ttft_2slot": med(c, "ttft"),
                 "prefill_solo": med(s, "prefill_tps"), "prefill_2slot": med(c, "prefill_tps"),
                 "decode_solo": med(s, "decode_tps"), "decode_2slot": med(c, "decode_tps")})
    print(f"{'':>7} {'ratio':<12} "
          f"{med(c,'ttft')/med(s,'ttft'):>8.2f}x "
          f"{med(c,'prefill_tps')/med(s,'prefill_tps'):>9.2f}x "
          f"{med(c,'decode_tps')/med(s,'decode_tps'):>9.2f}x")
print()
print(json.dumps(rows, indent=2))
