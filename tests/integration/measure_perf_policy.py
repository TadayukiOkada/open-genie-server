#!/usr/bin/env python3
"""Does Genie_PerformancePolicy_t change anything measurable?

    python3 measure_perf_policy.py [base_url] [reps] [policy,policy,...]

Sets each policy through POST /v1/server/performance_policy, confirms it reads
back, then reads the SDK's own KPIs for each request from
GET /v1/server/profile — time to first token, prefill rate and decode rate as
GenieProfile measured them, in microseconds, inside libGenie.

**Requires GENIE_PROFILE=true in env_config.json.** The profiler binds to the
dialog at creation time, so it cannot be switched on at runtime; without it this
script stops rather than quietly reporting something weaker. Host wall time is
recorded alongside, but only to show how much of the request is not the model —
a policy that changes decode by a few percent would be lost in it.

Two outcomes are worth telling apart:

  * the policies separate — decode rate tracks the policy, and the ordering
    follows burst > balanced > power_saver
  * they do not — the numbers sit inside each other's spread, which means the
    call is accepted and has no effect here. That is a real result: the model
    bundles already ask for perf_profile "burst" in their HTP extension config,
    so a runtime policy may have nothing left to change.

Read the spread, not the medians alone. A 5% gap with a 10% spread is nothing.
"""
import json
import statistics
import sys
import time

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.1.2:8080"
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
# Ordered from most to least performance, so a real effect shows as a trend.
DEFAULT_POLICIES = ["burst", "sustained_high_performance", "high_performance",
                    "balanced", "low_balanced", "high_power_saver",
                    "power_saver", "low_power_saver", "extreme_power_saver"]
POLICIES = sys.argv[3].split(",") if len(sys.argv) > 3 else DEFAULT_POLICIES

MAX_TOKENS = 64
PROMPT_WORDS = 96          # long enough for a meaningful prefill number


def one_request():
    """One completion, then the SDK's KPIs for exactly that query."""
    prompt = ("filler " * PROMPT_WORDS) + "Write one short paragraph about the sea."
    body = {"model": "genie-local", "max_tokens": MAX_TOKENS, "temperature": 0,
            "enable_thinking": False,
            "messages": [{"role": "user", "content": prompt}]}
    t0 = time.perf_counter()
    r = requests.post(f"{BASE}/v1/chat/completions", json=body, timeout=600)
    r.raise_for_status()
    host_total = time.perf_counter() - t0

    p = requests.get(f"{BASE}/v1/server/profile", timeout=60)
    if p.status_code == 409:
        sys.exit("GENIE_PROFILE is not enabled on this server. Set "
                 '"GENIE_PROFILE": true in env_config.json and restart — the '
                 "profiler binds to the dialog at creation time, so it cannot "
                 "be turned on at runtime.")
    p.raise_for_status()
    s = p.json().get("summary") or {}
    if not s:
        sys.exit("the profile endpoint returned no summary; cannot measure")
    return {"host_total_s": host_total,
            "ttft_ms": s.get("ttft_ms"),
            "prefill_tps": s.get("prefill_tokens_per_s"),
            "decode_tps": s.get("decode_tokens_per_s")}


def get_policy():
    r = requests.get(f"{BASE}/v1/server/performance_policy",
                     params={"model": "genie-local"}, timeout=30)
    r.raise_for_status()
    return r.json().get("policy")


def set_policy(name):
    r = requests.post(f"{BASE}/v1/server/performance_policy",
                      json={"model": "genie-local", "policy": name}, timeout=60)
    return r.status_code, r.text[:200]


def spread(vals):
    """Median and half-range, so a difference can be read against the noise."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return float("nan"), float("nan")
    return statistics.median(vals), (max(vals) - min(vals)) / 2


def main():
    original = get_policy()
    print(f"base={BASE}  reps={REPS}  policy on entry={original!r}")
    print("KPIs are the SDK's own (GenieProfile), not host timing.\n")
    print(f"{'policy':28s} {'TTFT ms':>9s} {'prefill tok/s':>14s} "
          f"{'decode tok/s':>13s} {'decode ±':>9s} {'host s':>8s}")
    rows = []
    try:
        for name in POLICIES:
            code, body = set_policy(name)
            if code != 200:
                print(f"{name:28s}  set failed: HTTP {code} {body}")
                continue
            got = get_policy()
            if got != name:
                print(f"{name:28s}  read back as {got!r} — skipping")
                continue
            one_request()                     # warm the slot; not recorded
            recs = [one_request() for _ in range(REPS)]
            ttft, _ = spread([r["ttft_ms"] for r in recs])
            pre, _ = spread([r["prefill_tps"] for r in recs])
            dec, dec_sd = spread([r["decode_tps"] for r in recs])
            host, _ = spread([r["host_total_s"] for r in recs])
            rows.append((name, ttft, pre, dec, dec_sd, host))
            print(f"{name:28s} {ttft:9.2f} {pre:14.1f} {dec:13.2f} "
                  f"{dec_sd:8.2f}± {host:8.3f}", flush=True)
    finally:
        if original:
            set_policy(original)
            print(f"\nrestored policy to {original!r}")

    if len(rows) >= 2:
        decs = [r[3] for r in rows]
        worst_spread = max(r[4] for r in rows)
        gap = max(decs) - min(decs)
        print(f"\ndecode tok/s across policies: {min(decs):.2f}-{max(decs):.2f} "
              f"(gap {gap:.2f}); widest within-policy spread ±{worst_spread:.2f}")
        if gap <= 2 * worst_spread:
            print("=> the policies do not separate: the gap between them is within "
                  "the noise of a single one. On this target the call is accepted "
                  "and changes nothing the SDK's own profiler can see.")
        else:
            print("=> the policies separate. Check that the ordering follows the "
                  "list above before calling it causal.")
    json.dump([{"policy": r[0], "ttft_ms": r[1], "prefill_tps": r[2],
                "decode_tps": r[3], "decode_half_range": r[4],
                "host_total_s": r[5]} for r in rows],
              open("/tmp/perf_policy_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
