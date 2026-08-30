# Platform Notes

*English | [日本語](./PLATFORM_NOTES.ja.md)*

Every measured number in this documentation — throughput, latency, how much
fits on a device, which model combinations co-exist — came from one machine.
This page says what that machine is, which of its properties are **not**
properties of the SoC, and how to find out what yours does instead.

> [!IMPORTANT]
> **We cannot tell you what your device is configured for, and neither can this
> documentation.** Guest memory and storage allocations differ between Qualcomm
> releases and are set when the image is built, so they are also yours to
> change. How many Hexagon NSP cores you may use is gated by your SKU's
> licence. Determine both on your own hardware, or ask Qualcomm. The figures
> below are a reference point, not a default and not a specification.

## The bench these numbers come from

| | |
|---|---|
| SoC | SA8255P |
| Execution environment | A Linux guest under a hypervisor, not bare metal. A second guest (Android) runs on the same SoC |
| RAM visible to the guest | **12.1 GiB** (`MemTotal: 12661020 kB`) |
| CPUs visible to the guest | 8 |
| Storage | 29.4 G filesystem holding the models |
| Hexagon NSP cores in use | **2** — `/dsp/image/dsp/cdsp0` and `cdsp1`, addressed as `device_id` 0 and 1 |
| Subsystem restart (SSR) | **Not available.** `/sys/class/remoteproc` is empty inside the guest, so a wedged cDSP is recovered by power-cycling the board, not by restarting the subsystem |
| QAIRT | 2.49.40.260810 and 2.49.1.260821, `aarch64-oe-linux-gcc11.2` |
| Model bundles | Qwen3 w4a16 context binaries (0.6B, 1.7B, 4B, VL-4B), Gemma-4 E2B |

The Android guest on the same board is a second data point: **6.0 GiB** of RAM,
8 CPUs, QAIRT 2.48.40 `aarch64-android`. See
[Running on Android](./MANUAL.md#running-on-android).

**The memory and storage rows above are not defaults.** This guest was built
with more of both than the images it started from. There is no default we can
quote you: the allocation is a property of the image someone built for that
board, and different Qualcomm releases start from different numbers.

## What varies between platforms, and what it changes here

| What varies | Set by | What it changes in these docs |
|---|---|---|
| **How many NSP cores you may use — 1 or 2** | Your SKU's licence (Qualcomm gates SoC features per SKU) | Everything in [Multi Text Slots](./MANUAL.md#multi-text-slots). With one usable core, a `TEXT_SLOTS` entry at `device_id: 1` has nothing to bind to, the 1.31× concurrency figure does not apply, and two models cannot be made co-resident — one core holds one model ([Loading two models at once](./MANUAL.md#loading-two-models-at-once)) |
| **RAM and storage given to the guest** | Whoever built the image; changeable at build time | Which bundles load at all, whether a second one fits beside the first, and how much of the `err 1002` behaviour you will meet |
| **Guest vs. bare metal** | Board integration | Subsystem restart. Under a hypervisor the guest may not reach `/sys/class/remoteproc`, and then a wedged cDSP needs the board power-cycled |
| **QAIRT version and ABI** | You | Which SDK defects you inherit — see [QAIRT Version Issues](./QAIRT_VERSIONS.md) — and which library directory the server loads from ([Running on Android](./MANUAL.md#running-on-android)) |
| **The model bundle** | Whoever exported it | Context-length variants and AR length set the usable token budget; the quantization recipe decides whether you see defects like the mangled `<tool_call>` marker. Two bundles of the same model at different export settings do not behave alike |

Qualcomm's own documentation treats the second core the same way: selecting
which HTP device executes a model is described as an Auto-platform (SA-series)
facility, reached through an extension library, rather than as something every
target has.

## Finding out what your device does

**NSP cores.** Configure a slot at `device_id: 1` and start the server. It
either comes up and reports both slots ready, or it fails to create the device.
There is no reliable way to ask in advance — in particular, the presence of
`/dsp/image/dsp/cdsp1` tells you what the guest image exposes, not what your
licence permits, so treat it as a hint rather than an answer.

```bash
# Two slots, one per core. If the second cannot bind, startup says so.
# See MANUAL.md, "Multi Text Slots", for the ordering rule.
curl -sS http://<device>:8080/v1/server/status | python3 -m json.tool
```

**Memory and storage.**

```bash
grep -E 'MemTotal|MemAvailable' /proc/meminfo
df -h <the filesystem holding your models>
```

**Subsystem restart.** `ls /sys/class/remoteproc` — empty means you cannot
restart the cDSP from inside the guest, and recovery is a power cycle.

**Anything else — what the defaults are, what your SKU licenses, how the guests
were sized.** Those are Qualcomm's to answer, and the answer depends on your
board and your release. Ask them rather than inferring it from this page.

## Reading the rest of this documentation

These claims are measurements from the bench above. They are honest about that
machine and say nothing about a differently configured one:

- **~1.31× throughput from a second slot**, and the prefill/decode split that
  explains where the rest went — [Multi Text Slots](./MANUAL.md#multi-text-slots)
- **Which model pairs are co-resident**, and that the answer depends on which
  model loads first rather than on the total size —
  [Loading two models at once](./MANUAL.md#loading-two-models-at-once)
- **`err 1002` thresholds**, including that a second core's allocation counts
  against the first
- **`QnnHtp.poll` costing ~260% CPU for no latency gain** —
  [QnnHtp.poll](./MANUAL.md#qnnhtppoll-costs-260-cpu-and-buys-nothing-here)
- **The 36 measured hot-swaps** whose outcome did not follow from the models
  involved — [Switching Models and LoRA](./MANUAL.md#switching-models-and-lora)
- **Per-model behaviour** — the mangled `<tool_call>` marker, the token budgets,
  the template families. These are properties of specific bundles, not of the
  SoC or of the models' names

Where a number matters to your deployment, measure it on your device.
[tests/integration/](../tests/integration/) runs the hardware suite from a host
that can reach the board, and `measure_ttft_tps.py` / `measure_parallelism.py`
beside it are the scripts that produced the throughput tables above.
