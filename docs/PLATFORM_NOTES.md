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
| QAIRT | 2.49.40.260810, 2.49.1.260821 and 2.50.0.260828, `aarch64-oe-linux-gcc11.2` |
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
| **The SMMU page-table pool behind DSP mappings** | Board integration; on this bench it belongs to the hypervisor and is 16 MB, and it is not visible from the guest | The actual `err 1002` ceiling, and therefore how many slots co-reside — see [Where the `err 1002` budget actually lives](#where-the-err-1002-budget-actually-lives) |
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

## Where the `err 1002` budget actually lives

[Loading two models at once](./MANUAL.md#loading-two-models-at-once) describes
what `err 1002` does to you. This section says what it *is* on this bench, and
why almost none of it is a property of the SoC.

**The failure is not a DSP-side allocation.** Turning the SDK's own logging on
(`GENIE_LOG_LEVEL: "info"`, see [Configuration](./MANUAL.md#configuration-env_configjson)) puts
three lines in front of the one you normally see:

```
<E> fastrpc memory map for fd: 129 with length: 312475648 failed with error: 0x1
<E> Mapping buffer fd 129 to FastRpc failed on domain 3
<E> Failed to map weights buffer on host, err 3
<E> Failed to map shared weights for contextId 9 on deviceId 0, coreId 0, pdId 0, err: 1002
```

A context's shared-weights buffer — 298 MiB for the bundle in [the slot sweep](./MANUAL.md#how-many-slots-of-one-small-model-fit-and-what-each-costs) —
could not be mapped into a DSP protection domain **from the host side**.
`err 1002` is QNN's generic memory-allocation code, so it reads like the DSP ran
out of memory; the mapping is what failed. The guest kernel says the same thing
in its own words, and this is the layer where a hypervisor shows up:

```
virtio-mmio ...: invalid smmu da 0
virtio-mmio ...: can't get map da
adsprpc: genie-server: hfastrpc_mem_map: failed to map fd 129, len 0x12a00000, ... err -14
```

**On this bench the exhausted resource is a fixed SMMU page-table pool that
belongs to the hypervisor**, 16,384 KB, and it is neither DSP memory nor guest
memory — the system had 10.9 GB free at the moment of the failure.

**This consumption cannot be seen from the guest.** Nothing in the guest's
`/proc` accounts for it, which is why the manual spent so long unable to
predict `err 1002` from anything measurable inside the Linux side. It has to be
read from the hypervisor — QNX on this board — where the pool and its usage are
reported. Measured there:

| State | QNN contexts | Pool used |
|---|---|---|
| No model loaded, straight after a power cycle | 0 | **3,332 KB** |
| Four slots | 8 | **11,524 KB** |
| Six slots | 12 | **15,620 KB** |

**1,024 KB per QNN context, exactly** — which is what a 4 KB-granule page table
for the ~462 MB a context maps comes to, rounded up to the pool's 256 KB
allocation unit. That makes the cap arithmetic rather than folklore:
`(16,384 − 3,332) / 1,024 = 12` contexts, and the bundle in [the slot sweep](./MANUAL.md#how-many-slots-of-one-small-model-fit-and-what-each-costs)
creates two per slot, so **six slots fit and a seventh needs 2,048 KB against
the 764 KB left**. Both numbers match what that sweep measured from the outside.

A bundle with more context-length variants creates more QNN contexts, and each
one is rounded up separately, so **the cap is per context, not per byte**. That
is also why two bundles of the same model that allocate byte-identical amounts
do not leave the same room behind them.

**Which protection domain a context lands on depends only on how many contexts
the process has already created** — not on which core the slot is on. Across
seven configurations the assignment was identical: contexts 1–7 to pd 0, the
8th to pd 2, the 9th back to pd 0, the 10th onwards to pd 2. So whichever core
loads first collects contexts 1–7 in one domain, and its ninth context comes
back to a domain that is already full. **That is the mechanism behind the
load-order rule** in the manual: the same six slots pass or fail depending on
the order alone. Four slots on one core plus two on the other passes with seven
contexts in one domain, because the ninth goes to the other core.

> [!WARNING]
> **The pool is a high-water mark within one boot and does not come back.**
> Stopping the server does not lower it; reloading does not raise it; only a
> power cycle returns it to baseline. This is the reason the manual tells you to
> judge co-residency on the first startup after a power cycle — and the reason a
> **failed** startup is worse than it looks. A startup that fails with
> `err 1002` does not return its page tables either, so repeated attempts walk
> the pool down until nothing can be created at all and startup fails with
> `Failed to create device: 14001`. At that point the server process may not
> exit even under `SIGKILL`, a guest reboot cannot complete because it still
> holds a filesystem, and recovery is a board power cycle.

**None of this transfers to a differently integrated board.** The page-table
pool is the hypervisor's, its size was set when that image was built, and on a
platform without one the accounting will be somewhere else entirely. What
should transfer is the shape of the problem: the budget `err 1002` runs into is
a *mapping* budget, it is charged per QNN context, and it is not visible from
the guest. Where to look on your own platform is Qualcomm's to answer.

## Reading the rest of this documentation

These claims are measurements from the bench above. They are honest about that
machine and say nothing about a differently configured one:

- **~1.31× throughput from a second slot**, and the prefill/decode split that
  explains where the rest went — [Multi Text Slots](./MANUAL.md#multi-text-slots)
- **Which model pairs are co-resident**, and that the answer depends on which
  model loads first rather than on the total size —
  [Loading two models at once](./MANUAL.md#loading-two-models-at-once)
- **`err 1002` thresholds**, including that a second core's allocation counts
  against the first — and [where that budget actually
  lives](#where-the-err-1002-budget-actually-lives), which is a property of this
  board's integration and not of the SoC
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
