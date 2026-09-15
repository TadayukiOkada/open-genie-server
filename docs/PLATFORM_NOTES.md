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
| Storage | 29.4 G filesystem holding the models, mounted at both `/home` and `/data`. The root filesystem is read-only — see [Installing on the device](#installing-on-the-device) |
| System Python | 3.10.14, without pip |
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
| **Which filesystems are writable, and what the system Python ships with** | Whoever built the image | Where the server and its dependencies can be installed — [Installing on the device](#installing-on-the-device) |
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

**Where you can install.**

```bash
mount | grep -E ' on / | on /home | on /data '   # "ro" on / means the system site-packages is read-only
python3 -m pip --version                         # "No module named pip" means use a venv
```

**Subsystem restart.** `ls /sys/class/remoteproc` — empty means you cannot
restart the cDSP from inside the guest, and recovery is a power cycle.

**Anything else — what the defaults are, what your SKU licenses, how the guests
were sized.** Those are Qualcomm's to answer, and the answer depends on your
board and your release. Ask them rather than inferring it from this page.

## Installing on the device

**On this bench, the server and its dependencies have to go into a virtualenv
under `/home/root`.** That is not a style preference. The guest's root
filesystem is mounted read-only and its system `python3` has no pip, so a venv
on the writable filesystem is the one place pip can run at all.

What the guest can write to:

| Path | Writable | |
|---|---|---|
| `/`, including `/usr` and the system `site-packages` | **No** | ext4 mounted `ro`, 1.5 G |
| `/home` and `/data` | **Yes** | **One** 29.4 G ext4 filesystem mounted at both, so `/home/root` and `/data/root` are the same directory. The models, the SDK and the venv all live here |
| `/tmp`, `/var`, `/run` | Until the next boot | tmpfs: held in RAM and emptied by a reboot — and a wedged cDSP is recovered by power-cycling the board. Not a place for a venv |
| `/persist` | Yes | The image's own settings (it backs `/etc/bluetooth`, `/etc/usb` and `/etc/build.prop`). Not for your files |

The system interpreter is Python 3.10.14 at `/usr/bin/python3`.
`python3 -m pip` fails with `No module named pip`, but `venv` and `ensurepip`
are there, and that is all a venv needs:

```bash
cd /home/root
python3 -m venv .venv                                    # ensurepip puts pip inside the venv
.venv/bin/pip install 'open-genie-server[logprobs,vlm]'  # or '.[logprobs,vlm]' from a checkout
.venv/bin/genie-server --config env_config.json
```

Running from a checkout without installing the package works the same way,
with the venv's interpreter: `.venv/bin/python3 genie-server.py`. Call the
venv's binaries by path rather than activating it — each `adb shell` invocation
starts a new shell, so an activation does not carry over to the next command.

**If the guest cannot reach PyPI.** This bench's guest reaches the network
through its host, but that is a property of the bench. Without a route,
download the wheels on any machine that has one — with `--platform`, pip picks
wheels for the platform you name rather than the machine it runs on — copy the
directory over, and install from it:

```bash
# On a machine with network access
pip download 'open-genie-server[logprobs,vlm]' -d wheels \
    --python-version 3.10 --only-binary=:all: \
    --platform manylinux2014_aarch64 --platform manylinux_2_28_aarch64
# On the device, after copying wheels/ to /home/root/wheels
.venv/bin/pip install --no-index --find-links=/home/root/wheels 'open-genie-server[logprobs,vlm]'
```

Match `--python-version` to the guest's `python3 --version`, and list the
manylinux tags its glibc accepts (`getconf GNU_LIBC_VERSION`; 2.35 here, so
both of the above). **With `manylinux2014_aarch64` alone the download still
succeeds, but quietly resolves to older releases** — when this was written,
pillow 12.2.0 instead of 12.3.0, and a `huggingface_hub` (a dependency of
`tokenizers`) old enough not to need `hf_xet`, which ships only
`manylinux_2_28` wheels.

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

**2,048 KB per slot of the bundle in [the slot sweep](./MANUAL.md#how-many-slots-of-one-small-model-fit-and-what-each-costs)**,
which makes the cap arithmetic rather than folklore: `(16,384 − 3,332) / 2,048
= 6` slots, and **a seventh needs 2,048 KB against the 764 KB left**. Both
numbers match what that sweep measured from the outside.

> [!NOTE]
> *(Corrected.)* An earlier revision said this pool is charged **1,024 KB per
> QNN context**, so that the cap is per context rather than per byte. That held
> for this one bundle only. The charge follows **the address space a slot
> maps**: a 4 KB-granule page table takes 4 KB for every 2 MiB of DSP address
> space mapped, and the pool hands it out in 256 KB units. The
> multi-context-length export of the same 0.6B creates the same two QNN
> contexts and costs **4,864–5,120 KB**, because its six graphs map more —
> their I/O tensors, spill-fill and op data, and, while a context loads, one
> I/O buffer as large as all of them together (732 MiB for its second
> context). Counting 2 MiB regions in the address-space map below gives the
> pool's usage to its 256 KB unit wherever the map could be read.

**There are two limits, not one.** The pool above caps how much can be mapped
in total. Separately, **each DSP protection domain has its own 4 GiB address
space**, and every mapping has to find a contiguous run in it:

- A mapping starts on a multiple of the largest power of two not above its
  size — a 298 MiB weights buffer on a 256 MiB boundary, a 732 MiB buffer on a
  512 MiB one — at the lowest address where it fits.
- That fragments. Two multi-context-length 0.6B slots on one core fail with
  1,347 MiB of that domain's address space free, because no hole runs 732 MiB
  from a 512 MiB boundary: the first slot's load-time buffer was freed, and
  that slot's own 262 MiB I/O buffer took its place.
- The pool had room at that moment. The hypervisor reports running out of
  address space and running out of pool differently; the guest sees the same
  `err 1002` for both.

**Which protection domain a context lands on is decided by a budget, not by a
count.** Contexts go to pd 0, pd 2, pd 4 … — each to the first domain whose
budget still has room. The budget is **one per domain number, shared by both
cores**, and a context is charged its shared weights plus its graphs' I/O
tensors, spill-fill buffers and op data, as the context binary's metadata
lists them. Across 24 start-ups every placement fits a budget between
3,403,453,184 and 3,503,631,616 bytes, about 3.4–3.5 GB. For the CL1024 bundle
alone this comes out as contexts 1–7 to pd 0, the 8th to pd 2, the 9th back to
pd 0 and the 10th onwards to pd 2, which an earlier revision described as the
rule itself. *(Corrected: that sequence breaks as soon as bundles are mixed —
with two multi-context-length slots and one CL1024 slot, the 5th context goes
to pd 2.)*

**That is the mechanism behind the load-order rule** in the manual. With five
CL1024 slots on one core listed first, contexts 1–7 fill that core's pd 0, and
the 9th comes back to pd 0 by budget — but its 298 MiB weights find no free
256 MiB boundary left in that domain's address space, while the pool still has
4.6 MB free. List the other core's slot first and its two contexts take part of
pd 0's budget, so the first core's pd 0 holds five contexts instead of seven and
the 9th fits. Four slots on one core plus two on the other passes because the
9th context goes to the other core. And since the budget is shared across
cores, **a slot on one core moves contexts on the other**: two
multi-context-length 0.6B slots that cannot share a core on their own do load
there once a CL1024 slot on the other core has loaded first, because it pushes
the second slot's larger context to pd 2. The same three slots on one core —
two CL1024 and one multi-context-length — load in one order and fail in
another. All four of those outcomes were predicted from these rules before they
were measured.

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
should transfer is the shape of the problem: the budgets `err 1002` runs into
are *mapping* budgets — page tables and the DSP's address space — they follow
how much is mapped rather than how many contexts there are, and they are not
visible from the guest. Where to look on your own platform is Qualcomm's to answer.

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
