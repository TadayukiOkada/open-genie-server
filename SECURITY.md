# Security Policy

*English | [日本語](./SECURITY.ja.md)*

## What this server is, and what that means here

open-genie-server is a bench instrument for the Qualcomm Genie C API and for
quantized model bundles — see [What this is for](./README.md#what-this-is-for).
It is **not** a production serving stack, and several things a production
server would be expected to do are deliberately absent. Those are design
decisions, documented as such, and reporting them as vulnerabilities tells us
nothing we do not already say on the front page:

- **There is no authentication and no rate limiting.** Every endpoint is open
  to anyone who can reach the port.
- **`POST /v1/models/switch` will open any path the server process can read.**
  `MODELS_BASE_DIR` shortens paths; it does not confine them. An allow-list was
  considered and rejected — see
  [Where model paths resolve](./docs/MANUAL.md#where-model-paths-resolve).
- **One process, one slot lock, no multi-process scaling.** A slow request
  blocks that slot, by design.
- **Defects in the QAIRT SDK are not this server's to fix.** The stock-library
  slot wedge in particular is reported, not worked around; the per-version
  matrix is in [QAIRT Version Issues](./docs/QAIRT_VERSIONS.md), and defects in
  `libGenie.so` belong to Qualcomm.

**Run it on a network you control.** If you need it reachable from anywhere
else, put authentication and access control in front of it — a reverse proxy
is the usual answer — and treat model directories as trusted input.

## What we do want to hear about

Anything that lets a request do something **beyond** the powers listed above:

- Reading or writing a path that no documented feature would open, or reaching
  outside the process's own model and cache directories through an endpoint
  that is not `/v1/models/switch`.
- Memory corruption, a crash, or a hang reachable from a well-formed request —
  the ctypes layer hands buffers to a C API, so a length or lifetime mistake
  there is a real bug rather than a missing feature.
- Anything that leaks one request's data into another's: prefix-cache
  namespacing across models or LoRA adapters, slot routing, or the streaming
  filters.
- A dependency advisory that actually reaches code we run.

A report is much more useful with the request that triggers it, the
`env_config.json` shape (paths redacted), and the QAIRT version — the offline
test suite runs without an NPU, so a reproduction against
[`tests/fake_genie.py`](./tests/fake_genie.py) is ideal where the defect does
not need the device.

## Reporting

Use GitHub's private vulnerability reporting on this repository: **Security →
Report a vulnerability**. That keeps the report unpublished while it is being
looked at.

For anything that is not sensitive — including the design decisions listed
above, if you think one of them is wrong for a use case we did not consider —
open a normal issue instead. That discussion is more useful in the open.

## Supported versions

Development happens on `master`, and fixes land there. There is no long-term
support branch and no backporting to earlier tags: if you are running an older
version, the answer will be to move to the current one. State the version you
saw the problem on (`genie-server --version`, or `genie_server.__version__`),
since it tells us which commits you have.
