# VLM (Qwen3-VL) testing example

*English | [日本語](./README.ja.md)*

Steps for trying out the VLM support (`genie_server/vlm.py`, with the ctypes
bindings in `genie_node.py`, the bundle-layout auto-read in `vlm_layout.py`,
and the per-model preprocessing in `vlm_specs/`) against a real Qwen3-VL
bundle. See the "VLM (multimodal) support" section of docs/MANUAL.md for the
full reference.

## Prerequisites

- `numpy`/`Pillow` installed (`pip install .[vlm]` from the repository root, or `pip install numpy pillow`).
- A Qwen3-VL bundle, as it ships from the export tooling — no models ship
  with this repository, see the note at the top of the main README.
  `vlm_layout.py` reads which files to load straight from the bundle's own
  genie-app script (or `metadata.json`), so nothing here needs listing by
  name; see [Bundle layout auto-read](../../docs/MANUAL.md#bundle-layout-auto-read)
  if you want the exact rule.

## 1. Point env_config.json at the VLM bundle

```json
{
  "QAIRT_SDK_ROOT": "/home/root/qairt/2.49.40.260810",
  "HEXAGON_VERSION": "v73",
  "VLM_SLOTS": [
    {
      "name": "vision",
      "device_id": 0,
      "model_root": "/path/to/qwen3_vl_4b_instruct-genie-w4a16-qualcomm_sa8775p",
      "max_tokens": 1024
    }
  ]
}
```

`VLM_SLOTS` is a separate, parallel setting to `TEXT_SLOTS`; a chat request
carrying an `image_url` content part is routed to a VLM slot automatically.
`spec` (the VLM family) is left out here — it auto-detects from the bundle's
own tokenizer and node configs. Pass `"spec": "qwen3_vl"` explicitly if you
ever need to force it.

> **Leaving `TEXT_SLOTS` out keeps this sample to one thing.** A text model and
> a VLM *can* be resident together on the SA8255P this was tested on, but only
> when the text bundle is exported at a **single context length** and the two
> slots sit on **different `device_id`**. With a multi-context-length text
> bundle the second slot fails with `err 1002` whichever loads first. For that
> configuration start from
> [examples/config/env_config.text-vlm.sample.json](../config/env_config.text-vlm.sample.json)
> instead, and read
> [Loading two models at once](../../docs/MANUAL.md#loading-two-models-at-once)
> — the pairs that fit have to be measured, not calculated.

`max_tokens` bounds generation for the whole slot. The request's own
`max_tokens` cannot reach this path — GenieNode reads it once, at node creation
— so this is the only lever, and `0` means uncapped.

## 2. Start the server

```bash
python3 genie-server.py
```

Confirm the startup log prints `VLM slot 'vision' ready: ...`.

## 3. Base64-encode a test image and send it

```bash
python3 - "http://192.168.1.2:8080" photo.jpg <<'PYEOF'
import base64, json, sys, requests

base_url, image = sys.argv[1], sys.argv[2]
with open(image, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

resp = requests.post(f"{base_url}/v1/chat/completions", json={
    # Route by slot name, or by the model directory name in "model". A bare
    # "model": "vision" only lands on the right slot because a single VLM slot is
    # also the fallback.
    "slot": "vision",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": "Describe this image in one sentence."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]},
    ],
    "stream": False,
})
print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
PYEOF
```

`stream: true` streams the same answer over SSE — verified byte-identical to the
non-streaming reply for the same request.

## Things to check

- Whether the response actually matches the image content. This is the real test
  of the normalization constants and patch ordering — see the comments on
  `_qwen3vl_normalize` / `_qwen3vl_patchify` in `genie_server/vlm_specs/qwen3_vl.py`. A
  wrong patch order does not error; it produces confident nonsense.
- That several images in one message do not break anything.
- That `stream: true` and the non-streaming call return the same text.
- What happens when a client hangs up mid-stream: the generation **does not
  stop**. GenieNode exposes no abort call, so the slot stays busy until the
  answer finishes on its own, and the next request waits. That is expected, not
  a bug — the text path aborts, this one cannot.

Automated coverage for the last three lives in the integration suite as
`V01`–`V05` (`tests/integration/`), gated on `vlm.enabled` in `test_config.json`.
