# Grammar-constrained decoding example

*English | [日本語](./README.ja.md)*

A pair of `genie_config.json` (with `dialog.context.grammar` set) and `grammar_schema.txt` (a JSON Schema definition).

## Usage

1. Copy the following files into this directory from a real model bundle (e.g.
   `MODELS/qwen3_vl_4b_instruct-genie-w4a16-qualcomm_sa8775p/`) — or instead
   rewrite the paths in `genie_config.json` to point directly at the original
   bundle directory:
   - `tokenizer.json`
   - `htp_backend_ext_config.json`
   - `part1_of_4.bin` through `part4_of_4.bin` (the number of shards varies by model)
   - `embedding_weights.raw`
2. Point `env_config.json`'s `TEXT_SLOTS[].model_root` at this directory.
3. Start `genie-server.py` and send a normal chat completion request; confirm the
   output is forced into the schema in `grammar_schema.txt`
   (`{"answer": string, "confidence": number}`).

## Notes

- `dialog.context.grammar` is the slot's default (see "Grammar-constrained
  decoding" in docs/MANUAL.md). Every request to this slot is constrained by this
  schema unless it brings its own grammar in `response_format` or
  `structured_outputs`, which needs QAIRT 2.51.0 or later. The next request
  without one gets this schema back. Before 2.51.0 the schema is fixed per model.
- You do not need this example for a per-request grammar: on QAIRT 2.51.0 any
  bundle takes one, with a `libGenie.so` that has the grammar backend.
- On a stock library the response ends with `<|im_end|>` as text, an SDK defect
  the server does not hide (D4 in docs/QAIRT_VERSIONS.md).
- `backend` must be `"xgrammar"`. `file` points at a plain text file containing
  the schema definition itself (`grammar_schema.txt`) — the filename is
  arbitrary and doesn't need a `.json` extension.
