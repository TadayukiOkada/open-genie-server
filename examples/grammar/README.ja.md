# Grammar制約デコーディングの動作確認用サンプル

*[English](./README.md) | 日本語*

`genie_config.json`(dialog.context.grammar付き)と`grammar_schema.txt`(JSON Schema定義)のペア。

## 使い方

1. 実際のモデルバンドル(例: `MODELS/qwen3_vl_4b_instruct-genie-w4a16-qualcomm_sa8775p/`)から
   以下のファイルをこのディレクトリにコピーする(または`genie_config.json`側のパスを
   書き換えて元のバンドルディレクトリを直接指す):
   - `tokenizer.json`
   - `htp_backend_ext_config.json`
   - `part1_of_4.bin` 〜 `part4_of_4.bin`(モデルによってシャード数は異なる)
   - `embedding_weights.raw`
2. `env_config.json`の`TEXT_SLOTS[].model_root`をこのディレクトリに向ける。
3. `genie-server.py`を起動し、通常のchat completionsを叩いて出力が`grammar_schema.txt`の
   スキーマ(`{"answer": string, "confidence": number}`)に強制されることを確認する。

## 注意

- `dialog.context.grammar`はスロット/モデル単位で固定(docs/MANUAL.ja.md「Grammar制約デコーディング」参照)。
  このスロットへのリクエストは常にこのスキーマに制約される。
- `backend`は`"xgrammar"`固定。`file`はスキーマ定義そのものを書いたテキストファイル
  (`grammar_schema.txt`)を指す — ファイル名は任意、拡張子`.json`である必要もない。
