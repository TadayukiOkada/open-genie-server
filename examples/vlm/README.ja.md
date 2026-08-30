# VLM(Qwen3-VL)動作確認の例

*[English](./README.md) | 日本語*

VLM 対応(`genie_server/vlm.py`。ctypes バインディングは `genie_node.py`、
モデルごとの前処理は `vlm_specs.py`)を実際の Qwen3-VL バンドルで試す手順です。
詳細なリファレンスは docs/MANUAL.ja.md の「VLM(マルチモーダル)対応」節を参照してください。

## 前提

- `numpy`/`Pillow` がインストール済みであること(リポジトリ直下で `pip install .[vlm]`、または `pip install numpy pillow`)。
- Qwen3-VL バンドル一式(`img-enc-htp.json`、`text-encoder.json`、
  `text-generator.json`、`vision_encoder.bin`、`part*_of_4.bin`、
  `embedding_weights.raw`、`tokenizer.json`、`sample_inputs/*.raw`)。
  **モデルはこのリポジトリには含まれていません** — メインREADME冒頭の注記を参照してください。

## 1. env_config.json を VLM バンドルに向ける

```json
{
  "QAIRT_SDK_ROOT": "/home/root/qairt/2.49.40.260810",
  "HEXAGON_VERSION": "v73",
  "VLM_SLOTS": [
    {
      "name": "vision",
      "device_id": 0,
      "model_root": "/path/to/qwen3_vl_4b_instruct-genie-w4a16-qualcomm_sa8775p",
      "spec": "qwen3_vl",
      "max_tokens": 1024
    }
  ]
}
```

`VLM_SLOTS` は `TEXT_SLOTS` と並列の独立した設定です。`image_url` パートを含む
チャットリクエストは自動的に VLM スロットへ振り分けられます。

> **`TEXT_SLOTS` を書いていないのは、このサンプルを1つの話題に絞るためです。**
> 検証した SA8255P では、テキストモデルと VLM を**同時に常駐させられます**。
> ただし**テキスト側が単一コンテキスト長でエクスポートされており、かつ2つのスロットが
> 別の `device_id` にあるとき**に限ります。複数コンテキスト長のテキストバンドルでは、
> どちらを先にロードしても2本目が `err 1002` で失敗します。
> その構成を組むなら
> [examples/config/env_config.text-vlm.sample.json](../config/env_config.text-vlm.sample.json)
> から始め、
> [2つのモデルを同時にロードする](../../docs/MANUAL.ja.md#2つのモデルを同時にロードする)
> を読んでください — **載る組み合わせは計算ではなく実測で決めるしかありません。**

`max_tokens` はスロット全体の生成長を制限します。**リクエスト側の `max_tokens` は
この経路に届きません** — GenieNode がノード生成時に一度だけ読むためです。したがって
これが唯一のレバーで、`0` は無制限を意味します。

## 2. サーバを起動する

```bash
python3 genie-server.py
```

起動ログに `VLM slot 'vision' ready: ...` が出ることを確認してください。

## 3. テスト画像を base64 で送る

```bash
python3 - "http://192.168.1.2:8080" photo.jpg <<'PYEOF'
import base64, json, sys, requests

base_url, image = sys.argv[1], sys.argv[2]
with open(image, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

resp = requests.post(f"{base_url}/v1/chat/completions", json={
    # スロット名で指定するか、"model" にモデルディレクトリ名を入れる。
    # "model": "vision" が当たるのは、VLMスロットが1つだけでフォールバック先に
    # なっているからにすぎない。
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

`stream: true` にすると同じ応答が SSE で流れます — 同一リクエストに対して
非ストリーミングの応答と**バイト単位で一致する**ことを確認済みです。

## 確認すべきこと

- **応答が実際に画像の内容と合っているか。** これが正規化定数とパッチ順序の
  本当のテストです(`genie_server/vlm_specs.py` の `_qwen3vl_normalize` /
  `_qwen3vl_patchify` のコメント参照)。**パッチ順序が誤っていてもエラーにはならず、
  自信たっぷりの出鱈目が返ります。**
- 1つのメッセージに複数画像を入れても壊れないこと。
- `stream: true` と非ストリーミングで同じテキストが返ること。
- **ストリーミング中にクライアントが切断したときの挙動**: 生成は**止まりません**。
  GenieNode に abort API が無いため、応答が自然に終わるまでスロットは占有され、
  次のリクエストは待たされます。これは仕様であり不具合ではありません —
  テキスト経路は中断できますが、この経路はできません。

後ろ3つは統合テストの `V01`–`V05`(`tests/integration/`)で自動化されています。
`test_config.json` の `vlm.enabled` で有効化してください。
