# open-genie-server に対して lm_eval を実行する

*[English](./README.md) | 日本語*

`lm_eval` の `local-completions` バックエンドは `/v1/completions` を叩きます。
多肢選択タスク(hellaswag、arc、mmlu 等)は各選択肢を `echo` + `logprobs` の
リクエストでスコアリングし(=サーバのプロンプトスコアリングモード)、
生成タスク(gsm8k 等)は通常の completions を使います。

| ファイル | 内容 |
|---|---|
| `run_lm_eval.sh` | ラッパー。プロンプトスコアリングのスイッチをONにして(終了時に元へ戻す)、本サーバに必要な model_args を付けて `lm_eval` を呼ぶ |
| `compare_runs.py` | 2つの `--log_samples` 実行を問題ごとに比較する(例: ボード vs 同一モデルのfp32) |

## 1. lm_eval を入れる

**`[api]` extra が必須**です(無いと `local-completions` が `tenacity` 不足で落ちる)。
CPU版torchにすればインストールは約1GBで済みます:

```bash
python3 -m venv /tmp/lmeval
/tmp/lmeval/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
/tmp/lmeval/bin/pip install "lm_eval[api]" transformers
```

実行するのはボード自身ではなく、**ボードにHTTPで到達できるホスト**です。

## 2. 評価したいモデルをロードする

```bash
curl -X POST http://192.168.1.2:8080/v1/models/switch \
  -H 'Content-Type: application/json' \
  -d '{"slot": "chat", "model_dir": "/home/root/models/qwen3_4b_instruct_2507-genie-w4a16-qualcomm_sa8775p", "unload_first": true}'
```

## 3. 実行する

```bash
LM_EVAL=/tmp/lmeval/bin/lm_eval \
  ./run_lm_eval.sh http://192.168.1.2:8080 Qwen/Qwen3-4B-Instruct-2507 hellaswag 100
```

**トークナイザ引数はボードにロードされているモデルと同一のもの**を指定すること。
lm_eval はローカルでトークナイズしてリクエストのうちどこまでがコンテキストかを求め、
返ってきた logprobs をその境界で切ります。違うトークナイザだと**全スコアが黙ってずれます**。
`model=genie-local` はHFのリポジトリIDではないので、lm_eval は自動では決められません。

結果は `./lm_eval_out`(`OUT=` で変更可)に、`--log_samples` の問題ごとの詳細も併せて出ます。

## 遅いことを前提にすること

プロンプトスコアリングは**プロンプト全体を decode 速度で流します**(だから loglikelihood が
厳密になる)。さらにリクエストはスロット単位で直列化されます。SA8255P +
`qwen3_4b_instruct_2507`(w4a16)での実測:

| | |
|---|---|
| 1リクエスト | 約3.4秒(hellaswag程度のプロンプト長) |
| hellaswag `--limit 100` | 400リクエスト、約23分 |

Gemma 4 E2B の QAT のエクスポート(w4a16)はこれより速く、1リクエスト約2.2秒、hellaswag `--limit 100` で約15分です。

実質 `--limit` は必須です。hellaswag 全問は 40,168 リクエスト(約38時間)になります。
`num_concurrent=1` のままにしてください — スロットはどのみち1リクエストずつしか処理しないので、
並列度を上げてもキューが伸びるだけです。

## 量子化前のモデルとの比較

```bash
# 同じタスク・同じ --limit で、fp32 モデルをCPU実行
/tmp/lmeval/bin/lm_eval --model hf --tasks hellaswag --limit 100 --batch_size 4 \
    --device cpu --output_path ./lm_eval_hf --log_samples \
    --model_args pretrained=Qwen/Qwen3-4B-Instruct-2507,dtype=float32

/tmp/lmeval/bin/python compare_runs.py \
    ./lm_eval_out/genie-local/samples_hellaswag_*.jsonl \
    ./lm_eval_hf/Qwen__Qwen3-4B-Instruct-2507/samples_hellaswag_*.jsonl
```

`compare_runs.py` は両者の問題ごとの argmax と一致率を出します。**意味があるのは一致率**です —
4bit重みでは生の loglikelihood はずれますが、多肢選択タスクが評価しているのは順位だからです。

### BOS を揃える

dialog の context に `bos-token` を持つバンドル(Gemma 4 のものはそう)では、SDK がすべてのクエリの
先頭にそのトークンを付けます。プロンプトスコアリングも同じです。サーバのトークナイザは BOS を付けないので、
返る logprobs はプロンプトの最初のトークンから始まり、lm_eval の切り出し位置と合います。
**ボード側で設定するものはありません。**

参照側もその BOS を見る必要がありますが、`--model hf` が付けるとは限りません。Gemma 4 の HF トークナイザは
既定で BOS を付けず、lm_eval(0.4.13)も強制しません。`add_bos_token=True` を渡してください:

```bash
--model_args pretrained=google/gemma-4-E2B-it,dtype=float32,add_bos_token=True
```

付けないと参照は BOS の無いテキストを採点し、**ボードが自分の fp32 モデルに勝っているように見えます** —
Gemma 4 E2B では、hellaswag の最初の問題の選択肢 4 つが 9〜52 nats 低く出ました。Qwen3 のトークナイザには
BOS トークンが無いので、参照に何もいりません。他のモデルでの判断: トークナイザに `bos_token_id` があるのに
`tokenizer("hello").input_ids` がそれで始まらないなら、このフラグを渡します。

### QAT のバンドルには参照が 2 つ要る

量子化を意識した学習(QAT)のチェックポイントから量子化したバンドルは、fp32 モデルではなく
そのチェックポイントから作られており、両者は生の loglikelihood で大きく離れることがあります。
Gemma 4 E2B、hellaswag `--limit 100`、どちらも CPU での実測:
`google/gemma-4-E2B-it-qat-mobile-transformers` は `google/gemma-4-E2B-it` より選択肢ごとに 20 nats 高く、
同じ選択肢を選んだのは 100 問中 88 問。その QAT チェックポイントから作った w4a16 のバンドル 2 本は
SA8255P 上でチェックポイントから 0.4 nats 以内(相関 0.998)でした。**元の fp32 だけと比べると、
その 20 nats がボードのせいに見えます。**

QAT チェックポイントを 2 つ目の参照として回してください。transformers が専用の量子化器で読み込みます
(5.17.0 で確認):

```bash
--model_args pretrained=google/gemma-4-E2B-it-qat-mobile-transformers,dtype=float32,add_bos_token=True
```

CPU では fp32 モデルより遅く、上の実行で約 12 分(fp32 は約 4 分)かかります。量子化された層を Python で展開するためです。

## 結果の読み方

- **ボードのスコアを公開されている fp32 のリーダーボード値と直接比較しない。**
  比較対象は「同じモデル・同じタスク・同じ `--limit` の自分の fp32 実行」です。
- **`acc_norm` より `acc` を優先する。** 長さ正規化は短い continuation で
  per-token の差を増幅します。
- `--limit 100` なら標準誤差は概ね ±0.05 なので、**10ポイント未満の差は何の証拠にもなりません**。

スコアリングモードのゲート・コスト・数値の検証状況は
[docs/MANUAL.ja.md](../../docs/MANUAL.ja.md#logprobs) を参照してください。
