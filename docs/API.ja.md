# APIリファレンス

*[English](./API.md) | 日本語*

open-genie-server が提供する全エンドポイントを、用途ごとにまとめたもの。概念と設定は
[MANUAL.ja.md](./MANUAL.ja.md) にあり、本ファイルは契約(インタフェース)を扱う。

以下 `base_url = http://<ホスト>:8080`。

| グループ | エンドポイント |
|---|---|
| [OpenAI互換](#openai互換エンドポイント) | `GET /v1/models` · `GET /v1/models/{id}` · `POST /v1/completions` · `POST /v1/chat/completions` |
| [サーバの状態と制御](#サーバの状態と制御) | `GET /v1/server/status` · `GET /v1/server/idle` · `GET\|POST /v1/server/performance_policy` · `GET\|POST /v1/server/prompt_logprobs` · `GET /v1/server/profile` |
| [Prefix KVキャッシュ](#prefix-kvキャッシュ) | `GET /v1/prefix/cache` · `DELETE /v1/prefix/cache/{key}` · `POST /v1/prefix/warmup` |
| [モデルとLoRA](#モデルとlora) | `POST /v1/models/switch` · `POST /v1/lora/apply` · `POST /v1/lora/strength` · `POST /v1/lora/release` · `GET /v1/lora/current` |
| [エラー](#エラー形式) | 全ての失敗が使う共通エンベロープ |

全体に共通する2点:

- OpenAI互換エンドポイントは **`/v1` プレフィックス無しでも登録されている**
  (`/models`、`/completions`、`/chat/completions`)。base URL の設定を誤った
  クライアント向け。
- `GET /health` と `GET /v1/health` は `{"status": "ok"}` を返す
  (livenessプローブ。vLLM互換の形状)。

最初のグループ以外は本サーバ独自のものであり、OpenAIクライアントが目にすることはない。

## OpenAI互換エンドポイント

### GET /v1/models

利用可能なモデル一覧。固定の `genie-local`(lm_eval互換用プレースホルダ)と、**全スロット**の実際にロード中のモデルID(`active_model_id`)を返します。

```bash
curl $base_url/v1/models
```

```json
{"object": "list", "data": [{"id": "genie-local", ...}, {"id": "tool-model", ...}, {"id": "general", ...}]}
```

### GET /v1/models/{model_id}

任意の `model_id` を受け付け、そのままモデルオブジェクトとして返します(検証用の固定許可リストは存在しません)。

### POST /v1/completions

生テキスト補完(`lm_eval` の `local-completions` バックエンド用)。チャットテンプレートやprefixキャッシュは適用されません。`model` でスロットを選択します。

| フィールド | 型 | 説明 |
|---|---|---|
| `prompt` | string \| string[] \| int[] \| int[][] | 必須。文字列配列の場合は各プロンプトを**順番に**処理し、プロンプトごとに1つのchoiceを返す(`index` 0..n-1。非ストリーミングのみ)。トークンID配列(`lm_eval`の`local-completions`が`tokenizer_backend=huggingface`で送る形式)はスロット自身のトークナイザでデコードされる。 |
| `model` | string | どのスロットに送るか選択(`SlotManager.select`)。一致するスロットが無ければプライマリスロット。**レスポンスにはこの文字列をechoしません** — 応答本体もストリーミングチャンクも、**実際に応答したモデルのID**(選択されたスロットが現在保持しているもの)を返します。エイリアス(`genie-local`、lm_evalの固定プレースホルダ等)で振り分けた場合と、ホットスワップ後に `env_config.json` の記載と異なるモデルが載っている場合に、両者は食い違います。 |
| `slot` | string | 任意。スロット**名**を明示指定(例: `"chat"`) — `model`より優先されます。同じモデルディレクトリを複数スロットにロードしている場合、`model`だけでは区別できないために必要([制約事項](./MANUAL.ja.md#制約事項)参照)。存在しない名前は`404`。 |
| `stream` | bool | 既定 `false`。 |
| `max_completion_tokens` / `max_tokens` | int | 前者が優先(OpenAIの非推奨方針に準拠)。どちらも未指定の場合、`genie_config.json`の`dialog.context.size`からプロンプトのトークン数を引いた値(=残りコンテキスト容量)がデフォルトになる — Qualcomm自身のqai-appbuilderリファレンスサーバと同じ挙動。`DEFAULT_MAX_TOKENS`(`env_config.json`)を正の値に設定した場合は、その値と残りコンテキスト容量のうち小さい方が使われる。詳細は[トラブルシューティング](./MANUAL.ja.md#トラブルシューティング)参照。 |
| `stop` | string \| string[] | 停止シーケンス。マッチングはSDK内で行われ、ストリーミング中は部分一致テキストをホールドバックし、一致した停止シーケンス自体は出力からトリムされる(OpenAIセマンティクス)。 |
| `temperature` / `top_p` / `top_k` | number | サンプリングパラメータ。リクエストごとに再適用され、省略したパラメータは**モデル自身の`genie_config.json`既定値**に戻る(直前のリクエストの設定が漏れない)。`temperature=0` で貪欲デコード(SDKのランタイムサンプラ設定はtemp=0を受け付けないため、`top-k=1`として実装)。 |
| `seed` | int | ベストエフォートのサンプリングシード。SDKサンプラに転送される。 |
| `logprobs` | int (0-20) | **生成**トークンのトークン単位logprobsと、各位置の上位N候補を返す([Logprobs](./MANUAL.ja.md#logprobs)参照)。非ストリーミングのみ。`echo`と組み合わせると**プロンプトスコアリング**(lm_eval loglikelihood)に切り替わる — `POST /v1/server/prompt_logprobs`によるゲートあり。 |
| `suffix` / `best_of` | — | 非対応 → `400`。 |
| `echo` | bool | trueならプロンプトを応答に前置(ストリーミング時は最初のchunkとして送出)。 |
| `n` | int | `1` のみサポート。`>1` は `400`。 |
| `stream_options.include_usage` | bool | trueなら `[DONE]` 直前に `usage` を含む `text_completion` chunkを送出。 |

```bash
curl $base_url/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of Japan is", "max_tokens": 16, "temperature": 0}'

# 2番目のスロット(chatスロットにロードされたモデル)を明示的に狙う場合
curl $base_url/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "general", "prompt": "...", "max_tokens": 16}'
```

`finish_reason` のセマンティクス(本エンドポイントと`/v1/chat/completions`共通): 自然終了・停止シーケンス一致は`"stop"`、`max_tokens`到達またはコンテキスト超過は`"length"`、function callingの応答は`"tool_calls"`(chatのみ、後述)。`max_tokens: 0`は`echo`+`logprobs`(プロンプトスコアリング)との組み合わせでのみ有効。

### POST /v1/chat/completions

チャット補完(`lm_eval` の `local-chat-completions` バックエンド、Open WebUI、OpenAI SDK用)。テンプレート適用・prefixキャッシュ適用あり。`model` でスロットを選択します。ストリーミング処理は `/v1/completions` と共通の実装(`engine.py` + `_sse_stream`)を使い、chunk形式(chat vs. text_completion)だけが異なります。

メッセージの`content`はプレーン文字列と、OpenAIのparts配列(`[{"type": "text", ...}]`、Open WebUIが送る形式)のどちらでも受け付けます。テキストパーツは自動でフラット化され、`image_url`パーツがあるとVLMスロットにルーティングされます([VLM(マルチモーダル)対応](./MANUAL.ja.md#vlmマルチモーダル対応)参照)。

`/v1/completions` のフィールドに加えて:

| フィールド | 型 | 説明 |
|---|---|---|
| `messages` | array | 必須、非空。`{"role": "...", "content": "..."}` の配列。 |
| `enable_thinking` / `chat_template_kwargs.enable_thinking` | bool | 既定 `true`。OpenAI標準フィールドではなく、Qwen3向けの独自拡張。トップレベルの`enable_thinking`、または`chat_template_kwargs`にネストした形式(vLLM/SGLangの流儀 — このdictをそのままHFの`apply_chat_template()`に渡す実装で、`enable_thinking`はQwen3自身のチャットテンプレートが読むkwarg名そのもの。両方指定時は`chat_template_kwargs`側が優先)のどちらでも受け付ける。`false`を指定すると、systemプロンプトに文字列`/no_think`をそのまま追記する(systemメッセージが無ければ新規作成する) — Qwen3自身が公式にドキュメント化しているチャットテンプレート向けのソフトスイッチで、モデルは自前の推論をスキップして直接回答する。**空の`<think>\n\n</think>\n\n`ブロックを事前に埋め込む方式(HuggingFaceのチャットテンプレートの仕組み)では実装していない** — Qualcommの公式リファレンスサーバ(`qai-appbuilder/samples/genie/c++/Service`)が実機検証で、この方式だと短いプロンプトでQwen3が退化する(直前のターンをそのまま繰り返した直後に終了する)ことを確認しているため、本サーバは彼らが検証済みの`/no_think`方式に合わせている。Qwen3系以外のテンプレート/モデルには影響しない(単なるプロンプト文字列であり、SDK側に推論ON/OFFの切り替え機能自体が存在しないため)。 |
| `tools` | array | OpenAI function callingのツール定義 — 後述の[Function calling](#function-calling-tools)参照。 |
| `tool_choice` | string | `"auto"`(既定)と`"none"`(ツール注入を無効化)のみ。`"required"` および `{"type":"function", ...}` の関数指定形式は**`400`で拒否**する — どちらもOpenAIのセマンティクスでは呼び出しを保証するもので、本サーバが実装していない制約付きデコーディングを要するため。`"auto"`を使い、応答に実際に`tool_calls`が入っているかを確認すること。 |
| `logprobs` / `top_logprobs` | bool / int (0-20) | 生成トークンのOpenAI chat形式logprobs(`choices[0].logprobs.content[...]`)。非ストリーミングのみ。[Logprobs](./MANUAL.ja.md#logprobs)参照。 |

```bash
curl $base_url/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "genie-local",
        "messages": [
          {"role": "system", "content": "あなたは親切なアシスタントです。"},
          {"role": "user", "content": "こんにちは"}
        ],
        "stream": true,
        "max_tokens": 128
      }'
```

ストリーミング応答はSSE(`text/event-stream`)、OpenAI仕様通り最初に `delta.role="assistant"` の空チャンクを送出します。

#### Function calling (`tools`)

リクエストにOpenAI `tools` が付いていると、サーバはそれを**Hermes形式**でsystemプロンプトに描画します — Qwen3系モデルが実際に学習しているフォーマットで、Qualcomm自身の`qai-appbuilder` GenieAPIServiceリファレンスと同じアプローチです:

- 関数シグネチャは `<tools></tools>` XMLタグでsystemプロンプト末尾に追記されます(systemメッセージが無ければ新規作成)。このブロックはキャッシュ可能なsystem prefixの一部なので、prefix KVキャッシュの恩恵を受けます。
- モデル出力中の `<tool_call>{"name": ..., "arguments": ...}</tool_call>` ブロックはOpenAIの `message.tool_calls`(`call_...` idを生成)にパースされ、`finish_reason` は `"tool_calls"` になります。複数ブロックは複数(並列)ツール呼び出しに。JSONとしてパースできないブロックは黙って捨てずに `content` に残します。
- **モデルが開いたまま閉じなかったブロックはテキストのまま残る**(`TOOL_CALL_RECOVERY` が ON のときだけ回収する)。小さいモデルは`</tool_call>`を落としてJSON直後にEOSを出すことがある(`qwen3_0_6b` w4a16 で観測。gemma4 も自身のマーカーで同じことをする)。**回収は可能**だが — 生成は終わっており JSON も完全 — **自分の呼び出しを閉じないモデルには、開きマーカーを化けさせるのと同種の欠陥がある**。既定で修復すると、回収の余地が無い `/v1/completions` に比べてこのエンドポイントでだけバンドルの成績が良く見えることになる。`max_tokens`でJSONの途中で切れた場合は、どちらの設定でも`content`に残る(引数を捏造しないため)。
- **マーカーが化けた・欠落した呼び出しも、その`name`がこのリクエストで宣言されたツール名なら回収する**(`TOOL_CALL_RECOVERY`、**既定OFF** — 計測対象のバンドルの欠陥を隠すものであり、しかもこのエンドポイントにしか効かず `/v1/completions` には効かないため)。SA8255Pのモデル2つがこれを必要とする。`qwen3_4b_instruct_2507`のw4a16は`<tool_call>`トークンの代わりにキリル文字(`ФРАГМЕНТ`、`Флагорное`など、**リクエストごとに違う文字列**)を出し、`temperature: 0`・20プロンプトの実測で呼び出しの**約半数**を失う。`qwen3_0_6b`はタグ自体を出さずthinkブロックの後に裸のJSONを置き、約4分の1を失う。どちらも呼び出しの中身は正しいので、この機能が無ければ、クライアントのコードが`message.tool_calls`を読んでいるのに`finish_reason: "stop"`のプレーンテキストが返る。`qwen3_1_7b`は喪失0%でこの機能を必要としない。
- **判別の決め手は宣言済みツール名との一致である。** 閉じタグだけが無い場合に裸のJSONをテキストのまま残していたのは、JSONで正当に回答するモデルを関数呼び出しと誤読しないためだった。`name`が実際に送られてきたツールであることを要求すれば、その懸念は無くなる。それ以外の名前のJSONは`content`に残り、`TOOL_CALL_RECOVERY`が`false`(既定)のときは全て`content`に残る — マーカーを確実に出すモデルにとっても、これが正しい設定である。化けたマーカー自体も呼び出しと一緒に取り除く: 隣接する行のうち空白を含まないものをマーカーとみなすため、**ツール呼び出しの隣にある本物の一語だけの行も失われる**。
- 往復の履歴も理解します: `tool_calls` 付きassistantメッセージは `<tool_call>` ブロックに再描画され、`role: "tool"` の結果メッセージは `<tool_response>` ブロック(chatml)/ `ipython` ターン(llama3)として、モデル自身のチャットテンプレート通りに描画されます。
- **ストリーミング**: `<tool_call>` ブロックはホールドバックされ(テキストとしてクライアントに漏れません)、生成完了後に完全な呼び出しを載せた `delta.tool_calls` チャンクを1つ送出し、`finish_reason: "tool_calls"` で終わります。`TOOL_CALL_RECOVERY`をONにしたとき、化けたマーカーはタグではないため、気づく前にcontentとして流れ出てしまい取り消せません。そこでテキストを1行ずつ保留し、呼び出しの本体でもその隣のマーカーでもないと確定してから流します。散文は通常どおりストリーミングされますが、最初の1語だけは「その行がマーカーではない」ことを示す空白が来るまで待ちます。

```bash
curl $base_url/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "東京の天気は?"}],
  "tools": [{"type": "function", "function": {
    "name": "get_weather",
    "description": "指定した都市の現在の天気を取得",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]
}'
```

Function callingはQwen3系(chatml)モデルで最もよく機能します。llama3系にも同じHermesブロックをベストエフォートで注入します。整形式の呼び出しを実際に出力するかどうかはモデル側の性質であり、サーバは保証しません。

## サーバの状態と制御

### GET /v1/server/status

非ブロッキングな現在状態の取得。`bench_ttft.py` 等が最上位の `phase`/`detail` を参照する前提のため、これらは**プライマリスロット(`slots[0]`)の値**を維持したまま返します。加えて `slots` に全スロットの内訳が入ります。

```json
{
  "phase": "idle",
  "detail": "",
  "active_model": "tool-model",
  "active_lora": "",
  "context_occupancy": 128,
  "slots": [
    {"name": "tool_call", "device_id": 0, "active_model": "tool-model", "active_lora": "",
     "phase": "idle", "detail": "", "context_occupancy": 128},
    {"name": "chat", "device_id": 1, "active_model": "general", "active_lora": "finetune-v2",
     "phase": "idle", "detail": "", "context_occupancy": 0}
  ]
}
```

- 各スロットの `context_occupancy` は、そのスロットのロックが即時取得できた場合のみ `GenieDialog_getValue(GENIE_DIALOG_PARAM_CONTEXT_OCCUPANCY)` で取得(取得できなければ `null`)。推論中でも本エンドポイント自体はブロックしません。

### GET /v1/server/idle

対象スロットのロックが解放されるまでブロックします(既定120秒でタイムアウト・`503`)。ベンチマークのフェーズ間同期に使用。`?slot=<名前>` で対象を選択(省略時はプライマリスロット)。

```bash
curl $base_url/v1/server/idle
curl "$base_url/v1/server/idle?slot=chat"
```

### GET /v1/server/performance_policy

現在の `Genie_PerformancePolicy_t` を取得(`?model=` でスロット選択、`/v1/lora/current` 同様の非ブロッキング試行)。

```json
{"slot": "chat", "policy": "balanced", "raw_value": 40, "live": true}
```

この値は**SDKに最後に伝えた内容**であって、ハードウェアの状態ではありません(ゲッターはデバイスに問い合わせず、保存済みのコピーを返します)。往復が成功しても「値が受理された」以上の意味はありません。判断の前に [パフォーマンスポリシー](./MANUAL.ja.md#パフォーマンスポリシー) を読んでください。

### POST /v1/server/performance_policy

パフォーマンスポリシーを設定します。`model` でスロットを選択。ベンチマーク実行前に `burst` に固定し、終了後に元へ戻す運用を想定しています。

**実際に性能が変わるかはターゲット依存で、少なくとも1つのターゲットでは全く変わりません** — [パフォーマンスポリシー](./MANUAL.ja.md#パフォーマンスポリシー) を参照してください。

```bash
curl -X POST $base_url/v1/server/performance_policy \
  -H "Content-Type: application/json" \
  -d '{"policy": "burst"}'
```

`policy` に指定できる値:

| 値 | `Genie_PerformancePolicy_t` |
|---|---|
| `burst` | `GENIE_PERFORMANCE_BURST` (10) |
| `sustained_high_performance` | 20 |
| `high_performance` | 30 |
| `balanced` | 40 |
| `low_balanced` | 50 |
| `high_power_saver` | 60 |
| `power_saver` | 70 |
| `low_power_saver` | 80 |
| `extreme_power_saver` | 90 |

### GET /v1/server/prompt_logprobs

`{"enabled": bool, "max_tokens": int}` を返します — プロンプトスコアリングが現在有効かどうか。

### POST /v1/server/prompt_logprobs

Body: `{"enabled": true|false}`。lm_evalのloglikelihood計測の前に有効化し、終わったら無効化してください([Logprobs](./MANUAL.ja.md#logprobs)参照)。

### GET /v1/server/profile

そのスロットの直近クエリについてSDK自身が計測したKPI(`time-to-first-token`、
`prompt-processing-rate`、`token-generation-rate` とその根拠となるトークン数)。
`?slot=<名前>` でスロットを選択(省略時はプライマリスロット)。

env_config.json の `GENIE_PROFILE: true` が必要。無効時は `409` を返す
(プロファイラは dialog 生成時にバインドされるため実行時に有効化できない)。
**意図的にOpenAI APIの外**に置いており、プロファイリングの有無で chat/completions の
レスポンス形状は変わらない。詳細と実測値は
[プロファイリング](./MANUAL.ja.md#プロファイリングsdk側のkpi)を参照。

```json
{"slot": "chat", "model": "...",
 "summary": {"ttft_ms": 184.6, "prefill_tokens_per_s": 135.4,
             "decode_tokens_per_s": 11.01, "prompt_tokens": 25,
             "generated_tokens": 32, "generation_ms": 2815.6},
 "profile": {"header": {}, "components": []}}
```

`host_measured` にはSDKがプロファイルしない値が入る: prefix cacheの
`restore_state_ms` と `save_state_ms`(ブロッキング呼び出しである
`GenieDialog_restore`/`_save` をサーバ側で挟んで計測)。どちらも未実行なら空。
詳細は[コストと損益分岐](./MANUAL.ja.md#コストと損益分岐)。

## Prefix KVキャッシュ

### GET /v1/prefix/cache

保存済みprefix KVキャッシュの一覧(全スロット共通のディレクトリ、キーで論理的に分離)。

```json
{"entries": [{"key": "...", "path": "...", "kind": "file", "size_bytes": 12345, "mtime": 1700000000}]}
```

### DELETE /v1/prefix/cache/{key}

指定キーのキャッシュを削除。存在しなければ `404`。

### POST /v1/prefix/warmup

指定システムプロンプトのprefix KVキャッシュを事前生成します。`model` でスロットを選択します。

```bash
curl -X POST $base_url/v1/prefix/warmup \
  -H "Content-Type: application/json" \
  -d '{"system_prompt": "あなたは親切なアシスタントです。"}'
```

- 選択されたスロットのアクティブなモデル/テンプレートに対してウォームアップします。
- テンプレートがllama2/mistral(分割不可)の場合は `422`。
- 既にキャッシュ済みなら即座に `{"status": "already_cached", "slot": "...", ...}`。
- ウォームアップに `WARMUP_JOIN_TIMEOUT`(既定600秒)以上かかる場合、`202` を返してバックグラウンド続行(`/v1/prefix/cache` でポーリング)。他スロットはブロックされません。

> [!NOTE]
> **実機で確認済み**(LoRAアダプタを持つバンドル)。適用すると生成が変わり
> SDKから読み戻せること、強度を変えるとさらに変わること、解放すると戻ること、
> 未知のアダプタ名は `GenieDialog_applyLora failed: -1` で失敗することを確認しました。
>
> LoRAを使う前に知っておく価値のある点が2つあります。ひとつは、バンドルが
> **実質恒等のアダプタ**を同梱していることがある点(選んでも何も起きないように
> 見えるのは、実際に何も起きていないため)。**別のアダプタとではなく、解放した状態と
> 比較してください**。もうひとつは、**`dialog.type` が `ssd-q1` のバンドルでは、
> ライブラリにパッチを当てない限り LoRA が一切使えない**点です —
> [D5](./QAIRT_VERSIONS.ja.md#d5--リセットが投機デコードのダイアログを壊す)。

> [!IMPORTANT]
> `enable_thinking: false` はQwen3の `/no_think` をシステムターンに追記するため、
> **別のキー**でキャッシュされます。**実際に送る形と同じもの**を温めてください —
> このエンドポイントにも同じ `"enable_thinking": false` を渡します。素のプロンプトを
> 温めてからフラグ付きで送ると、**永久に静かなMISS**になります(リクエスト自体は成功するので
> 気付きにくい)。4B・システムプロンプト234トークンでの実測: MISS 361ms に対し HIT 184ms。

## スロットの選び方

1つのスロットに作用するエンドポイントは、すべて同じ2段階の規則に従います。
各エンドポイントで繰り返さず、ここに一度だけ書きます。

1. **明示された `slot`** — `TEXT_SLOTS[].name` で付けたスロット自身の名前
   (`"chat"`、`"tool_call"` など)。POST はリクエストボディ、GET は `?slot=`。
   **未知の名前は `404`** です。
2. **無ければ `model`** — 各スロットがロードしているモデルと突き合わせます。
   **誰もロードしていない名前は、失敗ではなくプライマリスロットに落ちます** —
   `lm_eval` が全リクエストに固定のプレースホルダを送るためです。

`slot` があるのは、`model` では答えが出ない場合があるからです —
**同じモデルディレクトリを2つのスロットに載せていると、モデル名では区別できず**、
2本目には `slot` でしか到達できません。同じモデルを二度ロードしないなら、
`model` だけで足ります。

## モデルとLoRA

### POST /v1/models/switch

**1つのハードウェアスロット**にロードされているモデルをホットスワップします。

```bash
curl -X POST $base_url/v1/models/switch \
  -H "Content-Type: application/json" \
  -d '{"slot": "chat", "model_dir": "llama3-8b-htp"}'
```

- `slot`: 対象スロット名(`env_config.json`の`TEXT_SLOTS[].name`、単一スロット構成では`"default"`)。省略時はプライマリスロット(`slots[0]`)。
- `model_dir`: 相対パスなら `MODELS_BASE_DIR`(未設定時はサーバの作業ディレクトリ)基準で解決し、絶対パスならそのまま使います。これは起動時の `TEXT_SLOTS`/`VLM_SLOTS` の `model_root` と同一のルールなので、同じディレクトリ名は両方で同じモデルを指します。 **許可リストも認証もありません** — このエンドポイントは、サーバプロセスが読めるあらゆるパスを開きます(意図的な設計です。[モデルパスの解決ルール](./MANUAL.ja.md#モデルパスの解決ルール)参照)。
- `config_file`: `model_dir` 内のダイアログ設定ファイル名。既定はそのスロット自身の設定(`TEXT_SLOTS[].config_file`、さらにその既定は `genie_config.json`)。**切り替え先のバンドルが別の名前を使っている場合に指定します。** 指定した名前はそのままスロットに残ります。
- `unload_first`(bool、既定`true`): 下記参照。
- 既定(`"unload_first": true`)では、**新モデルをロードする前に**そのスロットの旧 `GenieDialog` ハンドルを解放します。新モデルはHTPデバイスを独占した状態でロードされるため、**この順序が確実に切り替わります**。代償として、その後のロードが失敗した場合、そのスロットは**モデルが一切ロードされていない状態**になります(`GenieDialog_query`等の呼び出しは`503`を返します — [制約事項](./MANUAL.ja.md#制約事項)参照)。次のswitchが成功するまでこの状態が続きます。
- `"unload_first": false` では**先に新モデルをロードし**、成功してから旧ハンドルを解放します。ロードに失敗してもそのスロットは旧モデルのまま稼働を続けるという利点がありますが、HTPデバイス上に**新旧両モデル分**(小さい方のワーキングセットの約2倍)の空きが同時に必要で、**SA8255Pボードではこの同時常駐が当てにならないことが実測で分かっています**: 36回のスワップを測った結果、**成否はどのモデルの組み合わせかでは決まりませんでした** — 同じ組み合わせが、あるときは6回中6回成功し、別のときは8回中8回失敗し、6回の連続実行の途中で失敗から成功に転じたこともあります。決めているのはホストからは観測できないデバイス側の状態です。またデバイス自体が健全でも、2つのモデルを同時に保持する余地がそもそも無い場合、SDKレベルのバッファ登録エラー(例: `memRegister ERROR(8003)`)で失敗することがあります。
- **`false` を使うのは、デバイスのメモリに十分な余裕があり、かつ実際に運用で行うスワップの組み合わせを、コールドスタートからの反復も含めて十分にテストした場合だけにしてください。** そこで安定しないなら、既定のまま「一時的にスロットが空になる」方を扱う設計にしてください。
- 新モデルは**同じスロットの`device_id`を引き継ぎます**(スロットのハードウェア割り当ては、モデルの入れ替えでは変わりません)。
- **対象スロット自身のロックのみ**を取得するため、進行中の推論完了を待ってから切り替わります(既定タイムアウト600秒)が、**他スロットの推論はブロックしません**。
- 成功時、そのスロットのprefix KVキャッシュ namespace も自動的に切り替わります(明示的なキャッシュクリアは不要)。
- 成功時、そのスロットの適用中LoRAは自動的にリセットされます(`active_lora_adapter = ""`)。

レスポンス:

```json
{"status": "switched", "slot": "chat", "model": "llama3-8b-htp", "template": "llama3"}
```

失敗時: `400`(`model_dir` 未指定)/ `404`(存在しないスロット名、またはディレクトリに `genie_config.json` が無い)/ `500`(SDKロード失敗 — 既定の `unload_first` ではそのスロットが未ロード状態になる。`"unload_first": false` なら旧モデルのまま稼働を続ける。上記参照)/ `503`(対象スロットのロック取得タイムアウト)。

### POST /v1/lora/apply

LoRAアダプタを適用します。適用後、自動的に `GenieDialog_reset` されます。`slot` でスロットを、`model` ではロード中のモデル名で選択します([スロットの選び方](#スロットの選び方)を参照)。

```bash
curl -X POST $base_url/v1/lora/apply \
  -H "Content-Type: application/json" \
  -d '{"model": "genie-local", "engine": "primary", "lora_adapter_name": "my-lora"}'
```

- `engine` は対象のエンジンロール名(`genie_config.json` のエンジン設定名。単一エンジン構成では通常 `"primary"`)。既定 `"primary"`。
- 適用結果はSDKから読み戻して(`GenieDialog_getValue`)レスポンスに含めます。

### POST /v1/lora/strength

LoRAのalpha強度を変更します(適用後 `GenieDialog_reset`)。`slot` でスロットを、`model` ではロード中のモデル名で選択します([スロットの選び方](#スロットの選び方)を参照)。

```bash
curl -X POST $base_url/v1/lora/strength \
  -H "Content-Type: application/json" \
  -d '{"engine": "primary", "tensor_name": "lora_alpha_0", "alpha": 0.8}'
```

### POST /v1/lora/release

LoRAアダプタのメモリを解放します(適用後 `GenieDialog_reset`)。再適用時はディスクから再ロードされるため低速になります。`slot` でスロットを、`model` ではロード中のモデル名で選択します([スロットの選び方](#スロットの選び方)を参照)。

```bash
curl -X POST $base_url/v1/lora/release \
  -H "Content-Type: application/json" \
  -d '{"engine": "primary", "lora_adapter_name": "my-lora"}'
```

### GET /v1/lora/current

現在適用中のLoRAアダプタ名を取得します。`?model=` でスロットを選択。対象スロットのロックを1秒だけ試行し、取得できなければキャッシュ済みの値を `"live": false` で返します(推論をブロックしないため)。

```json
{"slot": "chat", "lora_adapter_name": "my-lora", "live": true}
```

## エラー形式

パラメータ検証・存在しないスロット・SDK失敗・不正なJSONボディ・内部の`HTTPException`まで、**全ての**エラーがOpenAIのエラーエンベロープで返るため、OpenAI SDK / lm_eval / LiteLLMは常に失敗をパースできます:

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": "model_dir", "code": null}}
```

主なステータスコード:

| コード | 意味 |
|---|---|
| `400` | リクエストパラメータ不正(必須フィールド欠落、`n>1` 等) |
| `404` | 存在しないリソース(prefixキャッシュキー、モデルディレクトリ、存在しないスロット名) |
| `422` | 意味的に処理不能(llama2テンプレートでのprefix warmup要求など) |
| `500` | SDK呼び出し失敗、モデルロード失敗 |
| `503` | 対象スロットのロック取得タイムアウト(そのスロットがビジー) |
| `504` | 推論タイムアウト |

SSEストリーミングの途中(トークン送出後)で失敗した場合は、`data: [DONE]` の前に最後のイベントとして `data: {"error": {...}}` を送出します(vLLM流)。

