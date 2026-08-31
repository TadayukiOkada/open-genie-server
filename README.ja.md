# open-genie-server

*[English](./README.md) | 日本語*

Qualcomm Genie C API (`libGenie.so`) を OpenAI互換のREST APIとして公開する、単一プロセスのFastAPIサーバです。[Hexagon NPU](./docs/MANUAL.ja.md#用語集)(本サーバが QNN HTP バックエンド経由で使うアクセラレータの、Qualcomm での呼び名)上で動くLLMを、`lm_eval`・`curl`・OpenAI SDK・[Open WebUI](https://github.com/open-webui/open-webui)等、一般的なOpenAI互換HTTPクライアントから叩けるようにします。実装本体は `genie_server` パッケージ(`src/genie_server/`)にあり、`genie-server.py` はランチャーです。

> [!IMPORTANT]
> このリポジトリは open-genie-server 自体のソースのみを含みます。動作させるには別途、Qualcomm から **QAIRT SDK**(`libGenie.so` を含む、Qualcomm の proprietary licenseで配布されるツールチェーン)と、Hexagon NPU向けにコンパイル済みのモデル一式(`genie_config.json` を含むモデルディレクトリ)を用意する必要があります。SDK・モデルはこのリポジトリには含まれておらず、[Qualcomm AI Hub](https://aihub.qualcomm.com/) 等から別途取得してください。

設定と挙動は [MANUAL.ja.md](./docs/MANUAL.ja.md)、エンドポイントのリファレンスは [API.ja.md](./docs/API.ja.md)、そして**本ドキュメントの実測値がデバイスについて何を前提にしているか**は [プラットフォーム別の注意点](./docs/PLATFORM_NOTES.ja.md) を参照してください。

## このプロジェクトの目的

**Genie C API と量子化済みモデルバンドルを検証するための計測器であって、本番用の推論サーバではありません。** 他のすべてはここから、この優先順位で導かれます:

1. **Genie C API を、APIが許す限り広く使えるようにする。** チャットだけではありません。SDK側のプロファイリングカウンタ、パフォーマンスポリシー、カスタムサンプラーフック経由のプロンプトスコアリング、LoRA、prefixキャッシュのスナップショット、VLMスロットの背後にある `GenieNode`/`GeniePipeline` — これらすべてにHTTPから手が届きます。**触れるようにすること自体が目的**だからです。
2. **標準的なベンチマークを、そのままデバイスに向けられるようにする。** `lm_eval` は無改造で通ります(生成タスクもloglikelihoodタスクも)。[examples/bfcl](./examples/bfcl) は Berkeley Function Calling Leaderboard を同じやり方で回します。**サーバに手を入れないと動かないベンチマークは、結局回さない**ものです。
3. **SDKやモデルの問題を、既定では隠さない。** 見えない欠陥は、そのまま出荷される欠陥です。よって素のライブラリによるスロット恒久故障は検知も回避もしません。grammar が漏らす終端トークンは、取り除かずに報告します。応答が名乗るのは、クライアントが送ってきた文字列ではなく**実際にロードされているモデル**です。prefixキャッシュは明示的なウォームアップでしか埋まらないので、TTFTの計測を黙って改善してしまうことがありません。回避策はありますが、**何を隠すかを承知のうえで自分で入れるスイッチ**として用意してあり、どちらも**自分で ON にするまで効きません** — マーカーが化けたツール呼び出しを組み立て直す `TOOL_CALL_RECOVERY` と、コンテキストに収まらない枚数の動画を拒否する `VLM_VISION_BUDGET_GUARD`(SDK に素通しすると、スロットが恒久故障するかプロセスが死にます)。
4. **OpenAI API および他の主要な推論サーバの挙動との互換性を保つ — 3と衝突しない範囲で。** 両者が食い違うときは、実際に起きたことを報告する側を採ります。上に挙げた `model` フィールドがその実例です。OpenAI も vLLM もリクエストをそのまま返しますが、本サーバは返しません。**ホットスワップされたモデルで走ったベンチマークは、そう名乗るべき**だからです。

**後方互換性はこの中に入りません。** 意味もなく壊すことはしませんし、既存の応答形式や既定値を変える変更は [CHANGELOG](./docs/CHANGELOG.md) の Breaking に明記します。ただし本サーバは **Genie C API を覗くための窓**であり、**その API が動けば、こちらも動きます** — 既存の呼び出し側を守るためだけに古い形を残すことは、SDK が今どうなっているかについて**計測器に嘘をつかせる**ことになります。**自分たちの既定値についても同じ**です。ある既定値が何かを隠していると実測で分かれば、それだけで変更の理由になります。**動かない表面が必要なら、バージョンを固定してください。**

**目指していないもの。** 認証もレート制限もマルチプロセスのスケーリングもありません。`POST /v1/models/switch` はプロセスが読めるあらゆるパスを開きます。1つのテキストスロットは、単一の `GenieDialog` ハンドルの後ろでリクエストを直列化します。**自分で管理しているベンチ用ネットワークで動かしてください**([SECURITY.ja.md](./SECURITY.ja.md) に、それが何を意味するかと、報告してほしいことを書いてあります)。Hexagon上に本番のサービング基盤が要るなら、ここは出発点として間違っています — ただし、**あなたのバンドルとSDKが実際に何をしているかは、詳細に教えてくれます**。

## 特徴

- `/v1/completions`・`/v1/chat/completions` — OpenAI互換のテキスト/チャット補完(ストリーミング対応。`/v1` なしのパスにも登録)
- **Function calling(`tools`)対応** — プロンプトの方言を2つ実装し、スロットのチャットテンプレートから自動で選択。Qwen3系向けの Hermes `<tool_call>` JSON と、gemma4 独自の `<|tool_call>call:NAME{...}` トークン。どちらでもワイヤー形式はOpenAIのままで、`message.tool_calls` / `finish_reason: "tool_calls"` に変換し、ストリーミング中にテキストとして漏らさない
- `lm_eval`(`local-completions` / `local-chat-completions`)にそのまま対応。トークンID形式のプロンプトもサーバ側でデコード
- **Logprobs対応**(SDKのカスタムサンプラーフック経由): 生成トークンの`logprobs`/`top_logprobs`(トークンあたり数msのオーバーヘッド、未使用時はゼロ)に加え、**プロンプトスコアリング**(`echo`+`logprobs`のteacher forcing)でlm_evalのloglikelihoodタスク(hellaswag, arc, mmlu等)も実行可能 — デコード速度で走るため`POST /v1/server/prompt_logprobs`によるゲート付き
- Open WebUIフレンドリー — parts配列形式 `content` のフラット化、`GET /health`、CORS、ストリーミング `usage` チャンク(`stream_options.include_usage`)
- システムプロンプトのprefix KVキャッシュ(モデル/LoRAごとにnamespace化)
- `GenieDialog_applyLora` 等によるLoRAアダプタのホットスワップ — 適用・強度・解放・読み戻しとも**実機で確認済み**
- `/v1/models/switch` によるモデルのホットスワップ(既定では旧モデルを解放してから新モデルをロードします。この順序が確実に切り替わりますが、ロードに失敗するとスロットは空になります。`"unload_first": false` は新旧を同時に載せることで旧モデルをフォールバックとして残せます — 使う前に下記の注意を参照)
- `Genie_PerformancePolicy_t` の切り替え(ベンチマーク時に `burst` 固定など)
- Context occupancy(KVキャッシュ占有量)を含む非ブロッキングなステータス監視
- **SDK側プロファイリング** — `GENIE_PROFILE` で Genie 自身が計測した TTFT/プレフィル/デコードのKPIを `GET /v1/server/profile` から取得([プロファイリング](./docs/MANUAL.ja.md#プロファイリングsdk側のkpi)参照)。OpenAIのレスポンス形状は一切変更しない
- **マルチテキストスロット対応** — `TEXT_SLOTS` 設定で、**使える**Hexagon NSPコア(cdsp0/cdsp1。HTP/NSP/cDSP/NPU の関係は[用語集](./docs/MANUAL.ja.md#用語集)参照)それぞれに独立した `GenieDialog` ハンドル(別ロック・別モデル可)を割り当て。別スロット宛のリクエストは実際に重なって進みますが、当方のベンチでの実測向上は **2倍ではなく約1.3倍**でした([マルチテキストスロット](./docs/MANUAL.ja.md#マルチテキストスロット)参照)。**何本使えるかは品番ではなくSKUのライセンスで決まります** — [プラットフォーム別の注意点](./docs/PLATFORM_NOTES.ja.md)参照
- **Grammar制約デコーディング** — JSON Schema/正規表現/EBNFで出力を制約(XGrammarバックエンド、モデル/スロット単位の固定設定)
- **VLM(マルチモーダル)対応** — Qwen3-VL等、`GenieNode`/`GeniePipeline` composable pipeline APIを使う画像入力モデルを、`VLM_SLOTS`設定で `TEXT_SLOTS` と完全に並列に追加可能([examples/vlm](./examples/vlm/README.ja.md)参照)
- オフラインテストスイート(`tests/`、pytest + fake SDK)— HTTP/エンジン/テンプレートの全スタックがNPUなしで動作確認可能

## 必要要件

- Python 3.10+(`int | None` 等の型構文を使用)
- QAIRT SDK(`libGenie.so` とその依存ライブラリ)、Hexagon NPU上で動作するモデル一式(上記の注意参照)

```bash
pip install .[logprobs,vlm]     # 全部入り
pip install .                   # サーバ本体のみ: fastapi, uvicorn, tokenizers
```

配布名は `open-genie-server`、import するパッケージ名は `genie_server` です。
`pip install -r requirements.txt` も従来どおり使えます(上の1行目と同じ内容)。

| | 区分 | 用途 |
|---|---|---|
| `fastapi`, `uvicorn` | 本体 | サーバ |
| `tokenizers` | 本体 | 正確なトークン数カウント。無いと `text.split()` で数えるため**実際55トークンの日本語の段落が「1」になり**、usage だけでなくコンテキストチェックと既定 `max_tokens` にも効きます。[トークン数のカウント](./docs/MANUAL.ja.md#トークン数のカウント)を参照 |
| `numpy` | `[logprobs]`, `[vlm]` | **logprobsとプロンプトスコアリング**、およびVLM。無いとこれらのリクエストは HTTP 400 で拒否されます |
| `pillow` | `[vlm]` | 画像入力 |
| `pytest`, `httpx`, `requests`, `jsonschema` | `[test]` | オフラインテストスイート |

インストールすると `genie-server` コマンドが使えるようになります。リポジトリ直下の
`genie-server.py` ランチャーも同じもので、こちらはインストール不要です。
Android ではこのうち3つに wheel がありません —
[Androidで動かす](./docs/MANUAL.ja.md#androidで動かす)を参照してください。

> [!WARNING]
> **デプロイ前に、どのQAIRTバージョンを指しているか確認してください。**
> どのSDK欠陥を抱えることになるかはそのバージョンで決まります。そして
> **当方が検証した 2.49.x はいずれも、3つの欠陥を同じ一箇所に抱えています —
> `GenieDialog_reset()` が元に戻しそこねるもの**です。サーバはリクエストを独立させるために
> 毎回リセットするので、そこは**あなたが処理するすべてのリクエストの下を通る経路**です。
>
> 現れ方は、**過大なリクエスト1回でスロットが恒久的に壊れる**、
> **直前が短かったせいで長いリクエストが空のコンテキストで失敗する**、
> **投機デコード向けのバンドルで、最初のリセット以降の応答が流暢なまま間違って返る** —
> のいずれかです。**3つとも「成功」を報告します。**
>
> バージョン別の一覧、[手元のSDKで試せる確認手順](./docs/QAIRT_VERSIONS.ja.md#手元の-sdk-を確認する)、
> ライブラリの選び方は **[QAIRT バージョン別の問題点](./docs/QAIRT_VERSIONS.ja.md)** に
> まとめてあります。**そこに載っていないバージョンは
> 「問題が無いと分かっている」のではなく「未検証」**です。

## クイックスタート

> [!NOTE]
> この節の例は `192.168.1.2:8080` でデバイスに接続しています。これは本サーバ固有の
> 値ではなく、**SA8255P の LV GVM が既定で起動してくるアドレス**です。当方の検証環境も
> これを使っているため、ドキュメント全体と
> `tests/integration/test_config.sample.json` に同じ値が出てきます。
> ご自身のデバイスのアドレスに置き換えてください(サーバとクライアントが同じマシン上
> なら `localhost` で構いません)。

1. `env_config.json` をサーバ起動ディレクトリ(カレントディレクトリ)に用意します。

   ```json
   {
     "QAIRT_SDK_ROOT": "/path/to/qairt-dir",
     "HEXAGON_VERSION": "v73",
     "MODELS_BASE_DIR": "/path/to/models",
     "PREFIX_CACHE_DIR": "/path/to/prefix_cache",
     "TEXT_SLOTS": [{"model_root": "model-dir"}]
   }
   ```

   `model_root` は `genie_config.json` を含むディレクトリを指します。スロットに必須なのはこのキーだけで、`name` と `device_id` は既定値が使われます。

   相対パスの `model_root` は `MODELS_BASE_DIR` 配下として解決されるので、上の例は `/path/to/models/model-dir` をロードします。絶対パスも使えます — [モデルパスの解決ルール](./docs/MANUAL.ja.md#モデルパスの解決ルール)を参照。

   複数NSPコアを積んだSoCでは、コアごとにエントリを足すとそれぞれにモデルを常駐させられます(詳細と、2本目が必ず載るとは限らないという順序の制約は [MANUAL.ja.md](./docs/MANUAL.ja.md#マルチテキストスロット)):

   ```json
   {
     "QAIRT_SDK_ROOT": "/path/to/qairt-dir",
     "HEXAGON_VERSION": "v73",
     "MODELS_BASE_DIR": "/path/to/models",
     "PREFIX_CACHE_DIR": "/path/to/prefix_cache",
     "TEXT_SLOTS": [
       {"name": "tool_call", "device_id": 0, "model_root": "model-fast"},
       {"name": "chat", "device_id": 1, "model_root": "model-general"}
     ]
   }
   ```

2. サーバを起動します。

   ```bash
   python3 genie-server.py            # フラグ: --config/--host/--port
   # または、パッケージをインストール済みなら
   genie-server                       # フラグは同じ
   # または uvicorn を直接
   uvicorn genie_server.asgi:app --host 0.0.0.0 --port 8080 --workers 1
   ```

3. 動作確認:

   ```bash
   curl http://192.168.1.2:8080/v1/models

   curl http://192.168.1.2:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"genie-local","messages":[{"role":"user","content":"こんにちは"}]}'
   ```

Grammar制約デコーディングの設定例は [examples/grammar](./examples/grammar/README.ja.md)、VLM(画像入力)の設定・テスト手順は [examples/vlm](./examples/vlm/README.ja.md)、`lm_eval` の実行手順(量子化前モデルとの比較方法を含む)は [examples/lm_eval](./examples/lm_eval/README.ja.md) を参照してください。

## lm_eval での利用

```bash
lm_eval --model local-chat-completions \
  --model_args model=genie-local,base_url=http://192.168.1.2:8080/v1,\
tokenizer_backend=huggingface,tokenizer=<hf_model>,max_tokens=512,num_concurrent=1 \
  --tasks mmlu_generative --apply_chat_template --batch_size 1
```

設定と挙動の詳細は [MANUAL.ja.md](./docs/MANUAL.ja.md)、エンドポイントのリファレンスは [API.ja.md](./docs/API.ja.md) を参照してください。

## ディレクトリ構成

```
genie-server.py       — ランチャー(CLIフラグ: --config/--host/--port)
src/genie_server/     — サーバ実装本体。関心ごとに1モジュール
pyproject.toml        — パッケージング設定: 依存・extras・genie-server コマンド
requirements.txt      — pyproject.toml の依存を指すだけのファイル(互換用)
SECURITY.md           — 意図的に持っていないもの/報告してほしいもの(日本語版あり)
LICENSE

tests/                — オフラインテストスイート(pytest。fake SDKでNPU不要)
tests/integration/    — 実機に対するホスト側テストランナー(MD/JSONレポート出力)

docs/MANUAL.ja.md     — 設定・挙動と、その理由(英語版は MANUAL.md)
docs/API.ja.md        — 全エンドポイントを用途別にまとめたもの(英語版は API.md)
docs/PLATFORM_NOTES.ja.md — 実測値がデバイスについて前提にしていること(英語版あり)
docs/QAIRT_VERSIONS.ja.md — QAIRT バージョン別のSDK欠陥(英語版あり)
docs/CHANGELOG.md     — リリースノート

examples/config/      — env_config.json サンプル(シングルスロット/デュアルNSP/VLM)
examples/grammar/     — Grammar制約デコーディング
examples/vlm/         — VLM(マルチモーダル)の設定・テスト手順
examples/lm_eval/     — lm_eval の実行と、量子化前モデルとの比較
examples/bfcl/        — Berkeley Function Calling Leaderboard
```

`src/genie_server/` のモジュールごとの内訳は、ここで繰り返さず
[MANUAL.ja.md § アーキテクチャ概要](./docs/MANUAL.ja.md#アーキテクチャ概要) にまとめてあります。

オフラインテストの実行:

```bash
pip install -e .[logprobs,vlm,test]
python3 -m pytest tests/
```

`[test]` の依存は「任意」ではありません。`requests` と `jsonschema` が無いと
grammar のテストが8件 `ModuleNotFoundError` で落ち、`numpy` が無いと
logprobs のテストが8件落ちます。同じスイートは push と
pull request のたびに Python 3.10 / 3.12 で実行されます
(`.github/workflows/offline-tests.yml`)。

ホストPCから実機をエンドツーエンドで検査する統合テスト(Markdown/JSONレポート、サーバ停止検出付き)は [tests/integration/](./tests/integration/README.ja.md) を参照してください。

## 既知の制約

- **本サーバは、このページ冒頭で説明した素のライブラリでのリセット系欠陥を防ぎません** — 検知も復旧もしません。回避はデプロイ側の選択です — [QAIRT バージョン別の問題点](./docs/QAIRT_VERSIONS.ja.md)を参照。
- **2.49.x のライブラリでは、投機デコード向けにビルドされたバンドル(`"dialog": {"type": "ssd-q1"}`)にパッチ版が要り、それ無しでは LoRA も使えません。** これはバンドル側の性質ではなく **2.49 の回帰**で、2.48.40.260702 では正しく動きます。理由と、バンドル側の1行の書き換えで回避する方法は [D5](./docs/QAIRT_VERSIONS.ja.md#d5--リセットが投機デコードのダイアログを壊す) を参照。
- 1テキストスロット = 1 `GenieDialog` ハンドルで、スロット内のリクエストはそのスロット自身のロックで直列化されます(`TEXT_SLOTS` 未設定時は単一スロットのみで、従来通り全リクエストが直列化されます)。
- `n > 1`(1リクエストでの複数補完同時生成)は非対応で、`400` で拒否されます。
- Llama2/Mistralテンプレートはシステムプロンプトを `[INST]` に埋め込むため、prefix KVキャッシュの対象外。
- `POST /v1/models/switch` は既定で旧モデルを解放してから新モデルをロードするため、ロードに失敗するとそのスロットは次のswitchが成功するまでモデル未ロードのままになります。その間、そのスロットに触れる全エンドポイントは `503` を返します。
- `"unload_first": false` にすると新旧を同時にHTPデバイスへ載せてこれを避けられますが、**SA8255Pボードではこの同時常駐は当てになりません**。36回のスワップを実測した結果、成否は**どのモデルの組み合わせかでは決まりませんでした** — 同じ組み合わせがあるときは6回中6回成功し、別のときは8回中8回失敗し、6回の連続実行の途中で失敗から成功に転じたこともあります。決めているのはホストからは観測できないデバイス側の状態です。**デバイスのメモリに十分な余裕があり、かつ実運用で行うスワップをコールドスタートからの反復も含めてテスト済みの場合にだけ**使ってください。
- VLMスロットはシングルターンのみ(会話履歴の保持なし)、LoRA・Prefix KVキャッシュ・grammar制約・ホットスワップ非対応。詳細は [MANUAL.ja.md](./docs/MANUAL.ja.md#vlmマルチモーダル対応) を参照。

他の詳細な制約事項は [MANUAL.ja.md の制約事項セクション](./docs/MANUAL.ja.md#制約事項) を参照してください。

## 謝辞

本プロジェクトは [Claude Code](https://claude.com/claude-code) を使って開発しました。
最初のコミットは 2026-08-19 で、そこからの11日間で、2,700行の単一スクリプトから、
オフラインテスト311件・実機統合テスト・英日2言語のマニュアルを備えたパッケージに
なりました。

**時間がかかったのはコードではありません。** QAIRT SDK のリファレンス実装を
「SDK の欠陥」と「こちらのバグ」を切り分けられる程度まで読み込むこと、その一つひとつを
実機で再現させてどちらなのかを確定させること、そして**推測ではなく実測を書き残す**こと —
本ドキュメントの記述にも、当時は自明に見えた結論を後から覆したものがいくつもあります。
これを**一人でこの速度でやることは、Claude Code なしでは不可能でした。**

## License

[MIT](./LICENSE)

本リポジトリのライセンスはopen-genie-server自体のソースコードにのみ適用されます。Qualcomm QAIRT SDK・`libGenie.so`・Hexagon NPU向けモデルは対象外で、それぞれQualcommおよび各モデルの配布元のライセンス条件に従います。

**MIT は既定値ではなく選択です。** この種の計測器はコピーレフトで公開されていることが
多いので、そうしなかった理由を書いておきます。理由は2つあり、どちらもこのプロジェクト
固有のものです。ひとつは、本サーバが proprietary な `libGenie.so` を ctypes で
**実行時にロードする**こと。コピーレフトにすると、**このサーバと SDK を同梱した
ボードイメージを配布する**人に結合著作物の解釈問題を負わせることになります。これは
このハードウェアでは普通の配布形態であり、モデルを1つ測る前に法務レビューを挟ませるだけの
価値はありません。もうひとつは、**このコードの正しい使い方が「分解すること」**だという点です。
`logprobs.py` を自分の評価ハーネスに持っていく、自分のボードに必要なエンドポイント1つの
ために fork する — そこに課税するライセンスは、このプロジェクトの目的に逆行します。
