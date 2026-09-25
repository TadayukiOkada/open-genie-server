# open-genie-server リポジトリ詳細レビュー

- 対象: `TadayukiOkada/open-genie-server`、`master` ブランチ（HEAD `a1499eb` "Release 1.4.0"）
  - 依頼には `main` とありましたが、このリポジトリに `main` ブランチはなく、デフォルトブランチは `master` です。
- レビュー日: 2026-09-25
- 範囲: リポジトリ全体（`src/genie_server/`、`tests/`、`examples/`、`docs/`、CI、パッケージング）
- やったこと: 全モジュールを通読し、pytest・カバレッジ・ruff・mypy・pip-audit を実行しました。主要な指摘はリポジトリ同梱のフェイク SDK（`tests/fake_genie.py`）で再現しています。

> 凡例: **[再現]** はこのレビューでフェイク SDK を使い実際に挙動を確認したもの。**[コード]** はコードを読んで成立を確認したもの（実行による再現はしていない）。**[要確認]** は実機 SDK の挙動次第のもの。

---

## 1. サマリー

open-genie-server は、Qualcomm の `libGenie.so`（QAIRT Genie C API）を ctypes で包み、OpenAI 互換の REST API として公開する単一プロセスの FastAPI サーバです。README では「本番サーバではなく計測器（bench instrument）」と位置付けています。コードベースは Python 約 6,000 行（src）です。SDK の不具合を「隠さない」方針が一貫しており、コメントには実機計測の根拠が並んでいます。505 件のオフラインテストはすべて green でした（カバレッジ 81%）。一方で、**非同期ハンドラ内のブロッキング処理でサーバ全体が止まる問題**、**タイムアウトで途中打ち切りされた出力が正常終了（200 / `finish_reason: "stop"`）として返る問題**、**リクエスト間で状態が漏れる問題（サンプラー設定の持ち越し、prefix cache の名前空間競合）**、**2 つ目以降の system メッセージを黙って捨てる問題**が確認できました。どれも「計測結果が正しいこと」というこのプロジェクトの存在意義を直接損ないます。セキュリティ面では、「認証なし」は設計上の割り切りとして明記されています。ただ、**CORS `*` と Content-Type を検査しない JSON パースの組み合わせで、同じネットワーク内のブラウザ経由で任意の Web ページから状態変更 API やプロセスクラッシュを引き起こせる**点は、SECURITY.md の想定（「制御下のネットワークで動かせば安全」）を超えるリスクです。

全体評価: **ドメイン知識の蓄積と文書化は非常に高水準です。ただし並行性と入力検証まわりに、計測値の信頼性を損なう実バグがあります。** 修正はどれも局所的で済み、構造的な作り直しは要りません。

---

## 2. 重大度別の問題一覧

### Critical

#### C-1. 同じネットワーク上のブラウザを踏み台に、任意の Web サイトから状態変更 API とプロセスクラッシュを誘発できる **[再現]**
- 箇所:
  - `src/genie_server/app.py:352-358`（`allow_origins=["*"]`, `allow_methods=["*"]`, `allow_headers=["*"]`）
  - `src/genie_server/protocol.py:244-252`（`read_json_body` が Content-Type を検査せずに `request.json()` する）
  - `src/genie_server/config.py:245`（`HOST` の既定値が `0.0.0.0`）
  - `src/genie_server/config.py:244` と `src/genie_server/vlm.py:563-653`（`VLM_VISION_BUDGET_GUARD` の既定値が off）
- 問題: `Content-Type: text/plain` の POST は CORS 上 “simple request” として扱われ、preflight なしで送信されます。このサーバは Content-Type を見ずに JSON としてパースするため、**悪意ある Web ページを開いただけのブラウザ**から次の操作ができます。
  - `POST /v1/models/switch` で任意パスのモデルをロードする（既定は `unload_first=true` なので、失敗すればスロットが空になる）
  - `POST /v1/server/prompt_logprobs` や `/v1/lora/*`、`/v1/server/performance_policy` で設定を書き換える
  - 過大な VLM リクエストを送り、スロットを恒久的に wedge させるか、プロセスを落とす（コメントに「`free(): invalid next size`」と記録済み = ヒープ破壊）
  - さらに `Access-Control-Allow-Origin: *` なので、応答（プロンプトの中身、`/v1/prefix/cache` が返す絶対パスなど）も読み取れます。
  - 再現: `Content-Type: text/plain`、`Origin: http://evil.example` で `/v1/server/prompt_logprobs` に POST すると 200 が返り、設定が変わり、ACAO は `*` でした。
- なぜ問題か: SECURITY.md は「制御下のネットワークで動かす」ことを前提に認証なしを正当化しています。しかしこの経路は**ネットワークの外にいる攻撃者**が、ラボ内 PC のブラウザを経由して到達できます。また SECURITY.md 自身が「well-formed なリクエストで到達できるメモリ破壊・クラッシュ」を報告対象としている一方、その一例を既定値のまま出荷しています。
- 推奨修正:
  1. JSON を受けるエンドポイントでは `Content-Type: application/json` を必須にする（これだけで simple request 経由の CSRF は塞がる）。
  2. CORS の `allow_origins` を設定で指定できるようにし、既定値を空か `localhost` 系にする。
  3. 管理系エンドポイント（`/v1/models/switch`、`/v1/lora/*`、`/v1/server/*`（POST）、`/v1/prefix/*`）に、任意の共有トークン（例: `ADMIN_TOKEN`）を設けられるようにする。
  4. `HOST` の既定値を `127.0.0.1` にする案も検討する（計測器として外部公開が標準なら、ドキュメントで明示する）。

---

### High

#### H-1. async ハンドラ内のブロッキング呼び出しで、サーバ全体（全スロット、`/health`、ストリーミング）が停止する **[再現]**
- 箇所:
  - `src/genie_server/app.py:426` `slot.lock.acquire(timeout=cfg.warmup_join_timeout_s)` → **最大 600 秒**ブロック
  - `src/genie_server/app.py:437` `manager.switch_model(...)`（モデルロード。実機で数秒から数十秒）をイベントループ上で同期実行
  - `src/genie_server/app.py:1125-1129` `_locked_slot`（LoRA 3 種と performance_policy POST。最大 `inference_timeout_s` = 既定 120 秒）
  - `src/genie_server/app.py:1221`, `:1238`（`timeout=1.0`）
  - `src/genie_server/app.py:738` `vlm.decode_media_sources(sources)`（数百フレームの JPEG デコードをループ上で実行）
- 問題: `threading.Lock.acquire` と重い同期処理を `async def` の中で直接呼んでいます。その間イベントループが止まるため、**別スロットの推論トークン配信**（`call_soon_threadsafe` 経由）や `/health`、ステータスのポーリングまで止まります。再現では、生成中のスロットに `/v1/lora/apply` を送ると、`/health` の応答に 3.3 秒かかりました（`inference_timeout_s=4` の設定で、watchdog がロックを解放するまで）。
- なぜ問題か: マルチスロットは「別スロットは独立して動く」ことが売りです（`slots.py` 冒頭のコメント）。管理操作 1 回で全体が固まると、TTFT/TPS 計測値が汚染され、外部の監視からはハングに見えます。
- 推奨修正: ロック取得とモデルロードは `await asyncio.to_thread(...)` に逃がす（`/v1/server/idle` は既にこのパターンで実装されている: `app.py:915-921`）。画像のデコードも `to_thread` に移す。

#### H-2. 推論タイムアウトが「正常終了」として返る（途中で切れた出力が 200 / `finish_reason: "stop"`） **[再現]**
- 箇所:
  - `src/genie_server/engine.py:192`（watchdog が `generation.abort` を呼ぶ）
  - `src/genie_server/engine.py:241-254`（`ret == WARNING_ABORTED(1)` を `"stop"` にマップ）
  - `src/genie_server/capi.py:84-89`
  - `src/genie_server/app.py:313-322`（同期経路の待ち時間は `inference_timeout_s * 2` なので、先に watchdog が発火する）
- 問題: watchdog による中断も SDK からは warning 1 で返ります。これが通常の `stop` と区別されず、**途中までのテキストが成功応答として返ります**。再現結果は `status 200, finish_reason "stop", content "partial "` でした。ストリーミングでも同様で、エラーイベントは出ません。
- あわせて、504 を返す経路（`app.py:319-322`）は `gen.abort()` を呼ばないため、クライアントには失敗を返した後もワーカーがロックを待ち続け、取得後に生成を最後まで走らせます。非ストリーミング要求では切断も検知しないので、放棄されたリクエストがスロットを占有し続けます。
- なぜ問題か: lm_eval の生成タスクでは、途中で切れた出力が正常なサンプルとして採点されます。「SDK の問題を隠さない」という README の原則 3 に正面から反します。
- 推奨修正: `Generation` に `timed_out` フラグを持たせ、watchdog 起因の中断は明示的なエラー（504 または `finish_reason: "length"` とログ）にする。504 の経路と非ストリーミングの切断時にも `gen.abort()` を呼ぶ。

#### H-3. 2 つ目以降の system メッセージが黙って捨てられる **[再現]**
- 箇所: `src/genie_server/templates.py:233-250`（`sys_msgs[0]` だけを prefix にし、`non_sys` からは全 system を除外）
- 問題: system メッセージが 1 つでもあれば prefix cache 分割の経路に入り、`sys_msgs[1:]` は**プロンプトから消えます**。会話途中の system メッセージ（エージェント系クライアントや Open WebUI の一部機能が使う）も失われ、位置も先頭に寄せられます。再現では `[system A, user, assistant, system B, user]` を送ると、SDK に渡ったプロンプトに `SYS-B` が含まれていませんでした。
- なぜ問題か: モデルに届く指示が黙って変わります。chatml と llama3 は `render_chat_prompt` 単体なら全メッセージを順番どおり描画できるので、分割経路だけの不具合です。
- 推奨修正: 分割できるのは「先頭が system で、かつ system が 1 つだけ」の場合に限り、それ以外は `cacheable=False` にして `render_chat_prompt(messages)` にフォールバックする。テストも追加する。

#### H-4. サンプラー設定が次のリクエストに漏れる（greedy の top-k=1 や seed が残る。VLM では全パラメータが残る） **[再現]**
- 箇所:
  - `src/genie_server/capi.py:118-140`（`make_sampler_params`）
  - `src/genie_server/vlm.py:678-679`（`make_sampler_params({}, ...)`: VLM ではモデル既定値が空）
- 問題: docstring 自身が「`GenieSampler_applyConfig` は部分マージなので、毎回完全なパラメータセットを適用しないと前のリクエストの値が残る」と書いています。それなのに次のように漏れます。
  - `temperature<=0` で `top-k: "1"` が設定された後、次のリクエストが `temperature=0.7` だけを指定し、かつモデルの `genie_config.json` の sampler に `top-k` が無い場合、`top-k` は送られず **1 のまま残り、greedy が継続します**。再現結果: `make_sampler_params({}, temperature=0.7)` は `{'type': 'basic', 'temp': '0.7'}` を返しました。
  - `seed` は指定がなければ何も送らないため、前のリクエストの seed が残ります。
  - VLM の経路は既定値として `{}` を渡すので、temp/top-k/top-p/seed の**すべて**が前のリクエストから引き継がれます。
- なぜ問題か: 同じリクエストでも、直前に誰が何を送ったかで分布が変わり、再現性が壊れます。SECURITY.md の「あるリクエストのデータが別のリクエストへ漏れること」の精神にも反します。
- 推奨修正: スロットの初期化時に「SDK 既定値を含む完全なベースライン」（top-k=0/無効、top-p=1.0、temp=モデル既定値）を決め、毎回それをベースに上書きする。VLM でも text-generator ノード設定の sampler 既定値を読み込む。

#### H-5. リクエスト計画とロック取得の間の競合（TOCTOU）で、別モデルや別 LoRA の prefix KV を復元しうる **[コード]**
- 箇所:
  - `src/genie_server/app.py:842-873`（テンプレート、`max_tokens`、`cache_key = key(prefix, slot.cache_namespace)` をロック外で確定）
  - `src/genie_server/engine.py:166-169` と `:215-219`（ロックを取った後で、その key のまま restore）
  - `src/genie_server/app.py:426-437` と `:1152-1159`（model switch と LoRA apply が同じロックを奪い合う）
- 問題: `threading.Lock` は FIFO ではありません。推論ワーカーがロック待ちの間に `/v1/lora/apply` や `/v1/models/switch` がロックを先に取ると、ワーカーは次の状態で走ります。
  1. 旧モデルのチャットテンプレートでレンダリングしたプロンプト
  2. 旧コンテキスト長から算出した `max_tokens`
  3. **旧名前空間（旧 LoRA/旧モデル）の prefix cache key** で `GenieDialog_restore` を実行
  4. 応答の `model` 名も旧モデルのまま
  - LoRA 切り替えは KV の形状が同じため、エラーにならず**誤った KV で生成が続く**可能性が高いです。
  - `unload_first` の切り替えに失敗して `slot.handle is None` になった後も、ワーカーは `lib.reset(None)` や `lib.query(None, ...)` を呼びます（`require_loaded` がロック外でしか検査されない: `app.py:814`）。
- なぜ問題か: SECURITY.md が報告対象に挙げている「prefix-cache namespacing across models or LoRA adapters」にそのまま該当します。
- 推奨修正: 計画（テンプレート、名前空間、`max_tokens`）をロック取得後に確定するか、ロック内で `slot` の世代番号（swap や LoRA のたびに増やす）を照合して不一致なら 409 や再計画にする。`require_loaded` もロック内で再チェックする。

#### H-6. VLM 入力に上限がなく、メモリ枯渇 DoS が可能 **[コード]**
- 箇所:
  - `src/genie_server/vlm.py:407-419`（`Image.open(...).load()`）
  - `src/genie_server/vlm.py:422-433`（全フレームを一括デコードし、同時に保持）
  - `src/genie_server/vlm.py:503-510`（フレーム数に上限なし）
  - リクエストボディサイズの上限はどこにもない
- 問題: Pillow の既定の `MAX_IMAGE_PIXELS`（約 8,900 万画素で警告、約 1.79 億画素でエラー）の手前まではロードが通るため、1 枚で数百 MB を確保できます。video_url にフレームを大量に並べると、guard が off（既定）ならそのまま全フレームをデコードします。しかもそれがイベントループ上で行われます（H-1）。
- 推奨修正: 1 リクエストあたりのフレーム数、1 画像あたりの画素数（`Image.MAX_IMAGE_PIXELS` を明示的に下げる）、ボディサイズ（uvicorn の前段、または middleware）に上限を設け、base64 デコードも `validate=True` にする。

---

### Medium

#### M-1. 生成パラメータの型を検証しておらず、500 やサイレントな誤動作になる **[再現]**
- 箇所: `src/genie_server/app.py:95-102`（`temperature`, `top_p`, `top_k`, `seed` を素通し）、`:115`（`n`）、`:798-800`（`chat_template_kwargs`）、`:1179`（`alpha`）
- 再現結果:
  - `temperature: "hot"` → ワーカースレッドで `ValueError` → 500
  - `n: "2"` → `TypeError` → 500（しかも本文は OpenAI 形式ではない平文。M-7 参照）
  - `chat_template_kwargs: [1]` → 500
  - `top_k: 1.5` → `int()` で 1 に切り捨てられ、黙って greedy になる
  - logprobs 経路では、例外が `capi.py:535-537` で握りつぶされ、**毎ステップ token 0 が出力されます**（ログに error が出るだけで 200 が返る）。
- 推奨修正: `_parse_gen_params` で型と範囲を検証して 400 を返す（`isinstance(x, (int, float)) and not isinstance(x, bool)`、`0 <= top_p <= 1`、`top_k` は整数、など）。Pydantic のリクエストモデルに寄せるのも一案。

#### M-2. logprobs を付けるとサンプリング分布が変わる（モデル既定値を無視する） **[コード]**
- 箇所: `src/genie_server/app.py:833-835`, `:677-679`, `src/genie_server/logprobs.py:95-99`
- 問題: 通常経路は `genie_config.json` の sampler 既定値（temp/top-k/top-p）を補いますが、logprobs 経路の `LogprobsCollector` は未指定時に `temp=1.0`、top-k/top-p なしでサンプリングします。`logprobs: true` を足すだけで出力分布が変わります。
- 推奨修正: `slot.sampler_defaults` を collector に渡し、`make_sampler_params` と同じ規則で補完する。

#### M-3. スロット名の重複を検証していない **[コード]**
- 箇所: `src/genie_server/config.py:453-495`、`src/genie_server/slots.py:125-126`, `:397`、`src/genie_server/capi.py:529-530`
- 問題: 同じ `name` のスロットが 2 つあると、次のことが起きます。
  - `_by_name` の衝突（一方にしかルーティングされない）
  - `status` の上書き
  - **custom sampler のコールバック名（`genie-server-logprobs-{name}`）の衝突**: 2 つ目のスロットは登録がスキップされ、1 つ目のスロットのクロージャ、つまり 1 つ目の `active_collector` に logits が流れます。1 つ目で並行して logprobs リクエストが走っていれば、**他のリクエストの collector を汚染します**。
  - `.htp_ext_cache/{slot_name}_...` の上書き
  - TEXT と VLM の名前衝突もチェックしていない。
- 推奨修正: `load_config` で TEXT と VLM を合わせた名前の一意性を検証し、起動時エラーにする。

#### M-4. シャットダウン時の解放漏れと、解放の順序違反 **[コード]**
- 箇所: `src/genie_server/app.py:340-343`、`src/genie_server/slots.py:407-414`
- 問題:
  - `free_all()` はスロットのロックを取らずに `GenieDialog_free` を呼びます。デーモンスレッド（切断済みリクエスト、202 を返した warmup、504 後のワーカー）が `GenieDialog_query` を実行中なら use-after-free になります。
  - `GenieProfile`（`slot.profile`）は `free_profile` が一度も呼ばれません。VLM の `Node`/`Pipeline` も `free()` は定義されていますが呼ばれません（`grep` で確認）。ライフスパンのログは「Releasing HTP context memory」と言いつつ、VLM のコンテキストは解放していません。
- 推奨修正: 各スロットのロックを（タイムアウト付きで）取ってから解放する。profile と VLM のノード・パイプラインも解放する。

#### M-5. トークンコールバックで UTF-8 を `errors="ignore"` でデコードしている **[要確認]**
- 箇所: `src/genie_server/capi.py:437`（dialog は `ignore`）、`src/genie_server/genie_node.py:204`（VLM は `replace`）
- 問題: SDK が byte-fallback トークンなど、マルチバイト文字の途中で区切ってコールバックを返す場合、日本語や絵文字が**黙って欠落します**。2 つの経路で方針も一致していません。
- 推奨修正: `codecs.getincrementaldecoder("utf-8")` をリクエスト単位で保持し、分割されたバイト列を連結してからデコードする。SDK が常に完全な UTF-8 を返すなら無害なので、どちらにしても安全側の実装です。

#### M-6. 真偽値や列挙値の設定をパースするとき、typo が黙って通る **[コード]**
- 箇所:
  - `src/genie_server/config.py:553-558`（`bool(raw.get("PROMPT_LOGPROBS"))` なので、`"false"` が True になる）
  - `src/genie_server/app.py:414`（`unload_first`）、`:1005`（`enable_thinking`）
  - `src/genie_server/tool_formats.py:370-373`（未知の `TOOL_FORMAT` は黙って Hermes にフォールバック）
  - `CHAT_TEMPLATE` の typo は chatml にフォールバック（`templates.py:22-37`）
- 推奨修正: 真偽値は `isinstance(v, bool)` を必須にする（`_parse_poll` と同じ方針）。override の値は既知の集合で検証する。

#### M-7. 予期しない例外が OpenAI エラー形式にならない **[再現]**
- 箇所: `src/genie_server/protocol.py:215-241`（汎用の `Exception` ハンドラがない）
- 問題: モジュールの docstring は「Every error this server returns ... is rendered in OpenAI's error envelope」と明言していますが、M-1 の `n: "2"` では平文の `Internal Server Error` が返りました。
- 推奨修正: `@app.exception_handler(Exception)` を追加し、500 と `server_error` の形式で返す（内部の例外文字列はログにだけ出す）。

#### M-8. `/health` がスロットの状態に関係なく常に `ok` を返す **[コード]**
- 箇所: `src/genie_server/app.py:367-371`
- 問題: `unload_first` の切り替えに失敗して全スロットが空でも、VLM スロットが wedge していても `ok` を返します。監視やオーケストレーションから異常が見えません。
- 推奨修正: liveness（`/health`）とは別に、readiness（例: `/ready`。全スロットがロード済みなら 200、そうでなければ 503）を追加する。

#### M-9. prefix cache が際限なく増え、掃除する仕組みがない **[コード]**
- 箇所: `src/genie_server/prefix_cache.py:25-35`（名前空間が変わると古いエントリは「到達不能になるだけ」）、`:72-82`
- 問題: モデルや LoRA を切り替えるたびに到達不能なスナップショットが溜まります（KV スナップショットは 1 件あたり数十 MB から GB 級になりうる）。ディスクの上限も LRU もありません。
- 推奨修正: サイズ上限と LRU で削除する仕組みか、少なくとも「現在の名前空間に属さないエントリ」を一括削除する API を追加する。`DELETE /v1/prefix/cache/{key}` の key は 16 桁の hex であることを検証する（現状は Starlette のルーティングのおかげでパストラバーサルにはならないが、防御は多重にする）。

#### M-10. テンプレートごとのツール関連の扱いが揃っていない **[コード]**
- 箇所:
  - `src/genie_server/templates.py:157-168`（llama2 は `tool` ロールのメッセージを黙って破棄）
  - `:149-152`（llama3 は `slot.tool_format` を無視して Hermes で描画）
  - `:205-211`（gemma4 のツール結果が `m["name"]` を参照するが、OpenAI の tool メッセージは通常 `tool_call_id` しか持たないため、`response:{...}` の関数名が空になる）
  - `src/genie_server/tool_formats.py:263-268`（gemma4 は壊れた arguments を `{}` に置き換え、Hermes は生の文字列を保持する。arguments が配列なら `AttributeError` で 500）
- 推奨修正: gemma4 は、直前の assistant の `tool_calls[].id` から `tool_call_id` を引いて関数名を解決する。未対応のロールは 400 にするか、警告ログを出す。

#### M-11. Qwen3-VL の patchify が 2 フレーム固定なのに、`temporal_patch_size` は設定で変えられる **[コード]**
- 箇所: `src/genie_server/vlm_specs/qwen3_vl.py:39-43`, `:83`, `:264`
- 問題: `metadata.json` の `temporal_patch_size` が 2 以外だと、起動時には通り、リクエストのたびに reshape エラー（500）になります。
- 推奨修正: `qwen3vl_bind` で `temporal_patch_size == 2` を検証するか、patchify を N フレームに一般化する。

#### M-12. 依存関係のバージョンをまったく固定していない **[コード]**
- 箇所: `pyproject.toml:33-37`、`:44`、`:49`、`:55`（`fastapi`, `uvicorn`, `tokenizers`, `numpy`, `pillow` がすべて無制約）
- 問題: 「計測器」として再現性が要るのに、実行環境の依存バージョンが不定です。現時点で pip-audit は実行時依存に既知の脆弱性を検出しませんでした（検出されたのは venv 内の pip/setuptools のみ）。しかしテストは既に `StarletteDeprecationWarning`（httpx から httpx2 への移行）を出しており、将来の破壊的変更を受ける経路が開いています。
- 推奨修正: 下限（と必要に応じて上限）を指定し、ボード配布用には lock ファイル（`pip-compile` / `uv lock`）を用意する。Dependabot か Renovate も導入する。

#### M-13. CI が pytest だけで、静的解析がなく、サプライチェーンの防御も弱い **[コード]**
- 箇所: `.github/workflows/offline-tests.yml`
- 問題:
  - ruff と mypy がない。今回の実行では mypy が 30 件のエラーを出しました（その中には `gemma4.py:35-45` の `None` 算術や `app.py:1245` の `dict.get(None)` など、型の穴がある）。ruff は `F841`/`F401`/`B904`/`ASYNC110` などを出しました。
  - `actions/*@v4`/`@v5` がタグ指定で、SHA で固定していない。
  - `permissions:` ブロックがない（GITHUB_TOKEN が既定権限のまま）。
- 推奨修正: ruff と mypy（まずは `--ignore-missing-imports`）の job を追加し、`permissions: contents: read` を付け、アクションを SHA で固定する。

#### M-14. リスクが最も高い ctypes 層のテストが最も薄い **[再現]**
- カバレッジ（`coverage run -m pytest tests`）: **`capi.py` 33%、`genie_node.py` 30%**、`cli.py` 23%、`asgi.py` 0%。全体は 81%。
- さらに、今回見つけた H-2（watchdog、`grep watchdog tests/` は 0 件）、H-3（複数 system）、H-4（サンプラーの持ち越し）、H-5（競合）、H-1（ループのブロック）はテストでカバーされていません。
- 推奨修正: ctypes 層は `ctypes.CDLL` を差し替えるスタブ（関数ポインタを Python の CFUNCTYPE で差し込む）で、引数の型、バッファの寿命、`None` ハンドルの扱いを単体テストする。上の各バグには回帰テストを追加する。

---

### Low

| # | 箇所 | 問題 | 推奨 |
|---|---|---|---|
| L-1 | `README.md:209`, `:231`, `README.ja.md:228` | 「311 offline tests」とあるが実際は 505 件。「CI は 3.10 と 3.12」とあるがワークフローは 3.10/3.12/3.14 | 数値をハードコードせず、CI バッジなどに寄せる |
| L-2 | `src/genie_server/slots.py:66-71` | `ModelAssets` の docstring が「新モデルを先にロードするので slot が空になることはない」と書いているが、既定の `unload_first=True` と矛盾する | docstring を実装に合わせる |
| L-3 | `src/genie_server/vlm_layout.py:85`, `:450`、`vlm_specs/qwen3_vl.py:8`, `:33` | リポジトリに存在しない `.claude/rules/*.md` と `preprocess.py` を参照している | 参照先をコミットするか、記述を削る |
| L-4 | `src/genie_server/genie_node.py:7-10`, `:96-98` | 実在しない `vlm_stream.py` と「genie-server.py's `genie_lib`」を参照（旧構成の名残） | 更新する |
| L-5 | `src/genie_server/config.py:265` | 「Derived timeouts (kept relative to inference_timeout_s)」とあるが、3 つのうち 2 つは定数 | コメントを直すか、設定可能にする |
| L-6 | `src/genie_server/app.py:994`, `:1236`, `:1253` | warmup と performance_policy だけ `slot` 指定ができない（`model` だけ）。同じモデルを載せた 2 つ目のスロットを操作できない | `select_for_request` に統一する |
| L-7 | `src/genie_server/app.py:615-618` と `:79` | echo+logprobs の経路だけ `max_tokens` を `max_completion_tokens` より優先しており、他の経路と逆 | 解決済みの `params.max_tokens` を使う |
| L-8 | `src/genie_server/app.py:671-672` | 非ストリーミングで複数プロンプトを送ると、コンテキスト超過チェックがプロンプトを順に実行する途中で走る（前のプロンプトの計算が無駄になる） | 全プロンプトを先に検証する |
| L-9 | `src/genie_server/templates.py:105-124` ほか | ユーザーの content に含まれる `<\|im_end\|>` などの特殊トークン文字列をそのまま連結する（ロール注入） | 計測器としての許容範囲だが、MANUAL に明記するか、エスケープオプションを設ける |
| L-10 | `src/genie_server/app.py:441`, `:1086`、`protocol.py` | 例外文字列やモデルの絶対パスをそのままクライアントに返す | 詳細はログにだけ出す（C-1 と併せて対応） |
| L-11 | `src/genie_server/genie_node.py:207` | コールバックのエラーを `print` で出しており、logging を通らない | `logger.error` にする |
| L-12 | `src/genie_server/cli.py:81` | `args.port or config.port` のため `--port 0` を指定できない | `is not None` で判定する |
| L-13 | `src/genie_server/slots.py:418-424` | 未知の `model` 名は黙ってプライマリスロットに回る（typo が検出されない） | 設計上の選択としては妥当。ただし `KNOWN_MODEL_ID` 以外の未知名はログで警告するか、strict モードを設ける |
| L-14 | `src/genie_server/app.py:271`（ruff ASYNC110） | `while ...: await asyncio.sleep(0.05)` のビジーウェイト | `asyncio.to_thread(gen.done.wait, timeout)` にする |

---

## 3. アーキテクチャ上の所感

### 強み
- **SDK の境界がはっきりしている**: ctypes は `capi.GenieLib` と `genie_node` に閉じ込められています。`FakeGenieLib` が同じ API 面を持つため、NPU なしで HTTP からエンジン、テンプレートまで通しで検証できます。この設計のおかげで今回の再現も容易でした。
- **設定がイミュータブル**: `ServerConfig` と `SlotSpec` は `frozen=True` で、起動時に一度だけ検証されます（`device_id` の型チェックなどは丁寧）。
- **拡張点がレジストリになっている**: `tool_formats.FORMATS` と `vlm_specs.FAMILIES`、`vlm_layout` の「バンドル自身のレイアウトを読む」優先順位付きの探索は、新しいモデルファミリーを足すときの変更範囲を最小にしています。
- **ドメイン知識の記録**: SDK の不具合、計測値、判断理由がコードの近くに残っていて、後から参加する人が「なぜこうなっているか」を追えます。README の「What this is for」で設計原則の優先順位を明文化している点も優れています。
- **OpenAI 互換の細部**（最初の role チャンク、`include_usage`、`context_length_exceeded`、ツール呼び出しのストリーミングフィルタ）への配慮が行き届いています。

### 弱み
- **並行性の規律が分散している**: `slot.lock` の取得が `engine.py`（ワーカースレッド）、`app.py`（イベントループ上で直接）、`app.py` 内の warmup スレッドの 3 系統に散らばっています。「ロック内で何を再検証すべきか」（ハンドル、名前空間、テンプレート）の規約がありません。H-1/H-2/H-5/M-4 はすべてこの構造から来ています。**スロット単位の作業キュー（1 スロット 1 ワーカースレッド + `asyncio.Future`）**に集約すれば、ロック、abort、タイムアウト、世代チェックを 1 か所で扱えます。
- **`app.py` が 1,271 行の単一クロージャ**: `create_app` の中にすべてのルートとヘルパーが入れ子で定義されていて、個別の単体テストや再利用が難しい構造です。completions と chat の重複（同じ 8 行のコメントブロックが 2 回出てくる: `app.py:576-583` と `:815-822`）もここから生まれています。ルートをドメイン別に分割（`APIRouter`: openai / admin / lora / prefix）し、リクエストのパースと検証をスキーマ層に分ける余地があります。
- **状態が可変で共有されている**: `manager.status`（生の dict）、`Slot` の各属性、`ServerState.prompt_logprobs_enabled` を、複数スレッドとイベントループから GIL 頼みで読み書きしています。今は壊れていませんが、ステータス更新（`_score_prompt` がロック取得前に他リクエストのフェーズを上書きする: `app.py:527-528`）のような小さな不整合を生んでいます。
- **入力検証の層がない**: FastAPI を使いながら Pydantic モデルを使わず、`body.get()` で手作業のパースをしています。M-1/M-6/M-7 の根本原因です（OpenAI の緩い互換性を保つためなら `extra="allow"` のモデルで足ります）。
- **VLM とテキストでライフサイクルが非対称**: VLM は abort もタイムアウトも解放処理も持たず、テキストスロットとは別の暗黙ルールで動いています。SDK の制約による部分は仕方ありませんが、サーバ側でできること（タイムアウト時にスロットを「wedge 疑い」として報告する、解放する）も揃っていません。

---

## 4. クイックウィン（すぐ直せて効果が高いもの）

1. **JSON エンドポイントで `Content-Type: application/json` を必須にする**（`protocol.read_json_body`）。C-1 の CSRF 経路が数行で塞がります。
2. **lock の acquire とモデル切り替えを `asyncio.to_thread` に移す**（`app.py:426-444`, `:1125-1129`, `:1221`, `:1238`, `:738`）。H-1 が解消します。
3. **watchdog による中断を成功扱いしない**（`Generation.timed_out` を追加し、504/`length` とログにする）。504 の経路で `gen.abort()` を呼ぶ（H-2）。
4. **system が複数あるときは prefix 分割しない**（`templates.split_prompt_for_prefix_cache`）と回帰テスト（H-3）。
5. **`make_sampler_params` に完全なベースライン（top-k/top-p/seed のリセット値）を持たせる**。VLM ではノード設定の既定値を渡す（H-4）。
6. **`_parse_gen_params` で型と範囲を検証**し、汎用の `Exception` ハンドラで OpenAI 形式の 500 を返す（M-1, M-7）。
7. **設定の真偽値と列挙値を厳格に検証**し、スロット名の一意性をチェックする（M-3, M-6）。
8. **VLM のフレーム数、画素数、ボディサイズに上限**を設ける（H-6）。
9. **CI に ruff と mypy を追加**し、`permissions: contents: read` を付け、アクションを SHA で固定する（M-13）。
10. **README のテスト件数と CI の Python バージョン、存在しないファイルへの参照を修正**する（L-1, L-3, L-4）。

---

## 5. 中長期的な改善提案

1. **スロット実行モデルの再設計（最優先）**: 各スロットに専用のワーカースレッドとジョブキューを持たせ、HTTP 層は `await future` するだけにします。ジョブには「計画時点のスロット世代番号」を持たせ、実行時に照合します（H-5 の根本対策）。abort、タイムアウト、キュー長の上限（バックプレッシャ）、`/v1/server/status` のフェーズ管理もここに集約できます。管理操作（switch/LoRA/policy/warmup）も同じキューに流せば、ロック取得コードが HTTP 層から消えます。
2. **リクエストスキーマ層の導入**: `ChatCompletionRequest` と `CompletionRequest` を Pydantic（`extra="allow"`）で定義し、型、範囲、既定値の解決（`max_completion_tokens` の優先順位など）を 1 か所にまとめます。OpenAPI ドキュメントも自動で正確になります。
3. **`app.py` の分割**: `routers/openai.py`、`routers/admin.py`、`routers/lora.py`、`routers/prefix.py` に分け、completions と chat の共通処理（ストリーミング／非ストリーミング、usage、エラー）をサービス層に切り出します。
4. **セキュリティの「オプトイン強化モード」**: 設計原則（計測器、認証なし）は尊重したまま、`ADMIN_TOKEN`、`CORS_ALLOW_ORIGINS`、`MODEL_ROOT_ALLOWLIST`（オプション）、VLM の入力上限を設定で有効にできるようにします。SECURITY.md の「design decisions」は、ブラウザ経由の CSRF という脅威モデルを追記したうえで再評価してください。
5. **ctypes 層のテスト戦略**: `ctypes.CDLL` を `CFUNCTYPE` のスタブで置き換える「フェイク共有ライブラリ」ハーネスを作り、引数型、バッファ寿命（`get_value_string` / `get_profile_json` の alloc callback、`Node.set_buffer` の `_keep`）、`None` ハンドル、UTF-8 境界を検証します。可能なら小さな C のスタブ `.so` を CI でビルドし、実際の ABI を通す統合テストにします。
6. **依存と配布の再現性**: lock ファイル、ボード向けのオフライン wheelhouse、Dependabot を用意します。計測レポート（`tests/integration/`）に依存バージョンと QAIRT のバージョンを自動で記録すると、計測器としての価値がさらに上がります。
7. **運用の可観測性**: readiness エンドポイント、Prometheus 形式のメトリクス（スロットごとの待ち行列長、TTFT/TPS、wedge 疑いの回数、prefix cache のサイズ）、構造化ログ（request_id をキーにする）を追加します。prefix cache には容量上限と LRU を入れます。
8. **テンプレートの正確性**: 可能なら、バンドルの `tokenizer_config.json` の `chat_template`（Jinja）を直接レンダリングするモードを追加し、手書きテンプレートとの差分を CI で比較します。複数 system、tool ロール、tool_call_id の扱いのような、手書きの抜けを構造的に防げます。

---

## 付録: 実行した検証

| 項目 | 結果 |
|---|---|
| `pytest tests/`（Python 3.11、`.[logprobs,vlm,test]`） | **505 passed**、1 warning（Starlette TestClient の httpx 非推奨警告） |
| coverage | 全体 81%（`capi.py` 33%、`genie_node.py` 30%、`cli.py` 23%、`asgi.py` 0%） |
| `ruff check --select E,F,W,B,UP,SIM,ASYNC` | 157 件（E501 が 134 件、実害候補: F841, F401, B904 ×5, ASYNC110, B905） |
| `mypy --ignore-missing-imports src/genie_server` | 8 ファイルで 30 件のエラー（`None` 算術、`Optional` の未処理、Pillow の定数の型スタブなど） |
| `pip-audit` | 実行時依存（fastapi 0.141.1 / starlette 1.7.0 / uvicorn 0.54.0 / tokenizers 0.23.2 / numpy 2.4.6 / pillow 12.3.0）に既知の脆弱性なし。検出は venv の pip/setuptools のみ |
| 再現スクリプト（フェイク SDK） | C-1、H-1、H-2、H-3、H-4、M-1、M-7 を確認（本文の **[再現]** 印） |
| 実機（Hexagon NPU / `libGenie.so`） | **未実施**（環境がないため）。M-5 と H-5 の実害の程度は実機での確認が必要 |
