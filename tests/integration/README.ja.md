# ホスト側統合テスト

*[English](./README.md) | 日本語*

**ホストPC**から、実機で稼働中のopen-genie-serverに対してREST API経由で全機能領域を検査し、Markdown + JSONのレポートを書き出します。実行途中でサーバが落ちた場合は、どのテスト中に停止したか・実行中だったリクエスト・最後に成功したテストをレポートに記録し、残りのテストはABORTED扱いになります。

## 準備

ホストPC側:

```bash
pip install requests
cp test_config.sample.json test_config.json   # パスやトグルを編集
```

実機側: [examples/config/](../../examples/config/) のサンプル(または自前の`env_config.json`)でopen-genie-serverを起動しておきます。

## 実行

```bash
python3 run_integration_tests.py --config test_config.json
# configを編集せずに接続先だけ上書き:
python3 run_integration_tests.py --config test_config.json --base-url http://192.168.1.2:8080

python3 run_integration_tests.py --list          # テストID一覧
python3 run_integration_tests.py --only C01,L03  # 一部だけ実行
```

レポートは `./reports/integration_report_<timestamp>.{md,json}` に出力されます。終了コード: 0=全PASS、1=FAILまたはサーバ停止あり、2=開始時点でサーバに到達不能。

## カバー範囲

| ID | 領域 |
|---|---|
| S01-S04 | health、モデル一覧(`/v1`なしエイリアス含む)、スロット別ステータス、idle待ち |
| C01-C07 | completions: 同期/echo/バッチ/ストリーミング/stopシーケンス/greedy再現性/`finish_reason` |
| CH01-CH05 | chat: 同期/ストリーミング+usage/`enable_thinking=false`/prefix KVキャッシュのwarmup+HIT/function calling往復 |
| D01 | ストリーミング中にクライアントが切断したら生成を中断する(放置して完走させない) |
| E01 | OpenAIエラーエンベロープ: 不正JSON、`n>1`、`stream`+`logprobs`、存在しないスロット |
| L01-L03 | logprobs(chat + completions)とプロンプトスコアリング(lm_eval loglikelihood形状。無効時→400のゲート確認込み) |
| P01, P02 | パフォーマンスポリシー: 1往復と、`Genie_PerformancePolicy_t` 全9値の往復 + 未知の名前が拒否されること。ポリシーが実際に効くかは `measure_perf_policy.py` が別途計測(SDK自身のKPIを読むため `GENIE_PROFILE=true` が必要) |
| M01, M02 | モデルホットスワップ(+復元)、LoRA適用/解放 — **config設定時のみ** |
| V01-V07 | VLM: 画像チャット、ストリーミングSSE、ストリーム↔同期の一致、usage集計、切断後のスロット復帰(abort APIが無いので生成自体は完走する)、動画入力(2フレームずつ1エンコーダステップに詰まること、およびそれを裏付けるusage)、視覚入力の予算ガードの400 — **config設定時のみ**。V07はさらにサーバを`VLM_VISION_BUDGET_GUARD`をONにして起動している必要がある |
| G01-G08 | Grammar制約デコーディング: JSON Schema(同期/ストリーミング/連続実行)、マスク下のlogprobs、正規表現、EBNF、非対応バックエンドの拒否、復元 — **config設定時のみ**、下記参照 |
| Z01 | 実行後のサーバ生存確認 |

configに設定がない領域(`switch`/`lora`/`vlm`)はFAILではなくSKIPとして報告されます。モデル依存の挙動(ツールを実際に呼ぶか、`<think>`の扱い)は、サーバの不具合ではないためnote付きPASSになります。

## ローカルドライラン(実機なし)

`fake_server.py` はユニットテストと同じfake SDKで同一のFastAPIアプリを起動します。ハーネス自体の開発・確認用です:

```bash
python3 fake_server.py --port 18080 &
python3 run_integration_tests.py --base-url http://127.0.0.1:18080
```

## Grammarテスト(G01-G08)

Grammarはモデル/スロット単位で固定されるため、種別ごとに別のモデルディレクトリが
必要になる。実機上で生成する(重いアセットは全てベースバンドルへの絶対パス参照なので
各数KBしかない):

```bash
adb push setup_grammar_models.py /home/root/
adb shell "cd /home/root && .venv/bin/python3 setup_grammar_models.py \
    --base /home/root/models/qwen3_0_6b-genie-w4a16-qualcomm_sa8775p \
    --out  /home/root/grammar_test"
```

そのうえで `test_config.json` の `grammar.enabled` を `true` にする。各G0xは必要な
ディレクトリへ自分でスロットを切り替え、G08が `grammar.restore_model_dir` に戻す。

**これらのテストには `ENABLE_GRAMMAR` 付きでビルドされた `libGenie.so` が必要。**
XGrammarバックエンドはQualcommのprebuiltライブラリの中にしか無い
(`GrammarBackend::create` は `qualla/grammar.hpp` で宣言されているが、配布ソースに
定義が無い)。`examples/Genie/` から再ビルドしたライブラリにはgrammar機能が入らず、
G0xのモデルロードは全て `GenieDialog_create failed: -1` になり、サーバログに
`"Grammar backend configured but qualla was built without ENABLE_GRAMMAR"` が出る。
