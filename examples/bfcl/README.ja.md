# open-genie-server に対して BFCL を実行する

*[English](./README.md) | 日本語*

[Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
は function calling を採点するベンチマークです。OSS 向けハンドラは OpenAI の
`/v1/completions` を話すので、本サーバに対してそのまま実行できます。

| ファイル | 内容 |
|---|---|
| `run_bfcl.sh` | ラッパ。サーバ疎通確認 → BFCL の向き先設定 → 生成 → 評価 → スコア表示 |
| `subset_ids.py` | BFCL の ID ファイルを書き出し、カテゴリの先頭 N 件だけを対象にする |
| `bfcl_analyze.py` | 失敗を「マーカーが化けた」と「呼び出しを間違えた」に分類する |
| `bfcl_marker_cost.py` | マーカーだけ補修したコピーを作り、BFCL自身の採点器でマーカーの代償を測る。**分析であってスコアではない** |
| `bfcl_format_cost.py` | 同じ発想を「**形**が違う」モデルに適用したもの。認識できる JSON 形式の呼び出しを、prompting モードが要求する `[func(param=value)]` 構文に書き換える |
| `gemma4_handler.py`, `gemma4_fc_handler.py`, `install_gemma4_handler.sh` | gemma4 を BFCL に登録する(prompting 版とネイティブ関数呼び出し版)。[gemma4](#gemma4) を参照 |
| `prompt_lengths.py` | BFCL が送るプロンプトのトークン長分布をカテゴリ別に出す(指定コンテキストとの比較つき)。オフラインで完結し、ボード不要 |

## 数字を読む前に必ず読むこと

**BFCL は本サーバのチャットテンプレートとツールパースをバイパスします。**
各ハンドラがプロンプトを**自前で組み立て**(Qwen3 系ハンドラは HF tokenizer の
チャットテンプレートから Hermes 形式、下記 [gemma4](#gemma4) のハンドラは
gemma4 自身のマーカー)、`/v1/completions` に生テキストとして送り、応答から
呼び出しを**自前でパース**します。`/v1/chat/completions`・`tools`・
`tool_choice`・`TOOL_CALL_RECOVERY` はどれも関与しません。**サーバ側の方言
レジストリも関与しません** — ハンドラとサーバが同じ方言を実装していても実装が
食い違うことはあり得るので、BFCL のスコアはサーバのツール対応の検査ではありません。

リーダーボードとしては正しい設計(サーバ実装によらずモデルを測れる)ですが、
ここでは重要な帰結があります。

**`<tool_call>` マーカーが不安定なモデルは実力より大幅に低い点数になり、
本サーバの回収機能では救えません。** 当ボードの `qwen3_4b_instruct_2507`(w4a16)は
呼び出しの約半数で `<tool_call>` トークンがキリル文字に化けます。呼び出しの中身は
正しく `/v1/chat/completions` なら回収できますが、BFCL が見るのはこうなります:

```
ФРАГМЕНТ
{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5, "unit": "units"}}
ФРАГМЕНТ
```

そして `Wrong number of functions.` と採点されます。**正しい呼び出しが、マーカーが
無いという理由だけで不正解になります。**

当ボードでの実測(`simple_python` 全400件、**両モデルとも BFCL の `-FC` ハンドラ**、
現行の単一CL再export版):

| Python Simple AST | BFCL の採点結果 | マーカー補修後(再採点値) | bf16参照 |
|---|---|---|---|
| `qwen3_4b_instruct_2507` | **48.50%** | **88.75%** | 95.75% |
| `qwen3_1_7b` | **90.25%** | *(補修対象なし)* | 92.00% |

補修列は `bfcl_marker_cost.py` で復元したものを BFCL 自身の採点器にかけ直した値、
bf16列は同じ2モデルをホストPC(GPU)で走らせた参照値です(ボードではありません)。

4B の失敗206件のうち **186件(90.3%)がマーカー起因** — 完全で名前も正しい呼び出しが
BFCL からは見えないだけです。実際に呼び出しを間違えたのは19件、呼び出しを出さなかったのが1件。
一方 **1.7B にはマーカー起因の失敗が1件もありません**(39件すべてが本物の誤り)。

**マーカーの代償は 4B で 40.25ポイント**(採点値 48.50%、タグを戻すと 88.75%)。
BFCL は 4B を 1.7B より 41.75ポイント下に並べますが、呼び出しが読める状態にすると
差は 1.5ポイントです。**どちらを載せるべきかという結論は間違っていませんが、
差の大きさとその理由は間違っています。**

補修しても全部は戻りません。88.75% は同じモデルの bf16 参照になお 7.00ポイント届かず、
この残差はマーカー起因ではありません。引用するのは 88.75% で、
`bfcl_analyze.py` が出す `at most 95.00%` **ではありません** — あちらは上限推定であって、
再採点した補修版が実測値です。

### 引用するときに踏む2つの落とし穴

**スコアは「モデル」ではなく「1つの export」を指しています。** 同じ2モデルの
旧・複数CL export は、同じカテゴリで **38.50%**(4B。補修後 79.00%)と **55.25%**(1.7B)でした。
1.7B の 55.25% は量子化による劣化ではありません — あの export は thinking を
出さなくなっており(1件あたり平均42出力トークン、現行は328)、**思考しないモデルを
思考するモデルと比べていた**だけです。再exportで35ポイント戻りました。
**export をまたいで数値を持ち回らないこと。思考の有無が違う実行同士を比べないこと。**

**スコアは「1つの BFCL モード」も指しています。** 上表はすべて `-FC` です。
同じ 1.7B を prompting ハンドラで走らせると **87.00%** で、ここでは近い値ですが、
それはこのモデルについての事実であって一般則ではありません
([gemma4](#gemma4) ではモード差が24ポイントあります)。
CSV の `Model` 列末尾に `(FC)` / `(Prompt)` が出るので、
**2つの数値を同じ表に並べる前に必ず確認**してください。

### `TOOL_CALL_RECOVERY` を使えばスコアは上がるか — 上がりません。ただしアプリからは変わります

**BFCL の数字は変わりません。** 回収機能は `/v1/chat/completions` の中にあり、
BFCL は `/v1/completions` を使って自前でパースするため、この経路に回収機能は存在しません。
ONにして再実行しても 48.50% のままです。

**実クライアントから見える挙動は、上表のぶんだけ変わります。** 4B のマーカー起因失敗
186件は**すべて、リクエストが宣言したツール名を名乗っています** — これは回収機能が
判定条件にしているものそのものです。したがって186件すべてが、chat エンドポイントなら
`message.tool_calls` として返る呼び出しです。`/v1/chat/completions` に `tools` を渡す
アプリケーションは 88.75% 相当の挙動を得ます。BFCL が 48.50% と報告するのは、
**測っているものが違う**からです。

引用するときは両者を混同しないでください。**48.50% がこのモデルの BFCL スコア**、
**88.75% は呼び出しが読める状態での価値**であり、本サーバ経由で運用したときの数字です。

したがって **このボードで出した BFCL スコアはモデルの function calling 能力の
下界であり、この 4B export では相当に緩い下界です。** 公開リーダーボードと同じものを
測っているかのように比較しないでください。**採点前にマーカーを「修復」することも
しないでください** — 化けは実際のモデル出力であり、消してから採点するのは結果の
偽装になります。

この欠陥ではなく function calling 能力を測りたい場合は、マーカーが安定している
モデルで実行してください。`Qwen/Qwen3-1.7B-FC` は BFCL に登録済みで、上表のとおり
**現行の 1.7B export は BFCL 自身のプロンプト400件でマーカー起因の失敗がゼロ**でした。

## 1. インストール

```bash
python3 -m venv /tmp/bfclvenv
/tmp/bfclvenv/bin/pip install bfcl-eval soundfile
```

`soundfile` は省略できません。`bfcl_eval` が `qwen_agent` を import し、そこが
無条件に `soundfile` を import するため、無いと CLI が起動しません。

ボード上ではなく、ボードに HTTP で到達できるホストで実行してください。

初回実行時にモデルの tokenizer と config を Hugging Face から取得します
(数MB。重みは落としません)。レート制限に当たるなら `HF_TOKEN` を設定するか、
ホストがオフラインなら `bfcl generate` に `--local-model-path` を渡してください。

## 2. 評価したいモデルをロードする

BFCL は自前のモデルID(`Qwen/Qwen3-4B-Instruct-2507`)を送りますが、これは
どのスロット名とも一致しないため、**プライマリスロットに載っているものが応答します**。
`run_bfcl.sh` はそれを表示します。**パイプラインの他のどこもこれを教えてくれない**ので、
必ず確認してください。

```bash
curl -sS "$BASE_URL/v1/server/status" | python3 -m json.tool
```

BFCL のモデルIDは、**ロード中のモデルとチャットテンプレートが一致するもの**を選びます
(プロンプトを組むのはこのテンプレートです)。`bfcl models` で一覧が出ます。
`-FC` 付きはネイティブ function calling 用テンプレート、無しはプロンプト方式です。

## 3. 実行

まず小さく始めてください。1カテゴリは数百〜千件あり、各件が NPU での実生成です。

```bash
BFCL=/tmp/bfclvenv/bin/bfcl ./run_bfcl.sh \
    --base-url http://<board>:8080 \
    --categories simple_python \
    --subset 8 \
    --workdir /tmp/bfcl-smoke
```

カテゴリ全体:

```bash
BFCL=/tmp/bfclvenv/bin/bfcl ./run_bfcl.sh \
    --base-url http://<board>:8080 \
    --categories simple_python \
    --workdir /tmp/bfcl-simple
```

1件あたりの所要時間は**モデルが何トークン出すか**でほぼ決まります。デコード速度は
測定したどのモデルでも当ボードで 19〜20 tok/s なので、**思考するモデルは
しないモデルの数倍**かかります。

| 実行(シングルスレッド) | 秒/件 | 400件の所要 |
|---|---|---|
| `qwen3_4b_instruct_2507`、`-FC` — 思考しない | 4.94 | 約33分 |
| `qwen3_1_7b`、prompting — 92%の件で思考、出力中央値179トークン | 10.2 | 約68分 |
| `gemma4-e2b-it`、`-FC` — 思考しない | 2.3 | 約15分 |

4B の値は旧 export での実測ですが、4B は新旧とも非thinkingなのでそのまま使えます。
1.7B の旧 export は 2.53秒/件 で、現行の4倍速かったのは**思考しなくなっていたから**です。
他カテゴリは件数で按分してください。

| カテゴリ | 件数 |
|---|---|
| `simple_python` | 400 |
| `multiple` | 199 |
| `parallel` | 199 |
| `irrelevance` | 239 |
| `live_simple` | 257 |
| `live_multiple` | 1053 |

`bfcl test-categories` で全一覧が出ます(`all` / `ast` / `live` などの
まとめ指定も含む)。

**`--threads 1` のままにしてください。** テキストスロットはどのみち逐次処理なので
並列化しても速くならず、タイムアウトの原因切り分けが難しくなるだけです。

## 4. スコアの読み方

`run_bfcl.sh` が最後に CSV を表示します。部分実行では
**`Overall Acc` ではなくカテゴリ別の列を読んでください** — 全体値は実行していない
カテゴリを 0 として平均するためです。8件の `simple_python` で 50% だった実行は、
全体では `0.42%`、`Python Simple AST` 列で `50.00%` と出ます。

失敗は中身を見る価値があります。`$WORKDIR/score/**/..._score.json` に各失敗の
**生のモデル出力**が入っており、上のマーカー化けもこれで特定できました(推測ではなく):

```bash
python3 - <<'EOF'
import glob, json
for f in glob.glob("/tmp/bfcl-simple/score/**/*_score.json", recursive=True):
    for i, line in enumerate(open(f)):
        d = json.loads(line)
        if i == 0:
            print(f, d); continue
        print(d["id"], d.get("error"), repr(str(d.get("model_result_raw"))[:120]))
EOF
```

## このサーバ固有の注意

- **コンテキストは 4096 トークンだが、ほとんど制約にならない**。BFCL はプロンプトに加えて
  最大 4096 の `max_tokens` を要求します(サーバではなく HF config の
  `max_position_embeddings` = このモデルでは 262144 から算出するため)。明示された
  `max_tokens` はそのまま尊重され、コンテキストが埋まった時点で生成が止まるのでここは無害です。
  **プロンプト単体**がコンテキストを超えるエントリは `400 context_length_exceeded` に
  なりますが、BFCL が実際に組み立てるプロンプトを実測したところ、該当するのは
  **`live_multiple` の1053件中2件(0.2%)だけ**で、`simple_python` / `multiple` /
  `parallel` / `live_simple` / `live_parallel_multiple` には**1件もありません**:

  | カテゴリ | 中央値 | p90 | p99 | 最大 | 4096超 |
  |---|---|---|---|---|---|
  | `simple_python` | 240 | 292 | 354 | 394 | 0 |
  | `multiple` | 476 | 714 | 920 | 944 | 0 |
  | `parallel` | 276 | 353 | 424 | 427 | 0 |
  | `live_simple` | 279 | 470 | 672 | 819 | 0 |
  | `live_multiple` | 905 | 1628 | 2335 | 5166 | 2 |
  | `live_parallel_multiple` | 805 | 1599 | 2386 | 2386 | 0 |

  収まらない2件(`live_multiple_217-93-0` = 4597トークン、`live_multiple_985-216-0`
  = 5166トークン)を実際にボードへ送ったところ、どちらもトークン数を明示した
  `400 context_length_exceeded` が返りました。スコア上は失敗として数えられますが、
  0.2% なので結果を左右するものではありません。
- **プロンプトスコアリングは無関係**。BFCL は `echo`+`logprobs` を使わないので
  `PROMPT_LOGPROBS` は off のままで構いません。
- **実行後は prefix cache にエントリが残ります**。無害で、出力も変わりません
  (greedy 出力はキャッシュの有無にかかわらずバイト単位で同一)。

## gemma4

BFCL はモデルIDからチャットテンプレートを決めており、`GemmaHandler` は
Gemma 2/3 の `<start_of_turn>` をハードコードしている。**gemma4 の語彙にこのトークンは存在しない**ため、
gemma4 を `google/gemma-3-*` として評価すると、1マーカーあたり約9個の通常トークンに分解された
プロンプトを食わせることになる。gemma4 自身のモデルIDは登録されておらず、
`google/gemma-3-*` のリポジトリは Hugging Face で gated でもある。

`install_gemma4_handler.sh` は、導入済みの `bfcl-eval` に2つのIDを登録する:

| ID | ハンドラ | 内容 |
|---|---|---|
| `google/gemma4-e2b-it` | `Gemma4Handler` | gemma4 の `<\|turn>` マーカーを使う prompting モード |
| `google/gemma4-e2b-it-FC` | `Gemma4FCHandler` | **ネイティブ関数呼び出し** — ツールを `<\|tool>declaration:...<tool\|>` で宣言し、`<\|tool_call>call:NAME{...}<tool_call\|>` を解析する |

```bash
./install_gemma4_handler.sh /tmp/bfclvenv

BFCL=/tmp/bfclvenv/bin/bfcl ./run_bfcl.sh \
    --base-url http://<board>:8080 \
    --model google/gemma4-e2b-it-FC \
    --local-model-path /path/to/tokenizer-dir \
    --categories simple_python \
    --workdir /tmp/bfcl-gemma4
```

`--local-model-path` には、バンドルの `tokenizer.json` と並べて
`config.json`(`max_position_embeddings` を含む)と `tokenizer_config.json` を置いたディレクトリを渡す。
gated な `google/*` リポジトリを回避する手段であり、そもそも gemma4 には取得元の HF リポジトリが無い。

**どちらのモードを選ぶかが数値を支配する。** 実機・`simple_python` 全400件での実測:

| | Python Simple AST |
|---|---|
| `gemma4-e2b-it` — prompting | **7.50%** |
| 同じ生成結果の呼び出し形式を書き換えたもの(`bfcl_format_cost.py`) | 28.25% |
| **`gemma4-e2b-it` — ネイティブ FC** | **31.50%** |

prompting モードは `[func_name(param=value)]` を要求するが gemma4 は JSON で答えるため、
失敗370件のうち330件が `Failed to decode AST` になる — **そもそも話していない構文に対する構文判定**である。
**引用するなら FC の数値にすること**。prompting の数値は、モデルと同じくらい「食い違い」を測っている。

とはいえ 31.50% は、同じカテゴリでの Qwen3-1.7B(単一CL版)の 90.25% には遠く及ばない。
**ただしこの比較は交絡している。** 1.7B は400件中369件で思考する(出力中央値179トークン)のに対し、
gemma4 はこのハンドラで思考を有効にしていないため0件である。
差のうちどれだけが chain-of-thought によるもので、どれだけが関数呼び出し能力の差なのかは
分けられていない。揃えるにはどちらかが要る: **gemma4 の思考を有効にする**
(公式テンプレートは system ターン冒頭に `<|think|>` を注入する)か、
**1.7B の思考を切る**(`/no_think` をプロンプトに直接書く。BFCL は `/v1/completions` を使うので
サーバ側の `enable_thinking` は経路に無い)。**どちらも未実施。**
なお **BFCL は思考を `result` から `reasoning_content` へ分離する**ので、
`result` だけを数えると思考しているモデルを「思考していない」と誤判定する
(ここで実際に2度やらかした)。

FC の失敗274件の内訳は、**呼び出したが内容を間違えたもの205件**、**呼び出さず散文で答えたもの69件**、
そして**正しく組み立てたのに終端しなかったもの14件**
(`<|tool_call>call:math.gcd{num1:12,num2:15}` で `<tool_call|>` が無い)。
この14件はマーカー破損と同じ扱いで**失敗に数えている** — 自分の呼び出しを閉じないのはモデルの挙動であり、
それを許すパーサは別のものを測ることになる。許容すれば 3.5 ポイント上がる。
