# LLASA-Trainer

**LLASA-3B TTS モデルのファインチューニングツール**

LLASA-3B（Large Language Audio Speech Analysis）は、テキストから音声を生成する日本語TTSモデルです。このリポジトリは、独自のデータセットでLLASA-3BをファインチューニングするためのトレーニングツールとUIを提供します。
**注意**: `requirements.txt`には特定のバージョンのtransformersライブラリが含まれています。

## 🔄 Transformers対応XCodec2モデルの作成

このプロジェクトはHugging Face Transformers対応のXCodec2モデルを使用します。オリジナルのAnime-XCodec2チェックポイントをTransformers形式に変換する場合は、以下の手順に従ってください。

### 変換手順

#### 1. オリジナルチェックポイントのダウンロード
[model.safetensors](https://huggingface.co/NandemoGHS/Anime-XCodec2/blob/main/model.safetensors)


#### 2. 重みキーの変換

オリジナルのチェックポイントはHugging Faceの変換スクリプトと互換性のないキー名を使用しています。`script/convert_weight_norm_key.py`を使用してキーを変換します：

```bash
python script/convert_weight_norm_key.py
```

このスクリプトは以下の変換を実行します：
- `parametrizations.weight.original0` → `weight_g`
- `parametrizations.weight.original1` → `weight_v`
- `act.bias` → `act.beta`

変換後のモデルは`origin_ckpt/model_c.safetensors`として保存されます。

#### 3. Transformers形式への変換

Hugging Faceの公式変換スクリプトを使用して、PyTorch形式に変換します：

```bash
python venv/lib/python3.12/site-packages/transformers/models/xcodec2/convert_xcodec2_checkpoint_to_pytorch.py \
  --checkpoint_path origin_ckpt/model_c.safetensors \
  --config_path origin_ckpt/config.json \
  --pytorch_dump_folder_path Anime-XCodec2-hf
```

## 📂 データセット準備

### データ形式

学習には以下のデータが必要です：

1. **音声ファイル**: WAV形式の音声ファイル
2. **テキストファイル**: 音声に対応するテキスト（ファイル名:テキスト形式）

### ディレクトリ構成例

```
your_dataset/
├── audio/
│   ├── file001.wav
│   ├── file002.wav
│   └── file003.wav
└── text.txt
```

### テキストファイル形式例

デフォルトでは、`:` (コロン) で区切られたファイルIDとテキストのペアを記述します：

```
file001:こんにちは、今日はいい天気ですね。
file002:ありがとうございます。
file003:お疲れ様でした。
```

**注意**: ファイルIDは音声ファイル名から拡張子を除いたものと一致させる必要があります。

### データセット作成

#### 基本的な使い方

```bash
python create_dataset.py <音声フォルダ> <テキストファイル> -o dataset/data.jsonl
```

#### コマンドライン引数

| 引数 | 説明 | デフォルト値 |
|------|------|------------|
| `audio_dir` | 音声ファイルが格納されたディレクトリ | (必須) |
| `text_file` | テキストファイルのパス | (必須) |
| `-o, --output` | 出力するJSONLファイルのパス | `dataset/data.jsonl` |
| `--split_word` | テキストファイルの区切り文字 | `:` |
| `--file_index` | ファイルIDのインデックス（0始まり） | `0` |
| `--text_index` | テキストのインデックス（0始まり） | `1` |
| `--ext` | 音声ファイルの拡張子 | `.wav` |

#### 使用例

**基本的な使用例**:
```bash
python create_dataset.py ./audio ./text.txt -o dataset/data.jsonl
```

**タブ区切りのテキストファイルを使用する場合**:
```bash
python create_dataset.py ./audio ./text.txt -o dataset/data.jsonl --split_word $'\t'
```

**カラムの順序が逆の場合（テキスト:ファイルID）**:
```bash
python create_dataset.py ./audio ./text.txt -o dataset/data.jsonl --file_index 1 --text_index 0
```

**MP3ファイルを使用する場合**:
```bash
python create_dataset.py ./audio ./text.txt -o dataset/data.jsonl --ext .mp3
```

#### 処理の流れ

このスクリプトは以下を実行します：

1. **テキストファイルの読み込み**: 指定されたテキストファイルからファイルIDとテキストのペアを読み込みます
2. **XCodec2モデルのロード**: Anime-XCodec2-hfモデルを自動的にダウンロード・ロードします
3. **音声コードへの変換**: 各音声ファイルをXCodec2で音声コード（トークン）に変換します
4. **JSONL形式での保存**: テキストと音声コードのペアをJSONL形式で保存します

各行は以下の形式のJSON形式で保存されます：
```json
{"text": "こんにちは、今日はいい天気ですね。", "code": [1234, 5678, ...]}
```

#### ヒント

- 音声ファイルは16kHz以上のサンプリングレートを推奨します
- 長すぎる音声（10秒以上）は学習に時間がかかる可能性があります
- テキストは事前に正規化されている必要はありません（学習時に自動的に正規化されます）
- データセット作成には時間がかかるため、初回はGPU環境での実行を推奨します

## 🎓 トレーニング

### 1. 設定ファイルの準備

`config/example.yaml`をコピーして編集します：

```bash
cp config/example.yaml config/my_config.yaml
```

### 2. 主要な設定パラメータ

```yaml
# データパス
data_dir: dataset/data.jsonl          # 学習データ
output_dir: ./trained/MyModel         # 出力先
model_name: NandemoGHS/Anime-Llasa-3B # ベースモデル

# LoRA設定
lora:
  r: 16                    # LoRAランク（8-64推奨）
  lora_alpha: 32          # スケーリング係数
  lora_dropout: 0.05      # ドロップアウト率

# 学習設定
training:
  num_train_epochs: 20                  # エポック数
  per_device_train_batch_size: 1        # バッチサイズ
  gradient_accumulation_steps: 8        # 勾配蓄積ステップ
  learning_rate: 1e-4                   # 学習率
  fp16: false                           # FP16精度
  bf16: true                            # BF16精度（A100推奨）
```

### 3. トレーニング開始

```bash
python main.py --config config/my_config.yaml
```

### 4. VRAM節約方法

学習時にVRAMが不足する場合、以下の方法でメモリ使用量を削減できます。

#### 📊 VRAM使用量の目安

| GPU | VRAM | 推奨設定 |
|-----|------|---------|
| RTX 3060 | 12GB | LoRA r=8, batch_size=1, gradient_accumulation=16 |
| RTX 3090 / 4090 | 24GB | LoRA r=16, batch_size=1, gradient_accumulation=8 |
| A100 | 40GB+ | LoRA r=32, batch_size=2, gradient_accumulation=4 |

#### 🔧 主要な最適化パラメータ

##### 1. **LoRAランクを下げる**

LoRAランク（`r`）を小さくすると、学習可能なパラメータが減り、VRAM使用量が削減されます。

```yaml
lora:
  r: 8              # 16から8に下げる（VRAM使用量 約30-40%削減）
  lora_alpha: 16    # 通常はrの2倍に設定
```

**推奨値**:
- VRAM 8-12GB: `r=8`
- VRAM 12-16GB: `r=16`
- VRAM 24GB以上: `r=32`

##### 2. **バッチサイズと勾配蓄積の調整**

バッチサイズを1に固定し、勾配蓄積ステップ（`gradient_accumulation_steps`）で実効バッチサイズを調整します。

```yaml
training:
  per_device_train_batch_size: 1        # 常に1を推奨
  gradient_accumulation_steps: 16       # VRAMが少ない場合は大きく設定
```

実効バッチサイズ = `per_device_train_batch_size × gradient_accumulation_steps`

- VRAM不足時: `gradient_accumulation_steps`を16-32に増やす
- 十分なVRAM時: `per_device_train_batch_size`を2に増やすことも可能

##### 3. **混合精度学習の活用**

メモリ使用量を約半分に削減できます。

```yaml
training:
  fp16: true       # Ampere以前のGPU（RTX 20/30シリーズ）
  bf16: false      # fp16使用時はfalseに
```

または

```yaml
training:
  fp16: false      
  bf16: true       # Ampere以降のGPU（RTX 40シリーズ、A100）推奨
```

**選択ガイド**:
- RTX 20/30シリーズ、GTX 16シリーズ → `fp16: true`
- RTX 40シリーズ、A100、H100 → `bf16: true`（より安定）

##### 4. **勾配チェックポイント**

計算時間が増えますが、VRAM使用量を大幅に削減できます。

```yaml
training:
  gradient_checkpointing: true    # VRAM使用量 約20-30%削減
```

**トレードオフ**: 学習速度が約20-30%低下しますが、VRAM不足を解消できます。

##### 5. **学習対象レイヤーの削減**

LoRAを適用するレイヤーを減らすことでVRAM使用量を削減できます。

```yaml
lora:
  target_modules:
    - q_proj        # 最小構成: q_projとv_projのみ
    - v_proj
    # k_proj、o_projをコメントアウト
```

#### 💡 実践的な設定例

##### 例1: VRAM 8-12GB（RTX 3060など）

```yaml
lora:
  r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  target_modules:
    - q_proj
    - v_proj

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  fp16: true
  bf16: false
  gradient_checkpointing: true
```

##### 例2: VRAM 16-24GB（RTX 3090/4090など）

```yaml
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  fp16: false
  bf16: true
  gradient_checkpointing: false
```

##### 例3: VRAM 40GB以上（A100など）

```yaml
lora:
  r: 32
  lora_alpha: 64
  lora_dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj

training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  fp16: false
  bf16: true
  gradient_checkpointing: false
```

#### ⚠️ トラブルシューティング

**"CUDA out of memory"エラーが発生する場合**:

1. バッチサイズを1に固定: `per_device_train_batch_size: 1`
2. LoRAランクを下げる: `r: 8`
3. 勾配チェックポイントを有効化: `gradient_checkpointing: true`
4. 学習対象レイヤーを削減: `target_modules`を`q_proj`と`v_proj`のみに
5. それでも不足する場合: FFT（Full Fine-tuning）ではなく必ずLoRAを使用

**学習が遅すぎる場合**:

1. 勾配チェックポイントを無効化: `gradient_checkpointing: false`
2. `gradient_accumulation_steps`を減らす（VRAMが許す範囲で）
3. 混合精度学習を有効化: `fp16: true`または`bf16: true`

#### 📈 その他のヒント

- **不要なプロセスを終了**: 学習前にブラウザやその他のGPUを使用するアプリケーションを終了
- **システムモニタリング**: `nvidia-smi`コマンドでVRAM使用状況を確認
- **段階的な調整**: 設定を変更する際は一度に一つずつ変更し、効果を確認
- **データセット長の考慮**: 音声が長いほどVRAMを多く使用するため、極端に長い音声は分割を検討

## 🎤 音声生成（推論）

### Gradio UIの起動

```bash
python app.py -m [モデルパス] -c [コーデックモデルパス]
```

#### コマンドライン引数

- `-m, --model_path`: モデルのパス（デフォルト: `server`）
- `-c, --codec_model_path`: コーデックモデルのパス（デフォルト: `Anime-XCodec2-hf`）
- `--host`: ホスト名（省略可）
- `--port`: ポート番号（デフォルト: 7860）
- `--cuda_visible_devices`: 使用するCUDAデバイス（デフォルト: `0`）

#### UIの機能

起動後、以下の3つのタブが利用可能です：

1. **🗣️ TTS**: テキストから音声を生成
   - テキスト入力と参照音声を使用した音声生成
   - Temperature、Top-p、Repeat Penaltyなどの生成パラメータ調整
   - 音声の自動文字起こし機能

2. **🔤 トークナイザー**: 音声のトークン化と復元
   - 音声ファイルを音声トークンに変換
   - 音声トークンから音声を復元

3. **🎤 類似度計算**: 音声間の類似度を計算
   - ターゲット音声と複数の参照音声の類似度を測定

## 📖 主要なスクリプト

| スクリプト | 説明 |
|----------|------|
| `main.py` | トレーニングのメインスクリプト |
| `create_dataset.py` | データセット作成ツール |
| `app.py` | Gradio WebUIの起動 |
| `modules/llasa.py` | LLASAモデルクラス |
| `modules/train.py` | トレーニングロジック |
| `modules/llasa_utils.py` | ユーティリティ関数 |


## 📁 プロジェクト構造

```
llasa-trainer/
├── config/
│   └── example.yaml          # 設定ファイル例
├── modules/
│   ├── llasa.py              # LLASAモデルクラス
│   ├── llasa_utils.py        # ユーティリティ
│   ├── train.py              # トレーニングロジック
│   └── train_utils.py        # トレーニングユーティリティ
├── ui/
│   ├── llasa_processor.py    # LLASAモデル処理
│   ├── tts.py                # TTSインターフェース
│   ├── tokenizer.py          # トークナイザーインターフェース
│   └── similarity.py         # 類似度計算インターフェース
├── script/
│   └── convert_weight_norm_key.py  # XCodec2変換スクリプト
├── app.py                    # Gradio UI
├── create_dataset.py         # データセット作成
├── main.py                   # トレーニングメイン
└── requirements.txt          # 依存関係
```

## 📜 ライセンスと謝辞

### このリポジトリのライセンス

このリポジトリのコードは **MIT License** の下で公開されています。詳細は [LICENSE](LICENSE) ファイルをご覧ください。

### 使用している外部モデルとライセンス

このプロジェクトは以下の外部モデルを使用しています。各モデルには独自のライセンスが適用されます：

#### 1. **Anime-Llasa-3B**
- **提供元**: [NandemoGHS/Anime-Llasa-3B](https://huggingface.co/NandemoGHS/Anime-Llasa-3B)
- **説明**: ベースとなるLLASA-3B TTSモデル
- **ライセンス**: モデルのHugging Faceページでライセンスをご確認ください

#### 2. **Anime-XCodec2**
- **提供元**: [NandemoGHS/Anime-XCodec2](https://huggingface.co/NandemoGHS/Anime-XCodec2)
- **説明**: 音声エンコーダー/デコーダーモデル
- **ライセンス**: モデルのHugging Faceページでライセンスをご確認ください

#### 3. **anime-whisper**
- **提供元**: [litagin/anime-whisper](https://huggingface.co/litagin/anime-whisper)
- **説明**: 音声認識モデル（文字起こし機能に使用）
- **ライセンス**: モデルのHugging Faceページでライセンスをご確認ください

#### 4. **spkrec-ecapa-voxceleb (SpeechBrain)**
- **提供元**: [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- **説明**: 話者認識モデル（音声の埋め込みベクトル取得に使用）
- **ライセンス**: Apache License 2.0 ([SpeechBrain GitHub](https://github.com/speechbrain/speechbrain))
- **注**: Apache 2.0ライセンスはMITライセンスと互換性があります

### 重要な注意事項

⚠️ **このリポジトリのコードはMITライセンスですが、使用する外部モデルには各モデル独自のライセンスが適用されます。**

- 外部モデルを使用する前に、必ず各モデルのHugging Faceページでライセンス条項を確認してください
- 商用利用や配布を行う場合は、各モデルのライセンス条項を遵守してください
- モデルの使用に関する詳細な条件は、各モデルの提供元にお問い合わせください

### 謝辞

以下のプロジェクトとその開発者の皆様に感謝いたします：

- [NandemoGHS](https://huggingface.co/NandemoGHS) 様 - Anime-Llasa-3BおよびAnime-XCodec2モデルの開発と公開
- [litagin](https://huggingface.co/litagin) 様 - anime-whisperモデルの開発と公開
- [SpeechBrain](https://speechbrain.github.io/) - spkrec-ecapa-voxcelebモデルの開発と公開
- Hugging Face Transformers - モデルの統合とツールの提供
