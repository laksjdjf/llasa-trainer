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

### テキストファイル形式例

```
file001:こんにちは、今日はいい天気ですね。
file002:ありがとうございます。
file003:お疲れ様でした。
```

### データセット作成

```bash
python create_dataset.py <音声フォルダ> <テキストファイル> -o dataset/data.jsonl
```

このスクリプトは以下を実行します：
1. 音声ファイルを読み込み
2. XCodec2で音声コードに変換
3. テキストと音声コードをJSONL形式で保存

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
