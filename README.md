# LLASA-Trainer

**LLASA-3B TTS モデルのファインチューニングツール**

LLASA-3B（Large Language Audio Speech Analysis）は、テキストから音声を生成する日本語TTSモデルです。このリポジトリは、独自のデータセットでLLASA-3BをファインチューニングするためのトレーニングツールとUIを提供します。

## 🌟 特徴

- ✨ **LoRAベースのファインチューニング**: 効率的な学習で独自の音声スタイルを学習
- 🎤 **XCodec2統合**: 高品質な音声コーデックを使用
- 🖥️ **Gradio UI**: 簡単に使えるWebベースのTTSインターフェース
- 📊 **柔軟な設定**: YAMLベースの設定ファイルで簡単にカスタマイズ
- 🔄 **学習中テスト**: トレーニング中に音声生成をテストして進捗確認

## 📋 必要な環境

- Python 3.8以上
- CUDA対応GPU（推奨: 16GB以上のVRAM）
- PyTorch

## 🚀 インストール

### 1. リポジトリのクローン

```bash
git clone https://github.com/laksjdjf/llasa-trainer.git
cd llasa-trainer
```

### 2. 依存関係のインストール

```bash
pip install -r requirements.txt
```

**注意**: `requirements.txt`には特定のバージョンのtransformersライブラリが含まれています。

## 📂 データセット準備

### データ形式

学習には以下のデータが必要です：

1. **音声ファイル**: WAV形式の音声ファイル（16kHz推奨）
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

**オプション:**
- `-o, --output`: 出力ファイルパス（デフォルト: `dataset/data.jsonl`）
- `--max-samples`: 最大サンプル数

**例:**
```bash
python create_dataset.py ./audio_data ./transcripts.txt -o dataset/my_data.jsonl --max-samples 1000
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

トレーニング中は以下が実行されます：
- モデルの学習
- 定期的なチェックポイント保存
- テスト音声の生成（設定した間隔で）

## 🎤 音声生成（推論）

### Gradio UIの起動

```bash
python gradio_tts_ui.py [モデルパス]
```

**例:**
```bash
# デフォルト（./lora_checkpoints）
python gradio_tts_ui.py

# カスタムパス
python gradio_tts_ui.py ./trained/MyModel
```

### UIの使い方

1. ブラウザで表示されるURLにアクセス
2. テキストを入力
3. パラメータを調整（オプション）：
   - **Temperature**: ランダム性（0.1-2.0）
   - **Top-p**: サンプリング範囲（0.1-1.0）
   - **Repeat Penalty**: 繰り返し抑制（0.1-2.0）
   - **最大トークン数**: 生成する最大トークン数
4. 「音声生成」ボタンをクリック

### プログラムから使用

```python
from modules.llasa import LLASA

# モデルロード
llasa = LLASA.from_pretrained("./trained/MyModel")

# 音声生成
audio_path, status, tokens = llasa.generate(
    text="こんにちは、今日はいい天気ですね。",
    temperature=0.7,
    top_p=0.9,
    repeat_penalty=1.1,
    max_tokens=300
)

print(f"音声ファイル: {audio_path}")
```

## 📖 主要なスクリプト

| スクリプト | 説明 |
|----------|------|
| `main.py` | トレーニングのメインスクリプト |
| `create_dataset.py` | データセット作成ツール |
| `gradio_tts_ui.py` | Gradio WebUIの起動 |
| `modules/llasa.py` | LLASAモデルクラス |
| `modules/train.py` | トレーニングロジック |
| `modules/llasa_utils.py` | ユーティリティ関数 |

## 🔧 トラブルシューティング

### CUDA out of memory

- `per_device_train_batch_size`を減らす
- `gradient_accumulation_steps`を増やす
- `gradient_checkpointing`を有効にする

### 音声が生成されない

- モデルパスが正しいか確認
- CUDAが利用可能か確認（`torch.cuda.is_available()`）
- データセットの形式が正しいか確認

### 学習が進まない

- 学習率を調整
- データセットのサイズを確認
- エポック数を増やす

## 🔄 Transformers対応XCodec2モデルの作成

このプロジェクトはHugging Face Transformers対応のXCodec2モデルを使用します。オリジナルのAnime-XCodec2チェックポイントをTransformers形式に変換する場合は、以下の手順に従ってください。

### 前提条件

- `safetensors`ライブラリがインストールされていること
- Hugging Face Transformersライブラリがインストールされていること

### 変換手順

#### 1. オリジナルチェックポイントのダウンロード

```bash
# Hugging Faceから元のモデルをダウンロード
git lfs install
git clone https://huggingface.co/NandemoGHS/Anime-XCodec2 origin_ckpt
```

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

**注意**: 
- パスは環境に応じて調整してください（`venv/lib/python3.12/`の部分など）
- 変換には数分かかる場合があります
- 変換後のモデルは`Anime-XCodec2-hf`ディレクトリに保存されます

#### 4. 変換モデルの使用

変換が完了したら、ローカルパスを指定してモデルを使用できます：

```python
from transformers import Xcodec2Model, Xcodec2FeatureExtractor

# ローカルの変換済みモデルを使用
codec_model = Xcodec2Model.from_pretrained("./Anime-XCodec2-hf")
feature_extractor = Xcodec2FeatureExtractor.from_pretrained("./Anime-XCodec2-hf")
```

または、データセット作成時にローカルモデルを使用するように`create_dataset.py`を修正することもできます。

### トラブルシューティング

**ImportError: cannot import name 'Xcodec2Model'**
- Transformersライブラリのバージョンを確認してください
- このプロジェクトは特定のTransformersフォークを使用しています（`requirements.txt`参照）

**KeyError during conversion**
- ステップ2の重みキー変換が正しく実行されたか確認してください
- `origin_ckpt/model_c.safetensors`が存在するか確認してください

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
├── script/
│   └── convert_weight_norm_key.py  # XCodec2変換スクリプト
├── create_dataset.py         # データセット作成
├── gradio_tts_ui.py          # Gradio UI
├── main.py                   # トレーニングメイン
└── requirements.txt          # 依存関係
```

## 🤝 貢献

プルリクエストを歓迎します！バグ報告や機能リクエストはIssueで報告してください。

## 📄 ライセンス

MIT License - 詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 🙏 謝辞

- [LLASA-3B](https://huggingface.co/NandemoGHS/Anime-Llasa-3B) - ベースモデル
- [XCodec2](https://huggingface.co/Anime-XCodec2-hf) - 音声コーデック
- Hugging Face Transformers, PEFT, TRL

## 📞 サポート

質問や問題がある場合は、[Issues](https://github.com/laksjdjf/llasa-trainer/issues)で報告してください。