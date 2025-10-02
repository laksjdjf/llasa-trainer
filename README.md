# LLASA-Trainer

[日本語](#日本語) | [English](#english)

---

## 日本語

### 概要

LLASA-Trainerは、LLASA-3B（Large Language Audio Speech Architecture）モデルをLoRAを使用してファインチューニングするためのトレーニングフレームワークです。このプロジェクトは、テキストから音声を生成する高品質な日本語TTSシステムの構築を支援します。

### 特徴

- 🎯 **LoRAベースのファインチューニング**: 効率的なメモリ使用でモデルをカスタマイズ
- 🎵 **XCodec2統合**: 高品質な音声生成のためのニューラルコーデック
- 🎤 **Gradio Web UI**: ブラウザベースの音声生成インターフェース
- 📊 **学習中テスト**: トレーニング中に音声品質を自動確認
- ⚙️ **柔軟な設定**: YAMLベースの設定管理
- 🚀 **簡単なデプロイ**: シンプルなコマンドラインインターフェース

### 必要要件

- Python 3.8+
- CUDA対応GPU（推奨）
- 必要なPythonパッケージ:
  - PyTorch
  - Transformers
  - PEFT (Parameter-Efficient Fine-Tuning)
  - TRL (Transformer Reinforcement Learning)
  - XCodec2
  - Gradio
  - OmegaConf
  - soundfile
  - datasets

### インストール

```bash
# リポジトリのクローン
git clone https://github.com/laksjdjf/llasa-trainer.git
cd llasa-trainer

# 必要なパッケージのインストール（例）
pip install torch transformers peft trl xcodec2 gradio omegaconf soundfile datasets
```

### 使い方

#### 1. データセットの準備

JSONL形式でデータセットを準備します：

```jsonl
{"text": "こんにちは、今日はいい天気ですね。", "code": [1234, 5678, ...]}
{"text": "ありがとうございます！", "code": [9012, 3456, ...]}
```

- `text`: 音声に変換するテキスト
- `code`: 対応する音声コード（オプション）

#### 2. 設定ファイルの作成

`config/example.yaml`を参考に設定ファイルを作成します：

```yaml
# データとモデルパス
data_dir: dataset/data.jsonl
output_dir: ./trained/MyModel
model_name: NandemoGHS/Anime-Llasa-3B

# CUDA設定
cuda_visible_devices: "0"

# LoRA設定
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  bias: none
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj

# 学習設定
training:
  num_train_epochs: 20
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1e-4
  save_steps: 100
  fp16: true

# テスト設定
test:
  text: 天使ちゃんマジ天使。
  interval: 50
```

#### 3. トレーニングの実行

```bash
python main.py --config config/your_config.yaml
```

トレーニング中、指定された間隔でテスト音声が生成され、進捗を確認できます。

#### 4. 推論（音声生成）

学習済みモデルを使用してGradio UIで音声を生成：

```bash
python gradio_tts_ui.py /path/to/trained/model
```

デフォルトでは`./lora_checkpoints`からモデルを読み込みます。

ブラウザでインターフェースが開き、以下の機能が利用できます：
- テキスト入力と正規化プレビュー
- 生成パラメータの調整（Temperature、Top-p、Repeat Penalty）
- リアルタイム音声生成
- サンプルテキストのクイック選択

### プロジェクト構造

```
llasa-trainer/
├── main.py                 # トレーニングエントリーポイント
├── gradio_tts_ui.py       # Gradio Web UIインターフェース
├── config/
│   └── example.yaml       # 設定ファイルの例
└── modules/
    ├── llasa.py           # LLASAモデルクラス
    ├── llasa_utils.py     # ユーティリティ関数
    ├── train.py           # トレーニングロジック
    └── train_utils.py     # トレーニング用ヘルパー
```

### モデル詳細

#### LLASAクラス

`modules.llasa.LLASA`クラスは、テキストから音声生成までの完全なパイプラインを提供します：

```python
from modules.llasa import LLASA

# モデルの読み込み
llasa = LLASA.from_pretrained("./lora_checkpoints")

# 音声生成
audio_path, status, tokens = llasa.generate(
    text="こんにちは、今日はいい天気ですね。",
    temperature=0.7,
    top_p=0.9,
    repeat_penalty=1.1,
    max_tokens=300
)
```

### ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

### 謝辞

- ベースモデル: [NandemoGHS/Anime-Llasa-3B](https://huggingface.co/NandemoGHS/Anime-Llasa-3B)
- オーディオコーデック: [NandemoGHS/Anime-XCodec2](https://huggingface.co/NandemoGHS/Anime-XCodec2)

---

## English

### Overview

LLASA-Trainer is a training framework for fine-tuning the LLASA-3B (Large Language Audio Speech Architecture) model using LoRA. This project helps build high-quality Japanese TTS systems that generate speech from text.

### Features

- 🎯 **LoRA-based Fine-tuning**: Efficient memory usage for model customization
- 🎵 **XCodec2 Integration**: Neural codec for high-quality speech generation
- 🎤 **Gradio Web UI**: Browser-based speech generation interface
- 📊 **Training Tests**: Automatic quality checks during training
- ⚙️ **Flexible Configuration**: YAML-based configuration management
- 🚀 **Easy Deployment**: Simple command-line interface

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- Required Python packages:
  - PyTorch
  - Transformers
  - PEFT (Parameter-Efficient Fine-Tuning)
  - TRL (Transformer Reinforcement Learning)
  - XCodec2
  - Gradio
  - OmegaConf
  - soundfile
  - datasets

### Installation

```bash
# Clone the repository
git clone https://github.com/laksjdjf/llasa-trainer.git
cd llasa-trainer

# Install required packages (example)
pip install torch transformers peft trl xcodec2 gradio omegaconf soundfile datasets
```

### Usage

#### 1. Prepare Dataset

Prepare your dataset in JSONL format:

```jsonl
{"text": "Hello, it's nice weather today.", "code": [1234, 5678, ...]}
{"text": "Thank you very much!", "code": [9012, 3456, ...]}
```

- `text`: Text to convert to speech
- `code`: Corresponding speech codes (optional)

#### 2. Create Configuration File

Create a configuration file based on `config/example.yaml`:

```yaml
# Data and model paths
data_dir: dataset/data.jsonl
output_dir: ./trained/MyModel
model_name: NandemoGHS/Anime-Llasa-3B

# CUDA settings
cuda_visible_devices: "0"

# LoRA configuration
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  bias: none
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj

# Training settings
training:
  num_train_epochs: 20
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1e-4
  save_steps: 100
  fp16: true

# Test settings
test:
  text: Sample test text.
  interval: 50
```

#### 3. Run Training

```bash
python main.py --config config/your_config.yaml
```

During training, test audio will be generated at specified intervals to monitor progress.

#### 4. Inference (Speech Generation)

Use the trained model with Gradio UI for speech generation:

```bash
python gradio_tts_ui.py /path/to/trained/model
```

By default, it loads the model from `./lora_checkpoints`.

The browser interface opens with the following features:
- Text input with normalization preview
- Generation parameter adjustment (Temperature, Top-p, Repeat Penalty)
- Real-time speech generation
- Quick sample text selection

### Project Structure

```
llasa-trainer/
├── main.py                 # Training entry point
├── gradio_tts_ui.py       # Gradio Web UI interface
├── config/
│   └── example.yaml       # Example configuration file
└── modules/
    ├── llasa.py           # LLASA model class
    ├── llasa_utils.py     # Utility functions
    ├── train.py           # Training logic
    └── train_utils.py     # Training helpers
```

### Model Details

#### LLASA Class

The `modules.llasa.LLASA` class provides a complete pipeline from text to speech generation:

```python
from modules.llasa import LLASA

# Load model
llasa = LLASA.from_pretrained("./lora_checkpoints")

# Generate speech
audio_path, status, tokens = llasa.generate(
    text="Hello, it's nice weather today.",
    temperature=0.7,
    top_p=0.9,
    repeat_penalty=1.1,
    max_tokens=300
)
```

### License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

### Acknowledgments

- Base model: [NandemoGHS/Anime-Llasa-3B](https://huggingface.co/NandemoGHS/Anime-Llasa-3B)
- Audio codec: [NandemoGHS/Anime-XCodec2](https://huggingface.co/NandemoGHS/Anime-XCodec2)