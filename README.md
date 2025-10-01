# llasa-trainer

Training code for [NandemoGHS/Anime-Llasa-3B](https://huggingface.co/NandemoGHS/Anime-Llasa-3B) model.

## Features

- Full fine-tuning support
- LoRA (Low-Rank Adaptation) training support
- 8-bit and 4-bit quantization support
- Wandb integration for experiment tracking
- Easy-to-use configuration via YAML
- Data utilities for dataset preparation

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Dataset

Create a JSONL file with your training data:

```json
{"text": "User: こんにちは\nAssistant: こんにちは！お手伝いできることはありますか？"}
{"text": "User: 今日の天気は？\nAssistant: 申し訳ございませんが、リアルタイムの天気情報にはアクセスできません。"}
```

Or use the provided example data:
```bash
cp example_data.json train_data.json
```

### 2. Configure Training

Edit `config.yaml` to specify your training parameters:

```yaml
model:
  name: "NandemoGHS/Anime-Llasa-3B"

training:
  output_dir: "./output"
  num_train_epochs: 3
  per_device_train_batch_size: 4
  learning_rate: 2.0e-5

data:
  train_file: "train_data.json"
  validation_file: null
  max_length: 2048

lora:
  enabled: true  # Use LoRA for efficient fine-tuning
  r: 8
  lora_alpha: 16
```

### 3. Train the Model

```bash
python train.py --config config.yaml
```

Or use the example script:
```bash
bash example_train.sh
```

### 4. Run Inference

```bash
python inference.py \
    --model "NandemoGHS/Anime-Llasa-3B" \
    --lora "./output/final" \
    --prompt "User: こんにちは\nAssistant: "
```

Or use the example script:
```bash
bash example_inference.sh
```

## Configuration Options

### Model Configuration

- `name`: Model name or path
- `load_in_8bit`: Enable 8-bit quantization
- `load_in_4bit`: Enable 4-bit quantization
- `use_flash_attention`: Enable Flash Attention 2

### Training Configuration

- `output_dir`: Directory to save model checkpoints
- `num_train_epochs`: Number of training epochs
- `per_device_train_batch_size`: Batch size per device
- `gradient_accumulation_steps`: Gradient accumulation steps
- `learning_rate`: Learning rate
- `warmup_steps`: Number of warmup steps
- `logging_steps`: Log every N steps
- `save_steps`: Save checkpoint every N steps
- `fp16`: Enable FP16 training
- `bf16`: Enable BF16 training
- `gradient_checkpointing`: Enable gradient checkpointing

### Data Configuration

- `train_file`: Path to training data file
- `validation_file`: Path to validation data file
- `max_length`: Maximum sequence length
- `preprocessing_num_workers`: Number of workers for preprocessing

### LoRA Configuration

- `enabled`: Enable LoRA training
- `r`: LoRA rank
- `lora_alpha`: LoRA alpha parameter
- `lora_dropout`: LoRA dropout rate
- `target_modules`: List of modules to apply LoRA to
- `bias`: Bias configuration ("none", "all", or "lora_only")

### Wandb Configuration

- `enabled`: Enable Wandb logging
- `project`: Wandb project name
- `name`: Run name
- `entity`: Wandb entity

## Data Utilities

### Split Dataset

Split your dataset into training and validation sets:

```bash
python data_utils.py \
    --split \
    --input data.jsonl \
    --train train.jsonl \
    --val val.jsonl \
    --val-ratio 0.1
```

### Prepare Training Data

Use the data utilities module to prepare your data:

```python
from data_utils import create_conversation_dataset

conversations = [
    {"user": "こんにちは", "assistant": "こんにちは！"},
    {"user": "ありがとう", "assistant": "どういたしまして！"},
]

create_conversation_dataset(
    conversations,
    "train_data.jsonl",
    system_prompt="あなたは親切なアシスタントです。"
)
```

## Advanced Usage

### LoRA Training

For efficient fine-tuning with limited GPU memory:

```yaml
lora:
  enabled: true
  r: 8
  lora_alpha: 16
  target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj
```

### Quantized Training

For training on limited GPU memory:

```yaml
model:
  load_in_4bit: true  # or load_in_8bit: true
  
lora:
  enabled: true  # LoRA is recommended with quantization
```

### Wandb Integration

Track your experiments with Wandb:

```yaml
wandb:
  enabled: true
  project: "anime-llasa-3b"
  name: "experiment-1"
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- 16GB+ GPU memory for full fine-tuning
- 8GB+ GPU memory for LoRA fine-tuning with quantization

## License

MIT License - See LICENSE file for details

## Acknowledgments

This trainer is built for the [Anime-Llasa-3B](https://huggingface.co/NandemoGHS/Anime-Llasa-3B) model by NandemoGHS.