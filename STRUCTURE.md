# Project Structure

This document describes the structure of the llasa-trainer project.

## Core Files

### Training
- **train.py**: Main training script with support for full fine-tuning and LoRA
  - Supports 8-bit/4-bit quantization
  - Integrated Wandb logging
  - Configurable via YAML
  - Automatic checkpoint saving

### Inference
- **inference.py**: Inference script for testing trained models
  - Supports base models and LoRA adapters
  - Configurable generation parameters (temperature, top-p, top-k)
  - Easy command-line interface

### Data Utilities
- **data_utils.py**: Utilities for data preparation
  - Dataset splitting (train/validation)
  - Format conversion (JSON, JSONL)
  - Conversation dataset creation

## Configuration

### config.yaml
Main configuration file containing:
- Model settings (name, quantization, flash attention)
- Training hyperparameters (learning rate, batch size, epochs)
- Data settings (file paths, max length)
- LoRA configuration (rank, alpha, target modules)
- Wandb settings (project, entity, run name)

## Examples

### Example Scripts
- **example_train.sh**: Shell script demonstrating training usage
- **example_inference.sh**: Shell script demonstrating inference usage
- **example_data.json**: Sample training data with Japanese conversations

## Setup & Installation

### setup.py
Python package setup file for installing llasa-trainer as a package:
```bash
pip install -e .
```

This enables command-line tools:
- `llasa-train`: Run training
- `llasa-inference`: Run inference

### requirements.txt
Lists all Python dependencies:
- PyTorch (>=2.0.0)
- Transformers (>=4.30.0)
- Datasets (>=2.14.0)
- Accelerate (>=0.20.0)
- PEFT (>=0.4.0)
- And more...

## Documentation

### README.md
Comprehensive guide covering:
- Installation instructions
- Quick start guide
- Configuration options
- Usage examples
- Advanced features (LoRA, quantization, Wandb)

### CONTRIBUTING.md
Guidelines for contributors:
- How to report issues
- How to submit pull requests
- Code style guidelines
- Development setup

### LICENSE
MIT License file

## Configuration Files

### .gitignore
Specifies files/directories to exclude from version control:
- Python bytecode (__pycache__)
- Virtual environments
- Build artifacts
- IDE settings

### .gitattributes
Ensures consistent line endings across platforms

## Usage Flow

1. **Prepare Data**: Use `data_utils.py` to prepare and split your dataset
2. **Configure**: Edit `config.yaml` to set training parameters
3. **Train**: Run `python train.py --config config.yaml`
4. **Inference**: Run `python inference.py` with your trained model
5. **Iterate**: Adjust configuration and repeat

## Key Features

- **Flexible Training**: Full fine-tuning or efficient LoRA training
- **Memory Efficient**: 8-bit/4-bit quantization support
- **Easy Configuration**: YAML-based configuration
- **Experiment Tracking**: Integrated Wandb support
- **Production Ready**: Includes setup.py for package installation
- **Well Documented**: Comprehensive README and examples
