#!/bin/bash

# Example inference script for Anime-Llasa-3B

# Basic inference with base model
python inference.py \
    --model "NandemoGHS/Anime-Llasa-3B" \
    --prompt "User: こんにちは\nAssistant: " \
    --max-length 256 \
    --temperature 0.7

# Inference with fine-tuned LoRA weights
# python inference.py \
#     --model "NandemoGHS/Anime-Llasa-3B" \
#     --lora "./output/final" \
#     --prompt "User: こんにちは\nAssistant: " \
#     --max-length 256
