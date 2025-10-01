"""
Utilities for data preparation and processing
"""
import json
from typing import List, Dict, Any
from pathlib import Path


def prepare_training_data(
    input_file: str,
    output_file: str,
    text_field: str = "text",
    format: str = "jsonl"
):
    """
    Prepare training data from various formats
    
    Args:
        input_file: Path to input file
        output_file: Path to output file
        text_field: Field name containing text data
        format: Output format ('jsonl' or 'json')
    """
    data = []
    
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        if input_file.endswith('.jsonl'):
            for line in f:
                data.append(json.loads(line))
        else:
            data = json.load(f)
    
    # Write output file
    with open(output_file, 'w', encoding='utf-8') as f:
        if format == 'jsonl':
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            json.dump(data, f, ensure_ascii=False, indent=2)


def create_conversation_dataset(
    conversations: List[Dict[str, Any]],
    output_file: str,
    system_prompt: str = None
):
    """
    Create a dataset from conversations
    
    Args:
        conversations: List of conversation dicts with 'user' and 'assistant' messages
        output_file: Path to output file
        system_prompt: Optional system prompt to prepend
    """
    formatted_data = []
    
    for conv in conversations:
        text = ""
        if system_prompt:
            text += f"System: {system_prompt}\n\n"
        
        text += f"User: {conv['user']}\n"
        text += f"Assistant: {conv['assistant']}\n"
        
        formatted_data.append({"text": text})
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def split_dataset(
    input_file: str,
    train_file: str,
    val_file: str,
    val_ratio: float = 0.1
):
    """
    Split dataset into train and validation sets
    
    Args:
        input_file: Path to input file
        train_file: Path to training output file
        val_file: Path to validation output file
        val_ratio: Ratio of validation data (default 0.1)
    """
    data = []
    
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        if input_file.endswith('.jsonl'):
            for line in f:
                data.append(json.loads(line))
        else:
            data = json.load(f)
    
    # Split data
    split_idx = int(len(data) * (1 - val_ratio))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # Write training data
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Write validation data
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Split dataset: {len(train_data)} training, {len(val_data)} validation")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Data preparation utilities")
    parser.add_argument('--split', action='store_true', help='Split dataset')
    parser.add_argument('--input', type=str, help='Input file')
    parser.add_argument('--train', type=str, help='Training output file')
    parser.add_argument('--val', type=str, help='Validation output file')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation ratio')
    
    args = parser.parse_args()
    
    if args.split:
        if not all([args.input, args.train, args.val]):
            print("Error: --input, --train, and --val are required for split operation")
            exit(1)
        split_dataset(args.input, args.train, args.val, args.val_ratio)
