"""
Training script for Anime-Llasa-3B model
"""
import os
import argparse
import yaml
from typing import Dict, Any
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_model_and_tokenizer(config: Dict[str, Any]):
    """Setup model and tokenizer"""
    model_config = config['model']
    model_name = model_config['name']
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model_kwargs = {}
    if model_config.get('load_in_8bit', False):
        model_kwargs['load_in_8bit'] = True
    if model_config.get('load_in_4bit', False):
        model_kwargs['load_in_4bit'] = True
    if model_config.get('use_flash_attention', False):
        model_kwargs['attn_implementation'] = "flash_attention_2"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if model_config.get('load_in_8bit') or model_config.get('load_in_4bit') else torch.float32,
        device_map="auto",
        **model_kwargs
    )
    
    # Setup LoRA if enabled
    if config['lora']['enabled']:
        if model_config.get('load_in_8bit') or model_config.get('load_in_4bit'):
            model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=config['lora']['r'],
            lora_alpha=config['lora']['lora_alpha'],
            target_modules=config['lora']['target_modules'],
            lora_dropout=config['lora']['lora_dropout'],
            bias=config['lora']['bias'],
            task_type=config['lora']['task_type'],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def prepare_dataset(config: Dict[str, Any], tokenizer):
    """Prepare dataset for training"""
    data_config = config['data']
    
    # Load dataset
    data_files = {}
    if data_config['train_file']:
        data_files['train'] = data_config['train_file']
    if data_config['validation_file']:
        data_files['validation'] = data_config['validation_file']
    
    if not data_files:
        raise ValueError("No training or validation file specified in config")
    
    # Determine file extension
    file_extension = data_config['train_file'].split('.')[-1] if data_config['train_file'] else None
    
    if file_extension == 'json' or file_extension == 'jsonl':
        dataset = load_dataset('json', data_files=data_files)
    elif file_extension == 'csv':
        dataset = load_dataset('csv', data_files=data_files)
    elif file_extension == 'txt':
        dataset = load_dataset('text', data_files=data_files)
    else:
        # Try to auto-detect
        dataset = load_dataset('json', data_files=data_files)
    
    # Tokenize dataset
    def tokenize_function(examples):
        # Assuming the dataset has a 'text' field
        text_field = 'text' if 'text' in examples else list(examples.keys())[0]
        return tokenizer(
            examples[text_field],
            padding='max_length',
            truncation=True,
            max_length=data_config['max_length'],
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=data_config['preprocessing_num_workers'],
        remove_columns=dataset['train'].column_names,
    )
    
    return tokenized_dataset


def main():
    parser = argparse.ArgumentParser(description="Train Anime-Llasa-3B model")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup wandb if enabled
    if config['wandb']['enabled']:
        wandb.init(
            project=config['wandb']['project'],
            name=config['wandb']['name'],
            entity=config['wandb']['entity'],
            config=config,
        )
    
    # Setup model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Prepare dataset
    print("Preparing dataset...")
    dataset = prepare_dataset(config, tokenizer)
    
    # Setup training arguments
    training_config = config['training']
    training_args = TrainingArguments(
        output_dir=training_config['output_dir'],
        num_train_epochs=training_config['num_train_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=training_config['learning_rate'],
        warmup_steps=training_config['warmup_steps'],
        logging_steps=training_config['logging_steps'],
        save_steps=training_config['save_steps'],
        eval_steps=training_config['eval_steps'],
        save_total_limit=training_config['save_total_limit'],
        fp16=training_config['fp16'],
        bf16=training_config['bf16'],
        gradient_checkpointing=training_config['gradient_checkpointing'],
        max_grad_norm=training_config['max_grad_norm'],
        report_to="wandb" if config['wandb']['enabled'] else "none",
        evaluation_strategy="steps" if 'validation' in dataset else "no",
        save_strategy="steps",
        load_best_model_at_end=True if 'validation' in dataset else False,
    )
    
    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset.get('validation'),
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Saving final model...")
    trainer.save_model(os.path.join(training_config['output_dir'], 'final'))
    tokenizer.save_pretrained(os.path.join(training_config['output_dir'], 'final'))
    
    print("Training complete!")


if __name__ == "__main__":
    main()
