"""
Main training script for Mask Diffusion LLASA

This script trains LLASA using mask diffusion instead of causal LM.
"""

import os
import torch
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

from modules.llasa import LLASA
from modules.mask_diffusion_train import MaskDiffusionTrainer, create_mask_diffusion_collator
from modules.train_utils import load_dataset


class MaskDiffusionTrainerWrapper(Trainer):
    """Custom Trainer for mask diffusion training"""
    
    def __init__(self, mask_trainer: MaskDiffusionTrainer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_trainer = mask_trainer
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """Override compute_loss to use mask diffusion loss"""
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        labels = inputs.get("labels", None)
        
        # Use mask diffusion trainer's loss computation
        loss, metrics = self.mask_trainer.compute_loss(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        # Log metrics
        if self.state.is_world_process_zero:
            self.log(metrics)
        
        return (loss, None) if return_outputs else loss


def main(config):
    """Main training function for mask diffusion"""
    
    # CUDA設定
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
    
    # Load LLASA model
    print("🎯 Loading LLASA model...")
    llasa = LLASA.from_pretrained(
        model_path=config.model_name,
        codec_model_path=config.get('codec_model_name', "Anime-XCodec2-hf")
    )
    
    # LoRA設定
    if config.lora is not None:
        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            target_modules=list(config.lora.target_modules),
            lora_dropout=config.lora.lora_dropout,
            bias=config.lora.bias,
            task_type="CAUSAL_LM",
        )
        llasa.model = get_peft_model(llasa.model, lora_config)
        print(f"🔧 LoRA設定: r={config.lora.r}, alpha={config.lora.lora_alpha}")
    else:
        print("🔧 Full fine-tuning mode")
    
    # Initialize mask diffusion trainer
    mask_ratio = config.get('mask_diffusion', {}).get('mask_ratio', 0.15)
    print(f"🎭 Initializing Mask Diffusion Trainer (mask_ratio={mask_ratio})...")
    
    mask_trainer = MaskDiffusionTrainer(
        model=llasa.model,
        tokenizer=llasa.tokenizer,
        mask_ratio=mask_ratio,
    )
    
    # Training arguments
    training_kwargs = {
        "output_dir": config.output_dir,
    }
    
    if hasattr(config, 'training') and config.training is not None:
        for key, value in config.training.items():
            training_kwargs[key] = value
            print(f"🔧 Training setting: {key} = {value}")
    
    training_args = TrainingArguments(**training_kwargs)
    
    # Load dataset
    print("📂 Loading dataset...")
    train_dataset = load_dataset(config.data_dir)
    
    # Tokenize dataset
    def tokenize_function(examples):
        """Tokenize the dataset"""
        tokenized = llasa.tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=2048,
        )
        return tokenized
    
    print("🔤 Tokenizing dataset...")
    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing dataset",
    )
    
    # Create data collator
    print("🎭 Creating mask diffusion data collator...")
    data_collator = create_mask_diffusion_collator(
        tokenizer=llasa.tokenizer,
        mask_ratio=mask_ratio,
    )
    
    # Create trainer
    print("🏋️ Creating trainer...")
    trainer = MaskDiffusionTrainerWrapper(
        mask_trainer=mask_trainer,
        model=llasa.model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("🚀 Starting training...")
    trainer.train()
    
    # Save model
    print("💾 Saving model...")
    trainer.save_model()
    
    print("✅ Training complete!")


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config)
    main(config)
