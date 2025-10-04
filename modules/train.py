import os
from typing import Optional, List

from transformers import TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from modules.llasa import LLASA
from modules.train_utils import TTSTestCallback, load_dataset
from modules.llasa_utils import SPEECH_GENERATION_START


def create_lora_config(lora_config) -> Optional[LoraConfig]:
    """Create LoRA configuration from config object.
    
    Args:
        lora_config: Configuration object with LoRA parameters
        
    Returns:
        LoraConfig instance or None if not using LoRA
    """
    if lora_config is None:
        print("🔧 Using FFT (Full Fine-tuning) mode")
        return None
    
    config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        target_modules=list(lora_config.target_modules),
        task_type="CAUSAL_LM"
    )
    print(f"🔧 LoRA configuration: r={lora_config.r}, alpha={lora_config.lora_alpha}")
    
    return config


def create_training_arguments(config) -> TrainingArguments:
    """Create training arguments from config object.
    
    Args:
        config: Configuration object with training parameters
        
    Returns:
        TrainingArguments instance
    """
    training_kwargs = {
        "output_dir": config.output_dir,
        "overwrite_output_dir": True,
    }
    
    # Add all training settings dynamically
    if hasattr(config, 'training') and config.training is not None:
        for key, value in config.training.items():
            training_kwargs[key] = value
            print(f"🔧 Training setting: {key} = {value}")
    
    return TrainingArguments(**training_kwargs)


def create_callbacks(config, llasa) -> List[TTSTestCallback]:
    """Create training callbacks from config.
    
    Args:
        config: Configuration object with callback parameters
        llasa: LLASA model instance
        
    Returns:
        List of callback instances
    """
    callbacks = []
    
    if hasattr(config, 'test') and config.test is not None:
        test_callback = TTSTestCallback(
            llasa=llasa,
            test_text=config.test.text,
            test_interval=config.test.interval,
            save_path=config.output_dir
        )
        callbacks.append(test_callback)
        print(f"🧪 Test callback configured: interval={config.test.interval}")
    else:
        print("🧪 No test callback configured")
    
    return callbacks


def main(config):
    """Main training function.
    
    Args:
        config: Configuration object with all training parameters
    """
    
    # Set CUDA environment
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices

    # Create LoRA configuration
    lora_config = create_lora_config(config.lora if hasattr(config, 'lora') else None)
    
    # Create training arguments
    training_args = create_training_arguments(config)
    
    # Initialize LLASA instance (includes XCodec2)
    print("🎯 Creating LLASA instance...")
    llasa = LLASA.from_pretrained(lora_path=config.model_name)

    # Create data collator for completion-only training
    collator = DataCollatorForCompletionOnlyLM(
        SPEECH_GENERATION_START,
        tokenizer=llasa.tokenizer,
    )
    
    # Create callbacks
    callbacks = create_callbacks(config, llasa)

    # Load dataset
    print("📂 Loading dataset...")
    train_dataset = load_dataset(config.data_dir)

    # Create trainer
    trainer = SFTTrainer(
        model=llasa.model,
        tokenizer=llasa.tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        dataset_text_field="text",
        callbacks=callbacks,
        peft_config=lora_config,
    )
    
    # Start training
    print("🚀 Starting training...")
    trainer.train()
    
    # Save model
    print("💾 Saving model...")
    trainer.save_model()
    
    print("✅ Training complete!")


if __name__ == "__main__":
    main()
