import os

from transformers import TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# LLASAクラスをインポート
from modules.llasa import LLASA
from modules.train_utils import TTSTestCallback, load_dataset

def main(config):
    """メイン関数"""
    
    # CUDA設定
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices

    # LoRA設定
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        target_modules=list(config.lora.target_modules),
        task_type="CAUSAL_LM"
    )
    
    # 学習設定
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        fp16=config.training.fp16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        report_to=config.training.report_to,
    )
    
    # LLASAインスタンスを最初に作成（XCodec2も含む）
    print("🎯 LLASAインスタンスを作成中...")
    llasa = LLASA.from_pretrained(lora_path=config.model_name)

    collator = DataCollatorForCompletionOnlyLM(
        "<|SPEECH_GENERATION_START|>",
        tokenizer=llasa.tokenizer,
    )
    
    # テスト用コールバック
    test_callback = TTSTestCallback(
        llasa=llasa,
        test_text=config.test.text,
        test_interval=config.test.interval,
        save_path=config.output_dir
    )

    # データセットの読み込み
    print("📂 データセットを読み込み中...")
    train_dataset = load_dataset(config.data_dir)

    # トレーナー
    trainer = SFTTrainer(
        model=llasa.model,
        tokenizer=llasa.tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        dataset_text_field="text",
        callbacks=[test_callback],
        peft_config=lora_config,
    )
    
    print("学習を開始します...")
    trainer.train()
    
    print("モデルを保存中...")
    trainer.save_model()
    
    print("学習完了！")

if __name__ == "__main__":
    main()