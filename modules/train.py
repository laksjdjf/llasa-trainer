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
        fp16=config.training.get("fp16", False),
        bf16=config.training.get("bf16", False),
        gradient_checkpointing=config.training.gradient_checkpointing,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        report_to=config.training.report_to,
        logging_steps=config.training.get("logging_steps", 10),
        logging_dir=config.training.get("logging_dir", f"{config.output_dir}/logs"),
        evaluation_strategy=config.training.get("evaluation_strategy", "no"),
        eval_steps=config.training.get("eval_steps", 100),
        save_strategy=config.training.get("save_strategy", "steps"),
        load_best_model_at_end=config.training.get("load_best_model_at_end", False),
        metric_for_best_model=config.training.get("metric_for_best_model", "loss"),
        greater_is_better=config.training.get("greater_is_better", False),
        dataloader_num_workers=config.training.get("dataloader_num_workers", 0),
        dataloader_pin_memory=config.training.get("dataloader_pin_memory", True),
        max_grad_norm=config.training.get("max_grad_norm", 1.0),
        optim=config.training.get("optim", "adamw_torch"),
        group_by_length=config.training.get("group_by_length", False),
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
    
    # バリデーションデータセットの読み込み（オプション）
    eval_dataset = None
    if hasattr(config, 'eval_data_dir') and config.eval_data_dir:
        print("📂 バリデーションデータセットを読み込み中...")
        eval_dataset = load_dataset(config.eval_data_dir)

    # コールバックのリスト
    callbacks = [test_callback]
    
    # Early Stoppingコールバックの追加（オプション）
    if config.training.get("use_early_stopping", False):
        from transformers import EarlyStoppingCallback
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=config.training.get("early_stopping_patience", 3),
            early_stopping_threshold=config.training.get("early_stopping_threshold", 0.0)
        )
        callbacks.append(early_stopping)
        print("⏰ Early Stopping を有効化")

    # トレーナー
    trainer = SFTTrainer(
        model=llasa.model,
        tokenizer=llasa.tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        dataset_text_field="text",
        callbacks=callbacks,
        peft_config=lora_config,
    )
    
    print("学習を開始します...")
    trainer.train()
    
    print("モデルを保存中...")
    trainer.save_model()
    
    print("学習完了！")

if __name__ == "__main__":
    main()