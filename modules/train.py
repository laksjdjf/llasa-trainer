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

    # LoRA設定（nullの場合はFFTを使用）
    lora_config = None
    if config.lora is not None:
        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            bias=config.lora.bias,
            target_modules=list(config.lora.target_modules),
            task_type="CAUSAL_LM"
        )
        print(f"🔧 LoRA設定: r={config.lora.r}, alpha={config.lora.lora_alpha}")
    else:
        print("🔧 FFT (Full Fine-tuning) モードを使用")
    
    # 学習設定（動的に引数を取得）
    training_kwargs = {
        "output_dir": config.output_dir,
        "overwrite_output_dir": True,
    }
    
    # config.trainingの全ての設定を動的に追加
    if hasattr(config, 'training') and config.training is not None:
        for key, value in config.training.items():
            training_kwargs[key] = value
            print(f"🔧 学習設定: {key} = {value}")
    
    training_args = TrainingArguments(**training_kwargs)
    
    # LLASAインスタンスを最初に作成（XCodec2も含む）
    print("🎯 LLASAインスタンスを作成中...")
    llasa = LLASA.from_pretrained(lora_path=config.model_name)

    collator = DataCollatorForCompletionOnlyLM(
        "<|SPEECH_GENERATION_START|>",
        tokenizer=llasa.tokenizer,
    )
    
    # テスト用コールバック（設定があれば）
    callbacks = []
    if hasattr(config, 'test') and config.test is not None:
        test_callback = TTSTestCallback(
            llasa=llasa,
            test_text=config.test.text,
            test_interval=config.test.interval,
            save_path=config.output_dir
        )
        callbacks.append(test_callback)
        print(f"🧪 テストコールバック設定: interval={config.test.interval}")
    else:
        print("🧪 テストコールバックなし")

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