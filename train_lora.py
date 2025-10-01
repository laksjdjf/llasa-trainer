"""
LoRAを使ったLLASA-3Bモデルのファインチューニングスクリプト
音スト→スピーチトークンモデルの学習
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import re

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training
)

# LLASAクラスをインポート
from llasa import LLASA
from xcodec2.modeling_xcodec2 import XCodec2Model

# グローバルでLLASAインスタンスを管理
_global_llasa = None

class TTSTestCallback(TrainerCallback):
    """学習中にTTSのテストを行うコールバック"""
    
    def __init__(self, test_text: str = "やっほー！ボクの名前はアストルフォ！", test_interval: int = 100):
        self.test_text = test_text  # 正規化はLLASAで行う
        self.test_interval = test_interval
        self.step_count = 0
    
    def test_generation(self, model, tokenizer, step=None):
        """音声生成をテスト"""
        try:
            global _global_llasa
            
            if _global_llasa is None:
                print("❌ LLASAインスタンスがありません")
                return
            
            # 短い音声生成でテスト（LLASAが自動でテキスト正規化）
            audio_path, status, tokens = _global_llasa.generate(
                self.test_text, 
                temperature=0.7, 
                top_p=0.9, 
                max_tokens=200  # 短めにして高速化
            )
            
            if audio_path:
                print(f"🎵 テスト生成成功: '{self.test_text}' -> {status}")
                
                # 音声ファイルを保存（ステップ番号付き）
                if step is not None:
                    save_path = f"./test_audio/step_{step}.wav"
                    os.makedirs("./test_audio", exist_ok=True)
                    import shutil
                    shutil.copy2(audio_path, save_path)
                    print(f"🎵 テスト音声保存: {save_path}")
            else:
                print(f"⚠️ テスト生成失敗: {status}")
                
        except Exception as e:
            print(f"❌ テスト生成エラー: {e}")
        finally:
            model.train()
    
    def on_step_end(self, args, state, control, model, tokenizer=None, **kwargs):
        """ステップ終了時に呼ばれる"""
        self.step_count += 1
        
        if self.step_count % self.test_interval == 0 and tokenizer is not None:
            print(f"\n--- Step {state.global_step}: テスト実行中 ---")
            self.test_generation(model, tokenizer, step=state.global_step)
            print("--- テスト完了 ---\n")

@dataclass
class TTSDataCollator:
    """TTS用のデータコレクター"""
    tokenizer: AutoTokenizer
    max_length: int = 2048

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # バッチ内の最大長を取得
        max_len = min(max([len(f["input_ids"]) for f in features]), self.max_length)
        
        input_ids = []
        attention_mask = []
        labels = []
        
        for feature in features:
            ids = feature["input_ids"][:max_len]
            prompt_len = feature["prompt_length"]
            mask = [1] * len(ids)
            
            # labelsを作成
            label_ids = ids.copy()
            
            # プロンプト部分（学習対象外）を-100でマスク
            for i in range(min(prompt_len, len(label_ids))):
                label_ids[i] = -100
            
            # パディング
            padding_length = max_len - len(ids)
            ids.extend([self.tokenizer.pad_token_id] * padding_length)
            mask.extend([0] * padding_length)
            label_ids.extend([-100] * padding_length)  # パディング部分もlossから除外
            
            input_ids.append(ids)
            attention_mask.append(mask)
            labels.append(label_ids)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

class TTSDataset(Dataset):
    """音声合成用データセット"""
    
    def __init__(self, data_dir: str, tokenizer: AutoTokenizer, max_length: int = 2048):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # データファイルのリストを作成
        self.data_files = []
        txt_files = sorted(list(self.data_dir.glob("*.txt")))
        
        for txt_file in txt_files:
            code_file = txt_file.with_name(txt_file.stem + "_codes.npy")
            if code_file.exists():
                self.data_files.append({
                    "text_file": txt_file,
                    "codes_file": code_file
                })
        
        print(f"データセットに {len(self.data_files)} 個のサンプルが見つかりました")
    
    def __len__(self):
        return len(self.data_files)
    
    def ids_to_speech_tokens(self, speech_ids):
        """音声IDを音声トークン文字列に変換"""
        speech_tokens_str = []
        for speech_id in speech_ids:
            speech_tokens_str.append(f"<|s_{speech_id}|>")
        return "".join(speech_tokens_str)
    
    def __getitem__(self, idx):
        data_info = self.data_files[idx]
        
        # テキストを読み込み
        with open(data_info["text_file"], "r", encoding="utf-8") as f:
            text = f.read().strip()
        
        # LLASAの正規化を使用（グローバルLLASAインスタンスから）
        global _global_llasa
        if _global_llasa is not None:
            text = _global_llasa.normalize_text(text)
        else:
            # フォールバック: 基本的な正規化
            text = text.strip()
        
        # 音声コードを読み込み
        speech_codes = np.load(data_info["codes_file"])
        
        # 音声トークンに変換
        speech_tokens = self.ids_to_speech_tokens(speech_codes)
        
        # チャット形式のテンプレートを作成
        formatted_text = f"<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|>"
        
        # プロンプト部分（学習対象外）
        chat_prompt = [
            {"role": "user", "content": "Convert the text to speech:" + formatted_text},
            {"role": "assistant", "content": f"<|SPEECH_GENERATION_START|>"}
        ]
        
        # 完全な会話（学習対象を含む）
        chat_full = [
            {"role": "user", "content": "Convert the text to speech:" + formatted_text},
            {"role": "assistant", "content": f"<|SPEECH_GENERATION_START|>{speech_tokens}<|SPEECH_GENERATION_END|>"}
        ]
        
        # プロンプト部分をトークン化
        prompt_ids = self.tokenizer.apply_chat_template(
            chat_prompt,
            tokenize=True,
            return_tensors=None,
            continue_final_message=True
        )
        
        # 完全な会話をトークン化
        full_ids = self.tokenizer.apply_chat_template(
            chat_full,
            tokenize=True,
            return_tensors=None,
            add_generation_prompt=False
        )
        
        # 最大長チェック
        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]
            # プロンプト部分が切り取られる場合は調整
            if len(prompt_ids) > self.max_length:
                prompt_ids = prompt_ids[:self.max_length]
        
        return {
            "input_ids": full_ids,
            "prompt_length": len(prompt_ids)  # プロンプト長を保存
        }



def setup_model_and_tokenizer(model_name: str = "NandemoGHS/Anime-Llasa-3B"):
    """モデルとトークナイザーを設定"""
    print(f"モデル '{model_name}' を読み込み中...")
    
    # トークナイザー
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # モデル
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # モデルを明示的にcuda:0に移動
    model = model.to('cuda:0')
    
    # モデルを明示的にトレーニングモードに設定
    model.train()
    
    return model, tokenizer

def setup_lora_model(model, lora_config: LoraConfig):
    """LoRAを適用したモデルを設定"""
    print("LoRAを適用中...")
    
    # モデルを学習モードに設定
    model.train()
    
    # 勾配を有効にする
    for param in model.parameters():
        param.requires_grad = False
    
    # LoRAを適用
    model = get_peft_model(model, lora_config)
    
    # 学習可能パラメータ数を表示
    model.print_trainable_parameters()
    
    return model

def main():
    """メイン関数"""
    
    # CUDA:0のみを使用するように設定
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # torchでもデバイス設定を確実にする
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"CUDA_VISIBLE_DEVICES=0 に設定、使用デバイス: {torch.cuda.current_device()}")
        print(f"利用可能なGPU数: {torch.cuda.device_count()}")
    else:
        print("CUDAが利用できません")
    
    # 設定
    DATA_DIR = "dataset/kama1"
    OUTPUT_DIR = "./kama1_lora"
    MODEL_NAME = "NandemoGHS/Anime-Llasa-3B"
    
    # LoRA設定
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # ランク
        lora_alpha=32,  # スケーリングパラメータ
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 基本的なattentionモジュールのみ
    )
    
    # 学習設定
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=20,
        per_device_train_batch_size=1,  # 小さくして安全に
        gradient_accumulation_steps=8,  # 実効バッチサイズを維持
        learning_rate=1e-4,  # 少し小さめに
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=True,  # メモリ節約
        gradient_checkpointing=False,  # メモリ節約のため有効化
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to=None,  # wandbなどを使う場合はここを変更
        ddp_find_unused_parameters=False,  # 追加
        dataloader_num_workers=0,  # マルチプロセシング無効化でCUDAコンテキスト問題回避
    )
    
    # モデルとトークナイザーを設定
    model, tokenizer = setup_model_and_tokenizer(MODEL_NAME)
    
    # LLASAインスタンスを最初に作成（XCodec2も含む）
    print("🎯 LLASAインスタンスを作成中...")
    global _global_llasa
    
    # XCodec2モデルを読み込み
    print("📦 XCodec2モデル読み込み中...")
    codec_model = XCodec2Model.from_pretrained(
        "NandemoGHS/Anime-XCodec2",
        torch_dtype=torch.float32
    ).eval().to('cuda:0')
    
    # LLASAインスタンスを作成
    _global_llasa = LLASA(model=model, tokenizer=tokenizer, codec_model=codec_model)
    print("✅ LLASAインスタンス作成完了")
    
    # LoRAを適用
    model = setup_lora_model(model, lora_config)
    
    # データセット
    train_dataset = TTSDataset(DATA_DIR, tokenizer)
    
    # データコレクター
    data_collator = TTSDataCollator(tokenizer)
    
    print(f"訓練可能なパラメータを確認中...")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"訓練可能: {trainable_params:,} / 全体: {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # テスト用コールバック
    test_callback = TTSTestCallback(
        test_text="こんにちは、マスターさん。",
        test_interval=50  # 50ステップごとにテスト
    )
    
    # トレーナー（音声トークン制約はLLASAが担当）
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[test_callback],
    )
    
    # 学習前の初期テスト
    print("\n--- 学習前の初期テスト ---")
    test_callback.test_generation(model, tokenizer, step="initial")
    print("--- 初期テスト完了 ---\n")
    
    # 学習開始
    print("学習を開始します...")
    trainer.train()
    
    # 学習後の最終テスト
    print("\n--- 学習後の最終テスト ---")
    test_callback.test_generation(model, tokenizer, step="final")
    print("--- 最終テスト完了 ---\n")
    
    # モデルを保存
    print("モデルを保存中...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("学習完了！")

if __name__ == "__main__":
    main()