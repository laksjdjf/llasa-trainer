import os
import json
from transformers import TrainerCallback
from datasets import Dataset
from modules.llasa_utils import get_prompt
import random

class TTSTestCallback(TrainerCallback):
    """学習中にTTSのテストを行うコールバック"""
    
    def __init__(self, llasa, test_text: str = "こんにちは、お元気ですか。", test_interval: int = 100, save_path: str = "./test_audio"):
        self.llasa = llasa
        self.test_text = test_text
        self.test_interval = test_interval
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
    
    def test_generation(self, step=None):
        """音声生成をテスト"""
        
        # 短い音声生成でテスト（LLASAが自動でテキスト正規化）
        audio_path, tokens = self.llasa.generate(
            self.test_text,
            temperature=0.7,
            top_p=0.9, 
            max_tokens=300,
        )
        
        if audio_path:
            print(f"🎵 テスト生成成功: '{self.test_text}' -> {len(tokens)} tokens")
            
            # 音声ファイルを保存（ステップ番号付き）
            if step is not None:
                save_path = os.path.join(self.save_path, f"test_step_{step}.wav")
                import shutil
                shutil.copy2(audio_path, save_path)
                print(f"🎵 テスト音声保存: {save_path}")
    
    def on_step_end(self, args, state, control, **kwargs):
        """ステップ終了時に呼ばれる"""
        
        if state.global_step % self.test_interval == 0:
            print(f"\n--- Step {state.global_step}: テスト実行中 ---")
            self.test_generation(step=state.global_step)
            print("--- テスト完了 ---\n")

def load_dataset(file_path: str):
    # データセットの読み込み
    print("📂 データセットを読み込み中...")
    # 1) JSONL -> プロンプトのリスト
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            samples.append(get_prompt(text=obj["text"], code=obj.get("code"), add_bos_token=False, add_end_token=True))

    # 2) HF Datasets へ
    random.shuffle(samples)
    train_ds = Dataset.from_dict({"text": samples})
    return train_ds