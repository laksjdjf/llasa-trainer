from transformers import LlamaForCausalLM, AutoTokenizer, Xcodec2Model, Xcodec2FeatureExtractor, TrainingArguments
import torch
from dataclasses import dataclass
from typing import List, Dict, Tuple
from torch.nn.utils.rnn import pad_sequence
from modules.llasa import LLASA
from peft import AutoPeftModelForCausalLM, LoraConfig
from modules.train_utils import TTSTestCallback, load_dataset
import os
from trl import SFTTrainer

class LlasaForMaskedLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.mask_token_id = 128002

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=300, step=20, temperature=1.0, top_p=0.9, **kwargs):
        if step <= 0:
            raise ValueError(f"step must be positive, got {step}")
        
        batch_size = input_ids.size(0)
        len_inputs = input_ids.size(1)
        device = input_ids.device
        
        # 初期化：全てマスクトークン
        prediction = torch.full((batch_size, max_new_tokens), self.mask_token_id, device=device, dtype=torch.long)
        
        # Diffusionステップごとにトークンを生成
        for i in range(step):
            # 現在のステップでマスクする割合を計算（線形減衰）
            mask_ratio = 1.0 - (i / step)
            num_masked = max(1, int(max_new_tokens * mask_ratio))
            
            # 現在の状態で推論（1回だけ）
            current_ids = torch.cat([input_ids, prediction], dim=1)
            
            # attention_maskを生成（4次元、全てのトークンが見える状態）
            current_seq_len = current_ids.size(1)
            attention_mask = torch.ones(
                (batch_size, self.config.num_attention_heads, current_seq_len, current_seq_len),
                device=device, dtype=torch.bool
            )
            
            outputs = self(current_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, len_inputs:]
            
            probs = torch.softmax(logits / temperature, dim=-1)
            new_tokens = torch.multinomial(probs.view(-1, self.config.vocab_size), num_samples=1)
            new_prediction = new_tokens.view(batch_size, max_new_tokens)
            
            # マスクされた部分のみ更新
            mask_positions = (prediction == self.mask_token_id)
            prediction = torch.where(mask_positions, new_prediction, prediction)
            
            # 次のステップのためのマスキング
            if i < step - 1:  # 最後のステップではマスキングしない
                # ランダムマスキング
                for b in range(batch_size):
                    mask_indices = torch.randperm(max_new_tokens, device=device)[:num_masked]
                    prediction[b, mask_indices] = self.mask_token_id
                
        current_ids = torch.cat([input_ids, prediction], dim=1)
        return current_ids

@dataclass
class DataCollatorGenMask:
    sep_token_id: int
    mask_token_id: int
    pad_token_id: int
    p_range: Tuple[float, float] = (0.01, 1.0)
    num_heads: int = 16

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # ---- 1) 取り出しと張り合わせ（可変長 -> パディングは後でまとめて） ----
        input_ids_list: List[torch.Tensor] = []

        for ex in features:
            ids = torch.tensor(ex["input_ids"], dtype=torch.long)
            input_ids_list.append(ids)

        # ---- 2) 各サンプルで生成部分のインデックスを特定 ----
        # sep_token_id が最初に現れる位置の「次」から末尾を生成部分とみなす
        gen_spans: List[slice] = []
        for ids in input_ids_list:
            sep_pos = (ids == self.sep_token_id).nonzero(as_tuple=False)
            if sep_pos.numel() == 0:
                raise ValueError(f"sep_token_id {self.sep_token_id} not found in input_ids")
            start = int(sep_pos[0].item()) + 1
            gen_spans.append(slice(start, len(ids)))

        # ---- 3) マスク処理（改善版）----
        masked_inputs: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []

        a, b = self.p_range
        for ids, span in zip(input_ids_list, gen_spans):
            labels = torch.full_like(ids, fill_value=-100)
            if span.start < span.stop:
                p = torch.empty(1).uniform_(a, b).item()
                
                # 生成部分に対して独立ベルヌーイ（最低1つはマスク）
                gen_len = span.stop - span.start
                bern = torch.rand(gen_len) < p
                
                # 少なくとも1つはマスクする
                if not bern.any():
                    random_idx = torch.randint(0, gen_len, (1,))
                    bern[random_idx] = True
                
                idxs = torch.nonzero(bern, as_tuple=False).view(-1) + span.start
                if idxs.numel() > 0:
                    labels[idxs] = ids[idxs]
                    ids = ids.clone()
                    ids[idxs] = self.mask_token_id
            masked_inputs.append(ids)
            labels_list.append(labels)

        input_ids = pad_sequence(masked_inputs, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
        attention_mask = torch.ones(
            (input_ids.size(0), self.num_heads, input_ids.size(1), input_ids.size(1)), dtype=torch.bool
        )

        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

        return batch

class LlasaDiffusion(LLASA):
    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "./lora_checkpoints",
        codec_model_path: str = "Anime-XCodec2-hf",
        dtype=torch.float16,
    ):
        """フォルダパスから LLASA モデルを読み込み"""
        
        # モデル読み込み
        print("📦 LoRAモデル読み込み中...")
        try:
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,
                dtype=dtype,
                device_map="auto"
            )
        except Exception as e:
            print(f"⚠️ 通常モデルとして再試行中... (Error: {e})")
            model = LlasaForMaskedLM.from_pretrained(
                model_path,
                dtype=dtype,
                device_map="auto",
            )
        
        print("📝 トークナイザー読み込み中...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print("🎵 XCodec2モデル読み込み中...")
        codec_model = Xcodec2Model.from_pretrained(codec_model_path, device_map="auto", dtype=dtype).eval()
        # avoid half precision error
        codec_model.decoder.head.to(dtype=torch.float32)
        def hook_fn(self):
            def forward(x):
                x = self.backbone(x)
                x = x.to(dtype=torch.float32)
                x = self.head(x)[0]
                return x
            return forward
        codec_model.decoder.forward = hook_fn(codec_model.decoder)
        feature_extractor = Xcodec2FeatureExtractor.from_pretrained(codec_model_path)
        
        return cls(model=model, tokenizer=tokenizer, codec_model=codec_model, feature_extractor=feature_extractor)
    
def main(config):
    """メイン関数"""
    
    # CUDA設定
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices

    # LoRA設定（nullの場合はFFTを使用）
    if config.lora is not None:
        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            target_modules=list(config.lora.target_modules),
            lora_dropout=config.lora.lora_dropout,
            bias=config.lora.bias,
            task_type="CAUSAL_LM",
        )
        print(f"🔧 LoRA設定: r={config.lora.r}, alpha={config.lora.lora_alpha}")
    else:
        lora_config = None
        print("🔧 FFT (Full Fine-tuning) モードを使用")
    
    # 学習設定（動的に引数を取得）
    training_kwargs = {
        "output_dir": config.output_dir,
    }
    
    # config.trainingの全ての設定を動的に追加
    if hasattr(config, 'training') and config.training is not None:
        for key, value in config.training.items():
            training_kwargs[key] = value
            print(f"🔧 学習設定: {key} = {value}")
    
    training_args = TrainingArguments(**training_kwargs)
    
    # LLASAインスタンスを最初に作成（XCodec2も含む）
    print("🎯 LLASAインスタンスを作成中...")
    dtype = getattr(torch, config.get('dtype', 'float16'))
    llasa = LlasaDiffusion.from_pretrained(model_path=config.model_name, codec_model_path=config.get('codec_model_name', "Anime-XCodec2-hf"), dtype=dtype)
    
    collator = DataCollatorGenMask(
        128260, #<|SPEECH_GENERATION_START|>
        llasa.model.mask_token_id,
        llasa.tokenizer.pad_token_id,
        num_heads=llasa.model.config.num_attention_heads,
    )
    
    # テスト用コールバック（設定があれば）
    callbacks = []
    if hasattr(config, 'test') and config.test is not None:
        test_callback = TTSTestCallback(
            llasa=llasa,
            test_text=config.test.text,
            test_interval=config.test.interval,
            save_path=os.path.join(config.output_dir, "samples")
        )
        callbacks.append(test_callback)
        print(f"🧪 テストコールバック設定: interval={config.test.interval}")
    else:
        print("🧪 テストコールバックなし")

    # データセットの読み込み
    train_dataset = load_dataset(config.data_dir)
    def formatting_func(example):
        return [example["text"][i] for i in range(len(example["text"]))]

    # トレーナー
    trainer = SFTTrainer(
        model=llasa.model,
        tokenizer=llasa.tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        formatting_func=formatting_func,
        callbacks=callbacks,
        peft_config=lora_config,
    )

    # ステップ0でテスト生成
    if callbacks:
        print("\n--- 初期状態でのテスト生成 ---")
        callbacks[0].test_generation(step=0)
        print("--- 初期テスト完了 ---\n")
    
    print("学習を開始します...")
    trainer.train()
    
    print("モデルを保存中...")
    trainer.save_model()

    # 最終ステップでテスト生成
    if callbacks:
        print("\n--- 最終状態でのテスト生成 ---")
        callbacks[0].test_generation(step='final')
        print("--- 最終テスト完了 ---\n")
    
    print("学習完了！")