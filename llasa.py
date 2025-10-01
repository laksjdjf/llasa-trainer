"""
LLASA-3B TTS 生成クラス
"""

import torch
import numpy as np
import tempfile
import soundfile as sf
import re
from transformers import AutoTokenizer, LogitsProcessor
from peft import AutoPeftModelForCausalLM
from xcodec2.modeling_xcodec2 import XCodec2Model


class LLASA:
    def __init__(self, model=None, tokenizer=None, codec_model=None):
        """LLASA-3B TTS モデルを初期化
        
        Args:
            model: 学習済みモデル（既に読み込み済みの場合）
            tokenizer: トークナイザー（既に読み込み済みの場合）
            codec_model: XCodec2モデル（既に読み込み済みの場合）
        """
        
        # 直接指定された場合はそれを使用
        if model is not None and tokenizer is not None and codec_model is not None:
            print("🎯 既存のモデルを使用...")
            self.model = model
            self.tokenizer = tokenizer
            self.codec_model = codec_model
        else:
            # 指定されていない場合はエラー
            raise ValueError("モデル、トークナイザー、コーデックをすべて指定してください")
        
        # 音声トークン設定
        self._setup_speech_tokens()
        
        # テキスト正規化設定
        self._setup_normalizer()
        
        print("✅ LLASA 初期化完了！")
    
    @classmethod
    def from_pretrained(cls, lora_path: str = "./lora_checkpoints"):
        """フォルダパスから LLASA モデルを読み込み"""
        
        print("🚀 LLASA モデル読み込み開始...")
        
        # CUDA設定
        torch.cuda.empty_cache()
        
        # モデル読み込み
        print("📦 LoRAモデル読み込み中...")
        model = AutoPeftModelForCausalLM.from_pretrained(
            lora_path,
            torch_dtype=torch.float16,
        ).eval().to('cuda:0')
        model.merge_and_unload()
        
        print("📝 トークナイザー読み込み中...")
        tokenizer = AutoTokenizer.from_pretrained(lora_path)
        
        print("🎵 XCodec2モデル読み込み中...")
        codec_model = XCodec2Model.from_pretrained(
            "NandemoGHS/Anime-XCodec2",
        ).eval().to('cuda:0')
        
        return cls(model=model, tokenizer=tokenizer, codec_model=codec_model)
    
    def _setup_speech_tokens(self):
        """音声トークン設定を初期化"""
        self.speech_start_id = self.tokenizer.convert_tokens_to_ids('<|s_0|>')
        self.speech_end_id = self.tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
        
        # LogitsProcessor設定
        speech_token_ids = list(range(self.speech_start_id, self.speech_start_id + 65536))
        allowed_tokens = torch.tensor(speech_token_ids + [self.speech_end_id], dtype=torch.long)
        mask = torch.full((193800,), float('-inf'))
        mask[allowed_tokens] = 0.0
        mask = mask.unsqueeze(0).to("cuda:0", dtype=torch.float16)
        self.speech_processor = SpeechOnlyProcessor(mask)
        
        # テキスト正規化設定
        self._setup_normalizer()
        
        print("✅ LLASA 初期化完了！")
    
    def _setup_normalizer(self):
        """テキスト正規化の設定"""
        self.replace_map = {
            r"\t": "",
            r"\[n\]": "",
            r" ": "",
            r"　": "",
            r"[;▼♀♂《》≪≫①②③④⑤⑥]": "",
            r"[\u02d7\u2010-\u2015\u2043\u2212\u23af\u23e4\u2500\u2501\u2e3a\u2e3b]": "",
            r"[\uff5e\u301C]": "ー",
            r"？": "?",
            r"！": "!",
            r"[●◯〇]": "○",
            r"♥": "♡",
        }
        
        self.fullwidth_alpha_to_halfwidth = str.maketrans({
            chr(full): chr(half)
            for full, half in zip(
                list(range(0xFF21, 0xFF3B)) + list(range(0xFF41, 0xFF5B)),
                list(range(0x41, 0x5B)) + list(range(0x61, 0x7B)),
            )
        })
        
        self.halfwidth_katakana_to_fullwidth = str.maketrans({
            chr(half): chr(full)
            for half, full in zip(range(0xFF61, 0xFF9F), range(0x30A1, 0x30FB))
        })
        
        self.fullwidth_digits_to_halfwidth = str.maketrans({
            chr(full): chr(half)
            for full, half in zip(range(0xFF10, 0xFF1A), range(0x30, 0x3A))
        })
        
        self.invalid_pattern = re.compile(
            r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
            r"\u0041-\u005A\u0061-\u007A"
            r"\u0030-\u0039"
            r"。、!?…♪♡○]"
        )
    
    def normalize_text(self, text: str) -> str:
        """テキストを正規化"""
        for pattern, replacement in self.replace_map.items():
            text = re.sub(pattern, replacement, text)
        
        text = text.translate(self.fullwidth_alpha_to_halfwidth)
        text = text.translate(self.fullwidth_digits_to_halfwidth)
        text = text.translate(self.halfwidth_katakana_to_fullwidth)
        text = self.invalid_pattern.sub("", text)
        text = re.sub(r"…{3,}", "……", text)
        
        return text
    
    @torch.no_grad()
    def generate(
        self,
        text: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.0,
        max_tokens: int = 300,
    ) -> tuple[str, str, str]:
        """テキストから音声を生成
        
        Returns:
            tuple[audio_path, status_msg, token_info]
        """
        
        # テキスト正規化
        text = self.normalize_text(text)
        
        # プロンプト作成
        formatted_text = f"<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|>"
        chat = [
            {"role": "user", "content": "Convert the text to speech:" + formatted_text},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"}
        ]
        
        # トークン化
        input_ids = self.tokenizer.apply_chat_template(
            chat, tokenize=True, return_tensors='pt', continue_final_message=True
        ).to('cuda:0')
        
        # 音声トークン生成
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            eos_token_id=self.speech_end_id,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repeat_penalty,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            logits_processor=[self.speech_processor],
        )
        
        # 音声IDを抽出
        generated_ids = outputs[0][input_ids.shape[1]:]
        speech_ids = []
        
        for token_id in generated_ids:
            token_id_val = token_id.item()
            
            # 終了トークンで停止
            if token_id_val == self.speech_end_id:
                break
                
            # 音声トークンの範囲内かチェック
            if self.speech_start_id <= token_id_val < self.speech_start_id + 65536:
                speech_id = token_id_val - self.speech_start_id
                speech_ids.append(speech_id)
        
        if not speech_ids:
            return None, "❌ 有効な音声トークンが生成されませんでした", ""
        
        # 音声波形生成
        speech_codes = torch.tensor(speech_ids, dtype=torch.long).to('cuda:0').unsqueeze(0).unsqueeze(0)
        gen_wav = self.codec_model.decode_code(speech_codes)
        
        # ファイル保存
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            sf.write(tmp_file.name, gen_wav[0, 0, :].cpu().numpy(), 16000)
            audio_path = tmp_file.name
        
        status_msg = f"✅ 生成完了 ({len(speech_ids)} tokens)"
        token_info = str(speech_ids[:10]) + ("..." if len(speech_ids) > 10 else "")
        
        return audio_path, status_msg, token_info


class SpeechOnlyProcessor(LogitsProcessor):
    """音声トークンのみを許可するLogitsProcessor"""
    
    def __init__(self, mask):
        self.mask = mask
    
    def __call__(self, input_ids, scores):
        return scores + self.mask
