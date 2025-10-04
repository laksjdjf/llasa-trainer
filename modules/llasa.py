import torch
import tempfile
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForCausalLM, Xcodec2Model, Xcodec2FeatureExtractor
from peft import AutoPeftModelForCausalLM
from modules.llasa_utils import get_prompt, SpeechOnlyProcessor


class LLASA:
    def __init__(self, model=None, tokenizer=None, codec_model=None, feature_extractor=None, compile_model=False):
        """LLASA-3B TTS モデルを初期化
        
        Args:
            model: 学習済みモデル（既に読み込み済みの場合）
            tokenizer: トークナイザー（既に読み込み済みの場合）
            codec_model: XCodec2モデル（既に読み込み済みの場合）
            compile_model: torch.compile()を使用するかどうか（PyTorch 2.0+）
        """
        
        self.model = model
        self.tokenizer = tokenizer
        self.codec_model = codec_model
        self.feature_extractor = feature_extractor
        self.device = model.device
        self.dtype = next(model.parameters()).dtype
        
        # torch.compile()による最適化（PyTorch 2.0+）
        if compile_model and hasattr(torch, 'compile'):
            print("⚡ torch.compile()でモデルを最適化中...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
            self.codec_model = torch.compile(self.codec_model, mode="reduce-overhead")
        
        self.logits_processor = SpeechOnlyProcessor(tokenizer=self.tokenizer, device=self.device, dtype=self.dtype)
        self.speech_start_id = self.tokenizer.convert_tokens_to_ids('<|s_0|>')
        self.speech_end_id = self.tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
        
        print("✅ LLASA 初期化完了！")
    
    @classmethod
    def from_pretrained(cls, lora_path: str = "./lora_checkpoints", compile_model: bool = False, use_bf16: bool = False):
        """フォルダパスから LLASA モデルを読み込み
        
        Args:
            lora_path: モデルのパス
            compile_model: torch.compile()を使用するかどうか（PyTorch 2.0+）
            use_bf16: bfloat16を使用するかどうか（A100などで高速化）
        """
        
        # データ型の選択
        dtype = torch.bfloat16 if use_bf16 and torch.cuda.is_bf16_supported() else torch.float16
        print(f"📊 使用するデータ型: {dtype}")
        
        # モデル読み込み
        print("📦 LoRAモデル読み込み中...")
        try:
            model = AutoPeftModelForCausalLM.from_pretrained(
                lora_path,
                dtype=dtype,
                device_map="auto",
            ).eval()
        except:
            print("⚠️ 通常モデルとして再試行中...")
            model = AutoModelForCausalLM.from_pretrained(
                lora_path,
                dtype=dtype,
                device_map="auto",
            ).eval()
        
        print("📝 トークナイザー読み込み中...")
        tokenizer = AutoTokenizer.from_pretrained(lora_path)
        
        print("🎵 XCodec2モデル読み込み中...")
        codec_model = Xcodec2Model.from_pretrained(
            "Anime-XCodec2-hf",
            torch_dtype=dtype,
        ).eval().to(model.device)
        feature_extractor = Xcodec2FeatureExtractor.from_pretrained("Anime-XCodec2-hf")
        
        return cls(model=model, tokenizer=tokenizer, codec_model=codec_model, feature_extractor=feature_extractor, compile_model=compile_model)
    
    @torch.no_grad()
    def generate(
        self,
        text: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        max_tokens: int = 300,
    ) -> tuple[str, str, str]:
        """テキストから音声を生成
        
        Returns:
            tuple[audio_path, status_msg, token_info]
        """
        
        # プロンプト作成
        prompt = get_prompt(text)
        
        # トークン化（デバイスを自動で合わせる）
        input_ids = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        # 音声トークン生成
        outputs = self.model.generate(
            **input_ids,
            max_new_tokens=max_tokens,
            eos_token_id=self.speech_end_id,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repeat_penalty,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            logits_processor=[self.logits_processor],
        )
        
        # 音声IDを抽出（GPU上で処理を継続）
        generated_ids = outputs[:, input_ids.input_ids.shape[1]:][0]
        speech_ids = []
        
        # ベクトル化された処理でパフォーマンス向上
        mask = (generated_ids >= self.speech_start_id) & (generated_ids < self.speech_start_id + 65536)
        end_token_positions = (generated_ids == self.speech_end_id).nonzero(as_tuple=True)[0]
        
        if len(end_token_positions) > 0:
            end_pos = end_token_positions[0].item()
            mask[end_pos:] = False
        
        valid_tokens = generated_ids[mask]
        speech_ids = (valid_tokens - self.speech_start_id).tolist()
        
        if not speech_ids:
            return None, "❌ 有効な音声トークンが生成されませんでした", ""
        
        # 音声波形生成（tensor操作を最小限に）
        speech_codes = torch.tensor(speech_ids, dtype=torch.long, device=self.device).unsqueeze(0).unsqueeze(0)
        gen_wav = self.codec_model.decode(speech_codes).audio_values
        
        # CPU転送は最後の1回のみ
        gen_wav_cpu = gen_wav[0, 0, :].cpu().numpy()
        
        # ファイル保存
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            sf.write(tmp_file.name, gen_wav_cpu, 16000)
            audio_path = tmp_file.name
        
        status_msg = f"✅ 生成完了 ({len(generated_ids)} tokens)"
        token_info = str(generated_ids.cpu().numpy().tolist())
        
        return audio_path, status_msg, token_info