import torch
import tempfile
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForCausalLM, Xcodec2Model, Xcodec2FeatureExtractor, pipeline
from peft import AutoPeftModelForCausalLM
from modules.llasa_utils import get_prompt, preprocess_audio, SAMPLING_RATE
from pathlib import Path


class BaseAudioDecoder:
    """XCodec2デコード機能の共通基底クラス"""
    
    def __init__(
        self, 
        model=None,
        tokenizer=None,
        codec_model=None, 
        feature_extractor=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.codec_model = codec_model
        self.feature_extractor = feature_extractor
        self.speech_start_id = 128264  # <|s_0|>
        self.speech_end_id = 128261    # <|SPEECH_GENERATION_END|>
        
        print("✅ LLASA サーバークライアント 初期化完了！")

    @torch.no_grad()
    def encode_audio(self, audio_path: str) -> list[int]:
        waveform = preprocess_audio(Path(audio_path))
        inputs = self.feature_extractor(
            audio=waveform,
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
            use_torch=True,
        ).to(self.codec_model.device)
        vq_code = self.codec_model.encode(**inputs).audio_codes
        codes = vq_code[0, 0, :].cpu().numpy().tolist()
        return codes
    
    @torch.no_grad()
    def decode_tokens(self, speech_ids: list[int]) -> str:
        if not speech_ids:
            raise ValueError("音声トークンが空です")
        
        # 音声波形生成
        speech_codes = torch.tensor(speech_ids, dtype=torch.long)
        speech_codes = speech_codes.to(self.codec_model.device).unsqueeze(0).unsqueeze(0)
        gen_wav = self.codec_model.decode(speech_codes).audio_values
        
        # ファイル保存
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            sf.write(tmp_file.name, gen_wav[0, 0, :].cpu().numpy(), SAMPLING_RATE)
            audio_path = tmp_file.name
        
        return audio_path
    
    @torch.no_grad()
    def generate(
        self,
        text: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        max_tokens: int = 300,
        reference_text: str = "",
        reference_audio: list[int] = None,
    ) -> tuple[str, str]:
        """テキストから音声を生成（サーバー版）
        
        Returns:
            tuple[audio_path, status_msg, token_info]
        """
        
        text = reference_text + text if reference_text else text
        reference_codes = self.encode_tokens(reference_audio) if reference_audio else None
        prompt = get_prompt(text, reference_codes, add_bos_token=True, add_end_token=False)
        speech_ids = self.generate_tokens(prompt, temperature, top_p, repeat_penalty, max_tokens)
        
        if not speech_ids:
            return None, speech_ids
        
        audio_path = self.decode_tokens(speech_ids)
        
        return audio_path, speech_ids
    
    def load_whisper(self, model_path: str = "litagin/anime-whisper"):
        self.whisper = pipeline(
            "automatic-speech-recognition",
            model=model_path,
            device="cuda",
            torch_dtype=torch.float16,
            chunk_length_s=30.0,
            batch_size=64,
        )

    def transcribe(self, audio_path: str) -> str:
        if not hasattr(self, 'whisper'):
            self.load_whisper()
        """音声ファイルをテキストに文字起こし"""
    
        generate_kwargs = {
            "language": "Japanese",
            "no_repeat_ngram_size": 0,
            "repetition_penalty": 1.0,
        }
        result = self.whisper(audio_path, generate_kwargs=generate_kwargs)
        return result['text']
    
    def load_classifier(self):
        """話者認識モデルの読み込み"""
        from speechbrain.inference.speaker import EncoderClassifier
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
        )

    def get_embedding(self, audio_path: str):
        """話者認識モデルで音声の埋め込みベクトルを取得"""
        if not hasattr(self, 'classifier'):
            self.load_classifier()
        audio = preprocess_audio(audio_path).unsqueeze(0)
        embedding = self.classifier.encode_batch(audio)
        return embedding.squeeze(0).cpu()

    def calc_similarity(self, target_audio: str, reference_audios: list[str]) -> float:
        """話者認識モデルで音声の類似度を計算"""
        target_emb = self.get_embedding(target_audio)
        ref_embs = [self.get_embedding(ref) for ref in reference_audios]
        similarities = [torch.cosine_similarity(target_emb, ref_emb, dim=1).item() for ref_emb in ref_embs]
        return similarities

class LLASA(BaseAudioDecoder):

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "./lora_checkpoints",
        codec_model_path: str = "Anime-XCodec2-hf",
        dtype=torch.float16
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
        except:
            print("⚠️ 通常モデルとして再試行中...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=dtype,
                device_map="auto",
            )
        
        print("📝 トークナイザー読み込み中...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print("🎵 XCodec2モデル読み込み中...")
        codec_model = Xcodec2Model.from_pretrained(codec_model_path, device_map="auto").eval()
        feature_extractor = Xcodec2FeatureExtractor.from_pretrained(codec_model_path)
        
        return cls(model=model, tokenizer=tokenizer, codec_model=codec_model, feature_extractor=feature_extractor)
    
    @torch.no_grad()
    def generate_tokens(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        max_tokens: int = 300,
        min_tokens: int = 0,
    ) -> list[int]:
        """テキストから音声トークンを生成
        
        Returns:
            list[int]: 生成された音声トークン列
        """
        
        # トークン化
        input_ids = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)

        # 音声トークン生成
        outputs = self.model.generate(
            **input_ids,
            max_new_tokens=max_tokens,
            min_new_tokens=min_tokens,
            eos_token_id=self.speech_end_id,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repeat_penalty,
            use_cache=True,
        )
        
        # 音声IDを抽出
        generated_ids = outputs[:, input_ids.input_ids.shape[1]:][0]
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
        
        return speech_ids