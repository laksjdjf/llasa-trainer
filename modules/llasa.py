import torch
import tempfile
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForCausalLM, Xcodec2Model, Xcodec2FeatureExtractor
from peft import AutoPeftModelForCausalLM
from modules.llasa_utils import get_prompt, SpeechOnlyProcessor


class BaseAudioDecoder:
    """XCodec2デコード機能の共通基底クラス"""
    
    def __init__(self, codec_model=None, feature_extractor=None):
        self.codec_model = codec_model
        self.feature_extractor = feature_extractor
    
    @torch.no_grad()
    def decode_tokens(self, speech_ids: list[int]) -> str:
        """音声トークンからwav波形を生成
        
        Args:
            speech_ids: 音声トークン列
            
        Returns:
            str: 生成されたwavファイルのパス
        """
        if not speech_ids:
            raise ValueError("音声トークンが空です")
        
        # 音声波形生成
        speech_codes = torch.tensor(speech_ids, dtype=torch.long).to('cuda:0').unsqueeze(0).unsqueeze(0)
        gen_wav = self.codec_model.decode(speech_codes).audio_values
        
        # ファイル保存
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            sf.write(tmp_file.name, gen_wav[0, 0, :].cpu().numpy(), 16000)
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
    ) -> tuple[str, str, str]:
        """テキストから音声を生成（サーバー版）
        
        Returns:
            tuple[audio_path, status_msg, token_info]
        """
        
        # トークン生成
        prompt = get_prompt(text)
        try:
            speech_ids = self.generate_tokens(prompt, temperature, top_p, repeat_penalty, max_tokens)
        except RuntimeError as e:
            return None, f"❌ {str(e)}", ""
        
        if not speech_ids:
            return None, "❌ 有効な音声IDが生成されませんでした", ""
        
        # デコード
        try:
            audio_path = self.decode_tokens(speech_ids)
            status_msg = f"✅ サーバー生成完了 ({len(speech_ids)} tokens)"
            token_info = str(speech_ids)
            
            return audio_path, status_msg, token_info
            
        except Exception as e:
            return None, f"❌ 音声デコードエラー: {e}", ""
        
    def generate_multilines(
        self,
        text: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        max_tokens: int = 300,
    ):
        
        texts = [line.strip() for line in text.splitlines() if line.strip()]
        speech_ids = []

        all_speech_ids = []
        pre_line = ""
        for line in texts:
            prompt = get_prompt(pre_line + line, speech_ids, add_end_token=False)
            pre_line = line + "。" if not line.endswith("。") else line
            try:
                speech_ids = self.generate_tokens(prompt, temperature, top_p, repeat_penalty, max_tokens)
                all_speech_ids.extend(speech_ids)
            except RuntimeError as e:
                print(f"❌ {str(e)}")
                continue

        if not all_speech_ids:
            return None, "❌ 有効な音声IDが生成されませんでした", ""

        # デコード
        try:
            audio_path = self.decode_tokens(all_speech_ids)
            status_msg = f"✅ サーバー生成完了 ({len(all_speech_ids)} tokens)"
            token_info = str(all_speech_ids)
            
            return audio_path, status_msg, token_info
            
        except Exception as e:
            return None, f"❌ 音声デコードエラー: {e}", ""

class LLASA(BaseAudioDecoder):
    def __init__(self, model=None, tokenizer=None, codec_model=None, feature_extractor=None):
        """LLASA-3B TTS モデルを初期化
        
        Args:
            model: 学習済みモデル（既に読み込み済みの場合）
            tokenizer: トークナイザー（既に読み込み済みの場合）
            codec_model: XCodec2モデル（既に読み込み済みの場合）
        """
        
        super().__init__(codec_model, feature_extractor)
        self.model = model
        self.tokenizer = tokenizer
        self.logits_processor = SpeechOnlyProcessor(tokenizer=self.tokenizer, device=model.device, dtype=next(model.parameters()).dtype)
        self.speech_start_id = self.tokenizer.convert_tokens_to_ids('<|s_0|>')
        self.speech_end_id = self.tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
        
        print("✅ LLASA 初期化完了！")
    
    @classmethod
    def from_pretrained(cls, lora_path: str = "./lora_checkpoints"):
        """フォルダパスから LLASA モデルを読み込み"""
        
        # モデル読み込み
        print("📦 LoRAモデル読み込み中...")
        try:
            model = AutoPeftModelForCausalLM.from_pretrained(
                lora_path,
                dtype=torch.float16,
            ).eval().to('cuda:0')
        except:
            print("⚠️ 通常モデルとして再試行中...")
            model = AutoModelForCausalLM.from_pretrained(
                lora_path,
                dtype=torch.float16,
            ).eval().to('cuda:0')
        
        print("📝 トークナイザー読み込み中...")
        tokenizer = AutoTokenizer.from_pretrained(lora_path)
        
        print("🎵 XCodec2モデル読み込み中...")
        codec_model = Xcodec2Model.from_pretrained("Anime-XCodec2-hf").eval().to('cuda:0')
        feature_extractor = Xcodec2FeatureExtractor.from_pretrained("Anime-XCodec2-hf")
        
        return cls(model=model, tokenizer=tokenizer, codec_model=codec_model, feature_extractor=feature_extractor)
    
    @torch.no_grad()
    def generate_tokens(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        max_tokens: int = 300,
    ) -> list[int]:
        """テキストから音声トークンを生成
        
        Returns:
            list[int]: 生成された音声トークン列
        """
        
        # トークン化
        input_ids = self.tokenizer(prompt, return_tensors='pt').to('cuda:0')

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