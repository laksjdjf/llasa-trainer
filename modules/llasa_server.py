import torch
import requests
from transformers import Xcodec2Model, Xcodec2FeatureExtractor
from modules.llasa import BaseAudioDecoder
from modules.llasa_utils import extract_speech_ids

class LLASAServer(BaseAudioDecoder):
    def __init__(
            self,
            model="http://localhost:8000",
            tokenizer=None,
            codec_model=None,
            feature_extractor=None
        ):
        """LLASA サーバー版クライアント
        
        Args:
            server_url: vLLMサーバーのURL
            codec_model: XCodec2モデル（既に読み込み済みの場合）
            feature_extractor: XCodec2 feature extractor（既に読み込み済みの場合）
        """

        self.server_url = model.rstrip('/')
        super().__init__(model=None, tokenizer=tokenizer, codec_model=codec_model, feature_extractor=feature_extractor)
    
    @classmethod
    def from_pretrained(
        cls, 
        model_path="server",
        codec_model_path: str="Anime-XCodec2-hf",
        dtype=torch.float16,
    ):
        """XCodec2とサーバークライアントを初期化"""
        print("🔄 XCodec2モデルを読み込み中...")
        
        # XCodec2の読み込み
        codec_model = Xcodec2Model.from_pretrained(codec_model_path, device_map="auto", dtype=dtype).eval()
        
        # avoide half error
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
        print("✅ XCodec2モデル読み込み完了")
        
        # サーバークライアント初期化
        client = cls(
            model="http://localhost:8000",
            tokenizer=None,
            codec_model=codec_model,
            feature_extractor=feature_extractor
        )
        
        print(f"✅ LLASA サーバークライアント準備完了 (vLLMサーバー: {model_path})")
        return client
    
    def _call_server(self, prompt: str, temperature: float, top_p: float, repeat_penalty: float, max_tokens: int, min_tokens: int):
        """サーバーAPIを呼び出し"""
        
        data = {
            "llasa": "llasa",
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "min_tokens": min_tokens,
            "repetition_penalty": repeat_penalty, # vllm
            "repeat_penalty": repeat_penalty, # llama.cpp
            "stop_token_ids": [self.speech_end_id],  # 直接トークンIDを指定
            "stream": False
        }
        
        response = requests.post(
            f"{self.server_url}/v1/completions",
            headers={"Content-Type": "application/json"},
            json=data,
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["text"], None
        else:
            return None, f"API エラー: {response}"
    
    @torch.no_grad()
    def generate_tokens(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        max_tokens: int = 300,
        min_tokens: int = 0,
        step: int = 20,
    ) -> list[int]:
        """テキストから音声トークンを生成（サーバー版）
        
        Returns:
            list[int]: 生成された音声トークン列
        """
        
        # サーバーで生成
        generated_text, error = self._call_server(prompt, temperature, top_p, repeat_penalty, max_tokens, min_tokens)
        
        if error:
            raise RuntimeError(f"生成エラー: {error}")
        speech_ids = extract_speech_ids(generated_text)

        if not generated_text:
            raise RuntimeError("空のレスポンスが返されました")
        
        return speech_ids
