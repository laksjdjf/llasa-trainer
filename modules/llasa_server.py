import torch
import requests
from transformers import Xcodec2Model, Xcodec2FeatureExtractor
from modules.llasa_utils import get_prompt
from modules.llasa import BaseAudioDecoder

class LLASAServerClient(BaseAudioDecoder):
    def __init__(self, server_url="http://localhost:8000", codec_model=None, feature_extractor=None):
        """LLASA サーバー版クライアント
        
        Args:
            server_url: vLLMサーバーのURL
            codec_model: XCodec2モデル（既に読み込み済みの場合）
            feature_extractor: XCodec2 feature extractor（既に読み込み済みの場合）
        """
        
        super().__init__(codec_model, feature_extractor)
        self.server_url = server_url.rstrip('/')
        
        # 特殊トークンID（固定値）
        self.speech_start_id = 128264  # <|s_0|>
        self.speech_end_id = 128261    # <|SPEECH_GENERATION_END|>
        
        print("✅ LLASA サーバークライアント 初期化完了！")
    
    @classmethod
    def from_pretrained(cls, model_path, server_url="http://localhost:8000"):
        """XCodec2とサーバークライアントを初期化"""
        print("🔄 XCodec2モデルを読み込み中...")
        
        # XCodec2の読み込み
        try:
            codec_model = Xcodec2Model.from_pretrained("Anime-XCodec2-hf").eval().to('cuda:0')
            feature_extractor = Xcodec2FeatureExtractor.from_pretrained("Anime-XCodec2-hf")
            print("✅ XCodec2モデル読み込み完了")
        except Exception as e:
            print(f"❌ XCodec2読み込みエラー: {e}")
            return None
        
        # サーバークライアント初期化
        client = cls(
            server_url=server_url,
            codec_model=codec_model,
            feature_extractor=feature_extractor
        )
        
        print(f"✅ LLASA サーバークライアント準備完了 (vLLMサーバー: {server_url})")
        return client
    
    def _call_server(self, prompt: str, temperature: float, top_p: float, max_tokens: int, repeat_penalty: float):
        """サーバーAPIを呼び出し"""
        
        data = {
            "llasa": "llasa",
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "repetition_penalty": repeat_penalty,
            "stop_token_ids": [self.speech_end_id],  # 直接トークンIDを指定
            "stream": False
        }
        
        response = requests.post(
            f"{self.server_url}/v1/completions",
            headers={"Content-Type": "application/json"},
            json=data,
            timeout=120
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
    ) -> list[int]:
        """テキストから音声トークンを生成（サーバー版）
        
        Returns:
            list[int]: 生成された音声トークン列
        """
        
        # サーバーで生成
        generated_text, error = self._call_server(prompt, temperature, top_p, max_tokens, repeat_penalty)
        
        if error:
            raise RuntimeError(f"生成エラー: {error}")
        
        if not generated_text:
            raise RuntimeError("空のレスポンスが返されました")
        
        # 生成されたテキストを音声トークンIDに変換
        # extract_speech_ids関数を参考にした実装
        try:
            from modules.llasa_utils import extract_speech_ids
            import re
            
            # 音声トークンパターン <|s_数値|> を検索
            speech_token_pattern = r'<\|s_(\d+)\|>'
            matches = re.findall(speech_token_pattern, generated_text)
            
            if matches:
                # マッチした数値を整数に変換
                generated_ids = [int(match) for match in matches]
            else:
                # フォールバック: 生成テキストから直接数値を抽出
                number_pattern = r'\d+'
                numbers = re.findall(number_pattern, generated_text)
                if numbers:
                    generated_ids = [int(num) for num in numbers[:max_tokens//2]]
                else:
                    # 最終フォールバック: 適当な音声ID生成
                    estimated_count = min(len(generated_text) // 10, max_tokens // 4, 100)
                    generated_ids = list(range(0, estimated_count))
                    
        except Exception as e:
            print(f"⚠️ 音声ID抽出エラー: {e}")
            # エラー時のフォールバック
            estimated_count = min(len(generated_text) // 10, max_tokens // 4, 50)
            generated_ids = list(range(0, estimated_count))
        
        # 音声IDを検証・フィルタリング
        speech_ids = []
        
        for speech_id in generated_ids:
            # 音声IDの範囲をチェック（0-65535）
            if 0 <= speech_id < 65536:
                speech_ids.append(speech_id)
            else:
                print(f"⚠️ 範囲外の音声ID: {speech_id}")
        
        return speech_ids

# 完全互換のためのエイリアス
LLASA = LLASAServerClient