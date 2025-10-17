#!/usr/bin/env python3
"""
OpenAI Compatible Speech Generation Server for LLASA
OpenAI Speech API互換のLLASA音声生成サーバー

Usage:
    python openai_speech_server.py --port 8001

API Endpoints:
    POST /v1/audio/speech - OpenAI Speech API互換エンドポイント
    GET /v1/models - 利用可能なモデル一覧
"""

import argparse
import io
import time
import tempfile
import traceback
from typing import Optional, Literal
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import soundfile as sf
import torch

from modules.llasa_server import LLASAServer
from modules.llasa_utils import normalize_text

# Pydantic models for OpenAI API compatibility
class SpeechRequest(BaseModel):
    model: str = Field(default="llasa-3b", description="使用するモデル名")
    input: str = Field(..., description="音声に変換するテキスト")
    voice: str = Field(default="alloy", description="音声スタイル（現在は未使用）")
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(default="mp3", description="音声フォーマット")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="再生速度（現在は未使用）")
    # LLASA specific parameters
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="生成の多様性")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling")
    repeat_penalty: float = Field(default=1.1, ge=0.0, le=2.0, description="繰り返しペナルティ")
    max_tokens: int = Field(default=300, ge=1, le=2048, description="最大トークン数")
    reference_text: Optional[str] = Field(default=None, description="参照テキスト")
    reference_audio: Optional[str] = Field(default=None, description="参照音声ファイルパス")

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "llasa"

class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]

class ErrorResponse(BaseModel):
    error: dict

# グローバル変数
llasa_model: Optional[LLASAServer] = None

def initialize_model(model_path: str = "server", codec_model_path: str = "Anime-XCodec2-hf"):
    """LLASA モデルを初期化"""
    global llasa_model
    
    print(f"🔄 LLASA モデルを初期化中... (codec: {codec_model_path})")
    try:
        llasa_model = LLASAServer.from_pretrained(
            model_path=model_path,
            codec_model_path=codec_model_path
        )
        print("✅ LLASA モデル初期化完了")
        return True
    except Exception as e:
        print(f"❌ モデル初期化エラー: {e}")
        traceback.print_exc()
        return False

def audio_to_bytes(audio_data, sample_rate: int = 24000, format: str = "wav") -> bytes:
    """音声データをバイト列に変換"""
    # 音声データを正規化 (-1.0 to 1.0)
    if torch.is_tensor(audio_data):
        audio_data = audio_data.cpu().numpy()
    
    # メモリ上のバッファに音声を書き込み
    buffer = io.BytesIO()
    
    if format == "wav":
        sf.write(buffer, audio_data.T, sample_rate, format='WAV')
        content_type = "audio/wav"
    elif format == "mp3":
        # WAVとして一時ファイルに保存してからMP3に変換（実装は簡略化）
        # 実際の本番環境では ffmpeg などを使用することを推奨
        sf.write(buffer, audio_data.T, sample_rate, format='WAV')
        content_type = "audio/mpeg"
    else:
        # デフォルトはWAV
        sf.write(buffer, audio_data.T, sample_rate, format='WAV')
        content_type = "audio/wav"
    
    buffer.seek(0)
    return buffer.getvalue(), content_type

# FastAPI アプリケーション
app = FastAPI(
    title="LLASA OpenAI Compatible Speech API",
    description="OpenAI Speech API互換のLLASA音声生成サーバー",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "LLASA OpenAI Compatible Speech API Server", "status": "running"}

@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """利用可能なモデル一覧を返す"""
    models = [
        ModelInfo(
            id="llasa-3b",
            created=int(time.time()),
            owned_by="llasa"
        )
    ]
    return ModelsResponse(data=models)

@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    """OpenAI Speech API互換エンドポイント"""
    
    if llasa_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not initialized. Please restart the server."
        )
    
    try:
        # テキストの前処理
        text = normalize_text(request.input)
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="Input text is empty after normalization"
            )
        
        print(f"🎯 音声生成開始: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # LLASAで音声トークン生成
        audio_path, speech_tokens = llasa_model.generate(
            text=text,
            max_tokens=4000,
        )
        
        if not audio_path:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate speech tokens"
            )
        
        # 音声デコード
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            # 音声ファイルを読み込み
            audio_data, sample_rate = sf.read(audio_path)
            
            # 指定されたフォーマットに応じてバイト列に変換
            audio_bytes, content_type = audio_to_bytes(
                audio_data, 
                sample_rate, 
                request.response_format
            )
            
            print(f"✅ 音声生成完了: {len(speech_tokens)} tokens, {len(audio_bytes)} bytes")
            
            # ストリーミングレスポンスとして返す
            return StreamingResponse(
                io.BytesIO(audio_bytes),
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "X-Generated-Tokens": str(len(speech_tokens)),
                    "X-Audio-Duration": str(len(audio_data) / sample_rate)
                }
            )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 音声生成エラー: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Speech generation failed: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """グローバル例外ハンドラー"""
    print(f"❌ Unhandled exception: {exc}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error={
                "message": f"Internal server error: {str(exc)}",
                "type": "internal_error",
                "code": "internal_error"
            }
        ).dict()
    )

def main():
    parser = argparse.ArgumentParser(description="LLASA OpenAI Compatible Speech API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--model-path", default="server", help="LLASA model path")
    parser.add_argument("--codec-model-path", default="Anime-XCodec2-hf", help="XCodec2 model path")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--cuda_visible_devices", type=str, default="0", help="使用するCUDAデバイス (例: '0', '0,1')")
    
    args = parser.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    
    print("🚀 LLASA OpenAI Compatible Speech API Server を起動中...")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Model: {args.model_path}")
    print(f"   Codec: {args.codec_model_path}")
    
    # モデル初期化
    if not initialize_model(args.model_path, args.codec_model_path):
        print("❌ モデル初期化に失敗しました。終了します。")
        return 1
    
    # サーバー起動
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )
    
    return 0

if __name__ == "__main__":
    exit(main())