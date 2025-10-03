#!/usr/bin/env python3
import json
from pathlib import Path
import argparse
from typing import Dict, List, Optional
import torch
import torchaudio
from transformers import AutoFeatureExtractor, Xcodec2Model

def parse_text_file(text_file: Path) -> Dict[str, str]:
    texts = {}
    with text_file.open('r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                file_id, text = line.strip().split(':', 1)
                texts[file_id.strip()] = text.strip()
    return texts

def preprocess_audio(audio_path: Path):
    waveform, sample_rate = torchaudio.load(str(audio_path))
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0)

def extract_codes(audio_path: Path, codec_model, feature_extractor) -> Optional[List[int]]:
    try:
        waveform = preprocess_audio(audio_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        inputs = feature_extractor(
            audio=waveform,
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt",
            use_torch=True,
        ).to(device)
    
        with torch.no_grad():
            vq_code = codec_model.encode(**inputs).audio_codes
            codes = vq_code[0, 0, :].cpu().numpy().tolist()
        
        del waveform, vq_code
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return codes
    except Exception as e:
        print(f"エラー: {audio_path}: {e}")
        return None

def load_codec_model():
    """XCodec2モデルと特徴抽出器を読み込み"""
    print("XCodec2モデルロード中...")
    codec_model = Xcodec2Model.from_pretrained("Anime-XCodec2-hf")
    feature_extractor = AutoFeatureExtractor.from_pretrained("Anime-XCodec2-hf")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    codec_model = codec_model.eval().to(device)
    print(f"モデルロード完了 ({device})")
    return codec_model, feature_extractor

def create_dataset(audio_dir: Path, text_file: Path, output_file: Path, max_samples: Optional[int] = None):
    # テキスト読み込み
    print("テキスト読み込み中...")
    texts = parse_text_file(text_file)
    print(f"{len(texts)}個のテキストを取得")
    
    # モデル読み込み
    codec_model, feature_extractor = load_codec_model()
    
    # データセット作成
    print("データセット作成中...")
    created = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with output_file.open('w', encoding='utf-8') as f:
        for file_id, text in texts.items():
            if max_samples and created >= max_samples:
                break
                
            audio_path = audio_dir / f"{file_id}.wav"
            if not audio_path.exists():
                continue
                
            codes = extract_codes(audio_path, codec_model, feature_extractor)
            if codes is None:
                continue
                
            entry = {"text": text, "codes": codes}
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            created += 1
            
            if created % 10 == 0:
                print(f"処理済み: {created}")
    
    print(f"完了: {created}サンプル作成")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_dir", help="音声フォルダのパス")
    parser.add_argument("text_file", help="テキストファイルのパス")
    parser.add_argument("-o", "--output", default="dataset/data.jsonl", help="出力ファイル")
    parser.add_argument("--max-samples", type=int, help="最大サンプル数")
    args = parser.parse_args()
    
    audio_dir = Path(args.audio_dir)
    text_file = Path(args.text_file)
    output_file = Path(args.output)
    
    create_dataset(audio_dir, text_file, output_file, args.max_samples)

if __name__ == "__main__":
    main()