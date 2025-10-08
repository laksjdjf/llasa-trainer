#!/usr/bin/env python3
import json
from pathlib import Path
import argparse
from modules.llasa import LLASA
from transformers import Xcodec2Model, AutoFeatureExtractor
from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 必要に応じて変更

def parse_text_file(text_file: Path, split_word: str = ":", file_index: int = 0, text_index: int = 1):
    """テキストファイルを解析して辞書を作成"""
    texts = {}
    with text_file.open('r', encoding='utf-8') as f:
        for line in f:
            splited = line.strip().split(split_word)
            file_id = splited[file_index] if len(splited) > file_index else None
            text = splited[text_index] if len(splited) > text_index else None
            if file_id and text:
                texts[file_id.strip()] = text.strip()
    return texts

def create_dataset(audio_dir: Path, text_file: Path, output_file: Path, split_word: str = ":", file_index: int = 0, text_index: int = 1, ext: str = ".wav"):
    # テキスト読み込み
    print("テキスト読み込み中...")
    texts = parse_text_file(text_file, split_word, file_index, text_index)
    print(f"{len(texts)}個のテキストを取得")
    
    # モデル読み込み
    print("XCodec2モデルロード中...")
    codec_model = Xcodec2Model.from_pretrained("Anime-XCodec2-hf", device_map='auto').eval()
    feature_extractor = AutoFeatureExtractor.from_pretrained("Anime-XCodec2-hf")
    llasa = LLASA(codec_model=codec_model, feature_extractor=feature_extractor)
    print(f"モデルロード完了")
    
    # データセット作成
    print("データセット作成中...")
    created = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with output_file.open('w', encoding='utf-8') as f:
        for file_id, text in tqdm(texts.items()):
                
            audio_path = audio_dir / (file_id + ext)
            if not audio_path.exists():
                continue
                
            codes = llasa.encode_audio(audio_path)
            if codes is None:
                continue
                
            entry = {"text": text, "code": codes}
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            created += 1
    
    print(f"完了: {created}サンプル作成")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_dir", help="音声フォルダのパス")
    parser.add_argument("text_file", help="テキストファイルのパス")
    parser.add_argument("-o", "--output", default="dataset/data.jsonl", help="出力ファイル")
    parser.add_argument("--split_word", default=":", help="テキストファイルの区切り文字")
    parser.add_argument("--file_index", type=int, default=0, help="テキストファイルのファイルIDのインデックス")
    parser.add_argument("--text_index", type=int, default=1, help="テキストファイルのテキストのインデックス")
    parser.add_argument("--ext", default=".wav", help="音声ファイルの拡張子")
    args = parser.parse_args()
    
    audio_dir = Path(args.audio_dir)
    text_file = Path(args.text_file)
    output_file = Path(args.output)
    
    create_dataset(audio_dir, text_file, output_file, args.split_word, args.file_index, args.text_index, args.ext)

if __name__ == "__main__":
    main()