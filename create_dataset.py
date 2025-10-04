#!/usr/bin/env python3
"""Dataset creation script for LLASA training.

This script converts audio files and text pairs into a JSONL dataset
suitable for training LLASA TTS models.
"""
import json
from pathlib import Path
import argparse
from typing import Dict, List, Optional
import torch
import torchaudio

from modules.llasa_utils import load_codec_model


def parse_text_file(text_file: Path) -> Dict[str, str]:
    """Parse text file with format 'file_id: text'.
    
    Args:
        text_file: Path to text file
        
    Returns:
        Dictionary mapping file IDs to text strings
    """
    texts = {}
    with text_file.open('r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                file_id, text = line.strip().split(':', 1)
                texts[file_id.strip()] = text.strip()
    return texts


def preprocess_audio(audio_path: Path, target_sample_rate: int = 16000) -> torch.Tensor:
    """Load and preprocess audio file.
    
    Args:
        audio_path: Path to audio file
        target_sample_rate: Target sample rate for resampling
        
    Returns:
        Preprocessed audio waveform tensor
    """
    waveform, sample_rate = torchaudio.load(str(audio_path))
    
    # Resample if necessary
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    return waveform.squeeze(0)


def extract_codes(audio_path: Path, codec_model, feature_extractor) -> Optional[List[int]]:
    """Extract codec codes from audio file.
    
    Args:
        audio_path: Path to audio file
        codec_model: XCodec2 model instance
        feature_extractor: Feature extractor instance
        
    Returns:
        List of codec codes, or None if extraction failed
    """
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
        
        # Clean up memory
        del waveform, vq_code
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return codes
    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")
        return None


def create_dataset(
    audio_dir: Path, 
    text_file: Path, 
    output_file: Path, 
    max_samples: Optional[int] = None
):
    """Create dataset from audio files and text file.
    
    Args:
        audio_dir: Directory containing audio files
        text_file: Path to text file with transcriptions
        output_file: Path to output JSONL file
        max_samples: Maximum number of samples to process (None for all)
    """
    # Load texts
    print("📝 Loading text file...")
    texts = parse_text_file(text_file)
    print(f"✅ Found {len(texts)} text entries")
    
    # Load codec model using shared utility
    codec_model, feature_extractor = load_codec_model()
    
    # Create dataset
    print("🎵 Creating dataset...")
    created = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with output_file.open('w', encoding='utf-8') as f:
        for file_id, text in texts.items():
            if max_samples and created >= max_samples:
                break
                
            audio_path = audio_dir / f"{file_id}.wav"
            if not audio_path.exists():
                print(f"⚠️ Audio file not found: {audio_path}")
                continue
                
            codes = extract_codes(audio_path, codec_model, feature_extractor)
            if codes is None:
                continue
                
            entry = {"text": text, "code": codes}
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            created += 1
            
            if created % 10 == 0:
                print(f"📊 Processed: {created} samples")
    
    print(f"✅ Dataset created: {created} samples")


def main():
    """Main entry point for dataset creation script."""
    parser = argparse.ArgumentParser(
        description="Create LLASA training dataset from audio files and text"
    )
    parser.add_argument("audio_dir", help="Directory containing audio files")
    parser.add_argument("text_file", help="Text file with transcriptions")
    parser.add_argument(
        "-o", "--output", 
        default="dataset/data.jsonl", 
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--max-samples", 
        type=int, 
        help="Maximum number of samples to process"
    )
    args = parser.parse_args()
    
    audio_dir = Path(args.audio_dir)
    text_file = Path(args.text_file)
    output_file = Path(args.output)
    
    create_dataset(audio_dir, text_file, output_file, args.max_samples)


if __name__ == "__main__":
    main()