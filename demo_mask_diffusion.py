"""
Demo script for LLASA Mask Diffusion

This script demonstrates how to use the mask diffusion training and sampling.
"""

import torch
from modules.llasa_mask_diffusion import LLASAMaskDiffusion
from modules.llasa_utils import get_prompt


def demo_generation():
    """Demo: Generate audio using mask diffusion"""
    
    print("=" * 60)
    print("LLASA Mask Diffusion - Generation Demo")
    print("=" * 60)
    
    # Load model
    print("\n📦 Loading LLASA Mask Diffusion model...")
    llasa_md = LLASAMaskDiffusion.from_pretrained(
        model_path="./trained/MaskDiffusionModel",  # Path to trained model
        codec_model_path="Anime-XCodec2-hf",
    )
    
    # Test text
    test_text = "こんにちは、今日はいい天気ですね。"
    
    print(f"\n🎯 Generating audio for: '{test_text}'")
    print(f"   Using mask diffusion with 10 iterations...")
    
    # Generate audio
    audio_path, speech_ids = llasa_md.generate_with_mask_diffusion(
        text=test_text,
        num_audio_tokens=300,
        num_iterations=10,
        temperature=0.7,
        top_p=0.9,
        confidence_threshold=0.9,
    )
    
    if audio_path:
        print(f"\n✅ Audio generated successfully!")
        print(f"   Audio path: {audio_path}")
        print(f"   Generated {len(speech_ids)} speech tokens")
    else:
        print("\n❌ Generation failed")


def demo_inpainting():
    """Demo: Inpaint audio using mask diffusion"""
    
    print("\n" + "=" * 60)
    print("LLASA Mask Diffusion - Inpainting Demo")
    print("=" * 60)
    
    # Load model
    print("\n📦 Loading LLASA Mask Diffusion model...")
    llasa_md = LLASAMaskDiffusion.from_pretrained(
        model_path="./trained/MaskDiffusionModel",
        codec_model_path="Anime-XCodec2-hf",
    )
    
    # Create dummy audio codes (in practice, you'd get these from encoding audio)
    print("\n🎯 Creating dummy audio sequence for inpainting demo...")
    dummy_codes = list(range(100, 400))  # 300 tokens
    
    # Inpaint a region (tokens 100-200)
    mask_start = 100
    mask_end = 200
    
    print(f"   Inpainting region [{mask_start}:{mask_end}]...")
    
    inpainted_codes = llasa_md.inpaint_audio(
        original_codes=dummy_codes,
        mask_start=mask_start,
        mask_end=mask_end,
        num_iterations=10,
        temperature=0.7,
        top_p=0.9,
    )
    
    print(f"\n✅ Inpainting complete!")
    print(f"   Original length: {len(dummy_codes)} tokens")
    print(f"   Inpainted length: {len(inpainted_codes)} tokens")


def demo_training_collator():
    """Demo: Show how the mask diffusion collator works"""
    
    print("\n" + "=" * 60)
    print("LLASA Mask Diffusion - Training Collator Demo")
    print("=" * 60)
    
    from transformers import AutoTokenizer
    from modules.mask_diffusion_train import create_mask_diffusion_collator
    
    # Load tokenizer
    print("\n📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("NandemoGHS/Anime-Llasa-3B")
    
    # Create sample data
    sample_text = "こんにちは、今日はいい天気ですね。"
    prompt = get_prompt(sample_text, code=list(range(100)), add_bos_token=True, add_end_token=True)
    
    print(f"\n📝 Sample prompt length: {len(prompt)} characters")
    
    # Tokenize
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"][0]
    
    print(f"   Tokenized length: {len(input_ids)} tokens")
    
    # Create collator
    collator = create_mask_diffusion_collator(tokenizer, mask_ratio=0.15)
    
    # Apply collator
    examples = [{"input_ids": input_ids}]
    batch = collator(examples)
    
    # Count masked tokens
    labels = batch["labels"][0]
    num_masked = (labels != -100).sum().item()
    
    print(f"\n🎭 Masking applied:")
    print(f"   Mask ratio: 0.15")
    print(f"   Masked tokens: {num_masked}")
    print(f"   Total tokens: {len(input_ids)}")
    print(f"   Actual mask ratio: {num_masked / len(input_ids):.3f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLASA Mask Diffusion Demo")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["generation", "inpainting", "collator", "all"],
        default="all",
        help="Demo mode to run"
    )
    
    args = parser.parse_args()
    
    print("\n🎭 LLASA Mask Diffusion Demo Script")
    print("=" * 60)
    
    if args.mode in ["generation", "all"]:
        try:
            demo_generation()
        except Exception as e:
            print(f"\n⚠️  Generation demo skipped: {e}")
    
    if args.mode in ["inpainting", "all"]:
        try:
            demo_inpainting()
        except Exception as e:
            print(f"\n⚠️  Inpainting demo skipped: {e}")
    
    if args.mode in ["collator", "all"]:
        try:
            demo_training_collator()
        except Exception as e:
            print(f"\n⚠️  Collator demo skipped: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)
