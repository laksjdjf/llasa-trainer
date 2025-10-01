"""
Inference script for Anime-Llasa-3B model
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(model_path: str, lora_path: str = None, device: str = "cuda"):
    """Load model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
    )
    
    # Load LoRA weights if specified
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
    
    model.eval()
    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
):
    """Generate text from prompt"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Generate text with Anime-Llasa-3B")
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model or model name')
    parser.add_argument('--lora', type=str, default=None,
                        help='Path to LoRA weights')
    parser.add_argument('--prompt', type=str, required=True,
                        help='Input prompt')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='Nucleus sampling parameter')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-k sampling parameter')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for inference')
    
    args = parser.parse_args()
    
    print("Loading model...")
    model, tokenizer = load_model(args.model, args.lora, args.device)
    
    print("Generating text...")
    generated = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    
    print("\n" + "="*50)
    print("Generated text:")
    print("="*50)
    print(generated)


if __name__ == "__main__":
    main()
