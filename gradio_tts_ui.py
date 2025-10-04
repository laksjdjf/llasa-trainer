"""Simple Gradio TTS UI using LLASA model.

This script provides a web interface for text-to-speech generation
using trained LLASA models.
"""
import gradio as gr
import os
import sys
import time
from modules.llasa import LLASA
from modules.llasa_utils import normalize_text

# CUDA configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def generate_speech(
    text: str, 
    temperature: float = 0.7, 
    top_p: float = 0.9, 
    repeat_penalty: float = 1.0, 
    max_tokens: int = 300
):
    """Generate speech with timing information.
    
    Args:
        text: Input text to convert to speech
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        repeat_penalty: Repetition penalty
        max_tokens: Maximum tokens to generate
        
    Returns:
        Tuple of (audio_path, status_message, token_info)
    """
    start_time = time.time()
    
    audio_path, status, tokens = llasa.generate(
        text, temperature, top_p, repeat_penalty, max_tokens
    )
    
    elapsed_time = time.time() - start_time
    
    # Add timing information to status
    if audio_path:
        status_with_time = f"✅ Generation complete! ⏱️ {elapsed_time:.2f}s"
    else:
        status_with_time = f"❌ {status} ⏱️ {elapsed_time:.2f}s"
    
    return audio_path, status_with_time, tokens


def create_ui() -> gr.Blocks:
    """Create the Gradio UI interface.
    
    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks(title="LLASA TTS", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎤 LLASA-3B TTS")
        
        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="テキスト入力",
                    value="",
                    lines=3
                )

                normalized_text = gr.Textbox(
                    label="正規化テキスト",
                    interactive=False
                )
                
                with gr.Row():
                    temperature = gr.Slider(0.1, 2.0, 0.7, step=0.01, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, 0.9, step=0.01, label="Top-p")
                    repeat_penalty = gr.Slider(0.1, 2.0, 1.1, step=0.01, label="Repeat Penalty")
                
                max_tokens = gr.Slider(50, 2000, 500, step=25, label="最大トークン数")
                generate_btn = gr.Button("🎵 音声生成", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                audio_output = gr.Audio(label="生成音声", type="filepath")
                status_output = gr.Textbox(label="状態", interactive=False)
                token_output = gr.Textbox(label="トークン情報", interactive=False)
        
        # Sample texts
        samples = [
            "こんにちは、今日はいい天気ですね。",
            "ありがとうございます！",
            "頑張って！応援してるよ！",
            "おつかれさまでした。"
        ]
        
        with gr.Row():
            for sample in samples:
                btn = gr.Button(f"「{sample}...」", size="sm")
                btn.click(lambda s=sample: s, outputs=text_input)
        
        # Event handlers
        generate_btn.click(
            generate_speech,
            inputs=[text_input, temperature, top_p, repeat_penalty, max_tokens],
            outputs=[audio_output, status_output, token_output]
        )

        text_input.change(
            normalize_text,
            inputs=text_input,
            outputs=normalized_text
        )
        
        text_input.submit(
            generate_speech,
            inputs=[text_input, temperature, top_p, repeat_penalty, max_tokens],
            outputs=[audio_output, status_output, token_output]
        )
    
    return demo


if __name__ == "__main__":
    # Get model path from command line argument
    llasa_model = sys.argv[1] if len(sys.argv) > 1 else "./lora_checkpoints"
    
    # Initialize LLASA model
    print(f"🎯 Loading LLASA model from: {llasa_model}")
    llasa = LLASA.from_pretrained(llasa_model)
    
    # Merge LoRA weights if applicable
    if hasattr(llasa.model, 'merge_and_unload'):
        print("🔀 Merging LoRA weights...")
        llasa.model.merge_and_unload()
    
    # Launch UI
    demo = create_ui()
    demo.launch()