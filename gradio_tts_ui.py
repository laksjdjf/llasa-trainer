"""
シンプルなGradio TTS UI - LLASAクラス使用版
"""

import gradio as gr
import os
import time
import argparse
from modules.llasa import LLASA
from modules.llasa_utils import normalize_text

# CUDA設定
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def generate_speech(text: str, temperature: float = 0.7, top_p: float = 0.9, repeat_penalty: float = 1.0, max_tokens: int = 300):
    """時間計測付き音声生成"""
    start_time = time.time()
    
    audio_path, status, tokens = llasa.generate(text, temperature, top_p, repeat_penalty, max_tokens)
    
    elapsed_time = time.time() - start_time
    
    # ステータスに時間情報を追加
    if audio_path:
        status_with_time = f"✅ 生成完了！ ⏱️ {elapsed_time:.2f}秒"
    else:
        status_with_time = f"❌ {status} ⏱️ {elapsed_time:.2f}秒"
    
    return audio_path, status_with_time, tokens

# Gradio UI
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
    
    # サンプルテキスト
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
    
    # イベント設定
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLASA TTS Gradio UI")
    parser.add_argument("model_path", nargs="?", default="./lora_checkpoints", help="Path to the LLASA model")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile() for optimization (PyTorch 2.0+)")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision (recommended for A100)")
    args = parser.parse_args()
    
    print("=" * 50)
    print("🚀 最適化オプション:")
    print(f"  - torch.compile(): {'有効' if args.compile else '無効'}")
    print(f"  - bfloat16: {'有効' if args.bf16 else '無効 (float16)'}")
    print("=" * 50)
    
    # LLASAモデル初期化（最適化オプション付き）
    llasa = LLASA.from_pretrained(args.model_path, compile_model=args.compile, use_bf16=args.bf16)
    llasa.model.merge_and_unload()
    demo.launch()