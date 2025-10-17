import gradio as gr
from ui.llasa_processor import generate, generate_multiple, transcribe
from modules.llasa_utils import normalize_text
import time

SAMPLES = [
    "こんにちは、今日はいい天気ですね。",
    "ありがとうございます！",
    "頑張って！応援してるよ！",
    "おつかれさまでした。"
]

def generate_speech(
    text: str,
    reference_text: str,
    reference_audio: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repeat_penalty: float = 1.0,
    max_tokens: int = 300
):
    
    """時間計測付き音声生成"""
    start_time = time.time()
    audio_path, tokens = generate(text, temperature, top_p, repeat_penalty, max_tokens, reference_text, reference_audio)
    elapsed_time = time.time() - start_time
    
    # ステータスに時間情報を追加
    if audio_path:
        num_tokens = len(tokens)
        status_with_time = f"✅ 生成完了！ ⏱️ {elapsed_time:.2f}秒 | {num_tokens} tokens | {num_tokens/elapsed_time:.2f} t/s"
    else:
        status_with_time = f"❌ 生成失敗^q^ ⏱️ {elapsed_time:.2f}秒"
    
    return audio_path, status_with_time, str(tokens)

def generate_multiple_speech(
    text: str,
    reference_text: str,
    reference_audio: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repeat_penalty: float = 1.0,
    max_tokens: int = 300
):
    """時間計測付き複数文音声生成"""
    start_time = time.time()
    texts = [s.strip() for s in text.splitlines() if s.strip()]
    audio_path, tokens = generate_multiple(texts, temperature, top_p, repeat_penalty, max_tokens, reference_text, reference_audio)
    elapsed_time = time.time() - start_time
    
    # ステータスに時間情報を追加
    if audio_path:
        num_tokens = len(tokens)
        status_with_time = f"✅ 生成完了！ ⏱️ {elapsed_time:.2f}秒 | {num_tokens} tokens | {num_tokens/elapsed_time:.2f} t/s"
    else:
        status_with_time = f"❌ 生成失敗^q^ ⏱️ {elapsed_time:.2f}秒"
    
    return audio_path, status_with_time, str(tokens)

def transcribe_audio(audio_path: str):
    if not audio_path:
        return ""
    return transcribe(audio_path)

def tts_interface():
    with gr.Blocks() as tts_ui:
        gr.Markdown("## 🗣️ Text-to-Speech (TTS) インターフェース")
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(label="📝 テキスト入力", placeholder="ここにテキストを入力してください...", lines=3)
                
                with gr.Row():
                    reference_audio = gr.Audio(label="🎧 参照音声 (オプション)", type="filepath", scale=8)
                    transcribe_audio_btn = gr.Button("📝 文字起こし", scale=2)
                reference_text = gr.Textbox(label="🔤 参照テキスト (オプション)", placeholder="参照音声の内容を入力してください...", lines=3)
                with gr.Row():
                    generate_button = gr.Button("▶️ 音声生成", variant="primary")
                    generate_multiple_button = gr.Button("▶️ 複数文生成", variant="secondary")

                with gr.Row():
                    temperature = gr.Slider(0.0, 1.0, value=0.7, step=0.01, label="🌡️ Temperature")
                    top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.01, label="🧮 Top-p")
                    repeat_penalty = gr.Slider(0.0, 2.0, value=1.3, step=0.01, label="🔁 Repeat Penalty")
                    max_tokens = gr.Slider(10, 4000, value=1000, step=10, label="🔢 Max Tokens")

                with gr.Row():
                    for sample in SAMPLES:
                        btn = gr.Button(f"「{sample}...」", size="sm")
                        btn.click(lambda s=sample: s, outputs=text_input)

                with gr.Accordion("正規化済みテキストを確認", open=False):
                    normalized_text = gr.Textbox(label="✅正規化済みテキスト", interactive=False, lines=3)
                    reference_normalized_text = gr.Textbox(label="✅正規化済み参照テキスト", interactive=False, lines=3)
                
            with gr.Column():
                audio_output = gr.Audio(label="🔊 生成された音声", type="filepath")
                status_output = gr.Textbox(label="📊 ステータス", interactive=False)
                tokens = gr.Textbox(label="🔢 生成トークン", interactive=False, lines=5)
    
        generate_button.click(
            fn=generate_speech,
            inputs=[
                text_input, reference_text, reference_audio, temperature, top_p, repeat_penalty, max_tokens
            ],
            outputs=[audio_output, status_output, tokens]
        )

        generate_multiple_button.click(
            fn=generate_multiple_speech,
            inputs=[
                text_input, reference_text, reference_audio, temperature, top_p, repeat_penalty, max_tokens
            ],
            outputs=[audio_output, status_output, tokens]
        )

        transcribe_audio_btn.click(
            fn=transcribe_audio,
            inputs=reference_audio,
            outputs=reference_text
        )

        text_input.change(
            fn=normalize_text,
            inputs=text_input,
            outputs=normalized_text
        )

        reference_text.change(
            fn=normalize_text,
            inputs=reference_text,
            outputs=reference_normalized_text
        )
    return tts_ui