import gradio as gr
from ui.llasa_processor import encode_audio, decode_tokens

def encode_decode(audio_path: str):
    speech_ids = encode_audio(audio_path)
    restored_audio_path = decode_tokens(speech_ids)
    return str(speech_ids), restored_audio_path

def decode(tokens_str: str):
    speech_ids = [int(t.strip("[").strip("]").strip()) for t in tokens_str.split(",")]
    restored_audio_path = decode_tokens(speech_ids)
    return restored_audio_path

def tokenizer_interface():
    with gr.Blocks() as tokenizer_ui:
        gr.Markdown("## 🛠️ トークナイザー インターフェース")
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(label="🎧 音声入力", type="filepath")
                encode_button = gr.Button("🔤 トークン化+復元", variant="primary")
            with gr.Column():
                audio_output = gr.Audio(label="🔊 復元音声", type="filepath")
                tokens_output = gr.Textbox(label="🪙 音声トークン", placeholder="ここに音声トークンが表示されます...", lines=10)
                decode_button = gr.Button("🎵 デコード", variant="secondary")

        encode_button.click(
            encode_decode,
            inputs=audio_input,
            outputs=[tokens_output, audio_output]
        )

        decode_button.click(
            decode,
            inputs=tokens_output,
            outputs=audio_output
        )

    return tokenizer_ui
