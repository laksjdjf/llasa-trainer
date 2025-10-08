from ui.llasa_processor import load
from ui.tts import tts_interface
from ui.tokenizer import tokenizer_interface
from ui.similarity import similarity_interface
import argparse
import gradio as gr
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLASA TTS UI")
    parser.add_argument("-m", "--model_path", type=str, default="server", help="モデルのパスまたは'server'を指定")
    parser.add_argument("-c", "--codec_model_path", type=str, default="Anime-XCodec2-hf", help="コーデックモデルのパス")
    parser.add_argument("--host", type=str, help="ホスト名")
    parser.add_argument("--port", type=int, default=7860, help="ポート番号")
    parser.add_argument("--cuda_visible_devices", type=str, default="0", help="使用するCUDAデバイス (例: '0', '0,1')")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    # モデルのロード
    load(args.model_path, args.codec_model_path)

    demo = gr.TabbedInterface(
        [
            tts_interface(),
            tokenizer_interface(),
            similarity_interface()
        ],
        [
            "🗣️ TTS",
            "🔤 トークナイザー",
            "🎤 類似度計算"
        ],
        theme=gr.themes.Soft(),
        title="LLASA TTS Interface"
    )

    demo.launch(server_name=args.host, server_port=args.port)