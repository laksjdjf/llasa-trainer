import gradio as gr
from pathlib import Path
from ui.llasa_processor import calc_similarity

def calculate_similarity(target, references):
    if not target or not references:
        return "", "⚠️ ターゲット音声と参照音声を両方指定してください。"
    similarities = calc_similarity(target, references)
    similarity_dict = {Path(ref).name: sim for ref, sim in zip(references, similarities)}

    similarity_csv = "\n".join([f"{k},{v:.4f}" for k, v in similarity_dict.items()])
    return similarity_dict, similarity_csv, "✅ 類似度計算完了！"

def similarity_interface():
    """音声類似度計算UIを作成"""
    with gr.Blocks(title="LLASA Similarity Calculator", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎤 LLASA Similarity Calculator")
        gr.Markdown("音声ファイル間の類似度を計算するツール")
        
        with gr.Row():
            with gr.Column():
                target_audio = gr.Audio(label="📁 ターゲット音声ファイル", type="filepath")
                reference_audios = gr.File(label="📂 参照音声ファイル（複数選択可）", file_types=["audio"], file_count="multiple")
                calc_btn = gr.Button("🔍 類似度計算実行", variant="primary")
            
            with gr.Column():
                similarity_output = gr.Label(
                    label="📊 類似度結果",
                    num_top_classes=10,
                )
                similarity_table = gr.Textbox(label="類似度詳細", interactive=False)
                status_output = gr.Textbox(label="📈 ステータス", interactive=False)
        
        calc_btn.click(
            calculate_similarity,
            inputs=[target_audio, reference_audios],
            outputs=[similarity_output, similarity_table, status_output],
        )
    
    return app