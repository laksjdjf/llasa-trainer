"""
Make weight keys compatible with hf convert script.

1.download origin checkpoint from: https://huggingface.co/NandemoGHS/Anime-XCodec2
2.run this script to convert keys
3.run the official convert script:
python venv/lib/python3.12/site-packages/transformers/models/xcodec2/convert_xcodec2_checkpoint_to_pytorch.py
 --checkpoint_path origin_ckpt/model.safetensors
 --config_path origin_ckpt/config.json 
 --pytorch_dump_folder_path Anime-XCodec2-hf
"""


from safetensors.torch import save_file, load_file
pt = load_file("origin_ckpt/model.safetensors")

new_pt = {}

replace = {
    "parametrizations.weight.original0": "weight_g",
    "parametrizations.weight.original1": "weight_v",
    "act.bias": "act.beta"
}

for k, v in pt.items():
    ok = k
    for rk, rv in replace.items():
        k = k.replace(rk, rv)
    if ok != k:
        print(f"{ok} -> {k}")
    new_pt[k] = v

save_file(new_pt, "origin_ckpt/model_c.safetensors")