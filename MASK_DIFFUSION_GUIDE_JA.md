# LLASAマスク拡散（Mask Diffusion）使用ガイド

## 概要

このドキュメントは、LLASAトレーナーのマスク拡散機能の使い方を説明します。

## マスク拡散とは？

マスク拡散は、従来の自己回帰的な生成（Causal LM）とは異なるアプローチです：

- **学習時**: 音声トークンの一部をランダムにマスクし、マスクされたトークンを予測
- **推論時**: 全ての音声トークンをマスクした状態から始め、反復的に修復

### メリット

1. **非自己回帰的な生成**: 複数のトークンを並行して予測可能
2. **音声の編集が可能**: 音声の特定部分を修復（インペインティング）できる
3. **高品質な生成**: 反復的な改善により、より高品質な音声生成が期待できる

## 実装内容

### 1. マスク予測用の学習コード

**ファイル**: `modules/mask_diffusion_train.py`

- 音声トークンのみをマスク（テキストトークンはマスクしない）
- マスク予測のための損失計算
- Causal LMのforwardを書き換えずに、データコレーターとカスタムトレーナーで実現

**主要機能**:
- `MaskDiffusionTrainer`: マスク拡散トレーニングクラス
- `mask_tokens()`: 音声トークンをランダムにマスク
- `compute_loss()`: マスク予測損失を計算

### 2. マスクを修復するサンプリングコード

**ファイル**: `modules/mask_diffusion_sampling.py`

- 反復的にマスクされたトークンを修復
- 信頼度ベースで段階的に予測
- Top-pサンプリングをサポート

**主要機能**:
- `MaskDiffusionSampler`: マスク拡散サンプリングクラス
- `iterative_decode()`: 反復的なデコード
- `generate_with_mask_diffusion()`: マスク拡散による音声生成
- `inpaint_audio()`: 音声の特定部分を修復

## 使い方

### 学習

1. **設定ファイルを作成**:
```bash
cp config/mask_diffusion_example.yaml config/my_mask_diffusion.yaml
```

2. **設定を編集** (`config/my_mask_diffusion.yaml`):
```yaml
# マスク拡散の設定
mask_diffusion:
  mask_ratio: 0.15  # マスクする音声トークンの割合（0.10-0.30推奨）

# データとモデルパス
data_dir: dataset/data.jsonl
output_dir: ./trained/MaskDiffusionModel
model_name: NandemoGHS/Anime-Llasa-3B

# LoRA設定（オプション）
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj

# 学習設定
training:
  num_train_epochs: 20
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1e-4
  bf16: true
```

3. **学習を開始**:
```bash
python main_mask_diffusion.py --config config/my_mask_diffusion.yaml
```

### 推論

**Pythonコードで使用**:

```python
from modules.llasa_mask_diffusion import LLASAMaskDiffusion

# モデルを読み込み
llasa_md = LLASAMaskDiffusion.from_pretrained(
    model_path="./trained/MaskDiffusionModel",
    codec_model_path="Anime-XCodec2-hf",
)

# 音声を生成
audio_path, speech_ids = llasa_md.generate_with_mask_diffusion(
    text="こんにちは、今日はいい天気ですね。",
    num_audio_tokens=300,      # 生成する音声トークン数
    num_iterations=10,          # 反復回数（多いほど高品質）
    temperature=0.7,            # サンプリング温度
    top_p=0.9,                  # Top-pサンプリング
    confidence_threshold=0.9,   # 信頼度のしきい値
)

print(f"音声ファイル: {audio_path}")
print(f"生成されたトークン数: {len(speech_ids)}")
```

### 音声の修復（インペインティング）

```python
# 音声の特定部分を修復
inpainted_codes = llasa_md.inpaint_audio(
    original_codes=original_audio_codes,  # 元の音声トークン
    mask_start=100,                        # 修復開始位置
    mask_end=200,                          # 修復終了位置
    num_iterations=10,                     # 反復回数
    temperature=0.7,
    top_p=0.9,
)
```

### デモスクリプト

機能を確認するためのデモスクリプトが用意されています：

```bash
# すべてのデモを実行
python demo_mask_diffusion.py --mode all

# 生成デモのみ
python demo_mask_diffusion.py --mode generation

# インペインティングデモのみ
python demo_mask_diffusion.py --mode inpainting

# データコレーターのデモのみ
python demo_mask_diffusion.py --mode collator
```

## パラメータの説明

### 学習時のパラメータ

- **mask_ratio** (0.15): マスクする音声トークンの割合
  - 0.10-0.30を推奨
  - 大きすぎると学習が難しくなる
  - 小さすぎると学習が不十分になる

### 推論時のパラメータ

- **num_audio_tokens** (300): 生成する音声トークンの数
  - 音声の長さを制御
  
- **num_iterations** (10): 反復回数
  - 多いほど高品質だが、時間がかかる
  - 5-20を推奨
  
- **temperature** (0.7): サンプリング温度
  - 低い値: より確定的な生成
  - 高い値: より多様性のある生成
  
- **top_p** (0.9): Top-pサンプリング
  - 0.8-0.95を推奨
  
- **confidence_threshold** (0.9): 信頼度のしきい値
  - 高い値: より慎重に生成
  - 低い値: より早く収束

## 注意点

1. **音声トークンのみマスク**: テキストトークンはマスクされません
2. **反復的な生成**: 通常のCausal LMより時間がかかります
3. **モデルサイズ**: LoRAを使用することでメモリ使用量を削減できます
4. **データセット**: 通常の学習と同じデータセット形式を使用します

## トラブルシューティング

### メモリ不足

- `per_device_train_batch_size`を1に減らす
- `gradient_accumulation_steps`を増やす
- LoRAを使用する
- `gradient_checkpointing: true`を設定

### 学習が遅い

- `num_train_epochs`を減らす
- `save_steps`を大きくする
- `bf16: true`を使用（A100等のGPUで）

### 生成品質が低い

- `num_iterations`を増やす
- `mask_ratio`を調整（0.15前後が適切）
- 学習エポック数を増やす
- より大きなデータセットを使用

## まとめ

マスク拡散実装により、以下が可能になりました：

- ✅ マスク予測用の学習コード（音声トークンのみ対象）
- ✅ マスクを修復するサンプリングコード
- ✅ 音声のインペインティング（部分的な修復）
- ✅ 非自己回帰的な音声生成

詳細な技術情報は `MASK_DIFFUSION_SUMMARY.md`（英語）を参照してください。
