# 学習最適化ガイド

このガイドでは、LLASA TTSトレーナーの学習精度と効率を向上させるための具体的な方法を説明します。

## 目次
1. [クイックスタート](#クイックスタート)
2. [GPU別の推奨設定](#gpu別の推奨設定)
3. [精度向上のテクニック](#精度向上のテクニック)
4. [速度向上のテクニック](#速度向上のテクニック)
5. [メモリ最適化](#メモリ最適化)
6. [トラブルシューティング](#トラブルシューティング)

## クイックスタート

### 1. 最適化済み設定を使う

```bash
# 標準的な設定（推奨）
python main.py --config config/optimized_training.yaml

# 省メモリ設定（VRAM 8-12GB）
python main.py --config config/low_memory.yaml
```

### 2. データセットの分割

学習精度を正しく評価するため、データセットを学習用とバリデーション用に分割します：

```python
# split_dataset.py
import json
import random

# データを読み込み
with open('dataset/all_data.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# シャッフル
random.shuffle(data)

# 90%学習、10%バリデーション
split_idx = int(len(data) * 0.9)
train_data = data[:split_idx]
val_data = data[split_idx:]

# 保存
with open('dataset/train.jsonl', 'w') as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

with open('dataset/val.jsonl', 'w') as f:
    for item in val_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"学習: {len(train_data)}件, バリデーション: {len(val_data)}件")
```

## GPU別の推奨設定

### A100 / H100 (80GB VRAM)
```yaml
training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 2
  bf16: true
  gradient_checkpointing: false  # VRAMに余裕あり
  optim: adamw_torch_fused
  dataloader_num_workers: 8

lora:
  r: 64  # 高い表現力
  lora_alpha: 128
```

### A100 / H100 (40GB VRAM)
```yaml
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  bf16: true
  gradient_checkpointing: true
  optim: adamw_torch_fused
  dataloader_num_workers: 4

lora:
  r: 32
  lora_alpha: 64
```

### RTX 4090 (24GB VRAM)
```yaml
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  bf16: true
  gradient_checkpointing: true
  optim: adamw_torch_fused
  dataloader_num_workers: 4

lora:
  r: 32
  lora_alpha: 64
```

### RTX 3090 (24GB VRAM)
```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  bf16: true  # Ampere世代はBF16対応
  gradient_checkpointing: true
  optim: adamw_torch
  dataloader_num_workers: 2

lora:
  r: 16
  lora_alpha: 32
```

### V100 (16GB VRAM)
```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  fp16: true  # V100はBF16非対応
  gradient_checkpointing: true
  optim: adamw_8bit  # メモリ節約
  dataloader_num_workers: 2

lora:
  r: 16
  lora_alpha: 32
```

### RTX 3060 / 2080 Ti (8-12GB VRAM)
```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  fp16: true
  gradient_checkpointing: true
  optim: adamw_8bit
  dataloader_num_workers: 0
  load_best_model_at_end: false  # メモリ節約

lora:
  r: 8
  lora_alpha: 16
  target_modules: [q_proj, v_proj]  # 最小限
```

## 精度向上のテクニック

### 1. バリデーション設定

```yaml
# 設定ファイル
eval_data_dir: dataset/val.jsonl

training:
  evaluation_strategy: steps
  eval_steps: 50
  load_best_model_at_end: true
  metric_for_best_model: eval_loss
```

### 2. Early Stopping

過学習を自動で防ぎます：

```yaml
training:
  use_early_stopping: true
  early_stopping_patience: 5  # 5回改善なしで停止
  early_stopping_threshold: 0.001
```

### 3. 学習率の調整

```yaml
training:
  learning_rate: 5e-5  # 低めの学習率で安定
  warmup_ratio: 0.05   # ウォームアップ
  lr_scheduler_type: cosine_with_restarts
```

### 4. LoRAパラメータの最適化

```yaml
lora:
  r: 32  # ランクを上げて表現力向上
  lora_alpha: 64
  lora_dropout: 0.05  # 過学習防止
  target_modules:  # より多くのレイヤー
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
```

### 5. 正則化

```yaml
training:
  weight_decay: 0.01  # L2正則化
  max_grad_norm: 1.0  # 勾配クリッピング

lora:
  lora_dropout: 0.05  # ドロップアウト
```

## 速度向上のテクニック

### 1. オプティマイザの選択

```yaml
# 最速（Ampere以降のGPU）
optim: adamw_torch_fused

# バランス型
optim: adamw_torch

# メモリ重視
optim: adamw_8bit
```

### 2. データローダーの最適化

```yaml
training:
  dataloader_num_workers: 4      # データ読み込みを並列化
  dataloader_pin_memory: true    # GPU転送高速化
  group_by_length: true          # 同じ長さをまとめる
```

### 3. 勾配蓄積の調整

```yaml
training:
  per_device_train_batch_size: 2  # 可能な限り大きく
  gradient_accumulation_steps: 4   # 合計バッチサイズ = 8
```

### 4. 混合精度学習

```yaml
# BF16（推奨、Ampere以降）
bf16: true
fp16: false

# FP16（古いGPU）
fp16: true
bf16: false
```

### 5. グラディエントチェックポイント

速度 vs メモリのトレードオフ：

```yaml
# 速度重視（VRAMに余裕あり）
gradient_checkpointing: false

# メモリ重視（VRAM不足気味）
gradient_checkpointing: true
```

## メモリ最適化

### 1. バッチサイズとGradient Accumulation

```yaml
# 設定例：実効バッチサイズ = 8
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
```

### 2. LoRAランクの削減

```yaml
lora:
  r: 8  # 16 → 8に削減
  target_modules: [q_proj, v_proj]  # 最小限
```

### 3. 8bit Optimizer

```yaml
training:
  optim: adamw_8bit  # メモリ使用量が大幅に削減
```

### 4. Gradient Checkpointing

```yaml
training:
  gradient_checkpointing: true  # メモリ40-50%削減
```

### 5. 不要な機能をオフ

```yaml
training:
  load_best_model_at_end: false  # メモリ節約
  dataloader_num_workers: 0      # メモリ節約
  save_total_limit: 2            # ディスク容量節約
```

## トラブルシューティング

### OOM (Out of Memory) エラー

**症状**: `CUDA out of memory` エラー

**解決策**:
1. `per_device_train_batch_size: 1` に削減
2. `gradient_checkpointing: true` を有効化
3. `optim: adamw_8bit` に変更
4. LoRAの`r`を8に削減
5. `dataloader_num_workers: 0` に設定
6. `load_best_model_at_end: false` に設定

### 学習が遅い

**症状**: 1エポックに数時間かかる

**解決策**:
1. `dataloader_num_workers: 4` に増やす
2. `optim: adamw_torch_fused` に変更
3. `gradient_checkpointing: false` に設定（メモリに余裕があれば）
4. `per_device_train_batch_size` を増やす
5. `group_by_length: true` を有効化

### 過学習している

**症状**: 学習ロスは下がるがバリデーションロスが上がる

**解決策**:
1. バリデーションデータを設定
2. `use_early_stopping: true` を有効化
3. `weight_decay: 0.01-0.05` を増やす
4. `lora_dropout: 0.1` を増やす
5. エポック数を減らす
6. 学習率を下げる（`5e-5`など）

### 学習が不安定

**症状**: ロスが振動する、NaNになる

**解決策**:
1. 学習率を下げる（`1e-5`など）
2. `max_grad_norm: 0.5` でクリッピングを強化
3. `warmup_ratio: 0.1` でウォームアップを長く
4. `bf16` または `fp16` を有効化
5. `gradient_accumulation_steps` を増やす

### バリデーションロスが改善しない

**症状**: eval_lossが下がらない

**解決策**:
1. データセットの質を確認
2. バリデーションデータと学習データの分布を確認
3. LoRAの`r`を増やす（16 → 32）
4. より多くの`target_modules`を追加
5. 学習率を調整（`5e-5` - `1e-4`）

## 推奨ワークフロー

### 1. 初期実験（小規模）
```yaml
num_train_epochs: 3
save_steps: 50
eval_steps: 50
```

- 小さいエポック数で動作確認
- OOMエラーがないか確認
- バリデーションロスの挙動を観察

### 2. パラメータチューニング
```yaml
num_train_epochs: 10
use_early_stopping: true
```

- 学習率を調整（1e-5, 5e-5, 1e-4を試す）
- LoRAのランクを調整（8, 16, 32を試す）
- バリデーションロスが最良の設定を選択

### 3. 本番学習
```yaml
num_train_epochs: 20-30
early_stopping_patience: 5
```

- 最適なハイパーパラメータで長時間学習
- TensorBoardでモニタリング
- 定期的に生成音声を確認

## モニタリング

### TensorBoardの使用

```yaml
training:
  report_to: tensorboard
  logging_dir: ./logs
```

起動:
```bash
tensorboard --logdir ./logs
```

### 監視すべき指標

1. **train_loss**: 順調に減少しているか
2. **eval_loss**: 過学習していないか
3. **learning_rate**: スケジューリングが適切か
4. **生成音声**: 定期的に音質を確認

## まとめ

- **精度優先**: バリデーション + Early Stopping + 大きめのLoRAランク
- **速度優先**: BF16 + Fused Optimizer + 並列データローダー
- **メモリ優先**: 小バッチ + Gradient Checkpointing + 8bit Optimizer

環境とニーズに応じて設定を調整し、最適なバランスを見つけてください。
