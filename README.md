# llasa-trainer

LLASA-3B TTS（Text-to-Speech）モデルのLoRA微調整トレーナー

## 概要

このリポジトリは、LLASA-3B音声合成モデルをLoRA（Low-Rank Adaptation）で効率的に微調整するためのツールです。

## 特徴

- ✅ LoRAによる効率的なファインチューニング
- ✅ 混合精度学習（FP16/BF16）対応
- ✅ 勾配蓄積による大バッチサイズシミュレーション
- ✅ 学習中の音声生成テスト機能
- ✅ バリデーションとEarly Stopping対応
- ✅ 最適化されたデータローダー設定

## セットアップ

```bash
# 依存パッケージのインストール
pip install transformers peft trl datasets torch soundfile xcodec2
```

## 使い方

### 1. データセットの準備

JSONL形式でデータを準備：

```jsonl
{"text": "こんにちは、今日はいい天気ですね。"}
{"text": "ありがとうございます！"}
```

### 2. 設定ファイルの作成

`config/example.yaml`を参考に設定ファイルを作成：

```bash
cp config/example.yaml config/my_config.yaml
# my_config.yamlを編集
```

### 3. 学習の実行

```bash
python main.py --config config/my_config.yaml
```

## 学習精度・効率を上げるための推奨設定

### GPU別の推奨設定

#### A100/H100などの最新GPU
```yaml
training:
  bf16: true                           # BFloat16を使用
  fp16: false
  gradient_checkpointing: true         # メモリ節約
  optim: adamw_torch_fused             # 最速オプティマイザ
  dataloader_num_workers: 4            # 並列データ読み込み
```

#### RTX 3090/4090などのAmpere以降
```yaml
training:
  bf16: true                           # BFloat16対応
  fp16: false
  gradient_checkpointing: true
  optim: adamw_torch_fused
  dataloader_num_workers: 2
```

#### V100などの古めのGPU
```yaml
training:
  fp16: true                           # FP16を使用
  bf16: false
  gradient_checkpointing: true
  optim: adamw_torch                   # 互換性重視
  dataloader_num_workers: 2
```

### 過学習を防ぐための設定

```yaml
# バリデーションデータの設定
eval_data_dir: dataset/val.jsonl

training:
  evaluation_strategy: steps
  eval_steps: 100
  load_best_model_at_end: true
  
  # Early Stopping
  use_early_stopping: true
  early_stopping_patience: 3
  
  # 正則化
  weight_decay: 0.01
  lora_dropout: 0.05
```

### 学習速度を上げるための設定

```yaml
training:
  gradient_accumulation_steps: 8       # 実効バッチサイズを増やす
  dataloader_num_workers: 4            # データ読み込みを並列化
  dataloader_pin_memory: true          # GPU転送を高速化
  gradient_checkpointing: false        # メモリに余裕があればoff
  optim: adamw_torch_fused             # 最速オプティマイザ
  max_grad_norm: 1.0                   # 勾配クリッピングで安定化
```

### LoRAパラメータの調整

```yaml
lora:
  r: 32                                # ランクを増やす（表現力UP、メモリUP）
  lora_alpha: 64                       # rの2倍が一般的
  lora_dropout: 0.05                   # 過学習防止
  target_modules:                      # より多くのレイヤーに適用
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj                        # 追加
    - up_proj                          # 追加
    - down_proj                        # 追加
```

## 設定パラメータ詳細

### 基本設定
- `data_dir`: 学習データのパス（JSONL形式）
- `eval_data_dir`: バリデーションデータのパス（オプション）
- `output_dir`: 学習結果の保存先
- `model_name`: ベースモデル名

### LoRA設定
- `r`: LoRAのランク（8-64推奨、大きいほど表現力UP）
- `lora_alpha`: スケーリング係数（通常はrの2倍）
- `lora_dropout`: ドロップアウト率（0.0-0.1、過学習防止）
- `target_modules`: LoRAを適用するレイヤー

### 学習設定
- `num_train_epochs`: エポック数
- `per_device_train_batch_size`: デバイス当たりのバッチサイズ
- `gradient_accumulation_steps`: 勾配蓄積ステップ数
- `learning_rate`: 学習率（1e-4〜5e-4推奨）
- `weight_decay`: 重み減衰（0.01推奨）
- `bf16/fp16`: 混合精度学習（BF16推奨）
- `gradient_checkpointing`: 勾配チェックポイント（メモリ節約）
- `optim`: オプティマイザ（adamw_torch_fused推奨）
- `max_grad_norm`: 勾配クリッピング（1.0推奨）

### バリデーション設定
- `evaluation_strategy`: 評価戦略（steps/epoch/no）
- `eval_steps`: 評価間隔
- `load_best_model_at_end`: 最良モデルをロード
- `use_early_stopping`: Early Stopping有効化
- `early_stopping_patience`: 改善なしで停止までのステップ数

### データローダー設定
- `dataloader_num_workers`: ワーカー数（2-4推奨）
- `dataloader_pin_memory`: メモリピン留め（GPU転送高速化）

## トラブルシューティング

### OOM（Out of Memory）エラー
1. `per_device_train_batch_size`を減らす（1または2）
2. `gradient_checkpointing: true`に設定
3. `bf16: true`または`fp16: true`を有効化
4. LoRAの`r`を減らす（8-16）

### 学習が遅い
1. `dataloader_num_workers`を増やす（2-4）
2. `optim: adamw_torch_fused`に変更
3. `gradient_checkpointing: false`に設定（メモリに余裕があれば）
4. `gradient_accumulation_steps`を減らす

### 過学習している
1. バリデーションデータを設定
2. `use_early_stopping: true`を有効化
3. `weight_decay`を増やす（0.01-0.1）
4. `lora_dropout`を増やす（0.05-0.1）
5. エポック数を減らす

## ライセンス

このプロジェクトのライセンスについては、LICENSEファイルを参照してください。