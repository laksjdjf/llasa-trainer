#!/usr/bin/env python3
"""
データセット分割スクリプト

学習データをトレーニング用とバリデーション用に分割します。
過学習を防ぎ、モデルの汎化性能を向上させるために必要です。

使用方法:
    python scripts/split_dataset.py input.jsonl --train-ratio 0.9

オプション:
    --train-ratio: 学習データの割合（デフォルト: 0.9）
    --output-dir: 出力先ディレクトリ（デフォルト: dataset）
    --shuffle: データをシャッフル（デフォルト: True）
    --seed: 乱数シード（デフォルト: 42）
"""

import argparse
import json
import random
from pathlib import Path


def split_dataset(
    input_file: str,
    train_ratio: float = 0.9,
    output_dir: str = "dataset",
    shuffle: bool = True,
    seed: int = 42
):
    """データセットを学習用とバリデーション用に分割"""
    
    # 入力ファイルの存在確認
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {input_file}")
    
    # データを読み込み
    print(f"📂 データセットを読み込み中: {input_file}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    total_count = len(data)
    print(f"📊 総データ数: {total_count}件")
    
    if total_count == 0:
        raise ValueError("データが空です")
    
    # シャッフル
    if shuffle:
        random.seed(seed)
        random.shuffle(data)
        print(f"🔀 データをシャッフル（seed={seed}）")
    
    # 分割
    split_idx = int(total_count * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    train_count = len(train_data)
    val_count = len(val_data)
    
    print(f"✂️ データを分割:")
    print(f"   学習: {train_count}件 ({train_count/total_count*100:.1f}%)")
    print(f"   バリデーション: {val_count}件 ({val_count/total_count*100:.1f}%)")
    
    # 出力ディレクトリの作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存
    train_file = output_path / "train.jsonl"
    val_file = output_path / "val.jsonl"
    
    print(f"💾 保存中...")
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"   学習: {train_file}")
    
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"   バリデーション: {val_file}")
    
    print("✅ 完了！")
    print("\n次のステップ:")
    print("1. config/example.yaml を編集")
    print(f"   data_dir: {train_file}")
    print(f"   eval_data_dir: {val_file}")
    print("2. 学習を実行")
    print("   python main.py --config config/example.yaml")


def main():
    parser = argparse.ArgumentParser(
        description="データセットを学習用とバリデーション用に分割",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    # 基本的な使い方（90%学習、10%バリデーション）
    python scripts/split_dataset.py dataset/all_data.jsonl
    
    # 比率を変更（80%学習、20%バリデーション）
    python scripts/split_dataset.py dataset/all_data.jsonl --train-ratio 0.8
    
    # 出力先を変更
    python scripts/split_dataset.py dataset/all_data.jsonl --output-dir my_dataset
    
    # シャッフルなし
    python scripts/split_dataset.py dataset/all_data.jsonl --no-shuffle
        """
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        help="入力ファイル（JSONL形式）"
    )
    
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="学習データの割合（デフォルト: 0.9）"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset",
        help="出力先ディレクトリ（デフォルト: dataset）"
    )
    
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="データをシャッフルしない"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード（デフォルト: 42）"
    )
    
    args = parser.parse_args()
    
    try:
        split_dataset(
            input_file=args.input_file,
            train_ratio=args.train_ratio,
            output_dir=args.output_dir,
            shuffle=not args.no_shuffle,
            seed=args.seed
        )
    except Exception as e:
        print(f"❌ エラー: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
