# Tests for LLASA-Trainer

このディレクトリには、LLASA-Trainerのユニットテストが含まれています。

## テストの実行

すべてのテストを実行:
```bash
pytest tests/
```

特定のテストファイルを実行:
```bash
pytest tests/test_llasa_utils.py
```

詳細な出力:
```bash
pytest tests/ -v
```

カバレッジレポート付き:
```bash
pytest tests/ --cov=modules --cov=create_dataset
```

## テストファイル

### `test_llasa_utils.py`
`modules/llasa_utils.py`のユーティリティ関数のテスト:
- **TestNormalizeText**: テキスト正規化関数のテスト
  - 全角/半角文字の変換
  - 特殊文字の削除と置換
  - スペースやタブの処理
  
- **TestSpeechTokens**: 音声トークン変換関数のテスト
  - IDから音声トークンへの変換
  - 音声トークンからIDへの抽出
  
- **TestGetPrompt**: プロンプト生成関数のテスト
  - 基本的なプロンプト生成
  - キャプションメタデータの処理
  - テキスト正規化の統合
  
- **TestPreprocessAudio**: 音声前処理関数のテスト
  - サンプリングレート変換
  - ステレオからモノラルへの変換
  - 辞書形式の入力処理

### `test_create_dataset.py`
`create_dataset.py`のデータセット作成ロジックのテスト:
- **TestParseTextFile**: テキストファイル解析関数のテスト
  - コロン区切りテキストの解析
  - タブ区切りテキストの解析
  - カラム順序の柔軟性
  - Unicode文字の処理

### `test_train_utils.py`
`modules/train_utils.py`のトレーニングユーティリティのテスト:
- **TestLoadDataset**: データセット読み込み関数のテスト
  - JSONL形式の読み込み
  - プロンプトへの変換
  - データセットのシャッフル
  - 大きなコード配列の処理

## テストカバレッジ

現在のテストカバレッジ:
- `modules/llasa_utils.py`: 主要な関数を完全にカバー
- `create_dataset.py`: テキスト解析ロジックをカバー
- `modules/train_utils.py`: データセット読み込みをカバー

## 依存関係

テストを実行するには以下のパッケージが必要です:
```bash
pip install pytest pytest-mock
```

## 注意事項

- 一部の音声処理テストは、torchaudioのバックエンド依存関係のためスキップされます
- テストは軽量で、大きなモデルやデータセットをロードしません
- テストは独立して実行でき、相互依存関係はありません

## CI/CD統合

これらのテストは、GitHubアクションやその他のCI/CDパイプラインで自動実行できます。

```yaml
# .github/workflows/test.yml の例
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - run: pip install -r requirements.txt
      - run: pytest tests/
```
