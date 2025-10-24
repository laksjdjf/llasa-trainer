"""Tests for modules/train_utils.py"""
import pytest
import json
import tempfile
from pathlib import Path

from modules.train_utils import load_dataset


class TestLoadDataset:
    """Test dataset loading function"""
    
    def test_load_dataset_basic(self):
        """Test loading a basic JSONL dataset"""
        data = [
            {"text": "こんにちは", "code": [1, 2, 3, 4, 5]},
            {"text": "ありがとう", "code": [6, 7, 8, 9, 10]},
            {"text": "さようなら", "code": [11, 12, 13, 14, 15]},
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.jsonl') as tmp:
            for item in data:
                tmp.write(json.dumps(item, ensure_ascii=False) + '\n')
            tmp_path = tmp.name
        
        try:
            dataset = load_dataset(tmp_path)
            assert len(dataset) == 3
            # Each entry should have been converted to a prompt
            for item in dataset:
                assert "text" in item
                assert isinstance(item["text"], str)
                # The dataset is shuffled, so we can't check specific indices
                # Just verify that all original texts appear somewhere in the dataset
            
            # Check that all original texts are present
            all_prompts = [item["text"] for item in dataset]
            combined_prompts = " ".join(all_prompts)
            for original_item in data:
                assert original_item["text"] in combined_prompts
        finally:
            Path(tmp_path).unlink()
    
    def test_load_dataset_with_prompts(self):
        """Test that loaded dataset contains properly formatted prompts"""
        data = [
            {"text": "こんにちは", "code": [1, 2, 3]},
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.jsonl') as tmp:
            for item in data:
                tmp.write(json.dumps(item, ensure_ascii=False) + '\n')
            tmp_path = tmp.name
        
        try:
            dataset = load_dataset(tmp_path)
            assert len(dataset) == 1
            prompt = dataset[0]["text"]
            # Check prompt structure
            assert "<|start_header_id|>system<|end_header_id|>" in prompt
            assert "<|TEXT_UNDERSTANDING_START|>" in prompt
            assert "<|TEXT_UNDERSTANDING_END|>" in prompt
            assert "<|SPEECH_GENERATION_START|>" in prompt
            assert "<|SPEECH_GENERATION_END|>" in prompt
            # Check speech tokens are included
            assert "<|s_1|>" in prompt
            assert "<|s_2|>" in prompt
            assert "<|s_3|>" in prompt
        finally:
            Path(tmp_path).unlink()
    
    def test_load_dataset_empty(self):
        """Test loading an empty dataset"""
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.jsonl') as tmp:
            tmp_path = tmp.name
        
        try:
            dataset = load_dataset(tmp_path)
            assert len(dataset) == 0
        finally:
            Path(tmp_path).unlink()
    
    def test_load_dataset_without_code(self):
        """Test loading dataset where some entries don't have code"""
        data = [
            {"text": "こんにちは", "code": [1, 2, 3]},
            {"text": "ありがとう"},  # No code
            {"text": "さようなら", "code": [4, 5, 6]},
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.jsonl') as tmp:
            for item in data:
                tmp.write(json.dumps(item, ensure_ascii=False) + '\n')
            tmp_path = tmp.name
        
        try:
            dataset = load_dataset(tmp_path)
            assert len(dataset) == 3
            # All entries should be loaded
            for item in dataset:
                assert "text" in item
        finally:
            Path(tmp_path).unlink()
    
    def test_load_dataset_is_shuffled(self):
        """Test that dataset is shuffled (statistical test)"""
        # Create a dataset with identifiable order
        data = [{"text": f"text_{i:04d}", "code": [i]} for i in range(100)]
        
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.jsonl') as tmp:
            for item in data:
                tmp.write(json.dumps(item, ensure_ascii=False) + '\n')
            tmp_path = tmp.name
        
        try:
            dataset = load_dataset(tmp_path)
            assert len(dataset) == 100
            
            # Check if the order has changed (very unlikely to be in original order after shuffle)
            # Extract the indices from text
            original_order_count = 0
            for i, item in enumerate(dataset):
                if f"text_{i:04d}" in item["text"]:
                    original_order_count += 1
            
            # If shuffled, very unlikely to have most items in original positions
            # Allow for some coincidental matches
            assert original_order_count < 50, "Dataset does not appear to be shuffled"
        finally:
            Path(tmp_path).unlink()
    
    def test_load_dataset_large_codes(self):
        """Test loading dataset with large code arrays"""
        data = [
            {"text": "長いテキスト", "code": list(range(1000))},
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.jsonl') as tmp:
            for item in data:
                tmp.write(json.dumps(item, ensure_ascii=False) + '\n')
            tmp_path = tmp.name
        
        try:
            dataset = load_dataset(tmp_path)
            assert len(dataset) == 1
            # Should successfully create prompt even with large code array
            assert "text" in dataset[0]
        finally:
            Path(tmp_path).unlink()
    
    def test_load_dataset_unicode_text(self):
        """Test loading dataset with various unicode characters"""
        data = [
            {"text": "こんにちは♡テスト…", "code": [1, 2, 3]},
            {"text": "ありがとう！？", "code": [4, 5, 6]},
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.jsonl') as tmp:
            for item in data:
                tmp.write(json.dumps(item, ensure_ascii=False) + '\n')
            tmp_path = tmp.name
        
        try:
            dataset = load_dataset(tmp_path)
            assert len(dataset) == 2
            # Text should be normalized in prompts
            for item in dataset:
                assert "text" in item
        finally:
            Path(tmp_path).unlink()
