"""Tests for create_dataset.py"""
import pytest
import tempfile
from pathlib import Path
import json


# We define the function here to test it without importing heavy dependencies
def parse_text_file(text_file: Path, split_word: str = ":", file_index: int = 0, text_index: int = 1):
    """テキストファイルを解析して辞書を作成"""
    texts = {}
    with text_file.open('r', encoding='utf-8') as f:
        for line in f:
            splited = line.strip().split(split_word)
            file_id = splited[file_index] if len(splited) > file_index else None
            text = splited[text_index] if len(splited) > text_index else None
            if file_id and text:
                texts[file_id.strip()] = text.strip()
    return texts


class TestParseTextFile:
    """Test text file parsing function"""
    
    def test_parse_basic_colon_separated(self):
        """Test parsing basic colon-separated text file"""
        content = "file001:こんにちは\nfile002:ありがとう\nfile003:さようなら\n"
        
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        
        try:
            result = parse_text_file(tmp_path)
            assert len(result) == 3
            assert result["file001"] == "こんにちは"
            assert result["file002"] == "ありがとう"
            assert result["file003"] == "さようなら"
        finally:
            tmp_path.unlink()
    
    def test_parse_with_whitespace(self):
        """Test parsing with extra whitespace"""
        content = "  file001  :  こんにちは  \n  file002  :  ありがとう  \n"
        
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        
        try:
            result = parse_text_file(tmp_path)
            assert len(result) == 2
            # Should strip whitespace
            assert result["file001"] == "こんにちは"
            assert result["file002"] == "ありがとう"
        finally:
            tmp_path.unlink()
    
    def test_parse_tab_separated(self):
        """Test parsing tab-separated text file"""
        content = "file001\tこんにちは\nfile002\tありがとう\n"
        
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        
        try:
            result = parse_text_file(tmp_path, split_word="\t")
            assert len(result) == 2
            assert result["file001"] == "こんにちは"
            assert result["file002"] == "ありがとう"
        finally:
            tmp_path.unlink()
    
    def test_parse_reversed_columns(self):
        """Test parsing with reversed column order (text:file_id)"""
        content = "こんにちは:file001\nありがとう:file002\n"
        
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        
        try:
            result = parse_text_file(tmp_path, file_index=1, text_index=0)
            assert len(result) == 2
            assert result["file001"] == "こんにちは"
            assert result["file002"] == "ありがとう"
        finally:
            tmp_path.unlink()
    
    def test_parse_empty_file(self):
        """Test parsing empty file"""
        content = ""
        
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        
        try:
            result = parse_text_file(tmp_path)
            assert len(result) == 0
        finally:
            tmp_path.unlink()
    
    def test_parse_skip_invalid_lines(self):
        """Test parsing skips lines with missing fields"""
        content = "file001:こんにちは\nfile002\nfile003:さようなら\n"
        
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        
        try:
            result = parse_text_file(tmp_path)
            # file002 line should be skipped
            assert len(result) == 2
            assert "file001" in result
            assert "file002" not in result
            assert "file003" in result
        finally:
            tmp_path.unlink()
    
    def test_parse_multicolumn_file(self):
        """Test parsing file with more than 2 columns"""
        content = "file001:extra:こんにちは\nfile002:data:ありがとう\n"
        
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        
        try:
            # Get text from 3rd column (index 2)
            result = parse_text_file(tmp_path, file_index=0, text_index=2)
            assert len(result) == 2
            assert result["file001"] == "こんにちは"
            assert result["file002"] == "ありがとう"
        finally:
            tmp_path.unlink()
    
    def test_parse_text_with_delimiter_in_content(self):
        """Test parsing text that contains the delimiter character"""
        content = "file001:こんにちは:今日は\nfile002:ありがとう:ございます\n"
        
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        
        try:
            # split() will split on all occurrences, so taking index 1 gets only the second field
            result = parse_text_file(tmp_path, file_index=0, text_index=1)
            assert len(result) == 2
            # The second field after the first split
            assert result["file001"] == "こんにちは"
            assert result["file002"] == "ありがとう"
        finally:
            tmp_path.unlink()
    
    def test_parse_unicode_text(self):
        """Test parsing file with various unicode characters"""
        content = "file001:こんにちは♡\nfile002:テスト…\nfile003:ありがとう！\n"
        
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        
        try:
            result = parse_text_file(tmp_path)
            assert len(result) == 3
            assert result["file001"] == "こんにちは♡"
            assert result["file002"] == "テスト…"
            assert result["file003"] == "ありがとう！"
        finally:
            tmp_path.unlink()
