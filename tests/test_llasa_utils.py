"""Tests for modules/llasa_utils.py"""
import pytest
import torch
import torchaudio
from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf

from modules.llasa_utils import (
    normalize_text,
    ids_to_speech_tokens,
    extract_speech_ids,
    get_prompt,
    preprocess_audio,
    SAMPLING_RATE,
    BOS_TOKEN,
    PROMPT_FORMAT,
    DEFAULT_SYSTEM_PROMPT,
    CAPTION_FORMAT,
)


class TestNormalizeText:
    """Test text normalization function"""
    
    def test_normalize_basic_text(self):
        """Test basic text normalization"""
        text = "こんにちは"
        result = normalize_text(text)
        assert result == "こんにちは"
    
    def test_remove_spaces(self):
        """Test space removal (both half and full width)"""
        text = "こん にち は　テスト"
        result = normalize_text(text)
        assert " " not in result
        assert "　" not in result
        assert result == "こんにちはテスト"
    
    def test_remove_tab(self):
        """Test tab removal"""
        text = "こんにちは\tテスト"
        result = normalize_text(text)
        assert "\t" not in result
        assert result == "こんにちはテスト"
    
    def test_fullwidth_alpha_to_halfwidth(self):
        """Test full-width alphabets to half-width conversion"""
        text = "ＡＢＣａｂｃ"
        result = normalize_text(text)
        assert result == "ABCabc"
    
    def test_fullwidth_digits_to_halfwidth(self):
        """Test full-width digits to half-width conversion"""
        text = "０１２３４５６７８９"
        result = normalize_text(text)
        assert result == "0123456789"
    
    def test_halfwidth_katakana_to_fullwidth(self):
        """Test half-width katakana to full-width conversion"""
        text = "ｱｲｳｴｵ"
        result = normalize_text(text)
        # Half-width katakana should be converted to full-width
        assert "ｱ" not in result
        assert len(result) == 5  # Still 5 characters
    
    def test_replace_special_marks(self):
        """Test special character replacements"""
        text = "？！"
        result = normalize_text(text)
        assert result == "?!"
    
    def test_replace_tilde_to_longvowel(self):
        """Test tilde to long vowel mark conversion"""
        text = "〜テスト～"
        result = normalize_text(text)
        assert "〜" not in result
        assert "～" not in result
        assert "ー" in result
    
    def test_normalize_ellipsis(self):
        """Test ellipsis normalization (3 or more consecutive … to ……)"""
        text = "あ………い"
        result = normalize_text(text)
        assert result == "あ……い"
    
    def test_remove_invalid_characters(self):
        """Test removal of invalid characters"""
        text = "こんにちは●◯〇"
        result = normalize_text(text)
        assert "●" not in result
        # ○ should remain as it's replaced with ○ in REPLACE_MAP
        assert "○" in result
    
    def test_complex_text(self):
        """Test complex text with multiple transformations"""
        text = "こんにちは！　ＡＢＣ　１２３　？　テスト"
        result = normalize_text(text)
        assert " " not in result
        assert "　" not in result
        assert "ABC" in result
        assert "123" in result
        assert "!" in result
        assert "?" in result


class TestSpeechTokens:
    """Test speech token conversion functions"""
    
    def test_ids_to_speech_tokens(self):
        """Test conversion of IDs to speech tokens"""
        ids = [0, 1, 100, 65535]
        tokens = ids_to_speech_tokens(ids)
        assert tokens == ["<|s_0|>", "<|s_1|>", "<|s_100|>", "<|s_65535|>"]
    
    def test_ids_to_speech_tokens_empty(self):
        """Test empty ID list"""
        ids = []
        tokens = ids_to_speech_tokens(ids)
        assert tokens == []
    
    def test_extract_speech_ids(self):
        """Test extraction of speech IDs from token string"""
        token_str = "<|s_0|><|s_1|><|s_100|><|s_65535|>"
        ids = extract_speech_ids(token_str)
        assert ids == [0, 1, 100, 65535]
    
    def test_extract_speech_ids_empty(self):
        """Test extraction from empty string"""
        token_str = ""
        ids = extract_speech_ids(token_str)
        assert ids == []
    
    def test_extract_speech_ids_mixed(self):
        """Test extraction from string with mixed content"""
        token_str = "some text <|s_42|> more text <|s_123|> end"
        ids = extract_speech_ids(token_str)
        assert ids == [42, 123]
    
    def test_roundtrip_conversion(self):
        """Test roundtrip conversion of IDs"""
        original_ids = [10, 20, 30, 40, 50]
        tokens = ids_to_speech_tokens(original_ids)
        token_str = "".join(tokens)
        extracted_ids = extract_speech_ids(token_str)
        assert extracted_ids == original_ids


class TestGetPrompt:
    """Test prompt generation function"""
    
    def test_get_prompt_basic(self):
        """Test basic prompt generation"""
        text = "こんにちは"
        prompt = get_prompt(text)
        assert BOS_TOKEN in prompt
        assert "こんにちは" in prompt
        assert "<|SPEECH_GENERATION_START|>" in prompt
        assert "<|SPEECH_GENERATION_END|>" in prompt
    
    def test_get_prompt_no_bos(self):
        """Test prompt without BOS token"""
        text = "こんにちは"
        prompt = get_prompt(text, add_bos_token=False)
        assert BOS_TOKEN not in prompt
        assert "こんにちは" in prompt
    
    def test_get_prompt_no_end_token(self):
        """Test prompt without end token"""
        text = "こんにちは"
        prompt = get_prompt(text, add_end_token=False)
        assert "<|SPEECH_GENERATION_END|>" not in prompt
        assert "<|SPEECH_GENERATION_START|>" in prompt
    
    def test_get_prompt_with_code(self):
        """Test prompt with speech codes"""
        text = "こんにちは"
        code = [0, 1, 2, 3]
        prompt = get_prompt(text, code=code)
        assert "<|s_0|>" in prompt
        assert "<|s_1|>" in prompt
        assert "<|s_2|>" in prompt
        assert "<|s_3|>" in prompt
    
    def test_get_prompt_text_normalization(self):
        """Test that text is normalized in prompt"""
        text = "こんにちは　テスト"
        prompt = get_prompt(text)
        # Space should be removed by normalization
        assert "こんにちはテスト" in prompt
        assert "　" not in prompt
    
    def test_get_prompt_with_captions(self):
        """Test prompt with caption metadata"""
        text = "こんにちは"
        captions = {
            "emotion": "happy",
            "profile": "character1",
            "mood": "excited",
            "speed": "fast",
            "prosody": "emphatic",
            "pitch_timbre": "high",
            "style": "anime",
            "notes": "test notes",
            "caption": "test caption"
        }
        prompt = get_prompt(text, captions=captions)
        assert "emotion: happy" in prompt
        assert "profile: character1" in prompt
        assert "mood: excited" in prompt
        assert "speed: fast" in prompt
    
    def test_get_prompt_with_partial_captions(self):
        """Test prompt with partial caption metadata"""
        text = "こんにちは"
        captions = {"emotion": "sad"}
        prompt = get_prompt(text, captions=captions)
        assert "emotion: sad" in prompt
        # Default values should be used for missing fields
        assert "profile: default" in prompt
        assert "mood: normal" in prompt
    
    def test_get_prompt_default_system_prompt(self):
        """Test that default system prompt is included when no captions"""
        text = "こんにちは"
        prompt = get_prompt(text)
        assert DEFAULT_SYSTEM_PROMPT in prompt


class TestPreprocessAudio:
    """Test audio preprocessing function"""
    
    @pytest.mark.skip(reason="Requires torchaudio backend with audio file support")
    def test_preprocess_audio_correct_sample_rate(self):
        """Test preprocessing audio with correct sample rate"""
        # Create dummy audio at correct sample rate
        duration = 1.0  # 1 second
        sample_rate = SAMPLING_RATE
        waveform = torch.randn(1, int(sample_rate * duration))
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, waveform.numpy().T, sample_rate)
            tmp_path = Path(tmp.name)
            
            try:
                result = preprocess_audio(tmp_path)
                assert isinstance(result, torch.Tensor)
                assert result.dim() == 1  # Should be 1D
                assert len(result) == int(sample_rate * duration)
            finally:
                tmp_path.unlink()
    
    @pytest.mark.skip(reason="Requires torchaudio backend with audio file support")
    def test_preprocess_audio_resample(self):
        """Test preprocessing audio with different sample rate"""
        # Create dummy audio at different sample rate
        duration = 1.0
        original_rate = 8000
        waveform = torch.randn(1, int(original_rate * duration))
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, waveform.numpy().T, original_rate)
            tmp_path = Path(tmp.name)
            
            try:
                result = preprocess_audio(tmp_path)
                assert isinstance(result, torch.Tensor)
                assert result.dim() == 1
                # Should be resampled to SAMPLING_RATE
                assert len(result) == int(SAMPLING_RATE * duration)
            finally:
                tmp_path.unlink()
    
    @pytest.mark.skip(reason="Requires torchaudio backend with audio file support")
    def test_preprocess_audio_stereo_to_mono(self):
        """Test preprocessing stereo audio to mono"""
        # Create dummy stereo audio
        duration = 1.0
        sample_rate = SAMPLING_RATE
        waveform = torch.randn(2, int(sample_rate * duration))  # 2 channels
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, waveform.numpy().T, sample_rate)
            tmp_path = Path(tmp.name)
            
            try:
                result = preprocess_audio(tmp_path)
                assert isinstance(result, torch.Tensor)
                assert result.dim() == 1  # Should be mono (1D)
                assert len(result) == int(sample_rate * duration)
            finally:
                tmp_path.unlink()
    
    def test_preprocess_audio_dict_input(self):
        """Test preprocessing audio from dict format (HF datasets format)"""
        duration = 1.0
        sample_rate = SAMPLING_RATE
        waveform = torch.randn(int(sample_rate * duration))
        
        audio_dict = {
            "array": waveform.numpy(),
            "sampling_rate": sample_rate
        }
        
        result = preprocess_audio(audio_dict)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 1
        assert len(result) == int(sample_rate * duration)
    
    def test_preprocess_audio_dict_input_resample(self):
        """Test preprocessing audio from dict with different sample rate"""
        duration = 1.0
        original_rate = 8000
        waveform = torch.randn(int(original_rate * duration))
        
        audio_dict = {
            "array": waveform.numpy(),
            "sampling_rate": original_rate
        }
        
        result = preprocess_audio(audio_dict)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 1
        # Should be resampled to SAMPLING_RATE
        expected_length = int(SAMPLING_RATE * duration)
        # Allow small difference due to resampling
        assert abs(len(result) - expected_length) < 100
