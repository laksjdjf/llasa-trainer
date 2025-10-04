import re
from typing import List, Optional, Tuple
from transformers import LogitsProcessor
import torch

# Default codec model
DEFAULT_CODEC_MODEL = "Anime-XCodec2-hf"

# Constants
SPEECH_TOKEN_PREFIX = "<|s_"
SPEECH_TOKEN_SUFFIX = "|>"
SPEECH_GENERATION_START = "<|SPEECH_GENERATION_START|>"
SPEECH_GENERATION_END = "<|SPEECH_GENERATION_END|>"
TEXT_UNDERSTANDING_START = "<|TEXT_UNDERSTANDING_START|>"
TEXT_UNDERSTANDING_END = "<|TEXT_UNDERSTANDING_END|>"

# Number of speech tokens in the vocabulary
NUM_SPEECH_TOKENS = 65536

# Total vocabulary size
TOTAL_VOCAB_SIZE = 193800

# Text replacement patterns
REPLACE_MAP: dict[str, str] = {
    r"\t": "",
    r"\[n\]": "",
    r" ": "",
    r"　": "",
    r"[;▼♀♂《》≪≫①②③④⑤⑥]": "",
    r"[\u02d7\u2010-\u2015\u2043\u2212\u23af\u23e4\u2500\u2501\u2e3a\u2e3b]": "",
    r"[\uff5e\u301C]": "ー",
    r"？": "?",
    r"！": "!",
    r"[●◯〇]": "○",
    r"♥": "♡",
}

# Character translation tables
FULLWIDTH_ALPHA_TO_HALFWIDTH = str.maketrans(
    {
        chr(full): chr(half)
        for full, half in zip(
            list(range(0xFF21, 0xFF3B)) + list(range(0xFF41, 0xFF5B)),
            list(range(0x41, 0x5B)) + list(range(0x61, 0x7B)),
        )
    }
)

HALFWIDTH_KATAKANA_TO_FULLWIDTH = str.maketrans(
    {
        chr(half): chr(full)
        for half, full in zip(range(0xFF61, 0xFF9F), range(0x30A1, 0x30FB))
    }
)

FULLWIDTH_DIGITS_TO_HALFWIDTH = str.maketrans(
    {
        chr(full): chr(half)
        for full, half in zip(range(0xFF10, 0xFF1A), range(0x30, 0x3A))
    }
)

# Pattern for invalid characters
INVALID_PATTERN = re.compile(
    r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
    r"\u0041-\u005A\u0061-\u007A"
    r"\u0030-\u0039"
    r"。、!?…♪♡○]"
)

# Prompt template for text-to-speech generation
PROMPT_FORMAT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 02 Oct 2025

<|eot_id|><|start_header_id|>user<|end_header_id|>

Convert the text to speech:{TEXT_UNDERSTANDING_START}{text}{TEXT_UNDERSTANDING_END}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{SPEECH_GENERATION_START}
"""

OUTPUT_FORMAT = "{{speech}}{SPEECH_GENERATION_END}"


def normalize_text(text: str) -> str:
    """Normalize Japanese text for TTS processing.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text string
    """
    for pattern, replacement in REPLACE_MAP.items():
        text = re.sub(pattern, replacement, text)

    text = text.translate(FULLWIDTH_ALPHA_TO_HALFWIDTH)
    text = text.translate(FULLWIDTH_DIGITS_TO_HALFWIDTH)
    text = text.translate(HALFWIDTH_KATAKANA_TO_FULLWIDTH)

    text = re.sub(r"…{3,}", "……", text)
    text = INVALID_PATTERN.sub("", text)

    return text


def ids_to_speech_tokens(speech_ids: List[int]) -> List[str]:
    """Convert speech IDs to speech token strings.
    
    Args:
        speech_ids: List of speech ID integers
        
    Returns:
        List of formatted speech token strings
    """
    return [f"{SPEECH_TOKEN_PREFIX}{speech_id}{SPEECH_TOKEN_SUFFIX}" for speech_id in speech_ids]


def extract_speech_ids(speech_tokens_str: List[str]) -> List[int]:
    """Extract speech IDs from speech token strings.
    
    Args:
        speech_tokens_str: List of speech token strings
        
    Returns:
        List of extracted speech ID integers
    """
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith(SPEECH_TOKEN_PREFIX) and token_str.endswith(SPEECH_TOKEN_SUFFIX):
            num_str = token_str[len(SPEECH_TOKEN_PREFIX):-len(SPEECH_TOKEN_SUFFIX)]
            try:
                num = int(num_str)
                speech_ids.append(num)
            except ValueError:
                print(f"Unexpected token format: {token_str}")
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids


def get_prompt(text: str, code: Optional[List[int]] = None) -> str:
    """Generate a prompt for text-to-speech generation.
    
    Args:
        text: Input text to convert to speech
        code: Optional list of speech token codes for training
        
    Returns:
        Formatted prompt string
    """
    prompt = PROMPT_FORMAT.format(
        text=text,
        TEXT_UNDERSTANDING_START=TEXT_UNDERSTANDING_START,
        TEXT_UNDERSTANDING_END=TEXT_UNDERSTANDING_END,
        SPEECH_GENERATION_START=SPEECH_GENERATION_START
    )
    if code:
        speech_tokens = ids_to_speech_tokens(code)
        prompt += "".join(speech_tokens)
        prompt += SPEECH_GENERATION_END
    return prompt


class SpeechOnlyProcessor(LogitsProcessor):
    """LogitsProcessor that only allows speech tokens and end token.
    
    This processor masks out all non-speech tokens to ensure the model
    only generates valid speech tokens or the end-of-speech token.
    """
    
    def __init__(self, tokenizer, device: str = "cuda:0", dtype=torch.float16):
        """Initialize the speech token processor.
        
        Args:
            tokenizer: HuggingFace tokenizer instance
            device: Device to place the mask tensor on
            dtype: Data type for the mask tensor
        """
        # Get token IDs for speech tokens
        speech_start_id = tokenizer.convert_tokens_to_ids(f'{SPEECH_TOKEN_PREFIX}0{SPEECH_TOKEN_SUFFIX}')
        speech_end_id = tokenizer.convert_tokens_to_ids(SPEECH_GENERATION_END)
        
        # Create list of allowed token IDs
        speech_token_ids = list(range(speech_start_id, speech_start_id + NUM_SPEECH_TOKENS))
        allowed_tokens = torch.tensor(speech_token_ids + [speech_end_id], dtype=torch.long)
        
        # Create mask that blocks all tokens except allowed ones
        mask = torch.full((TOTAL_VOCAB_SIZE,), float('-inf'))
        mask[allowed_tokens] = 0.0
        self.mask = mask.unsqueeze(0).to(device, dtype=dtype)
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Apply the speech-only mask to logits.
        
        Args:
            input_ids: Current input token IDs
            scores: Current logit scores
            
        Returns:
            Modified scores with non-speech tokens masked
        """
        return scores + self.mask


def load_codec_model(model_name: str = DEFAULT_CODEC_MODEL, device: Optional[str] = None) -> Tuple:
    """Load XCodec2 model and feature extractor.
    
    Args:
        model_name: Name or path of the codec model
        device: Device to load model on. If None, uses cuda if available, else cpu
        
    Returns:
        Tuple of (codec_model, feature_extractor)
    """
    from transformers import Xcodec2Model, Xcodec2FeatureExtractor
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🎵 Loading XCodec2 model from {model_name}...")
    codec_model = Xcodec2Model.from_pretrained(model_name).eval().to(device)
    feature_extractor = Xcodec2FeatureExtractor.from_pretrained(model_name)
    print(f"✅ XCodec2 model loaded on {device}")
    
    return codec_model, feature_extractor