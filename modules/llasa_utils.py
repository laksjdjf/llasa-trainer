import re
from transformers import LogitsProcessor
import torch

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
INVALID_PATTERN = re.compile(
    r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
    r"\u0041-\u005A\u0061-\u007A"
    r"\u0030-\u0039"
    r"。、!?…♪♡○]"
)

PROMPT_FORMAT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 02 Oct 2025

<|eot_id|><|start_header_id|>user<|end_header_id|>

Convert the text to speech:<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|><|eot_id|><|start_header_id|>assistant<|end_header_id|>

<|SPEECH_GENERATION_START|>
"""

OUTPUT_FORMAT = "{speech}<|SPEECH_GENERATION_END|>"

def normalize_text(text: str) -> str:
    for pattern, replacement in REPLACE_MAP.items():
        text = re.sub(pattern, replacement, text)

    text = text.translate(FULLWIDTH_ALPHA_TO_HALFWIDTH)
    text = text.translate(FULLWIDTH_DIGITS_TO_HALFWIDTH)
    text = text.translate(HALFWIDTH_KATAKANA_TO_FULLWIDTH)

    text = re.sub(r"…{3,}", "……", text)
    text = INVALID_PATTERN.sub("", text)

    return text

def ids_to_speech_tokens(speech_ids):
 
    speech_tokens_str = []
    for speech_id in speech_ids:
        speech_tokens_str.append(f"<|s_{speech_id}|>")
    return speech_tokens_str

def extract_speech_ids(speech_tokens_str):
 
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]

            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids

def get_prompt(text: str, code: list[int] | None = None) -> str:
    prompt = PROMPT_FORMAT.format(text=text)
    if code:
        speech_tokens = ids_to_speech_tokens(code)
        prompt += "".join(speech_tokens)
        prompt += "<|SPEECH_GENERATION_END|>"
    return prompt

class SpeechOnlyProcessor(LogitsProcessor):
    """音声トークンのみを許可するLogitsProcessor"""
    
    def __init__(self, tokenizer, device: str = "cuda:0", dtype=torch.float16):
        speech_start_id = tokenizer.convert_tokens_to_ids('<|s_0|>')
        speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
        speech_token_ids = list(range(speech_start_id, speech_start_id + 65536))
        allowed_tokens = torch.tensor(speech_token_ids + [speech_end_id], dtype=torch.long)
        mask = torch.full((193800,), float('-inf'))
        mask[allowed_tokens] = 0.0
        mask = mask.unsqueeze(0).to(device, dtype=dtype)
        self.mask = mask
    
    def __call__(self, input_ids, scores):
        return scores + self.mask