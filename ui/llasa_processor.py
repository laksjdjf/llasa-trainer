llasa = None

def load(model_path, codec_model_path):
    global llasa
    if model_path == "server":
        from modules.llasa_server import LLASAServer
        llasa = LLASAServer.from_pretrained(model_path, codec_model_path)
    else:
        from modules.llasa import LLASA
        llasa = LLASA.from_pretrained(model_path, codec_model_path)

def generate(
    text: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repeat_penalty: float = 1.1,
    max_tokens: int = 300,
    reference_text: str = "",
    reference_audio: list[int] = None,
) -> tuple[str, str]:
    """テキストから音声を生成"""
    global llasa

    return llasa.generate(
        text,
        temperature,
        top_p,
        repeat_penalty,
        max_tokens,
        reference_text,
        reference_audio,
    )

def transcribe(audio_path: str) -> str:
    """音声ファイルをテキストに文字起こし"""
    global llasa
    return llasa.transcribe(audio_path)

def encode_audio(audio_path: str) -> list[int]:
    """音声ファイルを音声トークンにエンコード"""
    global llasa
    return llasa.encode_audio(audio_path)

def decode_tokens(speech_ids: list[int]) -> str:
    """音声トークンを音声ファイルにデコード"""
    global llasa
    return llasa.decode_tokens(speech_ids)

def calc_similarity(target_audio: str, reference_audios: list[str]) -> float:
    """話者認識モデルで音声の類似度を計算"""
    global llasa
    return llasa.calc_similarity(target_audio, reference_audios)