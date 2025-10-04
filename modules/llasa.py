import torch
import tempfile
import soundfile as sf
from typing import Optional, Tuple, List
from transformers import AutoTokenizer, AutoModelForCausalLM, Xcodec2Model, Xcodec2FeatureExtractor
from peft import AutoPeftModelForCausalLM
from modules.llasa_utils import (
    get_prompt, 
    SpeechOnlyProcessor, 
    load_codec_model,
    SPEECH_GENERATION_END,
    SPEECH_TOKEN_PREFIX,
    SPEECH_TOKEN_SUFFIX,
    NUM_SPEECH_TOKENS,
    DEFAULT_CODEC_MODEL
)


class LLASA:
    """LLASA Text-to-Speech model wrapper.
    
    This class provides an interface for loading and using LLASA models
    for text-to-speech generation.
    """
    
    def __init__(
        self, 
        model=None, 
        tokenizer=None, 
        codec_model: Optional[Xcodec2Model] = None, 
        feature_extractor: Optional[Xcodec2FeatureExtractor] = None
    ):
        """Initialize LLASA TTS model.
        
        Args:
            model: Pre-loaded language model
            tokenizer: Pre-loaded tokenizer
            codec_model: Pre-loaded XCodec2 model
            feature_extractor: Pre-loaded feature extractor
        """
        
        self.model = model
        self.tokenizer = tokenizer
        self.codec_model = codec_model
        self.feature_extractor = feature_extractor
        
        # Initialize speech token processor
        self.logits_processor = SpeechOnlyProcessor(
            tokenizer=self.tokenizer, 
            device=model.device, 
            dtype=next(model.parameters()).dtype
        )
        
        # Cache token IDs for speech processing
        self.speech_start_id = self.tokenizer.convert_tokens_to_ids(
            f'{SPEECH_TOKEN_PREFIX}0{SPEECH_TOKEN_SUFFIX}'
        )
        self.speech_end_id = self.tokenizer.convert_tokens_to_ids(SPEECH_GENERATION_END)
        
        print("✅ LLASA initialized successfully!")
    
    @classmethod
    def from_pretrained(
        cls, 
        lora_path: str = "./lora_checkpoints",
        codec_model_name: str = DEFAULT_CODEC_MODEL
    ):
        """Load LLASA model from a folder path.
        
        Args:
            lora_path: Path to the LoRA checkpoint or base model
            codec_model_name: Name or path of the codec model
            
        Returns:
            Initialized LLASA instance
        """
        
        # Load model
        print("📦 Loading model...")
        try:
            # Try loading as LoRA model first
            model = AutoPeftModelForCausalLM.from_pretrained(
                lora_path,
                dtype=torch.float16,
            ).eval().to('cuda:0')
            print("✅ LoRA model loaded")
        except Exception as e:
            # Fall back to loading as regular model
            print(f"⚠️ Failed to load as LoRA model, trying as regular model...")
            model = AutoModelForCausalLM.from_pretrained(
                lora_path,
                dtype=torch.float16,
            ).eval().to('cuda:0')
            print("✅ Regular model loaded")
        
        # Load tokenizer
        print("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(lora_path)
        print("✅ Tokenizer loaded")
        
        # Load codec model using shared utility
        codec_model, feature_extractor = load_codec_model(codec_model_name, device='cuda:0')
        
        return cls(
            model=model, 
            tokenizer=tokenizer, 
            codec_model=codec_model, 
            feature_extractor=feature_extractor
        )
    
    def _extract_speech_ids(self, generated_ids: torch.Tensor) -> List[int]:
        """Extract speech IDs from generated token IDs.
        
        Args:
            generated_ids: Tensor of generated token IDs
            
        Returns:
            List of speech ID integers
        """
        speech_ids = []
        
        for token_id in generated_ids:
            token_id_val = token_id.item()
            
            # Stop at end token
            if token_id_val == self.speech_end_id:
                break
                
            # Check if token is within speech token range
            if self.speech_start_id <= token_id_val < self.speech_start_id + NUM_SPEECH_TOKENS:
                speech_id = token_id_val - self.speech_start_id
                speech_ids.append(speech_id)
        
        return speech_ids
    
    @torch.no_grad()
    def generate(
        self,
        text: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        max_tokens: int = 300,
    ) -> Tuple[Optional[str], str, str]:
        """Generate speech from text.
        
        Args:
            text: Input text to convert to speech
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            repeat_penalty: Penalty for repeating tokens
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Tuple of (audio_path, status_msg, token_info)
            - audio_path: Path to generated audio file (None if failed)
            - status_msg: Status message describing the result
            - token_info: String representation of generated tokens
        """
        
        # Create prompt
        prompt = get_prompt(text)
        
        # Tokenize
        input_ids = self.tokenizer(prompt, return_tensors='pt').to('cuda:0')

        # Generate speech tokens
        outputs = self.model.generate(
            **input_ids,
            max_new_tokens=max_tokens,
            eos_token_id=self.speech_end_id,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repeat_penalty,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            logits_processor=[self.logits_processor],
        )
        
        # Extract speech IDs from generated tokens
        generated_ids = outputs[:, input_ids.input_ids.shape[1]:][0]
        speech_ids = self._extract_speech_ids(generated_ids)
        
        if not speech_ids:
            return None, "❌ No valid speech tokens were generated", ""
        
        # Generate audio waveform from speech codes
        speech_codes = torch.tensor(speech_ids, dtype=torch.long).to('cuda:0').unsqueeze(0).unsqueeze(0)
        gen_wav = self.codec_model.decode(speech_codes).audio_values
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            sf.write(tmp_file.name, gen_wav[0, 0, :].cpu().numpy(), 16000)
            audio_path = tmp_file.name
        
        status_msg = f"✅ Generation complete ({len(generated_ids)} tokens)"
        token_info = str(generated_ids.cpu().numpy().tolist())
        
        return audio_path, status_msg, token_info