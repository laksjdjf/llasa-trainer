"""
LLASA with Mask Diffusion Support

This module extends the LLASA class to support mask diffusion generation.
"""

import torch
from modules.llasa import LLASA, BaseAudioDecoder
from modules.mask_diffusion_sampling import MaskDiffusionSampler
from modules.llasa_utils import get_prompt
from typing import Optional


class LLASAMaskDiffusion(BaseAudioDecoder):
    """LLASA with mask diffusion generation support"""
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "./lora_checkpoints",
        codec_model_path: str = "Anime-XCodec2-hf",
        dtype=torch.float16,
    ):
        """Load LLASA model trained with mask diffusion"""
        
        # Use LLASA's from_pretrained to load the model
        llasa = LLASA.from_pretrained(model_path, codec_model_path, dtype)
        
        # Create instance
        instance = cls(
            model=llasa.model,
            tokenizer=llasa.tokenizer,
            codec_model=llasa.codec_model,
            feature_extractor=llasa.feature_extractor,
        )
        
        # Initialize mask diffusion sampler
        instance.mask_sampler = MaskDiffusionSampler(
            model=instance.model,
            tokenizer=instance.tokenizer,
        )
        
        print("✅ LLASA Mask Diffusion loaded!")
        
        return instance
    
    @torch.no_grad()
    def generate_tokens_mask_diffusion(
        self,
        prompt: str,
        num_audio_tokens: int = 300,
        num_iterations: int = 10,
        temperature: float = 0.7,
        top_p: float = 0.9,
        confidence_threshold: float = 0.9,
    ) -> list[int]:
        """
        Generate audio tokens using mask diffusion
        
        Args:
            prompt: Text prompt
            num_audio_tokens: Number of audio tokens to generate
            num_iterations: Number of iterative refinement steps
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            confidence_threshold: Confidence threshold for early stopping
            
        Returns:
            List of generated speech token IDs
        """
        speech_ids = self.mask_sampler.generate_with_mask_diffusion(
            prompt_text=prompt,
            num_audio_tokens=num_audio_tokens,
            num_iterations=num_iterations,
            temperature=temperature,
            top_p=top_p,
            confidence_threshold=confidence_threshold,
        )
        
        return speech_ids
    
    @torch.no_grad()
    def generate_with_mask_diffusion(
        self,
        text: str,
        num_audio_tokens: int = 300,
        num_iterations: int = 10,
        temperature: float = 0.7,
        top_p: float = 0.9,
        confidence_threshold: float = 0.9,
        reference_text: str = "",
        reference_audio: Optional[str] = None,
        reference_codes: Optional[list[int]] = None,
        decode_audio: bool = True,
        captions: Optional[dict] = None,
    ) -> tuple[str, list[int]]:
        """
        Generate audio using mask diffusion with reference support
        
        Args:
            text: Text to generate audio for
            num_audio_tokens: Number of audio tokens to generate
            num_iterations: Number of iterative refinement steps
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            confidence_threshold: Confidence threshold
            reference_text: Reference text for context
            reference_audio: Reference audio path
            reference_codes: Pre-encoded reference codes
            decode_audio: Whether to decode audio
            captions: Optional captions dictionary
            
        Returns:
            Tuple of (audio_path, speech_ids)
        """
        # Prepare text with reference
        text = reference_text + text if reference_text else text
        
        # Encode reference audio if provided
        reference_codes = reference_codes or (
            self.encode_audio(reference_audio) if reference_audio else None
        )
        
        # Get prompt
        prompt = get_prompt(
            text,
            reference_codes,
            add_bos_token=False,
            add_end_token=False,
            captions=captions
        )
        
        # Generate using mask diffusion
        speech_ids = self.generate_tokens_mask_diffusion(
            prompt=prompt,
            num_audio_tokens=num_audio_tokens,
            num_iterations=num_iterations,
            temperature=temperature,
            top_p=top_p,
            confidence_threshold=confidence_threshold,
        )
        
        if not speech_ids or not decode_audio:
            return None, speech_ids
        
        # Decode to audio
        audio_path = self.decode_tokens(speech_ids)
        
        return audio_path, speech_ids
    
    @torch.no_grad()
    def inpaint_audio(
        self,
        original_codes: list[int],
        mask_start: int,
        mask_end: int,
        num_iterations: int = 10,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> list[int]:
        """
        Inpaint (restore) a specific region of audio tokens
        
        Args:
            original_codes: Original audio token sequence
            mask_start: Start index of region to inpaint
            mask_end: End index of region to inpaint
            num_iterations: Number of refinement iterations
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Inpainted audio token sequence
        """
        # Convert codes to token IDs
        token_ids = [self.speech_start_id + code for code in original_codes]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        
        # Create mask positions
        mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)
        mask_positions[0, mask_start:mask_end] = True
        
        # Inpaint using mask diffusion
        decoded_ids = self.mask_sampler.inpaint_audio(
            input_ids=input_ids,
            mask_positions=mask_positions,
            num_iterations=num_iterations,
            temperature=temperature,
            top_p=top_p,
        )
        
        # Convert back to speech IDs
        speech_ids = []
        for token_id in decoded_ids[0].cpu().tolist():
            if self.speech_start_id <= token_id < self.speech_start_id + 65536:
                speech_ids.append(token_id - self.speech_start_id)
        
        return speech_ids
