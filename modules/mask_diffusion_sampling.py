"""
Mask Diffusion Sampling Module for LLASA

This module implements iterative sampling to restore masked audio tokens.
The approach is similar to BERT's masked language modeling but applied iteratively
to gradually refine the generated audio.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List
import numpy as np


class MaskDiffusionSampler:
    """Sampler for mask diffusion approach"""
    
    def __init__(
        self,
        model,
        tokenizer,
        speech_start_id: int = 128264,
        speech_end_id: int = 128261,
        mask_token_id: Optional[int] = None,
    ):
        """
        Initialize mask diffusion sampler
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            speech_start_id: Start ID for speech tokens
            speech_end_id: End ID for speech tokens
            mask_token_id: Token ID used for masking
        """
        self.model = model
        self.tokenizer = tokenizer
        self.speech_start_id = speech_start_id
        self.speech_end_id = speech_end_id
        
        # Get mask token
        if mask_token_id is None:
            if "[MASK]" in tokenizer.vocab:
                self.mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")
            else:
                raise ValueError("Mask token not found in tokenizer vocabulary")
        else:
            self.mask_token_id = mask_token_id
            
        print(f"🎭 Mask Diffusion Sampler initialized")
        print(f"🎭 Mask token ID: {self.mask_token_id}")
    
    def is_speech_token(self, token_id: int) -> bool:
        """Check if a token is a speech token"""
        return self.speech_start_id <= token_id < self.speech_start_id + 65536
    
    @torch.no_grad()
    def iterative_decode(
        self,
        input_ids: torch.Tensor,
        num_iterations: int = 10,
        temperature: float = 1.0,
        top_p: float = 0.9,
        confidence_threshold: float = 0.9,
    ) -> torch.Tensor:
        """
        Iteratively decode masked tokens
        
        Args:
            input_ids: Input token IDs with some positions masked
            num_iterations: Number of iterative refinement steps
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            confidence_threshold: Confidence threshold for early stopping
            
        Returns:
            Decoded token IDs with masks filled
        """
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        # Find masked positions
        mask_positions = (input_ids == self.mask_token_id)
        
        if not mask_positions.any():
            # No masks to fill
            return input_ids
        
        # Track which positions are still masked
        remaining_masks = mask_positions.clone()
        
        for iteration in range(num_iterations):
            if not remaining_masks.any():
                # All masks have been filled
                break
            
            # Forward pass
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits
            
            # Apply temperature
            logits = logits / temperature
            
            # Get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # For each masked position, get prediction
            batch_size, seq_len = input_ids.shape
            
            for i in range(batch_size):
                masked_positions = remaining_masks[i].nonzero(as_tuple=False).squeeze(-1)
                
                if len(masked_positions) == 0:
                    continue
                
                # Get confidence scores for masked positions
                confidences = []
                predictions = []
                
                for pos in masked_positions:
                    pos_probs = probs[i, pos]
                    
                    # Apply top-p filtering
                    sorted_probs, sorted_indices = torch.sort(pos_probs, descending=True)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=0)
                    
                    # Find cutoff index for top-p
                    cutoff_idx = (cumsum_probs > top_p).nonzero(as_tuple=False)
                    if len(cutoff_idx) > 0:
                        cutoff_idx = cutoff_idx[0].item()
                    else:
                        cutoff_idx = len(sorted_probs) - 1
                    
                    # Keep only top-p tokens
                    top_probs = sorted_probs[:cutoff_idx + 1]
                    top_indices = sorted_indices[:cutoff_idx + 1]
                    
                    # Renormalize
                    top_probs = top_probs / top_probs.sum()
                    
                    # Sample from top-p distribution
                    sampled_idx = torch.multinomial(top_probs, 1).item()
                    predicted_token = top_indices[sampled_idx].item()
                    confidence = top_probs[sampled_idx].item()
                    
                    # Only consider speech tokens
                    if self.is_speech_token(predicted_token):
                        confidences.append((confidence, pos.item(), predicted_token))
                        predictions.append((pos.item(), predicted_token, confidence))
                
                # Sort by confidence (highest first)
                confidences.sort(reverse=True, key=lambda x: x[0])
                
                # Fill in high-confidence predictions
                num_to_fill = max(1, len(confidences) // (num_iterations - iteration))
                
                for conf, pos, token in confidences[:num_to_fill]:
                    if conf >= confidence_threshold or iteration == num_iterations - 1:
                        input_ids[i, pos] = token
                        remaining_masks[i, pos] = False
            
            # Print progress
            num_remaining = remaining_masks.sum().item()
            print(f"  Iteration {iteration + 1}/{num_iterations}: {num_remaining} masks remaining")
        
        return input_ids
    
    @torch.no_grad()
    def generate_with_mask_diffusion(
        self,
        prompt_text: str,
        num_audio_tokens: int = 300,
        num_iterations: int = 10,
        temperature: float = 0.7,
        top_p: float = 0.9,
        confidence_threshold: float = 0.9,
    ) -> List[int]:
        """
        Generate audio tokens using mask diffusion
        
        Args:
            prompt_text: Text prompt (should include proper formatting)
            num_audio_tokens: Number of audio tokens to generate
            num_iterations: Number of iterative refinement steps
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            confidence_threshold: Confidence threshold for early stopping
            
        Returns:
            List of generated audio token IDs
        """
        device = next(self.model.parameters()).device
        
        # Tokenize prompt
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        
        # Create initial sequence with all audio tokens masked
        masked_audio_tokens = torch.full(
            (1, num_audio_tokens),
            self.mask_token_id,
            dtype=torch.long,
            device=device
        )
        
        # Concatenate prompt and masked audio tokens
        input_ids = torch.cat([prompt_ids, masked_audio_tokens], dim=1)
        
        print(f"🎭 Starting mask diffusion generation...")
        print(f"   Prompt length: {prompt_ids.shape[1]} tokens")
        print(f"   Audio tokens to generate: {num_audio_tokens}")
        
        # Iteratively decode
        decoded_ids = self.iterative_decode(
            input_ids,
            num_iterations=num_iterations,
            temperature=temperature,
            top_p=top_p,
            confidence_threshold=confidence_threshold,
        )
        
        # Extract audio token IDs
        audio_token_ids = decoded_ids[0, prompt_ids.shape[1]:].cpu().tolist()
        
        # Convert to speech IDs (subtract speech_start_id offset)
        speech_ids = []
        for token_id in audio_token_ids:
            if self.is_speech_token(token_id):
                speech_id = token_id - self.speech_start_id
                speech_ids.append(speech_id)
        
        print(f"✅ Generated {len(speech_ids)} speech tokens")
        
        return speech_ids
    
    @torch.no_grad()
    def inpaint_audio(
        self,
        input_ids: torch.Tensor,
        mask_positions: torch.Tensor,
        num_iterations: int = 10,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Inpaint (restore) specific masked positions in audio
        
        Args:
            input_ids: Full sequence with some positions to inpaint
            mask_positions: Boolean tensor indicating which positions to inpaint
            num_iterations: Number of refinement iterations
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Inpainted sequence
        """
        # Apply mask token to specified positions
        input_ids = input_ids.clone()
        input_ids[mask_positions] = self.mask_token_id
        
        # Run iterative decoding
        decoded_ids = self.iterative_decode(
            input_ids,
            num_iterations=num_iterations,
            temperature=temperature,
            top_p=top_p,
        )
        
        return decoded_ids
