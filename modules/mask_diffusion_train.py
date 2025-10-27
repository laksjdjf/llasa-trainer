"""
Mask Diffusion Training Module for LLASA

This module implements mask prediction training for audio tokens.
Unlike causal LM training, this uses a BERT-like masking strategy
where random audio tokens are masked and the model learns to predict them.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import random


class MaskDiffusionTrainer:
    """Trainer for mask diffusion approach"""
    
    def __init__(
        self,
        model,
        tokenizer,
        mask_ratio: float = 0.15,
        speech_start_id: int = 128264,
        speech_end_id: int = 128261,
        mask_token_id: Optional[int] = None,
    ):
        """
        Initialize mask diffusion trainer
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            mask_ratio: Ratio of tokens to mask (default: 0.15)
            speech_start_id: Start ID for speech tokens
            speech_end_id: End ID for speech tokens
            mask_token_id: Token ID to use for masking (if None, uses [MASK] token)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.mask_ratio = mask_ratio
        self.speech_start_id = speech_start_id
        self.speech_end_id = speech_end_id
        
        # Get or add mask token
        if mask_token_id is None:
            if "[MASK]" not in tokenizer.vocab:
                # Add [MASK] token if it doesn't exist
                self.tokenizer.add_special_tokens({"additional_special_tokens": ["[MASK]"]})
                self.model.resize_token_embeddings(len(self.tokenizer))
            self.mask_token_id = self.tokenizer.convert_tokens_to_ids("[MASK]")
        else:
            self.mask_token_id = mask_token_id
            
        print(f"🎭 Mask token ID: {self.mask_token_id}")
        print(f"🎭 Speech token range: {self.speech_start_id} - {self.speech_start_id + 65536}")
        print(f"🎭 Mask ratio: {self.mask_ratio}")
    
    def is_speech_token(self, token_id: int) -> bool:
        """Check if a token is a speech token"""
        return self.speech_start_id <= token_id < self.speech_start_id + 65536
    
    def mask_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mask audio tokens for training
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            masked_input_ids: Input IDs with some tokens masked
            labels: Original token IDs (-100 for tokens that should not be predicted)
            mask_positions: Boolean mask indicating which positions were masked
        """
        batch_size, seq_len = input_ids.shape
        
        # Initialize labels with -100 (ignore index)
        labels = torch.full_like(input_ids, -100)
        
        # Initialize mask positions
        mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)
        
        # Clone input_ids for masking
        masked_input_ids = input_ids.clone()
        
        # Process each sequence in the batch
        for i in range(batch_size):
            # Find speech token positions
            speech_positions = []
            for j in range(seq_len):
                if self.is_speech_token(input_ids[i, j].item()):
                    speech_positions.append(j)
            
            if not speech_positions:
                continue
            
            # Randomly select positions to mask
            num_to_mask = max(1, int(len(speech_positions) * self.mask_ratio))
            positions_to_mask = random.sample(speech_positions, num_to_mask)
            
            # Apply masking
            for pos in positions_to_mask:
                # Save original token for prediction
                labels[i, pos] = input_ids[i, pos]
                # Mask the token
                masked_input_ids[i, pos] = self.mask_token_id
                mask_positions[i, pos] = True
        
        return masked_input_ids, labels, mask_positions
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute mask prediction loss
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Optional pre-computed labels (if None, will mask tokens automatically)
            
        Returns:
            loss: The computed loss
            metrics: Dictionary of metrics for logging
        """
        # If labels not provided, create masked inputs
        if labels is None:
            masked_input_ids, labels, mask_positions = self.mask_tokens(input_ids, attention_mask)
        else:
            masked_input_ids = input_ids
            mask_positions = labels != -100
        
        # Forward pass
        outputs = self.model(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        loss = outputs.loss
        
        # Compute metrics
        with torch.no_grad():
            # Count masked tokens
            num_masked = (labels != -100).sum().item()
            
            # Compute accuracy for masked tokens
            if num_masked > 0:
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                correct = (predictions == labels) & (labels != -100)
                accuracy = correct.sum().item() / num_masked
            else:
                accuracy = 0.0
        
        metrics = {
            "loss": loss.item(),
            "num_masked_tokens": num_masked,
            "mask_accuracy": accuracy,
        }
        
        return loss, metrics


def create_mask_diffusion_collator(tokenizer, mask_ratio: float = 0.15):
    """
    Create a data collator for mask diffusion training
    
    Args:
        tokenizer: The tokenizer
        mask_ratio: Ratio of tokens to mask
        
    Returns:
        A collator function
    """
    speech_start_id = 128264
    
    def is_speech_token(token_id: int) -> bool:
        return speech_start_id <= token_id < speech_start_id + 65536
    
    # Get or add mask token
    if "[MASK]" not in tokenizer.vocab:
        tokenizer.add_special_tokens({"additional_special_tokens": ["[MASK]"]})
    mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")
    
    def collate_fn(examples):
        """Collate function that applies masking"""
        # Tokenize examples
        batch = tokenizer.pad(
            [{"input_ids": ex["input_ids"]} for ex in examples],
            padding=True,
            return_tensors="pt",
        )
        
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", torch.ones_like(input_ids))
        
        batch_size, seq_len = input_ids.shape
        labels = torch.full_like(input_ids, -100)
        
        # Mask tokens
        for i in range(batch_size):
            speech_positions = []
            for j in range(seq_len):
                if is_speech_token(input_ids[i, j].item()):
                    speech_positions.append(j)
            
            if speech_positions:
                num_to_mask = max(1, int(len(speech_positions) * mask_ratio))
                positions_to_mask = random.sample(speech_positions, num_to_mask)
                
                for pos in positions_to_mask:
                    labels[i, pos] = input_ids[i, pos]
                    input_ids[i, pos] = mask_token_id
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    return collate_fn
