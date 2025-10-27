# Mask Diffusion Implementation Summary

## Overview

This document provides a comprehensive summary of the mask diffusion implementation for LLASA trainer.

## What is Mask Diffusion?

Mask diffusion is an alternative training and inference approach for the LLASA TTS model that differs from the traditional autoregressive (causal language modeling) approach:

### Traditional Approach (Causal LM)
- **Training**: Model learns to predict the next token given previous tokens
- **Inference**: Generates tokens sequentially from left to right
- **Limitation**: Cannot easily edit or inpaint specific parts of the generated audio

### Mask Diffusion Approach
- **Training**: Random audio tokens are masked, and the model learns to predict masked tokens (similar to BERT)
- **Inference**: All audio tokens start as masked, and are iteratively refined over multiple steps
- **Advantage**: Allows for audio inpainting and editing of specific regions

## Implementation Components

### 1. Training Module (`modules/mask_diffusion_train.py`)

**Key Classes:**
- `MaskDiffusionTrainer`: Main trainer class for mask diffusion
  - Handles masking of audio tokens (only speech tokens, not text tokens)
  - Computes mask prediction loss
  - Tracks metrics (mask accuracy, number of masked tokens)

**Key Functions:**
- `mask_tokens()`: Randomly masks audio tokens at the specified ratio
- `compute_loss()`: Computes the mask prediction loss
- `create_mask_diffusion_collator()`: Creates a data collator that applies masking during training

**Key Parameters:**
- `mask_ratio`: Proportion of audio tokens to mask (default: 0.15, recommended: 0.10-0.30)
- `speech_start_id`: Start ID for speech tokens (128264)
- `speech_end_id`: Token ID for the speech end marker `<|SPEECH_GENERATION_END|>` (128261)

### 2. Sampling Module (`modules/mask_diffusion_sampling.py`)

**Key Classes:**
- `MaskDiffusionSampler`: Sampler for iterative mask diffusion inference
  - Iteratively refines masked tokens over multiple steps
  - Supports top-p (nucleus) sampling
  - Confidence-based early stopping

**Key Methods:**
- `iterative_decode()`: Main iterative refinement loop
- `generate_with_mask_diffusion()`: Generate audio from text using mask diffusion
- `inpaint_audio()`: Restore specific masked regions in audio

**Key Parameters:**
- `num_iterations`: Number of refinement iterations (default: 10)
- `temperature`: Sampling temperature (default: 0.7)
- `top_p`: Top-p sampling parameter (default: 0.9)
- `confidence_threshold`: Confidence threshold for early stopping (default: 0.9)

### 3. LLASA Mask Diffusion Wrapper (`modules/llasa_mask_diffusion.py`)

**Key Classes:**
- `LLASAMaskDiffusion`: Extends BaseAudioDecoder to support mask diffusion
  - Integrates mask diffusion sampling with LLASA
  - Supports reference audio and captions
  - Provides audio inpainting functionality

**Key Methods:**
- `generate_with_mask_diffusion()`: Generate audio with reference support
- `inpaint_audio()`: Inpaint specific regions of audio tokens

### 4. Training Script (`main_mask_diffusion.py`)

Main entry point for mask diffusion training:
- Loads LLASA model with optional LoRA
- Initializes MaskDiffusionTrainer
- Creates custom trainer that uses mask prediction loss
- Handles dataset tokenization and training

**Usage:**
```bash
python main_mask_diffusion.py --config config/mask_diffusion_example.yaml
```

### 5. Demo Script (`demo_mask_diffusion.py`)

Demonstrates mask diffusion functionality:
- **Generation Demo**: Shows how to generate audio using mask diffusion
- **Inpainting Demo**: Shows how to inpaint/restore masked audio regions
- **Collator Demo**: Shows how the masking collator works during training

**Usage:**
```bash
# Run all demos
python demo_mask_diffusion.py --mode all

# Run specific demo
python demo_mask_diffusion.py --mode generation
python demo_mask_diffusion.py --mode inpainting
python demo_mask_diffusion.py --mode collator
```

### 6. Configuration (`config/mask_diffusion_example.yaml`)

Example configuration for mask diffusion training:
```yaml
mask_diffusion:
  mask_ratio: 0.15  # Proportion of audio tokens to mask
```

## Key Design Decisions

### 1. Masking Strategy
- **Only audio tokens are masked** (not text tokens)
- Audio tokens are identified by their ID range: `[128264, 128264 + 65536)`
- Mask token `[MASK]` is added to the tokenizer if it doesn't exist

### 2. Loss Computation
- Only masked positions contribute to the loss
- Uses standard cross-entropy loss on masked tokens
- Tracks mask accuracy as a metric

### 3. Iterative Sampling
- Starts with all audio tokens masked
- Each iteration:
  1. Predicts all masked positions
  2. Selects high-confidence predictions to fill
  3. Leaves low-confidence predictions masked for next iteration
- Progressively refines the audio over multiple iterations

### 4. Integration with Existing Code
- Extends existing LLASA classes
- Reuses existing utilities (get_prompt, preprocess_audio, etc.)
- Compatible with LoRA training
- Maintains compatibility with XCodec2 encoder/decoder

## Usage Examples

### Training with Mask Diffusion

1. Prepare your dataset (same as regular training):
```bash
python create_dataset.py ./audio ./text.txt -o dataset/data.jsonl
```

2. Create a configuration file:
```bash
cp config/mask_diffusion_example.yaml config/my_config.yaml
# Edit my_config.yaml as needed
```

3. Start training:
```bash
python main_mask_diffusion.py --config config/my_config.yaml
```

### Inference with Mask Diffusion

```python
from modules.llasa_mask_diffusion import LLASAMaskDiffusion

# Load trained model
llasa_md = LLASAMaskDiffusion.from_pretrained(
    model_path="./trained/MaskDiffusionModel",
    codec_model_path="Anime-XCodec2-hf",
)

# Generate audio
audio_path, speech_ids = llasa_md.generate_with_mask_diffusion(
    text="こんにちは、今日はいい天気ですね。",
    num_audio_tokens=300,
    num_iterations=10,
    temperature=0.7,
    top_p=0.9,
)
```

### Audio Inpainting

```python
# Inpaint a specific region
inpainted_codes = llasa_md.inpaint_audio(
    original_codes=[...],  # Original audio tokens
    mask_start=100,        # Start of region to inpaint
    mask_end=200,          # End of region to inpaint
    num_iterations=10,
)
```

## Comparison with Causal LM

| Aspect | Causal LM | Mask Diffusion |
|--------|-----------|----------------|
| Training | Predicts next token | Predicts masked tokens |
| Inference | Sequential generation | Iterative refinement |
| Speed | Fast (single pass) | Slower (multiple iterations) |
| Quality | Good | Potentially better with iterations |
| Editing | Difficult | Easy (inpainting) |
| Use Cases | Standard TTS | TTS + audio editing |

## Files Created/Modified

### New Files
1. `modules/mask_diffusion_train.py` - Training module
2. `modules/mask_diffusion_sampling.py` - Sampling module
3. `modules/llasa_mask_diffusion.py` - LLASA wrapper
4. `main_mask_diffusion.py` - Training script
5. `demo_mask_diffusion.py` - Demo script
6. `config/mask_diffusion_example.yaml` - Configuration example
7. `MASK_DIFFUSION_SUMMARY.md` - This file

### Modified Files
1. `README.md` - Added mask diffusion documentation

## Technical Notes

### Speech Token Range
- Speech tokens: `[128264, 193800)` (i.e., 128264 to 128264+65536, representing 65536 possible codes)
- Speech start marker: `<|s_0|>` through `<|s_65535|>`
- Speech end marker: `<|SPEECH_GENERATION_END|>` (ID: 128261, used to mark end of generation)

### Mask Token
- Uses `[MASK]` special token
- Automatically added to tokenizer if not present
- Model is resized to accommodate new token

### Training Process
1. Load prompt with text and optional reference audio codes
2. Tokenize the full sequence
3. Identify speech token positions
4. Randomly mask a proportion of speech tokens
5. Train model to predict original values of masked tokens
6. Only masked positions contribute to loss

### Sampling Process
1. Create prompt from text
2. Append N masked tokens (all `[MASK]`)
3. For each iteration:
   - Forward pass to get predictions
   - Apply temperature and top-p sampling
   - Fill high-confidence predictions
   - Keep low-confidence positions masked
4. Return final predicted tokens

## Future Improvements

Possible enhancements to consider:
1. **Adaptive masking**: Vary mask ratio during training
2. **Scheduled sampling**: Gradually increase confidence threshold
3. **Parallel decoding**: Decode multiple positions simultaneously
4. **Conditioning**: Add more fine-grained control over generation
5. **Integration with UI**: Add mask diffusion option to Gradio UI

## Testing

All files have been validated:
- ✅ Python syntax checked with AST parsing
- ✅ YAML configuration validated
- ✅ No security vulnerabilities (CodeQL scan)

## Conclusion

The mask diffusion implementation provides an alternative training and inference approach for LLASA that enables:
- Non-autoregressive generation
- Audio inpainting and editing
- Potentially higher quality with iterative refinement

The implementation is modular, well-documented, and integrates seamlessly with the existing LLASA trainer codebase.
