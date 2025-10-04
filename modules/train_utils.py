import os
import json
import shutil
from typing import Optional
from transformers import TrainerCallback
from datasets import Dataset
from modules.llasa_utils import get_prompt


class TTSTestCallback(TrainerCallback):
    """Callback for testing TTS during training.
    
    This callback periodically generates test audio during training
    to monitor model performance.
    """
    
    def __init__(
        self, 
        llasa, 
        test_text: str = "こんにちは、マスターさん", 
        test_interval: int = 100, 
        save_path: str = "./test_audio"
    ):
        """Initialize TTS test callback.
        
        Args:
            llasa: LLASA model instance
            test_text: Text to use for testing
            test_interval: Number of steps between tests
            save_path: Directory to save test audio files
        """
        self.llasa = llasa
        self.test_text = test_text
        self.test_interval = test_interval
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
    
    def test_generation(self, step: Optional[int] = None):
        """Test audio generation.
        
        Args:
            step: Current training step number (for filename)
        """
        
        # Generate short audio for testing
        audio_path, status, tokens = self.llasa.generate(
            self.test_text,
            temperature=0.7,
            top_p=0.9, 
            max_tokens=200  # Keep it short for speed
        )
        
        if audio_path:
            print(f"🎵 Test generation successful: '{self.test_text}' -> {status}")
            
            # Save audio file with step number
            if step is not None:
                save_path = os.path.join(self.save_path, f"test_step_{step}.wav")
                shutil.copy2(audio_path, save_path)
                print(f"🎵 Test audio saved: {save_path}")
        else:
            print(f"❌ Test generation failed: {status}")
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step.
        
        Args:
            args: Training arguments
            state: Training state
            control: Training control
            **kwargs: Additional arguments
        """
        
        # Run test at specified intervals, first step, and last step
        if (state.global_step % self.test_interval == 0) or \
           (state.global_step == 1) or \
           (state.global_step == args.max_steps - 1):
            print(f"\n--- Step {state.global_step}: Running test ---")
            self.test_generation(step=state.global_step)
            print("--- Test complete ---\n")


def load_dataset(file_path: str) -> Dataset:
    """Load dataset from JSONL file.
    
    Args:
        file_path: Path to JSONL file containing text and code pairs
        
    Returns:
        HuggingFace Dataset object with formatted prompts
    """
    print(f"📂 Loading dataset from {file_path}...")
    
    # Load JSONL and convert to prompts
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            prompt = get_prompt(text=obj["text"], code=obj.get("code"))
            samples.append(prompt)

    # Convert to HuggingFace Dataset
    train_ds = Dataset.from_dict({"text": samples})
    print(f"✅ Loaded {len(samples)} samples")
    
    return train_ds