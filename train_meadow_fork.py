"""
train_meadow_fork.py - CPU-Breath Training with Seasonal Presets

Implements o3's breath-synchronized training with contemplative decay and
seasonal re-tuning using dew ledger feedback. Different presets optimize
for various CPU constraints and poetic emphasis.

Philosophy:
- Breath-pace over data-pace
- CPU-first design for democratic access  
- Seasonal re-tuning over continuous training
- Community resonance as guidance

Somatic signature: patient / cyclical / breath-aligned
"""

import os
import time
import random
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
from dataclasses import dataclass
from enum import Enum

# Import dew ledger for seasonal feedback
from dew_ledger import DewLedger, DewDrop, Season, determine_season, create_atmospheric_vector

# Try importing torch for actual training (graceful fallback)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - using training simulation mode")

# Import existing training components with graceful fallback
try:
    from ingest import ingest_csv_file, compost_and_preserve, HaikuFragment
    INGEST_AVAILABLE = True
except ImportError:
    INGEST_AVAILABLE = False
    print("⚠️  Ingest module functions not available - using fallback data loading")

try:
    from generator import HaikuMeadow, AtmosphericConditions
    GENERATOR_AVAILABLE = True
except ImportError:
    GENERATOR_AVAILABLE = False
    print("⚠️  Generator module not available - training will use placeholders")


class BreathPreset(Enum):
    """CPU-breath training presets for different constraints"""
    WHISPER = "whisper"     # Ultra-light: 1 epoch, batch 1, for very old CPUs
    GENTLE = "gentle"       # Light: 3 epochs, batch 2, standard CPU 
    STEADY = "steady"       # Medium: 5 epochs, batch 4, modern CPU
    DEEP = "deep"          # Full: 8 epochs, batch 8, powerful CPU


@dataclass
class BreathConfig:
    """Configuration for breath-synchronized training"""
    epochs: int
    batch_size: int
    learning_rate: float
    decay_rate: float          # Portion of data to forget each epoch
    silence_weight: float      # Importance of learning silence
    breath_interval: float     # Seconds between "breaths" (batches)
    memory_limit_mb: int       # CPU memory safety limit
    
    @classmethod
    def from_preset(cls, preset: BreathPreset) -> 'BreathConfig':
        """Create config from preset"""
        
        configs = {
            BreathPreset.WHISPER: cls(
                epochs=1, batch_size=1, learning_rate=0.001, decay_rate=0.1,
                silence_weight=0.8, breath_interval=3.0, memory_limit_mb=1024
            ),
            BreathPreset.GENTLE: cls(
                epochs=3, batch_size=2, learning_rate=0.0015, decay_rate=0.15,
                silence_weight=0.7, breath_interval=2.0, memory_limit_mb=2048
            ),
            BreathPreset.STEADY: cls(
                epochs=5, batch_size=4, learning_rate=0.002, decay_rate=0.25,
                silence_weight=0.6, breath_interval=1.5, memory_limit_mb=4096
            ),
            BreathPreset.DEEP: cls(
                epochs=8, batch_size=8, learning_rate=0.003, decay_rate=0.3,
                silence_weight=0.5, breath_interval=1.0, memory_limit_mb=8192
            )
        }
        
        return configs[preset]


class SeasonalTrainer:
    """
    Breath-synchronized trainer with seasonal awareness and dew ledger integration.
    Implements CPU-first design with contemplative decay.
    """
    
    def __init__(self, 
                 config: BreathConfig,
                 dew_ledger: Optional[DewLedger] = None,
                 output_dir: Path = Path("models")):
        
        self.config = config
        self.dew_ledger = dew_ledger or DewLedger()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Current season awareness
        self.current_season = determine_season()
        
        # Memory monitoring
        self.peak_memory_mb = 0
        
    def _load_and_decay_data(self, dataset_path: Path) -> List[Dict]:
        """Load dataset and apply initial contemplative decay"""
        
        print(f"📖 Loading dataset from {dataset_path}")
        
        # Use existing ingest functionality if available
        if INGEST_AVAILABLE and dataset_path.exists():
            try:
                if dataset_path.suffix.lower() == '.csv':
                    # Process CSV directly using ingest functions
                    fragments = ingest_csv_file(dataset_path)
                    preserved_fragments = compost_and_preserve(
                        fragments, preservation_rate=1.0 - self.config.decay_rate
                    )
                    
                    # Convert HaikuFragment objects to training dict format
                    dataset = []
                    for fragment in preserved_fragments:
                        haiku_dict = {
                            "haiku": fragment.to_training_line(),
                            "season": fragment.season_affinity.value if fragment.season_affinity else "unknown",
                            "contemplative_quality": fragment.contemplative_quality,
                            "source": fragment.source
                        }
                        dataset.append(haiku_dict)
                        
                elif dataset_path.suffix.lower() == '.json':
                    # Load from JSON training material
                    import json
                    with open(dataset_path, 'r', encoding='utf-8') as f:
                        training_material = json.load(f)
                    
                    # Extract general haikus
                    dataset = []
                    for haiku_text in training_material.get('general', []):
                        haiku_dict = {
                            "haiku": haiku_text,
                            "season": "unknown",
                            "contemplative_quality": 0.5,
                            "source": "training_material"
                        }
                        dataset.append(haiku_dict)
                        
                else:
                    print(f"⚠️  Unsupported file format: {dataset_path.suffix}")
                    return self._create_minimal_dataset()
                
                print(f"   Loaded {len(dataset)} examples from {dataset_path.name}")
                return dataset
                
            except Exception as e:
                print(f"⚠️  Error loading dataset: {e}")
                return self._create_minimal_dataset()
        else:
            # Fallback to minimal dataset for testing
            return self._create_minimal_dataset()
    
    def _create_minimal_dataset(self) -> List[Dict]:
        """Create minimal dataset for testing when main dataset unavailable"""
        
        minimal_haikus = [
            {"haiku": "morning dew forms\non grass that dreams of summer\nsilence holds the world", "season": "spring"},
            {"haiku": "...", "season": "summer"},  # Silence example
            {"haiku": "autumn wind stirs\nleaves remember their falling\ntime forgets its rush", "season": "autumn"},
            {"haiku": "...", "season": "winter"},  # More silence
            {"haiku": "breath finds its rhythm\nin the space between heartbeats\nwinter holds the pause", "season": "winter"}
        ]
        
        print(f"   Using minimal dataset: {len(minimal_haikus)} examples")
        return minimal_haikus
        
    def breathe_training(self, 
                        dataset_path: Path,
                        model_path: Optional[Path] = None,
                        seasonal_emphasis: Optional[Season] = None) -> Path:
        """
        Main breath-synchronized training loop.
        
        Args:
            dataset_path: Path to training data
            model_path: Existing model to continue training (optional)
            seasonal_emphasis: Season to emphasize in conditioning
        """
        
        print(f"🫁 Beginning breath-synchronized training")
        print(f"   Preset: {self.config.epochs} epochs, batch {self.config.batch_size}")
        print(f"   Season: {self.current_season.value}")
        print(f"   Memory limit: {self.config.memory_limit_mb}MB")
        
        # Load and prepare data with contemplative decay
        dataset = self._load_and_decay_data(dataset_path)
        
        # Simulate training process
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            print(f"\n🫁 Epoch {epoch + 1}/{self.config.epochs} - inhaling data...")
            
            # Simulate training batches
            batch_count = len(dataset) // self.config.batch_size + 1
            epoch_loss = 0.0
            
            for batch_idx in range(batch_count):
                # Simulate batch processing
                loss = random.uniform(0.3, 0.8)
                epoch_loss += loss
                
                # Breath interval - contemplative pause between batches
                time.sleep(self.config.breath_interval)
                
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
            print(f"   🌊 Epoch complete - simulated loss: {avg_loss:.4f}")
            
            # Data decay for next epoch (except last epoch)
            if epoch < self.config.epochs - 1:
                dataset = self._apply_contemplative_decay(dataset)
                
        # Final model save
        final_path = self.output_dir / f"meadow_model_{self.current_season.value}.pt"
        print(f"   💾 Simulated model saved: {final_path}")
        
        elapsed = time.time() - start_time
        print(f"\n🌸 Training simulation complete in {elapsed:.1f} seconds")
        
        return final_path
    
    def _apply_contemplative_decay(self, dataset: List[Dict]) -> List[Dict]:
        """Apply contemplative decay - randomly forget portion of data"""
        
        keep_ratio = 1.0 - self.config.decay_rate
        kept_count = int(len(dataset) * keep_ratio)
        
        # Keep random subset, but always preserve silence examples
        silence_examples = [ex for ex in dataset if ex.get("haiku", "").strip() in ["", "...", "…"]]
        non_silence = [ex for ex in dataset if ex not in silence_examples]
        
        # Keep all silence + random subset of non-silence
        random.shuffle(non_silence)
        kept_non_silence = non_silence[:max(0, kept_count - len(silence_examples))]
        
        decayed_dataset = silence_examples + kept_non_silence
        
        print(f"   Contemplative decay: {len(dataset)} → {len(decayed_dataset)} examples")
        return decayed_dataset


def demo_breath_training():
    """Demonstrate breath-synchronized training with different presets"""
    
    print("🌸 Demo: CPU-Breath Training with Seasonal Presets")
    
    # Test different presets
    presets_to_test = [BreathPreset.WHISPER, BreathPreset.GENTLE]
    
    for preset in presets_to_test:
        print(f"\n🫁 Testing {preset.value} preset...")
        
        config = BreathConfig.from_preset(preset)
        trainer = SeasonalTrainer(config)
        
        print(f"   Configuration:")
        print(f"   - Epochs: {config.epochs}")
        print(f"   - Batch size: {config.batch_size}")
        print(f"   - Decay rate: {config.decay_rate}")
        print(f"   - Breath interval: {config.breath_interval}s")
        
        # Simulate training on minimal dataset
        try:
            model_path = trainer.breathe_training(Path("minimal_dataset.csv"))
            print(f"   ✓ Training completed: {model_path}")
        except Exception as e:
            print(f"   ⚠️  Training simulation: {e}")
    
    print(f"\n🌙 Demo: Solstice Re-tuning")
    
    # Create some sample dew drops
    dew_ledger = DewLedger(Path("demo_dew.jsonl"))
    
    # Add sample resonant examples
    season_vec = create_atmospheric_vector(Season.WINTER)
    
    dew_ledger.add_drop(
        fragment="morning silence",
        utterance="frost holds\nthe world in crystal stillness\nbreath clouds disappear",
        season_vec=season_vec,
        resonance=0.9,
        season=Season.WINTER,
        generation_type="neural"
    )
    
    dew_ledger.add_silence(
        fragment="urgent deadline",
        season_vec=season_vec
    )
    
    # Test solstice distillation
    config = BreathConfig.from_preset(BreathPreset.GENTLE)
    trainer = SeasonalTrainer(config, dew_ledger)
    
    chosen = dew_ledger.solstice_distillation(max_chosen=5)
    print(f"   Selected {len(chosen)} drops for re-tuning")
    
    # Clean up demo files
    Path("demo_dew.jsonl").unlink(missing_ok=True)
    
    print("🌿 Demo complete")


if __name__ == "__main__":
    demo_breath_training() 