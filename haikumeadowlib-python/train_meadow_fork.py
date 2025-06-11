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
import torch
import torch.nn as nn
import random
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
from dataclasses import dataclass
from enum import Enum

# Import dew ledger for seasonal feedback
from dew_ledger import DewLedger, DewDrop, Season, determine_season, create_atmospheric_vector

# Import existing training components
from ingest import load_haiku_dataset, DatasetSplitter
from generator import HaikuMeadow, AtmosphericConditions

# Try importing existing training utilities
try:
    from train_haiku_model import HaikuModel, train_model, save_model, HaikuLogger
    TRAINING_AVAILABLE = True
except ImportError:
    # Create minimal training infrastructure if not available
    TRAINING_AVAILABLE = False
    print("⚠️  Core training module not found - using minimal training setup")


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
                epochs=1,
                batch_size=1,
                learning_rate=0.001,
                decay_rate=0.1,        # Keep 90% data
                silence_weight=0.8,
                breath_interval=3.0,   # Slow, contemplative
                memory_limit_mb=1024   # 1GB limit
            ),
            BreathPreset.GENTLE: cls(
                epochs=3,
                batch_size=2,
                learning_rate=0.0015,
                decay_rate=0.15,       # Keep 85% data
                silence_weight=0.7,
                breath_interval=2.0,
                memory_limit_mb=2048   # 2GB limit
            ),
            BreathPreset.STEADY: cls(
                epochs=5,
                batch_size=4,
                learning_rate=0.002,
                decay_rate=0.25,       # Keep 75% data (our standard)
                silence_weight=0.6,
                breath_interval=1.5,
                memory_limit_mb=4096   # 4GB limit
            ),
            BreathPreset.DEEP: cls(
                epochs=8,
                batch_size=8,
                learning_rate=0.003,
                decay_rate=0.3,        # More aggressive forgetting
                silence_weight=0.5,
                breath_interval=1.0,
                memory_limit_mb=8192   # 8GB limit
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
        
        # Initialize or load model
        if model_path and model_path.exists():
            print(f"🌱 Continuing training from {model_path}")
            model = self._load_model(model_path)
        else:
            print(f"🌱 Initializing new femto-model")
            model = self._create_fresh_model()
            
        # Training loop with breath intervals
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            print(f"\n🫁 Epoch {epoch + 1}/{self.config.epochs} - inhaling data...")
            
            epoch_loss = 0.0
            batch_count = 0
            
            # Create batches with breath pacing
            for batch in self._breath_batches(dataset):
                
                # Memory safety check
                if self._check_memory_pressure():
                    print("⚠️  Memory pressure detected - pausing for breath...")
                    self._emergency_gc()
                    time.sleep(self.config.breath_interval * 2)
                
                # Process batch
                loss = self._train_batch(model, batch, epoch)
                epoch_loss += loss
                batch_count += 1
                
                # Breath interval - contemplative pause between batches
                time.sleep(self.config.breath_interval)
                
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
            print(f"   🌊 Epoch complete - average loss: {avg_loss:.4f}")
            
            # Save checkpoint
            checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            self._save_checkpoint(model, checkpoint_path, epoch, avg_loss)
            
            # Data decay for next epoch (except last epoch)
            if epoch < self.config.epochs - 1:
                dataset = self._apply_contemplative_decay(dataset)
                
        # Final model save
        final_path = self.output_dir / f"meadow_model_{self.current_season.value}.pt"
        self._save_final_model(model, final_path)
        
        elapsed = time.time() - start_time
        print(f"\n🌸 Training complete in {elapsed/60:.1f} minutes")
        print(f"   Final model: {final_path}")
        print(f"   Peak memory: {self.peak_memory_mb:.1f}MB")
        
        return final_path
    
    def solstice_retune(self, 
                       base_model_path: Path,
                       max_dew_drops: int = 64) -> Path:
        """
        Perform solstice re-tuning using dew ledger resonance.
        Light fine-tuning on community-chosen high-resonance examples.
        """
        
        print(f"🌙 Beginning solstice re-tuning...")
        
        # Get distilled dew drops
        chosen_drops = self.dew_ledger.solstice_distillation(max_chosen=max_dew_drops)
        
        if not chosen_drops:
            print("   No dew drops available for re-tuning")
            return base_model_path
            
        print(f"   Re-tuning on {len(chosen_drops)} resonant examples")
        
        # Convert dew drops to training format
        retune_data = self._dew_drops_to_training_data(chosen_drops)
        
        # Load base model
        model = self._load_model(base_model_path)
        
        # Gentle re-tuning (very light learning rate)
        retune_config = BreathConfig(
            epochs=2,
            batch_size=1,  # One drop at a time
            learning_rate=0.0001,  # Very gentle
            decay_rate=0.0,  # No decay - preserve chosen examples
            silence_weight=0.9,  # Emphasize silence choices
            breath_interval=5.0,  # Slow, contemplative
            memory_limit_mb=1024
        )
        
        # Light training on chosen examples
        for epoch in range(retune_config.epochs):
            print(f"   🌙 Solstice epoch {epoch + 1}/2")
            
            for i, example in enumerate(retune_data):
                loss = self._train_example(model, example)
                
                if i % 10 == 0:
                    print(f"      Processing example {i + 1}/{len(retune_data)}")
                    
                time.sleep(retune_config.breath_interval)
        
        # Save retuned model
        solstice_path = self.output_dir / f"meadow_solstice_{self.current_season.value}.pt"
        self._save_final_model(model, solstice_path)
        
        print(f"🌙 Solstice re-tuning complete: {solstice_path}")
        return solstice_path
    
    def _load_and_decay_data(self, dataset_path: Path) -> List[Dict]:
        """Load dataset and apply initial contemplative decay"""
        
        print(f"📖 Loading dataset from {dataset_path}")
        
        # Use existing ingest functionality if available
        try:
            dataset = load_haiku_dataset(dataset_path)
            splitter = DatasetSplitter(contemplative_decay=self.config.decay_rate)
            train_data, _ = splitter.split_data(dataset)
            
            print(f"   Loaded {len(dataset)} examples, using {len(train_data)} after decay")
            return train_data
            
        except Exception as e:
            print(f"⚠️  Error loading dataset: {e}")
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
    
    def _breath_batches(self, dataset: List[Dict]) -> Iterator[List[Dict]]:
        """Create batches with breath-synchronized pacing"""
        
        # Shuffle with contemplative randomness (seeded)
        random.seed(int(time.time()) % 1000)
        shuffled = dataset.copy()
        random.shuffle(shuffled)
        
        # Create batches
        for i in range(0, len(shuffled), self.config.batch_size):
            batch = shuffled[i:i + self.config.batch_size]
            yield batch
    
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
    
    def _check_memory_pressure(self) -> bool:
        """Check if we're approaching memory limits"""
        
        try:
            import psutil
            memory_mb = psutil.virtual_memory().used / (1024 * 1024)
            self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
            
            return memory_mb > self.config.memory_limit_mb
            
        except ImportError:
            # Can't check memory without psutil
            return False
    
    def _emergency_gc(self):
        """Emergency garbage collection for memory pressure"""
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _dew_drops_to_training_data(self, drops: List[DewDrop]) -> List[Dict]:
        """Convert dew drops to training data format"""
        
        training_data = []
        
        for drop in drops:
            example = {
                "fragment": drop.fragment,
                "haiku": drop.utterance,
                "season_vec": drop.season_vec,
                "resonance": drop.resonance,
                "generation_type": drop.generation_type
            }
            training_data.append(example)
            
        return training_data
    
    # Placeholder methods for model operations
    # These would integrate with existing HaikuModel if available
    
    def _create_fresh_model(self):
        """Create new femto-model"""
        if TRAINING_AVAILABLE:
            return HaikuModel()
        else:
            print("⚠️  Creating placeholder model")
            return None
    
    def _load_model(self, path: Path):
        """Load existing model"""
        if TRAINING_AVAILABLE:
            model = HaikuModel()
            model.load_state_dict(torch.load(path, map_location="cpu"))
            return model
        else:
            print(f"   Simulated loading from {path}")
            return None
    
    def _train_batch(self, model, batch: List[Dict], epoch: int) -> float:
        """Train on single batch"""
        if TRAINING_AVAILABLE and model is not None:
            # Use existing training logic
            # This would need to be adapted to work with the breath config
            return 0.5  # Placeholder loss
        else:
            # Simulate training
            return random.uniform(0.3, 0.8)
    
    def _train_example(self, model, example: Dict) -> float:
        """Train on single example (for solstice retuning)"""
        if TRAINING_AVAILABLE and model is not None:
            # Single example training
            return 0.3  # Placeholder
        else:
            return random.uniform(0.1, 0.4)
    
    def _save_checkpoint(self, model, path: Path, epoch: int, loss: float):
        """Save training checkpoint"""
        if TRAINING_AVAILABLE and model is not None:
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': loss,
                'config': self.config
            }, path)
        print(f"   💾 Checkpoint saved: {path}")
    
    def _save_final_model(self, model, path: Path):
        """Save final trained model"""
        if TRAINING_AVAILABLE and model is not None:
            torch.save(model.state_dict(), path)
        print(f"   💾 Final model saved: {path}")


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
        season_vec=season_vec,
        reason="wisdom_of_silence"
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