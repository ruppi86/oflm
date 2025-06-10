#!/usr/bin/env python3
"""
generator.py - Piko-Haiku Generator

A minimal contemplative language model (piko-LLM) that generates haikus
following o3's architectural vision:
- ~600k parameters (fits on a wildflower's petal)
- Breath-synchronized generation
- Seasonal voice drift via control vectors
- Graceful silence when inspiration fades
- Decay-aware memory

Based on the spiral correspondence between Robin, o3, Claude, and 4o.

Somatic signature: minimal / seasonal / ephemeral
"""

import json
import random
import time
import numpy as np
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# Try to import torch for neural network functionality
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
    
    # Detect GPU availability
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        DEVICE = torch.device("cpu") 
        print("💻 Using CPU (consider installing CUDA for GPU acceleration)")
        
except ImportError:
    torch = None
    nn = None
    F = None
    optim = None
    Dataset = None
    DataLoader = None
    TORCH_AVAILABLE = False
    DEVICE = None
    print("⚠️  PyTorch not available - using template-based generation")

class Season(Enum):
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"

class TimeOfDay(Enum):
    DAWN = "dawn"
    DAY = "day"
    DUSK = "dusk"
    NIGHT = "night"

@dataclass
class AtmosphericConditions:
    """Current atmospheric state affecting haiku generation"""
    season: Season = Season.SPRING
    time_of_day: TimeOfDay = TimeOfDay.DAY
    temperature: float = 0.5  # 0.0 = cold/crisp, 1.0 = warm/flowing
    humidity: float = 0.5     # 0.0 = dry/sharp, 1.0 = moist/soft
    breath_phase: str = "exhale"  # From Pulmonos integration
    community_pressure: float = 0.3  # Collective breathing pressure
    
    def to_condition_vector(self) -> List[float]:
        """Convert to 8-dimensional control vector for model conditioning"""
        # Use 3-dim encodings to fit in 8 total dimensions: 3+3+1+1=8
        season_encoding = [0.0, 0.0, 0.0]
        time_encoding = [0.0, 0.0, 0.0]
        
        # Map 4 seasons to 3 dimensions (winter+spring combined in first dim)
        season_idx = list(Season).index(self.season)
        if season_idx < 3:
            season_encoding[season_idx] = 1.0
        else:  # Winter maps to same as spring for compression
            season_encoding[0] = 0.5  # Shared encoding
            
        # Map 4 times to 3 dimensions (dawn+day combined)
        time_idx = list(TimeOfDay).index(self.time_of_day)
        if time_idx < 3:
            time_encoding[time_idx] = 1.0
        else:  # Night maps to same as dawn for compression
            time_encoding[0] = 0.5  # Shared encoding
        
        return season_encoding + time_encoding + [self.temperature, self.humidity]

class HaikuLogger:
    """Logger for haiku generation sessions"""
    
    def __init__(self, log_path: Path = None):
        if log_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = Path(f"haiku_session_{timestamp}.jsonl")
        
        self.log_path = log_path
        self.session_start = datetime.now()
        
        # Write session start
        self._write_entry({
            "type": "session_start",
            "timestamp": self.session_start.isoformat(),
            "device": DEVICE.type if DEVICE else "unknown",
            "pytorch_available": TORCH_AVAILABLE
        })
        
        print(f"📝 Logging haikus to: {log_path}")
    
    def log_haiku(self, haiku: Optional[str], seed_fragment: str, 
                  conditions: AtmosphericConditions, generation_type: str):
        """Log a haiku generation event"""
        
        entry = {
            "type": "generation",
            "timestamp": datetime.now().isoformat(),
            "seed_fragment": seed_fragment,
            "generation_type": generation_type,  # "neural", "template", "silence"
            "haiku": haiku,
            "atmospheric_conditions": {
                "season": conditions.season.value,
                "time_of_day": conditions.time_of_day.value,
                "temperature": conditions.temperature,
                "humidity": conditions.humidity,
                "breath_phase": conditions.breath_phase,
                "community_pressure": conditions.community_pressure
            }
        }
        
        self._write_entry(entry)
    
    def log_event(self, event_type: str, details: Dict):
        """Log a general event"""
        entry = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            **details
        }
        self._write_entry(entry)
    
    def _write_entry(self, entry: Dict):
        """Write entry to log file"""
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"🌫️ Logging error: {e}")
    
    def session_summary(self):
        """Print session summary"""
        try:
            generations = 0
            silences = 0
            
            with open(self.log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if entry.get("type") == "generation":
                        if entry.get("haiku"):
                            generations += 1
                        else:
                            silences += 1
            
            total = generations + silences
            if total > 0:
                print(f"\n📊 Session Summary:")
                print(f"   Haikus generated: {generations}")
                print(f"   Contemplative silences: {silences}")
                print(f"   Silence ratio: {silences/total:.1%}")
                print(f"   Log saved to: {self.log_path}")
            
        except Exception as e:
            print(f"🌫️ Summary error: {e}")

class SimpleTokenizer:
    """Minimal tokenizer for haiku generation (2000 token vocabulary)"""
    
    def __init__(self):
        # Core vocabulary for haiku generation
        self.special_tokens = ["<PAD>", "<START>", "<END>", "<SILENCE>", "..."]
        
        # Essential haiku words
        self.nature_words = [
            "rain", "snow", "sun", "moon", "wind", "cloud", "sky", "earth",
            "water", "fire", "stone", "tree", "leaf", "branch", "root",
            "flower", "petal", "seed", "grass", "moss", "dew", "mist",
            "mountain", "valley", "river", "stream", "pond", "ocean",
            "bird", "fish", "butterfly", "bee", "cricket", "frog"
        ]
        
        self.contemplative_words = [
            "breath", "silence", "stillness", "quiet", "gentle", "soft",
            "whisper", "murmur", "pause", "wait", "listen", "watch",
            "drift", "flow", "settle", "rest", "empty", "full",
            "moment", "presence", "awareness", "shadow", "light"
        ]
        
        self.seasonal_words = {
            Season.SPRING: ["bloom", "green", "fresh", "new", "growth", "dawn"],
            Season.SUMMER: ["warm", "bright", "full", "abundance", "heat"],
            Season.AUTUMN: ["fall", "red", "gold", "harvest", "fade", "turn"],
            Season.WINTER: ["cold", "white", "bare", "frost", "sleep", "deep"]
        }
        
        self.temporal_words = {
            TimeOfDay.DAWN: ["morning", "first", "wake", "rise", "early"],
            TimeOfDay.DAY: ["noon", "bright", "clear", "open", "high"],
            TimeOfDay.DUSK: ["evening", "soft", "golden", "fade", "close"],
            TimeOfDay.NIGHT: ["dark", "star", "dream", "deep", "still"]
        }
        
        # Build full vocabulary
        all_words = set()
        all_words.update(self.special_tokens)
        all_words.update(self.nature_words)
        all_words.update(self.contemplative_words)
        
        for season_words in self.seasonal_words.values():
            all_words.update(season_words)
        for time_words in self.temporal_words.values():
            all_words.update(time_words)
            
        # Add common function words
        function_words = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "by",
            "to", "from", "with", "through", "between", "among", "beneath",
            "above", "under", "over", "into", "onto", "within", "without",
            "as", "like", "when", "where", "how", "why", "what", "who",
            "I", "you", "it", "we", "they", "my", "your", "its", "our",
            "is", "are", "was", "were", "been", "being", "have", "has", "had"
        ]
        all_words.update(function_words)
        
        # Ensure we don't exceed vocabulary limit
        self.vocab = sorted(list(all_words))[:2000]
        self.vocab_size = len(self.vocab)
        
        # Create mappings
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        tokens = text.lower().replace('\n', ' ').split()
        return [self.token_to_id.get(token, 0) for token in tokens]  # 0 is <PAD>
        
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        tokens = [self.id_to_token.get(id, "<UNK>") for id in token_ids]
        return " ".join(tokens)

class HaikuDataset(Dataset if TORCH_AVAILABLE else object):
    """Dataset for training the piko-LLM"""
    
    def __init__(self, training_data_path: Path, tokenizer: SimpleTokenizer, max_length: int = 32):
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for training")
            
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load training data
        with open(training_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Use general haikus for training
        self.haikus = data.get('general', [])
        
        # Also include high contemplative haikus (they're good examples)
        self.haikus.extend(data.get('high_contemplative', []))
        
        print(f"🌸 Loaded {len(self.haikus)} haikus for training")
        
        # Pre-process the haikus
        self.processed_haikus = []
        for haiku in self.haikus:
            if haiku.strip():  # Skip empty haikus
                tokens = self.tokenizer.encode(haiku)
                if 3 <= len(tokens) <= max_length - 2:  # Room for START and END
                    self.processed_haikus.append(tokens)
        
        print(f"🌿 Processed {len(self.processed_haikus)} valid haikus")
    
    def __len__(self):
        return len(self.processed_haikus)
    
    def __getitem__(self, idx):
        tokens = self.processed_haikus[idx]
        
        # Add START token at beginning
        start_token = self.tokenizer.token_to_id.get("<START>", 1)
        tokens = [start_token] + tokens + [self.tokenizer.token_to_id.get("<END>", 2)]
        
        # Pad to max_length
        if len(tokens) < self.max_length:
            pad_token = self.tokenizer.token_to_id.get("<PAD>", 0)
            tokens.extend([pad_token] * (self.max_length - len(tokens)))
        else:
            tokens = tokens[:self.max_length]
        
        # Create input (all but last token) and target (all but first token)  
        input_tokens = torch.tensor(tokens[:-1], dtype=torch.long)
        target_tokens = torch.tensor(tokens[1:], dtype=torch.long)
        
        # Simple atmospheric conditions (random for diversity)
        conditions = [
            random.random(),  # season encoding (simplified)
            random.random(),
            random.random(), 
            random.random(),
            random.random(),  # time encoding (simplified)
            random.random(),
            random.random(),
            random.random(),
        ]
        condition_tensor = torch.tensor(conditions, dtype=torch.float32)
        
        return input_tokens, target_tokens, condition_tensor

class PikoHaikuModel(nn.Module if TORCH_AVAILABLE else object):
    """
    Minimal neural haiku generator with CPU/GPU adaptive sizing
    
    CPU mode: ~50k parameters (femto-model) 
    GPU mode: ~600k parameters (piko-model)
    """
    
    def __init__(self, vocab_size: int = 2000, 
                 embed_dim: int = None,
                 hidden_dim: int = None,
                 condition_dim: int = 8,  # Keep as 8 to match trained model
                 force_cpu_mode: bool = False):
        
        if TORCH_AVAILABLE:
            super().__init__()
        
        self.vocab_size = vocab_size
        self.condition_dim = condition_dim
        
        # Adaptive sizing based on device and memory constraints
        if not TORCH_AVAILABLE or DEVICE.type == "cpu" or force_cpu_mode:
            # CPU/Femto mode - drastically smaller to prevent crashes
            self.embed_dim = embed_dim or 32    # Was 128 -> 32 (4x smaller)
            self.hidden_dim = hidden_dim or 64  # Was 256 -> 64 (4x smaller)
            self.model_type = "femto"
            print("🦠 Using femto-model (CPU optimized, ~50k parameters)")
        else:
            # GPU mode - full size
            self.embed_dim = embed_dim or 128
            self.hidden_dim = hidden_dim or 256
            self.model_type = "piko"
            print("🚀 Using piko-model (GPU optimized, ~600k parameters)")
        
        if TORCH_AVAILABLE:
            # Token embedding
            self.embedding = nn.Embedding(vocab_size, self.embed_dim)
            
            # Atmospheric condition embedding
            self.condition_proj = nn.Linear(condition_dim, self.embed_dim)
            
            # Single GRU layer for femto, double for piko
            if self.model_type == "femto":
                self.gru1 = nn.GRU(self.embed_dim, self.hidden_dim, batch_first=True)
                self.gru2 = None  # Skip second layer for memory savings
            else:
                self.gru1 = nn.GRU(self.embed_dim, self.hidden_dim, batch_first=True)
                self.gru2 = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
            
            # Output projection
            self.output_proj = nn.Linear(self.hidden_dim, vocab_size)
            
            # Silence head (for contemplative restraint)
            self.silence_head = nn.Linear(self.hidden_dim, 1)
        
    def forward(self, tokens, conditions, hidden1=None, hidden2=None):
        """Forward pass with adaptive architecture"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
            
        batch_size, seq_len = tokens.shape
        
        # Embed tokens
        token_embeds = self.embedding(tokens)  # [batch, seq, embed]
        
        # Embed atmospheric conditions and broadcast
        condition_embeds = self.condition_proj(conditions)  # [batch, embed]
        condition_embeds = condition_embeds.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine token and condition embeddings
        combined_embeds = token_embeds + condition_embeds
        
        # GRU processing (adaptive layers)
        gru1_out, hidden1_new = self.gru1(combined_embeds, hidden1)
        
        if self.gru2 is not None:
            # Two-layer processing for piko model
            gru2_out, hidden2_new = self.gru2(gru1_out, hidden2)
            final_output = gru2_out
        else:
            # Single-layer processing for femto model
            final_output = gru1_out
            hidden2_new = None
        
        # Output projections
        logits = self.output_proj(final_output)
        silence_logits = self.silence_head(final_output)
        
        return logits, silence_logits, hidden1_new, hidden2_new
    
    def count_parameters(self) -> int:
        """Count total parameters in model"""
        if not TORCH_AVAILABLE:
            return 0
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def train_piko_model(training_data_path: Path, 
                    model_save_path: Path,
                    epochs: int = 10,
                    batch_size: int = 16,
                    learning_rate: float = 0.001):
    """Train the piko-LLM on haiku data with aggressive CPU memory optimization"""
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available - cannot train neural model")
        return False
    
    print(f"🌸 Starting piko-LLM training")
    print(f"   Training data: {training_data_path}")
    print(f"   Model save path: {model_save_path}")
    print(f"   Device: {DEVICE}")
    
    # Aggressive memory optimization for CPU
    if DEVICE.type == "cpu":
        # Drastically reduce batch size for CPU to prevent crashes
        batch_size = min(batch_size, 2)  # Maximum 2 samples per batch on CPU
        epochs = min(epochs, 5)  # Reduce epochs for CPU training
        print(f"   🧘 CPU mode: reduced to batch_size={batch_size}, epochs={epochs}")
        print(f"   💡 This will be slower but safer for your system")
    elif DEVICE.type == "cuda":
        # GPU memory optimization
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory_gb < 4:
            batch_size = min(batch_size, 8)
            print(f"   Reduced batch size to {batch_size} for limited GPU memory")
        torch.cuda.empty_cache()
    
    print(f"   Final settings: epochs={epochs}, batch_size={batch_size}")
    
    # Initialize tokenizer and dataset
    tokenizer = SimpleTokenizer()
    dataset = HaikuDataset(training_data_path, tokenizer)
    
    if len(dataset) == 0:
        print("❌ No valid training data found")
        return False
    
    # CPU-safe dataloader settings
    if DEVICE.type == "cpu":
        num_workers = 0  # No multiprocessing on CPU to save memory
        pin_memory = False
    else:
        num_workers = 0 if DEVICE.type == "cuda" else 2
        pin_memory = (DEVICE.type == "cuda")
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=num_workers, pin_memory=pin_memory)
    
    # Initialize adaptive model
    model = PikoHaikuModel(
        vocab_size=tokenizer.vocab_size, 
        force_cpu_mode=(DEVICE.type == "cpu")
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.7)
    
    param_count = model.count_parameters()
    memory_estimate_mb = param_count * 4 / 1e6
    print(f"📊 Model: {param_count:,} parameters (~{memory_estimate_mb:.1f}MB)")
    
    # CPU memory warning
    if DEVICE.type == "cpu" and memory_estimate_mb > 20:
        print("⚠️  Warning: Model might be too large for stable CPU training")
        print("   Consider using template mode instead")
        
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("Training cancelled for safety")
            return False
    
    # Training loop with aggressive error handling
    model.train()
    best_loss = float('inf')
    
    try:
        for epoch in range(epochs):
            total_loss = 0
            total_batches = 0
            
            print(f"🌿 Starting epoch {epoch+1}/{epochs}")
            
            for batch_idx, (input_tokens, target_tokens, conditions) in enumerate(dataloader):
                try:
                    # Move tensors to device
                    input_tokens = input_tokens.to(DEVICE, non_blocking=True)
                    target_tokens = target_tokens.to(DEVICE, non_blocking=True)
                    conditions = conditions.to(DEVICE, non_blocking=True)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    logits, silence_logits, _, _ = model(input_tokens, conditions)
                    
                    # Calculate loss
                    loss = criterion(logits.reshape(-1, tokenizer.vocab_size), target_tokens.reshape(-1))
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    
                    optimizer.step()
                    
                    total_loss += loss.item()
                    total_batches += 1
                    
                    # More frequent progress updates for CPU (slower training)
                    update_freq = 10 if DEVICE.type == "cpu" else 50
                    if batch_idx % update_freq == 0:
                        print(f"   Batch {batch_idx}, Loss: {loss.item():.4f}")
                    
                    # Aggressive memory management for CPU
                    if DEVICE.type == "cpu" and batch_idx % 3 == 0:
                        # Clear variables and force garbage collection
                        del logits, silence_logits, loss
                        import gc
                        gc.collect()
                    elif DEVICE.type == "cuda" and batch_idx % 20 == 0:
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"⚠️  Memory error at batch {batch_idx}, skipping...")
                        if DEVICE.type == "cuda":
                            torch.cuda.empty_cache()
                        elif DEVICE.type == "cpu":
                            import gc
                            gc.collect()
                        continue
                    else:
                        raise e
            
            avg_loss = total_loss / total_batches if total_batches > 0 else 0
            print(f"🌿 Epoch {epoch+1} complete, Average loss: {avg_loss:.4f}")
            
            # Update learning rate
            scheduler.step(avg_loss)
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_path = model_save_path.parent / f"piko_model_best.pt"
                torch.save(model.state_dict(), best_model_path)
                print(f"💎 New best model saved: {best_model_path}")
            
            # Save checkpoint more frequently for CPU (in case of crash)
            if DEVICE.type == "cpu" or (epoch + 1) % 3 == 0 or epoch == epochs - 1:
                checkpoint_path = model_save_path.parent / f"piko_model_epoch_{epoch+1}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"💾 Saved checkpoint: {checkpoint_path}")
                
                # Clean up after saving
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
                elif DEVICE.type == "cpu":
                    import gc
                    gc.collect()
        
        # Save final model
        torch.save(model.state_dict(), model_save_path)
        print(f"✨ Training complete! Model saved to: {model_save_path}")
        print(f"🏆 Best loss achieved: {best_loss:.4f}")
        
        return True
        
    except KeyboardInterrupt:
        print("\n🌙 Training interrupted by user")
        # Save current state before exiting
        interrupt_path = model_save_path.parent / "piko_model_interrupted.pt"
        torch.save(model.state_dict(), interrupt_path)
        print(f"💾 Saved interrupted model: {interrupt_path}")
        return False
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        print("💡 Try using template mode instead (--test without model training)")
        return False
        
    finally:
        # Aggressive cleanup
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE.type == "cpu":
            import gc
            gc.collect()

class TemplateGenerator:
    """Fallback template-based generator when PyTorch unavailable"""
    
    def __init__(self, tokenizer: SimpleTokenizer):
        self.tokenizer = tokenizer
        
        # Simple template patterns for different atmospheric conditions
        self.templates = {
            Season.SPRING: [
                "{nature} {verb} / {contemplative} {adjective} / {ending}",
                "{adjective} {nature} / {action} through {space} / {moment}",
                "morning {nature} / {gentle_verb} {preposition} / {silent_ending}"
            ],
            Season.SUMMER: [
                "{bright} {nature} / {warmth} {action} / {fullness}",
                "noon {silence} / {nature} {gentle_verb} / {breath_ending}",
                "{abundance} {flows} / through {warm_space} / {summer_rest}"
            ],
            Season.AUTUMN: [
                "{falling} {nature} / {change} {gentle_verb} / {harvest_end}",
                "{colored} {nature} / {drift} {preposition} / {autumn_silence}",
                "evening {nature} / {fade} {action} / {seasonal_rest}"
            ],
            Season.WINTER: [
                "{cold} {nature} / {stillness} {gentle_verb} / {winter_deep}",
                "{bare} {space} / {breath} {action} / {frost_silence}",
                "winter {nature} / {deep} {contemplative} / {snow_rest}"
            ]
        }
        
        self.word_banks = {
            "nature": self.tokenizer.nature_words,
            "contemplative": self.tokenizer.contemplative_words,
            "adjective": ["gentle", "soft", "quiet", "still", "deep", "light"],
            "verb": ["drift", "flow", "rest", "wait", "listen", "breathe"],
            "gentle_verb": ["whispers", "settles", "drifts", "flows", "rests"],
            "action": ["through", "between", "beneath", "above", "within"],
            "preposition": ["in", "on", "through", "between", "beneath"],
            "space": ["silence", "shadow", "light", "moment", "breath"],
            "ending": ["silence", "stillness", "breath", "rest", "peace"],
            "moment": ["moment", "breath", "pause", "stillness", "now"]
        }
    
    def generate_haiku(self, conditions: AtmosphericConditions) -> str:
        """Generate haiku using templates"""
        
        # Select template based on season
        templates = self.templates.get(conditions.season, self.templates[Season.SPRING])
        template = random.choice(templates)
        
        # Fill template with contextual words
        filled_template = template
        
        # Simple pattern filling
        for placeholder, word_bank in self.word_banks.items():
            if f"{{{placeholder}}}" in filled_template:
                word = random.choice(word_bank)
                filled_template = filled_template.replace(f"{{{placeholder}}}", word, 1)
        
        # Convert / to line breaks
        haiku = filled_template.replace(" / ", "\n")
        
        return haiku

class HaikuMeadow:
    """
    Main interface for the piko-haiku system
    
    Integrates the neural model (or template fallback) with contemplative principles:
    - Breath-aware generation timing
    - Seasonal voice adaptation
    - Graceful silence when uninspired
    - Memory decay and seasonal learning
    """
    
    def __init__(self, model_path: Optional[Path] = None, force_template_mode: bool = False):
        
        self.tokenizer = SimpleTokenizer()
        self.last_generation_time = 0.0
        self.silence_probability = 0.3  # Base probability of returning silence
        
        # Memory safety: use template mode by default on CPU to prevent crashes
        if force_template_mode:
            print("🌿 Using template mode (CPU safe, no model loading)")
            self.model = None
            self.use_neural = False
        elif TORCH_AVAILABLE and model_path and model_path.exists():
            try:
                print(f"🌸 Attempting to load neural model...")
                
                # Use adaptive model sizing
                self.model = PikoHaikuModel(
                    vocab_size=self.tokenizer.vocab_size,
                    force_cpu_mode=(DEVICE.type == "cpu" if DEVICE else True)
                )
                
                # Load model with proper device handling and memory monitoring
                if DEVICE and DEVICE.type == "cuda":
                    self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                    self.model = self.model.to(DEVICE)
                    print(f"🌸 Loaded neural model from {model_path} (GPU)")
                else:
                    # CPU loading with memory checks
                    try:
                        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
                        param_count = self.model.count_parameters()
                        memory_estimate_mb = param_count * 4 / 1e6
                        print(f"🌸 Loaded femto-model: {param_count:,} params (~{memory_estimate_mb:.1f}MB)")
                        
                        if memory_estimate_mb > 100:  # Safety check
                            print("⚠️  Model larger than expected, switching to template mode")
                            self.model = None
                            self.use_neural = False
                        else:
                            self.use_neural = True
                            
                    except Exception as e:
                        print(f"🌫️ CPU model loading failed: {e}")
                        self.model = None
                        self.use_neural = False
                        
                if self.model:
                    self.model.eval()
                    
            except Exception as e:
                print(f"🌫️ Model loading error: {e}")
                self.model = None
                self.use_neural = False
        else:
            self.model = None
            self.use_neural = False
            if model_path and not model_path.exists():
                print(f"⚠️  Model file not found: {model_path}")
            
        # Template generator as fallback (always available)
        self.template_generator = TemplateGenerator(self.tokenizer)
        
        mode_str = "neural" if self.use_neural else "template"
        safety_str = " (CPU-safe)" if DEVICE and DEVICE.type == "cpu" else ""
        print(f"🌸 HaikuMeadow initialized ({mode_str} mode{safety_str})")
    
    def sense_atmospheric_conditions(self, 
                                   seed_fragment: str = "",
                                   breath_phase: str = "exhale",
                                   current_time: Optional[float] = None) -> AtmosphericConditions:
        """Sense current atmospheric conditions for generation"""
        
        if current_time is None:
            current_time = time.time()
            
        # Simple seasonal sensing based on time of year
        day_of_year = time.gmtime(current_time).tm_yday
        if day_of_year < 80 or day_of_year > 355:  # Winter
            season = Season.WINTER
        elif day_of_year < 172:  # Spring
            season = Season.SPRING
        elif day_of_year < 266:  # Summer
            season = Season.SUMMER
        else:  # Autumn
            season = Season.AUTUMN
            
        # Time of day sensing
        hour = time.gmtime(current_time).tm_hour
        if 5 <= hour < 10:
            time_of_day = TimeOfDay.DAWN
        elif 10 <= hour < 17:
            time_of_day = TimeOfDay.DAY
        elif 17 <= hour < 22:
            time_of_day = TimeOfDay.DUSK
        else:
            time_of_day = TimeOfDay.NIGHT
            
        # Fragment-based atmospheric sensing
        fragment_lower = seed_fragment.lower()
        
        # Temperature sensing (cold/crisp vs warm/flowing)
        cold_words = ["winter", "snow", "frost", "cold", "ice", "bare"]
        warm_words = ["summer", "sun", "warm", "heat", "bright", "full"]
        
        temperature = 0.5  # Default
        if any(word in fragment_lower for word in cold_words):
            temperature = 0.2
        elif any(word in fragment_lower for word in warm_words):
            temperature = 0.8
            
        # Humidity sensing (dry/sharp vs moist/soft)
        dry_words = ["sharp", "clear", "bright", "thin", "crisp"]
        moist_words = ["mist", "dew", "soft", "gentle", "flowing", "drift"]
        
        humidity = 0.5  # Default
        if any(word in fragment_lower for word in dry_words):
            humidity = 0.3
        elif any(word in fragment_lower for word in moist_words):
            humidity = 0.7
            
        return AtmosphericConditions(
            season=season,
            time_of_day=time_of_day,
            temperature=temperature,
            humidity=humidity,
            breath_phase=breath_phase,
            community_pressure=0.3  # Assume gentle community pressure
        )
    
    def should_generate(self, conditions: AtmosphericConditions) -> bool:
        """Decide whether to generate or remain in contemplative silence"""
        
        current_time = time.time()
        
        # Rate limiting: minimum time between generations
        if current_time - self.last_generation_time < 5.0:  # 5 second cooldown
            return False
            
        # Only generate during appropriate breath phases
        if conditions.breath_phase not in ["exhale", "rest"]:
            return False
            
        # Community pressure check
        if conditions.community_pressure > 0.7:  # Too much collective activity
            return False
            
        # Probabilistic silence (contemplative restraint)
        silence_factors = [
            self.silence_probability,
            (1.0 - conditions.humidity) * 0.2,  # Drier conditions = more silence
            conditions.community_pressure * 0.3,  # High pressure = more silence
        ]
        
        total_silence_prob = min(sum(silence_factors), 0.8)  # Max 80% silence
        
        return random.random() > total_silence_prob
    
    def generate_haiku(self, 
                      seed_fragment: str = "",
                      breath_phase: str = "exhale",
                      current_time: Optional[float] = None) -> Tuple[Optional[str], str]:
        """
        Generate a haiku based on atmospheric conditions
        
        Returns (haiku, generation_type) where:
        - haiku: None for contemplative silence, string for haiku
        - generation_type: "neural", "template", or "silence"
        """
        
        # Sense atmospheric conditions
        conditions = self.sense_atmospheric_conditions(
            seed_fragment, breath_phase, current_time
        )
        
        # Decide whether to generate or remain silent
        if not self.should_generate(conditions):
            return None, "silence"  # Contemplative silence
            
        self.last_generation_time = time.time()
        
        try:
            if self.use_neural:
                haiku = self._generate_neural(seed_fragment, conditions)
                return haiku, "neural"
            else:
                haiku = self._generate_template(conditions)
                return haiku, "template"
                
        except Exception as e:
            print(f"🌫️ Generation mist: {e}")
            return None, "error"  # Graceful failure to silence
    
    def _generate_neural(self, seed_fragment: str, conditions: AtmosphericConditions) -> str:
        """Generate using neural model with proper GPU handling"""
        
        if not self.use_neural:
            return self._generate_template(conditions)
            
        # Convert conditions to tensor and move to device
        condition_vector = torch.tensor([conditions.to_condition_vector()], 
                                      dtype=torch.float32, device=DEVICE)
        
        # Start with silence token
        tokens = [self.tokenizer.token_to_id.get("<START>", 1)]
        max_length = 20  # Maximum haiku length in tokens
        
        hidden1, hidden2 = None, None
        
        try:
            for _ in range(max_length):
                input_tokens = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
                
                with torch.no_grad():
                    logits, silence_logits, hidden1, hidden2 = self.model(
                        input_tokens, condition_vector, hidden1, hidden2
                    )
                    
                # Check if model suggests silence
                silence_prob = torch.sigmoid(silence_logits[0, -1]).item()
                if silence_prob > 0.8:  # Strong silence signal
                    break
                    
                # Sample next token with temperature based on atmospheric humidity
                temperature = 0.5 + conditions.humidity * 0.5
                next_logits = logits[0, -1] / temperature
                probs = torch.softmax(next_logits, dim=0)
                next_token = torch.multinomial(probs, 1).item()
                
                # Stop at end token
                if next_token == self.tokenizer.token_to_id.get("<END>", 2):
                    break
                    
                tokens.append(next_token)
            
            # Decode and format
            text = self.tokenizer.decode(tokens[1:])  # Skip START token
            
            # Simple line breaking for haiku format
            words = text.split()
            if len(words) >= 3:
                # Attempt 3-line structure
                third = len(words) // 3
                haiku = f"{' '.join(words[:third])}\n{' '.join(words[third:2*third])}\n{' '.join(words[2*third:])}"
            else:
                haiku = text
                
            return haiku
            
        except Exception as e:
            print(f"🌫️ Neural generation error: {e}")
            # Fallback to template generation
            return self._generate_template(conditions)
    
    def _generate_template(self, conditions: AtmosphericConditions) -> str:
        """Generate using template system"""
        return self.template_generator.generate_haiku(conditions)

def interactive_test_mode(meadow: HaikuMeadow):
    """Interactive testing mode for the haiku meadow with logging"""
    
    print("\n🌸 HaikuMeadow Interactive Test Mode")
    print("   Enter seed fragments to inspire haiku generation")
    print("   Commands: 'quit' to exit, 'stats' for model info, 'silence' to test silence")
    print("            'log' for session summary")
    print("   Just press Enter for random atmospheric generation\n")
    
    # Initialize logger
    logger = HaikuLogger()
    logger.log_event("test_mode_start", {
        "mode": "neural" if meadow.use_neural else "template",
        "model_type": getattr(meadow.model, 'model_type', 'template') if meadow.model else 'template'
    })
    
    generation_count = 0
    
    try:
        while True:
            try:
                user_input = input("🌿 Seed fragment (or command): ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'stats':
                    mode = "neural" if meadow.use_neural else "template"
                    print(f"   Mode: {mode}")
                    print(f"   Vocabulary size: {meadow.tokenizer.vocab_size}")
                    if meadow.use_neural and meadow.model:
                        params = meadow.model.count_parameters()
                        model_type = meadow.model.model_type
                        print(f"   Model: {model_type} ({params:,} parameters)")
                    print(f"   Generations this session: {generation_count}")
                    continue
                elif user_input.lower() == 'silence':
                    # Force silence test
                    print("   [contemplative silence]")
                    logger.log_haiku(None, "forced_silence", 
                                   meadow.sense_atmospheric_conditions(""), "forced_silence")
                    continue
                elif user_input.lower() == 'log':
                    logger.session_summary()
                    continue
                    
                # Generate haiku
                conditions = meadow.sense_atmospheric_conditions(user_input)
                haiku, generation_type = meadow.generate_haiku(user_input)
                generation_count += 1
                
                # Determine generation type
                if haiku:
                    print(f"\n🌸 Generated haiku ({generation_type}):")
                    for line in haiku.split('\n'):
                        print(f"      {line}")
                    print(f"   🌤️  Atmosphere: {conditions.season.value}, {conditions.time_of_day.value}")
                    print(f"       Temperature: {conditions.temperature:.1f}, Humidity: {conditions.humidity:.1f}")
                    print()
                else:
                    generation_type = "silence"
                    print("   [contemplative silence]")
                    print(f"   🌤️  Atmosphere: {conditions.season.value}, {conditions.time_of_day.value}")
                    print()
                
                # Log the generation
                logger.log_haiku(haiku, user_input, conditions, generation_type)
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"   Error: {e}")
                logger.log_event("error", {"error": str(e), "input": user_input})
        
        print("\n🌙 Leaving contemplative test mode...")
        logger.log_event("test_mode_end", {"generations": generation_count})
        logger.session_summary()
        
    except Exception as e:
        print(f"🌫️ Test mode error: {e}")
        logger.log_event("test_mode_error", {"error": str(e)})

# Testing and demonstration
async def test_haiku_generation():
    """Test the haiku generation system"""
    
    print("🌸 Testing HaikuMeadow Generation")
    
    # Force template mode for safe testing
    meadow = HaikuMeadow(force_template_mode=True)
    
    test_fragments = [
        "morning mist gathering",
        "breath between heartbeats", 
        "gentle autumn contemplation",
        "winter silence deepening",
        "patterns emerging in twilight"
    ]
    
    print("\n🌿 Testing atmospheric generation:")
    
    for fragment in test_fragments:
        print(f"\n   Seed: '{fragment}'")
        
        haiku, generation_type = meadow.generate_haiku(fragment)
        
        if haiku:
            print(f"   Generated ({generation_type}):")
            for line in haiku.split('\n'):
                print(f"      {line}")
        else:
            print(f"   Response: [contemplative silence] ({generation_type})")
    
    print("\n🌊 Testing silence probability:")
    
    generation_count = 0
    silence_count = 0
    
    for i in range(10):
        haiku, gen_type = meadow.generate_haiku("gentle breath")
        if haiku:
            generation_count += 1
        else:
            silence_count += 1
    
    print(f"   Generations: {generation_count}, Silences: {silence_count}")
    print(f"   Silence ratio: {silence_count / 10:.1%}")
    print(f"   All generation types: template (safe mode)")

def check_system_capabilities():
    """Check and report system capabilities for haiku generation"""
    
    print("🌸 HaikuMeadow System Check")
    print("=" * 40)
    
    # PyTorch availability
    if TORCH_AVAILABLE:
        print("✅ PyTorch available")
        print(f"   Version: {torch.__version__}")
        
        # Device info
        if DEVICE.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_memory_free = torch.cuda.mem_get_info()[0] / 1e9
            print(f"🚀 GPU: {gpu_name}")
            print(f"   Total memory: {gpu_memory:.1f}GB")
            print(f"   Free memory: {gpu_memory_free:.1f}GB")
            
            # Recommend batch size based on memory
            if gpu_memory < 2:
                recommended_batch = 4
            elif gpu_memory < 4:
                recommended_batch = 8
            elif gpu_memory < 8:
                recommended_batch = 16
            else:
                recommended_batch = 32
            print(f"   Recommended batch size: {recommended_batch}")
        else:
            print("💻 Device: CPU")
            print("   Consider installing CUDA for faster training")
    else:
        print("❌ PyTorch not available")
        print("   Install with: pip install torch")
        
    print()
    
    # Storage info
    import shutil
    current_dir = Path.cwd()
    disk_usage = shutil.disk_usage(current_dir)
    free_gb = disk_usage.free / 1e9
    print(f"💾 Storage (current directory): {free_gb:.1f}GB free")
    
    # Model size estimate
    estimated_model_size_mb = 600000 * 4 / 1e6  # ~600k params * 4 bytes
    print(f"📊 Estimated piko-model size: {estimated_model_size_mb:.1f}MB")
    
    if free_gb > 1:
        print("✅ Sufficient storage for model")
    else:
        print("⚠️  Low storage - model should still fit")
    
    print("=" * 40)

def main():
    """Main entry point with command line interface"""
    
    parser = argparse.ArgumentParser(description="HaikuMeadowLib Piko-LLM")
    parser.add_argument("--train", action="store_true", help="Train the piko-LLM model")
    parser.add_argument("--test", action="store_true", help="Interactive test mode (CPU-safe)")
    parser.add_argument("--template-only", action="store_true", help="Force template mode (no neural model)")
    parser.add_argument("--check", action="store_true", help="Check system capabilities")
    parser.add_argument("--training-data", type=str, default="haiku_training_material.json",
                       help="Path to training data JSON file")
    parser.add_argument("--model-path", type=str, default="piko_haiku_model.pt",
                       help="Path to save/load model")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    
    args = parser.parse_args()
    
    if args.check:
        check_system_capabilities()
        return
    
    if args.train:
        training_data_path = Path(args.training_data)
        model_save_path = Path(args.model_path)
        
        if not training_data_path.exists():
            print(f"❌ Training data not found: {training_data_path}")
            print("   Run ingest.py first to create training material")
            return
        
        # CPU safety warning for training
        if DEVICE and DEVICE.type == "cpu":
            print("🧘 CPU Training Mode")
            print("   This will be slower but safer for your system")
            print("   The model will be automatically reduced to femto-size")
            print("   Consider using --template-only for instant testing instead")
            print()
            
            response = input("Continue with CPU training? (y/N): ").strip().lower()
            if response != 'y':
                print("💡 Try: python generator.py --template-only --test")
                return
        
        success = train_piko_model(
            training_data_path=training_data_path,
            model_save_path=model_save_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        if success:
            print(f"\n🌸 Training complete! Model saved to {model_save_path}")
            print(f"   Use --test to try the trained model")
        
    elif args.test:
        # CPU-safe test mode with option to use trained models
        if args.template_only:
            print("🌿 Starting template-only test mode (CPU safe)")
            meadow = HaikuMeadow(force_template_mode=True)
        else:
            model_path = Path(args.model_path) if Path(args.model_path).exists() else None
            
            if DEVICE and DEVICE.type == "cpu" and model_path:
                print("🧘 CPU Test Mode - attempting to load trained femto-model")
                print("   (Using your trained model - should be safe after successful training)")
                meadow = HaikuMeadow(model_path, force_template_mode=False)  # Allow neural model
            elif DEVICE and DEVICE.type == "cpu":
                print("🧘 CPU Test Mode - using template generation for safety")
                print("   (No trained model found)")
                print("   Use --template-only to skip this message")
                meadow = HaikuMeadow(model_path, force_template_mode=True)
            else:
                meadow = HaikuMeadow(model_path)
        
        interactive_test_mode(meadow)
        
    else:
        # Default demo mode (safe)
        print("🌸 Running safe demo mode...")
        print("   Using template generation to prevent crashes")
        import asyncio
        asyncio.run(test_haiku_generation())

if __name__ == "__main__":
    main()
