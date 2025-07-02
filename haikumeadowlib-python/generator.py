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

# Try to import YAML for parameter loading
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    yaml = None
    YAML_AVAILABLE = False
    print("⚠️  PyYAML not available - using default parameters")

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

def load_model_parameters(model_name: str = None, param_file: str = "lm_parameters.yml") -> Dict:
    """Load model parameters from YAML configuration file"""
    
    # Default parameters (fallback if YAML not available)
    default_params = {
        "piko": {
            "description": "Default piko-scale model",
            "target_device": "cpu",
            "parameter_count": 35,
            "embed_dim": 32,
            "hidden_dim": 64,
            "num_layers": 1,
            "condition_dim": 8,
            "vocab_size": 2000,
            "max_sequence_length": 32,
            "training": {
                "epochs": 8,
                "batch_size": 2,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "gradient_clip_norm": 0.5
            },
            "memory_limit_mb": 50,
            "force_cpu_mode": True,
            "generation": {
                "temperature_base": 0.7,
                "silence_probability": 0.3,
                "rate_limit_seconds": 5.0
            }
        },
        "nano": {
            "description": "Default nano-scale model", 
            "target_device": "gpu",
            "parameter_count": 600,
            "embed_dim": 128,
            "hidden_dim": 256,
            "num_layers": 2,
            "condition_dim": 8,
            "vocab_size": 2000,
            "max_sequence_length": 48,
            "training": {
                "epochs": 20,
                "batch_size": 16,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "gradient_clip_norm": 1.0
            },
            "memory_limit_mb": 500,
            "force_cpu_mode": False,
            "generation": {
                "temperature_base": 0.8,
                "silence_probability": 0.25,
                "rate_limit_seconds": 3.0
            }
        }
    }
    
    # Try to load from YAML file
    if YAML_AVAILABLE:
        param_path = Path(param_file)
        if param_path.exists():
            try:
                with open(param_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                # Auto-detect model if not specified
                if model_name is None:
                    # Choose based on device capability
                    if DEVICE and DEVICE.type == "cuda":
                        model_name = config.get("default_model", "nano")
                    else:
                        model_name = "piko"  # Safe default for CPU
                else:
                    model_name = model_name.lower()
                
                # Load model-specific parameters
                if "models" in config and model_name in config["models"]:
                    params = config["models"][model_name].copy()
                    
                    # Add global config sections
                    if "paths" in config:
                        params["paths"] = config["paths"]
                    if "data" in config:
                        params["data"] = config["data"]
                    if "logging" in config:
                        params["logging"] = config["logging"]
                    if "atmospheric" in config:
                        params["atmospheric"] = config["atmospheric"]
                        
                    print(f"📋 Loaded {model_name} parameters from {param_file}")
                    print(f"   Model: {params.get('description', 'Unknown')}")
                    print(f"   Target: ~{params.get('parameter_count', '?')}k parameters")
                    return params
                else:
                    print(f"⚠️  Model '{model_name}' not found in {param_file}, using defaults")
                    
            except Exception as e:
                print(f"🌫️ Error loading {param_file}: {e}")
                print("   Using default parameters")
        else:
            print(f"⚠️  Parameter file {param_file} not found, using defaults")
    
    # Fallback to defaults
    if model_name is None:
        # Auto-select based on device
        if DEVICE and DEVICE.type == "cuda":
            model_name = "nano" 
        else:
            model_name = "piko"
    
    model_name = model_name.lower()
    if model_name in default_params:
        print(f"📋 Using default {model_name} parameters")
        return default_params[model_name]
    else:
        print(f"⚠️  Unknown model '{model_name}', using piko defaults")
        return default_params["piko"]

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
            # Ensure logs directory exists
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            log_path = logs_dir / f"haiku_session_{timestamp}.jsonl"
        
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
    Configurable neural haiku generator with adaptive sizing
    
    Now uses parameters from lm_parameters.yml:
    - Piko mode: ~35k parameters (CPU-optimized)
    - Nano mode: ~600k parameters (GPU-optimized)
    """
    
    def __init__(self, 
                 vocab_size: int = 2000, 
                 embed_dim: int = None,
                 hidden_dim: int = None,
                 num_layers: int = None,
                 condition_dim: int = 8,
                 force_cpu_mode: bool = None,
                 config: Dict = None):
        
        if TORCH_AVAILABLE:
            super().__init__()
        
        # Load parameters from config if provided
        if config:
            self.embed_dim = embed_dim or config.get("embed_dim", 32)
            self.hidden_dim = hidden_dim or config.get("hidden_dim", 64)
            self.num_layers = num_layers or config.get("num_layers", 1)
            self.condition_dim = condition_dim or config.get("condition_dim", 8)
            self.vocab_size = vocab_size or config.get("vocab_size", 2000)
            force_cpu_mode = force_cpu_mode if force_cpu_mode is not None else config.get("force_cpu_mode", False)
            
            # Determine model type from config
            target_device = config.get("target_device", "cpu")
            param_count = config.get("parameter_count", 35)
            
            if target_device == "cpu" or param_count < 100:
                self.model_type = "piko"
                model_desc = f"piko-model (CPU optimized, ~{param_count}k parameters)"
            else:
                self.model_type = "nano"
                model_desc = f"nano-model (GPU optimized, ~{param_count}k parameters)"
                
        else:
            # Fallback to original adaptive sizing if no config
            self.vocab_size = vocab_size
            self.condition_dim = condition_dim
            
            if not TORCH_AVAILABLE or DEVICE.type == "cpu" or force_cpu_mode:
                self.embed_dim = embed_dim or 32
                self.hidden_dim = hidden_dim or 64
                self.num_layers = num_layers or 1
                self.model_type = "piko"
                model_desc = "piko-model (CPU optimized, ~35k parameters)"
            else:
                self.embed_dim = embed_dim or 128
                self.hidden_dim = hidden_dim or 256
                self.num_layers = num_layers or 2
                self.model_type = "nano"
                model_desc = "nano-model (GPU optimized, ~600k parameters)"
        
        print(f"🦠 Using {model_desc}")
        
        if TORCH_AVAILABLE:
            # Token embedding
            self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
            
            # Atmospheric condition embedding
            self.condition_proj = nn.Linear(self.condition_dim, self.embed_dim)
            
            # GRU layers based on num_layers configuration
            if self.num_layers == 1:
                self.gru1 = nn.GRU(self.embed_dim, self.hidden_dim, batch_first=True)
                self.gru2 = None  # Single layer for memory efficiency
            else:
                # Multi-layer configuration (typically 2 layers)
                self.gru1 = nn.GRU(self.embed_dim, self.hidden_dim, batch_first=True)
                self.gru2 = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
            
            # Output projection
            self.output_proj = nn.Linear(self.hidden_dim, self.vocab_size)
            
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

def train_model(training_data_path: Path, 
                model_save_path: Path,
                config: Dict = None,
                model_name: str = "piko",
                epochs: int = None,
                batch_size: int = None,
                learning_rate: float = None):
    """Train the neural model with configurable parameters"""
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available - cannot train neural model")
        return False
    
    # Load configuration if not provided
    if config is None:
        config = load_model_parameters(model_name)
    
    # Extract training parameters from config
    training_config = config.get("training", {})
    epochs = epochs or training_config.get("epochs", 10)
    batch_size = batch_size or training_config.get("batch_size", 16)
    learning_rate = learning_rate or training_config.get("learning_rate", 0.001)
    weight_decay = training_config.get("weight_decay", 0.0001)
    gradient_clip_norm = training_config.get("gradient_clip_norm", 0.5)
    
    print(f"🌸 Starting {model_name} model training")
    print(f"   Training data: {training_data_path}")
    print(f"   Model save path: {model_save_path}")
    print(f"   Device: {DEVICE}")
    print(f"   Configuration: {config.get('description', 'Unknown')}")
    
    # Device-specific optimizations
    target_device = config.get("target_device", "cpu")
    memory_limit = config.get("memory_limit_mb", 100)
    
    if DEVICE.type == "cpu" or target_device == "cpu":
        # CPU optimizations from config
        print(f"   🧘 CPU mode: optimized for {memory_limit}MB memory limit")
        print(f"   💡 This will be slower but safer for your system")
    elif DEVICE.type == "cuda":
        # GPU memory optimization
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory_gb < 4:
            batch_size = min(batch_size, 8)
            print(f"   Reduced batch size to {batch_size} for limited GPU memory")
        torch.cuda.empty_cache()
    
    print(f"   Final settings: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
    
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
    
    # Initialize model with configuration
    model = PikoHaikuModel(
        config=config,
        vocab_size=tokenizer.vocab_size
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Learning rate scheduler for better convergence
    scheduler_patience = training_config.get("scheduler_patience", 2)
    scheduler_factor = training_config.get("scheduler_factor", 0.7)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=scheduler_patience, factor=scheduler_factor
    )
    
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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
                    
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
                best_model_path = model_save_path.parent / f"{model_name}_model_best.pt"
                torch.save(model.state_dict(), best_model_path)
                print(f"💎 New best model saved: {best_model_path}")
            
            # Save checkpoint more frequently for CPU (in case of crash)
            checkpoint_freq = config.get("logging", {}).get("checkpoint_frequency", 3)
            if DEVICE.type == "cpu" or (epoch + 1) % checkpoint_freq == 0 or epoch == epochs - 1:
                checkpoint_path = model_save_path.parent / f"{model_name}_model_epoch_{epoch+1}.pt"
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
        interrupt_path = model_save_path.parent / f"{model_name}_model_interrupted.pt"
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
    
    def __init__(self, model_path: Optional[Path] = None, force_template_mode: bool = False, 
                 config: Dict = None, model_name: str = None):
        
        self.tokenizer = SimpleTokenizer()
        self.last_generation_time = 0.0
        
        # Load configuration if not provided
        if config is None and not force_template_mode:
            config = load_model_parameters(model_name)
        
        # Set generation parameters from config
        if config and "generation" in config:
            gen_config = config["generation"]
            self.silence_probability = gen_config.get("silence_probability", 0.3)
            self.rate_limit = gen_config.get("rate_limit_seconds", 5.0)
            self.temperature_base = gen_config.get("temperature_base", 0.7)
        else:
            self.silence_probability = 0.3
            self.rate_limit = 5.0
            self.temperature_base = 0.7
        
        # Initialize use_neural flag first
        self.use_neural = False
        
        # Memory safety: use template mode by default on CPU to prevent crashes
        if force_template_mode:
            print("🌿 Using template mode (CPU safe, no model loading)")
            self.model = None
            self.use_neural = False
        elif TORCH_AVAILABLE and model_path and model_path.exists():
            try:
                print(f"🌸 Attempting to load neural model...")
                
                # Use configured model sizing
                self.model = PikoHaikuModel(
                    config=config,
                    vocab_size=self.tokenizer.vocab_size
                )
                
                # Load model with proper device handling and memory monitoring
                if DEVICE and DEVICE.type == "cuda":
                    self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                    self.model = self.model.to(DEVICE)
                    self.use_neural = True  # Enable neural mode on successful GPU load
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
                    # Final check - ensure neural mode is enabled if model loaded successfully
                    if not hasattr(self, 'use_neural') or not self.use_neural:
                        self.use_neural = True
                    
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
        if current_time - self.last_generation_time < self.rate_limit:
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
                temperature = self.temperature_base + conditions.humidity * 0.3
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
    parser.add_argument("--train", action="store_true", help="Train the neural model")
    parser.add_argument("--test", action="store_true", help="Interactive test mode (CPU-safe)")
    parser.add_argument("--template-only", action="store_true", help="Force template mode (no neural model)")
    parser.add_argument("--check", action="store_true", help="Check system capabilities")
    parser.add_argument("--model", type=str, choices=["piko", "nano"], default=None,
                       help="Model size: piko (CPU, ~35k params) or nano (GPU, ~600k params)")
    parser.add_argument("--training-data", type=str, default=None,
                       help="Path to training data JSON file")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to save/load model")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size (overrides config)")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate (overrides config)")
    
    args = parser.parse_args()
    
    if args.check:
        check_system_capabilities()
        return
    
    if args.train:
        # Load model configuration
        model_name = args.model or "piko"  # Default to piko for safety
        config = load_model_parameters(model_name)
        
        # Use config paths if not overridden by command line
        training_data_path = Path(args.training_data or config.get("data", {}).get("training_data_path", "training/haiku_training_material.json"))
        
        if args.model_path:
            model_save_path = Path(args.model_path)
        else:
            # Use config path
            paths_config = config.get("paths", {})
            model_save_path = Path(paths_config.get(model_name, {}).get("save_path", f"model/{model_name}/{model_name}_haiku_model.pt"))
        
        # Ensure model directory exists
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not training_data_path.exists():
            print(f"❌ Training data not found: {training_data_path}")
            print("   Run ingest.py first to create training material")
            return
        
        # Train model with configuration
        print(f"🌸 Training {model_name} model with configuration-based parameters")
        
        success = train_model(
            training_data_path=training_data_path,
            model_save_path=model_save_path,
            config=config,
            model_name=model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        if success:
            print(f"\n🌸 Training complete! Model saved to {model_save_path}")
            print(f"   Use --test --model {model_name} to try the trained model")
        
    elif args.test:
        # Load model configuration for testing
        model_name = args.model  # Can be None for auto-detection
        
        if args.template_only:
            print("🌿 Starting template-only test mode (CPU safe)")
            meadow = HaikuMeadow(force_template_mode=True)
        else:
            # Load configuration
            config = load_model_parameters(model_name)
            model_name = model_name or ("piko" if DEVICE and DEVICE.type == "cpu" else "nano")
            
            # Determine model path
            if args.model_path:
                model_path_obj = Path(args.model_path)
            else:
                # Use config path
                paths_config = config.get("paths", {})
                model_path_str = paths_config.get(model_name, {}).get("save_path", f"model/{model_name}/{model_name}_haiku_model.pt")
                model_path_obj = Path(model_path_str)
            
            # Ensure model directory exists
            model_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            model_path = model_path_obj if model_path_obj.exists() else None
            
            print(f"🌸 Starting {model_name} model test mode")
            meadow = HaikuMeadow(model_path, config=config, model_name=model_name)
        
        interactive_test_mode(meadow)
        
    else:
        # Default demo mode (safe)
        print("🌸 Running safe demo mode...")
        print("   Using template generation to prevent crashes")
        import asyncio
        asyncio.run(test_haiku_generation())

if __name__ == "__main__":
    main()
