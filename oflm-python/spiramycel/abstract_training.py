#!/usr/bin/env python3
"""
Abstract Training for Spiramycel Neural Model

Fast training using pre-generated abstract data files,
matching the performance of ecological training.
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import time
from datetime import datetime
import shutil
import re
import glob
import os

# Import from existing modules
from glyph_codec import SpiramycelGlyphCodec
from gpu_breathing import contemplative_pause
from spore_map import Season
from neural_trainer import (
    SpiramycelDataset,
    NetworkConditions,
    SpiramycelNeuralModel,
    load_spiramycel_parameters,
    PAD_TOKEN,  # re-exported from token_constants via neural_trainer
    START_TOKEN,
    END_TOKEN,
)

try:
    from .training_utils import determine_model_scale_and_folders, discover_training_data, set_deterministic
except ImportError:
    from training_utils import determine_model_scale_and_folders, discover_training_data, set_deterministic

# Ensure reproducibility
set_deterministic(42)

class AbstractDataset(Dataset):
    """Dataset for abstract spore echoes (mirrors EcologicalDataset)"""
    
    def __init__(self, jsonl_file: str, codec: SpiramycelGlyphCodec):
        self.codec = codec
        self.samples = []
        
        print(f"🔬 Loading abstract data from {jsonl_file}...")
        
        with open(jsonl_file, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    if line.strip():
                        data = json.loads(line)
                        self.samples.append(data)
                except Exception as e:
                    print(f"⚠ Skipping line {line_num}: {e}")
        
        print(f"✓ Loaded {len(self.samples)} abstract spore echoes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Fixed: Handle both sensor_readings and sensor_deltas formats (o3's data consistency fix)
        conditions_data = sample['conditions']
        
        # Try sensor_deltas first (new o3 format), fall back to sensor_readings (legacy)
        if 'sensor_deltas' in conditions_data:
            sensor_deltas = conditions_data['sensor_deltas']
            # Convert deltas back to absolute values for NetworkConditions
            conditions = NetworkConditions(
                latency=sensor_deltas.get('latency', 0.1),  # latency is absolute
                voltage=3.3 + sensor_deltas.get('voltage', 0.0),  # Convert delta to absolute
                temperature=25.0 + sensor_deltas.get('temperature', 0.0),  # Convert delta to absolute
                error_rate=sensor_deltas.get('error_rate', 0.02),  # error_rate is absolute
                bandwidth=0.8 + sensor_deltas.get('bandwidth', 0.0),  # Convert delta to absolute
            )
        elif 'sensor_readings' in conditions_data:
            # Legacy format support
            sensor_readings = conditions_data['sensor_readings']
            conditions = NetworkConditions(
                latency=sensor_readings.get('latency', 0.1),
                voltage=sensor_readings.get('voltage', 3.3),
                temperature=sensor_readings.get('temperature', 25.0),
                error_rate=sensor_readings.get('error_rate', 0.02),
                bandwidth=sensor_readings.get('bandwidth', 0.8),
            )
        else:
            # Fallback with default values
            print(f"⚠ Warning: No sensor data found in sample {idx}, using defaults")
            conditions = NetworkConditions(
                latency=0.1, voltage=3.3, temperature=25.0, error_rate=0.02, bandwidth=0.8
            )
        
        # Get glyph sequence 
        glyph_sequence = sample['repair_action']['glyph_sequence']
        
        # Add START and END tokens
        glyph_tokens = [START_TOKEN] + glyph_sequence + [END_TOKEN]
        
        # Pad to max_length of 16
        max_length = 16
        if len(glyph_tokens) < max_length:
            glyph_tokens.extend([PAD_TOKEN] * (max_length - len(glyph_tokens)))
        else:
            glyph_tokens = glyph_tokens[:max_length]
        
        # Create input/target sequences
        input_tokens = torch.tensor(glyph_tokens[:-1], dtype=torch.long)
        target_tokens = torch.tensor(glyph_tokens[1:], dtype=torch.long)
        
        condition_tensor = torch.tensor(conditions.to_condition_vector(), dtype=torch.float32)
        
        # Effectiveness as supervision signal
        effectiveness = torch.tensor(sample['repair_action']['effectiveness'], dtype=torch.float32)
        
        return input_tokens, target_tokens, condition_tensor, effectiveness

def train_abstract_model(data_file: str = "training_scenarios/abstract_large.jsonl",
                        config: Dict = None,
                        epochs: int = None):
    """Train Spiramycel on abstract data (mirrors train_ecological_model)"""
    
    print("🔬 Abstract Spiramycel Training")
    print("=" * 50)
    
    # Load configuration
    if config is None:
        config = load_spiramycel_parameters("abstract")
    
    # Get training parameters from config (ensure numeric types)
    training_config = config.get('training', {})
    epochs = epochs or int(training_config.get('epochs', 15))
    batch_size = int(training_config.get('batch_size', 4))
    learning_rate = float(training_config.get('learning_rate', 0.0008))
    weight_decay = float(training_config.get('weight_decay', 2e-5))
    gradient_clip_norm = float(training_config.get('gradient_clip_norm', 0.8))
    
    print(f"🔧 Using abstract paradigm configuration")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    
    # Initialize codec
    codec = SpiramycelGlyphCodec()
    print(f"📝 Glyph vocabulary: {len(codec.glyphs)} symbols")
    
    # Load abstract dataset
    dataset = AbstractDataset(data_file, codec)
    
    if len(dataset) == 0:
        print("❌ No training data loaded!")
        return None
    
    # Use SpiramycelNeuralModel with configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    model = SpiramycelNeuralModel(config=config, paradigm="abstract").to(device)
    print(f"🚀 Using device: {device} ({'GPU-accelerated!' if device.type == 'cuda' else 'CPU fallback'})")
    
    # Print actual model type that was selected
    print(f"🧠 Model: {model.model_type} ({model.count_parameters():,} parameters)")
    
    # Training setup using configuration
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=min(2, os.cpu_count() or 0))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Loss functions matching neural_trainer.py (PAD_TOKEN is ignore index)
    glyph_criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    effectiveness_criterion = nn.MSELoss()
    silence_criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    print(f"🚀 Training for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_glyph_loss = 0.0
        epoch_effectiveness_loss = 0.0
        epoch_silence_loss = 0.0
        num_batches = 0
        
        for batch_idx, (input_tokens, target_tokens, condition_tensor, effectiveness) in enumerate(dataloader):
            input_tokens = input_tokens.to(device)
            target_tokens = target_tokens.to(device)
            condition_tensor = condition_tensor.to(device)
            effectiveness = effectiveness.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            glyph_logits, eff_logits, silence_logits, _, _, _ = model(input_tokens, condition_tensor)
            
            # Calculate losses
            glyph_loss = glyph_criterion(
                glyph_logits.reshape(-1, glyph_logits.size(-1)),
                target_tokens.reshape(-1)
            )
            
            effectiveness_loss = effectiveness_criterion(
                eff_logits.squeeze(-1).mean(dim=1),
                effectiveness
            )
            
            # Silence loss – align with ecological training (first timestep only)
            silence_targets = (effectiveness < 0.3).float()
            first_silence_logits = silence_logits[:, 0].squeeze(-1)
            silence_loss = silence_criterion(first_silence_logits, silence_targets)
            
            # Combined loss
            total_loss = glyph_loss + 0.5 * effectiveness_loss + 0.3 * silence_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping from configuration
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            
            optimizer.step()
            
            # Accumulate losses
            epoch_glyph_loss += glyph_loss.item()
            epoch_effectiveness_loss += effectiveness_loss.item()
            epoch_silence_loss += silence_loss.item()
            num_batches += 1
            
            # Adaptive contemplative breathing based on GPU stress (skip for models < 1M)
            if model.count_parameters() >= 1000000:  # Only for large models (1M+)
                contemplative_pause("abstract_training")
        
        # Calculate average losses
        avg_glyph_loss = epoch_glyph_loss / num_batches if num_batches > 0 else 0.0
        avg_effectiveness_loss = epoch_effectiveness_loss / num_batches if num_batches > 0 else 0.0
        avg_silence_loss = epoch_silence_loss / num_batches if num_batches > 0 else 0.0
        
        print(f"Epoch {epoch+1:2d}: Glyph {avg_glyph_loss:.3f} | "
              f"Effectiveness {avg_effectiveness_loss:.4f} | "
              f"Silence {avg_silence_loss:.4f}")
    
    training_time = time.time() - start_time
    print(f"⏱ Training completed in {training_time:.1f} seconds")
    
    # Auto-detect model scale and use appropriate scale-specific folder
    scale_name, scale_model_dir, scale_suffix = determine_model_scale_and_folders(model, "abstract")
    models_dir = Path(scale_model_dir)
    models_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = models_dir / f"abstract_spiramycel_{timestamp}.pt"
    
    # Embed scale metadata into state_dict for future loaders
    state = model.state_dict()
    state["_meta"] = {"scale": scale_name}
    torch.save(state, model_path)
    print(f"💾 Model saved to {model_path}")
    
    # Latest model link using scale-specific naming
    latest_model_name = f"abstract_spiramycel_latest.pt"
    latest_path = models_dir / latest_model_name
    try:
        if latest_path.exists():
            latest_path.unlink()
        # On Windows, copy instead of symlink
        shutil.copy2(model_path, latest_path)
        print(f"📎 Latest model link: {latest_path}")
    except Exception as e:
        print(f"⚠ Could not create latest link: {e}")
    
    # Also save with standard naming for controlled comparison compatibility
    standard_names = {
        "abstract_calm_model.pt": "calm",
        "abstract_chaotic_model.pt": "chaotic"
    }
    
    # Determine model type based on config or training characteristics
    model_type = config.get('model_type', 'calm')  # Default to calm if not specified
    for standard_name, model_variant in standard_names.items():
        if model_variant in str(model_path).lower() or model_type == model_variant:
            standard_path = models_dir / standard_name
            try:
                shutil.copy2(model_path, standard_path)
                print(f"📋 Standard model link: {standard_path}")
                break
            except Exception as e:
                print(f"⚠ Could not create standard link: {e}")
    
    # Test abstract inference
    print("\n🔬 Testing abstract inference:")
    model.eval()
    with torch.no_grad():
        # Create a test scenario - urban fiber thermal overload
        test_conditions = NetworkConditions(
            latency=0.3,      # High latency from overheating
            voltage=2.9,      # Low voltage
            temperature=45.0, # High temperature 
            error_rate=0.15,  # High error rate
            bandwidth=0.2     # Low bandwidth from congestion
        )
        
        test_tensor = torch.tensor([test_conditions.to_condition_vector()], dtype=torch.float32).to(device)
        
        # Generate abstract repair sequence
        generated_tokens = [START_TOKEN]
        hidden1, hidden2 = None, None
        
        for step in range(10):  # Generate up to 10 tokens
            input_tensor = torch.tensor([generated_tokens[-1:]], dtype=torch.long).to(device)
            glyph_logits, _, silence_logits, hidden1, hidden2, hidden3 = model(input_tensor, test_tensor, hidden1, hidden2)
            
            # Check if we should use silence
            silence_prob = torch.sigmoid(silence_logits[0, -1]).item()
            
            if silence_prob > 0.7:  # High silence threshold
                print(f"   Step {step}: 🤫 (silence probability: {silence_prob:.2f})")
                break
            
            # Sample next token
            probs = torch.softmax(glyph_logits[0, -1], dim=0)
            next_token = torch.multinomial(probs, 1).item()
            
            if next_token == END_TOKEN:
                break
                
            generated_tokens.append(next_token)
            
            # Decode and display
            glyph_name = codec.decode_glyph(next_token)
            print(f"   Step {step}: {glyph_name} (0x{next_token:02X})")
    
    return model_path

def main():
    """Main training function"""
    
    # Dynamically discover abstract training data files
    print("🔍 Discovering abstract training data...")
    available_files = discover_training_data("abstract", "training_scenarios")
    
    if not available_files:
        print("❌ No abstract training data found!")
        print("   Expected files in training_scenarios/:")
        print("     abstract_YYYYMMDD_*.jsonl  (dated files - preferred)")
        print("     abstract_*.jsonl           (undated files)")
        print("   Run: python generate_abstract_data.py")
        return
    
    # Show discovered files (first 3 for brevity)
    print(f"📁 Found {len(available_files)} abstract dataset(s):")
    for i, file_path in enumerate(available_files[:3]):
        file_size = file_path.stat().st_size / (1024 * 1024)  # MB
        print(f"   {i+1}. {file_path.name} ({file_size:.1f} MB)")
    if len(available_files) > 3:
        print(f"   ... and {len(available_files) - 3} more files")
    
    # Use most recent (first in sorted list)
    data_file = str(available_files[0])
    print(f"\n📊 Using most recent dataset: {available_files[0].name}")
    
    # Show dataset size info
    try:
        file_size = available_files[0].stat().st_size / (1024 * 1024)  # MB
        print(f"   File size: {file_size:.1f} MB")
    except Exception as e:
        print(f"   ⚠ Could not get file size: {e}")
    
    # Train abstract model
    model_path = train_abstract_model(
        data_file=data_file,
        epochs=15
    )
    
    if model_path:
        print(f"\n✅ Abstract Spiramycel training complete!")
        print(f"🔬 Ready for contemplative inference")
        print(f"📁 Model: {model_path}")

if __name__ == "__main__":
    main() 