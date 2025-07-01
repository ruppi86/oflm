#!/usr/bin/env python3
"""
Ecological Training for Spiramycel Neural Model

Trains on realistic ecological spore echoes with multi-generational 
bioregional patterns instead of abstract scenarios.
"""

import json
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import time
from datetime import datetime
import shutil
import re
import glob

# Import from existing modules with token constants
from glyph_codec import SpiramycelGlyphCodec
from gpu_breathing import contemplative_pause
from neural_trainer import SpiramycelDataset, NetworkConditions, SpiramycelNeuralModel, START_TOKEN, END_TOKEN, PAD_TOKEN, load_spiramycel_parameters

try:
    from .training_utils import determine_model_scale_and_folders, discover_training_data, set_deterministic
except ImportError:
    from training_utils import determine_model_scale_and_folders, discover_training_data, set_deterministic

# Reproducibility
set_deterministic(42)

class EcologicalDataset(Dataset):
    """Dataset for ecological spore echoes with proper token handling"""
    
    def __init__(self, jsonl_file: str, codec: SpiramycelGlyphCodec):
        self.codec = codec
        self.samples = []
        
        print(f"🌱 Loading ecological data from {jsonl_file}...")
        
        # Robust data loading with error handling
        total_lines = 0
        skipped_lines = 0
        
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    total_lines += 1
                    try:
                        if line.strip():
                            data = json.loads(line)
                            
                            # Validate required fields
                            required_fields = ['conditions', 'repair_action']
                            if all(field in data for field in required_fields):
                                self.samples.append(data)
                            else:
                                print(f"⚠ Line {line_num+1}: Missing required fields")
                                skipped_lines += 1
                                
                    except json.JSONDecodeError as e:
                        print(f"⚠ Line {line_num+1}: JSON decode error - {e}")
                        skipped_lines += 1
                    except Exception as e:
                        print(f"⚠ Line {line_num+1}: Unexpected error - {e}")
                        skipped_lines += 1
                        
        except FileNotFoundError:
            print(f"❌ File not found: {jsonl_file}")
            return
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return
        
        success_rate = (total_lines - skipped_lines) / total_lines if total_lines > 0 else 0
        print(f"✓ Loaded {len(self.samples)} ecological spore echoes")
        print(f"  Success rate: {success_rate:.1%} ({skipped_lines} skipped of {total_lines})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert to format similar to SpiramycelDataset
        sensor_readings = sample['conditions']['sensor_readings']
        
        # Create network conditions from ecological data (normalized 0-1)
        conditions = NetworkConditions(
            latency=min(sensor_readings.get('soil_moisture', 0.5), 1.0),          # soil moisture as latency analog
            voltage=min(sensor_readings.get('nutrient_nitrogen', 0.5), 1.0),      # nitrogen as voltage analog  
            temperature=min(sensor_readings.get('temperature', 0.5), 1.0),       # direct temperature mapping
            error_rate=min(1.0 - sensor_readings.get('root_connections', 0.5), 1.0),  # connection health
            bandwidth=min(sensor_readings.get('nutrient_phosphorus', 0.5), 1.0),  # phosphorus as bandwidth
        )
        
        # Get glyph sequence 
        glyph_sequence = sample['repair_action']['glyph_sequence']
        
        # Add START and END tokens using dedicated constants (fixed collision)
        glyph_tokens = [START_TOKEN] + glyph_sequence + [END_TOKEN]
        
        # Pad to max_length of 16 using dedicated PAD token
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

def train_ecological_model(data_file: str = "training_scenarios/ecological_large.jsonl",
                          config: Dict = None,
                          epochs: int = None):
    """Train Spiramycel on ecological data"""
    
    print("🌍 Ecological Spiramycel Training")
    print("=" * 50)
    
    # Load configuration (respect passed config for scale-specific training)
    if config is None:
        config = load_spiramycel_parameters("ecological")
    
    # Get training parameters from config (ensure numeric types)
    training_config = config.get('training', {})
    epochs = epochs or int(training_config.get('epochs', 15))
    batch_size = int(training_config.get('batch_size', 4))
    learning_rate = float(training_config.get('learning_rate', 0.001))
    weight_decay = float(training_config.get('weight_decay', 1e-5))
    gradient_clip_norm = float(training_config.get('gradient_clip_norm', 1.0))
    
    print(f"🔧 Using ecological paradigm configuration")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    
    # Initialize codec
    codec = SpiramycelGlyphCodec()
    print(f"📝 Glyph vocabulary: {len(codec.glyphs)} symbols")
    
    # Load ecological dataset
    dataset = EcologicalDataset(data_file, codec)
    
    if len(dataset) == 0:
        print("❌ No training data loaded!")
        return None
    
    # Use SpiramycelNeuralModel with configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    model = SpiramycelNeuralModel(config=config, paradigm="ecological").to(device)
    print(f"🚀 Using device: {device} ({'GPU-accelerated!' if device.type == 'cuda' else 'CPU fallback'})")
    
    # Print actual model type that was selected
    print(f"🧠 Model: {model.model_type} ({model.count_parameters():,} parameters)")
    
    # Training setup using configuration
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=min(2, os.cpu_count() or 0))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Loss functions matching neural_trainer.py fixes
    glyph_criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)  # Fixed: ignore PAD, not START
    effectiveness_criterion = nn.MSELoss()
    silence_criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    print(f"🚀 Training for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_glyph_loss = 0
        epoch_effectiveness_loss = 0
        epoch_silence_loss = 0
        num_batches = 0
        
        model.train()
        for batch_idx, (input_tokens, target_tokens, conditions, effectiveness) in enumerate(dataloader):
            input_tokens = input_tokens.to(device)
            target_tokens = target_tokens.to(device)
            conditions = conditions.to(device)
            effectiveness = effectiveness.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass using the neural_trainer.py model structure
            glyph_logits, eff_logits, silence_logits, _, _, _ = model(input_tokens, conditions)
            
            # Glyph sequence loss
            glyph_loss = glyph_criterion(glyph_logits.reshape(-1, model.vocab_size), target_tokens.reshape(-1))
            
            # Effectiveness prediction loss (fixed: use final timestep only)
            final_eff_logits = eff_logits[:, -1].squeeze(-1)  # Last timestep only
            eff_loss = effectiveness_criterion(final_eff_logits, effectiveness)
            
            # Silence loss (fixed: only train on first position to avoid broadcasting punishment)
            silence_targets = (effectiveness < 0.3).float()  # Sequence-level target
            first_silence_logits = silence_logits[:, 0].squeeze(-1)  # First timestep only
            silence_loss = silence_criterion(first_silence_logits, silence_targets)
            
            # Combined loss
            total_loss = glyph_loss + 0.5 * eff_loss + 0.3 * silence_loss
            
            total_loss.backward()
            
            # Gradient clipping from configuration
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            
            optimizer.step()
            
            epoch_glyph_loss += glyph_loss.item()
            epoch_effectiveness_loss += eff_loss.item()
            epoch_silence_loss += silence_loss.item()
            num_batches += 1
            
            # Adaptive contemplative breathing based on GPU stress (skip for models < 1M)
            if model.count_parameters() >= 1000000:  # Only for large models (1M+)
                contemplative_pause("ecological_training")
        
        # Print epoch results
        avg_glyph_loss = epoch_glyph_loss / num_batches if num_batches > 0 else 0.0
        avg_effectiveness_loss = epoch_effectiveness_loss / num_batches if num_batches > 0 else 0.0
        avg_silence_loss = epoch_silence_loss / num_batches if num_batches > 0 else 0.0
        
        print(f"Epoch {epoch+1:2d}: Glyph {avg_glyph_loss:.3f} | "
              f"Effectiveness {avg_effectiveness_loss:.4f} | "
              f"Silence {avg_silence_loss:.4f}")
    
    training_time = time.time() - start_time
    print(f"⏱ Training completed in {training_time:.1f} seconds")
    
    # Auto-detect model scale and use appropriate scale-specific folder
    scale_name, scale_model_dir, scale_suffix = determine_model_scale_and_folders(model, "ecological")
    models_dir = Path(scale_model_dir)
    models_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = models_dir / f"ecological_spiramycel_{timestamp}.pt"
    
    state = model.state_dict()
    state["_meta"] = {"scale": scale_name}
    torch.save(state, model_path)
    print(f"💾 Model saved to {model_path}")
    
    # Latest model link using scale-specific naming
    latest_model_name = f"ecological_spiramycel_latest.pt"
    latest_path = models_dir / latest_model_name
    try:
        # Remove old latest if exists, then create new one
        if latest_path.exists():
            latest_path.unlink()
        shutil.copy2(model_path, latest_path)
        print(f"📎 Latest model link: {latest_path}")
    except Exception as e:
        print(f"⚠ Could not create latest link: {e}")
    
    # Also save with standard naming for controlled comparison compatibility
    standard_names = {
        "ecological_calm_model.pt": "calm",
        "ecological_chaotic_model.pt": "chaotic"
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
    
    # Test ecological inference
    print("\n🌿 Testing ecological inference:")
    model.eval()
    with torch.no_grad():
        # Simulate drought stress conditions
        drought_conditions = NetworkConditions(
            latency=0.1,       # low soil moisture  
            voltage=0.2,       # low nitrogen
            temperature=0.8,   # high temperature
            error_rate=0.7,    # poor root connections
            bandwidth=0.3      # low phosphorus
        )
        
        # Create dummy input tokens (using proper tokens) and move to same device as model
        dummy_input = torch.full((1, 15), PAD_TOKEN, dtype=torch.long).to(device)  # 15 = max_length - 1, use PAD_TOKEN  
        dummy_input[0, 0] = START_TOKEN  # Start with START token
        condition_tensor = torch.tensor(drought_conditions.to_condition_vector(), dtype=torch.float32).unsqueeze(0).to(device)
        
        glyph_logits, effectiveness_pred, silence_logits, _, _, _ = model(dummy_input, condition_tensor)
        
        # Decode predictions using fixed logic
        predicted_glyphs = torch.argmax(glyph_logits[0, :3], dim=1).tolist()  # First 3 glyphs
        predicted_effectiveness = effectiveness_pred[0, -1].item()  # Final timestep
        predicted_silence = torch.sigmoid(silence_logits[0, 0]).item()  # First timestep
        
        print(f"   Drought conditions → Glyphs: {predicted_glyphs}")
        print(f"   Predicted effectiveness: {predicted_effectiveness:.3f}")
        print(f"   Silence probability: {predicted_silence:.3f}")
        
        # Decode glyph meanings (with error handling)
        glyph_meanings = []
        for glyph in predicted_glyphs:
            if glyph in codec.glyphs:
                meaning = codec.glyphs[glyph].description
            else:
                meaning = f"unknown_glyph_{glyph:#04x}"
            glyph_meanings.append(meaning)
        
        print(f"   Meaning: {' → '.join(glyph_meanings)}")
    
    return model_path

def main():
    """Main training function"""
    
    # Dynamically discover ecological training data files
    print("🔍 Discovering ecological training data...")
    available_files = discover_training_data("ecological", "training_scenarios")
    
    if not available_files:
        print("❌ No ecological training data found!")
        print("   Expected files in training_scenarios/:")
        print("     ecological_YYYYMMDD_*.jsonl  (dated files - preferred)")
        print("     ecological_*.jsonl           (undated files)")
        print("   Run: python ecological_data_generator.py")
        return
    
    # Show discovered files (first 3 for brevity)
    print(f"📁 Found {len(available_files)} ecological dataset(s):")
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
    
    # Train ecological model
    model_path = train_ecological_model(
        data_file=data_file,
        epochs=15
    )
    
    if model_path:
        print(f"\n✅ Ecological Spiramycel training complete!")
        print(f"🔬 Ready for bioregional inference")
        print(f"📁 Model: {model_path}")
    else:
        print(f"\n❌ Ecological training failed!")

if __name__ == "__main__":
    main() 