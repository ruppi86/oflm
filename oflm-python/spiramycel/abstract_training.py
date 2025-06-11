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

# Import from existing modules
from glyph_codec import SpiramycelGlyphCodec
from neural_trainer import SpiramycelDataset, NetworkConditions, SpiramycelNeuralModel

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
        
        # Convert to format similar to SpiramycelDataset
        sensor_readings = sample['conditions']['sensor_readings']
        
        # Create network conditions from abstract data
        conditions = NetworkConditions(
            latency=sensor_readings.get('latency', 0.1),
            voltage=sensor_readings.get('voltage', 3.3),
            temperature=sensor_readings.get('temperature', 25.0),
            error_rate=sensor_readings.get('error_rate', 0.02),
            bandwidth=sensor_readings.get('bandwidth', 0.8),
        )
        
        # Get glyph sequence 
        glyph_sequence = sample['repair_action']['glyph_sequence']
        
        # Add START and END tokens like SpiramycelDataset
        start_token = 0x00
        end_token = 0x41
        glyph_tokens = [start_token] + glyph_sequence + [end_token]
        
        # Pad to max_length of 16
        max_length = 16
        if len(glyph_tokens) < max_length:
            pad_token = 0x00
            glyph_tokens.extend([pad_token] * (max_length - len(glyph_tokens)))
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
                        epochs: int = 10):
    """Train Spiramycel on abstract data (mirrors train_ecological_model)"""
    
    print("🔬 Abstract Spiramycel Training")
    print("=" * 50)
    
    # Initialize codec
    codec = SpiramycelGlyphCodec()
    print(f"📝 Glyph vocabulary: {len(codec.glyphs)} symbols")
    
    # Load abstract dataset
    dataset = AbstractDataset(data_file, codec)
    
    if len(dataset) == 0:
        print("❌ No training data loaded!")
        return None
    
    # Use SpiramycelNeuralModel with femto to match ecological training on CPU
    device = torch.device("cpu")  # Force CPU for compatibility
    model = SpiramycelNeuralModel(force_cpu_mode=True).to(device)
    
    # Print actual model type that was selected
    print(f"🧠 Model: {model.model_type} ({model.count_parameters():,} parameters)")
    
    # Training setup - match ecological training parameters exactly
    batch_size = 8  # Match ecological training batch size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Loss functions matching neural_trainer.py
    glyph_criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
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
            glyph_logits, eff_logits, silence_logits, _, _ = model(input_tokens, condition_tensor)
            
            # Calculate losses
            glyph_loss = glyph_criterion(
                glyph_logits.reshape(-1, glyph_logits.size(-1)),
                target_tokens.reshape(-1)
            )
            
            effectiveness_loss = effectiveness_criterion(
                eff_logits.squeeze(-1).mean(dim=1),
                effectiveness
            )
            
            # Silence loss - encourage silence when effectiveness is low
            silence_targets = (effectiveness < 0.3).float().unsqueeze(1).expand(-1, silence_logits.shape[1])
            silence_loss = silence_criterion(silence_logits.squeeze(-1), silence_targets)
            
            # Combined loss
            total_loss = glyph_loss + 0.5 * effectiveness_loss + 0.3 * silence_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses
            epoch_glyph_loss += glyph_loss.item()
            epoch_effectiveness_loss += effectiveness_loss.item()
            epoch_silence_loss += silence_loss.item()
            num_batches += 1
            
            # Match ecological training's contemplative breathing pause
            time.sleep(0.05)
        
        # Calculate average losses
        avg_glyph_loss = epoch_glyph_loss / num_batches if num_batches > 0 else 0.0
        avg_effectiveness_loss = epoch_effectiveness_loss / num_batches if num_batches > 0 else 0.0
        avg_silence_loss = epoch_silence_loss / num_batches if num_batches > 0 else 0.0
        
        print(f"Epoch {epoch+1:2d}: Glyph {avg_glyph_loss:.3f} | "
              f"Effectiveness {avg_effectiveness_loss:.4f} | "
              f"Silence {avg_silence_loss:.4f}")
    
    training_time = time.time() - start_time
    print(f"⏱ Training completed in {training_time:.1f} seconds")
    
    # Save model
    model_path = f"abstract_spiramycel_piko.pt"
    torch.save(model.state_dict(), model_path)
    
    print(f"💾 Model saved to {model_path}")
    
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
        
        test_tensor = torch.tensor([test_conditions.to_condition_vector()], dtype=torch.float32)
        start_token = torch.tensor([[0x00]], dtype=torch.long)  # START token
        
        # Generate abstract repair sequence
        generated_tokens = [0x00]  # Start with START token
        hidden1, hidden2 = None, None
        
        for step in range(10):  # Generate up to 10 tokens
            input_tensor = torch.tensor([generated_tokens[-1:]], dtype=torch.long)
            glyph_logits, _, silence_logits, hidden1, hidden2 = model(input_tensor, test_tensor, hidden1, hidden2)
            
            # Check if we should use silence
            silence_prob = torch.sigmoid(silence_logits[0, -1]).item()
            
            if silence_prob > 0.7:  # High silence threshold
                print(f"   Step {step}: 🤫 (silence probability: {silence_prob:.2f})")
                break
            
            # Sample next token
            probs = torch.softmax(glyph_logits[0, -1], dim=0)
            next_token = torch.multinomial(probs, 1).item()
            
            if next_token == 0x41:  # END token
                break
                
            generated_tokens.append(next_token)
            
            # Decode and display
            glyph_name = codec.decode_glyph(next_token)
            print(f"   Step {step}: {glyph_name} (0x{next_token:02X})")
    
    return model_path

def main():
    """Main training function"""
    
    # Check available data files
    data_files = [
        "training_scenarios/abstract_small_chaotic.jsonl",
        "training_scenarios/abstract_medium_chaotic.jsonl", 
        "training_scenarios/abstract_large_chaotic.jsonl"
    ]
    
    available_files = [f for f in data_files if Path(f).exists()]
    
    if not available_files:
        print("❌ No abstract training data found!")
        print("   Run: cd oflm-python/spiramycel && python generate_abstract_data.py")
        return
    
    # Use largest available dataset
    data_file = available_files[-1]
    print(f"📊 Using dataset: {data_file}")
    
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