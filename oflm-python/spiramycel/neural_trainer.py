#!/usr/bin/env python3
"""
Spiramycel Neural Trainer

Adapts the HaikuMeadowLib training infrastructure for mycelial network repair.
Trains on spore echoes to learn glyph patterns for network healing.

Based on the proven femto/piko architecture but for infrastructure repair
rather than poetry generation.
"""

import json
import random
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Iterator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# Try to import torch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
    
    # Detect device like HaikuMeadowLib
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(f"🚀 GPU detected for Spiramycel training: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device("cpu") 
        print("💻 Spiramycel using CPU (consider installing CUDA for GPU acceleration)")
        
except ImportError:
    torch = None
    nn = None
    F = None
    optim = None
    Dataset = None
    DataLoader = None
    TORCH_AVAILABLE = False
    DEVICE = None
    print("⚠️  PyTorch not available - Spiramycel will use template-based glyph generation")

# Import Spiramycel components
try:
    from .glyph_codec import SpiramycelGlyphCodec, GlyphCategory
    from .spore_map import SporeMapLedger, SporeEcho, Season
except ImportError:
    # Handle direct execution
    from glyph_codec import SpiramycelGlyphCodec, GlyphCategory
    from spore_map import SporeMapLedger, SporeEcho, Season

@dataclass
class NetworkConditions:
    """Current network state affecting glyph generation (like AtmosphericConditions in HaikuMeadowLib)"""
    latency: float = 0.1          # 0.0 = instant, 1.0 = very slow
    voltage: float = 0.5          # 0.0 = low power, 1.0 = high power  
    temperature: float = 0.5      # 0.0 = cold, 1.0 = hot
    error_rate: float = 0.02      # 0.0 = no errors, 1.0 = many errors
    bandwidth: float = 0.8        # 0.0 = congested, 1.0 = free
    uptime: float = 0.9           # 0.0 = just restarted, 1.0 = long stable
    season: Season = Season.SUMMER # Seasonal repair patterns
    bioregion: str = "local"      # Geographic/network context
    
    def to_condition_vector(self) -> List[float]:
        """Convert to 8-dimensional control vector for model conditioning (like HaikuMeadowLib)"""
        # Map season to 3 dimensions (reusing HaikuMeadowLib approach)
        season_encoding = [0.0, 0.0, 0.0]
        season_idx = list(Season).index(self.season)
        if season_idx < 3:
            season_encoding[season_idx] = 1.0
        else:  # Winter maps to same as spring for compression
            season_encoding[0] = 0.5
            
        # Network metrics
        return season_encoding + [self.latency, self.voltage, self.temperature, self.error_rate, self.bandwidth]

class SpiramycelDataset(Dataset if TORCH_AVAILABLE else object):
    """Dataset for training Spiramycel on spore echoes (adapts HaikuDataset)"""
    
    def __init__(self, spore_ledger: SporeMapLedger, codec: SpiramycelGlyphCodec, max_length: int = 16):
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for training")
            
        self.codec = codec
        self.max_length = max_length
        
        # Load spore echoes from ledger
        self.spores = spore_ledger.spores
        
        # Filter for high-quality repair patterns
        self.quality_spores = [s for s in self.spores if s.repair_effectiveness > 0.5 and s.spore_quality > 0.4]
        
        print(f"🍄 Loaded {len(self.spores)} spore echoes, {len(self.quality_spores)} high-quality for training")
    
    def __len__(self):
        return len(self.quality_spores)
    
    def __getitem__(self, idx):
        spore = self.quality_spores[idx]
        
        # Convert glyph sequence to tokens (like haiku tokenization)
        glyph_tokens = spore.glyph_sequence.copy()
        
        # Add START and END tokens (using special glyph IDs)
        start_token = 0x00  # Reserved for START
        end_token = 0x41    # After our 64-glyph vocabulary
        glyph_tokens = [start_token] + glyph_tokens + [end_token]
        
        # Pad to max_length
        if len(glyph_tokens) < self.max_length:
            pad_token = 0x00  # Use START token for padding
            glyph_tokens.extend([pad_token] * (self.max_length - len(glyph_tokens)))
        else:
            glyph_tokens = glyph_tokens[:self.max_length]
        
        # Create input/target sequences
        input_tokens = torch.tensor(glyph_tokens[:-1], dtype=torch.long)
        target_tokens = torch.tensor(glyph_tokens[1:], dtype=torch.long)
        
        # Network conditions (adapting atmospheric conditions)
        conditions = NetworkConditions(
            latency=spore.sensor_deltas.get("latency", 0.1),
            voltage=spore.sensor_deltas.get("voltage", 3.3) / 5.0,  # Normalize to 0-1
            temperature=spore.sensor_deltas.get("temperature", 25.0) / 50.0,  # Normalize to 0-1
            error_rate=min(spore.sensor_deltas.get("error_rate", 0.02), 1.0),
            season=spore.season if spore.season else Season.SUMMER
        )
        
        condition_tensor = torch.tensor(conditions.to_condition_vector(), dtype=torch.float32)
        
        # Effectiveness as additional supervision signal
        effectiveness = torch.tensor(spore.repair_effectiveness, dtype=torch.float32)
        
        return input_tokens, target_tokens, condition_tensor, effectiveness

class SpiramycelNeuralModel(nn.Module if TORCH_AVAILABLE else object):
    """
    Neural glyph generator based on HaikuMeadowLib's PikoHaikuModel
    
    Generates repair glyph sequences from network sensor conditions.
    """
    
    def __init__(self, 
                 vocab_size: int = 66,  # 64 glyphs + START + END
                 embed_dim: int = None,
                 hidden_dim: int = None,
                 condition_dim: int = 8,
                 force_cpu_mode: bool = False):
        
        if TORCH_AVAILABLE:
            super().__init__()
        
        self.vocab_size = vocab_size
        self.condition_dim = condition_dim
        
        # Adaptive sizing (copying HaikuMeadowLib approach)
        if not TORCH_AVAILABLE or DEVICE.type == "cpu" or force_cpu_mode:
            # CPU/Femto mode - smaller for stability
            self.embed_dim = embed_dim or 32
            self.hidden_dim = hidden_dim or 64
            self.model_type = "femto"
            print("🦠 Using Spiramycel femto-model (CPU optimized, ~50k parameters)")
        else:
            # GPU mode - full size
            self.embed_dim = embed_dim or 128
            self.hidden_dim = hidden_dim or 256
            self.model_type = "piko"
            print("🚀 Using Spiramycel piko-model (GPU optimized, ~600k parameters)")
        
        if TORCH_AVAILABLE:
            # Glyph embedding
            self.embedding = nn.Embedding(vocab_size, self.embed_dim)
            
            # Network condition embedding
            self.condition_proj = nn.Linear(condition_dim, self.embed_dim)
            
            # GRU layers (like HaikuMeadowLib)
            if self.model_type == "femto":
                self.gru1 = nn.GRU(self.embed_dim, self.hidden_dim, batch_first=True)
                self.gru2 = None
            else:
                self.gru1 = nn.GRU(self.embed_dim, self.hidden_dim, batch_first=True)
                self.gru2 = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
            
            # Output projection
            self.glyph_proj = nn.Linear(self.hidden_dim, vocab_size)
            
            # Effectiveness prediction head (like silence head in HaikuMeadowLib)
            self.effectiveness_head = nn.Linear(self.hidden_dim, 1)
            
            # Tystnadsmajoritet head (predicts when to stay silent)
            self.silence_head = nn.Linear(self.hidden_dim, 1)
    
    def forward(self, glyph_tokens, conditions, hidden1=None, hidden2=None):
        """Forward pass (based on HaikuMeadowLib architecture)"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
            
        batch_size, seq_len = glyph_tokens.shape
        
        # Embed glyph tokens
        glyph_embeds = self.embedding(glyph_tokens)
        
        # Embed network conditions and broadcast
        condition_embeds = self.condition_proj(conditions)
        condition_embeds = condition_embeds.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine embeddings
        combined_embeds = glyph_embeds + condition_embeds
        
        # GRU processing
        gru1_out, hidden1_new = self.gru1(combined_embeds, hidden1)
        
        if self.gru2 is not None:
            gru2_out, hidden2_new = self.gru2(gru1_out, hidden2)
            final_output = gru2_out
        else:
            final_output = gru1_out
            hidden2_new = None
        
        # Output projections
        glyph_logits = self.glyph_proj(final_output)
        effectiveness_logits = self.effectiveness_head(final_output)
        silence_logits = self.silence_head(final_output)
        
        return glyph_logits, effectiveness_logits, silence_logits, hidden1_new, hidden2_new
    
    def count_parameters(self) -> int:
        """Count total parameters"""
        if not TORCH_AVAILABLE:
            return 0
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class SpiramycelTrainer:
    """
    Neural trainer for Spiramycel (adapts the HaikuMeadowLib training approach)
    """
    
    def __init__(self, output_dir: Path = Path("spiramycel_models")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.codec = SpiramycelGlyphCodec()
    
    def train_on_spore_echoes(self,
                             spore_ledger: SporeMapLedger,
                             epochs: int = 10,
                             batch_size: int = 8,
                             learning_rate: float = 0.001,
                             save_checkpoints: bool = False) -> Optional[Path]:
        """Train neural model on spore echoes (adapts train_piko_model)"""
        
        if not TORCH_AVAILABLE:
            print("❌ PyTorch not available - cannot train neural Spiramycel model")
            return None
        
        print(f"🍄 Starting Spiramycel neural training")
        print(f"   Device: {DEVICE}")
        print(f"   Spore echoes: {len(spore_ledger.spores)}")
        
        # CPU optimization (copying HaikuMeadowLib approach)
        if DEVICE.type == "cpu":
            batch_size = min(batch_size, 4)  # Allow slightly larger batches for serious training
            print(f"   🧘 CPU mode: batch_size={batch_size}, epochs={epochs}")
        
        # Create dataset
        dataset = SpiramycelDataset(spore_ledger, self.codec)
        
        if len(dataset) == 0:
            print("❌ No quality spore echoes found for training")
            return None
        
        # DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # Initialize model
        model = SpiramycelNeuralModel(force_cpu_mode=(DEVICE.type == "cpu")).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Loss functions
        glyph_criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        effectiveness_criterion = nn.MSELoss()
        silence_criterion = nn.BCEWithLogitsLoss()
        
        param_count = model.count_parameters()
        print(f"📊 Spiramycel model: {param_count:,} parameters")
        
        # Training loop
        start_time = time.time()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\n🍄 Epoch {epoch + 1}/{epochs}")
            
            epoch_glyph_loss = 0.0
            epoch_effectiveness_loss = 0.0
            epoch_silence_loss = 0.0
            batch_count = 0
            
            for batch_idx, (input_tokens, target_tokens, conditions, effectiveness) in enumerate(dataloader):
                input_tokens = input_tokens.to(DEVICE)
                target_tokens = target_tokens.to(DEVICE)
                conditions = conditions.to(DEVICE)
                effectiveness = effectiveness.to(DEVICE)
                
                # Forward pass
                glyph_logits, eff_logits, silence_logits, _, _ = model(input_tokens, conditions)
                
                # Glyph sequence loss
                glyph_loss = glyph_criterion(glyph_logits.reshape(-1, model.vocab_size), target_tokens.reshape(-1))
                
                # Effectiveness prediction loss
                eff_loss = effectiveness_criterion(eff_logits.squeeze(-1).mean(dim=1), effectiveness)
                
                # Silence loss (encourage contemplative silence)
                # Target high silence for low effectiveness
                silence_targets = (effectiveness < 0.3).float().unsqueeze(1).expand(-1, silence_logits.shape[1])
                silence_loss = silence_criterion(silence_logits.squeeze(-1), silence_targets)
                
                # Combined loss (weighted for contemplative principles)
                total_loss = glyph_loss + 0.5 * eff_loss + 0.3 * silence_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Accumulate losses
                epoch_glyph_loss += glyph_loss.item()
                epoch_effectiveness_loss += eff_loss.item()
                epoch_silence_loss += silence_loss.item()
                batch_count += 1
                
                # Contemplative breathing pause (like HaikuMeadowLib)
                time.sleep(0.05)  # Shorter pause for serious training
            
            # Epoch summary
            avg_glyph_loss = epoch_glyph_loss / batch_count if batch_count > 0 else 0.0
            avg_eff_loss = epoch_effectiveness_loss / batch_count if batch_count > 0 else 0.0
            avg_silence_loss = epoch_silence_loss / batch_count if batch_count > 0 else 0.0
            total_avg_loss = avg_glyph_loss + 0.5 * avg_eff_loss + 0.3 * avg_silence_loss
            
            print(f"   🌊 Glyph loss: {avg_glyph_loss:.4f}")
            print(f"   📈 Effectiveness loss: {avg_eff_loss:.4f}")
            print(f"   🤫 Silence loss: {avg_silence_loss:.4f}")
            
            # Save checkpoint if best
            if total_avg_loss < best_loss:
                best_loss = total_avg_loss
                checkpoint_path = self.output_dir / f"spiramycel_model_best.pt"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"   💾 New best model saved: {checkpoint_path}")
            
            # Save epoch checkpoints if requested
            if save_checkpoints:
                epoch_path = self.output_dir / f"spiramycel_model_epoch_{epoch+1}.pt"
                torch.save(model.state_dict(), epoch_path)
        
        elapsed = time.time() - start_time
        print(f"\n🌸 Spiramycel training complete in {elapsed/60:.1f} minutes")
        
        # Save final model
        final_path = self.output_dir / f"spiramycel_model_final.pt"
        torch.save(model.state_dict(), final_path)
        
        return final_path
    
    def create_enhanced_training_data(self, num_examples: int = 5000) -> SporeMapLedger:
        """Create enhanced training data with more diversity and realism"""
        
        print(f"🧪 Creating {num_examples} enhanced spore echoes for serious training...")
        
        spore_ledger = SporeMapLedger("enhanced_training_spores.jsonl")
        
        # Enhanced network scenarios with more variety
        scenarios = [
            # Network issues
            {"name": "high_latency", "latency_range": (0.2, 1.0), "repair_glyphs": [0x01, 0x02, 0x0A], "effectiveness": (0.6, 0.95)},
            {"name": "packet_loss", "error_range": (0.05, 0.3), "repair_glyphs": [0x05, 0x03, 0x04], "effectiveness": (0.7, 0.9)},
            {"name": "bandwidth_congestion", "bandwidth_range": (0.1, 0.4), "repair_glyphs": [0x01, 0x03, 0x05], "effectiveness": (0.5, 0.8)},
            
            # Power issues  
            {"name": "low_voltage", "voltage_range": (2.0, 3.0), "repair_glyphs": [0x12, 0x14, 0x18], "effectiveness": (0.4, 0.8)},
            {"name": "power_fluctuation", "voltage_range": (2.8, 3.8), "repair_glyphs": [0x11, 0x12, 0x16], "effectiveness": (0.6, 0.85)},
            {"name": "battery_drain", "voltage_range": (2.5, 3.0), "repair_glyphs": [0x12, 0x14, 0x19], "effectiveness": (0.5, 0.8)},
            
            # Thermal issues
            {"name": "overheating", "temp_range": (35, 60), "repair_glyphs": [0x17, 0x03, 0x16], "effectiveness": (0.4, 0.75)},
            {"name": "thermal_cycling", "temp_range": (15, 45), "repair_glyphs": [0x17, 0x1C, 0x03], "effectiveness": (0.6, 0.8)},
            
            # System health
            {"name": "memory_pressure", "repair_glyphs": [0x25, 0x28, 0x22], "effectiveness": (0.5, 0.8)},
            {"name": "process_hanging", "repair_glyphs": [0x24, 0x25, 0x29], "effectiveness": (0.7, 0.9)},
            {"name": "disk_errors", "repair_glyphs": [0x23, 0x24, 0x2F], "effectiveness": (0.4, 0.7)},
            
            # Good conditions (mostly contemplative)
            {"name": "healthy_system", "repair_glyphs": [0x31, 0x32, 0x33, 0x35, 0x36], "effectiveness": (0.1, 0.4)},
            {"name": "optimal_conditions", "repair_glyphs": [0x37, 0x38, 0x39, 0x3F, 0x40], "effectiveness": (0.05, 0.3)},
        ]
        
        # Enhanced bioregional diversity
        bioregions = [
            "forest_meadow", "mountain_node", "coastal_sensor", "urban_mesh", "desert_relay",
            "arctic_station", "tropical_hub", "suburban_gateway", "rural_repeater", "industrial_node",
            "academic_cluster", "healthcare_network", "transport_junction", "energy_grid", "backup_site"
        ]
        
        for i in range(num_examples):
            scenario = random.choice(scenarios)
            bioregion = random.choice(bioregions)
            
            # Generate sensor deltas based on scenario
            sensor_deltas = {"latency": 0.0, "voltage": 0.0, "temperature": 0.0, "error_rate": 0.0, "bandwidth": 0.0}
            
            if "latency_range" in scenario:
                latency = random.uniform(*scenario["latency_range"])
                sensor_deltas["latency"] = latency - 0.15
            elif "voltage_range" in scenario:
                voltage = random.uniform(*scenario["voltage_range"])
                sensor_deltas["voltage"] = voltage - 3.3
            elif "temp_range" in scenario:
                temp = random.uniform(*scenario["temp_range"])
                sensor_deltas["temperature"] = temp - 25.0
            elif "error_range" in scenario:
                error = random.uniform(*scenario["error_range"])
                sensor_deltas["error_rate"] = error
            elif "bandwidth_range" in scenario:
                bandwidth = random.uniform(*scenario["bandwidth_range"])
                sensor_deltas["bandwidth"] = bandwidth - 0.8
            else:  # Good conditions
                sensor_deltas = {
                    "latency": random.uniform(-0.05, 0.01),
                    "voltage": random.uniform(0.0, 0.1),
                    "temperature": random.uniform(-3.0, 1.0),
                    "error_rate": random.uniform(0.0, 0.01),
                    "bandwidth": random.uniform(-0.1, 0.1)
                }
            
            # Select repair glyphs with realistic patterns
            primary_glyphs = random.choices(scenario["repair_glyphs"], k=random.randint(1, 3))
            
            # Add contemplative glyphs (Tystnadsmajoritet)
            contemplative_glyphs = self.codec.get_contemplative_glyphs()
            
            # Ensure contemplative majority - more silence for healthy systems
            if scenario["name"] in ["healthy_system", "optimal_conditions"]:
                silence_count = random.randint(8, 12)  # Heavy silence
            else:
                silence_count = random.randint(4, 8)   # Moderate silence
            
            contemplative_selection = random.choices(contemplative_glyphs, k=silence_count)
            glyph_sequence = primary_glyphs + contemplative_selection
            
            # Shuffle to mix repair and contemplative naturally
            random.shuffle(glyph_sequence)
            
            # Effectiveness based on scenario with some randomness
            effectiveness = random.uniform(*scenario["effectiveness"])
            
            # Add small amount of seasonal variation
            season = random.choice(list(Season))
            
            # Add to ledger
            spore_ledger.add_spore_echo(
                sensor_deltas=sensor_deltas,
                glyph_sequence=glyph_sequence,
                repair_effectiveness=effectiveness,
                bioregion=bioregion,
                season=season
            )
            
            if i % 500 == 0 and i > 0:
                print(f"   Generated {i}/{num_examples} enhanced spore echoes...")
        
        print(f"✅ Created {len(spore_ledger.spores)} enhanced spore echoes")
        print(f"   Scenarios: {len(scenarios)} different types")
        print(f"   Bioregions: {len(bioregions)} locations")
        print(f"   Seasonal variation: All 4 seasons represented")
        
        return spore_ledger
    
    def evaluate_model(self, model_path: Path, test_ledger: SporeMapLedger) -> Dict[str, float]:
        """Evaluate trained model performance"""
        
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}
        
        # Load trained model
        model = SpiramycelNeuralModel(force_cpu_mode=(DEVICE.type == "cpu")).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
        # Create test dataset
        test_dataset = SpiramycelDataset(test_ledger, self.codec)
        test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
        
        # Evaluation metrics
        total_glyph_loss = 0.0
        total_eff_loss = 0.0
        total_silence_loss = 0.0
        batch_count = 0
        
        glyph_criterion = nn.CrossEntropyLoss(ignore_index=0)
        effectiveness_criterion = nn.MSELoss()
        silence_criterion = nn.BCEWithLogitsLoss()
        
        with torch.no_grad():
            for input_tokens, target_tokens, conditions, effectiveness in test_dataloader:
                input_tokens = input_tokens.to(DEVICE)
                target_tokens = target_tokens.to(DEVICE)
                conditions = conditions.to(DEVICE)
                effectiveness = effectiveness.to(DEVICE)
                
                # Forward pass
                glyph_logits, eff_logits, silence_logits, _, _ = model(input_tokens, conditions)
                
                # Calculate losses
                glyph_loss = glyph_criterion(glyph_logits.reshape(-1, model.vocab_size), target_tokens.reshape(-1))
                eff_loss = effectiveness_criterion(eff_logits.squeeze(-1).mean(dim=1), effectiveness)
                
                silence_targets = (effectiveness < 0.3).float().unsqueeze(1).expand(-1, silence_logits.shape[1])
                silence_loss = silence_criterion(silence_logits.squeeze(-1), silence_targets)
                
                total_glyph_loss += glyph_loss.item()
                total_eff_loss += eff_loss.item()
                total_silence_loss += silence_loss.item()
                batch_count += 1
        
        # Calculate averages
        avg_glyph_loss = total_glyph_loss / batch_count if batch_count > 0 else 0.0
        avg_eff_loss = total_eff_loss / batch_count if batch_count > 0 else 0.0
        avg_silence_loss = total_silence_loss / batch_count if batch_count > 0 else 0.0
        
        # Calculate improvement (rough estimate vs. random baseline)
        random_glyph_loss = 4.19  # ln(66) for 66 classes
        improvement = (random_glyph_loss - avg_glyph_loss) / random_glyph_loss
        
        return {
            "glyph_loss": avg_glyph_loss,
            "effectiveness_loss": avg_eff_loss,
            "silence_loss": avg_silence_loss,
            "improvement": improvement
        }

    def create_training_data_from_simulation(self, num_examples: int = 1000) -> SporeMapLedger:
        """Create synthetic training data by simulating network repair events (backward compatibility)"""
        return self.create_enhanced_training_data(num_examples)

def demo_spiramycel_neural_training():
    """Demonstrate the complete neural training pipeline"""
    
    print("🍄 Spiramycel Neural Training Demo")
    print("=" * 60)
    
    trainer = SpiramycelTrainer()
    
    # Create synthetic training data
    spore_ledger = trainer.create_training_data_from_simulation(200)  # Slightly larger demo
    
    # Show training data statistics
    stats = spore_ledger.get_statistics()
    print(f"\n📊 Training Data Statistics:")
    print(f"   Total spores: {stats['total_spores']}")
    print(f"   Average effectiveness: {stats['avg_effectiveness']:.2f}")
    print(f"   Bioregional distribution: {stats['bioregional_distribution']}")
    
    # Train neural model
    if TORCH_AVAILABLE:
        print(f"\n🧠 Training neural Spiramycel model...")
        model_path = trainer.train_on_spore_echoes(spore_ledger, epochs=5, batch_size=4)
        
        if model_path:
            print(f"✅ Neural model trained and saved to: {model_path}")
            
            # Quick evaluation
            eval_results = trainer.evaluate_model(model_path, spore_ledger)
            print(f"\n📊 Model Performance:")
            print(f"   Glyph loss: {eval_results['glyph_loss']:.3f}")
            print(f"   Effectiveness loss: {eval_results['effectiveness_loss']:.3f}")
            print(f"   Silence loss: {eval_results['silence_loss']:.3f}")
            print(f"   Improvement over random: {eval_results['improvement']:.1%}")
        else:
            print("❌ Neural training failed")
    else:
        print("\n⚠️  PyTorch not available - skipping neural training")
    
    print("\n🌱 Demo complete - for serious training, run: python serious_training.py")
    print("🍄 Ready for integration with existing Spiramycel framework")

if __name__ == "__main__":
    demo_spiramycel_neural_training() 