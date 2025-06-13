#!/usr/bin/env python3
"""
Test script for the most recent Spiramycel neural model
"""

from spiramycel.neural_trainer import SpiramycelNeuralModel
from spiramycel.spore_codec import SpiramycelGlyphCodec
import torch

def test_recent_model():
    print("🦠 Testing most recent Spiramycel model...")
    
    # Create model (using correct vocab_size=67)
    model = SpiramycelNeuralModel(vocab_size=67, force_cpu_mode=True)
    print(f"✓ Model created successfully")
    print(f"   Model type: {model.model_type}")
    print(f"   Embed dim: {model.embed_dim}")
    print(f"   Hidden dim: {model.hidden_dim}")
    print(f"   Vocab size: {model.vocab_size}")
    print(f"   Parameters: {model.count_parameters():,}")
    
    # Test tokenization
    codec = SpiramycelGlyphCodec()
    print(f"✓ Codec vocab size: {len(codec.vocab)}")
    
    # Simple forward pass test
    batch_size = 2
    sequence_length = 8
    condition_dim = 8
    
    # Create dummy inputs
    glyph_tokens = torch.randint(0, model.vocab_size, (batch_size, sequence_length))
    conditions = torch.randn(batch_size, condition_dim)
    
    print(f"✓ Testing forward pass...")
    print(f"   Input shape: {glyph_tokens.shape}")
    print(f"   Conditions shape: {conditions.shape}")
    
    # Forward pass
    with torch.no_grad():
        glyph_logits, eff_logits, silence_logits, h1, h2 = model(glyph_tokens, conditions)
    
    print(f"✓ Forward pass successful!")
    print(f"   Glyph logits shape: {glyph_logits.shape}")
    print(f"   Effectiveness logits shape: {eff_logits.shape}")  
    print(f"   Silence logits shape: {silence_logits.shape}")
    
    print("🎉 Recent model test complete!")

if __name__ == "__main__":
    test_recent_model() 