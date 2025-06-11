"""
breath_trainer.py - CPU-Breath Training Demo

Simple test of the breath-synchronized training concept.
"""

import time
import random
from pathlib import Path
from dew_ledger import DewLedger, Season, create_atmospheric_vector

def simple_demo():
    """Simple demo of breath training and dew ledger"""
    
    print("🌸 Simple Breath Training Demo")
    
    # Test dew ledger
    ledger = DewLedger(Path("test_dew.jsonl"))
    
    season_vec = create_atmospheric_vector(Season.WINTER)
    
    ledger.add_drop(
        fragment="morning silence",
        utterance="frost holds\nthe world in stillness\nbreath fades",
        season_vec=season_vec,
        resonance=0.9,
        season=Season.WINTER,
        generation_type="neural"
    )
    
    print("   ✓ Added sample dew drop")
    
    # Test solstice distillation  
    chosen = ledger.solstice_distillation(max_chosen=2)
    print(f"   ✓ Solstice distillation: {len(chosen)} drops chosen")
    
    # Simulate breath training
    print("\n🫁 Simulating breath training...")
    for epoch in range(2):
        print(f"   Epoch {epoch + 1}: inhaling...")
        time.sleep(1.0)  # Breath interval
        loss = random.uniform(0.3, 0.8)
        print(f"   Epoch {epoch + 1}: loss {loss:.3f}")
    
    # Cleanup
    Path("test_dew.jsonl").unlink(missing_ok=True)
    
    print("🌿 Demo complete!")

if __name__ == "__main__":
    simple_demo() 