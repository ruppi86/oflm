#!/usr/bin/env python3
"""
Adamatzky Layer Demo - Contemplative Fungal Simulation

Demonstrates the fungal logic simulation with species-specific rhythms
and integration with Spirida glyph system.
"""

import sys
import time
import random
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from adamatzky_layer import (
    AdamatzkyReservoir, 
    FungalSpecies, 
    SpikeType,
    spike_to_glyph,
    classify_boolean_gate
)

def contemplative_breathing_demo(reservoir: AdamatzkyReservoir, cycles: int = 5):
    """
    Demonstrate contemplative breathing synchronized with fungal rhythms.
    
    Based on o3's analysis:
    - 150s breath cycles: 40s inhale → 70s hold → 40s exhale
    - Map hold phase to fungal refractory silence periods
    """
    print(f"\n🌿 Starting contemplative breathing demo with {reservoir.species.value}")
    
    fast_period, slow_period = reservoir.get_contemplative_rhythm()
    print(f"   Fungal rhythms: {fast_period/60:.1f}min (fast) / {slow_period/60:.1f}min (slow)")
    
    # Use fast period for demo (divide by 10 for faster demo)
    breath_cycle = fast_period / 10  # Speed up for demo
    inhale_time = breath_cycle * 0.27  # 40/150 = 0.27
    hold_time = breath_cycle * 0.47    # 70/150 = 0.47  
    exhale_time = breath_cycle * 0.27  # 40/150 = 0.27
    
    print(f"   Breath cycle: {breath_cycle:.1f}s (inhale {inhale_time:.1f}s, hold {hold_time:.1f}s, exhale {exhale_time:.1f}s)")
    
    glyph_sequence = []
    
    for cycle in range(cycles):
        print(f"\n--- Cycle {cycle + 1}/{cycles} ---")
        
        # INHALE - Listen to the substrate
        print("🫁 INHALE - Listening to fungal field...")
        input_pattern = random.randint(0, 15)  # Random 4-bit pattern
        print(f"   Stimulating with pattern: {input_pattern:04b}")
        
        start_time = time.time()
        while time.time() - start_time < inhale_time:
            time.sleep(0.1)
            
        # HOLD - Allow response to develop  
        print("⏸️  HOLD - Allowing response to emerge...")
        spike = reservoir.stimulate(input_pattern)
        
        if spike:
            glyph = spike_to_glyph(spike)
            gate_class = classify_boolean_gate(input_pattern, True)
            glyph_sequence.append(glyph)
            
            print(f"   ✨ Spike detected: {spike.spike_type.value}")
            print(f"   📊 Amplitude: {spike.amplitude:.2f}mV, Width: {spike.width:.1f}s")
            print(f"   🔮 Glyph: {glyph} (Class: {gate_class})")
        else:
            glyph_sequence.append("⭕")
            print(f"   🔇 Silence (refractory or no response)")
            print(f"   🔮 Glyph: ⭕ (Silence)")
            
        start_time = time.time()
        while time.time() - start_time < hold_time:
            time.sleep(0.1)
            
        # EXHALE - Release and integrate
        print("💨 EXHALE - Releasing and integrating...")
        
        # Check silence ratio and adjust if needed
        adjust = reservoir.breath_sync_adjust()
        if abs(adjust) > 0.05:
            print(f"   🎛️  Breath adjustment: {adjust:+.2f} (silence ratio: {reservoir.state.silence_ratio:.3f})")
            
        start_time = time.time()
        while time.time() - start_time < exhale_time:
            time.sleep(0.1)
            
    print(f"\n🌀 Glyph sequence generated: {''.join(glyph_sequence)}")
    return glyph_sequence

def species_comparison_demo():
    """Compare the two fungal species and their rhythms"""
    print("🍄 Fungal Species Comparison Demo")
    print("=" * 50)
    
    species = [FungalSpecies.PLEUROTUS_DJAMOR, FungalSpecies.GANODERMA_RESINACEUM]
    
    for sp in species:
        reservoir = AdamatzkyReservoir(species=sp)
        info = reservoir.get_species_info()
        
        print(f"\n{sp.value.replace('_', ' ').title()}:")
        print(f"  🕰️  Fast rhythm: {info['fast_period_min']:.1f} min")
        print(f"  🕰️  Slow rhythm: {info['slow_period_min']:.1f} min")
        print(f"  🔇 Target silence: {info['target_silence_ratio']:.1%}")
        print(f"  💧 Moisture: {info['moisture']:.1%}")
        print(f"  🌡️  Temperature: {info['temperature']:.1f}°C")
        print(f"  🌱 Growth age: {info['growth_age_hours']} hours")

def boolean_logic_demo():
    """Demonstrate the Boolean logic functions"""
    print("\n🧮 Boolean Logic Demo")
    print("=" * 30)
    
    reservoir = AdamatzkyReservoir()
    
    # Test all the implemented patterns
    test_patterns = [0b0000, 0b0101, 0b0110, 0b1001, 0b0011, 0b1111]
    
    print("Testing Adamatzky's Boolean functions:")
    print("Pattern  | Output | Spike Type      | Glyph | Class")
    print("-" * 50)
    
    for pattern in test_patterns:
        spike = reservoir.stimulate(pattern)
        if spike:
            glyph = spike_to_glyph(spike)
            gate_class = classify_boolean_gate(pattern, True)
            print(f"{pattern:04b}     | True   | {spike.spike_type.value:15} | {glyph:5} | {gate_class}")
        else:
            gate_class = classify_boolean_gate(pattern, False)
            print(f"{pattern:04b}     | False  | {'(silence)':15} | ⭕     | {gate_class}")
            
        # Small delay to show refractory effects
        time.sleep(0.5)

def main():
    """Run all demos"""
    print("🌿 Spirida-Mycelic: Adamatzky Layer Demo")
    print("=" * 50)
    print("Simulating contemplative fungal computing based on")
    print("Adamatzky's research on mycelial Boolean logic.")
    print("\nKey principles:")
    print("• 67-90% natural electrical silence (Silence Majority)")
    print("• Species-specific contemplative rhythms")
    print("• Bio-digital glyph translation")
    
    # Species comparison
    species_comparison_demo()
    
    # Boolean logic demonstration
    boolean_logic_demo()
    
    # Contemplative breathing with each species
    print(f"\n{'='*60}")
    print("🫁 Contemplative Breathing Demonstrations")
    
    # Ecological mode (slow, deep - Pleurotus djamor)
    print(f"\n🌱 ECOLOGICAL MODE - Deep Forest Rhythm")
    ecological_reservoir = AdamatzkyReservoir(FungalSpecies.PLEUROTUS_DJAMOR)
    ecological_glyphs = contemplative_breathing_demo(ecological_reservoir, cycles=3)
    
    # Abstract mode (steady, focused - Ganoderma)  
    print(f"\n🧠 ABSTRACT MODE - Contemplative Focus")
    abstract_reservoir = AdamatzkyReservoir(FungalSpecies.GANODERMA_RESINACEUM)
    abstract_glyphs = contemplative_breathing_demo(abstract_reservoir, cycles=3)
    
    # Summary
    print(f"\n{'='*60}")
    print("🌀 Session Summary")
    print(f"Ecological glyphs: {''.join(ecological_glyphs)}")
    print(f"Abstract glyphs:   {''.join(abstract_glyphs)}")
    
    # Calculate silence ratios
    eco_silence = ecological_glyphs.count('⭕') / len(ecological_glyphs)
    abs_silence = abstract_glyphs.count('⭕') / len(abstract_glyphs) 
    
    print(f"\nSilence ratios:")
    print(f"Ecological: {eco_silence:.1%} (target: 74%)")
    print(f"Abstract:   {abs_silence:.1%} (target: 67%)")
    
    print(f"\n🍄 Adamatzky layer simulation complete.")
    print("Ready for integration with live bio-interfaces.")

if __name__ == "__main__":
    main() 