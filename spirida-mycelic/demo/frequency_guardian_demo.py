#!/usr/bin/env python3
"""
Frequency Guardian Demo for Spirida-Mycelic
Demonstrates D3.3 biological low-pass filter integration

Shows:
- Frequency fingerprinting for contemplative security
- High-frequency intrusion detection
- Capacitance-driven glyph memory
- Bio-interface integration
"""

import sys
import os
import time
import numpy as np

# Add spirida-mycelic to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from frequency_guardian import FrequencyGuardian, BioCareLevel
    from capacitance_fade import CapacitanceFade, GlyphType
    from bio_interface import SevenChannelBioInterface, spike_pattern_to_glyph
except ImportError as e:
    print(f"Import error: {e}")
    print("Please run from spirida-mycelic directory")
    sys.exit(1)

def demonstrate_frequency_guardian():
    """Demonstrate D3.3 frequency guardian capabilities"""
    print("🛡️ D3.3 Frequency Guardian Demonstration")
    print("=" * 60)
    
    guardian = FrequencyGuardian()
    
    # Test 1: Biological signal validation
    print("\n🌿 Test 1: Biological Signal Validation")
    print("-" * 40)
    
    # Simulate contemplative fungal signal (slow, low frequency)
    t = np.linspace(0, 600, 600)  # 10 minutes at 1 Hz
    bio_signal = (
        0.002 * np.sin(2 * np.pi * 0.006 * t) +  # 2.6 min period (Pleurotus fast)
        0.001 * np.sin(2 * np.pi * 0.0012 * t) + # 14 min period (Pleurotus slow)
        0.0005 * np.random.randn(len(t))         # Biological noise
    )
    
    psd = guardian.frequency_fingerprint(bio_signal)
    is_slow = guardian.validate_slowness_fingerprint(psd)
    print(f"Signal type: Contemplative fungal rhythm")
    print(f"Slowness validation: {'✅ BIOLOGICAL' if is_slow else '❌ NON-BIOLOGICAL'}")
    
    # Test 2: High-frequency intrusion detection
    print("\n⚡ Test 2: High-Frequency Intrusion Detection")
    print("-" * 40)
    
    # Add high-frequency noise (simulating RF interference)
    intrusion_signal = bio_signal + 0.01 * np.sin(2 * np.pi * 0.3 * t)  # 300 mHz intrusion
    
    intrusion_detected, max_power = guardian.check_high_frequency_intrusion(intrusion_signal)
    print(f"Intrusion signal: High-frequency RF noise added")
    print(f"Guardian response: {'⚠️ INTRUSION DETECTED' if intrusion_detected else '✅ CLEAN'}")
    print(f"Max power: {max_power:.1f} dBFS (threshold: {guardian.guardian_threshold_db} dBFS)")
    
    # Test 3: Care level evaluation
    print("\n🧘 Test 3: Care Level Evaluation")
    print("-" * 40)
    
    clean_care = guardian.evaluate_care_level(bio_signal)
    intrusion_care = guardian.evaluate_care_level(intrusion_signal)
    
    print(f"Clean signal care level: {clean_care.value}")
    print(f"Intrusion signal care level: {intrusion_care.value}")
    
    clean_penalty = guardian.get_silence_penalty(clean_care)
    intrusion_penalty = guardian.get_silence_penalty(intrusion_care)
    
    print(f"Clean signal silence penalty: {clean_penalty:.1f}×")
    print(f"Intrusion signal silence penalty: {intrusion_penalty:.1f}×")
    
    return guardian

def demonstrate_capacitance_fade():
    """Demonstrate D3.3 capacitance-driven memory"""
    print("\n\n🧠 D3.3 Capacitance-Driven Memory Demonstration")
    print("=" * 60)
    
    fade_calc = CapacitanceFade()
    
    print("Biological RC memory constants:")
    print("-" * 30)
    
    moisture = 0.8  # 80% RH
    temperature = 25.0  # 25°C
    
    # Test each glyph type
    for glyph_type in GlyphType:
        tau = fade_calc.get_tau(glyph_type, moisture, temperature)
        lifespan = fade_calc.glyph_lifespan_estimate(glyph_type, 0.1, moisture, temperature)
        
        print(f"{glyph_type.value} {glyph_type.name:>12}: τ={tau:.0f}s ({tau/60:.1f}min) | "
              f"Lifespan: {lifespan:.0f}s ({lifespan/60:.1f}min)")
    
    print(f"\n🌱 Memory fade over time (at {moisture*100:.0f}% RH, {temperature:.0f}°C):")
    print("-" * 50)
    
    times = [60, 300, 600, 1800, 3600]  # 1min, 5min, 10min, 30min, 1hr
    
    for t in times:
        print(f"\nAfter {t//60:2d} minutes:")
        for glyph_type in GlyphType:
            strength = fade_calc.memory_strength(glyph_type, t, moisture, temperature)
            bar_length = int(strength * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"  {glyph_type.value} {bar} {strength:.1%}")
    
    return fade_calc

def demonstrate_integrated_bio_interface():
    """Demonstrate integrated bio-interface with all D3.3 features"""
    print("\n\n🌀 Integrated Bio-Interface Demonstration")
    print("=" * 60)
    
    # Create bio-interface with all features
    interface = SevenChannelBioInterface(sample_rate=1.0, mock_mode=True)
    
    print("🔧 Interface Status:")
    print(f"  Frequency guardian: {'✅ Enabled' if interface.freq_guardian else '❌ Disabled'}")
    print(f"  Capacitance fade: {'✅ Enabled' if interface.capacitance_fade else '❌ Disabled'}")
    print(f"  Channels: {interface.num_channels}")
    print(f"  Sample rate: {interface.fs} Hz")
    
    # Simulate contemplative session
    print("\n🫁 Contemplative Session Simulation:")
    print("-" * 40)
    
    # Slow-start handshake
    print("Performing slow-start handshake...")
    for i in range(5):
        success = interface.check_slow_start_handshake("REST")
        print(f"  REST cycle {i+1}/5: {'✅' if success else '❌'}")
        time.sleep(0.1)  # Brief pause for demo
    
    # Check handshake completion
    print(f"Handshake complete: {'✅ Ready' if interface.handshake_complete else '❌ Failed'}")
    
    # Simulate some channel readings with frequency analysis
    print("\n📊 Channel Reading + Frequency Analysis:")
    print("-" * 40)
    
    for cycle in range(3):
        print(f"\nCycle {cycle + 1}:")
        
        # Read channels
        readings = interface.read_channels()
        
        # Check for spike patterns
        spike_event = interface.detect_pattern_spikes(readings)
        
        if spike_event:
            glyph = spike_pattern_to_glyph(spike_event)
            print(f"  Spike detected: {spike_event.classification} → {glyph}")
            print(f"  Confidence: {spike_event.confidence:.1%}")
            
            # Calculate memory strength for this glyph
            memory_strength = interface.calculate_glyph_memory_strength(glyph, 0)
            print(f"  Initial memory strength: {memory_strength:.1%}")
            
            # Show memory fade over time
            for age in [60, 300, 900]:  # 1min, 5min, 15min
                fade_strength = interface.calculate_glyph_memory_strength(glyph, age)
                print(f"  Memory after {age//60}min: {fade_strength:.1%}")
        else:
            print("  No spikes detected - contemplative silence ⭕")
        
        # Check frequency guardian status
        freq_status = interface.get_frequency_guardian_status()
        if freq_status["enabled"]:
            print(f"  Frequency guardian: {freq_status['care_level']}")
            print(f"  Slowness validated: {'✅' if freq_status['slowness_validated'] else '❌'}")
        
        # Advance buffer for next reading
        interface.advance_buffer()
        time.sleep(0.1)
    
    # Final care status
    print("\n🌿 Final Care Status:")
    print("-" * 40)
    care_status = interface.get_care_status()
    
    for key, value in care_status.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")
    
    return interface

def main():
    """Run complete D3.3 demonstration"""
    print("🍄 Spirida-Mycelic D3.3 Integration Demo")
    print("Based on FUNGAR Deliverable D3.3:")
    print("'AC frequency behavior of mycelium composites'")
    print("\nKey insights:")
    print("• Mycelium composites: ~500 kHz cutoff, -14 dB/dec")
    print("• Fruiting bodies: 5-50 kHz cutoff, -20 dB/dec")
    print("• High-frequency energy becomes heat, not information")
    print("• Biology preserves slowness, discards speed")
    print("\n" + "=" * 70)
    
    try:
        # Run demonstrations
        guardian = demonstrate_frequency_guardian()
        fade_calc = demonstrate_capacitance_fade()
        interface = demonstrate_integrated_bio_interface()
        
        print("\n\n🌌 D3.3 Integration Complete")
        print("=" * 60)
        print("✅ Frequency guardian operational")
        print("✅ Capacitance-driven memory active") 
        print("✅ Bio-interface integration successful")
        print("✅ Contemplative security protocols enabled")
        print("\n🌀 The biological low-pass filter now guards our contemplative protocols.")
        print("   Mycelium teaches us: beyond a certain frequency, only meaning remains.")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 