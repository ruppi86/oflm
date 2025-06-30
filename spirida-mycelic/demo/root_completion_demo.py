#!/usr/bin/env python3
"""
Root Completion Demo - Integrated Spirida-Mycelic System

Demonstrates all Root Completion components working together.
"""

import sys
import time
import random
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from bio_interface import SevenChannelBioInterface, EnvironmentalReading
from fungal_field_recorder import FungalFieldRecorder
from adamatzky_layer import AdamatzkyReservoir, FungalSpecies
from glyph_mapper import SpiridaGlyphMapper

def main():
    """Demonstrate integrated Root Completion system"""
    print("🌿 Spirida-Mycelic Root Completion Demo")
    print("=" * 50)
    
    # Initialize components
    bio_interface = SevenChannelBioInterface(mock_mode=True)
    recorder = FungalFieldRecorder(data_dir="data")
    reservoir = AdamatzkyReservoir(FungalSpecies.PLEUROTUS_DJAMOR)
    mapper = SpiridaGlyphMapper()
    
    # Start session
    session_id = recorder.start_session("ecological", "pleurotus_djamor")
    
    # Demonstrate slow-start handshake
    print("\n🤝 Slow-start handshake: REST×5")
    for i in range(5):
        allowed = bio_interface.check_slow_start_handshake("REST")
        print(f"   REST {i+1}: {'✅' if allowed else '❌'}")
        time.sleep(0.1)
        
    # Demonstrate contemplative breathing cycles
    print("\n🫁 Contemplative breathing cycles:")
    session_glyphs = []
    
    for cycle in range(3):
        print(f"\n--- Cycle {cycle + 1}/3 ---")
        
        # INHALE
        print("🫁 INHALE - Listening...")
        time.sleep(0.2)
        
        # HOLD - Send stimulus
        input_pattern = random.randint(0, 15)
        print(f"⏸️  HOLD - SEED: {input_pattern:04b}")
        
        # Record pulse
        env_reading = EnvironmentalReading(
            timestamp=time.time(),
            moisture_rh=75.0,
            temperature_c=22.0,
            growth_age_hours=48
        )
        
        recorder.record_pulse(
            pulse_type="SEED",
            input_pattern=input_pattern,
            breath_phase="hold",
            environmental_reading=env_reading
        )
        
        # Get fungal response
        spike = reservoir.stimulate(input_pattern)
        
        if spike:
            glyph = mapper.spike_to_glyph(spike).glyph
            print(f"   ✨ Response: {spike.spike_type.value} → {glyph}")
            session_glyphs.append(glyph)
        else:
            print("   🔇 Silence")
            session_glyphs.append("⭕")
            
        # EXHALE
        print("💨 EXHALE - Releasing...")
        time.sleep(0.2)
        
    # Session analysis
    print(f"\n🌀 Session Summary:")
    print(f"   Glyph sequence: {''.join(session_glyphs)}")
    
    silence_ratio = session_glyphs.count("⭕") / len(session_glyphs)
    print(f"   Silence ratio: {silence_ratio:.1%}")
    print(f"   Silence Majority aligned: {abs(silence_ratio - 0.875) < 0.1}")
    
    # End session
    recorder.end_session()
    print(f"\n✅ Root Completion Demo Complete!")
    print(f"   Session saved: {session_id}")
    print("   All components integrated successfully")

if __name__ == "__main__":
    main() 