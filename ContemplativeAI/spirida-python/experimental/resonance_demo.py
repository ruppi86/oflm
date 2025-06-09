#!/usr/bin/env python3
"""
🌀 RESONANCE PATTERNS DEMO – Living Connections

This demonstration shows how pulses resonate with each other in 
meaningful ways, how different field ecosystems manage memory,
and how contemplative computing creates emergent behaviors.

Watch how:
- Pulses strengthen each other through resonance
- Different composting modes create natural rhythms
- Emotional and symbolic connections emerge organically
"""

import sys
import os
import time
from datetime import datetime

# Add the parent directory to the path so we can import spirida
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spirida.contemplative_core import ContemplativeSystem, SpiralField, PulseObject

def demonstrate_pulse_resonances():
    """Show how individual pulses discover resonance with each other."""
    print("🌊 Demonstrating Pulse Resonances")
    print("="*60)
    print("Watch how pulses recognize and strengthen each other...")
    print()
    
    # Create several pulses with different characteristics
    pulses = [
        PulseObject("🌿", "peaceful", amplitude=0.8, decay_rate=0.03),
        PulseObject("🌱", "growing", amplitude=0.7, decay_rate=0.05),
        PulseObject("🌊", "flowing", amplitude=0.9, decay_rate=0.02),
        PulseObject("💧", "calm", amplitude=0.6, decay_rate=0.04),
        PulseObject("🌙", "peaceful", amplitude=0.5, decay_rate=0.01),
    ]
    
    print("Created 5 pulses with different symbols and emotions...")
    for pulse in pulses:
        print(f"  {pulse}")
    
    print("\nExploring resonances between pulses...")
    
    # Test resonances between different pulse pairs
    interesting_pairs = [
        (pulses[0], pulses[1]),  # 🌿 peaceful + 🌱 growing
        (pulses[0], pulses[4]),  # 🌿 peaceful + 🌙 peaceful
        (pulses[2], pulses[3]),  # 🌊 flowing + 💧 calm
    ]
    
    for pulse_a, pulse_b in interesting_pairs:
        print(f"\n🔄 Resonance between {pulse_a.symbol} [{pulse_a.emotion}] and {pulse_b.symbol} [{pulse_b.emotion}]:")
        
        resonance = pulse_a.resonates_with(pulse_b)
        
        print(f"   Strength: {resonance['strength']:.3f}")
        print(f"   Poetic trace: {resonance['poetic_trace']}")
        print(f"   Components:")
        for component, value in resonance['components'].items():
            print(f"     {component}: {value:.3f}")
        
        # Show strengthening effect
        if resonance['strength'] > 0.6:
            original_attention_a = pulse_a.current_attention()
            original_attention_b = pulse_b.current_attention()
            
            pulse_a.strengthen_from_resonance(resonance['strength'])
            pulse_b.strengthen_from_resonance(resonance['strength'])
            
            print(f"   🌟 Strengthening effect:")
            print(f"     {pulse_a.symbol}: {original_attention_a:.3f} → {pulse_a.current_attention():.3f}")
            print(f"     {pulse_b.symbol}: {original_attention_b:.3f} → {pulse_b.current_attention():.3f}")
    
    print("\n" + "="*60 + "\n")

def demonstrate_field_ecosystems():
    """Show how different composting modes create natural rhythms."""
    print("🌱 Demonstrating Field Ecosystems")
    print("="*60)
    print("Three fields with different temporal relationships...")
    print()
    
    # Create fields with different composting behaviors
    natural_field = SpiralField("natural_memory", composting_mode="natural")
    seasonal_field = SpiralField("seasonal_cycles", composting_mode="seasonal") 
    seasonal_field.seasonal_cycle_hours = 0.01  # Very fast for demo (36 seconds)
    
    resonant_field = SpiralField("resonant_connections", composting_mode="resonant")
    
    fields = [natural_field, seasonal_field, resonant_field]
    
    print("🌿 Natural Field: Composts based on attention threshold")
    print("🍂 Seasonal Field: Composts based on seasonal cycles (fast demo cycle)")
    print("🌊 Resonant Field: Keeps pulses that resonate with others")
    print()
    
    # Emit initial pulses into each field
    print("Emitting initial pulses...")
    
    # Natural field - mixed decay rates
    natural_field.emit("🌱", "emerging", amplitude=0.8, decay_rate=0.1)  # Fast fade
    natural_field.emit("🌳", "stable", amplitude=0.6, decay_rate=0.01)   # Slow fade
    natural_field.emit("🍃", "gentle", amplitude=0.9, decay_rate=0.05)   # Medium fade
    
    # Seasonal field - various pulses
    seasonal_field.emit("🌸", "spring", amplitude=0.7, decay_rate=0.02)
    seasonal_field.emit("☀️", "summer", amplitude=0.8, decay_rate=0.03)
    seasonal_field.emit("🍂", "autumn", amplitude=0.9, decay_rate=0.04)
    
    # Resonant field - emotionally connected pulses
    resonant_field.emit("💖", "love", amplitude=0.8, decay_rate=0.08)
    resonant_field.emit("🤗", "connection", amplitude=0.7, decay_rate=0.08)
    resonant_field.emit("💙", "love", amplitude=0.6, decay_rate=0.08)  # Should resonate with 💖
    resonant_field.emit("⭐", "isolated", amplitude=0.5, decay_rate=0.08)  # Less connected
    
    print(f"Initial field states:")
    for field in fields:
        print(f"  {field.name}: {len(field.pulses)} pulses, resonance {field.resonance_field():.2f}")
    
    # Watch fields evolve over time
    for cycle in range(4):
        print(f"\n--- Time Cycle {cycle + 1} ---")
        
        # Show seasonal status for seasonal field
        if cycle == 0:
            seasonal_status = seasonal_field.seasonal_status()
            print(f"Seasonal field is in: {seasonal_status.get('season', 'unknown')} (phase {seasonal_status.get('phase', 0):.2f})")
        
        # Show resonances in resonant field
        if cycle == 1:
            resonances = resonant_field.find_resonances(min_strength=0.4)
            print(f"Resonant field has {len(resonances)} active resonances:")
            for res in resonances[:2]:  # Show first 2
                print(f"  {res['resonance']['poetic_trace']}")
        
        # Wait for decay/changes
        time.sleep(3)
        
        # Compost each field
        for field in fields:
            composted = field.compost()
            current_resonance = field.resonance_field()
            print(f"{field.name}: {len(field.pulses)} pulses, {composted} composted, resonance {current_resonance:.2f}")
            
            if field.composting_mode == "seasonal" and cycle == 2:
                new_status = field.seasonal_status()
                print(f"  Season shifted to: {new_status.get('season', 'unknown')}")
    
    print("\nFinal field comparison:")
    for field in fields:
        print(f"  {field.name}:")
        print(f"    Active pulses: {len(field.pulses)}")
        print(f"    Total composted: {field.total_composted}")
        print(f"    Final resonance: {field.resonance_field():.2f}")
        if field.pulses:
            print(f"    Surviving pulses: {[p.symbol for p in field.pulses]}")
        print()
    
    print("="*60 + "\n")

def demonstrate_living_conversation():
    """Show how a field develops a living conversation through resonances."""
    print("💬 Demonstrating Living Conversation")
    print("="*60)
    print("Watch how a contemplative field develops emergent behaviors...")
    print()
    
    # Create a field optimized for conversation
    conversation_field = SpiralField("living_dialogue", composting_mode="resonant")
    
    # Simulate a conversation developing over time
    conversation_moments = [
        ("🌅", "hopeful", "I'm feeling optimistic about today"),
        ("☁️", "uncertain", "But there's also some worry"),
        ("🌈", "hopeful", "Maybe the worry and hope can coexist"),
        ("🤲", "accepting", "Learning to hold both feelings"),
        ("💫", "grateful", "Grateful for this insight"),
        ("🌱", "growing", "Feeling something new growing from this"),
    ]
    
    print("Adding conversation moments one by one...")
    
    for symbol, emotion, description in conversation_moments:
        print(f"\n💭 {description}")
        
        # Create the pulse
        pulse = conversation_field.emit(symbol, emotion, amplitude=0.8, decay_rate=0.06)
        
        # Let it settle and check for immediate resonances
        time.sleep(1)
        
        recent_resonances = conversation_field.find_resonances(min_strength=0.5)
        if recent_resonances:
            print(f"   🌊 Resonates with {len(recent_resonances)} existing pulse(s):")
            for res in recent_resonances[-2:]:  # Show most recent
                print(f"      {res['resonance']['poetic_trace']}")
        
        # Show field evolution
        print(f"   Field now holds {len(conversation_field.pulses)} pulses")
        print(f"   Total resonance: {conversation_field.resonance_field():.2f}")
        
        # Occasional composting to show natural flow
        if len(conversation_field.pulses) > 4:
            composted = conversation_field.compost(threshold=0.02)
            if composted > 0:
                print(f"   🍂 {composted} faded pulse(s) released naturally")
    
    print(f"\nFinal conversation state:")
    print(f"Active pulses: {len(conversation_field.pulses)}")
    print(f"Total resonance: {conversation_field.resonance_field():.2f}")
    print(f"Surviving elements:")
    for pulse in conversation_field.pulses:
        print(f"  {pulse.symbol} [{pulse.emotion}] attention: {pulse.current_attention():.3f}")
    
    # Show final resonance web
    final_resonances = conversation_field.find_resonances(min_strength=0.3)
    if final_resonances:
        print(f"\nFinal resonance web ({len(final_resonances)} connections):")
        for res in final_resonances:
            strength = res['resonance']['strength']
            trace = res['resonance']['poetic_trace']
            print(f"  {strength:.2f}: {trace}")
    
    print("\n" + "="*60 + "\n")

def main():
    """Run all resonance and ecosystem demonstrations."""
    print("🌀 Welcome to the Resonance Patterns Demo")
    print("This demonstration shows how contemplative computing creates")
    print("emergent behaviors through pulse resonances and field ecosystems.\n")
    
    try:
        demonstrate_pulse_resonances()
        time.sleep(2)
        
        demonstrate_field_ecosystems()
        time.sleep(2)
        
        demonstrate_living_conversation()
        
        print("🙏 All demonstrations complete.")
        print("\nWhat you've witnessed:")
        print("• Pulses that strengthen each other through meaningful resonance")
        print("• Fields that compost according to natural temporal cycles")
        print("• Emergent conversational behaviors in contemplative systems")
        print("• Living memory that breathes with organic time")
        print()
        print("This is contemplative computing - where technology learns")
        print("to participate in the deeper rhythms of meaning-making.")
        print()
        print("To explore interactively:")
        print("  python experimental/contemplative_journal.py")
        print("  python contemplative_repl.py")
        
    except KeyboardInterrupt:
        print("\n\n🌙 Demo interrupted gently. Even interruption teaches us")
        print("about the natural rhythm of attention and release...")

if __name__ == "__main__":
    main() 