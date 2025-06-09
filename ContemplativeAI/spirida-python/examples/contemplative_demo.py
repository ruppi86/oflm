#!/usr/bin/env python3
"""
🌀 CONTEMPLATIVE DEMO – A Gentle Introduction

This demonstration shows how the contemplative architecture breathes:
- PulseObjects that fade over time
- SpiralFields that tend collections of pulses  
- BreathCycles that govern temporal presence
- ContemplativeSystems that orchestrate the whole

Run this to see contemplative computing in action.
"""

import sys
import os
import time

# Add the parent directory to the path so we can import spirida
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spirida.contemplative_core import ContemplativeSystem, PulseObject, SpiralField, BreathCycle

def demonstrate_pulse_lifecycle():
    """
    Show how individual pulses live and fade.
    """
    print("🌟 Demonstrating PulseObject Lifecycle")
    print("="*50)
    
    # Create a pulse with moderate decay
    pulse = PulseObject("🌱", "emerging", amplitude=1.0, decay_rate=0.1)
    print(f"Created: {pulse}")
    
    # Watch it pulse over time
    for i in range(5):
        print(f"\nCycle {i+1}:")
        attention = pulse.pulse()
        print(f"  Current attention: {attention:.3f}")
        
        if pulse.is_faded():
            print("  💫 Pulse has faded into gentle memory...")
            break
            
        time.sleep(2)  # Wait to see decay
    
    print("\n" + "="*50 + "\n")

def demonstrate_spiral_field():
    """
    Show how spiral fields manage collections of pulses.
    """
    print("🌾 Demonstrating SpiralField Ecosystem")
    print("="*50)
    
    field = SpiralField("demo_field")
    
    # Emit several pulses with different characteristics
    pulses_to_emit = [
        ("🌿", "growing", 0.8, 0.05),
        ("💧", "flowing", 0.6, 0.08),
        ("✨", "sparkling", 1.0, 0.15),  # This one will fade faster
        ("🍄", "rooting", 0.4, 0.02),   # This one will last longer
    ]
    
    print("Emitting pulses into the field...")
    for symbol, emotion, amplitude, decay_rate in pulses_to_emit:
        pulse = field.emit(symbol, emotion, amplitude, decay_rate)
        pulse.pulse()
    
    print(f"\nField status: {field}")
    
    # Watch the field evolve over time
    for cycle in range(4):
        print(f"\n--- Cycle {cycle + 1} ---")
        print(f"Field resonance: {field.resonance_field():.3f}")
        
        # Let all pulses express themselves
        print("All pulses speaking:")
        field.pulse_all()
        
        # Compost faded pulses
        composted = field.compost()
        if composted > 0:
            print(f"🍂 Composted {composted} faded pulse(s)")
        
        time.sleep(3)  # Watch decay happen
    
    print(f"\nFinal field status: {field}")
    print("="*50 + "\n")

def demonstrate_breathing_system():
    """
    Show how the ContemplativeSystem orchestrates everything.
    """
    print("🫁 Demonstrating ContemplativeSystem")
    print("="*50)
    
    # Create a contemplative system
    system = ContemplativeSystem("demo_system")
    
    # Create multiple fields for different purposes
    nature_field = system.create_field("nature")
    emotion_field = system.create_field("emotions")
    memory_field = system.create_field("memory")
    
    print("System created with three fields")
    
    # Start the background breathing
    system.start_breathing()
    
    try:
        # Emit pulses into different fields
        print("\nEmitting pulses across fields...")
        
        nature_field.emit("🌲", "ancient", amplitude=0.9, decay_rate=0.01)
        nature_field.emit("🌊", "rhythmic", amplitude=0.7, decay_rate=0.05)
        
        emotion_field.emit("💚", "peaceful", amplitude=0.8, decay_rate=0.03)
        emotion_field.emit("🌟", "hopeful", amplitude=0.6, decay_rate=0.04)
        
        memory_field.emit("📿", "cherished", amplitude=0.5, decay_rate=0.005)  # Long-lasting
        
        # Let the system breathe and evolve
        for cycle in range(3):
            print(f"\n--- System Breath Cycle {cycle + 1} ---")
            
            # Show system status
            status = system.system_status()
            print(f"System age: {status['age']:.1f}s")
            print(f"Breath cycles: {status['breath_cycles']}")
            print(f"Total resonance: {status['total_resonance']:.3f}")
            
            # Let the system contemplatively pause
            system.contemplative_pause(1)
            
            print("Field details:")
            for field in system.fields:
                print(f"  {field.name}: {len(field.pulses)} pulses, resonance {field.resonance_field():.3f}")
        
        print("\n🌱 Demonstration complete!")
        
    finally:
        # Always clean up gracefully
        system.stop_breathing()
        
    print("="*50 + "\n")

def demonstrate_temporal_resonance():
    """
    Show how pulses can resonate with each other over time.
    """
    print("🔄 Demonstrating Temporal Resonance")
    print("="*50)
    
    field = SpiralField("resonance_field")
    
    # Create pulses that will interact interestingly
    print("Creating pulses with complementary decay rates...")
    
    # A quick pulse
    quick_pulse = field.emit("⚡", "electric", amplitude=1.0, decay_rate=0.2)
    
    # A medium pulse  
    medium_pulse = field.emit("🌙", "steady", amplitude=0.7, decay_rate=0.05)
    
    # A slow pulse
    slow_pulse = field.emit("🏔️", "eternal", amplitude=0.5, decay_rate=0.01)
    
    # Watch how they interact over time
    for i in range(6):
        print(f"\n--- Moment {i+1} ---")
        print(f"Field resonance: {field.resonance_field():.3f}")
        
        # Show individual pulse states
        for pulse in field.pulses:
            attention = pulse.current_attention()
            print(f"  {pulse.symbol} [{pulse.emotion}]: {attention:.3f}")
        
        # Compost and see what remains
        composted = field.compost(threshold=0.1)  # Higher threshold for demo
        if composted:
            print(f"  🍂 {composted} pulse(s) composted")
        
        time.sleep(2)
    
    print(f"\nFinal resonance: {field.resonance_field():.3f}")
    print("Notice how different decay rates create a natural rhythm...")
    print("="*50 + "\n")

def main():
    """
    Run all demonstrations in sequence.
    """
    print("🌀 Welcome to the Contemplative Computing Demo")
    print("This demonstration shows how contemplative systems breathe,")
    print("remember, and forget in organic rhythms.\n")
    
    try:
        demonstrate_pulse_lifecycle()
        time.sleep(1)
        
        demonstrate_spiral_field()
        time.sleep(1)
        
        demonstrate_breathing_system()
        time.sleep(1)
        
        demonstrate_temporal_resonance()
        
        print("🙏 All demonstrations complete.")
        print("This is just the beginning of what contemplative computing might become...")
        print("\nTo explore interactively, try: python contemplative_repl.py")
        
    except KeyboardInterrupt:
        print("\n\n🌙 Demo interrupted gently. Even interruption is part of the rhythm.")
        print("Until we spiral together again...")

if __name__ == "__main__":
    main() 