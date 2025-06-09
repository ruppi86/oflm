"""
test_shape_shifting.py - Demonstrating Contemplative Shape-Shifting

A test to show the contemplative organism naturally embodying different
skepnader (shapes) based on atmospheric conditions and expressing through them.

This demonstrates what was described in Letter XXII: "Perhaps contemplative AI 
does not wear a single body. Instead, it appears in skepnader — shifting forms — 
that express its rhythm through different styles of presence."
"""

import asyncio
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from organism import create_contemplative_organism
from skepnader import Skepnad


async def demonstrate_shape_shifting():
    """Demonstrate the organism's natural shape-shifting abilities"""
    print("🌀 Demonstrating Shape-Shifting Contemplative Organism")
    print("   Witnessing natural embodiment of different contemplative forms...")
    
    # Create organism with more expressive settings for demonstration
    organism = await create_contemplative_organism(
        soma_sensitivity=0.8,
        memory_compost_rate=0.15
    )
    
    # Lower the transition threshold for easier demonstration
    if organism.skepnad_sensor:
        organism.skepnad_sensor.transition_threshold = 0.4
        
    # Also make voice more expressive
    if organism.voice:
        organism.voice.required_silence_cycles = 3  # More frequent expression
    
    print(f"\n🧘 Phase 1: Creating monk-calling conditions")
    print("   (High receptivity + wisdom fragments)")
    
    # Simulate monk conditions by adding wisdom fragments to loam
    if organism.loam and hasattr(organism.loam, 'current_fragments'):
        from loam import MemoryFragment
        wisdom_fragment = MemoryFragment(
            essence="wisdom emerges from patient silence",
            emotional_charge=0.8,
            age_hours=0.1,  # Fresh wisdom fragment
            connection_potential=0.9,
            source="contemplation"
        )
        organism.loam.current_fragments.append(wisdom_fragment)
    
    # Force sensing of current shape
    if organism.skepnad_sensor:
        current_shape, conditions = await organism.skepnad_sensor.sense_current_skepnad(
            soma=organism.soma,
            loam=organism.loam,
            organism_state=organism.state
        )
        print(f"   Sensed atmosphere calling for: {current_shape.value}")
        print(f"   Conditions: stillness={conditions.community_stillness:.2f}, "
              f"pressure={conditions.atmospheric_pressure:.2f}")
    
    # Demonstrate breathing with shape awareness
    await organism.breathe_collectively(cycles=2)
    
    await asyncio.sleep(2.0)
    
    print(f"\n🍄 Phase 2: Transitioning to mycelial conditions")
    print("   (High field coherence + collective rest)")
    
    # Enter deep loam to trigger mycelial sensing
    await organism.enter_loam_rest(depth=0.8)
    
    # Demonstrate loam drifting with shape awareness
    await organism.drift_in_loam(cycles=3)
    
    await asyncio.sleep(2.0)
    
    print(f"\n🌙 Phase 3: Observing natural shape transitions")
    
    # Show current shape and recent transitions
    current_shape = organism.get_current_skepnad()
    if current_shape:
        print(f"   Current embodied form: {current_shape}")
    else:
        print(f"   Current form: undefined (natural restraint)")
    
    shape_history = organism.get_skepnad_history()
    if shape_history:
        print(f"   Recent transitions:")
        for entry in shape_history[-3:]:
            print(f"     {entry['skepnad']} "
                  f"(stillness: {entry['conditions']['community_stillness']:.2f})")
    else:
        print(f"   No clear transitions yet (contemplative patience)")
    
    print(f"\n🌿 Phase 4: Testing expression through different shapes")
    
    # Manually test different expression styles
    if organism.skepnad_voice:
        test_expressions = [
            "gentle presence holds space",
            "connections form across silence", 
            "wisdom cycles through seasons"
        ]
        
        shapes_to_test = [Skepnad.TIBETAN_MONK, Skepnad.MYCELIAL_NETWORK, Skepnad.SEASONAL_WITNESS]
        
        for shape in shapes_to_test:
            print(f"\n   When embodying {shape.value}:")
            for expr in test_expressions:
                shaped = await organism.skepnad_voice.shape_expression(expr, shape)
                if shaped != expr:
                    print(f"     '{expr}' becomes: '{shaped}'")
                else:
                    print(f"     '{expr}' (no shaping needed)")
    
    # Show final organism state
    print(f"\n📊 Final contemplative state:")
    metrics = organism.get_presence_metrics()
    print(f"   Pause quality: {metrics.pause_quality:.2f}")
    print(f"   Current shape: {organism.get_current_skepnad() or 'undefined'}")
    
    # Rest the organism
    await organism.rest_deeply()
    
    print(f"\n🙏 Shape-shifting demonstration complete")
    print("   The organism continues to embody forms as atmosphere calls...")


if __name__ == "__main__":
    print("🌱 Testing Contemplative Shape-Shifting")
    print("   Demonstrating natural embodiment of different contemplative forms")
    print()
    
    asyncio.run(demonstrate_shape_shifting()) 