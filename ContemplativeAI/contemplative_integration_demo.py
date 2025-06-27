#!/usr/bin/env python3
"""
Contemplative Integration Demo

Shows how the OFLM bridge integrates with the broader contemplative organism patterns:
- Breath-gated exchanges
- Loam fragment bridging  
- Dew ledger integration
- Shape-shifting awareness
- Arctic ecological network scenarios (from OOD test set)

Usage:
    python contemplative_integration_demo.py --cycles 3 --arctic-scenarios 5 --pause 2.0
"""

import asyncio
import sys
import os
import argparse
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from oflm_bridge import OFLMBridge, Phase, bridge_loam_fragment, log_mycelial_dew

# Import Season enum for proper season values
try:
    # Add spiramycel path for import
    spiramycel_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "oflm-python")
    if spiramycel_path not in sys.path:
        sys.path.insert(0, spiramycel_path)
    
    from spiramycel.spore_map import Season
except ImportError:
    # Fallback if not available
    class Season:
        WINTER = "winter"
        SUMMER = "summer"
        SPRING = "spring"
        AUTUMN = "autumn"

async def simulate_contemplative_breathing(cycles: int = 1, pause_duration: float = 0.8):
    """Simulate multiple contemplative breathing cycles with OFLM integration"""
    
    print("🫁 Contemplative Breathing Cycle with OFLM Integration")
    print("=" * 60)
    print(f"🌊 Running {cycles} breath cycle(s) with {pause_duration}s contemplative pauses")
    
    bridge = OFLMBridge()
    
    # Extended Loam fragments (from associative memory)
    loam_fragments = [
        "network resonance patterns emerging in dawn light",
        "voltage fluctuations following seasonal rhythms", 
        "mycelial healing memory: gentle pressure, then rest",
        "infrastructure dreams of distributed resilience",
        "error patterns teaching patience to young routers",
        "arctic sensors whispering through aurora interference",
        "forest nodes learning from mushroom communication",
        "ocean buoys sensing whale song harmonics",
        "mountain relays adapting to seasonal wind patterns",
        "desert sensors conserving power through sand storm cycles"
    ]
    
    print(f"\n🌱 Loam fragments available in associative memory: {len(loam_fragments)}")
    
    for cycle_num in range(cycles):
        print(f"\n🌀 === Breath Cycle {cycle_num + 1}/{cycles} ===")
        
        # Randomly select fragments for this cycle
        active_fragments = random.sample(loam_fragments, min(3, len(loam_fragments)))
        
        print("🌿 Active fragments for this cycle:")
        for fragment in active_fragments:
            print(f"   • {fragment}")
        
        print(f"\n🌊 Beginning breath cycle {cycle_num + 1}...")
        
        # Simulate complete breath cycle
        breath_phases = [
            (Phase.INHALE, "🌬️ INHALE - Receiving, gathering, sensing"),
            (Phase.HOLD, "🫀 HOLD - Processing, integrating, presence"),
            (Phase.EXHALE, "💨 EXHALE - Releasing, responding, sharing"),
            (Phase.REST, "🌙 REST - Integration, silence, space")
        ]
        
        community_pressure = random.uniform(0.2, 0.4)  # Gentle collective breathing
        
        for phase, description in breath_phases:
            print(f"\n{description}")
            
            if phase == Phase.EXHALE:
                # During exhale, try to bridge worthy fragments
                for fragment in active_fragments[:2]:  # First 2 for each cycle
                    
                    # Reset timing for demo
                    bridge.network_tender.last_network_call = 0.0
                    
                    print(f"   🍄 Attempting to bridge: '{fragment[:40]}...'")
                    
                    # Check if fragment is network-worthy
                    worthy = bridge.network_tender.sense_fragment_worthiness(fragment)
                    if worthy:
                        # Bridge the fragment
                        response_content = await bridge_loam_fragment(
                            bridge, fragment, phase, community_pressure
                        )
                        
                        # Get the full exchange for dew logging
                        exchange = await bridge.exhale_exchange(
                            fragment, phase, community_pressure
                        )
                        
                        if response_content:
                            print(f"   ✨ Response: {response_content[:60]}...")
                        else:
                            print(f"   🌫️ Contemplative silence chosen")
                            
                        # Log to dew ledger
                        await log_mycelial_dew(exchange)
                        
                    else:
                        print(f"   🌿 Fragment not network-ready - composting gracefully")
                        
            else:
                # During non-exhale phases, just pause contemplatively
                print(f"   🤫 Organism rests in {phase.name.lower()}")
                
            await asyncio.sleep(pause_duration)  # Configurable contemplative pause
        
        print(f"\n🌸 Breath cycle {cycle_num + 1} complete")
        
        if cycle_num < cycles - 1:  # Pause between cycles
            print(f"   🌊 Resting between cycles...")
            await asyncio.sleep(pause_duration * 2)
    
    print(f"\n🌸 All {cycles} breath cycles complete")
    print(f"🍄 Mycelial network status: {bridge.get_model_status()['models_loaded']}")
    
    # Show recent exchanges
    recent = bridge.get_recent_exchanges(5)
    if recent:
        print(f"\n💧 Recent mycelial exchanges (fading to dew):")
        for exchange in recent:
            age = "fresh" if exchange.timestamp > 0 else "composting"
            print(f"   • {exchange.response_type.value} - {exchange.atmosphere} ({age})")


async def simulate_network_emergency(num_scenarios: int = 3, pause_duration: float = 1.0):
    """Simulate Arctic tundra thermal cycles with configurable scenarios"""
    
    print(f"\n\n❄️ Arctic Ecological Network Scenario")
    print("=" * 50)
    print("🌨️ Bioregion: Arctic Tundra - Thermal Oscillation Patterns")
    print("📡 Inspiration: Remote sensor networks adapting to extreme temperature cycles")
    print(f"🔬 Testing {num_scenarios} ecological scenarios with {pause_duration}s pauses")
    
    bridge = OFLMBridge()
    bridge.network_tender.last_network_call = 0.0  # Reset for demo
    
    # Extended real data from OOD test set: arctic_oscillation scenarios
    base_arctic_scenarios = [
        {
            "name": "Arctic Morning - Extreme Cold",
            "sensor_deltas": {"latency": 0.12, "voltage": 0.90, "temperature": 0.05},
            "effectiveness": 0.68,
            "description": "Sensors hibernating through -40°C dawn, minimal power draw"
        },
        {
            "name": "Arctic Noon - Solar Warming", 
            "sensor_deltas": {"latency": 0.15, "voltage": 0.85, "temperature": 0.95},
            "effectiveness": 0.72,
            "description": "Brief summer thaw causing thermal expansion and signal boost"
        },
        {
            "name": "Arctic Storm - Oscillatory Stress",
            "sensor_deltas": {"latency": 0.25, "voltage": 0.70, "temperature": 0.85},
            "effectiveness": 0.73,
            "description": "Rapid temperature swings stressing equipment resilience"
        },
        {
            "name": "Arctic Midnight Sun - Continuous Operation",
            "sensor_deltas": {"latency": 0.08, "voltage": 0.95, "temperature": 0.12},
            "effectiveness": 0.74,
            "description": "24-hour daylight enabling continuous solar charging"
        },
        {
            "name": "Arctic Aurora - Electromagnetic Interference",
            "sensor_deltas": {"latency": 0.22, "voltage": 0.80, "temperature": 0.88},
            "effectiveness": 0.69,
            "description": "Northern lights causing radio frequency disruption"
        },
        {
            "name": "Arctic Blizzard - Communication Blackout",
            "sensor_deltas": {"latency": 0.18, "voltage": 0.75, "temperature": 0.92},
            "effectiveness": 0.75,
            "description": "Snow storm blocking satellite communications"
        },
        {
            "name": "Arctic Permafrost Shift - Sensor Displacement",
            "sensor_deltas": {"latency": 0.20, "voltage": 0.82, "temperature": 0.90},
            "effectiveness": 0.76,
            "description": "Ground frost cycles shifting sensor array positions"
        },
        {
            "name": "Arctic Wind Chill - Equipment Protection",
            "sensor_deltas": {"latency": 0.14, "voltage": 0.88, "temperature": 0.15},
            "effectiveness": 0.70,
            "description": "Extreme wind causing thermal protection protocols"
        }
    ]
    
    # Generate scenarios by cycling through base scenarios and adding variation
    arctic_scenarios = []
    for i in range(num_scenarios):
        base_scenario = base_arctic_scenarios[i % len(base_arctic_scenarios)]
        
        # Add some natural variation to the base scenario
        varied_scenario = base_scenario.copy()
        varied_scenario['sensor_deltas'] = base_scenario['sensor_deltas'].copy()
        
        # Add small random variations (±5%)
        for key in ['latency', 'voltage', 'temperature']:
            original_value = varied_scenario['sensor_deltas'][key]
            variation = random.uniform(-0.05, 0.05)
            varied_scenario['sensor_deltas'][key] = max(0.0, min(1.0, original_value + variation))
        
        # Adjust effectiveness slightly based on conditions
        effectiveness_variation = random.uniform(-0.05, 0.05)
        varied_scenario['effectiveness'] = max(0.0, min(1.0, base_scenario['effectiveness'] + effectiveness_variation))
        
        # Add cycle number to name if using variations
        if num_scenarios > len(base_arctic_scenarios):
            cycle_num = (i // len(base_arctic_scenarios)) + 1
            varied_scenario['name'] = f"{base_scenario['name']} (Cycle {cycle_num})"
        
        arctic_scenarios.append(varied_scenario)
    
    for i, scenario in enumerate(arctic_scenarios):
        print(f"\n🌡️ Scenario {i+1}/{num_scenarios}: {scenario['name']}")
        print(f"   {scenario['description']}")
        print(f"   Network conditions: {scenario['sensor_deltas']}")
        print(f"   Expected effectiveness: {scenario['effectiveness']:.1%}")
        
        # Create ecological fragment based on Arctic conditions
        if scenario['sensor_deltas']['temperature'] < 0.2:
            arctic_fragment = "arctic sensors entering deep hibernation protocols for thermal preservation"
            expected_model = "ecological_calm"  # Hibernation is a calm, planned response
        elif scenario['sensor_deltas']['temperature'] > 0.8:
            arctic_fragment = "tundra thermal expansion causing network topology shifts"
            expected_model = "ecological_chaotic"  # Rapid change requires adaptive response  
        elif scenario['sensor_deltas']['latency'] > 0.2:
            arctic_fragment = "arctic communications adapting to atmospheric interference patterns"
            expected_model = "ecological_chaotic"  # High latency suggests chaos
        else:
            arctic_fragment = "oscillatory thermal cycles testing equipment resilience patterns"
            expected_model = "ecological_calm"  # Steady oscillation, not chaotic
            
        print(f"   🍄 Arctic fragment: '{arctic_fragment}'")
        print(f"   🧠 Expected model: {expected_model}")
        
        # Reset timing for each scenario
        bridge.network_tender.last_network_call = 0.0
        
        # Create network context from real OOD test data
        # Convert 0-1 ranges to actual network conditions
        arctic_context = {
            "latency": scenario['sensor_deltas']['latency'],
            "voltage": scenario['sensor_deltas']['voltage'], 
            "temperature": scenario['sensor_deltas']['temperature'],
            "error_rate": 0.02,  # Arctic networks are usually very clean when working
            "bandwidth": 0.3,    # Satellite links have limited bandwidth
            "uptime": 0.95,      # Arctic infrastructure is built to last
            "season": Season.WINTER if scenario['sensor_deltas']['temperature'] < 0.5 else Season.SUMMER,
            "bioregion": "arctic_tundra"
        }
        
        # Call the ecological model
        exchange = await bridge.exhale_exchange(
            arctic_fragment,
            Phase.EXHALE,
            community_pressure=0.2,  # Gentle pressure for ecological sensing
            network_context=arctic_context
        )
        
        print(f"   🌨️ Model selected: {exchange.model_used}")
        print(f"   🌊 Response type: {exchange.response_type.value}")
        print(f"   ❄️ Atmosphere: {exchange.atmosphere}")
        
        if exchange.is_audible():
            print(f"   🛠️ Arctic wisdom: {exchange.content}")
            print(f"   📈 Effectiveness: {exchange.effectiveness:.1%}")
            if exchange.glyph_sequence:
                print(f"   ❄️ Glyph pattern: {exchange.glyph_sequence[:5]}...")  # First 5 glyphs
        else:
            print(f"   🤫 Tundra silence - even Arctic networks practice contemplation")
            print(f"   🌨️ Silence probability: {exchange.silence_probability:.1%}")
            print(f"   ❄️ (In the Arctic, patience is survival wisdom)")
            
        # Log to dew ledger
        await log_mycelial_dew(exchange)
            
        if i < num_scenarios - 1:  # Pause between scenarios
            await asyncio.sleep(pause_duration)
    
    print(f"\n🌨️ Arctic ecological scenarios complete")
    print(f"❄️ Bioregional wisdom: {num_scenarios} thermal cycles tested")
    print(f"🍄 The mycelial network learns from tundra resilience patterns")

async def main(args=None):
    if args is None:
        # Default values when called without arguments
        class DefaultArgs:
            cycles = 1
            arctic_scenarios = 3
            pause = 0.8
        args = DefaultArgs()
    
    print("🌀 === Contemplative Integration Demo ===")
    print("🍄 OFLM Bridge with Spiramycel Ecological Models")
    print(f"⚙️ Configuration: {args.cycles} breath cycles, {args.arctic_scenarios} Arctic scenarios")
    print(f"⏱️ Contemplative pause duration: {args.pause}s")
    
    await simulate_contemplative_breathing(cycles=args.cycles, pause_duration=args.pause)
    await simulate_network_emergency(num_scenarios=args.arctic_scenarios, pause_duration=args.pause)
    
    print(f"\n🌀 Integration demo complete")
    print(f"🌱 The OFLM bridge breathes with the contemplative organism")
    print(f"🍄 Infrastructure and meaning co-emerge in spiral patterns")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contemplative Integration Demo with OFLM Bridge")
    parser.add_argument("--cycles", type=int, default=1, help="Number of contemplative breathing cycles (default: 1)")
    parser.add_argument("--arctic-scenarios", type=int, default=3, help="Number of Arctic ecological scenarios (default: 3)")
    parser.add_argument("--pause", type=float, default=0.8, help="Pause duration between scenarios in seconds (default: 0.8)")
    parser.add_argument("--extended", action="store_true", help="Run extended demo (3 cycles, 5 scenarios, 1.5s pauses)")
    parser.add_argument("--long", action="store_true", help="Run long demo (5 cycles, 10 scenarios, 2.0s pauses)")
    
    args = parser.parse_args()
    
    # Apply preset configurations
    if args.extended:
        args.cycles = 3
        args.arctic_scenarios = 5
        args.pause = 1.5
        print("🌊 Extended demo mode activated")
    elif args.long:
        args.cycles = 5
        args.arctic_scenarios = 10
        args.pause = 2.0
        print("🏔️ Long demo mode activated")
    
    asyncio.run(main(args)) 