#!/usr/bin/env python3
"""
Quick test of OFLM bridge with direct model interaction
"""

import asyncio
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from oflm_bridge import OFLMBridge, Phase

async def demo_oflm_responses():
    """Demonstrate actual OFLM model responses"""
    
    print("🍄 OFLM Bridge Direct Model Demo")
    print("=" * 50)
    
    bridge = OFLMBridge()
    
    # Temporarily reduce timing constraints for demo
    bridge.network_tender.last_network_call = 0.0
    
    # Test scenarios with network context
    scenarios = [
        {
            "fragment": "voltage drop causing network instability",
            "context": {"voltage": 0.3, "error_rate": 0.15, "temperature": 0.8},
            "description": "🔋 Power Crisis Scenario"
        },
        {
            "fragment": "gentle maintenance check for optimal performance", 
            "context": {"voltage": 0.7, "error_rate": 0.01, "temperature": 0.3},
            "description": "🌿 Calm Maintenance Scenario"
        },
        {
            "fragment": "critical system failure emergency response needed",
            "context": {"latency": 0.9, "error_rate": 0.4, "bandwidth": 0.1},
            "description": "⚡ Chaos Emergency Scenario"
        },
        {
            "fragment": "seasonal patterns in network resilience",
            "context": {"temperature": 0.5, "uptime": 0.95},
            "description": "🌱 Ecological Wisdom Query"
        }
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"\n{scenario['description']}")
        print(f"Fragment: '{scenario['fragment']}'")
        print(f"Context: {scenario['context']}")
        
        # Reset timing to allow call
        bridge.network_tender.last_network_call = 0.0
        
        # Make the call
        exchange = await bridge.exhale_exchange(
            scenario['fragment'],
            Phase.EXHALE,
            community_pressure=0.2,  # Gentle pressure
            network_context=scenario['context']
        )
        
        print(f"Response Type: {exchange.response_type.value}")
        print(f"Model Used: {exchange.model_used or 'simulation'}")
        print(f"Atmosphere: {exchange.atmosphere}")
        
        if exchange.is_audible():
            print(f"Content: {exchange.content}")
            if exchange.effectiveness > 0:
                print(f"Effectiveness: {exchange.effectiveness:.2f}")
            if exchange.glyph_sequence:
                print(f"Glyph Sequence: {exchange.glyph_sequence}")
        else:
            print("Content: [contemplative silence]")
            
        if exchange.silence_probability > 0.5:
            print(f"Silence Probability: {exchange.silence_probability:.1%}")
            
        # Add a small delay between scenarios
        await asyncio.sleep(0.5)
    
    print(f"\n🌸 Demo complete - mycelial network practicing contemplative repair")

if __name__ == "__main__":
    asyncio.run(demo_oflm_responses()) 