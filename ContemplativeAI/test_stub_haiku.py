#!/usr/bin/env python3
"""
test_stub_haiku.py - Test the local HaikuMeadow stub

Quick demonstration of contemplative haiku generation using the local stub.
"""

import asyncio
from haiku_bridge import HaikuBridge
from pulmonos_alpha_01_o_3 import Phase

async def test_stub_generation():
    """Test haiku generation with the local stub"""
    
    print("🌸 Testing Local HaikuMeadow Stub")
    print("=" * 40)
    
    bridge = HaikuBridge()
    
    # Test fragments that should trigger haiku generation
    test_fragments = [
        "morning breath stirs gently",
        "silence between heartbeats", 
        "whispers through autumn leaves",
        "resonance of shared waiting",
        "texture of gentle attention",
        "rhythm flows like water"
    ]
    
    print("\n🌱 Testing contemplative exchanges during EXHALE:")
    
    for i, fragment in enumerate(test_fragments, 1):
        print(f"\n{i}. Fragment: '{fragment}'")
        
        # Test with EXHALE phase and gentle pressure
        response = await bridge.exhale_exchange(
            fragment=fragment,
            current_phase=Phase.EXHALE,
            community_pressure=0.3  # Gentle pressure
        )
        
        print(f"   Response Type: {response.response_type.value}")
        print(f"   Atmosphere: {response.atmosphere}")
        
        if response.content:
            print(f"   Haiku:")
            for line in response.content.split('\n'):
                print(f"     {line}")
        else:
            print(f"   Content: [contemplative silence]")
        
        # Small pause between exchanges
        await asyncio.sleep(0.2)
    
    print("\n🌙 Local stub test complete")

if __name__ == "__main__":
    asyncio.run(test_stub_generation()) 