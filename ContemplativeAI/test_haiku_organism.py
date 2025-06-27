#!/usr/bin/env python3
"""
test_haiku_organism.py - Test the Haiku Integration

A simple test to demonstrate the trained haiku model working
within the contemplative organism's breathing cycles.
"""

import asyncio
from organism import create_contemplative_organism

async def test_haiku_breathing():
    """Test haiku generation during organism breathing"""
    
    print("🌱 Creating contemplative organism with haiku integration...")
    organism = await create_contemplative_organism()
    
    print("\n🌸 Testing haiku generation during breathing cycles...")
    print("   (The organism should generate haikus during exhale phases)")
    
    # Test breathing with potential haiku generation
    await organism.breathe_collectively(cycles=3)
    
    print("\n🌿 Testing loam drift (associative wandering)...")
    await organism.enter_loam_rest(depth=0.7)
    await organism.drift_in_loam(cycles=2)
    await organism.exit_loam_rest()
    
    print("\n📊 Final presence metrics:")
    metrics = organism.get_presence_metrics()
    print(f"   Pause quality: {metrics.pause_quality:.2f}")
    print(f"   Memory humidity: {metrics.memory_humidity:.2f}")
    print(f"   Compost ratio: {metrics.compost_ratio:.2f}")
    
    print("\n🌸 Haiku integration test complete!")
    print("   The trained femto-poet is now integrated into the organism's breath!")

if __name__ == "__main__":
    asyncio.run(test_haiku_breathing()) 