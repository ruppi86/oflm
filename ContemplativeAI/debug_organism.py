#!/usr/bin/env python3
"""Debug script to check organism.py imports"""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print("🔍 Debugging organism.py imports...")

# Test each import step by step
print("\n1. Testing pulmonos import directly:")
try:
    from pulmonos_alpha_01_o_3 import Phase as BreathPhase, BreathConfig, PHASE_ORDER
    print("✅ Pulmonos components imported successfully")
    print(f"   Phase: {BreathPhase}")
    print(f"   BreathConfig: {BreathConfig}")
    print(f"   PHASE_ORDER: {PHASE_ORDER}")
except Exception as e:
    print(f"❌ Pulmonos import failed: {e}")

print("\n2. Testing Pulmonos class creation:")
try:
    from pulmonos_alpha_01_o_3 import Phase as BreathPhase, BreathConfig, PHASE_ORDER
    
    class TestPulmonos:
        def __init__(self, breath_rhythm):
            self.config = BreathConfig(
                inhale=breath_rhythm.get("inhale", 2.0),
                hold=breath_rhythm.get("hold", 1.0),
                exhale=breath_rhythm.get("exhale", 2.0),
                rest=breath_rhythm.get("rest", 1.0)
            )
            print(f"   Created BreathConfig: {self.config}")
            
        async def broadcast_breathing(self, cycles):
            for cycle in range(cycles):
                for phase in PHASE_ORDER:
                    yield phase
                    
    test_pulmonos = TestPulmonos({"inhale": 2.0, "hold": 1.0, "exhale": 2.0, "rest": 1.0})
    print("✅ Pulmonos class creation works")
    
except Exception as e:
    print(f"❌ Pulmonos class creation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n3. Testing organism import:")
try:
    import organism
    print(f"✅ Organism module imported")
    print(f"   organism.Pulmonos: {organism.Pulmonos}")
    print(f"   organism.BreathPhase: {organism.BreathPhase}")
except Exception as e:
    print(f"❌ Organism import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n4. Testing organism creation:")
try:
    import asyncio
    from organism import create_contemplative_organism
    
    async def test_creation():
        print("   Creating organism...")
        organism = await create_contemplative_organism()
        print(f"   Organism created: {organism}")
        print(f"   Organism.pulmonos: {organism.pulmonos}")
        return organism
    
    organism = asyncio.run(test_creation())
    print("✅ Organism creation works")
    
except Exception as e:
    print(f"❌ Organism creation failed: {e}")
    import traceback
    traceback.print_exc() 