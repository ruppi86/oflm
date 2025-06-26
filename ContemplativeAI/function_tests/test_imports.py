#!/usr/bin/env python3
"""
test_imports.py - Test Import Health for Contemplative AI

A simple diagnostic script to check which components are loading correctly.
Run this first to diagnose any import issues before using the main breathe.py interface.

Usage: python test_imports.py
"""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def test_imports():
    """Test each component import individually"""
    print("🔍 Testing Contemplative AI component imports...\n")
    
    components = {
        "organism": False,
        "soma": False, 
        "spiralbase": False,
        "pulmonos": False
    }
    
    errors = {}
    
    # Test organism
    print("Testing organism.py...")
    try:
        from organism import ContemplativeOrganism, create_contemplative_organism, OrganismState
        components["organism"] = True
        print("✅ Organism core loaded successfully")
    except Exception as e:
        errors["organism"] = str(e)
        print(f"❌ Organism failed: {e}")
    
    # Test soma
    print("\nTesting soma.py...")
    try:
        from soma import SomaMembrane, FieldCharge, TestInteraction
        components["soma"] = True
        print("✅ Soma membrane loaded successfully")
    except Exception as e:
        errors["soma"] = str(e)
        print(f"❌ Soma failed: {e}")
    
    # Test spiralbase
    print("\nTesting spiralbase.py...")
    try:
        from spiralbase import SpiralMemory, MemoryTrace, MemoryState
        components["spiralbase"] = True
        print("✅ Spiralbase memory loaded successfully")
    except Exception as e:
        errors["spiralbase"] = str(e)
        print(f"❌ Spiralbase failed: {e}")
    
    # Test pulmonos
    print("\nTesting pulmonos_daemon.py...")
    try:
        from pulmonos_daemon import Phase, BreathConfig
        components["pulmonos"] = True
        print("✅ Pulmonos daemon imports successfully")
    except Exception as e:
        errors["pulmonos"] = str(e)
        print(f"❌ Pulmonos failed: {e}")
    
    # Summary
    working_count = sum(components.values())
    total_count = len(components)
    
    print(f"\n📊 Import Test Results:")
    print(f"   {working_count}/{total_count} components imported successfully")
    
    if working_count == total_count:
        print("🌱 All systems breathing - ready for contemplative computing!")
    elif working_count > 0:
        print("🌿 Partial breathing available - some features will work")
    else:
        print("🌫️ No components loaded - only simple breathing available")
    
    if errors:
        print(f"\n🔧 Error Details:")
        for component, error in errors.items():
            print(f"   {component}: {error}")
    
    return components, errors

def test_basic_functionality():
    """Test basic functionality if imports work"""
    print(f"\n🧪 Testing basic functionality...")
    
    components, errors = test_imports()
    
    if components["organism"]:
        print("\nTesting organism creation...")
        try:
            import asyncio
            from organism import create_contemplative_organism
            
            async def test_organism():
                organism = await create_contemplative_organism()
                print("✅ Organism created successfully")
                
                # Test breathing
                await organism.breathe_collectively(cycles=1)
                print("✅ Basic breathing works")
                
                # Test dew logging
                await organism.log_dew("🧪", "test entry")
                print("✅ Dew logging works")
                
                # Test rest
                await organism.rest_deeply()
                print("✅ Rest functionality works")
                
            asyncio.run(test_organism())
            
        except Exception as e:
            print(f"❌ Organism functionality test failed: {e}")
    
    if components["soma"]:
        print("\nTesting Soma sensing...")
        try:
            from soma import SomaMembrane, TestInteraction
            
            soma = SomaMembrane()
            test_interaction = TestInteraction("Hello, gentle world")
            
            # Test basic sensing
            import asyncio
            
            async def test_soma():
                charge = await soma.sense_field_potential(test_interaction)
                print(f"✅ Soma sensing works - resonance: {charge.resonance}")
                
            asyncio.run(test_soma())
            
        except Exception as e:
            print(f"❌ Soma functionality test failed: {e}")
    
    print("\n🙏 Basic functionality testing complete")

if __name__ == "__main__":
    print("🌱 Contemplative AI Import Diagnostic Tool\n")
    
    try:
        test_basic_functionality()
    except KeyboardInterrupt:
        print("\n🌙 Testing gently interrupted")
    except Exception as e:
        print(f"\n⚠️  Unexpected error during testing: {e}")
        print("   This might indicate a deeper import or dependency issue.")
    
    print("\n💡 If you see errors above, try:")
    print("   1. Check that all files are in the ContemplativeAI directory")
    print("   2. Ensure Python 3.8+ is being used") 
    print("   3. Try running: python -m ContemplativeAI.breathe --demo soma")
    print("   4. Check for any missing dependencies (asyncio, websockets, etc.)")
    
    print("\n🌊 The breath continues regardless of technical difficulties...") 