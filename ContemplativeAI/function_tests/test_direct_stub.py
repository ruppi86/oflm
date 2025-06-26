#!/usr/bin/env python3
"""
test_direct_stub.py - Direct test of the generator_stub

Shows contemplative haiku generation without bridge rate limiting.
"""

from generator_stub import HaikuMeadow

def test_direct_generation():
    """Test direct haiku generation with the stub"""
    
    print("🌸 Direct HaikuMeadow Stub Test")
    print("=" * 35)
    
    # Create the stub meadow
    meadow = HaikuMeadow(force_template_mode=True)
    
    # Test fragments
    test_fragments = [
        "morning breath stirs gently",
        "silence between heartbeats", 
        "whispers through autumn leaves",
        "resonance of shared waiting", 
        "texture of gentle attention",
        "rhythm flows like water",
        "breath carries scent",
        "patterns emerge slow",
        "moisture gathers wisdom"
    ]
    
    print(f"\n🌱 Generating contemplative haiku from fragments:")
    
    for i, fragment in enumerate(test_fragments, 1):
        haiku, generation_type = meadow.generate_haiku(fragment)
        
        print(f"\n{i}. Fragment: '{fragment}'")
        
        if haiku:
            print(f"   Generated ({generation_type}):")
            for line in haiku.split('\n'):
                print(f"     {line}")
        else:
            print(f"   Result: contemplative silence ({generation_type})")
    
    print(f"\n🌙 Direct stub test complete")
    print(f"   This demonstrates the local contemplative haiku generation")
    print(f"   that's now available when HaikuMeadowLib is not present.")

if __name__ == "__main__":
    test_direct_generation() 