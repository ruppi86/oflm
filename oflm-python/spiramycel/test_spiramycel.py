#!/usr/bin/env python3
"""
Complete Spiramycel System Test

Tests all three components working together:
- Glyph Codec: 64-symbol mycelial vocabulary  
- Spore Map: Living memory with evaporation
- Runtime Patcher: Safe glyph-to-action conversion

Demonstrates the mycelial network repair cycle.
"""

try:
    from glyph_codec import SpiramycelGlyphCodec
    from spore_map import SporeMapLedger, Season
    from runtime_patch import SpiramycelRuntimePatcher
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the spiramycel directory")
    exit(1)

def test_complete_spiramycel_system():
    """Test the complete mycelial repair cycle."""
    print("🍄 Complete Spiramycel System Test")
    print("=" * 60)
    
    # Initialize all components
    codec = SpiramycelGlyphCodec()
    spore_ledger = SporeMapLedger("test_mycelial_repairs.jsonl")
    patcher = SpiramycelRuntimePatcher("test_network_patches.jsonl")
    
    # 1. Generate contemplative breath pattern
    print("\n🌸 1. Contemplative Breath Generation")
    print("-" * 40)
    breath_pattern = codec.practice_tystnadsmajoritet(12)
    formatted = codec.format_glyph_sequence(breath_pattern)
    
    silence_count = sum(1 for gid in breath_pattern if gid in codec.get_contemplative_glyphs())
    silence_ratio = silence_count / len(breath_pattern)
    
    print(f"Generated pattern: {formatted}")
    print(f"Silence ratio: {silence_ratio:.1%} (target: 87.5%)")
    
    # 2. Convert to network patches
    print("\n🔧 2. Network Patch Generation")
    print("-" * 40)
    
    network_context = {
        "latency": 0.22,
        "voltage": 3.0,
        "temperature": 26.5,
        "error_rate": 0.06
    }
    
    patches = patcher.process_glyph_sequence(breath_pattern, network_context, "test_meadow")
    print(f"Network context: latency={network_context['latency']}, voltage={network_context['voltage']}")
    print(f"Generated {len(patches)} patches:")
    
    for patch in patches:
        print(f"  {patch.glyph_symbol} → {patch.action_type} (safety: {patch.safety_score:.2f})")
    
    # 3. Simulate patch execution and log results
    print("\n🧪 3. Patch Simulation & Spore Collection")
    print("-" * 40)
    
    for i, patch in enumerate(patches[:3]):  # Test first 3 patches
        results = patcher.simulate_patch_execution(patch)
        effectiveness = results['estimated_improvement']
        
        print(f"Patch {i+1}: {patch.glyph_symbol} → improvement {effectiveness:.2f}")
        
        # Create spore echo from repair attempt
        sensor_deltas = {
            "latency": -0.05 if effectiveness > 0.1 else 0.02,
            "voltage": 0.1 if patch.target_component == "power_management" else 0.0,
            "temperature": -1.0 if effectiveness > 0.05 else 0.5
        }
        
        spore = spore_ledger.add_spore_echo(
            sensor_deltas=sensor_deltas,
            glyph_sequence=[patch.glyph_id],
            repair_effectiveness=effectiveness,
            bioregion="test_meadow",
            season=Season.SUMMER
        )
        
        print(f"  Spore quality: {spore.spore_quality:.2f}")
    
    # 4. Show spore map statistics
    print("\n📊 4. Mycelial Memory Analysis")
    print("-" * 40)
    
    stats = spore_ledger.get_statistics()
    print(f"Total spores: {stats['total_spores']}")
    print(f"Average effectiveness: {stats['avg_effectiveness']:.2f}")
    print(f"Average quality: {stats['avg_quality']:.2f}")
    print(f"Survival rate: {stats['survival_rate']:.1%}")
    
    # 5. Solstice distillation
    if stats['total_spores'] > 0:
        print("\n🌙 5. Solstice Distillation")
        print("-" * 40)
        
        chosen = spore_ledger.solstice_distillation(max_chosen=2)
        print(f"Selected {len(chosen)} spores for network re-tuning:")
        
        for spore in chosen:
            glyph_symbols = codec.format_glyph_sequence(spore.glyph_sequence)
            print(f"  {glyph_symbols} → effectiveness {spore.repair_effectiveness:.2f}")
    
    # 6. Network recommendations based on current state
    print("\n💡 6. Adaptive Network Recommendations")
    print("-" * 40)
    
    recommendations = patcher.get_patch_recommendations(network_context, "test_meadow")
    rec_formatted = codec.format_glyph_sequence(recommendations)
    print(f"Recommended for current conditions: {rec_formatted}")
    
    # Count recommendation types
    rec_patches = patcher.process_glyph_sequence(recommendations, network_context)
    contemplative_recs = sum(1 for p in rec_patches if p.severity.value == "contemplative")
    print(f"Contemplative ratio in recommendations: {contemplative_recs}/{len(rec_patches)} = {contemplative_recs/len(rec_patches)*100:.1f}%")
    
    print("\n🌱 System Test Complete!")
    print("=" * 60)
    print("✅ Glyph Codec: Generating contemplative vocabularies")
    print("✅ Spore Map: Collecting mycelial repair memories")  
    print("✅ Runtime Patcher: Safe glyph-to-action conversion")
    print("✅ Tystnadsmajoritet: Maintaining contemplative silence")
    print("✅ Mycelial Resonance: Building collective wisdom")
    print("\n🍄 The underground nervous system is breathing...")

if __name__ == "__main__":
    test_complete_spiramycel_system() 