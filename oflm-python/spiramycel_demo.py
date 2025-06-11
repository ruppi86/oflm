#!/usr/bin/env python3
"""
Spiramycel Package Demo

Demonstrates the complete Organic Femto Language Model
working as an importable Python package.

Shows the mycelial network repair cycle:
1. Generate contemplative glyph patterns
2. Convert to safe network patches  
3. Collect repair memories as spore echoes
4. Build collective wisdom through solstice distillation
"""

import spiramycel

def main():
    print("🍄 Spiramycel: Organic Femto Language Model")
    print("=" * 60)
    
    # System information
    info = spiramycel.get_system_info()
    print(f"Version: {info['version']}")
    print(f"Philosophy: {info['philosophy']}")
    print()
    
    # Core principle demonstration
    print("🌸 Core Principles:")
    for key, value in info['principles'].items():
        print(f"  • {key}: {value}")
    print()
    
    # Quick component test
    print("🔬 Component Integration Test:")
    
    # 1. Glyph Codec
    codec = spiramycel.SpiramycelGlyphCodec()
    breath = codec.practice_tystnadsmajoritet(8)
    formatted = codec.format_glyph_sequence(breath)
    
    silence_glyphs = codec.get_contemplative_glyphs()
    silence_count = sum(1 for g in breath if g in silence_glyphs)
    silence_ratio = silence_count / len(breath)
    
    print(f"  🌱 Glyph Codec: {formatted}")
    print(f"     Silence ratio: {silence_ratio:.1%}")
    
    # 2. Spore Map
    spores = spiramycel.SporeMapLedger("demo_mycelial_memory.jsonl")
    spore = spores.add_spore_echo(
        sensor_deltas={"latency": -0.1, "voltage": 0.05, "temperature": -1.5},
        glyph_sequence=[0x01, 0x31],  # bandwidth + contemplative pause
        repair_effectiveness=0.82,
        bioregion="demo_meadow"
    )
    
    print(f"  🍄 Spore Map: Created spore with quality {spore.spore_quality:.2f}")
    print(f"     Bioregion: {spore.bioregion}, Season: {spore.season.value}")
    
    # 3. Runtime Patcher
    patcher = spiramycel.SpiramycelRuntimePatcher("demo_network_patches.jsonl")
    patches = patcher.process_glyph_sequence(breath[:3])
    
    safe_patches = sum(1 for p in patches if p.is_safe_to_execute())
    contemplative_patches = sum(1 for p in patches if p.severity == spiramycel.PatchSeverity.CONTEMPLATIVE)
    
    print(f"  🔧 Runtime Patcher: Generated {len(patches)} patches")
    print(f"     Safe patches: {safe_patches}/{len(patches)}")
    print(f"     Contemplative: {contemplative_patches}/{len(patches)}")
    
    print()
    print("✨ Integration Success:")
    print("  • 64-symbol mycelial vocabulary practicing Tystnadsmajoritet")
    print("  • Living memory with seasonal evaporation cycles")
    print("  • Safe patch system suggesting rather than commanding")
    print("  • Community consensus building for network wisdom")
    print()
    print("🌱 The underground nervous system is breathing...")
    print("🍄 Infrastructure and meaning co-emerge in contemplative spirals")

if __name__ == "__main__":
    main() 