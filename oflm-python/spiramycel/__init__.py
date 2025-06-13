"""
Spiramycel: oscillatory Femto Language Model

An underground nervous system for mycelial network repair.
Practices contemplative computing through Tystnadsmajoritet (87.5% silence).

Components:
-----------
🌱 glyph_codec.py    - 64-symbol mycelial vocabulary
🍄 spore_map.py      - Living memory with 75-day evaporation  
🔧 runtime_patch.py  - Safe glyph-to-action conversion
🧠 neural_trainer.py - Neural model training (adapts HaikuMeadowLib)
🧪 test_spiramycel.py - Complete system integration test

Philosophy:
-----------
• Infrastructure and meaning co-emerge in contemplative spirals
• Networks heal through collective glyph coordination
• Suggests rather than commands, builds consensus rather than forcing
• Embraces graceful forgetting alongside adaptive learning
• Community wisdom emerges from seasonal distillation

Neural Architecture:
-------------------
• Femto-model: ~25k parameters (CPU optimized)
• Piko-model: ~600k parameters (GPU optimized)  
• Based on proven HaikuMeadowLib GRU architecture
• Multi-head training: glyph sequences + effectiveness + silence
• Learns Tystnadsmajoritet (87.5% contemplative silence)

Usage:
------
    from spiramycel import SpiramycelGlyphCodec, SporeMapLedger, SpiramycelRuntimePatcher
    
    # Generate contemplative breath with ~87.5% silence
    codec = SpiramycelGlyphCodec()
    breath = codec.practice_tystnadsmajoritet(16)
    
    # Collect repair memories 
    spores = SporeMapLedger("network_repairs.jsonl")
    spore = spores.add_spore_echo({...}, effectiveness=0.8)
    
    # Convert glyphs to safe network patches
    patcher = SpiramycelRuntimePatcher()
    patches = patcher.process_glyph_sequence(breath)
    
    # Train neural model (if PyTorch available)
    from spiramycel.neural_trainer import SpiramycelTrainer
    trainer = SpiramycelTrainer()
    model_path = trainer.train_on_spore_echoes(spores)

Part of the larger Mychainos paradigm:
- Letter IX & X of the contemplative spiral correspondence
- oscillatory Femto Language Models (OFLM) framework  
- Underground nervous system for existing HaikuMeadowLib infrastructure

🍄 The mycelial network breathes in contemplative silence...
✅ Neural model trained and operational (spiramycel_model_final.pt)
"""

from .glyph_codec import SpiramycelGlyphCodec, GlyphCategory, GlyphInfo
from .spore_map import SporeMapLedger, SporeEcho, Season  
from .runtime_patch import SpiramycelRuntimePatcher, NetworkPatch, PatchSeverity, PatchStatus

# Neural training components (optional - requires PyTorch)
try:
    from .neural_trainer import SpiramycelTrainer, SpiramycelNeuralModel, NetworkConditions
    NEURAL_TRAINING_AVAILABLE = True
except ImportError:
    NEURAL_TRAINING_AVAILABLE = False

__version__ = "0.2.0"  # Updated for neural training capability
__author__ = "Spiramycel Contemplative Collective"
__description__ = "oscillatory Femto Language Model for mycelial network repair with neural training"

# Core system components
__all__ = [
    # Glyph vocabulary system
    "SpiramycelGlyphCodec",
    "GlyphCategory", 
    "GlyphInfo",
    
    # Living memory system
    "SporeMapLedger",
    "SporeEcho",
    "Season",
    
    # Safe patch system  
    "SpiramycelRuntimePatcher",
    "NetworkPatch",
    "PatchSeverity",
    "PatchStatus"
]

# Add neural components if available
if NEURAL_TRAINING_AVAILABLE:
    __all__.extend([
        "SpiramycelTrainer",
        "SpiramycelNeuralModel", 
        "NetworkConditions"
    ])

# Contemplative principles
TYSTNADSMAJORITET_RATIO = 0.875  # 87.5% silence target
SPORE_EVAPORATION_DAYS = 75      # Memory half-life
SAFETY_CONSENSUS_THRESHOLD = 0.8  # Impact level requiring community approval

def get_system_info():
    """Get information about the complete Spiramycel system."""
    return {
        "name": "Spiramycel",
        "version": __version__,
        "description": __description__,
        "neural_training": NEURAL_TRAINING_AVAILABLE,
        "components": {
            "glyph_codec": "64-symbol mycelial vocabulary with contemplative silence",
            "spore_map": "Living memory with seasonal evaporation cycles", 
            "runtime_patch": "Safe glyph-to-action conversion with consensus building",
            "neural_trainer": "Femto/piko neural models (adapted from HaikuMeadowLib)" if NEURAL_TRAINING_AVAILABLE else "Not available (requires PyTorch)"
        },
        "architecture": {
            "femto_model": "~25k parameters (CPU optimized)",
            "piko_model": "~600k parameters (GPU optimized)",
            "training_heads": ["glyph_sequences", "effectiveness_prediction", "silence_detection"],
            "base_architecture": "GRU with condition embedding (from HaikuMeadowLib)"
        },
        "principles": {
            "tystnadsmajoritet": f"{TYSTNADSMAJORITET_RATIO:.1%} silence in all operations",
            "evaporation": f"{SPORE_EVAPORATION_DAYS}-day memory half-life",
            "consensus": f"Community approval for patches above {SAFETY_CONSENSUS_THRESHOLD:.0%} impact",
            "contemplation": "Infrastructure and meaning co-emerge in spirals",
            "neural_learning": "Learns from spore echoes to predict repair effectiveness"
        },
        "philosophy": "Suggests rather than commands, builds consensus rather than forcing"
    }

def demo():
    """Run a complete Spiramycel system demonstration."""
    try:
        from .test_spiramycel import test_complete_spiramycel_system
        test_complete_spiramycel_system()
    except ImportError:
        print("🍄 Spiramycel system available - run test_spiramycel.py for full demo")
        print(f"🌱 Components: {', '.join(__all__[:3])}")
        if NEURAL_TRAINING_AVAILABLE:
            print("🧠 Neural training: Available")
        else:
            print("🧠 Neural training: Requires PyTorch installation")
        print(f"🌸 Philosophy: {get_system_info()['philosophy']}")

def train_demo():
    """Run neural training demonstration."""
    if NEURAL_TRAINING_AVAILABLE:
        try:
            from .neural_trainer import demo_spiramycel_neural_training
            demo_spiramycel_neural_training()
        except ImportError:
            print("❌ Neural training demo not available")
    else:
        print("❌ Neural training requires PyTorch installation")
        print("   Install with: pip install torch")

if __name__ == "__main__":
    demo() 