"""
Glyph Mapper - Translating Boolean Gates to Spirida Glyphs

Maps Adamatzky's 470 Boolean functions to contemplative Spirida glyph system.
Bridges fungal computing logic with contemplative AI symbolism.
"""

from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

try:
    from adamatzky_layer import SpikeType, SpikeEvent
except ImportError:
    # Fallback for standalone use
    from enum import Enum
    class SpikeType(Enum):
        S_ALPHA = "fast_narrow_single"
        S_BETA = "medium_broad"
        S_GAMMA = "paired_doublet"
        S_DELTA = "burst_3_5"

class ContemplativeClass(Enum):
    """Contemplative classifications based on Adamatzky's Boolean classes"""
    SILENCE = "absorbing"      # Class I - Deep contemplative silence
    FLOW = "periodic"          # Class II - Rhythmic contemplative flow  
    STORM = "chaotic"          # Class III - Dynamic contemplative processing
    CONSTELLATION = "universal" # Class IV - Universal contemplative wisdom

@dataclass
class GlyphEvent:
    """A glyph event with contemplative context"""
    glyph: str
    contemplative_class: ContemplativeClass
    spike_origin: Optional[SpikeType] = None
    boolean_input: Optional[int] = None
    timestamp: Optional[float] = None
    silence_context: float = 0.875  # Silence Majority context

class SpiridaGlyphMapper:
    """
    Maps fungal Boolean functions to contemplative Spirida glyphs.
    
    Implements the bridge between Adamatzky's experimental results
    and contemplative AI glyph system.
    """
    
    def __init__(self):
        self._init_glyph_mappings()
        self._init_contemplative_sequences()
        
    def _init_glyph_mappings(self):
        """Initialize core glyph mappings based on spike types and Boolean classes"""
        
        # Primary glyph mapping from spike types
        self.spike_glyphs = {
            SpikeType.S_ALPHA: "⭕",   # Fast/narrow - Information silence
            SpikeType.S_BETA: "🌊",    # Medium/broad - Metabolic flow
            SpikeType.S_GAMMA: "🌪️",   # Paired doublet - Bifurcation storm
            SpikeType.S_DELTA: "🌌",   # Burst - Constellation broadcast
            # Special case: spectral overcrowding detected
            "FOG": "🌁"                # Resonant fog - semantic protection
        }
        
        # Contemplative class glyphs
        self.class_glyphs = {
            ContemplativeClass.SILENCE: "⭕",        # Deep silence
            ContemplativeClass.FLOW: "🌊",           # Rhythmic flow
            ContemplativeClass.STORM: "🌪️",          # Dynamic processing  
            ContemplativeClass.CONSTELLATION: "🌌"   # Universal wisdom
        }
        
        # Boolean pattern -> contemplative interpretation
        self.boolean_contemplative_map = {
            # Silence patterns (Class I - Absorbing)
            0b0000: ContemplativeClass.SILENCE,  # All quiet
            0b1111: ContemplativeClass.SILENCE,  # All active -> silence
            
            # Flow patterns (Class II - Periodic)
            0b0101: ContemplativeClass.FLOW,     # Alternating rhythm
            0b1010: ContemplativeClass.FLOW,     # Complementary rhythm
            0b0011: ContemplativeClass.FLOW,     # Rising rhythm
            0b1100: ContemplativeClass.FLOW,     # Falling rhythm
            
            # Storm patterns (Class III - Chaotic)
            0b0110: ContemplativeClass.STORM,    # XOR pattern
            0b1001: ContemplativeClass.STORM,    # XNOR pattern
            0b0111: ContemplativeClass.STORM,    # Majority high
            0b1000: ContemplativeClass.STORM,    # Single low
            
            # Constellation patterns (Class IV - Universal)
            0b0001: ContemplativeClass.CONSTELLATION,  # Minimal activation
            0b1110: ContemplativeClass.CONSTELLATION,  # Maximal activation
            0b1011: ContemplativeClass.CONSTELLATION,  # Complex pattern
            0b0100: ContemplativeClass.CONSTELLATION,  # Isolated activation
        }
        
    def _init_contemplative_sequences(self):
        """Initialize meaningful glyph sequences for contemplative states"""
        
        # Breathing sequences aligned with fungal rhythms
        self.breathing_sequences = {
            "ecological_breath": ["⭕", "🌊", "⭕", "🌊", "⭕"],     # 14-min deep forest rhythm
            "abstract_breath": ["⭕", "🌌", "⭕", "🌌"],          # 5-8min focused rhythm
            "storm_integration": ["🌪️", "⭕", "🌊", "⭕", "🌌"],   # Processing -> silence -> flow -> wisdom
            "silence_majority": ["⭕"] * 7 + ["🌊"],              # 87.5% silence
        }
        
        # Ecological vs Abstract paradigm markers
        self.paradigm_markers = {
            "ecological": "🌱",    # Growth, adaptation, bioregional
            "abstract": "🧠",      # Philosophy, systematic, focused
            "bridge": "🌀",        # Spiral connection between paradigms
        }
        
    def spike_to_glyph(self, spike: SpikeEvent) -> GlyphEvent:
        """Convert a spike event to a contemplative glyph event"""
        glyph = self.spike_glyphs.get(spike.spike_type, "⭕")
        
        # Determine contemplative class from spike characteristics
        if spike.amplitude < 1.0:  # Very quiet
            contemp_class = ContemplativeClass.SILENCE
        elif spike.spike_type == SpikeType.S_BETA:
            contemp_class = ContemplativeClass.FLOW
        elif spike.spike_type in [SpikeType.S_GAMMA]:
            contemp_class = ContemplativeClass.STORM
        else:
            contemp_class = ContemplativeClass.CONSTELLATION
            
        return GlyphEvent(
            glyph=glyph,
            contemplative_class=contemp_class,
            spike_origin=spike.spike_type,
            timestamp=spike.timestamp
        )
        
    def boolean_to_glyph(self, input_pattern: int, output: bool) -> GlyphEvent:
        """Convert Boolean function result to contemplative glyph"""
        
        if not output:
            # No output = silence
            return GlyphEvent(
                glyph="⭕",
                contemplative_class=ContemplativeClass.SILENCE,
                boolean_input=input_pattern
            )
            
        # Look up contemplative class
        contemp_class = self.boolean_contemplative_map.get(
            input_pattern, 
            ContemplativeClass.FLOW  # Default to flow
        )
        
        glyph = self.class_glyphs[contemp_class]
        
        return GlyphEvent(
            glyph=glyph,
            contemplative_class=contemp_class,
            boolean_input=input_pattern
        )
        
    def sequence_to_contemplative_pattern(self, glyphs: List[str]) -> Dict:
        """Analyze a sequence of glyphs for contemplative patterns"""
        
        if not glyphs:
            return {"pattern": "empty", "silence_ratio": 1.0}
            
        # Calculate silence ratio
        silence_count = glyphs.count("⭕")
        silence_ratio = silence_count / len(glyphs)
        
        # Detect patterns
        pattern_type = "unknown"
        
        if silence_ratio >= 0.8:
            pattern_type = "deep_silence"
        elif silence_ratio >= 0.6:
            pattern_type = "contemplative_balance"  # Near Silence Majority
        elif "🌊" in glyphs and "⭕" in glyphs:
            pattern_type = "breathing_rhythm"
        elif "🌪️" in glyphs:
            pattern_type = "storm_processing"
        elif "🌌" in glyphs:
            pattern_type = "constellation_wisdom"
        elif all(g == "🌊" for g in glyphs):
            pattern_type = "pure_flow"
            
        # Check for breathing sequences
        breathing_match = None
        for breath_name, breath_seq in self.breathing_sequences.items():
            if self._sequence_matches(glyphs, breath_seq):
                breathing_match = breath_name
                break
                
        return {
            "pattern": pattern_type,
            "silence_ratio": silence_ratio,
            "silence_majority_aligned": abs(silence_ratio - 0.875) < 0.1,
            "breathing_match": breathing_match,
            "length": len(glyphs),
            "unique_glyphs": len(set(glyphs)),
            "glyph_counts": {g: glyphs.count(g) for g in set(glyphs)}
        }
        
    def _sequence_matches(self, sequence: List[str], pattern: List[str], tolerance: float = 0.3) -> bool:
        """Check if sequence approximately matches pattern with some tolerance"""
        if not pattern:
            return False
            
        # Simple fuzzy matching - allow some variations
        min_length = min(len(sequence), len(pattern))
        matches = sum(1 for i in range(min_length) if sequence[i] == pattern[i])
        match_ratio = matches / len(pattern)
        
        return match_ratio >= (1 - tolerance)
        
    def generate_contemplative_sequence(self, paradigm: str = "ecological", length: int = 8) -> List[str]:
        """Generate a contemplative glyph sequence for a given paradigm"""
        
        if paradigm == "ecological":
            # Deep forest rhythm - more silence, slower patterns
            base_pattern = self.breathing_sequences["ecological_breath"]
            silence_weight = 0.8  # 80% chance of silence
            
        elif paradigm == "abstract":
            # Focused contemplative rhythm - balanced but structured
            base_pattern = self.breathing_sequences["abstract_breath"]
            silence_weight = 0.7  # 70% chance of silence
            
        else:  # bridge or unknown
            base_pattern = self.breathing_sequences["silence_majority"]
            silence_weight = 0.875  # Exact Silence Majority
            
        # Generate sequence with appropriate silence weighting
        import random
        
        sequence = []
        pattern_index = 0
        
        for i in range(length):
            if random.random() < silence_weight:
                sequence.append("⭕")
            else:
                # Use pattern or random active glyph
                if pattern_index < len(base_pattern) and base_pattern[pattern_index] != "⭕":
                    sequence.append(base_pattern[pattern_index])
                else:
                    sequence.append(random.choice(["🌊", "🌪️", "🌌"]))
                    
            pattern_index = (pattern_index + 1) % len(base_pattern)
            
        return sequence
        
    def explain_glyph(self, glyph: str) -> str:
        """Provide contemplative explanation of a glyph"""
        explanations = {
            "⭕": "Silence - The foundation of contemplative wisdom. In fungal terms, this represents the refractory period where the mycelium integrates and prepares. Embodies the Silence Majority principle (87.5% of contemplative processing).",
            
            "🌊": "Flow - Rhythmic contemplative processing. Represents the metabolic transport waves in mycelium, the steady integration of information over time. Embodies the breath of contemplative intelligence.",
            
            "🌪️": "Storm - Dynamic contemplative processing. The chaotic patterns that emerge when contemplative systems encounter complexity. In fungi, these are the bifurcation events that create new pathways of understanding.",
            
            "🌌": "Constellation - Universal contemplative wisdom. The rare but profound insights that emerge from deep contemplative processing. In fungal computing, these are the Class IV universal patterns that exhibit computational completeness.",
            
            "🌁": "Resonant Fog - Semantic protection mode. Triggered when spectral overcrowding threatens bio-semantic integrity. Represents the mycelium's protective response to high-frequency intrusion or transmission chaos. All non-fog glyphs are blocked until spectral clarity returns.",
            
            "🌱": "Ecological Paradigm - Bioregional, adaptive contemplative intelligence that grows from place and responds to environmental wisdom.",
            
            "🧠": "Abstract Paradigm - Systematic, philosophical contemplative intelligence that processes universal patterns and principles.",
            
            "🌀": "Spiral Bridge - The connecting wisdom that links paradigms, allowing ecological and abstract contemplative intelligence to inform each other."
        }
        
        return explanations.get(glyph, f"Unknown glyph: {glyph}")


# Utility functions for integration

def create_contemplative_bridge(fungal_sequence: List[str], paradigm: str = "ecological") -> str:
    """Create a contemplative bridge sequence from fungal glyph sequence"""
    mapper = SpiridaGlyphMapper()
    
    # Add paradigm marker
    if paradigm == "ecological":
        bridge = "🌱" + "".join(fungal_sequence) + "🌀"
    elif paradigm == "abstract":
        bridge = "🧠" + "".join(fungal_sequence) + "🌀"
    else:
        bridge = "🌀" + "".join(fungal_sequence) + "🌀"
        
    return bridge

def analyze_contemplative_session(glyph_sequences: Dict[str, List[str]]) -> Dict:
    """Analyze multiple glyph sequences for contemplative insights"""
    mapper = SpiridaGlyphMapper()
    
    analysis = {}
    
    for session_name, sequence in glyph_sequences.items():
        analysis[session_name] = mapper.sequence_to_contemplative_pattern(sequence)
        
    # Cross-session analysis
    if "ecological" in analysis and "abstract" in analysis:
        eco_silence = analysis["ecological"]["silence_ratio"]
        abs_silence = analysis["abstract"]["silence_ratio"]
        
        analysis["paradigm_comparison"] = {
            "ecological_silence": eco_silence,
            "abstract_silence": abs_silence,
            "silence_difference": abs(eco_silence - abs_silence),
            "contemplative_balance": abs(eco_silence - 0.875) + abs(abs_silence - 0.875)
        }
        
    return analysis
