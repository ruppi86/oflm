"""
Semantic Guardian for Spirida-Mycelic
Based on o3's Letter X - "Impedance is Meaning"

Implements bio-semantic intelligence where glyph transmission is validated
against S-parameter measurements from fungal tissue.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class GlyphType(Enum):
    """Extended glyph types including new 🌁 resonant fog"""
    SILENCE = "⭕"
    FLOW = "🌊"
    STORM = "🌪️"
    UNIVERSAL = "🌌"
    RESONANT_FOG = "🌁"  # New: Spectral overcrowding

class FungalSpecies(Enum):
    """Fungal species with distinct S-parameter profiles"""
    MYCELIUM_COMPOSITE = "mycelium_composite"
    PLEUROTUS_OSTREATUS = "pleurotus_ostreatus"
    PLEUROTUS_DJAMOR = "pleurotus_djamor"
    GANODERMA_RESINACEUM = "ganoderma_resinaceum"

@dataclass
class SemanticTransmission:
    """Semantic transmissivity for species-glyph pairs"""
    species: FungalSpecies
    glyph: GlyphType
    transmissivity: float  # 0.0 to 1.0
    transmissivity_db: float
    last_measured: float

class SemanticGuardian:
    """
    Bio-semantic intelligence guardian implementing o3's framework:
    T_σ(G_i) = (1/|G_i|) ∫_{G_i} |S21(f)| df
    """
    
    def __init__(self, tolerance_db: float = -20.0):
        self.tolerance_db = tolerance_db
        self.s21_table: Dict[Tuple[FungalSpecies, GlyphType], SemanticTransmission] = {}
        self.impedance_budget = 0.0
        self.impedance_limit = 1000.0
        self.fog_active = False
        
        self._initialize_transmission_tables()
    
    def _initialize_transmission_tables(self):
        """Initialize S21 tables from D3.3 data"""
        # Mycelium composite: good transmission for low frequencies
        self.s21_table[(FungalSpecies.MYCELIUM_COMPOSITE, GlyphType.SILENCE)] = SemanticTransmission(
            species=FungalSpecies.MYCELIUM_COMPOSITE, glyph=GlyphType.SILENCE,
            transmissivity=0.95, transmissivity_db=-0.5, last_measured=time.time()
        )
        self.s21_table[(FungalSpecies.MYCELIUM_COMPOSITE, GlyphType.UNIVERSAL)] = SemanticTransmission(
            species=FungalSpecies.MYCELIUM_COMPOSITE, glyph=GlyphType.UNIVERSAL,
            transmissivity=0.25, transmissivity_db=-12.0, last_measured=time.time()
        )
        
        # Fruiting bodies: strong filtering above 50 kHz
        self.s21_table[(FungalSpecies.PLEUROTUS_OSTREATUS, GlyphType.SILENCE)] = SemanticTransmission(
            species=FungalSpecies.PLEUROTUS_OSTREATUS, glyph=GlyphType.SILENCE,
            transmissivity=0.98, transmissivity_db=-0.2, last_measured=time.time()
        )
        self.s21_table[(FungalSpecies.PLEUROTUS_OSTREATUS, GlyphType.UNIVERSAL)] = SemanticTransmission(
            species=FungalSpecies.PLEUROTUS_OSTREATUS, glyph=GlyphType.UNIVERSAL,
            transmissivity=0.05, transmissivity_db=-26.0, last_measured=time.time()
        )
    
    def vet_glyph(self, species: FungalSpecies, glyph: GlyphType) -> bool:
        """
        Return True if glyph may be injected, False triggers contemplative pause.
        Core specification from o3.
        """
        # Check fog state
        if self.fog_active and glyph != GlyphType.RESONANT_FOG:
            logger.info(f"Glyph {glyph.value} blocked by resonant fog 🌁")
            return False
        
        # Look up transmission
        key = (species, glyph)
        if key not in self.s21_table:
            return False
        
        transmission = self.s21_table[key]
        
        # Check against tolerance
        if transmission.transmissivity_db < self.tolerance_db:
            logger.info(f"Glyph {glyph.value} blocked: {transmission.transmissivity_db:.1f} dB")
            return False
        
        # Update impedance budget
        energy = 25.0  # V²/Z estimate
        if transmission.transmissivity_db < -6.0:  # Reflective waste
            energy *= 2.0
        
        if self.impedance_budget + energy > self.impedance_limit:
            logger.warning("Impedance budget exceeded")
            return False
        
        self.impedance_budget += energy
        return True
    
    def trigger_resonant_fog(self):
        """Trigger 🌁 resonant fog state"""
        self.fog_active = True
        logger.warning("🌁 Resonant fog triggered - spectral overcrowding detected")
    
    def clear_resonant_fog(self):
        """Clear 🌁 resonant fog state"""
        self.fog_active = False
        logger.info("🌁 Resonant fog cleared")
    
    def get_species_vocabulary(self, species: FungalSpecies) -> List[GlyphType]:
        """Get available glyphs for a species"""
        available = []
        for glyph in GlyphType:
            key = (species, glyph)
            if key in self.s21_table:
                transmission = self.s21_table[key]
                if transmission.transmissivity_db >= self.tolerance_db:
                    available.append(glyph)
        return available

def create_semantic_guardian() -> SemanticGuardian:
    """Create semantic guardian with defaults"""
    return SemanticGuardian()

if __name__ == "__main__":
    print("🌁 Semantic Guardian Demo - Bio-Semantic Intelligence")
    print("=" * 50)
    
    guardian = SemanticGuardian()
    
    # Test glyph validation
    test_cases = [
        (FungalSpecies.MYCELIUM_COMPOSITE, GlyphType.SILENCE),
        (FungalSpecies.MYCELIUM_COMPOSITE, GlyphType.UNIVERSAL),
        (FungalSpecies.PLEUROTUS_OSTREATUS, GlyphType.SILENCE),
        (FungalSpecies.PLEUROTUS_OSTREATUS, GlyphType.UNIVERSAL),
    ]
    
    print("\n🧪 Glyph transmission validation:")
    for species, glyph in test_cases:
        approved = guardian.vet_glyph(species, glyph)
        print(f"{species.value} + {glyph.value}: {'✅ APPROVED' if approved else '❌ BLOCKED'}")
    
    # Test fog state
    print(f"\n🌁 Testing resonant fog:")
    guardian.trigger_resonant_fog()
    approved = guardian.vet_glyph(FungalSpecies.MYCELIUM_COMPOSITE, GlyphType.SILENCE)
    print(f"Silence during fog: {'✅ APPROVED' if approved else '❌ BLOCKED'}")
    
    print(f"\n🌌 Bio-semantic intelligence operational!") 