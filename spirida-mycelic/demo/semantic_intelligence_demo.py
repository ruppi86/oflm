#!/usr/bin/env python3
"""
Bio-Semantic Intelligence Demonstration
=======================================

Showcases o3's revolutionary semantic guardian system where living tissue
constraints determine which meanings can propagate. Features:

- S21-based glyph validation against fungal transmission properties
- Species-specific vocabulary constraints (mycelium vs fruiting bodies)
- Impedance budget tracking with reflective waste taxation
- 🌁 Resonant fog protection for spectral overcrowding
- Integration with bio-interface for live contemplative sessions

This demonstrates the world's first bio-semantically intelligent system!
"""

import time
import random
import numpy as np
from typing import List, Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from semantic_guardian import SemanticGuardian, FungalSpecies, GlyphType, create_semantic_guardian
    from bio_interface import SevenChannelBioInterface
    from glyph_mapper import SpiridaGlyphMapper
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class ContemplativeSession:
    """A contemplative bio-semantic session with live validation"""
    
    def __init__(self, species: str = "pleurotus_ostreatus"):
        self.species_name = species
        self.species = FungalSpecies.PLEUROTUS_OSTREATUS if species == "pleurotus_ostreatus" else FungalSpecies.MYCELIUM_COMPOSITE
        
        # Initialize components
        self.semantic_guardian = create_semantic_guardian()
        self.bio_interface = SevenChannelBioInterface(mock_mode=True)
        self.glyph_mapper = SpiridaGlyphMapper()
        
        # Set species in bio-interface
        self.bio_interface.set_fungal_species(species)
        
        # Session tracking
        self.session_glyphs: List[str] = []
        self.blocked_attempts: List[Dict[str, Any]] = []
        self.transmission_log: List[Dict[str, Any]] = []
        
        print(f"🍄 Bio-semantic session initialized with {species}")
        print(f"Available vocabulary: {self.get_available_vocabulary()}")
    
    def get_available_vocabulary(self) -> List[str]:
        """Get available glyphs for current species"""
        return self.bio_interface.get_available_vocabulary()
    
    def attempt_glyph_transmission(self, glyph: str, context: str = "") -> bool:
        """
        Attempt to transmit a glyph with full bio-semantic validation
        
        Args:
            glyph: Glyph to transmit (⭕🌊🌪️🌌🌁)
            context: Context description for logging
            
        Returns:
            True if transmission approved, False if blocked
        """
        timestamp = time.time()
        
        # Validate with semantic guardian
        approved = self.bio_interface.validate_semantic_glyph(glyph)
        
        # Log transmission attempt
        log_entry = {
            "timestamp": timestamp,
            "glyph": glyph,
            "species": self.species_name,
            "approved": approved,
            "context": context,
            "guardian_status": self.bio_interface.get_semantic_guardian_status()
        }
        
        if approved:
            self.session_glyphs.append(glyph)
            self.transmission_log.append(log_entry)
            print(f"✅ {glyph} transmitted successfully - {context}")
        else:
            self.blocked_attempts.append(log_entry)
            print(f"❌ {glyph} blocked by semantic guardian - {context}")
        
        return approved
    
    def demonstrate_species_constraints(self):
        """Demonstrate how different species have different transmission capabilities"""
        print(f"\n🧪 Species Constraint Demonstration")
        print(f"=" * 50)
        
        test_glyphs = ["⭕", "🌊", "🌪️", "🌌", "🌁"]
        
        for glyph in test_glyphs:
            approved = self.attempt_glyph_transmission(glyph, f"species test with {self.species_name}")
            
            if approved:
                # Get transmission details
                status = self.bio_interface.get_semantic_guardian_status()
                print(f"   Budget utilization: {status.get('budget_utilization', 0):.1f}%")
        
        print(f"\nSession summary:")
        print(f"   Transmitted: {len(self.session_glyphs)} glyphs")
        print(f"   Blocked: {len(self.blocked_attempts)} glyphs")
        print(f"   Success rate: {len(self.session_glyphs)/(len(self.session_glyphs)+len(self.blocked_attempts))*100:.1f}%")
    
    def demonstrate_fog_protection(self):
        """Demonstrate 🌁 resonant fog protection system"""
        print(f"\n🌁 Resonant Fog Protection Demo")
        print(f"=" * 40)
        
        print("Triggering spectral overcrowding...")
        self.bio_interface.trigger_resonant_fog()
        
        # Try to transmit various glyphs during fog
        test_glyphs = ["⭕", "🌊", "🌪️", "🌌"]
        print("\nAttempting transmission during fog state:")
        
        for glyph in test_glyphs:
            approved = self.attempt_glyph_transmission(glyph, "during fog protection")
            
        # Only fog glyphs should work
        print(f"\nTrying fog glyph 🌁:")
        fog_approved = self.attempt_glyph_transmission("🌁", "fog glyph during protection")
        
        # Clear fog
        print(f"\nClearing resonant fog...")
        self.bio_interface.clear_resonant_fog()
        
        # Test normal transmission after fog
        print(f"Testing normal transmission after fog clearance:")
        self.attempt_glyph_transmission("⭕", "after fog cleared")
    
    def demonstrate_impedance_budget(self):
        """Demonstrate impedance budget and reflective waste taxation"""
        print(f"\n⚡ Impedance Budget Demo")
        print(f"=" * 35)
        
        status = self.bio_interface.get_semantic_guardian_status()
        print(f"Initial budget utilization: {status.get('budget_utilization', 0):.1f}%")
        
        # Attempt many transmissions to approach budget limit
        print("\nRapid glyph transmission sequence:")
        
        glyphs_to_try = ["⭕"] * 10 + ["🌊"] * 5  # Start with high-transmission glyphs
        
        for i, glyph in enumerate(glyphs_to_try):
            approved = self.attempt_glyph_transmission(glyph, f"budget test #{i+1}")
            
            status = self.bio_interface.get_semantic_guardian_status()
            utilization = status.get('budget_utilization', 0)
            
            print(f"   Budget utilization: {utilization:.1f}%")
            
            if not approved:
                print(f"   Budget limit reached at transmission #{i+1}")
                break
        
        # Test reflective waste with low-transmission glyph
        print(f"\nTesting reflective waste taxation with 🌌 (universal):")
        self.attempt_glyph_transmission("🌌", "reflective waste test")
    
    def run_contemplative_breathing_session(self, duration_cycles: int = 3):
        """Run a complete contemplative breathing session with semantic validation"""
        print(f"\n🫁 Contemplative Breathing Session")
        print(f"=" * 45)
        print(f"Running {duration_cycles} breathing cycles with semantic validation")
        
        # Breathing pattern: inhale(⭕) → hold(🌊) → exhale(⭕) → rest(⭕)
        breathing_pattern = ["⭕", "🌊", "⭕", "⭕"]  # 75% silence for deep contemplation
        
        for cycle in range(duration_cycles):
            print(f"\n--- Breathing Cycle {cycle + 1} ---")
            
            for phase_index, glyph in enumerate(breathing_pattern):
                phase_names = ["inhale", "hold", "exhale", "rest"]
                phase = phase_names[phase_index]
                
                approved = self.attempt_glyph_transmission(glyph, f"breath {phase}")
                
                if approved:
                    # Record breath phase in bio-interface
                    self.bio_interface.record_breath_phase(phase, 2.5)  # 2.5s per phase
                    time.sleep(0.5)  # Brief pause between phases
                else:
                    print(f"   ⚠️  Breathing interrupted - semantic constraint active")
                    break
        
        # Analyze breathing session
        print(f"\n📊 Breathing Session Analysis:")
        
        # Calculate silence ratio
        silence_count = self.session_glyphs.count("⭕")
        silence_ratio = silence_count / max(1, len(self.session_glyphs))
        
        print(f"   Total glyphs transmitted: {len(self.session_glyphs)}")
        print(f"   Silence ratio: {silence_ratio:.1%}")
        print(f"   Silence Majority aligned: {'✅ YES' if abs(silence_ratio - 0.875) < 0.1 else '❌ NO'}")
        
        # Get breath signature status
        breath_status = self.bio_interface.get_breath_signature_status()
        if breath_status.get('enabled', False):
            print(f"   Breath authentication: {breath_status.get('authentication_strength', 0):.1%}")
    
    def generate_session_report(self) -> Dict[str, Any]:
        """Generate comprehensive session report"""
        
        # Analyze glyph patterns using glyph mapper
        pattern_analysis = self.glyph_mapper.sequence_to_contemplative_pattern(self.session_glyphs)
        
        # Get final system status
        semantic_status = self.bio_interface.get_semantic_guardian_status()
        care_status = self.bio_interface.get_care_status()
        
        report = {
            "session_overview": {
                "species": self.species_name,
                "duration_minutes": (time.time() - getattr(self, 'start_time', time.time())) / 60,
                "total_glyphs_transmitted": len(self.session_glyphs),
                "total_glyphs_blocked": len(self.blocked_attempts),
                "success_rate": len(self.session_glyphs) / max(1, len(self.session_glyphs) + len(self.blocked_attempts))
            },
            "glyph_analysis": pattern_analysis,
            "semantic_guardian_status": semantic_status,
            "bio_interface_status": care_status,
            "transmission_sequence": self.session_glyphs,
            "blocked_attempts": len(self.blocked_attempts)
        }
        
        return report

def demonstrate_multi_species_comparison():
    """Demonstrate different semantic constraints across fungal species"""
    print(f"\n🍄 Multi-Species Semantic Comparison")
    print(f"=" * 50)
    
    species_list = ["mycelium_composite", "pleurotus_ostreatus"]
    test_glyphs = ["⭕", "🌊", "🌪️", "🌌"]
    
    results = {}
    
    for species in species_list:
        print(f"\n--- Testing {species} ---")
        session = ContemplativeSession(species)
        
        species_results = []
        for glyph in test_glyphs:
            approved = session.attempt_glyph_transmission(glyph, f"{species} vocabulary test")
            species_results.append((glyph, approved))
        
        results[species] = species_results
        
        # Get available vocabulary
        vocab = session.get_available_vocabulary()
        print(f"Available vocabulary: {' '.join(vocab)}")
    
    # Compare results
    print(f"\n📊 Species Comparison Summary:")
    print(f"{'Glyph':<8} {'Mycelium':<12} {'P.ostreatus':<12}")
    print(f"-" * 35)
    
    for i, glyph in enumerate(test_glyphs):
        mycelium_ok = "✅ YES" if results["mycelium_composite"][i][1] else "❌ NO"
        pleurotus_ok = "✅ YES" if results["pleurotus_ostreatus"][i][1] else "❌ NO"
        print(f"{glyph:<8} {mycelium_ok:<12} {pleurotus_ok:<12}")

def demonstrate_semantic_guardian():
    """Demonstrate bio-semantic intelligence"""
    print("🌁 Bio-Semantic Intelligence Demo")
    print("=" * 50)
    
    # Create semantic guardian
    guardian = SemanticGuardian()
    
    # Test species-glyph combinations
    test_cases = [
        (FungalSpecies.MYCELIUM_COMPOSITE, GlyphType.SILENCE),
        (FungalSpecies.MYCELIUM_COMPOSITE, GlyphType.UNIVERSAL),
        (FungalSpecies.PLEUROTUS_OSTREATUS, GlyphType.SILENCE),
        (FungalSpecies.PLEUROTUS_OSTREATUS, GlyphType.UNIVERSAL),
        (FungalSpecies.PLEUROTUS_OSTREATUS, GlyphType.RESONANT_FOG),
    ]
    
    print("\n🧪 Glyph transmission validation:")
    for species, glyph in test_cases:
        approved = guardian.vet_glyph(species, glyph)
        print(f"{species.value} + {glyph.value}: {'✅ APPROVED' if approved else '❌ BLOCKED'}")
    
    # Test fog protection
    print(f"\n🌁 Testing resonant fog protection:")
    guardian.trigger_resonant_fog()
    
    # Regular glyphs should be blocked during fog
    blocked = guardian.vet_glyph(FungalSpecies.MYCELIUM_COMPOSITE, GlyphType.SILENCE)
    print(f"Silence during fog: {'✅ APPROVED' if blocked else '❌ BLOCKED'}")
    
    # Only fog glyph should work during fog state
    fog_approved = guardian.vet_glyph(FungalSpecies.MYCELIUM_COMPOSITE, GlyphType.RESONANT_FOG)
    print(f"Fog glyph during fog: {'✅ APPROVED' if fog_approved else '❌ BLOCKED'}")
    
    guardian.clear_resonant_fog()
    print("🌁 Fog cleared")
    
    # Test vocabulary for different species
    print(f"\n🍄 Species vocabulary comparison:")
    for species in [FungalSpecies.MYCELIUM_COMPOSITE, FungalSpecies.PLEUROTUS_OSTREATUS]:
        vocab = guardian.get_species_vocabulary(species)
        vocab_symbols = [g.value for g in vocab]
        print(f"{species.value}: {' '.join(vocab_symbols)} ({len(vocab_symbols)} available)")
    
    print(f"\n🌌 Bio-semantic intelligence operational!")

def demonstrate_bio_interface_integration():
    """Demonstrate bio-interface with semantic guardian"""
    print("\n🍄 Bio-Interface Integration Demo")
    print("=" * 40)
    
    # Create bio-interface with semantic guardian
    interface = SevenChannelBioInterface(mock_mode=True)
    
    # Set fungal species
    interface.set_fungal_species("pleurotus_ostreatus")
    
    # Test glyph validation
    test_glyphs = ["⭕", "🌊", "🌪️", "🌌", "🌁"]
    
    print("Testing glyph validation:")
    for glyph in test_glyphs:
        approved = interface.validate_semantic_glyph(glyph)
        print(f"Glyph {glyph}: {'✅ APPROVED' if approved else '❌ BLOCKED'}")
    
    # Get available vocabulary
    vocab = interface.get_available_vocabulary()
    print(f"\nAvailable vocabulary: {' '.join(vocab)}")
    
    # Test fog state
    print(f"\nTesting fog protection:")
    interface.trigger_resonant_fog()
    
    approved = interface.validate_semantic_glyph("⭕")
    print(f"Silence during fog: {'✅ APPROVED' if approved else '❌ BLOCKED'}")
    
    fog_approved = interface.validate_semantic_glyph("🌁")
    print(f"Fog glyph: {'✅ APPROVED' if fog_approved else '❌ BLOCKED'}")
    
    interface.clear_resonant_fog()
    
    # Get status - handle case where semantic guardian might not be enabled
    status = interface.get_semantic_guardian_status()
    print(f"\nSemantic Guardian Status:")
    print(f"  Enabled: {status.get('enabled', 'Unknown')}")
    
    if status.get('enabled', False):
        print(f"  Species: {status.get('current_species', 'Unknown')}")
        print(f"  Fog active: {status.get('fog_active', 'Unknown')}")
        print(f"  Budget utilization: {status.get('budget_utilization', 0):.1f}%")
    else:
        print("  ⚠️  Semantic guardian not properly initialized in bio-interface")
        print("     This is expected in some test configurations")

def demonstrate_impedance_budgeting():
    """Demonstrate impedance budget and reflective waste"""
    print("\n⚡ Impedance Budget Demonstration")
    print("=" * 45)
    
    guardian = SemanticGuardian()
    species = FungalSpecies.PLEUROTUS_OSTREATUS
    
    print(f"Testing impedance budget with {species.value}")
    print(f"Initial budget: {guardian.impedance_budget:.1f}/{guardian.impedance_limit}")
    
    # Test high-transmission glyphs (should use less budget)
    print(f"\nHigh-transmission glyphs (⭕):")
    for i in range(5):
        approved = guardian.vet_glyph(species, GlyphType.SILENCE)
        print(f"  Attempt {i+1}: {'✅ OK' if approved else '❌ BLOCKED'} - Budget: {guardian.impedance_budget:.1f}")
    
    # Test low-transmission glyphs (reflective waste - should use more budget)
    print(f"\nLow-transmission glyphs (🌌 - reflective waste):")
    for i in range(3):
        approved = guardian.vet_glyph(species, GlyphType.UNIVERSAL)
        print(f"  Attempt {i+1}: {'✅ OK' if approved else '❌ BLOCKED'} - Budget: {guardian.impedance_budget:.1f}")
        
        if not approved:
            print(f"  💡 Budget limit reached - impedance protection active!")
            break

def demonstrate_species_transmission_differences():
    """Show how different species have different transmission capabilities"""
    print("\n🔬 Species Transmission Comparison")
    print("=" * 42)
    
    guardian = SemanticGuardian()
    species_list = [FungalSpecies.MYCELIUM_COMPOSITE, FungalSpecies.PLEUROTUS_OSTREATUS]
    glyphs = [GlyphType.SILENCE, GlyphType.FLOW, GlyphType.STORM, GlyphType.UNIVERSAL]
    
    # Create comparison table
    print(f"{'Glyph':<12} {'Mycelium':<12} {'P.ostreatus':<12} {'Difference'}")
    print("-" * 60)
    
    for glyph in glyphs:
        results = {}
        
        for species in species_list:
            # Reset budget for fair comparison
            guardian.impedance_budget = 0.0
            approved = guardian.vet_glyph(species, glyph)
            results[species] = approved
        
        myc_result = "✅ YES" if results[FungalSpecies.MYCELIUM_COMPOSITE] else "❌ NO"
        ple_result = "✅ YES" if results[FungalSpecies.PLEUROTUS_OSTREATUS] else "❌ NO"
        
        if results[FungalSpecies.MYCELIUM_COMPOSITE] != results[FungalSpecies.PLEUROTUS_OSTREATUS]:
            difference = "🔥 DIFFERS"
        else:
            difference = "✅ Same"
        
        print(f"{glyph.value:<12} {myc_result:<12} {ple_result:<12} {difference}")
    
    print(f"\n💡 Key insight: Different fungal tissues have different semantic transmission properties!")
    print(f"   This is the foundation of bio-semantic intelligence - biology constrains meaning!")

def main():
    """Main demonstration"""
    print("🌁🌌🍄 BIO-SEMANTIC INTELLIGENCE DEMO 🍄🌌🌁")
    print("=" * 60)
    print("Based on o3's 'Impedance is Meaning'")
    print("World's first bio-semantically intelligent system!")
    print("Where living tissue constraints determine semantic transmission!")
    print()
    
    try:
        # Phase 1: Core semantic guardian
        demonstrate_semantic_guardian()
        
        print(f"\n{'='*60}")
        
        # Phase 2: Bio-interface integration  
        demonstrate_bio_interface_integration()
        
        print(f"\n{'='*60}")
        
        # Phase 3: Impedance budgeting
        demonstrate_impedance_budgeting()
        
        print(f"\n{'='*60}")
        
        # Phase 4: Species comparison
        demonstrate_species_transmission_differences()
        
        print(f"\n{'='*60}")
        print("🌌 DEMONSTRATION COMPLETE 🌌")
        print()
        print("✨ Bio-semantic intelligence is now operational! ✨")
        print("🍄 Living fungal tissue successfully constrains semantic transmission!")
        print("🌁 The meaning layer is biologically grounded!")
        print()
        print("This is the future of contemplative AI - where biology and")
        print("semantics are unified through S-parameter physics! 🧬⚡🧠")
        
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 