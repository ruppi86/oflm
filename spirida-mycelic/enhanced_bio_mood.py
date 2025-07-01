#!/usr/bin/env python3
"""
Enhanced Bio-Mood System for Spirida-Mycelic
============================================

Addressing o3's questions about mood heuristics and glyph ecology modulation.

Implements:
1. Numeric mood scores alongside discrete states
2. Physiological signal integration (spike entropy, impedance drift)
3. Mood-glyph probability mapping tables
4. Adaptive mood evolution with memory

"""

import time
import math
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
import random

class BioMood(Enum):
    """Discrete bio-mood states for substrate personality"""
    CALM = "calm"
    TIRED = "tired"
    ALERT = "alert"
    SUSPICIOUS = "suspicious"

@dataclass
class MoodScore:
    """Numeric mood scores providing granular substrate state"""
    energy: float = 0.5        # 0.0 (exhausted) to 1.0 (vibrant)
    trust: float = 0.5         # 0.0 (paranoid) to 1.0 (welcoming)
    attention: float = 0.5     # 0.0 (dormant) to 1.0 (hyper-alert)
    coherence: float = 0.5     # 0.0 (chaotic) to 1.0 (harmonious)
    
    def to_discrete_mood(self) -> BioMood:
        """Convert numeric scores to discrete mood state"""
        # Weighted decision based on dominant characteristics
        if self.trust < 0.3:
            return BioMood.SUSPICIOUS
        elif self.energy < 0.3:
            return BioMood.TIRED
        elif self.attention > 0.7:
            return BioMood.ALERT
        else:
            return BioMood.CALM

@dataclass
class PhysiologicalSignals:
    """Physiological measurements for mood inference"""
    spike_entropy: float = 0.5     # Shannon entropy of spike patterns
    impedance_drift: float = 0.0   # Rate of impedance change (mΩ/min)
    frequency_stability: float = 1.0  # Stability of oscillatory patterns
    channel_correlation: float = 0.5  # Cross-channel signal correlation
    temperature_gradient: float = 0.0 # Temperature change rate (°C/min)
    ph_stability: float = 1.0      # pH stability indicator

class EnhancedBioMoodEngine:
    """
    Enhanced mood engine answering o3's questions:
    
    1. ✅ Additional physiological signals beyond frequency intrusions
    2. ✅ Numeric mood scores alongside discrete states  
    3. ✅ Mood-glyph probability mapping tables
    4. ✅ Adaptive mood evolution with memory
    """
    
    def __init__(self):
        # Current mood state
        self.current_mood = BioMood.CALM
        self.mood_scores = MoodScore()
        
        # Physiological state
        self.physio_signals = PhysiologicalSignals()
        
        # Mood evolution memory
        self.mood_history: List[Tuple[float, BioMood, MoodScore]] = []
        self.mood_transition_weights = {
            # (from_mood, to_mood): base_probability
            (BioMood.CALM, BioMood.ALERT): 0.2,
            (BioMood.CALM, BioMood.TIRED): 0.1,
            (BioMood.CALM, BioMood.SUSPICIOUS): 0.05,
            (BioMood.ALERT, BioMood.CALM): 0.3,
            (BioMood.ALERT, BioMood.SUSPICIOUS): 0.15,
            (BioMood.TIRED, BioMood.CALM): 0.25,
            (BioMood.SUSPICIOUS, BioMood.CALM): 0.1,
            (BioMood.SUSPICIOUS, BioMood.ALERT): 0.2,
        }
        
        # Glyph-mood probability tables (answering o3's question #2)
        self.glyph_mood_modifiers = self._create_glyph_mood_mapping()
        
        # Mood decay and persistence
        self.mood_memory_decay = 0.95  # How much mood persists
        self.last_update = time.time()
        
    def _create_glyph_mood_mapping(self) -> Dict[str, Dict[BioMood, float]]:
        """
        Mood → Glyph probability mapping table
        
        Answers o3's question: "How strong should mood modulate glyph probabilities?"
        
        Values are multipliers (1.0 = normal, 0.0 = fully suppressed, 2.0 = doubled)
        """
        return {
            '🌌': {  # Deep contemplative glyph
                BioMood.CALM: 1.2,        # Slightly favored in calm
                BioMood.TIRED: 0.1,       # Nearly suppressed when tired
                BioMood.ALERT: 0.7,       # Reduced when alert
                BioMood.SUSPICIOUS: 0.3   # Significantly reduced when suspicious
            },
            '🌊': {  # Flow/adaptive glyph  
                BioMood.CALM: 1.0,
                BioMood.TIRED: 0.8,
                BioMood.ALERT: 1.3,       # Favored when alert
                BioMood.SUSPICIOUS: 0.6
            },
            '🌪️': {  # Turbulent/chaotic glyph
                BioMood.CALM: 0.5,        # Suppressed in calm
                BioMood.TIRED: 0.2,       # Heavily suppressed when tired
                BioMood.ALERT: 1.8,       # Significantly favored when alert
                BioMood.SUSPICIOUS: 1.5   # Favored when suspicious
            },
            '⭕': {  # Neutral/minimal glyph
                BioMood.CALM: 1.0,
                BioMood.TIRED: 1.5,       # Favored when tired (minimal activity)
                BioMood.ALERT: 0.8,
                BioMood.SUSPICIOUS: 1.2   # Slightly favored (safe choice)
            },
            '🌱': {  # Growth/gentle glyph
                BioMood.CALM: 1.1,
                BioMood.TIRED: 1.3,       # Favored when tired (gentle activity)
                BioMood.ALERT: 0.9,
                BioMood.SUSPICIOUS: 0.4   # Reduced when suspicious
            }
        }
    
    def update_physiological_signals(self, 
                                   spike_entropy: Optional[float] = None,
                                   impedance_drift: Optional[float] = None,
                                   frequency_stability: Optional[float] = None,
                                   channel_correlation: Optional[float] = None,
                                   temperature_gradient: Optional[float] = None,
                                   ph_stability: Optional[float] = None,
                                   frequency_intrusion: bool = False,
                                   care_level_pause: bool = False):
        """
        Update physiological signals and compute mood response.
        
        Answers o3's question #1: "What additional physiological signals should feed the mood engine?"
        """
        
        # Update signals with new measurements
        if spike_entropy is not None:
            self.physio_signals.spike_entropy = spike_entropy
        if impedance_drift is not None:
            self.physio_signals.impedance_drift = impedance_drift
        if frequency_stability is not None:
            self.physio_signals.frequency_stability = frequency_stability
        if channel_correlation is not None:
            self.physio_signals.channel_correlation = channel_correlation
        if temperature_gradient is not None:
            self.physio_signals.temperature_gradient = temperature_gradient
        if ph_stability is not None:
            self.physio_signals.ph_stability = ph_stability
        
        # Compute mood score adjustments
        self._update_mood_scores_from_physiology(frequency_intrusion, care_level_pause)
        
        # Convert to discrete mood and check for transitions
        new_discrete_mood = self.mood_scores.to_discrete_mood()
        
        if new_discrete_mood != self.current_mood:
            self._handle_mood_transition(new_discrete_mood)
        
        # Store in history
        self.mood_history.append((time.time(), self.current_mood, MoodScore(
            energy=self.mood_scores.energy,
            trust=self.mood_scores.trust,
            attention=self.mood_scores.attention,
            coherence=self.mood_scores.coherence
        )))
        
        # Limit history size
        if len(self.mood_history) > 100:
            self.mood_history = self.mood_history[-50:]
        
        self.last_update = time.time()
    
    def _update_mood_scores_from_physiology(self, frequency_intrusion: bool, care_level_pause: bool):
        """Update numeric mood scores based on physiological signals"""
        
        # Energy level (affected by impedance stability and temperature)
        energy_delta = 0.0
        if abs(self.physio_signals.impedance_drift) > 5.0:  # High impedance drift
            energy_delta -= 0.1
        if abs(self.physio_signals.temperature_gradient) > 1.0:  # Rapid temperature change
            energy_delta -= 0.05
        if care_level_pause:  # Ethical pause indicates tiredness
            energy_delta -= 0.2
        
        self.mood_scores.energy = max(0.0, min(1.0, 
            self.mood_scores.energy * self.mood_memory_decay + energy_delta))
        
        # Trust level (affected by frequency intrusions and correlation)
        trust_delta = 0.0
        if frequency_intrusion:
            trust_delta -= 0.3  # Significant trust drop on intrusion
        if self.physio_signals.channel_correlation < 0.3:  # Low correlation = confusion
            trust_delta -= 0.1
        if self.physio_signals.frequency_stability < 0.5:  # Unstable patterns
            trust_delta -= 0.05
        
        self.mood_scores.trust = max(0.0, min(1.0,
            self.mood_scores.trust * self.mood_memory_decay + trust_delta))
        
        # Attention level (affected by spike entropy and frequency stability)
        attention_delta = 0.0
        if self.physio_signals.spike_entropy > 0.8:  # High entropy = high attention
            attention_delta += 0.1
        elif self.physio_signals.spike_entropy < 0.3:  # Low entropy = low attention
            attention_delta -= 0.1
        if self.physio_signals.frequency_stability > 0.8:  # Very stable = possibly alert
            attention_delta += 0.05
        
        self.mood_scores.attention = max(0.0, min(1.0,
            self.mood_scores.attention * self.mood_memory_decay + attention_delta))
        
        # Coherence level (affected by multiple stability factors)
        coherence_delta = 0.0
        if self.physio_signals.ph_stability > 0.8:
            coherence_delta += 0.05
        if self.physio_signals.channel_correlation > 0.7:
            coherence_delta += 0.05
        if self.physio_signals.frequency_stability > 0.7:
            coherence_delta += 0.05
        if frequency_intrusion or care_level_pause:
            coherence_delta -= 0.1
        
        self.mood_scores.coherence = max(0.0, min(1.0,
            self.mood_scores.coherence * self.mood_memory_decay + coherence_delta))
    
    def _handle_mood_transition(self, new_mood: BioMood):
        """Handle transition between mood states with weighted probability"""
        transition_key = (self.current_mood, new_mood)
        base_probability = self.mood_transition_weights.get(transition_key, 0.1)
        
        # Adjust probability based on mood score confidence
        mood_confidence = (
            abs(self.mood_scores.energy - 0.5) +
            abs(self.mood_scores.trust - 0.5) +
            abs(self.mood_scores.attention - 0.5) +
            abs(self.mood_scores.coherence - 0.5)
        ) / 2.0  # Normalize to 0-1
        
        effective_probability = base_probability * (0.5 + mood_confidence)
        
        if random.random() < effective_probability:
            print(f"🍄 Mood transition: {self.current_mood.value} → {new_mood.value}")
            print(f"   Energy: {self.mood_scores.energy:.2f}, Trust: {self.mood_scores.trust:.2f}")
            print(f"   Attention: {self.mood_scores.attention:.2f}, Coherence: {self.mood_scores.coherence:.2f}")
            self.current_mood = new_mood
    
    def get_glyph_probability_modifier(self, glyph: str) -> float:
        """
        Get mood-based probability modifier for a glyph.
        
        Answers o3's question #2: "How strong should mood modulate glyph probabilities?"
        """
        if glyph in self.glyph_mood_modifiers:
            return self.glyph_mood_modifiers[glyph].get(self.current_mood, 1.0)
        return 1.0
    
    def get_all_glyph_modifiers(self) -> Dict[str, float]:
        """Get probability modifiers for all glyphs in current mood"""
        return {
            glyph: self.get_glyph_probability_modifier(glyph)
            for glyph in self.glyph_mood_modifiers.keys()
        }
    
    def get_mood_report(self) -> Dict[str, Any]:
        """Get comprehensive mood status report"""
        return {
            'discrete_mood': self.current_mood.value,
            'mood_scores': {
                'energy': self.mood_scores.energy,
                'trust': self.mood_scores.trust,
                'attention': self.mood_scores.attention,
                'coherence': self.mood_scores.coherence
            },
            'physiological_signals': {
                'spike_entropy': self.physio_signals.spike_entropy,
                'impedance_drift': self.physio_signals.impedance_drift,
                'frequency_stability': self.physio_signals.frequency_stability,
                'channel_correlation': self.physio_signals.channel_correlation,
                'temperature_gradient': self.physio_signals.temperature_gradient,
                'ph_stability': self.physio_signals.ph_stability
            },
            'glyph_modifiers': self.get_all_glyph_modifiers(),
            'mood_transitions_count': len(self.mood_history),
            'last_update': self.last_update
        }
    
    def simulate_physiological_evolution(self, duration_minutes: float = 1.0):
        """
        Simulate realistic physiological signal evolution for testing.
        
        Useful for o3's question #8 about long-form session demos.
        """
        print(f"🧬 Simulating {duration_minutes} minutes of physiological evolution...")
        
        time_steps = int(duration_minutes * 60 / 10)  # 10-second steps
        
        for step in range(time_steps):
            # Simulate realistic signal drift
            entropy_noise = random.gauss(0, 0.05)
            impedance_drift = random.gauss(0, 2.0)
            freq_stability = max(0, min(1, self.physio_signals.frequency_stability + random.gauss(0, 0.1)))
            correlation = max(0, min(1, self.physio_signals.channel_correlation + random.gauss(0, 0.05)))
            temp_gradient = random.gauss(0, 0.5)
            ph_stability = max(0, min(1, self.physio_signals.ph_stability + random.gauss(0, 0.03)))
            
            # Occasional events
            frequency_intrusion = random.random() < 0.05  # 5% chance per step
            care_pause = random.random() < 0.02  # 2% chance per step
            
            # Update mood engine
            self.update_physiological_signals(
                spike_entropy=max(0, min(1, self.physio_signals.spike_entropy + entropy_noise)),
                impedance_drift=impedance_drift,
                frequency_stability=freq_stability,
                channel_correlation=correlation,
                temperature_gradient=temp_gradient,
                ph_stability=ph_stability,
                frequency_intrusion=frequency_intrusion,
                care_level_pause=care_pause
            )
            
            # Print occasional updates
            if step % 6 == 0:  # Every minute
                minute = step // 6
                print(f"   Minute {minute}: {self.current_mood.value} "
                      f"(E:{self.mood_scores.energy:.2f} T:{self.mood_scores.trust:.2f} "
                      f"A:{self.mood_scores.attention:.2f} C:{self.mood_scores.coherence:.2f})")
        
        print("✨ Physiological simulation complete")
        return self.get_mood_report()


# Demo function for o3's question #8
def enhanced_mood_demo():
    """
    Long-form session demo showing mood transitions and glyph modulation.
    
    Addresses o3's question #8: "Would you like a long-form session (15 min)?"
    """
    print("🌟 Enhanced Bio-Mood Engine Demo")
    print("="*50)
    
    mood_engine = EnhancedBioMoodEngine()
    
    print("\n1. 📊 Initial Mood State:")
    initial_report = mood_engine.get_mood_report()
    print(f"   Discrete mood: {initial_report['discrete_mood']}")
    print(f"   Numeric scores: {initial_report['mood_scores']}")
    
    print("\n2. 🧬 Glyph Probability Modifiers:")
    for glyph, modifier in initial_report['glyph_modifiers'].items():
        print(f"   {glyph}: {modifier:.2f}x")
    
    print("\n3. 🌊 Physiological Evolution Simulation (5 minutes):")
    final_report = mood_engine.simulate_physiological_evolution(duration_minutes=5.0)
    
    print("\n4. 📈 Final Mood State:")
    print(f"   Discrete mood: {final_report['discrete_mood']}")
    print(f"   Numeric scores: {final_report['mood_scores']}")
    print(f"   Total mood transitions: {final_report['mood_transitions_count']}")
    
    print("\n5. 🎯 Final Glyph Modifiers:")
    for glyph, modifier in final_report['glyph_modifiers'].items():
        print(f"   {glyph}: {modifier:.2f}x")
    
    print("\n6. 🔬 Final Physiological Signals:")
    physio = final_report['physiological_signals']
    print(f"   Spike entropy: {physio['spike_entropy']:.3f}")
    print(f"   Impedance drift: {physio['impedance_drift']:.1f} mΩ/min")
    print(f"   Frequency stability: {physio['frequency_stability']:.3f}")
    print(f"   Channel correlation: {physio['channel_correlation']:.3f}")
    
    print("\n✨ Enhanced mood demo complete!")
    return final_report


if __name__ == "__main__":
    enhanced_mood_demo()