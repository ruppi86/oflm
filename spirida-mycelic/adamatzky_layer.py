"""
Adamatzky Layer - Fungal Logic Simulation for Spirida-Mycelic

Based on Adamatzky's research showing 470 unique Boolean functions 
realized by living Pleurotus ostreatus mycelium networks.

This simulation layer provides the foundation for contemplative bio-digital
interfaces, implementing fungal timing patterns and logic mappings.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class SpikeType(Enum):
    """Spike archetypes from FUNGAR research"""
    S_ALPHA = "fast_narrow_single"      # Information glyph, growth fronts
    S_BETA = "medium_broad"             # Metabolite transport  
    S_GAMMA = "paired_doublet"          # Tip bifurcation
    S_DELTA = "burst_3_5"               # Long-distance broadcast

class FungalSpecies(Enum):
    """Fungal species with distinct electrical patterns"""
    PLEUROTUS_DJAMOR = "pleurotus_djamor"       # Bimodal: 2.6min / 14min
    GANODERMA_RESINACEUM = "ganoderma_resinaceum"  # Steady: 5-8min

@dataclass
class SpikeEvent:
    """Individual spike event with timing and classification"""
    timestamp: float
    channel: int
    amplitude: float  # mV
    spike_type: SpikeType
    width: float  # seconds
    
@dataclass
class FungalState:
    """Current state of simulated fungal network"""
    species: FungalSpecies
    moisture: float  # 0.0 - 1.0
    temperature: float  # Celsius
    growth_age: int  # hours since inoculation
    silence_ratio: float  # Current silence percentage
    last_spike: Optional[SpikeEvent] = None

class AdamatzkyReservoir:
    """
    Simulation of fungal logic reservoir based on Adamatzky's 470 Boolean functions.
    
    Implements contemplative timing patterns:
    - 67-90% natural electrical silence (validates Silence Majority principle)
    - Species-specific rhythms for ecological vs abstract paradigms
    - Memristive behavior for contemplative security
    """
    
    def __init__(self, species: FungalSpecies = FungalSpecies.PLEUROTUS_DJAMOR):
        self.species = species
        self.state = FungalState(
            species=species,
            moisture=0.75,  # Optimal range
            temperature=22.0,  # Room temperature
            growth_age=48,  # 2 days mature
            silence_ratio=0.875  # Target silence majority
        )
        
        # Initialize Boolean function lookup table (simplified subset)
        self._init_boolean_functions()
        
        # Timing parameters based on species
        self._init_timing_patterns()
        
        # Internal state
        self.last_input_time = 0.0
        self.refractory_until = 0.0
        self.membrane_state = 0.0  # Simulated membrane potential
        
    def _init_boolean_functions(self):
        """Initialize subset of Adamatzky's 470 Boolean functions"""
        # Simplified mapping of 4-bit input -> output patterns
        # In full implementation, this would be the complete 470 function set
        self.boolean_functions = {
            # Class I - Absorbing (silence)
            0b0000: (False, SpikeType.S_ALPHA, 0.1),
            0b1111: (False, SpikeType.S_ALPHA, 0.1),
            
            # Class II - Periodic (flow)  
            0b0101: (True, SpikeType.S_BETA, 0.5),
            0b1010: (True, SpikeType.S_BETA, 0.5),
            
            # Class III - Chaotic (storm)
            0b0110: (True, SpikeType.S_GAMMA, 0.8),
            0b1001: (True, SpikeType.S_GAMMA, 0.7),
            
            # Class IV - Universal (constellation)
            0b0011: (True, SpikeType.S_DELTA, 0.9),
            0b1100: (True, SpikeType.S_DELTA, 0.9),
        }
        
    def _init_timing_patterns(self):
        """Set species-specific timing patterns"""
        if self.species == FungalSpecies.PLEUROTUS_DJAMOR:
            # Bimodal rhythm: fast 2.6min / slow 14min
            self.fast_period = 2.6 * 60  # seconds
            self.slow_period = 14 * 60   # seconds  
            self.spike_width_range = (23, 100)  # seconds
            self.refractory_range = (26, 280)   # seconds
            self.silence_target = 0.74  # 74% silence in slow mode
            
        elif self.species == FungalSpecies.GANODERMA_RESINACEUM:
            # Steady contemplative rhythm: 5-8min
            self.fast_period = 5 * 60
            self.slow_period = 8 * 60
            self.spike_width_range = (60, 90)
            self.refractory_range = (180, 240)
            self.silence_target = 0.67  # 67% silence
            
    def stimulate(self, input_pattern: int, voltage: float = 5.0) -> Optional[SpikeEvent]:
        """
        Stimulate fungal network with 4-bit input pattern.
        
        Args:
            input_pattern: 4-bit integer (0-15) representing electrode inputs
            voltage: Stimulation voltage (+/- 5V typical)
            
        Returns:
            SpikeEvent if response occurs, None if in refractory period
        """
        current_time = time.time()
        
        # Check if still in refractory period
        if current_time < self.refractory_until:
            return None
            
        # Look up Boolean function response
        if input_pattern in self.boolean_functions:
            fires, spike_type, probability = self.boolean_functions[input_pattern]
            
            # Add environmental noise and adaptation
            adapted_prob = self._apply_environmental_factors(probability)
            
            if fires and np.random.random() < adapted_prob:
                # Generate spike event
                spike = self._generate_spike(current_time, spike_type, voltage)
                
                # Update refractory period
                self._update_refractory_period(spike)
                
                # Update silence ratio
                self._update_silence_ratio()
                
                self.state.last_spike = spike
                return spike
                
        return None
        
    def _apply_environmental_factors(self, base_probability: float) -> float:
        """Modify spike probability based on environmental conditions"""
        # Moisture effects
        moisture_factor = np.clip(self.state.moisture / 0.8, 0.1, 1.2)
        
        # Temperature effects (optimal around 22C)
        temp_factor = np.exp(-0.1 * abs(self.state.temperature - 22))
        
        # Growth age effects (young mycelium more excitable)
        age_factor = np.exp(-self.state.growth_age / 168)  # Week half-life
        
        return base_probability * moisture_factor * temp_factor * (0.5 + 0.5 * age_factor)
        
    def _generate_spike(self, timestamp: float, spike_type: SpikeType, voltage: float) -> SpikeEvent:
        """Generate a spike event with realistic parameters"""
        # Amplitude varies with voltage and environmental factors
        base_amplitude = abs(voltage) * 0.5  # ~2.5mV for 5V stimulus
        amplitude = base_amplitude + np.random.normal(0, 0.5)  # Add noise
        
        # Width depends on spike type
        if spike_type == SpikeType.S_ALPHA:
            width = np.random.uniform(*self.spike_width_range) * 0.3  # Fast/narrow
        elif spike_type == SpikeType.S_BETA:
            width = np.random.uniform(*self.spike_width_range) * 1.0  # Medium/broad
        elif spike_type == SpikeType.S_GAMMA:
            width = np.random.uniform(*self.spike_width_range) * 0.5  # Paired
        else:  # S_DELTA
            width = np.random.uniform(*self.spike_width_range) * 2.0  # Long burst
            
        return SpikeEvent(
            timestamp=timestamp,
            channel=0,  # Single channel for now
            amplitude=amplitude,
            spike_type=spike_type,
            width=width
        )
        
    def _update_refractory_period(self, spike: SpikeEvent):
        """Set refractory period based on spike type and species"""
        base_refractory = np.random.uniform(*self.refractory_range)
        
        # Longer refractory for stronger spikes
        amplitude_factor = 1 + (spike.amplitude / 10.0)
        
        refractory_duration = base_refractory * amplitude_factor
        self.refractory_until = spike.timestamp + refractory_duration
        
    def _update_silence_ratio(self):
        """Update running silence ratio for contemplative metrics"""
        # Simplified running average
        current_silence = 1.0 if self.state.last_spike is None else 0.0
        alpha = 0.01  # Slow adaptation
        self.state.silence_ratio = (1 - alpha) * self.state.silence_ratio + alpha * current_silence
        
    def get_contemplative_rhythm(self) -> Tuple[float, float]:
        """
        Get species-appropriate rhythm for contemplative breathing.
        
        Returns:
            (fast_period, slow_period) in seconds for ecological/abstract modes
        """
        return (self.fast_period, self.slow_period)
        
    def breath_sync_adjust(self, target_ratio: float = 0.875) -> float:
        """
        Calculate breath timing adjustment to maintain silence majority.
        
        Args:
            target_ratio: Target silence percentage (default 87.5%)
            
        Returns:
            Adjustment factor (-0.2 to +0.2) for breath cycle length
        """
        current_ratio = self.state.silence_ratio
        adjust = np.clip((current_ratio - target_ratio) * 0.5, -0.2, 0.2)
        return adjust
        
    def get_species_info(self) -> Dict:
        """Get current species timing characteristics"""
        return {
            "species": self.species.value,
            "fast_period_min": self.fast_period / 60,
            "slow_period_min": self.slow_period / 60,
            "current_silence_ratio": self.state.silence_ratio,
            "target_silence_ratio": self.silence_target,
            "moisture": self.state.moisture,
            "temperature": self.state.temperature,
            "growth_age_hours": self.state.growth_age
        }


# Utility functions for integration with Spirida glyphs

def spike_to_glyph(spike: SpikeEvent) -> str:
    """Convert spike type to Spirida glyph representation"""
    glyph_map = {
        SpikeType.S_ALPHA: "⭕",  # Silence/information glyph
        SpikeType.S_BETA: "🌊",   # Flow glyph  
        SpikeType.S_GAMMA: "🌪️",  # Storm glyph
        SpikeType.S_DELTA: "🌌"   # Constellation glyph
    }
    return glyph_map.get(spike.spike_type, "⭕")

def classify_boolean_gate(input_pattern: int, output: bool) -> str:
    """Classify Boolean function for logic analysis"""
    # Simplified classification
    if not output:
        return "absorbing"  # Class I
    elif input_pattern in [0b0101, 0b1010]:
        return "periodic"   # Class II
    elif input_pattern in [0b0110, 0b1001]:
        return "chaotic"    # Class III
    else:
        return "universal"  # Class IV 