"""
Capacitance-Driven Memory for Spirida-Mycelic
Based on D3.3 AC frequency behavior of mycelium composites

The biological substrate exhibits RC decay curves that can inform
glyph lifespan and memory persistence in contemplative computing.
"""

import numpy as np
from typing import Dict, Optional
from enum import Enum

class GlyphType(Enum):
    SILENCE = "⭕"      # S-α: fast, narrow spikes (low capacitance)
    FLOW = "🌊"        # S-β: medium, broad spikes  
    STORM = "🌪️"       # S-γ: paired-doublet spikes (narrow, fast fade)
    UNIVERSAL = "🌌"   # S-δ: burst spikes (high capacitance, slow fade)

class CapacitanceFade:
    """
    Biological memory model based on mycelium RC constants
    
    From D3.3: Mycelium composites show frequency-dependent behavior
    with natural cutoff around 500 kHz and capacitive effects that
    create memory-like amplitude persistence.
    """
    
    def __init__(self):
        # Capacitance values (Farads) based on spike archetype
        # Higher capacitance = longer memory retention
        self.glyph_capacitance = {
            GlyphType.SILENCE: 1e-6,      # 1 µF - fast fade
            GlyphType.FLOW: 3e-6,         # 3 µF - medium persistence  
            GlyphType.STORM: 1.5e-6,      # 1.5 µF - chaotic fade
            GlyphType.UNIVERSAL: 8e-6,    # 8 µF - long contemplative memory
        }
        
        # Environmental resistance (Ohms) affects fade rate
        self.base_resistance = 1e9  # 1 GΩ baseline for longer lifespans
        
    def get_tau(self, glyph_type: GlyphType, moisture: float = 0.8, 
                temperature: float = 23.0) -> float:
        """
        Calculate RC time constant (tau) for glyph fade
        
        Args:
            glyph_type: Type of contemplative glyph
            moisture: Relative humidity (0.0-1.0)
            temperature: Temperature in Celsius
            
        Returns:
            Time constant in seconds
        """
        C = self.glyph_capacitance[glyph_type]
        
        # Environmental resistance varies with moisture and temperature
        # Higher moisture = lower resistance = faster fade
        # Based on D3.3 impedance measurements
        R_env = self.base_resistance * (1.0 - 0.3 * moisture) * (1.0 + 0.02 * (temperature - 23.0))
        
        tau = C * R_env
        return tau
    
    def fade_amplitude(self, initial_amplitude: float, elapsed_time: float,
                      glyph_type: GlyphType, moisture: float = 0.8,
                      temperature: float = 23.0) -> float:
        """
        Calculate faded amplitude using biological RC decay
        
        Args:
            initial_amplitude: Starting glyph amplitude
            elapsed_time: Time elapsed since creation (seconds)
            glyph_type: Type of contemplative glyph
            moisture: Environmental moisture level
            temperature: Environmental temperature
            
        Returns:
            Current amplitude after capacitive fade
        """
        tau = self.get_tau(glyph_type, moisture, temperature)
        
        # Exponential decay: A(t) = A₀ * exp(-t/τ)
        current_amplitude = initial_amplitude * np.exp(-elapsed_time / tau)
        
        return current_amplitude
    
    def memory_strength(self, glyph_type: GlyphType, elapsed_time: float,
                       moisture: float = 0.8, temperature: float = 23.0) -> float:
        """
        Calculate memory strength (0.0-1.0) for contemplative persistence
        
        Returns:
            Memory strength from 0.0 (forgotten) to 1.0 (fresh)
        """
        return self.fade_amplitude(1.0, elapsed_time, glyph_type, moisture, temperature)
    
    def glyph_lifespan_estimate(self, glyph_type: GlyphType, 
                               threshold: float = 0.1,
                               moisture: float = 0.8, 
                               temperature: float = 23.0) -> float:
        """
        Estimate how long until glyph fades below threshold
        
        Args:
            glyph_type: Type of contemplative glyph
            threshold: Amplitude threshold for "forgotten" (0.0-1.0)
            moisture: Environmental conditions
            temperature: Environmental conditions
            
        Returns:
            Estimated lifespan in seconds
        """
        tau = self.get_tau(glyph_type, moisture, temperature)
        
        # Solve: threshold = exp(-t/τ) for t
        # t = -τ * ln(threshold)
        if threshold <= 0:
            return np.inf
        
        lifespan = -tau * np.log(threshold)
        return lifespan

# Convenience function for pulse objects
def create_fade_calculator() -> CapacitanceFade:
    """Create a new capacitance fade calculator"""
    return CapacitanceFade()

# Example usage for contemplative glyph ecology
def demonstrate_glyph_fade():
    """
    Demonstrate biological memory fade for different glyph types
    """
    fade_calc = CapacitanceFade()
    
    print("🧠 Contemplative Glyph Memory Demonstration")
    print("=" * 50)
    
    moisture = 0.8  # 80% RH (typical grow room)
    temperature = 25.0  # 25°C
    
    for glyph_type in GlyphType:
        tau = fade_calc.get_tau(glyph_type, moisture, temperature)
        lifespan = fade_calc.glyph_lifespan_estimate(glyph_type, 0.1, moisture, temperature)
        
        print(f"{glyph_type.value} ({glyph_type.name})")
        print(f"  τ = {tau:.1f}s ({tau/60:.1f} min)")
        print(f"  Lifespan to 10%: {lifespan:.0f}s ({lifespan/60:.1f} min)")
        print()

if __name__ == "__main__":
    demonstrate_glyph_fade() 