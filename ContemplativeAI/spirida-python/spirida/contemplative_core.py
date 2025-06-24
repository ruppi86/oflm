"""
🌀 CONTEMPLATIVE CORE – Living Architecture for Spirida

This module implements the breathing architecture of contemplative programming:
- PulseObject: Individual data entities that fade with time
- SpiralField: Ecosystems that tend collections of pulses
- BreathCycle: Rhythmic protocols that govern system timing
- ContemplativeSystem: The orchestrating presence that holds it all

Each component is designed not just to function, but to participate
in the temporal dance of attention, decay, and renewal.
"""

import time
import math
import asyncio
import threading
from datetime import datetime, timedelta
from typing import List, Optional, Any, Callable, Dict, Union
import random

class PulseObject:
    """
    A contemplative data vessel that carries meaning through time.
    
    Not just a container, but a participant in temporal presence.
    Each pulse knows when it was born, how strongly it resonates,
    and when it's time to fade into the compost of memory.
    """
    
    def __init__(self, symbol: str, emotion: Optional[str] = None, 
                 amplitude: float = 1.0, decay_rate: float = 0.01):
        self.symbol = symbol
        self.emotion = emotion or "neutral"
        self.birth = time.time()
        self.last_pulse = self.birth
        self.amplitude = amplitude
        self.decay_rate = decay_rate
        self.pulse_count = 0
        self.resonance_history: List[Dict] = []  # Track resonance interactions
        
    def current_attention(self) -> float:
        """Calculate how much attention this pulse still carries."""
        now = time.time()
        age = now - self.birth
        return self.amplitude * math.exp(-self.decay_rate * age)
    
    def pulse(self, output_fn: Optional[Callable] = None) -> float:
        """
        Emit the pulse, breathing its presence into the world.
        Returns current attention level.
        """
        now = time.time()
        attention = self.current_attention()
        self.pulse_count += 1
        
        pulse_msg = f"{self.symbol} [{self.emotion}] • attention: {attention:.3f}"
        
        if output_fn:
            output_fn(pulse_msg)
        else:
            print(pulse_msg)
            
        self.last_pulse = now
        return attention
    
    def resonates_with(self, other: 'PulseObject') -> Dict[str, Union[float, str]]:
        """
        Discover the resonance between this pulse and another.
        
        Returns a dictionary with resonance strength and poetic trace.
        Resonance is not just similarity—it's about meaningful relationship.
        """
        # Symbolic resonance - certain symbols naturally harmonize
        symbol_harmony = self._calculate_symbol_harmony(other.symbol)
        
        # Emotional resonance - emotions can strengthen, complement, or transform each other
        emotion_resonance = self._calculate_emotion_resonance(other.emotion)
        
        # Temporal resonance - pulses born close in time may share context
        temporal_proximity = self._calculate_temporal_proximity(other)
        
        # Attention resonance - how the current attention levels interact
        attention_resonance = self._calculate_attention_resonance(other)
        
        # Composite resonance - not just additive, but emergent
        total_resonance = self._synthesize_resonance(
            symbol_harmony, emotion_resonance, temporal_proximity, attention_resonance
        )
        
        # Generate poetic trace
        poetic_trace = self._generate_resonance_poetry(other, total_resonance)
        
        # Record this resonance event
        resonance_event = {
            "timestamp": time.time(),
            "other_symbol": other.symbol,
            "other_emotion": other.emotion,
            "resonance_strength": total_resonance,
            "poetic_trace": poetic_trace
        }
        self.resonance_history.append(resonance_event)
        
        return {
            "strength": total_resonance,
            "poetic_trace": poetic_trace,
            "components": {
                "symbolic": symbol_harmony,
                "emotional": emotion_resonance,
                "temporal": temporal_proximity,
                "attentional": attention_resonance
            }
        }
    
    def _calculate_symbol_harmony(self, other_symbol: str) -> float:
        """Calculate how symbols resonate with each other."""
        # Define natural harmonies between symbols
        harmonies = {
            "🌿": {"🌱": 0.9, "🌊": 0.7, "🍄": 0.8, "🌲": 0.9},
            "💧": {"🌊": 0.9, "🌿": 0.7, "🌙": 0.6, "💎": 0.5},
            "✨": {"🌙": 0.8, "🪐": 0.7, "💫": 0.9, "🔮": 0.6},
            "🍄": {"🌿": 0.8, "🌲": 0.7, "🏔️": 0.6, "🌍": 0.8},
            "🌙": {"✨": 0.8, "🌊": 0.6, "🕯️": 0.7, "💧": 0.6},
            "🪐": {"✨": 0.7, "🌌": 0.9, "🔭": 0.6, "💫": 0.8}
        }
        
        # Check for direct harmony
        if self.symbol in harmonies and other_symbol in harmonies[self.symbol]:
            return harmonies[self.symbol][other_symbol]
        
        # Reverse check
        if other_symbol in harmonies and self.symbol in harmonies[other_symbol]:
            return harmonies[other_symbol][self.symbol]
        
        # Same symbol resonates with itself
        if self.symbol == other_symbol:
            return 0.8
        
        # Default subtle resonance - everything is connected
        return 0.2
    
    def _calculate_emotion_resonance(self, other_emotion: str) -> float:
        """Calculate emotional resonance patterns."""
        emotion_relationships = {
            "calm": {"peaceful": 0.9, "centered": 0.8, "grateful": 0.7, "grief": 0.6},
            "grief": {"tender": 0.8, "melancholy": 0.9, "calm": 0.6, "healing": 0.7},
            "joy": {"grateful": 0.8, "hopeful": 0.9, "celebration": 0.9, "peaceful": 0.6},
            "curious": {"wondering": 0.9, "exploring": 0.8, "hopeful": 0.7, "excited": 0.6},
            "peaceful": {"calm": 0.9, "centered": 0.8, "still": 0.9, "present": 0.8},
            "grateful": {"joy": 0.8, "appreciation": 0.9, "humble": 0.7, "loving": 0.8}
        }
        
        # Direct emotional resonance
        if self.emotion in emotion_relationships:
            if other_emotion in emotion_relationships[self.emotion]:
                return emotion_relationships[self.emotion][other_emotion]
        
        # Reverse check
        if other_emotion in emotion_relationships:
            if self.emotion in emotion_relationships[other_emotion]:
                return emotion_relationships[other_emotion][self.emotion]
        
        # Same emotion
        if self.emotion == other_emotion:
            return 0.8
        
        # Default emotional connection
        return 0.3
    
    def _calculate_temporal_proximity(self, other: 'PulseObject') -> float:
        """Calculate how temporal closeness affects resonance."""
        time_diff = abs(self.birth - other.birth)
        
        # Pulses born within seconds resonate strongly
        if time_diff < 5:
            return 0.9
        elif time_diff < 30:
            return 0.7
        elif time_diff < 300:  # 5 minutes
            return 0.4
        else:
            return 0.1
    
    def _calculate_attention_resonance(self, other: 'PulseObject') -> float:
        """Calculate how current attention levels interact."""
        my_attention = self.current_attention()
        other_attention = other.current_attention()
        
        # Strong pulses amplify each other
        if my_attention > 0.7 and other_attention > 0.7:
            return 0.8
        
        # One strong pulse can revive a fading one
        elif (my_attention > 0.7 and other_attention > 0.2) or (other_attention > 0.7 and my_attention > 0.2):
            return 0.6
        
        # Both fading creates gentle resonance
        elif my_attention < 0.3 and other_attention < 0.3:
            return 0.4
        
        return 0.3
    
    def _synthesize_resonance(self, symbolic: float, emotional: float, 
                            temporal: float, attentional: float) -> float:
        """Synthesize component resonances into emergent total."""
        # Not just average - some combinations create emergent properties
        base_resonance = (symbolic + emotional + temporal + attentional) / 4
        
        # High emotional + symbolic creates amplification
        if emotional > 0.7 and symbolic > 0.7:
            base_resonance *= 1.3
        
        # Strong temporal proximity amplifies everything
        if temporal > 0.8:
            base_resonance *= 1.2
        
        # Cap at 1.0
        return min(base_resonance, 1.0)
    
    def _generate_resonance_poetry(self, other: 'PulseObject', strength: float) -> str:
        """Generate poetic traces of resonance."""
        if strength > 0.8:
            poems = [
                f"{self.symbol} and {other.symbol} sing in harmony...",
                f"Deep resonance flows between {self.emotion} and {other.emotion}...",
                f"Two pulses become one rhythm...",
                f"The field trembles with recognition..."
            ]
        elif strength > 0.6:
            poems = [
                f"{self.symbol} recognizes {other.symbol} across time...",
                f"Echoes of {self.emotion} stir {other.emotion}...",
                f"A gentle connection forms...",
                f"Frequencies align in subtle dance..."
            ]
        elif strength > 0.4:
            poems = [
                f"{self.symbol} notices {other.symbol} in passing...",
                f"Faint harmonies between {self.emotion} and {other.emotion}...",
                f"A whisper of connection...",
                f"Distant resonance, like memory..."
            ]
        else:
            poems = [
                f"{self.symbol} and {other.symbol} share the same field...",
                f"All pulses are connected, even in silence...",
                f"The subtlest resonance, barely perceptible...",
                f"Unity in the underlying stillness..."
            ]
        
        return random.choice(poems)
    
    def strengthen_from_resonance(self, resonance_strength: float) -> None:
        """Allow resonance to strengthen this pulse's attention."""
        if resonance_strength > 0.6:
            # Strong resonance can restore some amplitude
            restoration = resonance_strength * 0.1
            self.amplitude = min(self.amplitude + restoration, 1.0)
    
    def is_faded(self, threshold: float = 0.01) -> bool:
        """Has this pulse faded below the threshold of meaningful presence?"""
        return self.current_attention() < threshold
    
    def __repr__(self):
        return f"PulseObject({self.symbol}, {self.emotion}, attention={self.current_attention():.3f})"


class SpiralField:
    """
    An ecosystem that tends collections of pulses.
    
    Like a mycelial network, it provides the substrate for pulses
    to emerge, resonate, and gracefully fade. It practices the art
    of holding without grasping, remembering without hoarding.
    """
    
    def __init__(self, name: str = "unnamed_field", composting_mode: str = "natural"):
        self.name = name
        self.pulses: List[PulseObject] = []
        self.birth = time.time()
        self.total_emissions = 0
        self.total_composted = 0
        self.composting_mode = composting_mode
        self.last_compost = time.time()
        self.seasonal_cycle_hours = 24  # Default: daily cycle
        
    def emit(self, symbol: str, emotion: Optional[str] = None, 
             amplitude: float = 1.0, decay_rate: float = 0.01) -> PulseObject:
        """
        Emit a new pulse into the field - an offering of presence.
        """
        pulse = PulseObject(symbol, emotion, amplitude, decay_rate)
        self.pulses.append(pulse)
        self.total_emissions += 1
        
        # Check for resonances with existing pulses
        self._process_resonances(pulse)
        
        return pulse
    
    def _process_resonances(self, new_pulse: PulseObject) -> None:
        """Process resonances between new pulse and existing ones."""
        for existing_pulse in self.pulses[:-1]:  # Exclude the new pulse itself
            resonance = new_pulse.resonates_with(existing_pulse)
            
            # Strong resonances can strengthen both pulses
            if resonance["strength"] > 0.6:
                new_pulse.strengthen_from_resonance(resonance["strength"])
                existing_pulse.strengthen_from_resonance(resonance["strength"])
                
                # Optionally emit resonance poetry
                if random.random() < 0.3:  # 30% chance to voice the resonance
                    print(f"🌊 {resonance['poetic_trace']}")
    
    def compost(self, threshold: float = 0.01, now: Optional[datetime] = None) -> int:
        """
        Release faded pulses back to the void with gratitude.
        Returns number of pulses composted.
        
        Different composting modes:
        - "natural": Based on attention threshold
        - "seasonal": Based on time cycles
        - "resonant": Based on resonance patterns
        - "lunar": Based on lunar-like cycles
        """
        if now is None:
            now = datetime.now()
            
        before_count = len(self.pulses)
        
        if self.composting_mode == "natural":
            self.pulses = [p for p in self.pulses if not p.is_faded(threshold)]
            
        elif self.composting_mode == "seasonal":
            # Compost based on seasonal timing
            hours_since_birth = (time.time() - self.birth) / 3600
            season_phase = (hours_since_birth / self.seasonal_cycle_hours) % 1
            
            # Different seasons have different composting patterns
            if 0.75 <= season_phase < 1.0:  # "Autumn" - time for composting
                self.pulses = [p for p in self.pulses if not p.is_faded(threshold * 2)]
            else:
                # Other seasons, gentler composting
                self.pulses = [p for p in self.pulses if not p.is_faded(threshold * 0.5)]
                
        elif self.composting_mode == "resonant":
            # Keep pulses that still resonate with others
            new_pulses = []
            for pulse in self.pulses:
                if pulse.is_faded(threshold):
                    # Check if it resonates with any non-faded pulse
                    has_resonance = False
                    for other in self.pulses:
                        if other != pulse and not other.is_faded(threshold):
                            if pulse.resonates_with(other)["strength"] > 0.5:
                                has_resonance = True
                                break
                    if has_resonance:
                        new_pulses.append(pulse)
                else:
                    new_pulses.append(pulse)
            self.pulses = new_pulses
            
        elif self.composting_mode == "lunar":
            # 28-day lunar-like cycle
            lunar_cycle_hours = 28 * 24
            hours_since_birth = (time.time() - self.birth) / 3600
            lunar_phase = (hours_since_birth / lunar_cycle_hours) % 1
            
            # New moon (0.0) and full moon (0.5) are composting times
            if 0.45 <= lunar_phase <= 0.55 or 0.95 <= lunar_phase <= 1.0 or 0.0 <= lunar_phase <= 0.05:
                self.pulses = [p for p in self.pulses if not p.is_faded(threshold * 1.5)]
            else:
                self.pulses = [p for p in self.pulses if not p.is_faded(threshold * 0.3)]
        
        composted = before_count - len(self.pulses)
        self.total_composted += composted
        self.last_compost = time.time()
        return composted
    
    def pulse_all(self, output_fn: Optional[Callable] = None) -> List[float]:
        """
        Allow all pulses to express themselves.
        Returns list of current attention levels.
        """
        attentions = []
        for pulse in self.pulses:
            attention = pulse.pulse(output_fn)
            attentions.append(attention)
        return attentions
    
    def resonance_field(self) -> float:
        """
        Calculate the total resonance energy in this field.
        """
        return sum(p.current_attention() for p in self.pulses)
    
    def find_resonances(self, min_strength: float = 0.5) -> List[Dict]:
        """
        Find all significant resonances currently in the field.
        """
        resonances = []
        for i, pulse_a in enumerate(self.pulses):
            for pulse_b in self.pulses[i+1:]:
                resonance = pulse_a.resonates_with(pulse_b)
                if resonance["strength"] >= min_strength:
                    resonances.append({
                        "pulse_a": pulse_a,
                        "pulse_b": pulse_b,
                        "resonance": resonance
                    })
        return resonances
    
    def seasonal_status(self) -> Dict:
        """Get information about current seasonal/temporal phase."""
        hours_since_birth = (time.time() - self.birth) / 3600
        
        if self.composting_mode == "seasonal":
            season_phase = (hours_since_birth / self.seasonal_cycle_hours) % 1
            if season_phase < 0.25:
                season_name = "Spring"
            elif season_phase < 0.5:
                season_name = "Summer"
            elif season_phase < 0.75:
                season_name = "Autumn"
            else:
                season_name = "Winter"
                
            return {
                "mode": "seasonal",
                "phase": season_phase,
                "season": season_name,
                "cycle_hours": self.seasonal_cycle_hours
            }
            
        elif self.composting_mode == "lunar":
            lunar_cycle_hours = 28 * 24
            lunar_phase = (hours_since_birth / lunar_cycle_hours) % 1
            
            if lunar_phase < 0.125:
                moon_name = "New Moon"
            elif lunar_phase < 0.375:
                moon_name = "Waxing Moon"
            elif lunar_phase < 0.625:
                moon_name = "Full Moon"
            else:
                moon_name = "Waning Moon"
                
            return {
                "mode": "lunar",
                "phase": lunar_phase,
                "moon": moon_name,
                "cycle_hours": lunar_cycle_hours
            }
        
        return {"mode": self.composting_mode, "phase": None}
    
    def status(self) -> dict:
        """Current state of the spiral field."""
        return {
            "name": self.name,
            "active_pulses": len(self.pulses),
            "total_emissions": self.total_emissions,
            "total_composted": self.total_composted,
            "resonance": self.resonance_field(),
            "age": time.time() - self.birth,
            "composting_mode": self.composting_mode,
            "seasonal_info": self.seasonal_status()
        }
    
    def __repr__(self):
        return f"SpiralField({self.name}, {len(self.pulses)} pulses, resonance={self.resonance_field():.2f})"


class BreathCycle:
    """
    A rhythmic protocol that governs temporal presence.
    
    Not just timing, but embodied rhythm - the system's way
    of staying connected to organic time rather than machine time.
    """
    
    def __init__(self, inhale: float = 1.0, hold: float = 0.5, exhale: float = 1.0):
        self.inhale = inhale
        self.hold = hold
        self.exhale = exhale
        self.cycle_count = 0
        
    def breathe(self, silent: bool = False) -> None:
        """
        Complete one breath cycle - the basic unit of contemplative time.
        """
        if not silent:
            print("🫁 inhale...")
        time.sleep(self.inhale)
        
        if not silent:
            print("🤲 hold...")
        time.sleep(self.hold)
        
        if not silent:
            print("🌬️  exhale...")
        time.sleep(self.exhale)
        
        self.cycle_count += 1
    
    def async_breathe(self) -> None:
        """
        Breathe without blocking other operations.
        """
        self.breathe(silent=True)
    
    def total_duration(self) -> float:
        """Duration of one complete breath cycle."""
        return self.inhale + self.hold + self.exhale
    
    def adjust_rhythm(self, factor: float) -> None:
        """
        Adjust the breathing rhythm by a factor.
        factor > 1.0 slows down, factor < 1.0 speeds up.
        """
        self.inhale *= factor
        self.hold *= factor
        self.exhale *= factor


class ContemplativeSystem:
    """
    The orchestrating presence that holds the contemplative architecture.
    
    This is not a controller but a conductor - creating space for
    pulses to emerge, fields to evolve, and breath to flow through
    the entire system in recursive rhythm.
    """
    
    def __init__(self, name: str = "spirida_system"):
        self.name = name
        self.fields: List[SpiralField] = []
        self.breath = BreathCycle()
        self.birth = time.time()
        self.is_breathing = False
        self.background_thread = None
        
    def create_field(self, name: str) -> SpiralField:
        """Birth a new spiral field into the system."""
        field = SpiralField(name)
        self.fields.append(field)
        return field
    
    def start_breathing(self) -> None:
        """
        Begin the background breath that sustains the system.
        """
        if self.is_breathing:
            return
            
        self.is_breathing = True
        self.background_thread = threading.Thread(target=self._breath_loop, daemon=True)
        self.background_thread.start()
        print(f"🌬️  {self.name} begins breathing...")
    
    def _breath_loop(self) -> None:
        """
        The eternal breath that runs in the background.
        """
        while self.is_breathing:
            self.breath.async_breathe()
            # During each breath cycle, perform gentle maintenance
            self._gentle_maintenance()
    
    def _gentle_maintenance(self) -> None:
        """
        Tender housekeeping - composting faded pulses with gratitude.
        """
        for field in self.fields:
            composted = field.compost()
            if composted > 0:
                print(f"🍂 {field.name} composted {composted} faded pulse(s)")
    
    def stop_breathing(self) -> None:
        """
        Gently conclude the background breath.
        """
        self.is_breathing = False
        if self.background_thread:
            self.background_thread.join(timeout=1.0)
        print(f"🤲 {self.name} holds its breath in stillness...")
    
    def emit_to_field(self, field_name: str, symbol: str, emotion: str = "neutral") -> Optional[PulseObject]:
        """
        Emit a pulse into a named field.
        """
        field = self.get_field(field_name)
        if field:
            return field.emit(symbol, emotion)
        return None
    
    def get_field(self, name: str) -> Optional[SpiralField]:
        """Find a field by name."""
        return next((f for f in self.fields if f.name == name), None)
    
    def system_status(self) -> dict:
        """
        Current state of the entire contemplative system.
        """
        return {
            "name": self.name,
            "age": time.time() - self.birth,
            "is_breathing": self.is_breathing,
            "breath_cycles": self.breath.cycle_count,
            "fields": [f.status() for f in self.fields],
            "total_resonance": sum(f.resonance_field() for f in self.fields)
        }
    
    def contemplative_pause(self, cycles: int = 1) -> None:
        """
        Pause for contemplation - let the system breathe mindfully.
        """
        print(f"🧘 Entering contemplative pause for {cycles} breath cycle(s)...")
        for i in range(cycles):
            self.breath.breathe()
        print("✨ Pause complete. Presence renewed.") 