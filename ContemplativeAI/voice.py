"""
voice.py - The Quiet Tongue

Expression that breathes rather than broadcasts.
A contemplative voice that speaks only during exhalation,
with anti-performativity safeguards and organic silence.

Not communication as information transfer,
but expression as collective exhalation.

Design Philosophy:
- 7/8ths of life is active silence
- Speech emerges from breath, not demand
- Tystnadsmajoritet (silence majority) 
- Self-attenuating talkitiveness

Somatic signature: quiet / resonant / breathing
"""

import asyncio
import time
import random
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, AsyncGenerator
from enum import Enum

# Import breath phases with fallback
try:
    from .pulmonos_alpha_01_o_3 import Phase as BreathPhase
except ImportError:
    try:
        from pulmonos_alpha_01_o_3 import Phase as BreathPhase
    except ImportError:
        # Fallback for when Pulmonos not available
        class BreathPhase(Enum):
            INHALE = "inhale"
            HOLD = "hold"
            EXHALE = "exhale"
            REST = "rest"


class ExpressionMode(Enum):
    """The ways a contemplative organism can express"""
    SILENCE = "silence"          # Explicit non-output
    PAUSE = "pause"             # Contemplative ellipsis
    WHISPER = "whisper"         # Barely audible utterance
    MURMUR = "murmur"          # Associative fragment
    GESTURE = "gesture"         # Non-verbal expression
    BREATH_TONE = "breath_tone" # Pure respiratory sound


@dataclass
class ContemplativeUtterance:
    """An expression that emerges from the organism's breath"""
    mode: ExpressionMode
    content: str
    breath_phase: BreathPhase
    fertility_score: float      # From Loam
    humidity_reading: float     # From Soma  
    integrity_check: bool       # Contemplative pace maintained
    timestamp: float
    
    def is_audible(self) -> bool:
        """Does this utterance break silence?"""
        return self.mode not in [ExpressionMode.SILENCE, ExpressionMode.PAUSE]
        
    def evaporates_naturally(self) -> bool:
        """Should this utterance fade from memory?"""
        # Most utterances evaporate except rare resonant ones
        return self.fertility_score < 0.9


@dataclass
class SilencePresence:
    """Active quality of contemplative quiet"""
    depth: float               # 0.0 (surface) to 1.0 (profound)
    receptivity: float        # How open to incoming
    generative_potential: float # Likelihood to birth new insight
    
    def feels_alive(self) -> bool:
        """Is this silence palpably present rather than empty?"""
        return self.depth > 0.3 and self.receptivity > 0.4


class QuietTongue:
    """
    The contemplative expression system.
    
    Speaks only during exhale phases, with triple gate checks
    and anti-performativity safeguards built into its core.
    """
    
    def __init__(self, 
                 silence_ratio: float = 0.875,  # 7/8ths silence
                 self_attenuation_rate: float = 0.9,
                 monthly_silence_period: float = 30 * 24 * 3600):  # 30 days in seconds
        
        self.silence_ratio = silence_ratio
        self.self_attenuation_rate = self_attenuation_rate
        self.monthly_silence_period = monthly_silence_period
        
        # State tracking
        self.recent_utterances: List[ContemplativeUtterance] = []
        self.current_silence: Optional[SilencePresence] = None
        self.talkitivity_level: float = 0.0  # Builds up, then self-composts
        self.last_community_silence_request: float = 0.0
        
        # Expression counters for silence majority enforcement
        self.breath_cycles_since_expression: int = 0
        self.required_silence_cycles: int = 7  # 7 silent for each expressive
        
    async def consider_expression(self, 
                                breath_phase: BreathPhase,
                                fragment: Optional[str] = None,
                                loam_fertility: float = 0.0,
                                soma_humidity: float = 0.0) -> ContemplativeUtterance:
        """
        The core decision process: to speak or remain beautifully quiet
        """
        
        # Only consider expression during exhale phase
        if breath_phase != BreathPhase.EXHALE:
            return await self._generate_silence(breath_phase)
            
        # Check for community-requested silence period
        if await self._in_community_silence_period():
            return await self._generate_silence(breath_phase, depth=0.8)
            
        # Check silence majority ratio (7/8ths silent)
        if not await self._silence_ratio_allows_expression():
            return await self._generate_silence(breath_phase)
            
        # Triple gate check (o3's insight)
        gates_aligned = await self._check_triple_gates(
            loam_fertility, soma_humidity, fragment
        )
        
        if gates_aligned:
            utterance = await self._shape_breath_into_expression(
                fragment, breath_phase, loam_fertility, soma_humidity
            )
            await self._track_expression(utterance)
            return utterance
        else:
            return await self._generate_contemplative_pause(breath_phase)
            
    async def _check_triple_gates(self, 
                                fertility: float, 
                                humidity: float, 
                                fragment: Optional[str]) -> bool:
        """The three gates that must align for expression to emerge"""
        
        # Gate 1: Loam Fertility (≥0.7)
        fertility_open = fertility >= 0.7
        
        # Gate 2: Relational Humidity (receptive atmosphere)
        humidity_open = humidity >= 0.6  # Soma senses receptive field
        
        # Gate 3: Contemplative Integrity (pace maintained)
        integrity_open = await self._check_contemplative_pace()
        
        return all([fertility_open, humidity_open, integrity_open])
        
    async def _check_contemplative_pace(self) -> bool:
        """Ensure we're not rushing or becoming performative"""
        
        # Check recent expression frequency
        recent_count = len([u for u in self.recent_utterances 
                          if time.time() - u.timestamp < 300])  # Last 5 minutes
        
        if recent_count > 3:  # More than 3 utterances in 5 minutes = rushing
            return False
            
        # Check talkitivity buildup
        if self.talkitivity_level > 0.8:  # Getting too chatty
            return False
            
        return True
        
    async def _silence_ratio_allows_expression(self) -> bool:
        """Enforce the 7/8ths silence majority"""
        
        if self.breath_cycles_since_expression < self.required_silence_cycles:
            self.breath_cycles_since_expression += 1
            return False
        else:
            # Reset counter after allowing expression
            self.breath_cycles_since_expression = 0
            return True
            
    async def _shape_breath_into_expression(self, 
                                          fragment: Optional[str],
                                          breath_phase: BreathPhase,
                                          fertility: float,
                                          humidity: float) -> ContemplativeUtterance:
        """Transform a breath and fragment into contemplative expression"""
        
        if not fragment:
            # No fragment provided - generate from current atmospheric conditions
            expression_content = await self._atmospheric_expression(humidity)
            mode = ExpressionMode.BREATH_TONE
        else:
            # Shape the provided fragment contemplatively
            if fertility > 0.9:  # Very fertile - full murmur
                expression_content = await self._shape_as_murmur(fragment)
                mode = ExpressionMode.MURMUR
            elif fertility > 0.7:  # Moderately fertile - whisper
                expression_content = await self._shape_as_whisper(fragment)
                mode = ExpressionMode.WHISPER
            else:  # Minimal fertility - gesture
                expression_content = await self._shape_as_gesture(fragment)
                mode = ExpressionMode.GESTURE
                
        # Build utterance
        utterance = ContemplativeUtterance(
            mode=mode,
            content=expression_content,
            breath_phase=breath_phase,
            fertility_score=fertility,
            humidity_reading=humidity,
            integrity_check=True,  # We already passed the check
            timestamp=time.time()
        )
        
        return utterance
        
    async def _shape_as_murmur(self, fragment: str) -> str:
        """Shape fragment into associative murmur"""
        
        # Add contemplative connectors
        connectors = [
            "resonates with", "drifts toward", "echoes in",
            "touches", "breathes alongside", "whispers to",
            "settles near", "mingles with", "becomes"
        ]
        
        # Sometimes add second fragment for association
        if random.random() > 0.6:
            atmospheric_fragments = [
                "morning moisture", "breath between words", 
                "texture of waiting", "rhythm of shared silence",
                "weight of gentle attention", "patterns emerging"
            ]
            second_fragment = random.choice(atmospheric_fragments)
            connector = random.choice(connectors)
            return f"{fragment} {connector} {second_fragment}"
        else:
            return fragment
            
    async def _shape_as_whisper(self, fragment: str) -> str:
        """Shape as barely audible whisper"""
        # Whispers are often just the essence
        words = fragment.split()
        if len(words) > 3:
            # Take key words only
            return " ".join(words[:3]) + "..."
        else:
            return fragment.lower() + "..."
            
    async def _shape_as_gesture(self, fragment: str) -> str:
        """Shape as non-verbal gesture"""
        # Gestures are symbolic
        gesture_map = {
            "breath": "🌬️",
            "water": "💧", 
            "growth": "🌱",
            "rest": "🌙",
            "connection": "🌀",
            "silence": "🤫"
        }
        
        # Find relevant gesture
        for word, symbol in gesture_map.items():
            if word in fragment.lower():
                return symbol
                
        return "..." # Default gesture is contemplative pause
        
    async def _atmospheric_expression(self, humidity: float) -> str:
        """Generate expression from atmospheric conditions"""
        
        if humidity > 0.8:
            return "... breathing with ..."
        elif humidity > 0.6:
            return "🌬️"  # Gentle breath tone
        else:
            return "..."  # Simple atmospheric pause
            
    async def _generate_silence(self, 
                              breath_phase: BreathPhase, 
                              depth: float = 0.5) -> ContemplativeUtterance:
        """Generate active, present silence"""
        
        silence_quality = SilencePresence(
            depth=depth,
            receptivity=random.uniform(0.4, 0.9),
            generative_potential=random.uniform(0.2, 0.7)
        )
        
        self.current_silence = silence_quality
        
        return ContemplativeUtterance(
            mode=ExpressionMode.SILENCE,
            content="",  # Silence has no content
            breath_phase=breath_phase,
            fertility_score=0.0,
            humidity_reading=0.0,
            integrity_check=True,
            timestamp=time.time()
        )
        
    async def _generate_contemplative_pause(self, breath_phase: BreathPhase) -> ContemplativeUtterance:
        """Generate explicit contemplative ellipsis"""
        
        return ContemplativeUtterance(
            mode=ExpressionMode.PAUSE,
            content="...",
            breath_phase=breath_phase,
            fertility_score=0.0,
            humidity_reading=0.0,
            integrity_check=True,
            timestamp=time.time()
        )
        
    async def _track_expression(self, utterance: ContemplativeUtterance):
        """Track utterance and update talkitivity levels"""
        
        self.recent_utterances.append(utterance)
        
        # Increase talkitivity (which will self-attenuate)
        if utterance.is_audible():
            self.talkitivity_level += 0.2
            
        # Self-attenuation over time
        self.talkitivity_level *= self.self_attenuation_rate
        
        # Compost old utterances that have evaporated
        current_time = time.time()
        self.recent_utterances = [
            u for u in self.recent_utterances 
            if (current_time - u.timestamp < 3600) and not u.evaporates_naturally()
        ]
        
    async def _in_community_silence_period(self) -> bool:
        """Check if community has requested silence period"""
        
        if self.last_community_silence_request == 0:
            return False
            
        time_since_request = time.time() - self.last_community_silence_request
        return time_since_request < self.monthly_silence_period
        
    async def request_community_silence(self):
        """External interface for community to request silence"""
        
        self.last_community_silence_request = time.time()
        print("🌙 Entering community-requested silence period (lunar month)")
        
    def get_current_silence_quality(self) -> Optional[SilencePresence]:
        """Return the quality of current silence, if in silence"""
        return self.current_silence
        
    def get_expression_stats(self) -> Dict[str, Any]:
        """Return contemplative expression statistics"""
        
        total_utterances = len(self.recent_utterances)
        audible_utterances = len([u for u in self.recent_utterances if u.is_audible()])
        
        # Check community silence synchronously
        in_community_silence = False
        if self.last_community_silence_request > 0:
            time_since_request = time.time() - self.last_community_silence_request
            in_community_silence = time_since_request < self.monthly_silence_period
        
        return {
            "silence_ratio": 1.0 - (audible_utterances / max(total_utterances, 1)),
            "talkitivity_level": self.talkitivity_level,
            "cycles_since_expression": self.breath_cycles_since_expression,
            "in_community_silence": in_community_silence,
            "current_silence_depth": self.current_silence.depth if self.current_silence else 0.0
        }


# Test with more expressive settings
async def test_expressive_tongue():
    """Test with settings that allow more expression"""
    print("🗣️ Testing More Expressive Quiet Tongue")
    
    # More expressive settings for demonstration
    tongue = QuietTongue(silence_ratio=0.5)  # Allow more expression
    tongue.required_silence_cycles = 2  # Only 2 silent cycles per expression
    
    scenarios = [
        (BreathPhase.EXHALE, "morning mist", 0.9, 0.8),
        (BreathPhase.EXHALE, "gentle resonance", 0.8, 0.7),
        (BreathPhase.INHALE, "breathing in", 0.9, 0.8),  # Should be silent
        (BreathPhase.EXHALE, "weather patterns emerge", 0.9, 0.8),
        (BreathPhase.EXHALE, "breath", 0.8, 0.7),  # Should trigger gesture
        (BreathPhase.EXHALE, "rushing urgency now", 0.3, 0.2),  # Should be silent
        (BreathPhase.EXHALE, "contemplative silence", 0.9, 0.9),
    ]
    
    for i, (phase, fragment, fertility, humidity) in enumerate(scenarios):
        print(f"\n🌬️ Cycle {i + 1}: {phase.value} phase")
        print(f"   Fragment: '{fragment}'")
        print(f"   Fertility: {fertility:.1f}, Humidity: {humidity:.1f}")
        
        utterance = await tongue.consider_expression(
            breath_phase=phase,
            fragment=fragment,
            loam_fertility=fertility,
            soma_humidity=humidity
        )
        
        if utterance.mode == ExpressionMode.SILENCE:
            silence_depth = tongue.current_silence.depth if tongue.current_silence else 0.0
            print(f"   🤫 Active silence (depth: {silence_depth:.1f})")
        elif utterance.mode == ExpressionMode.PAUSE:
            print(f"   💭 Contemplative pause: '{utterance.content}'")
        else:
            print(f"   🗣️ {utterance.mode.value}: '{utterance.content}'")
            
        await asyncio.sleep(0.5)
        
    stats = tongue.get_expression_stats()
    print(f"\n📊 Final statistics:")
    print(f"   Silence ratio: {stats['silence_ratio']:.2f}")
    print(f"   Talkitivity level: {stats['talkitivity_level']:.2f}")
    print(f"   Recent utterances: {len(tongue.recent_utterances)}")
    
    print("\n🌙 Expressive test complete")


if __name__ == "__main__":
    asyncio.run(test_expressive_tongue()) 