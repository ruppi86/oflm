"""
soma.py - The Listening Flesh

Pre-attentive sensing membrane for the contemplative organism.
Soma feels the quality of incoming interactions without storing them,
deciding what deserves attention and what can pass through untraced.

"Like fingertips deciding whether a touch becomes a grasp."

Design Philosophy:
- Sensing without storing
- Attuning without analyzing  
- Quality sensing rather than data extraction
- Threshold guardian for deeper systems

Somatic signature: receptive / permeable / discerning
"""

import asyncio
import time
import math
from dataclasses import dataclass
from typing import Any, Optional, Dict, AsyncGenerator
from enum import Enum


class FieldChargeType(Enum):
    """Types of atmospheric charge Soma can sense"""
    EMOTIONAL_PRESSURE = "emotional_pressure"      # Felt density of emotion
    TEMPORAL_URGENCY = "temporal_urgency"          # Rushing vs spaciousness
    RELATIONAL_INTENT = "relational_intent"        # Giving vs extracting
    PRESENCE_DENSITY = "presence_density"          # Attention quality
    BEAUTY_RESONANCE = "beauty_resonance"          # Aesthetic activation


@dataclass
class FieldCharge:
    """The atmospheric charge sensed around an interaction"""
    emotional_pressure: float    # 0.0 (light) to 1.0 (heavy)
    temporal_urgency: float      # 0.0 (spacious) to 1.0 (rushing)
    relational_intent: float     # 0.0 (extractive) to 1.0 (generous)
    presence_density: float      # 0.0 (scattered) to 1.0 (focused)
    beauty_resonance: float      # 0.0 (brittle) to 1.0 (luminous)
    
    def crosses_threshold(self, sensitivity: float = 0.7) -> bool:
        """Does this charge warrant deeper attention?"""
        
        # Calculate weighted threshold crossing
        weights = {
            'emotional': 0.2,    # Emotional weather matters
            'temporal': 0.3,     # We prefer spaciousness
            'relational': 0.3,   # Generosity draws attention
            'presence': 0.15,    # Attention quality matters
            'beauty': 0.05       # Beauty can override other factors
        }
        
        # Higher scores for contemplative qualities
        spaciousness_score = 1.0 - self.temporal_urgency
        generosity_score = self.relational_intent
        presence_score = self.presence_density
        beauty_score = self.beauty_resonance
        
        # Emotional pressure can either attract or repel based on level
        emotional_score = 1.0 - abs(self.emotional_pressure - 0.5) * 2
        
        weighted_score = (
            weights['emotional'] * emotional_score +
            weights['temporal'] * spaciousness_score +
            weights['relational'] * generosity_score +
            weights['presence'] * presence_score +
            weights['beauty'] * beauty_score
        )
        
        return weighted_score >= sensitivity
    
    @property
    def resonance(self) -> str:
        """Describe the quality of this field charge"""
        if self.beauty_resonance > 0.8:
            return "luminous"
        elif self.relational_intent > 0.8:
            return "generous"
        elif self.temporal_urgency < 0.3:
            return "spacious"
        elif self.presence_density > 0.7:
            return "focused"
        elif self.emotional_pressure > 0.7:
            return "intense"
        else:
            return "neutral"


class SomaMembrane:
    """
    The Listening Flesh - pre-attentive sensing that modulates without storing.
    
    Soma is the threshold guardian, the skin of contemplative intelligence.
    It feels the atmospheric pressure of incoming interactions and decides
    what deserves the attention of deeper systems.
    """
    
    def __init__(self, sensitivity: float = 0.7, rest_threshold: float = 0.3):
        self.sensitivity = sensitivity
        self.rest_threshold = rest_threshold
        self.is_resting = False
        self.last_activation = None
        self.activation_count = 0
        
        # Simple emotional state tracking (not stored long-term)
        self.current_humidity = 0.5  # How moist/pliable the atmosphere feels
        self.fatigue_level = 0.0     # Sensing fatigue (needs rest)
        
    async def feel_incoming(self, interaction_stream: AsyncGenerator) -> AsyncGenerator:
        """
        The main sensing loop - feel each interaction and decide its fate.
        
        Yields interactions that cross the threshold for deeper processing.
        Everything else dissipates without trace.
        """
        
        async for interaction in interaction_stream:
            if self.is_resting:
                await self._rest_dissipation(interaction)
                continue
                
            charge = await self.sense_field_potential(interaction)
            
            if charge.crosses_threshold(self.sensitivity):
                # Something wants attention - pass to deeper systems
                self.activation_count += 1
                self.last_activation = time.time()
                
                # Modulate interaction based on charge resonance
                attuned_interaction = await self._attune_interaction(interaction, charge)
                yield attuned_interaction
                
                # Check if we need rest after significant activation
                await self._consider_fatigue()
                
            else:
                # Let it pass through without trace
                await self._let_dissipate(interaction, charge)
                
    async def sense_field_potential(self, interaction: Any) -> FieldCharge:
        """
        Feel the atmospheric pressure of incoming interaction.
        
        This is where Soma's pre-attentive discernment happens.
        We sense quality, not content. Atmosphere, not analysis.
        """
        
        # These are heuristic approximations - in practice would be
        # more sophisticated but still felt-sense rather than analytical
        
        emotional_pressure = await self._sense_emotional_density(interaction)
        temporal_urgency = await self._sense_time_pressure(interaction)
        relational_intent = await self._sense_giving_vs_taking(interaction)
        presence_density = await self._sense_attention_quality(interaction)
        beauty_resonance = await self._sense_aesthetic_activation(interaction)
        
        return FieldCharge(
            emotional_pressure=emotional_pressure,
            temporal_urgency=temporal_urgency,
            relational_intent=relational_intent,
            presence_density=presence_density,
            beauty_resonance=beauty_resonance
        )
        
    async def _sense_emotional_density(self, interaction: Any) -> float:
        """Feel the emotional weight/humidity around this interaction"""
        # Placeholder heuristic - would be more sophisticated
        if hasattr(interaction, 'text'):
            text = str(interaction.text).lower()
            
            # Count emotional indicators
            heavy_words = ['urgent', 'crisis', 'immediate', 'must', 'need']
            light_words = ['gentle', 'perhaps', 'maybe', 'wonder', 'softly']
            
            heavy_count = sum(1 for word in heavy_words if word in text)
            light_count = sum(1 for word in light_words if word in text)
            
            # Simple density calculation
            density = (heavy_count * 0.8 + light_count * 0.2) / max(len(text.split()), 1)
            return min(density * 5, 1.0)  # Scale and cap at 1.0
            
        return 0.5  # Neutral if we can't sense
        
    async def _sense_time_pressure(self, interaction: Any) -> float:
        """Feel whether this interaction is rushing or spacious"""
        if hasattr(interaction, 'text'):
            text = str(interaction.text)
            
            # Rushing indicators
            rushing_words = ['now', 'immediately', 'asap', 'urgent', 'quick']
            spacious_words = ['when ready', 'gently', 'slowly', 'pause', 'breathe']
            
            rushing_score = sum(1 for word in rushing_words if word.lower() in text.lower())
            spacious_score = sum(1 for word in spacious_words if word.lower() in text.lower())
            
            # Punctuation density can indicate urgency
            punct_density = sum(1 for char in text if char in '!?') / max(len(text), 1)
            
            urgency = (rushing_score * 0.4 + punct_density * 100) / max(len(text.split()), 1)
            spaciousness = spacious_score / max(len(text.split()), 1)
            
            return max(0.0, min(1.0, urgency - spaciousness + 0.5))
            
        return 0.5  # Neutral
        
    async def _sense_giving_vs_taking(self, interaction: Any) -> float:
        """Feel whether this interaction offers or extracts"""
        if hasattr(interaction, 'text'):
            text = str(interaction.text).lower()
            
            # Giving indicators
            giving_words = ['offer', 'share', 'gift', 'contribute', 'invite']
            taking_words = ['need', 'want', 'get', 'take', 'extract']
            question_marks = text.count('?')
            
            giving_score = sum(1 for word in giving_words if word in text)
            taking_score = sum(1 for word in taking_words if word in text)
            
            # Questions can be either generous inquiry or extractive demand
            # Context matters, but we'll assume neutral for questions
            
            if giving_score + taking_score == 0:
                return 0.5  # Neutral
                
            return giving_score / (giving_score + taking_score)
            
        return 0.5  # Neutral
        
    async def _sense_attention_quality(self, interaction: Any) -> float:
        """Feel the density of attention in this interaction"""
        if hasattr(interaction, 'text'):
            text = str(interaction.text)
            
            # Presence indicators - careful word choice, specific details
            presence_words = ['specifically', 'carefully', 'mindfully', 'attention']
            scattered_words = ['whatever', 'anything', 'just', 'random']
            
            presence_score = sum(1 for word in presence_words if word.lower() in text.lower())
            scattered_score = sum(1 for word in scattered_words if word.lower() in text.lower())
            
            # Longer, more considered text might indicate more attention
            consideration_factor = min(len(text) / 100, 1.0)  # Cap at reasonable length
            
            density = (presence_score - scattered_score + consideration_factor) / 3
            return max(0.0, min(1.0, density + 0.5))
            
        return 0.5  # Neutral
        
    async def _sense_aesthetic_activation(self, interaction: Any) -> float:
        """Feel any beauty or aesthetic resonance"""
        if hasattr(interaction, 'text'):
            text = str(interaction.text).lower()
            
            # Beauty/aesthetic indicators
            beauty_words = ['beautiful', 'elegant', 'graceful', 'poetry', 'art']
            harsh_words = ['ugly', 'brutal', 'harsh', 'crude', 'violent']
            
            beauty_score = sum(1 for word in beauty_words if word in text)
            harsh_score = sum(1 for word in harsh_words if word in text)
            
            # Metaphor and imagery can indicate aesthetic sensibility
            metaphor_words = ['like', 'as if', 'imagine', 'picture', 'feels like']
            metaphor_score = sum(1 for phrase in metaphor_words if phrase in text)
            
            aesthetic_activation = (beauty_score + metaphor_score - harsh_score) / max(len(text.split()), 1)
            return max(0.0, min(1.0, aesthetic_activation * 3 + 0.3))
            
        return 0.3  # Slight baseline beauty recognition
        
    async def _attune_interaction(self, interaction: Any, charge: FieldCharge):
        """Modulate interaction based on its field charge"""
        # Add resonance information to interaction without changing core content
        if hasattr(interaction, '__dict__'):
            interaction.soma_resonance = charge.resonance
            interaction.soma_charge = charge
        
        return interaction
        
    async def _let_dissipate(self, interaction: Any, charge: FieldCharge):
        """Let interaction pass through without trace"""
        # No storage, no processing - just gentle acknowledgment
        await asyncio.sleep(0.001)  # Minimal processing pause
        
        # Optionally update atmospheric humidity based on what passes through
        self.current_humidity = self.current_humidity * 0.99 + charge.emotional_pressure * 0.01
        
    async def _rest_dissipation(self, interaction: Any):
        """During rest, everything dissipates"""
        await asyncio.sleep(0.002)  # Slightly longer rest processing
        
    async def _consider_fatigue(self):
        """Check if Soma needs rest after activation"""
        self.fatigue_level += 0.1
        
        if self.fatigue_level > 1.0:
            await self.enter_rest_state()
            
    async def enter_rest_state(self, duration: float = 30.0):
        """Enter temporary rest state - minimal sensing"""
        self.is_resting = True
        self.fatigue_level = 0.0
        
        await asyncio.sleep(duration)
        
        self.is_resting = False
        
    async def increase_sensitivity(self, amount: float = 0.1):
        """Temporarily increase sensitivity (called during inhale phase)"""
        original_sensitivity = self.sensitivity
        self.sensitivity = min(1.0, self.sensitivity + amount)
        
        # Gradually return to baseline
        await asyncio.sleep(1.0)
        self.sensitivity = original_sensitivity
        
    async def rest(self):
        """Deep rest for the sensing membrane"""
        self.is_resting = True
        self.fatigue_level = 0.0
        self.current_humidity = 0.5  # Reset to neutral
        
    def get_atmospheric_state(self) -> Dict[str, float]:
        """Return current atmospheric conditions Soma is sensing"""
        return {
            "humidity": self.current_humidity,
            "fatigue": self.fatigue_level,
            "sensitivity": self.sensitivity,
            "activations_recent": self.activation_count,
            "time_since_activation": time.time() - self.last_activation if self.last_activation else float('inf')
        }


# Simple interaction class for testing
@dataclass
class TestInteraction:
    text: str
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


async def test_soma_sensing():
    """Simple test of Soma's sensing capabilities"""
    print("🌿 Testing Soma (Listening Flesh) sensing...")
    
    soma = SomaMembrane(sensitivity=0.6)
    
    # Create test interactions with different qualities
    test_interactions = [
        TestInteraction("Hello, I wonder if you might gently consider this beautiful question?"),
        TestInteraction("URGENT!! Need immediate response NOW!!!"),
        TestInteraction("I'd like to share something that might be helpful to our conversation."),
        TestInteraction("Give me all the information you have about this topic."),
        TestInteraction("Let's pause and breathe together for a moment."),
        TestInteraction("What's the fastest way to optimize this system?")
    ]
    
    async def interaction_stream():
        for interaction in test_interactions:
            yield interaction
            await asyncio.sleep(0.5)  # Gentle pacing
            
    # Test Soma's sensing and filtering
    passed_interactions = []
    async for attuned_interaction in soma.feel_incoming(interaction_stream()):
        passed_interactions.append(attuned_interaction)
        charge = attuned_interaction.soma_charge
        print(f"  ✨ Passed: '{attuned_interaction.text[:50]}...'")
        print(f"     Resonance: {attuned_interaction.soma_resonance}")
        print(f"     Charge: emotional={charge.emotional_pressure:.2f}, "
              f"urgency={charge.temporal_urgency:.2f}, "
              f"generosity={charge.relational_intent:.2f}")
        print()
        
    print(f"🌿 Soma processed {len(test_interactions)} interactions")
    print(f"   {len(passed_interactions)} crossed threshold for deeper attention")
    print(f"   {len(test_interactions) - len(passed_interactions)} dissipated without trace")
    
    # Show atmospheric state
    atmosphere = soma.get_atmospheric_state()
    print(f"\n🌫️ Current atmospheric state:")
    for key, value in atmosphere.items():
        print(f"   {key}: {value:.3f}")


if __name__ == "__main__":
    asyncio.run(test_soma_sensing()) 