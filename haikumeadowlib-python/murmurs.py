#!/usr/bin/env python3
"""
murmurs.py - Contemplative Expression for HaikuMeadowLib

A gentle output system that handles contemplative expression following
QuietTongue principles adapted for the haiku meadow:
- Tystnadsmajoritet (7/8ths silence) - most responses are quiet
- Atmospheric murmurs rather than direct answers
- Breath-synchronized expression timing
- Graceful degradation to silence when uninspired
- Multiple expression modes based on atmospheric conditions

Inspired by the QuietTongue architecture from the Contemplative Organism.

Somatic signature: whispered / atmospheric / mostly-silent
"""

import time
import random
import asyncio
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

class MurmurType(Enum):
    """Types of contemplative expressions"""
    HAIKU = "haiku"           # Full three-line haiku
    PETAL = "petal"           # Single poetic fragment
    WHISPER = "whisper"       # Gentle atmospheric phrase
    ELLIPSIS = "ellipsis"     # Contemplative pause markers
    SILENCE = "silence"       # Pure absence of speech
    BREATH = "breath"         # Breathing sound/rhythm
    RESONANCE = "resonance"   # Echo of input fragment

class AtmosphericMood(Enum):
    """Atmospheric moods affecting expression style"""
    CRISP = "crisp"           # Clear, bright expression
    MISTY = "misty"           # Soft, humid expression  
    DEEP = "deep"             # Profound, still expression
    FLOWING = "flowing"       # Gentle movement expression
    BARE = "bare"             # Minimal, essential expression

@dataclass
class Murmur:
    """A single contemplative expression from the meadow"""
    
    content: str
    murmur_type: MurmurType
    atmospheric_mood: AtmosphericMood
    created_at: float
    breath_phase: str = "exhale"
    silence_ratio: float = 0.875  # 7/8ths silence (QuietTongue principle)
    
    # Expression metadata
    contemplative_depth: float = 0.5  # How deep/contemplative this expression is
    seasonal_resonance: float = 0.5   # How well it resonates with current season
    intended_duration: float = 3.0    # How long this should be present
    
    def is_audible(self) -> bool:
        """Check if this murmur should be expressed (vs silent)"""
        return self.murmur_type != MurmurType.SILENCE
    
    def is_ephemeral(self) -> bool:
        """Check if this expression should fade quickly"""
        return self.murmur_type in [MurmurType.ELLIPSIS, MurmurType.BREATH, MurmurType.WHISPER]
    
    def should_fade(self) -> bool:
        """Check if enough time has passed for this expression to fade"""
        return time.time() - self.created_at > self.intended_duration

class MeadowVoice:
    """
    Contemplative voice system for the haiku meadow
    
    Implements QuietTongue principles with atmospheric sensitivity:
    - Maintains tystnadsmajoritet (silence majority)
    - Chooses expression mode based on atmospheric conditions
    - Gracefully degrades to silence under pressure
    - Coordinates with breath phases for timing
    """
    
    def __init__(self):
        
        # QuietTongue silence configuration
        self.target_silence_ratio = 0.875  # 7/8ths silence
        self.recent_expressions = []  # Track recent output to maintain ratio
        self.expression_window = 300.0  # 5 minute window for ratio calculation
        
        # Expression timing
        self.last_expression_time = 0.0
        self.min_expression_interval = 30.0  # Minimum seconds between expressions
        self.current_atmospheric_pressure = 0.3
        
        # Atmospheric sensitivity
        self.mood_thresholds = {
            AtmosphericMood.CRISP: {"humidity": (0.0, 0.3), "temperature": (0.6, 1.0)},
            AtmosphericMood.MISTY: {"humidity": (0.7, 1.0), "temperature": (0.4, 0.7)},
            AtmosphericMood.DEEP: {"humidity": (0.4, 0.7), "temperature": (0.0, 0.4)},
            AtmosphericMood.FLOWING: {"humidity": (0.5, 0.8), "temperature": (0.5, 0.8)},
            AtmosphericMood.BARE: {"humidity": (0.0, 0.4), "temperature": (0.0, 0.5)}
        }
        
        print("🤫 MeadowVoice initialized (tystnadsmajoritet: 87.5%)")
    
    def sense_atmospheric_mood(self, conditions: Dict[str, Any]) -> AtmosphericMood:
        """Sense current atmospheric mood from conditions"""
        
        humidity = conditions.get("humidity", 0.5)
        temperature = conditions.get("temperature", 0.5)
        pressure = conditions.get("pressure", 0.3)
        
        # Find best matching mood
        best_mood = AtmosphericMood.FLOWING  # Default
        best_score = 0.0
        
        for mood, thresholds in self.mood_thresholds.items():
            score = 0.0
            
            # Check humidity range
            h_min, h_max = thresholds["humidity"]
            if h_min <= humidity <= h_max:
                score += 0.5
            else:
                # Penalty for being outside range
                score -= min(abs(humidity - h_min), abs(humidity - h_max)) * 0.5
                
            # Check temperature range
            t_min, t_max = thresholds["temperature"]
            if t_min <= temperature <= t_max:
                score += 0.5
            else:
                score -= min(abs(temperature - t_min), abs(temperature - t_max)) * 0.5
            
            # Pressure influences mood selection
            if mood == AtmosphericMood.BARE and pressure > 0.7:
                score += 0.3  # High pressure favors bare expression
            elif mood == AtmosphericMood.DEEP and pressure < 0.2:
                score += 0.3  # Low pressure favors deep expression
                
            if score > best_score:
                best_score = score
                best_mood = mood
                
        return best_mood
    
    def should_express(self, 
                      conditions: Dict[str, Any],
                      force_expression: bool = False) -> bool:
        """Decide whether to express or maintain silence"""
        
        current_time = time.time()
        
        # Always respect minimum interval unless forced
        if not force_expression:
            if current_time - self.last_expression_time < self.min_expression_interval:
                return False
        
        # Check breath phase - only express during exhale/rest
        breath_phase = conditions.get("breath_phase", "rest")
        if breath_phase not in ["exhale", "rest"]:
            return False
            
        # Atmospheric pressure check - high pressure encourages silence
        pressure = conditions.get("pressure", 0.3)
        self.current_atmospheric_pressure = pressure
        
        if pressure > 0.7:  # Very high pressure - likely silence
            return False
        elif pressure > 0.5:  # Moderate pressure - reduced chance
            if random.random() < pressure:  # Higher pressure = more likely silence
                return False
        
        # Check silence ratio compliance
        if not force_expression:
            current_silence_ratio = self._calculate_current_silence_ratio()
            if current_silence_ratio < self.target_silence_ratio:
                # Below target silence - more likely to stay quiet
                return random.random() < 0.3  # 30% chance to express
        
        # Community pressure consideration
        community_pressure = conditions.get("community_pressure", 0.3)
        if community_pressure > 0.6:
            return False  # High community activity encourages individual silence
            
        # Seasonal and temporal factors
        season = conditions.get("season", "spring")
        time_of_day = conditions.get("time_of_day", "day")
        
        # Night and winter encourage more silence
        if time_of_day == "night":
            if random.random() < 0.7:  # 70% chance of silence at night
                return False
        
        if season == "winter":
            if random.random() < 0.6:  # 60% chance of silence in winter
                return False
        
        # If we've made it this far, expression is allowed
        return True
    
    def create_murmur(self, 
                     content: Optional[str],
                     conditions: Dict[str, Any],
                     preferred_type: Optional[MurmurType] = None) -> Murmur:
        """Create a contemplative murmur based on content and conditions"""
        
        current_time = time.time()
        atmospheric_mood = self.sense_atmospheric_mood(conditions)
        breath_phase = conditions.get("breath_phase", "exhale")
        
        # Determine murmur type
        if preferred_type:
            murmur_type = preferred_type
        elif not content:
            murmur_type = MurmurType.SILENCE
        else:
            murmur_type = self._sense_content_type(content, atmospheric_mood)
        
        # Transform content based on atmospheric mood
        transformed_content = self._transform_content_for_mood(
            content or "", murmur_type, atmospheric_mood
        )
        
        # Calculate contemplative depth
        contemplative_depth = self._assess_contemplative_depth(
            transformed_content, conditions
        )
        
        # Calculate intended duration based on type and depth
        duration_map = {
            MurmurType.HAIKU: 10.0,
            MurmurType.PETAL: 5.0,
            MurmurType.WHISPER: 3.0,
            MurmurType.ELLIPSIS: 2.0,
            MurmurType.BREATH: 1.0,
            MurmurType.RESONANCE: 4.0,
            MurmurType.SILENCE: 0.0
        }
        
        base_duration = duration_map.get(murmur_type, 3.0)
        intended_duration = base_duration * (0.5 + contemplative_depth * 0.5)
        
        murmur = Murmur(
            content=transformed_content,
            murmur_type=murmur_type,
            atmospheric_mood=atmospheric_mood,
            created_at=current_time,
            breath_phase=breath_phase,
            contemplative_depth=contemplative_depth,
            seasonal_resonance=self._assess_seasonal_resonance(transformed_content, conditions),
            intended_duration=intended_duration
        )
        
        # Record expression for silence ratio tracking
        self._record_expression(murmur)
        
        return murmur
    
    def express_contemplatively(self, 
                              content: Optional[str],
                              conditions: Dict[str, Any],
                              force_expression: bool = False) -> Optional[Murmur]:
        """
        Main interface for contemplative expression
        
        Returns a Murmur if expression occurs, None for silence
        """
        
        # First check if we should express at all
        if not self.should_express(conditions, force_expression):
            # Return a silence murmur to track the non-expression
            return self.create_murmur(None, conditions, MurmurType.SILENCE)
        
        # Create and return the murmur
        murmur = self.create_murmur(content, conditions)
        self.last_expression_time = time.time()
        
        return murmur
    
    def _sense_content_type(self, content: str, mood: AtmosphericMood) -> MurmurType:
        """Sense what type of murmur the content should be"""
        
        if not content.strip():
            return MurmurType.SILENCE
            
        content_lower = content.lower()
        
        # Check for haiku structure (line breaks or /)
        if "\n" in content and content.count("\n") == 2:
            return MurmurType.HAIKU
        elif "/" in content and content.count("/") == 2:
            return MurmurType.HAIKU
            
        # Check for ellipsis or contemplative pauses
        if "..." in content or content.strip() in [".", "..", "..."]:
            return MurmurType.ELLIPSIS
            
        # Check for breathing sounds
        if any(breath_word in content_lower for breath_word in 
               ["breath", "breathing", "inhale", "exhale", "sigh"]):
            return MurmurType.BREATH
            
        # Length-based classification
        word_count = len(content.split())
        
        if word_count <= 3:
            return MurmurType.PETAL
        elif word_count <= 8:
            return MurmurType.WHISPER
        elif word_count <= 15:
            # Could be haiku without line breaks
            return MurmurType.HAIKU
        else:
            # Longer content becomes whisper in contemplative context
            return MurmurType.WHISPER
    
    def _transform_content_for_mood(self, 
                                   content: str,
                                   murmur_type: MurmurType,
                                   mood: AtmosphericMood) -> str:
        """Transform content based on atmospheric mood"""
        
        if murmur_type == MurmurType.SILENCE:
            return ""
            
        if not content.strip():
            # Generate mood-appropriate silence expressions
            silence_expressions = {
                AtmosphericMood.CRISP: [".", "…", "clear silence"],
                AtmosphericMood.MISTY: ["...", "mist drifts", "soft quiet"],
                AtmosphericMood.DEEP: ["....", "depth", "stillness"],
                AtmosphericMood.FLOWING: ["~", "gentle flow", "breath moves"],
                AtmosphericMood.BARE: ["", ".", "bare"]
            }
            
            expressions = silence_expressions.get(mood, ["..."])
            return random.choice(expressions)
        
        # Apply mood-specific transformations
        if mood == AtmosphericMood.CRISP:
            # Crisp: Clean, clear language
            content = content.strip()
            
        elif mood == AtmosphericMood.MISTY:
            # Misty: Add soft, flowing elements
            if not any(word in content.lower() for word in ["mist", "dew", "soft", "gentle"]):
                content = f"gentle {content}"
                
        elif mood == AtmosphericMood.DEEP:
            # Deep: Add contemplative depth
            if len(content.split()) > 5:
                # Compress longer content to essence
                words = content.split()
                essential_words = words[:3] + ["..."] + words[-2:]
                content = " ".join(essential_words)
                
        elif mood == AtmosphericMood.FLOWING:
            # Flowing: Add movement words
            if not any(word in content.lower() for word in ["flow", "drift", "move", "through"]):
                content = content.replace(" ", " gently ")
                
        elif mood == AtmosphericMood.BARE:
            # Bare: Reduce to essential elements
            words = content.split()
            if len(words) > 3:
                content = " ".join(words[:3])
        
        return content
    
    def _assess_contemplative_depth(self, 
                                   content: str,
                                   conditions: Dict[str, Any]) -> float:
        """Assess the contemplative depth of content"""
        
        if not content:
            return 1.0  # Silence has maximum contemplative depth
            
        depth = 0.0
        content_lower = content.lower()
        
        # Contemplative words increase depth
        contemplative_indicators = [
            "breath", "silence", "stillness", "quiet", "gentle", "soft",
            "pause", "wait", "listen", "empty", "space", "moment",
            "mist", "dew", "shadow", "light", "depth", "essence"
        ]
        
        for indicator in contemplative_indicators:
            if indicator in content_lower:
                depth += 0.2
                
        # Punctuation patterns
        if "..." in content:
            depth += 0.3
        if content.count(".") > 1:
            depth += 0.1
            
        # Brevity increases contemplative depth
        word_count = len(content.split())
        if word_count <= 3:
            depth += 0.4
        elif word_count <= 6:
            depth += 0.2
            
        # Atmospheric conditions influence depth perception
        humidity = conditions.get("humidity", 0.5)
        depth += humidity * 0.2  # Higher humidity = perceived deeper
        
        return min(depth, 1.0)
    
    def _assess_seasonal_resonance(self, 
                                  content: str,
                                  conditions: Dict[str, Any]) -> float:
        """Assess how well content resonates with current season"""
        
        if not content:
            return 0.8  # Silence resonates with all seasons
            
        season = conditions.get("season", "spring")
        content_lower = content.lower()
        
        seasonal_words = {
            "spring": ["bloom", "green", "fresh", "new", "growth", "rain", "dawn"],
            "summer": ["warm", "bright", "full", "sun", "heat", "flower", "abundance"],
            "autumn": ["fall", "red", "gold", "harvest", "wind", "leaf", "fade"],
            "winter": ["cold", "snow", "frost", "bare", "ice", "still", "deep"]
        }
        
        season_indicators = seasonal_words.get(season, [])
        resonance = sum(0.2 for word in season_indicators if word in content_lower)
        
        # Time of day resonance
        time_of_day = conditions.get("time_of_day", "day")
        time_words = {
            "dawn": ["morning", "first", "early", "rise", "new"],
            "day": ["bright", "clear", "open", "full", "high"],
            "dusk": ["evening", "soft", "fade", "golden", "gentle"],
            "night": ["dark", "deep", "still", "quiet", "rest"]
        }
        
        time_indicators = time_words.get(time_of_day, [])
        resonance += sum(0.1 for word in time_indicators if word in content_lower)
        
        return min(resonance, 1.0)
    
    def _record_expression(self, murmur: Murmur):
        """Record expression for silence ratio tracking"""
        
        current_time = time.time()
        
        # Add to recent expressions
        self.recent_expressions.append({
            "time": current_time,
            "was_silent": murmur.murmur_type == MurmurType.SILENCE,
            "type": murmur.murmur_type.value
        })
        
        # Clean old expressions outside window
        cutoff_time = current_time - self.expression_window
        self.recent_expressions = [
            expr for expr in self.recent_expressions 
            if expr["time"] > cutoff_time
        ]
    
    def _calculate_current_silence_ratio(self) -> float:
        """Calculate current silence ratio over recent window"""
        
        if not self.recent_expressions:
            return 1.0  # No expressions = perfect silence
            
        silent_count = sum(1 for expr in self.recent_expressions if expr["was_silent"])
        total_count = len(self.recent_expressions)
        
        return silent_count / total_count
    
    def get_expression_stats(self) -> Dict[str, Any]:
        """Get current expression statistics"""
        
        current_silence_ratio = self._calculate_current_silence_ratio()
        
        type_counts = {}
        for expr in self.recent_expressions:
            expr_type = expr["type"]
            type_counts[expr_type] = type_counts.get(expr_type, 0) + 1
            
        return {
            "silence_ratio": current_silence_ratio,
            "target_silence_ratio": self.target_silence_ratio,
            "recent_expressions": len(self.recent_expressions),
            "expression_types": type_counts,
            "last_expression_age": time.time() - self.last_expression_time,
            "atmospheric_pressure": self.current_atmospheric_pressure
        }

# Utilities for formatting and display
class MurmurFormatter:
    """Formats murmurs for different output contexts"""
    
    @staticmethod
    def format_for_console(murmur: Murmur) -> str:
        """Format murmur for console display"""
        
        if murmur.murmur_type == MurmurType.SILENCE:
            return ""  # Silence is not displayed
            
        # Add atmospheric mood indicator
        mood_symbols = {
            AtmosphericMood.CRISP: "❄️",
            AtmosphericMood.MISTY: "🌫️",
            AtmosphericMood.DEEP: "🌊", 
            AtmosphericMood.FLOWING: "🌬️",
            AtmosphericMood.BARE: "🌙"
        }
        
        symbol = mood_symbols.get(murmur.atmospheric_mood, "🌸")
        
        if murmur.murmur_type == MurmurType.HAIKU:
            # Format haiku with proper line breaks
            lines = murmur.content.replace(" / ", "\n").split("\n")
            formatted = f"{symbol} "
            for line in lines:
                formatted += f"   {line.strip()}\n"
            return formatted
        else:
            return f"{symbol} {murmur.content}"
    
    @staticmethod  
    def format_for_bridge(murmur: Murmur) -> Dict[str, Any]:
        """Format murmur for haiku bridge transmission"""
        
        return {
            "haiku": murmur.content if murmur.is_audible() else None,
            "type": murmur.murmur_type.value,
            "mood": murmur.atmospheric_mood.value,
            "contemplative_depth": murmur.contemplative_depth,
            "seasonal_resonance": murmur.seasonal_resonance,
            "breath_phase": murmur.breath_phase
        }

# Testing and demonstration
async def test_meadow_voice():
    """Test the contemplative voice system"""
    
    print("🤫 Testing MeadowVoice system")
    
    voice = MeadowVoice()
    
    # Test atmospheric conditions
    test_conditions = [
        {
            "season": "spring",
            "time_of_day": "dawn",
            "breath_phase": "exhale",
            "humidity": 0.7,
            "pressure": 0.2,
            "temperature": 0.6
        },
        {
            "season": "winter", 
            "time_of_day": "night",
            "breath_phase": "rest",
            "humidity": 0.3,
            "pressure": 0.8,
            "temperature": 0.2
        },
        {
            "season": "autumn",
            "time_of_day": "dusk", 
            "breath_phase": "exhale",
            "humidity": 0.5,
            "pressure": 0.3,
            "temperature": 0.4
        }
    ]
    
    test_contents = [
        "morning mist rises through branches",
        "gentle breath gathering between heartbeats",
        "...",
        "snow falls on bare stone",
        "patterns emerging in twilight spaces"
    ]
    
    print("\n🌸 Testing contemplative expression:")
    
    for i, conditions in enumerate(test_conditions):
        print(f"\n   Condition {i+1}: {conditions['season']} {conditions['time_of_day']}")
        print(f"   Humidity: {conditions['humidity']:.1f}, Pressure: {conditions['pressure']:.1f}")
        
        for content in test_contents:
            murmur = voice.express_contemplatively(content, conditions)
            
            if murmur and murmur.is_audible():
                formatted = MurmurFormatter.format_for_console(murmur)
                print(f"   Input: '{content}' → {murmur.murmur_type.value}")
                print(f"   Output: {formatted.strip()}")
            else:
                print(f"   Input: '{content}' → [contemplative silence]")
    
    # Test silence ratio maintenance
    print(f"\n📊 Testing silence ratio maintenance...")
    
    silence_count = 0
    expression_count = 0
    
    # Generate many expressions to test ratio
    for i in range(50):
        test_content = f"test expression {i}"
        test_conditions = {
            "season": "spring",
            "breath_phase": "exhale", 
            "humidity": 0.5,
            "pressure": 0.3
        }
        
        murmur = voice.express_contemplatively(test_content, test_conditions)
        
        if murmur:
            if murmur.murmur_type == MurmurType.SILENCE:
                silence_count += 1
            else:
                expression_count += 1
        
        # Small delay to avoid timing restrictions
        await asyncio.sleep(0.01)
    
    total = silence_count + expression_count
    actual_silence_ratio = silence_count / total if total > 0 else 1.0
    
    print(f"   Generated {total} responses: {silence_count} silent, {expression_count} expressed")
    print(f"   Silence ratio: {actual_silence_ratio:.1%} (target: {voice.target_silence_ratio:.1%})")
    
    # Get final stats
    stats = voice.get_expression_stats()
    print(f"\n📈 Final expression statistics:")
    print(f"   Current silence ratio: {stats['silence_ratio']:.1%}")
    print(f"   Expression types: {stats['expression_types']}")
    
    print("\n🌙 Voice test complete")

if __name__ == "__main__":
    asyncio.run(test_meadow_voice())
