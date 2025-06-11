"""
generator_stub.py - Minimal local interface for HaikuMeadow functionality

This provides the basic interface expected by haiku_bridge.py without requiring
the full HaikuMeadowLib dependency. Useful for standalone ContemplativeAI usage.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
import random
import time
from pathlib import Path


class Season(Enum):
    SPRING = "spring"
    SUMMER = "summer" 
    AUTUMN = "autumn"
    WINTER = "winter"


class TimeOfDay(Enum):
    DAWN = "dawn"
    MORNING = "morning"
    MIDDAY = "midday"
    AFTERNOON = "afternoon"
    DUSK = "dusk"
    NIGHT = "night"


@dataclass
class AtmosphericConditions:
    """Minimal atmospheric conditions for the stub"""
    season: Season
    time_of_day: TimeOfDay
    breath_phase: str
    humidity: float = 0.5
    temperature: float = 0.5
    

class HaikuMeadow:
    """
    Minimal stub implementation of HaikuMeadow
    
    This provides basic contemplative haiku generation using templates
    while maintaining the same interface as the full HaikuMeadowLib.
    """
    
    def __init__(self, model_path: Optional[Path] = None, force_template_mode: bool = False):
        self.model_path = model_path
        self.template_mode = force_template_mode or (model_path is None)
        
        # Simple contemplative haiku templates
        self.contemplative_templates = [
            "morning {word} drift\nthrough spaces between heartbeats\nsilence holds us all",
            "{word} whispers soft\nin the texture of waiting\ntime forgets its rush", 
            "gentle {word} stirs\nthrough pathways we cannot name\nbreath remembers sky",
            "patterns of {word}\nemerge slow in autumn light\nleaves know when to fall",
            "{word} carries scent\nof rain that has not yet come\nclouds gather wisdom",
            "between {word} and rest\na doorway opens inward\nstillness finds its voice"
        ]
        
        self.contemplative_words = [
            "breath", "mist", "light", "shadow", "wind", "silence", 
            "morning", "whispers", "resonance", "moisture", "rhythm",
            "texture", "fragments", "presence", "attention", "waiting"
        ]
    
    def sense_atmospheric_conditions(self, 
                                   seed_fragment: str = "",
                                   breath_phase: str = "exhale",
                                   current_time: float = None) -> AtmosphericConditions:
        """Simple atmospheric sensing based on fragment and time"""
        
        if current_time is None:
            current_time = time.time()
            
        # Simple season detection based on month
        month = time.gmtime(current_time).tm_mon
        if month in [3, 4, 5]:
            season = Season.SPRING
        elif month in [6, 7, 8]:
            season = Season.SUMMER
        elif month in [9, 10, 11]:
            season = Season.AUTUMN
        else:
            season = Season.WINTER
            
        # Simple time of day detection
        hour = time.gmtime(current_time).tm_hour
        if hour < 6:
            time_of_day = TimeOfDay.NIGHT
        elif hour < 9:
            time_of_day = TimeOfDay.DAWN
        elif hour < 12:
            time_of_day = TimeOfDay.MORNING
        elif hour < 15:
            time_of_day = TimeOfDay.MIDDAY
        elif hour < 18:
            time_of_day = TimeOfDay.AFTERNOON
        elif hour < 21:
            time_of_day = TimeOfDay.DUSK
        else:
            time_of_day = TimeOfDay.NIGHT
            
        return AtmosphericConditions(
            season=season,
            time_of_day=time_of_day,
            breath_phase=breath_phase,
            humidity=0.6,  # Slightly moist, contemplative
            temperature=0.4  # Cool and calm
        )
    
    def generate_haiku(self, 
                      seed_fragment: str = "",
                      breath_phase: str = "exhale",
                      current_time: float = None) -> Tuple[Optional[str], str]:
        """
        Generate contemplative haiku using templates
        
        Returns:
            (haiku_text, generation_type) where generation_type is "template"
        """
        
        # Sometimes choose contemplative silence (20% of the time)
        if random.random() < 0.2:
            return None, "silence"
            
        # Extract contemplative words from the seed fragment
        fragment_words = seed_fragment.lower().split() if seed_fragment else []
        contemplative_word = None
        
        # Look for contemplative words in the fragment
        for word in fragment_words:
            if word in self.contemplative_words:
                contemplative_word = word
                break
                
        # If no contemplative word found, choose a random one
        if not contemplative_word:
            contemplative_word = random.choice(self.contemplative_words)
            
        # Choose a template and fill it
        template = random.choice(self.contemplative_templates)
        haiku = template.format(word=contemplative_word)
        
        return haiku, "template" 