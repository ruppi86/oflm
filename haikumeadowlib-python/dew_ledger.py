"""
dew_ledger.py - Seasonal Resonance Memory

A contemplative alternative to RLHF that collects, evaporates, and distills
community resonance with femto-poet utterances. Based on o3's Letter V design.

Philosophy:
- Dew as living memory, not static data
- Evaporation as graceful forgetting
- Solstice distillation for seasonal re-tuning
- Resonance over optimization

Somatic signature: ephemeral / cyclical / compost-ready
"""

import json
import time
import random
import math
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum


class Season(Enum):
    """Seasonal awareness for dew context"""
    SPRING = "spring"
    SUMMER = "summer" 
    AUTUMN = "autumn"
    WINTER = "winter"


@dataclass
class DewDrop:
    """
    A single drop of communal memory - one haiku or silence event.
    Based on o3's design with extensions for atmospheric awareness.
    """
    fragment: str                    # The seed fragment offered
    utterance: str                   # What the poet responded (or "..." for silence)
    season_vec: List[float]          # 8-dim atmospheric conditioning
    resonance: float                 # 0-1 community-felt moisture
    timestamp: float                 # When this dew formed
    chosen: bool = False             # Marked during solstice distillation
    
    # Extended atmospheric context
    season: Optional[Season] = None
    humidity: Optional[float] = None  # If available from sensors
    temperature: Optional[float] = None
    generation_type: str = "unknown" # "neural", "template", "silence"
    breath_phase: str = "exhale"     # Which phase this occurred in
    
    def age_days(self) -> float:
        """How many days old is this drop?"""
        return (time.time() - self.timestamp) / 86400
        
    def is_silence(self) -> bool:
        """Was this a contemplative silence?"""
        return self.utterance.strip() in ["", "...", "…"]
        
    def moisture_quality(self) -> float:
        """Combined quality measure considering resonance and atmospheric fit"""
        base_quality = self.resonance
        
        # Boost for successful neural generation
        if self.generation_type == "neural":
            base_quality *= 1.1
            
        # Seasonal harmony bonus
        if self.season and self.season_vec:
            seasonal_coherence = self._seasonal_coherence()
            base_quality *= (0.8 + 0.4 * seasonal_coherence)
            
        return min(1.0, base_quality)
        
    def _seasonal_coherence(self) -> float:
        """How well does the season_vec match the recorded season?"""
        if not self.season or not self.season_vec or len(self.season_vec) < 4:
            return 0.5
            
        # Simple coherence check - seasons map to first 4 dims
        season_expectations = {
            Season.SPRING: [0.7, 0.2, 0.1, 0.0],
            Season.SUMMER: [0.1, 0.8, 0.1, 0.0],
            Season.AUTUMN: [0.0, 0.2, 0.7, 0.1],
            Season.WINTER: [0.0, 0.0, 0.2, 0.8]
        }
        
        expected = season_expectations.get(self.season, [0.25, 0.25, 0.25, 0.25])
        actual = self.season_vec[:4]
        
        # Calculate similarity (inverse of euclidean distance)
        distance = sum((e - a) ** 2 for e, a in zip(expected, actual)) ** 0.5
        return max(0.0, 1.0 - distance)


class DewLedger:
    """
    The dew-ledger: a living memory that collects, evaporates, and distills
    seasonal resonance for contemplative AI organisms.
    """
    
    def __init__(self, 
                 ledger_path: Path = Path("dew_ledger.jsonl"),
                 half_life_days: float = 75.0,
                 max_entries: int = 10000):
        
        self.ledger_path = Path(ledger_path)
        self.half_life_days = half_life_days
        self.max_entries = max_entries
        
        # In-memory cache for recent entries
        self._cache: List[DewDrop] = []
        self._cache_dirty = False
        
        # Load existing ledger
        self._load_from_disk()
        
    def add_drop(self, 
                 fragment: str,
                 utterance: str, 
                 season_vec: List[float],
                 resonance: float = 0.5,
                 **kwargs) -> DewDrop:
        """
        Add a new dew drop to the ledger.
        
        Args:
            fragment: The seed fragment that was offered
            utterance: The poet's response (or "..." for silence)
            season_vec: 8-dimensional atmospheric conditioning
            resonance: Community-felt quality (0-1)
            **kwargs: Additional atmospheric context
        """
        
        drop = DewDrop(
            fragment=fragment,
            utterance=utterance,
            season_vec=season_vec.copy() if season_vec else [],
            resonance=resonance,
            timestamp=time.time(),
            **kwargs
        )
        
        self._cache.append(drop)
        self._cache_dirty = True
        
        # Periodic maintenance
        if len(self._cache) > self.max_entries:
            self._maintain_ledger()
            
        return drop
    
    def add_silence(self, 
                    fragment: str, 
                    season_vec: List[float],
                    reason: str = "contemplative_choice",
                    **kwargs) -> DewDrop:
        """Add a contemplative silence event"""
        
        return self.add_drop(
            fragment=fragment,
            utterance="...",
            season_vec=season_vec,
            resonance=0.6,  # Silence has its own value
            generation_type="silence",
            silence_reason=reason,
            **kwargs
        )
    
    def evaporate(self, force: bool = False) -> int:
        """
        Evaporate old entries based on half-life decay.
        Returns number of entries evaporated.
        """
        
        if not force and random.random() > 0.1:  # Only evaporate 10% of the time
            return 0
            
        current_time = time.time()
        evaporated_count = 0
        
        new_cache = []
        for drop in self._cache:
            age_days = drop.age_days()
            
            # Survival probability based on half-life decay
            survival_prob = 2 ** (-age_days / self.half_life_days)
            
            # Chosen entries get longevity bonus
            if drop.chosen:
                survival_prob = min(1.0, survival_prob * 3.0)
                
            # High-resonance entries also resist evaporation
            if drop.moisture_quality() > 0.8:
                survival_prob = min(1.0, survival_prob * 1.5)
            
            if random.random() < survival_prob:
                new_cache.append(drop)
            else:
                evaporated_count += 1
                
        self._cache = new_cache
        self._cache_dirty = True
        
        return evaporated_count
    
    def solstice_distillation(self, 
                             max_chosen: int = 64,
                             silence_ratio: float = 0.2) -> List[DewDrop]:
        """
        Perform solstice distillation - select the most resonant entries
        for seasonal re-tuning. Maintains balance between haikus and silences.
        
        Args:
            max_chosen: Maximum entries to select
            silence_ratio: Portion of selections that should be silence
        """
        
        # Separate haikus from silences
        haikus = [d for d in self._cache if not d.is_silence()]
        silences = [d for d in self._cache if d.is_silence()]
        
        # Sort by moisture quality
        haikus.sort(key=lambda d: d.moisture_quality(), reverse=True)
        silences.sort(key=lambda d: d.moisture_quality(), reverse=True)
        
        # Calculate distribution
        max_silences = int(max_chosen * silence_ratio)
        max_haikus = max_chosen - max_silences
        
        # Select top entries
        chosen_haikus = haikus[:max_haikus]
        chosen_silences = silences[:max_silences]
        
        # Mark as chosen
        chosen_drops = chosen_haikus + chosen_silences
        for drop in chosen_drops:
            drop.chosen = True
            
        self._cache_dirty = True
        
        return chosen_drops
    
    def get_recent_drops(self, 
                        limit: int = 100,
                        only_chosen: bool = False) -> List[DewDrop]:
        """Get recent dew drops, optionally filtering to chosen only"""
        
        drops = [d for d in self._cache if not only_chosen or d.chosen]
        drops.sort(key=lambda d: d.timestamp, reverse=True)
        return drops[:limit]
    
    def resonance_statistics(self) -> Dict[str, Any]:
        """Get statistics about current ledger state"""
        
        if not self._cache:
            return {"total_drops": 0}
            
        total = len(self._cache)
        chosen = sum(1 for d in self._cache if d.chosen)
        silences = sum(1 for d in self._cache if d.is_silence())
        
        resonances = [d.resonance for d in self._cache]
        qualities = [d.moisture_quality() for d in self._cache]
        
        return {
            "total_drops": total,
            "chosen_drops": chosen,
            "silence_ratio": silences / total if total > 0 else 0,
            "avg_resonance": sum(resonances) / len(resonances) if resonances else 0,
            "avg_quality": sum(qualities) / len(qualities) if qualities else 0,
            "oldest_age_days": max(d.age_days() for d in self._cache) if self._cache else 0,
            "generation_types": self._count_generation_types()
        }
    
    def _count_generation_types(self) -> Dict[str, int]:
        """Count entries by generation type"""
        counts = {}
        for drop in self._cache:
            counts[drop.generation_type] = counts.get(drop.generation_type, 0) + 1
        return counts
    
    def save_to_disk(self) -> bool:
        """Save current cache to disk as JSONL"""
        
        if not self._cache_dirty:
            return False
            
        try:
            with open(self.ledger_path, 'w', encoding='utf-8') as f:
                for drop in self._cache:
                    json.dump(asdict(drop), f, ensure_ascii=False)
                    f.write('\n')
                    
            self._cache_dirty = False
            return True
            
        except Exception as e:
            print(f"⚠️  Error saving dew ledger: {e}")
            return False
    
    def _load_from_disk(self):
        """Load existing ledger from disk"""
        
        if not self.ledger_path.exists():
            return
            
        try:
            with open(self.ledger_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        # Convert season string back to enum if present
                        if 'season' in data and data['season']:
                            data['season'] = Season(data['season'])
                        drop = DewDrop(**data)
                        self._cache.append(drop)
                        
        except Exception as e:
            print(f"⚠️  Error loading dew ledger: {e}")
            self._cache = []
            
    def _maintain_ledger(self):
        """Periodic maintenance - evaporate and save"""
        
        self.evaporate(force=True)
        self.save_to_disk()
        
        # Keep cache size reasonable
        if len(self._cache) > self.max_entries:
            # Keep most recent entries
            self._cache.sort(key=lambda d: d.timestamp, reverse=True)
            self._cache = self._cache[:self.max_entries]
            self._cache_dirty = True


def determine_season(timestamp: Optional[float] = None) -> Season:
    """Determine current season based on timestamp (Northern Hemisphere)"""
    
    if timestamp is None:
        timestamp = time.time()
        
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    month = dt.month
    
    if month in [12, 1, 2]:
        return Season.WINTER
    elif month in [3, 4, 5]:
        return Season.SPRING
    elif month in [6, 7, 8]:
        return Season.SUMMER
    else:  # [9, 10, 11]
        return Season.AUTUMN


def create_atmospheric_vector(season: Season = None,
                            humidity: float = None,
                            temperature: float = None,
                            time_of_day: str = "unknown") -> List[float]:
    """Create 8-dimensional atmospheric conditioning vector"""
    
    if season is None:
        season = determine_season()
        
    # Base seasonal encoding
    seasonal_base = {
        Season.SPRING: [0.7, 0.2, 0.1, 0.0],
        Season.SUMMER: [0.1, 0.8, 0.1, 0.0], 
        Season.AUTUMN: [0.0, 0.2, 0.7, 0.1],
        Season.WINTER: [0.0, 0.0, 0.2, 0.8]
    }
    
    vec = seasonal_base[season].copy()
    
    # Add atmospheric conditions
    vec.append(humidity if humidity is not None else 0.5)
    vec.append((temperature + 20) / 40 if temperature is not None else 0.5)  # Normalized
    
    # Time of day encoding
    if time_of_day == "dawn":
        vec.extend([0.8, 0.2])
    elif time_of_day == "dusk":
        vec.extend([0.2, 0.8])
    elif time_of_day == "night":
        vec.extend([0.1, 0.1])
    else:  # day or unknown
        vec.extend([0.5, 0.5])
        
    return vec


# Example usage and testing functions

def test_dew_ledger():
    """Test basic dew ledger functionality"""
    
    print("🌸 Testing Dew Ledger - Seasonal Resonance Memory")
    
    # Create temporary ledger
    ledger = DewLedger(Path("test_dew.jsonl"), half_life_days=30)
    
    # Add some test drops
    test_fragments = [
        ("morning mist gathering", "dew collects\non spider's patient web\nsilence holds", 0.8),
        ("urgent meeting now", "...", 0.9),  # Good silence choice
        ("breath between heartbeats", "stillness finds\nits own rhythm here\nclock forgets", 0.7),
        ("deadline pressure", "...", 0.8),  # Another good silence
    ]
    
    for fragment, utterance, resonance in test_fragments:
        season_vec = create_atmospheric_vector(Season.AUTUMN)
        drop = ledger.add_drop(
            fragment=fragment,
            utterance=utterance,
            season_vec=season_vec,
            resonance=resonance,
            season=Season.AUTUMN,
            generation_type="neural" if utterance != "..." else "silence"
        )
        print(f"   Added: '{fragment}' → quality={drop.moisture_quality():.2f}")
    
    # Test statistics
    stats = ledger.resonance_statistics()
    print(f"\n📊 Ledger statistics:")
    print(f"   Total drops: {stats['total_drops']}")
    print(f"   Silence ratio: {stats['silence_ratio']:.1%}")
    print(f"   Average quality: {stats['avg_quality']:.2f}")
    
    # Test solstice distillation
    chosen = ledger.solstice_distillation(max_chosen=3)
    print(f"\n🌙 Solstice distillation selected {len(chosen)} drops:")
    for drop in chosen:
        content = drop.utterance if not drop.is_silence() else "[silence]"
        print(f"   Quality {drop.moisture_quality():.2f}: {content}")
    
    # Test evaporation (forced)
    evaporated = ledger.evaporate(force=True)
    print(f"\n🌫️ Evaporation: {evaporated} drops faded")
    
    # Save and cleanup
    ledger.save_to_disk()
    Path("test_dew.jsonl").unlink(missing_ok=True)
    
    print("🌿 Dew ledger test complete")


if __name__ == "__main__":
    test_dew_ledger() 