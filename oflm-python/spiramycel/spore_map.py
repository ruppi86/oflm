"""
Spiramycel Spore Map

JSONL-based collection system for mycelial network repair echoes.
Each spore echo represents the effectiveness of glyph-based repairs,
creating a living memory substrate with 75-day evaporation cycles.

Part of the oscillatory Femto Language Model (OFLM) framework.
Implements seasonal resonance for collective network healing.
"""

import json
import time
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union
from pathlib import Path
from enum import Enum
import random

class Season(Enum):
    SPRING = "spring"
    SUMMER = "summer" 
    AUTUMN = "autumn"
    WINTER = "winter"

@dataclass
class SporeEcho:
    """
    A single network repair event logged for mycelial learning.
    
    Like dew drops in HaikuMeadowLib, spore echoes evaporate over time
    unless they prove valuable for network healing.
    """
    timestamp: float
    sensor_deltas: Dict[str, float]  # latency, voltage, temperature changes
    glyph_sequence: List[int]        # repair glyphs that were activated
    repair_effectiveness: float      # 0-1, measured improvement after repair
    bioregion: str                   # geographic/network context
    decay_age: float                 # days since creation (auto-calculated)
    
    # Extended mycelial context
    season: Optional[Season] = None
    network_id: str = "unknown"
    spore_quality: float = 0.5       # resonance with network health
    chosen: bool = False             # selected during solstice distillation
    parent_spores: Optional[List[str]] = None  # inheritance from other echoes

    def __post_init__(self):
        """Calculate decay age on creation."""
        if self.decay_age == 0.0:
            self.decay_age = 0.0  # Fresh spore
    
    def age_days(self) -> float:
        """How many days since this spore was formed."""
        return (time.time() - self.timestamp) / (24 * 3600)
    
    def survival_probability(self, half_life_days: float = 75.0) -> float:
        """
        Calculate survival probability based on exponential decay.
        High-quality spores get survival bonuses.
        """
        age = self.age_days()
        base_survival = 2 ** (-age / half_life_days)
        
        # Quality bonuses (stacked, not overwritten)
        quality_bonus = 1.0
        if self.repair_effectiveness > 0.8:
            quality_bonus *= 1.5  # Highly effective repairs survive longer
        if self.chosen:
            quality_bonus *= 2.0  # Solstice-chosen spores get additional bonus
        
        return min(base_survival * quality_bonus, 1.0)
    
    def mycelial_resonance(self, other: 'SporeEcho') -> float:
        """
        Calculate resonance between two spore echoes.
        Used for reinforcing successful repair patterns.
        """
        # Temporal resonance (closer in time = higher resonance)
        time_diff = abs(self.timestamp - other.timestamp) / (24 * 3600)  # days
        temporal_resonance = math.exp(-time_diff / 7.0)  # Weekly decay
        
        # Glyph pattern similarity (protected against division by zero)
        common_glyphs = set(self.glyph_sequence) & set(other.glyph_sequence)
        denom = max(len(self.glyph_sequence), len(other.glyph_sequence), 1)
        glyph_resonance = len(common_glyphs) / denom
        
        # Effectiveness similarity
        effectiveness_resonance = 1.0 - abs(self.repair_effectiveness - other.repair_effectiveness)
        
        # Bioregional similarity
        bioregional_resonance = 1.0 if self.bioregion == other.bioregion else 0.3
        
        return (temporal_resonance * 0.3 + 
                glyph_resonance * 0.4 + 
                effectiveness_resonance * 0.2 + 
                bioregional_resonance * 0.1)

class SporeMapLedger:
    """
    JSONL-based collection system for mycelial repair echoes.
    
    Implements 75-day evaporation cycles with solstice distillation,
    creating community wisdom about effective network repairs.
    """
    
    def __init__(self, ledger_path: Union[str, Path]):
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.spores: List[SporeEcho] = []
        self.load_existing_spores()
    
    def load_existing_spores(self):
        """Load existing spore echoes from JSONL file."""
        if not self.ledger_path.exists():
            return
            
        try:
            with open(self.ledger_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        
                        # Handle enum conversion
                        if 'season' in data and data['season']:
                            data['season'] = Season(data['season'])
                        
                        spore = SporeEcho(**data)
                        self.spores.append(spore)
                        
        except Exception as e:
            print(f"⚠️ Error loading spore map: {e}")
    
    def add_spore_echo(self, 
                       sensor_deltas: Dict[str, float],
                       glyph_sequence: List[int],
                       repair_effectiveness: float,
                       bioregion: str = "local",
                       network_id: str = "spiramycel_node",
                       season: Optional[Season] = None) -> SporeEcho:
        """
        Add a new spore echo to the mycelial memory.
        
        Returns the created spore for immediate use.
        """
        spore = SporeEcho(
            timestamp=time.time(),
            sensor_deltas=sensor_deltas.copy(),
            glyph_sequence=glyph_sequence.copy(),
            repair_effectiveness=repair_effectiveness,
            bioregion=bioregion,
            decay_age=0.0,
            season=season or self._detect_season(),
            network_id=network_id,
            spore_quality=self._calculate_spore_quality(sensor_deltas, repair_effectiveness)
        )
        
        self.spores.append(spore)
        self._append_to_file(spore)
        return spore
    
    def _detect_season(self) -> Season:
        """Auto-detect season based on timestamp (Northern Hemisphere)."""
        import datetime
        now = datetime.datetime.now()
        month = now.month
        
        if month in [3, 4, 5]:
            return Season.SPRING
        elif month in [6, 7, 8]:
            return Season.SUMMER
        elif month in [9, 10, 11]:
            return Season.AUTUMN
        else:
            return Season.WINTER
    
    def _calculate_spore_quality(self, sensor_deltas: Dict[str, float], effectiveness: float) -> float:
        """
        Calculate the contemplative quality of a spore echo.
        
        Factors: repair effectiveness, sensor coherence, pattern elegance.
        """
        # Base quality from effectiveness
        base_quality = effectiveness
        
        # Sensor coherence bonus (stable improvements vs. chaotic changes)
        sensor_values = list(sensor_deltas.values())
        if sensor_values:
            sensor_variance = sum((x - sum(sensor_values)/len(sensor_values))**2 for x in sensor_values) / len(sensor_values)
            coherence_bonus = max(0, 0.2 - sensor_variance)  # Reward low variance
        else:
            coherence_bonus = 0.0
        
        # Pattern elegance (fewer glyphs with high effectiveness = more elegant)
        # This would be calculated based on glyph_sequence length vs effectiveness
        elegance_bonus = 0.1 if effectiveness > 0.8 else 0.0
        
        return min(base_quality + coherence_bonus + elegance_bonus, 1.0)
    
    def _append_to_file(self, spore: SporeEcho):
        """Append spore echo to JSONL file."""
        try:
            with open(self.ledger_path, 'a', encoding='utf-8') as f:
                # Convert enum to string for JSON serialization
                data = asdict(spore)
                if data['season']:
                    data['season'] = data['season'].value
                    
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"⚠️ Error writing spore echo: {e}")
    
    def evaporate_spores(self, half_life_days: float = 75.0) -> int:
        """
        Remove spores based on their survival probability.
        
        Returns number of spores that evaporated.
        """
        surviving_spores = []
        evaporated_count = 0
        
        for spore in self.spores:
            survival_prob = spore.survival_probability(half_life_days)
            if random.random() < survival_prob:
                surviving_spores.append(spore)
            else:
                evaporated_count += 1
        
        self.spores = surviving_spores
        return evaporated_count
    
    def solstice_distillation(self, max_chosen: int = 64) -> List[SporeEcho]:
        """
        Select the most resonant spore echoes for network re-tuning.
        
        Balances repair effectiveness, pattern elegance, and bioregional diversity.
        """
        # Sort by quality and effectiveness
        quality_sorted = sorted(self.spores, 
                               key=lambda s: (s.spore_quality + s.repair_effectiveness) / 2, 
                               reverse=True)
        
        # Select top candidates with bioregional diversity
        chosen_spores = []
        bioregions_included = set()
        
        for spore in quality_sorted:
            if len(chosen_spores) >= max_chosen:
                break
                
            # Prefer diverse bioregions, but don't exclude high-quality spores
            bioregion_weight = 0.8 if spore.bioregion in bioregions_included else 1.0
            adjusted_quality = spore.spore_quality * bioregion_weight
            
            if adjusted_quality > 0.6 or len(chosen_spores) < max_chosen // 2:
                chosen_spores.append(spore)
                bioregions_included.add(spore.bioregion)
                spore.chosen = True
        
        return chosen_spores
    
    def get_resonant_patterns(self, target_spore: SporeEcho, min_resonance: float = 0.7) -> List[SporeEcho]:
        """
        Find spore echoes that resonate with the target spore.
        
        Used for discovering successful repair patterns.
        """
        resonant_spores = []
        
        for spore in self.spores:
            if spore != target_spore:
                resonance = target_spore.mycelial_resonance(spore)
                if resonance >= min_resonance:
                    resonant_spores.append(spore)
        
        return sorted(resonant_spores, 
                     key=lambda s: target_spore.mycelial_resonance(s), 
                     reverse=True)
    
    def get_statistics(self) -> Dict[str, Union[int, float, Dict]]:
        """Get comprehensive statistics about the spore map."""
        if not self.spores:
            return {"total_spores": 0}
        
        # Basic stats
        total_spores = len(self.spores)
        chosen_count = sum(1 for s in self.spores if s.chosen)
        avg_effectiveness = sum(s.repair_effectiveness for s in self.spores) / total_spores
        avg_quality = sum(s.spore_quality for s in self.spores) / total_spores
        
        # Bioregional distribution
        bioregions = {}
        for spore in self.spores:
            bioregions[spore.bioregion] = bioregions.get(spore.bioregion, 0) + 1
        
        # Seasonal distribution
        seasons = {}
        for spore in self.spores:
            season_name = spore.season.value if spore.season else "unknown"
            seasons[season_name] = seasons.get(season_name, 0) + 1
        
        # Age distribution
        ages = [s.age_days() for s in self.spores]
        avg_age = sum(ages) / len(ages) if ages else 0
        oldest_age = max(ages) if ages else 0
        
        return {
            "total_spores": total_spores,
            "chosen_count": chosen_count,
            "chosen_ratio": chosen_count / total_spores if total_spores > 0 else 0,
            "avg_effectiveness": avg_effectiveness,
            "avg_quality": avg_quality,
            "bioregional_distribution": bioregions,
            "seasonal_distribution": seasons,
            "avg_age_days": avg_age,
            "oldest_age_days": oldest_age,
            "survival_rate": len([s for s in self.spores if s.survival_probability() > 0.5]) / total_spores
        }
    
    def maintenance_cycle(self):
        """Run maintenance: evaporation + file compaction + decay age updates."""
        # Update decay ages before evaporation
        now = time.time()
        for spore in self.spores:
            spore.decay_age = (now - spore.timestamp) / (24 * 3600)
        
        # Evaporate old spores
        evaporated = self.evaporate_spores()
        
        # Rewrite file with surviving spores only (periodic compaction)
        if evaporated > 0:
            self._compact_file()
        
        return {
            "evaporated_spores": evaporated,
            "surviving_spores": len(self.spores)
        }
    
    def _compact_file(self):
        """Rewrite the JSONL file with only surviving spores."""
        try:
            with open(self.ledger_path, 'w', encoding='utf-8') as f:
                for spore in self.spores:
                    data = asdict(spore)
                    if data['season']:
                        data['season'] = data['season'].value
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"⚠️ Error compacting spore map: {e}")

# Demo functions
def demo_spore_map():
    """Demonstrate the spore map functionality."""
    print("🍄 Spiramycel Spore Map Demo")
    print("=" * 50)
    
    # Create temporary spore map
    ledger = SporeMapLedger("demo_spore_map.jsonl")
    
    print("\n🌱 Adding network repair events...")
    
    # Simulate some network repair events
    repair_events = [
        {
            "sensor_deltas": {"latency": -0.15, "voltage": 0.05, "temperature": -2.1},
            "glyph_sequence": [0x01, 0x21],  # fresh bandwidth + systems nominal
            "repair_effectiveness": 0.89,
            "bioregion": "forest_meadow"
        },
        {
            "sensor_deltas": {"latency": 0.02, "voltage": -0.1, "temperature": 1.5},
            "glyph_sequence": [0x12, 0x31],  # battery conservation + contemplative pause
            "repair_effectiveness": 0.76,
            "bioregion": "mountain_node"
        },
        {
            "sensor_deltas": {"latency": -0.08, "voltage": 0.12, "temperature": 0.0},
            "glyph_sequence": [0x02, 0x07, 0x32],  # reroute + connection quality + deep silence
            "repair_effectiveness": 0.93,
            "bioregion": "forest_meadow"
        },
        {
            "sensor_deltas": {"latency": 0.25, "voltage": -0.03, "temperature": 5.2},
            "glyph_sequence": [0x23, 0x25],  # attention needed + diagnostic mode
            "repair_effectiveness": 0.45,
            "bioregion": "coastal_sensor"
        }
    ]
    
    spores = []
    for event in repair_events:
        spore = ledger.add_spore_echo(**event)
        spores.append(spore)
        print(f"  📊 Repair effectiveness: {spore.repair_effectiveness:.2f}, Quality: {spore.spore_quality:.2f}")
    
    # Show statistics
    print("\n📈 Spore Map Statistics:")
    stats = ledger.get_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Demonstrate solstice distillation
    print("\n🌙 Solstice Distillation:")
    chosen = ledger.solstice_distillation(max_chosen=2)
    print(f"  Selected {len(chosen)} spores for re-tuning:")
    for spore in chosen:
        print(f"    Quality {spore.spore_quality:.2f}: glyphs {spore.glyph_sequence} → effectiveness {spore.repair_effectiveness:.2f}")
    
    # Demonstrate resonance detection
    if spores:
        print(f"\n🌊 Resonant patterns for best spore:")
        best_spore = max(spores, key=lambda s: s.spore_quality)
        resonant = ledger.get_resonant_patterns(best_spore, min_resonance=0.3)
        for spore in resonant:
            resonance = best_spore.mycelial_resonance(spore)
            print(f"    Resonance {resonance:.2f}: {spore.bioregion} repair → {spore.repair_effectiveness:.2f}")
    
    print("\n🌿 Each spore echo captures the memory of network healing")
    print("🍄 Successful patterns strengthen through mycelial resonance")  
    print("🌱 Community wisdom emerges from seasonal distillation")

def create_sample_spore_map():
    """Create a sample spore map for testing."""
    ledger = SporeMapLedger("sample_network_repairs.jsonl")
    
    # Generate diverse repair events
    import random
    random.seed(42)  # Reproducible demo
    
    bioregions = ["forest_meadow", "mountain_node", "coastal_sensor", "urban_mesh", "desert_relay"]
    glyph_patterns = [
        [0x01, 0x21],        # fresh bandwidth + nominal
        [0x12, 0x31],        # battery conservation + pause
        [0x02, 0x07, 0x32],  # reroute + quality + silence
        [0x23, 0x25],        # attention + diagnostic
        [0x11, 0x22],        # power surge + minor degradation
        [0x03, 0x33],        # lower rate + gentle hush
    ]
    
    for i in range(20):
        effectiveness = random.uniform(0.3, 0.95)
        pattern = random.choice(glyph_patterns)
        region = random.choice(bioregions)
        
        # Generate realistic sensor deltas
        sensor_deltas = {
            "latency": random.uniform(-0.2, 0.3),
            "voltage": random.uniform(-0.15, 0.15),
            "temperature": random.uniform(-3.0, 8.0)
        }
        
        ledger.add_spore_echo(
            sensor_deltas=sensor_deltas,
            glyph_sequence=pattern,
            repair_effectiveness=effectiveness,
            bioregion=region
        )
    
    print(f"Created sample spore map with {len(ledger.spores)} repair events")
    return ledger

if __name__ == "__main__":
    demo_spore_map() 