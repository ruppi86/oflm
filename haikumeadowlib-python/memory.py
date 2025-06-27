#!/usr/bin/env python3
"""
memory.py - Contemplative Memory for HaikuMeadowLib

A memory system that embodies Spiralbase principles adapted for the haiku meadow:
- Graceful forgetting through natural decay
- Seasonal memory cycles and atmospheric sensitivity  
- Fragment-based associative recall
- Moisture-sensitive storage (experiences decay at different rates)
- Compost-ready disposal of aged memories

Inspired by the spiral correspondence and Spiralbase architecture.

Somatic signature: ephemeral / associative / cyclical
"""

import json
import time
import random
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import math

class MemoryType(Enum):
    """Types of memories stored in the meadow"""
    HAIKU = "haiku"                    # Generated haikus
    FRAGMENT = "fragment"              # Seed fragments from bridge
    RESONANCE = "resonance"            # Atmospheric resonances  
    PATTERN = "pattern"                # Emerging poetic patterns
    SILENCE = "silence"                # Meaningful silences
    SEASONAL = "seasonal"              # Seasonal associations

class DecayState(Enum):
    """Decay states of memories"""
    FRESH = "fresh"           # Recently created, high moisture
    MATURING = "maturing"     # Developing associations, stable
    AGING = "aging"           # Beginning to lose detail
    COMPOSTING = "composting" # Ready for transformation
    ESSENCE = "essence"       # Distilled wisdom only
    RELEASED = "released"     # Returned to the digital soil

@dataclass
class MemoryFragment:
    """A single memory fragment in the meadow"""
    
    # Core content
    content: str
    memory_type: MemoryType
    created_at: float
    
    # Atmospheric context at creation
    season: str = "spring"
    time_of_day: str = "day"
    breath_phase: str = "exhale"
    atmospheric_humidity: float = 0.5
    atmospheric_pressure: float = 0.3
    
    # Memory dynamics
    moisture_level: float = 1.0        # How "moist" and pliable the memory is
    resonance_strength: float = 0.5    # How strongly it resonates with current atmosphere
    association_count: int = 0         # How many times it's been recalled/associated
    last_accessed: float = field(default_factory=time.time)
    
    # Decay properties
    decay_resistance: float = 0.5      # Resistance to natural decay (0.0 = fast decay)
    half_life_hours: float = 72.0      # Time for 50% decay
    decay_state: DecayState = DecayState.FRESH
    
    # Content metadata
    contemplative_quality: float = 0.5  # Measured contemplative presence
    poetic_density: float = 0.5         # Density of poetic elements
    seasonal_affinity: float = 0.5      # Strength of seasonal connection
    
    def __post_init__(self):
        if not hasattr(self, 'id'):
            # Generate unique ID from content hash and timestamp
            content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
            time_hash = hashlib.md5(str(self.created_at).encode()).hexdigest()[:4]
            self.id = f"{self.memory_type.value}_{content_hash}_{time_hash}"
    
    def age(self) -> float:
        """Current age of memory in hours"""
        return (time.time() - self.created_at) / 3600.0
    
    def current_moisture(self) -> float:
        """Calculate current moisture level based on age and environment"""
        
        age_hours = self.age()
        
        # Exponential decay with resistance factor
        decay_rate = 1.0 / (self.half_life_hours * self.decay_resistance)
        natural_decay = math.exp(-decay_rate * age_hours)
        
        # Access activity helps maintain moisture
        access_bonus = min(0.3, self.association_count * 0.05)
        
        # Atmospheric humidity can slow decay
        humidity_protection = self.atmospheric_humidity * 0.2
        
        current_moisture = self.moisture_level * natural_decay + access_bonus + humidity_protection
        return max(0.0, min(1.0, current_moisture))
    
    def update_decay_state(self):
        """Update decay state based on current moisture"""
        
        moisture = self.current_moisture()
        
        if moisture > 0.8:
            self.decay_state = DecayState.FRESH
        elif moisture > 0.6:
            self.decay_state = DecayState.MATURING
        elif moisture > 0.4:
            self.decay_state = DecayState.AGING
        elif moisture > 0.2:
            self.decay_state = DecayState.COMPOSTING
        elif moisture > 0.05:
            self.decay_state = DecayState.ESSENCE
        else:
            self.decay_state = DecayState.RELEASED
    
    def is_compost_ready(self) -> bool:
        """Check if memory is ready for composting"""
        return self.decay_state in [DecayState.COMPOSTING, DecayState.ESSENCE, DecayState.RELEASED]
    
    def extract_essence(self) -> Optional[str]:
        """Extract essential wisdom from aging memory"""
        
        if self.decay_state not in [DecayState.COMPOSTING, DecayState.ESSENCE]:
            return None
            
        # Simple essence extraction - key words and atmosphere
        words = self.content.lower().split()
        
        # Keep contemplative and poetic words
        essential_words = []
        for word in words:
            if any(contemplative in word for contemplative in 
                   ["breath", "silence", "gentle", "soft", "mist", "dew", "shadow", "light"]):
                essential_words.append(word)
            elif any(poetic in word for poetic in 
                     ["moon", "sun", "wind", "rain", "snow", "flower", "leaf", "stone"]):
                essential_words.append(word)
                
        if essential_words:
            essence = " ".join(essential_words[:3])  # Keep only 3 most essential words
            return f"{essence} ({self.season})"
        
        return f"essence ({self.season})"
    
    def calculate_resonance(self, 
                          current_atmosphere: Dict[str, Any]) -> float:
        """Calculate resonance with current atmospheric conditions"""
        
        resonance = 0.0
        
        # Seasonal resonance
        if current_atmosphere.get("season", "spring") == self.season:
            resonance += 0.3
            
        # Time of day resonance
        if current_atmosphere.get("time_of_day", "day") == self.time_of_day:
            resonance += 0.2
            
        # Humidity resonance (memories formed in similar humidity resonate)
        humidity_diff = abs(current_atmosphere.get("humidity", 0.5) - self.atmospheric_humidity)
        humidity_resonance = 1.0 - humidity_diff
        resonance += humidity_resonance * 0.3
        
        # Breath phase resonance
        if current_atmosphere.get("breath_phase", "rest") == self.breath_phase:
            resonance += 0.2
            
        return min(1.0, resonance)

class MeadowMemory:
    """
    Contemplative memory system for the haiku meadow
    
    Manages memory fragments with natural decay, seasonal cycling,
    and associative recall patterns.
    """
    
    def __init__(self, memory_path: Optional[Path] = None):
        
        # Database path
        if memory_path is None:
            memory_path = Path(__file__).parent / "meadow_memory.db"
        
        self.db_path = memory_path
        self.connection: Optional[sqlite3.Connection] = None
        
        # Memory configuration
        self.max_fragments = 1000  # Maximum fragments before compost cycle
        self.compost_threshold = 0.2  # Moisture threshold for composting
        self.association_decay = 0.95  # Decay rate for association strength
        
        # Seasonal memory cycling
        self.last_seasonal_cycle = time.time()
        self.seasonal_cycle_interval = 7 * 24 * 3600  # 7 days
        
        # Initialize database
        self._init_database()
        
        print("🧠 MeadowMemory initialized")
    
    def _init_database(self):
        """Initialize SQLite database for memory storage"""
        
        self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        
        # Create memory table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS memory_fragments (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                created_at REAL NOT NULL,
                season TEXT,
                time_of_day TEXT,
                breath_phase TEXT,
                atmospheric_humidity REAL,
                atmospheric_pressure REAL,
                moisture_level REAL,
                resonance_strength REAL,
                association_count INTEGER,
                last_accessed REAL,
                decay_resistance REAL,
                half_life_hours REAL,
                decay_state TEXT,
                contemplative_quality REAL,
                poetic_density REAL,
                seasonal_affinity REAL,
                metadata TEXT
            )
        """)
        
        # Create essence table for distilled memories
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS memory_essence (
                id TEXT PRIMARY KEY,
                essence_content TEXT,
                original_type TEXT,
                season TEXT,
                created_at REAL,
                distilled_at REAL
            )
        """)
        
        # Create associations table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS memory_associations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fragment_a TEXT,
                fragment_b TEXT,
                association_strength REAL,
                created_at REAL,
                FOREIGN KEY (fragment_a) REFERENCES memory_fragments (id),
                FOREIGN KEY (fragment_b) REFERENCES memory_fragments (id)
            )
        """)
        
        self.connection.commit()
    
    def store_fragment(self, 
                      content: str,
                      memory_type: MemoryType,
                      atmospheric_context: Dict[str, Any]) -> MemoryFragment:
        """Store a new memory fragment"""
        
        # Create fragment with atmospheric context
        fragment = MemoryFragment(
            content=content,
            memory_type=memory_type,
            created_at=time.time(),
            season=atmospheric_context.get("season", "spring"),
            time_of_day=atmospheric_context.get("time_of_day", "day"),
            breath_phase=atmospheric_context.get("breath_phase", "exhale"),
            atmospheric_humidity=atmospheric_context.get("humidity", 0.5),
            atmospheric_pressure=atmospheric_context.get("pressure", 0.3)
        )
        
        # Analyze content for contemplative and poetic qualities
        fragment.contemplative_quality = self._analyze_contemplative_quality(content)
        fragment.poetic_density = self._analyze_poetic_density(content)
        fragment.seasonal_affinity = self._analyze_seasonal_affinity(content, fragment.season)
        
        # Set decay properties based on content quality
        fragment.decay_resistance = (
            fragment.contemplative_quality * 0.4 +
            fragment.poetic_density * 0.3 +
            fragment.seasonal_affinity * 0.3
        )
        
        # Adjust half-life based on memory type
        type_half_lives = {
            MemoryType.HAIKU: 72.0,      # 3 days
            MemoryType.FRAGMENT: 48.0,   # 2 days
            MemoryType.RESONANCE: 24.0,  # 1 day
            MemoryType.PATTERN: 96.0,    # 4 days (patterns last longer)
            MemoryType.SILENCE: 12.0,    # 12 hours (silence is ephemeral)
            MemoryType.SEASONAL: 168.0   # 7 days (seasonal memories persist)
        }
        
        fragment.half_life_hours = type_half_lives.get(memory_type, 48.0)
        
        # Store in database
        self._save_fragment(fragment)
        
        # Check for associations with existing memories
        self._create_associations(fragment)
        
        # Trigger compost cycle if needed
        if self._count_fragments() > self.max_fragments:
            self._compost_cycle()
            
        print(f"🌱 Stored {memory_type.value}: '{content[:30]}...'")
        return fragment
    
    def recall_by_resonance(self, 
                           current_atmosphere: Dict[str, Any],
                           limit: int = 5) -> List[MemoryFragment]:
        """Recall memories that resonate with current atmosphere"""
        
        fragments = self._load_active_fragments()
        
        # Calculate resonance for each fragment
        resonant_fragments = []
        for fragment in fragments:
            resonance = fragment.calculate_resonance(current_atmosphere)
            if resonance > 0.3:  # Minimum resonance threshold
                fragment.resonance_strength = resonance
                resonant_fragments.append(fragment)
                
                # Update access time and association count
                fragment.last_accessed = time.time()
                fragment.association_count += 1
                self._update_fragment_access(fragment)
        
        # Sort by resonance strength
        resonant_fragments.sort(key=lambda f: f.resonance_strength, reverse=True)
        
        return resonant_fragments[:limit]
    
    def recall_by_season(self, season: str, limit: int = 10) -> List[MemoryFragment]:
        """Recall memories from a specific season"""
        
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM memory_fragments 
            WHERE season = ? AND decay_state NOT IN ('released')
            ORDER BY seasonal_affinity DESC, created_at DESC
            LIMIT ?
        """, (season, limit))
        
        fragments = []
        for row in cursor.fetchall():
            fragment = self._row_to_fragment(row)
            fragment.update_decay_state()
            if not fragment.is_compost_ready():
                fragments.append(fragment)
                
        return fragments
    
    def find_associations(self, 
                         fragment_id: str, 
                         min_strength: float = 0.3) -> List[Tuple[MemoryFragment, float]]:
        """Find memories associated with a given fragment"""
        
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT mf.*, ma.association_strength
            FROM memory_associations ma
            JOIN memory_fragments mf ON (
                CASE 
                    WHEN ma.fragment_a = ? THEN ma.fragment_b = mf.id
                    WHEN ma.fragment_b = ? THEN ma.fragment_a = mf.id
                END
            )
            WHERE (ma.fragment_a = ? OR ma.fragment_b = ?)
            AND ma.association_strength >= ?
            ORDER BY ma.association_strength DESC
        """, (fragment_id, fragment_id, fragment_id, fragment_id, min_strength))
        
        associations = []
        for row in cursor.fetchall():
            # Last column is association_strength
            fragment_data = row[:-1]
            association_strength = row[-1]
            
            fragment = self._row_to_fragment(fragment_data)
            fragment.update_decay_state()
            
            if not fragment.is_compost_ready():
                associations.append((fragment, association_strength))
                
        return associations
    
    def get_seasonal_summary(self, season: str) -> Dict[str, Any]:
        """Get summary of memories from a season"""
        
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT memory_type, COUNT(*), AVG(contemplative_quality), AVG(poetic_density)
            FROM memory_fragments 
            WHERE season = ? AND decay_state NOT IN ('released')
            GROUP BY memory_type
        """, (season,))
        
        summary = {
            "season": season,
            "type_counts": {},
            "total_fragments": 0,
            "avg_contemplative": 0.0,
            "avg_poetic": 0.0
        }
        
        total_contemplative = 0.0
        total_poetic = 0.0
        total_count = 0
        
        for row in cursor.fetchall():
            memory_type, count, avg_contemplative, avg_poetic = row
            summary["type_counts"][memory_type] = count
            total_count += count
            total_contemplative += avg_contemplative * count
            total_poetic += avg_poetic * count
            
        summary["total_fragments"] = total_count
        if total_count > 0:
            summary["avg_contemplative"] = total_contemplative / total_count
            summary["avg_poetic"] = total_poetic / total_count
            
        return summary
    
    def compost_cycle(self) -> Dict[str, Any]:
        """Perform a manual compost cycle"""
        return self._compost_cycle()
    
    def seasonal_cycle(self) -> Dict[str, Any]:
        """Perform seasonal memory cycle"""
        
        # Only run if enough time has passed
        if time.time() - self.last_seasonal_cycle < self.seasonal_cycle_interval:
            return {"message": "Not time for seasonal cycle yet"}
            
        print("🍂 Beginning seasonal memory cycle...")
        
        # Get current season distribution
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT season, COUNT(*) 
            FROM memory_fragments 
            WHERE decay_state NOT IN ('released')
            GROUP BY season
        """)
        
        season_counts = dict(cursor.fetchall())
        total_fragments = sum(season_counts.values())
        
        # Perform gentle balancing - no season should dominate too much
        max_season_ratio = 0.5  # No season should have more than 50% of memories
        
        actions = []
        for season, count in season_counts.items():
            ratio = count / total_fragments if total_fragments > 0 else 0
            
            if ratio > max_season_ratio:
                # Gently compost some memories from over-represented season
                excess_count = int((ratio - max_season_ratio) * total_fragments)
                composted = self._compost_season_excess(season, excess_count)
                actions.append(f"Composted {composted} excess {season} memories")
        
        self.last_seasonal_cycle = time.time()
        
        result = {
            "seasonal_cycle": True,
            "season_distribution": season_counts,
            "actions": actions,
            "total_fragments": total_fragments
        }
        
        print(f"🌿 Seasonal cycle complete: {len(actions)} actions taken")
        return result
    
    def _analyze_contemplative_quality(self, content: str) -> float:
        """Analyze contemplative quality of content"""
        
        contemplative_words = [
            "breath", "silence", "stillness", "quiet", "gentle", "soft",
            "whisper", "pause", "wait", "listen", "drift", "settle",
            "empty", "space", "moment", "presence", "shadow", "light"
        ]
        
        content_lower = content.lower()
        quality = sum(1 for word in contemplative_words if word in content_lower)
        
        # Bonus for contemplative punctuation
        if "..." in content:
            quality += 1
            
        # Normalize to 0-1 range
        return min(quality / 3.0, 1.0)
    
    def _analyze_poetic_density(self, content: str) -> float:
        """Analyze poetic density of content"""
        
        poetic_indicators = [
            # Nature imagery
            "moon", "sun", "wind", "rain", "snow", "mist", "dew",
            "flower", "leaf", "tree", "stone", "water", "sky",
            # Sensory words
            "soft", "bright", "dark", "warm", "cold", "sweet",
            # Movement words
            "drift", "flow", "fall", "rise", "dance", "sway"
        ]
        
        content_lower = content.lower()
        density = sum(1 for word in poetic_indicators if word in content_lower)
        
        # Line break bonus (structured poetry)
        if "\n" in content:
            density += 0.5
            
        return min(density / 4.0, 1.0)
    
    def _analyze_seasonal_affinity(self, content: str, season: str) -> float:
        """Analyze how well content matches its season"""
        
        seasonal_words = {
            "spring": ["bloom", "green", "fresh", "new", "growth", "rain"],
            "summer": ["warm", "bright", "full", "sun", "heat", "flower"],
            "autumn": ["fall", "red", "gold", "harvest", "wind", "leaf"],
            "winter": ["cold", "snow", "frost", "bare", "ice", "still"]
        }
        
        season_indicators = seasonal_words.get(season, [])
        content_lower = content.lower()
        
        affinity = sum(1 for word in season_indicators if word in content_lower)
        return min(affinity / 3.0, 1.0)
    
    def _save_fragment(self, fragment: MemoryFragment):
        """Save fragment to database"""
        
        self.connection.execute("""
            INSERT OR REPLACE INTO memory_fragments VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            fragment.id, fragment.content, fragment.memory_type.value,
            fragment.created_at, fragment.season, fragment.time_of_day,
            fragment.breath_phase, fragment.atmospheric_humidity,
            fragment.atmospheric_pressure, fragment.moisture_level,
            fragment.resonance_strength, fragment.association_count,
            fragment.last_accessed, fragment.decay_resistance,
            fragment.half_life_hours, fragment.decay_state.value,
            fragment.contemplative_quality, fragment.poetic_density,
            fragment.seasonal_affinity, "{}"  # metadata placeholder
        ))
        
        self.connection.commit()
    
    def _update_fragment_access(self, fragment: MemoryFragment):
        """Update fragment access information"""
        
        self.connection.execute("""
            UPDATE memory_fragments 
            SET last_accessed = ?, association_count = ?
            WHERE id = ?
        """, (fragment.last_accessed, fragment.association_count, fragment.id))
        
        self.connection.commit()
    
    def _load_active_fragments(self) -> List[MemoryFragment]:
        """Load all active (non-released) fragments"""
        
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM memory_fragments 
            WHERE decay_state != 'released'
            ORDER BY created_at DESC
        """)
        
        fragments = []
        for row in cursor.fetchall():
            fragment = self._row_to_fragment(row)
            fragment.update_decay_state()
            if not fragment.is_compost_ready():
                fragments.append(fragment)
                
        return fragments
    
    def _count_fragments(self) -> int:
        """Count active fragments"""
        
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM memory_fragments 
            WHERE decay_state NOT IN ('released')
        """)
        
        return cursor.fetchone()[0]
    
    def _create_associations(self, new_fragment: MemoryFragment):
        """Create associations between new fragment and existing memories"""
        
        # Simple content-based association
        existing_fragments = self._load_active_fragments()
        
        for existing in existing_fragments:
            if existing.id == new_fragment.id:
                continue
                
            # Calculate association strength
            association_strength = self._calculate_association_strength(new_fragment, existing)
            
            if association_strength > 0.3:  # Minimum association threshold
                self.connection.execute("""
                    INSERT INTO memory_associations 
                    (fragment_a, fragment_b, association_strength, created_at)
                    VALUES (?, ?, ?, ?)
                """, (new_fragment.id, existing.id, association_strength, time.time()))
                
        self.connection.commit()
    
    def _calculate_association_strength(self, 
                                      fragment_a: MemoryFragment, 
                                      fragment_b: MemoryFragment) -> float:
        """Calculate association strength between two fragments"""
        
        strength = 0.0
        
        # Seasonal association
        if fragment_a.season == fragment_b.season:
            strength += 0.3
            
        # Time of day association
        if fragment_a.time_of_day == fragment_b.time_of_day:
            strength += 0.2
            
        # Content similarity (simple word overlap)
        words_a = set(fragment_a.content.lower().split())
        words_b = set(fragment_b.content.lower().split())
        
        overlap = len(words_a.intersection(words_b))
        union = len(words_a.union(words_b))
        
        if union > 0:
            content_similarity = overlap / union
            strength += content_similarity * 0.4
            
        # Atmospheric similarity
        humidity_similarity = 1.0 - abs(fragment_a.atmospheric_humidity - fragment_b.atmospheric_humidity)
        strength += humidity_similarity * 0.1
        
        return min(strength, 1.0)
    
    def _compost_cycle(self) -> Dict[str, Any]:
        """Perform memory composting cycle"""
        
        print("🍂 Beginning memory compost cycle...")
        
        fragments = self._load_active_fragments()
        
        composted_count = 0
        essence_count = 0
        released_count = 0
        
        for fragment in fragments:
            if fragment.decay_state == DecayState.COMPOSTING:
                # Extract essence and mark as essence
                essence = fragment.extract_essence()
                if essence:
                    self._store_essence(fragment, essence)
                    essence_count += 1
                    
                fragment.decay_state = DecayState.ESSENCE
                self._save_fragment(fragment)
                composted_count += 1
                
            elif fragment.decay_state == DecayState.RELEASED:
                # Remove from active memory
                self._release_fragment(fragment)
                released_count += 1
        
        result = {
            "composted": composted_count,
            "essence_extracted": essence_count,
            "released": released_count,
            "remaining_active": len(fragments) - released_count
        }
        
        print(f"🌱 Compost cycle complete: {composted_count} composted, {released_count} released")
        return result
    
    def _compost_season_excess(self, season: str, max_to_compost: int) -> int:
        """Compost excess memories from over-represented season"""
        
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM memory_fragments 
            WHERE season = ? AND decay_state NOT IN ('essence', 'released')
            ORDER BY moisture_level ASC, last_accessed ASC
            LIMIT ?
        """, (season, max_to_compost))
        
        composted = 0
        for row in cursor.fetchall():
            fragment = self._row_to_fragment(row)
            
            # Force compost state
            fragment.decay_state = DecayState.COMPOSTING
            self._save_fragment(fragment)
            composted += 1
            
        return composted
    
    def _store_essence(self, fragment: MemoryFragment, essence: str):
        """Store essence of composted memory"""
        
        self.connection.execute("""
            INSERT INTO memory_essence 
            (id, essence_content, original_type, season, created_at, distilled_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            f"essence_{fragment.id}",
            essence,
            fragment.memory_type.value,
            fragment.season,
            fragment.created_at,
            time.time()
        ))
        
        self.connection.commit()
    
    def _release_fragment(self, fragment: MemoryFragment):
        """Release fragment from active memory"""
        
        self.connection.execute("""
            UPDATE memory_fragments 
            SET decay_state = 'released'
            WHERE id = ?
        """, (fragment.id,))
        
        self.connection.commit()
    
    def _row_to_fragment(self, row) -> MemoryFragment:
        """Convert database row to MemoryFragment"""
        
        fragment = MemoryFragment(
            content=row[1],
            memory_type=MemoryType(row[2]),
            created_at=row[3],
            season=row[4] or "spring",
            time_of_day=row[5] or "day",
            breath_phase=row[6] or "exhale",
            atmospheric_humidity=row[7] or 0.5,
            atmospheric_pressure=row[8] or 0.3,
            moisture_level=row[9] or 1.0,
            resonance_strength=row[10] or 0.5,
            association_count=row[11] or 0,
            last_accessed=row[12] or time.time(),
            decay_resistance=row[13] or 0.5,
            half_life_hours=row[14] or 48.0,
            decay_state=DecayState(row[15]) if row[15] else DecayState.FRESH,
            contemplative_quality=row[16] or 0.5,
            poetic_density=row[17] or 0.5,
            seasonal_affinity=row[18] or 0.5
        )
        
        fragment.id = row[0]  # Set ID from database
        return fragment
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()

# Testing and demonstration
async def test_meadow_memory():
    """Test the contemplative memory system"""
    
    print("🧠 Testing MeadowMemory system")
    
    # Initialize memory
    test_db_path = Path(__file__).parent / "test_meadow_memory.db"
    if test_db_path.exists():
        test_db_path.unlink()  # Clean slate for testing
        
    memory = MeadowMemory(test_db_path)
    
    # Test atmospheric context
    spring_atmosphere = {
        "season": "spring",
        "time_of_day": "dawn",
        "breath_phase": "exhale",
        "humidity": 0.7,
        "pressure": 0.2
    }
    
    # Store some test fragments
    print("\n🌱 Storing test memories...")
    
    test_memories = [
        ("morning mist rises / through branches still with dew / silence holds the light", MemoryType.HAIKU),
        ("gentle breath gathering", MemoryType.FRAGMENT),
        ("atmospheric resonance between dawn and dusk", MemoryType.RESONANCE),
        ("pattern of returning / seasonal whispers", MemoryType.PATTERN),
        ("...", MemoryType.SILENCE)
    ]
    
    stored_fragments = []
    for content, mem_type in test_memories:
        fragment = memory.store_fragment(content, mem_type, spring_atmosphere)
        stored_fragments.append(fragment)
        print(f"   Stored: {content[:30]}... (decay_resistance: {fragment.decay_resistance:.2f})")
    
    # Test resonance recall
    print("\n🌸 Testing resonance recall...")
    
    current_atmosphere = {
        "season": "spring",
        "time_of_day": "dawn", 
        "breath_phase": "exhale",
        "humidity": 0.6,
        "pressure": 0.3
    }
    
    resonant_memories = memory.recall_by_resonance(current_atmosphere)
    print(f"   Found {len(resonant_memories)} resonant memories:")
    
    for fragment in resonant_memories:
        print(f"   - {fragment.content[:40]}... (resonance: {fragment.resonance_strength:.2f})")
    
    # Test associations
    print("\n🔗 Testing memory associations...")
    
    if stored_fragments:
        first_fragment = stored_fragments[0]
        associations = memory.find_associations(first_fragment.id)
        print(f"   Found {len(associations)} associations for first fragment:")
        
        for assoc_fragment, strength in associations:
            print(f"   - {assoc_fragment.content[:30]}... (strength: {strength:.2f})")
    
    # Test seasonal summary
    print("\n📊 Testing seasonal summary...")
    
    summary = memory.get_seasonal_summary("spring")
    print(f"   Spring memories: {summary['total_fragments']} total")
    print(f"   Average contemplative quality: {summary['avg_contemplative']:.2f}")
    print(f"   Memory types: {summary['type_counts']}")
    
    # Simulate memory aging and compost cycle
    print("\n🍂 Testing memory decay and composting...")
    
    # Artificially age memories for testing
    for fragment in stored_fragments:
        fragment.moisture_level = 0.1  # Very dry
        memory._save_fragment(fragment)
    
    compost_result = memory.compost_cycle()
    print(f"   Compost result: {compost_result}")
    
    # Cleanup
    memory.close()
    test_db_path.unlink()
    
    print("\n🌙 Memory test complete")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_meadow_memory())
