#!/usr/bin/env python3
"""
ingest.py - Haiku Ingestion with Contemplative Decay

A breath-aware script that processes haiku CSV files and creates training material
for the piko-LLM. Follows contemplative principles:
- Graceful decay (not all haikus are preserved)
- Seasonal awareness (different haikus for different atmospheric conditions)
- Minimal memory footprint
- Breathing rhythm in processing

Supports multiple CSV formats:
- all_haiku.csv: columns [index, 0, 1, 2, source, hash] 
- documarianum_1_haikus.csv: columns [0, 1, 2, source, 0_syllables, 1_syllables, 2_syllables]
- Notgnoshi_haiku.csv: columns [index, haiku, colors, lines, syllables, total_syllables]

Somatic signature: cyclical / selective / composting
"""

import csv
import random
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

# Seasonal awareness for training material
class Season(Enum):
    SPRING = "spring"
    SUMMER = "summer" 
    AUTUMN = "autumn"
    WINTER = "winter"
    
class TimeOfDay(Enum):
    DAWN = "dawn"
    DAY = "day"
    DUSK = "dusk"
    NIGHT = "night"

@dataclass
class HaikuFragment:
    """A single haiku with atmospheric metadata"""
    text: str
    lines: int
    syllables: tuple
    total_syllables: int
    colors: List[str]
    source: str = ""
    season_affinity: Optional[Season] = None
    time_affinity: Optional[TimeOfDay] = None
    decay_resistance: float = 0.5  # 0.0 = easily composted, 1.0 = preserved
    contemplative_quality: float = 0.5  # Measured by presence of contemplative words
    
    def to_training_line(self) -> str:
        """Convert to training format with breathing pauses"""
        # Replace / with actual line breaks for training
        formatted_text = self.text.replace(" / ", "\n")
        return formatted_text
        
    def should_preserve(self, decay_rate: float = 0.3) -> bool:
        """Decide if this haiku survives the composting process"""
        survival_score = (
            self.decay_resistance * 0.5 +
            self.contemplative_quality * 0.3 +
            random.random() * 0.2  # Gentle randomness
        )
        return survival_score > decay_rate

def detect_csv_format(csv_path: Path) -> str:
    """Detect which CSV format we're dealing with"""
    
    with open(csv_path, 'r', encoding='utf-8') as file:
        # Read first few lines to detect format
        sample_lines = []
        for i, line in enumerate(file):
            sample_lines.append(line.strip())
            if i >= 2:  # Header + 2 data lines should be enough
                break
    
    # Check header to determine format
    header = sample_lines[0].lower()
    
    if 'haiku' in header and 'colors' in header:
        return 'notgnoshi'  # Notgnoshi_haiku.csv format
    elif '0_syllables' in header or (len(sample_lines) > 1 and ',' in sample_lines[1] and sample_lines[1].count(',') >= 6):
        return 'documarianum'  # documarianum_1_haikus.csv format  
    else:
        return 'all_haiku'  # all_haiku.csv format
        
def parse_syllables_flexible(syllable_data) -> tuple:
    """Parse syllable count from various formats"""
    if not syllable_data:
        return ()
        
    # Handle string representation like '(5, 7, 5)' 
    if isinstance(syllable_data, str):
        try:
            # Remove parentheses and quotes, split by comma
            clean_str = syllable_data.strip('()"\' ').replace(' ', '')
            if clean_str:
                return tuple(int(x) for x in clean_str.split(','))
        except:
            pass
    
    # Handle integer
    if isinstance(syllable_data, int):
        return (syllable_data,)
        
    return ()

def parse_haiku_notgnoshi_format(row: Dict) -> Optional[HaikuFragment]:
    """Parse Notgnoshi format: haiku already combined with '/'"""
    
    haiku_text = row.get('haiku', '').strip()
    if not haiku_text:
        return None
        
    # Parse colors (they're stored as string representation of list)
    colors_str = row.get('colors', '[]')
    try:
        colors = eval(colors_str) if colors_str != '[]' else []
    except:
        colors = []
        
    # Parse syllable counts
    syllables = parse_syllables_flexible(row.get('syllables', ''))
    
    lines_count = int(row.get('lines', 3))
    total_syllables = int(row.get('total_syllables', 0))
    
    return HaikuFragment(
        text=haiku_text,
        lines=lines_count,
        syllables=syllables,
        total_syllables=total_syllables,
        colors=colors,
        source="notgnoshi"
    )

def parse_haiku_documarianum_format(row: Dict) -> Optional[HaikuFragment]:
    """Parse Documarianum format: 3 separate columns for haiku parts"""
    
    # Combine the three parts
    part0 = row.get('0', '').strip()
    part1 = row.get('1', '').strip()  
    part2 = row.get('2', '').strip()
    
    if not (part0 and part1 and part2):
        return None
        
    haiku_text = f"{part0} / {part1} / {part2}"
    
    # Parse syllables from separate columns
    syllables = []
    for col in ['0_syllables', '1_syllables', '2_syllables']:
        syl_data = row.get(col, '')
        if syl_data:
            try:
                # Handle formats like "2,3" or just "5"
                if ',' in str(syl_data):
                    syllables.extend([int(x) for x in str(syl_data).split(',')])
                else:
                    syllables.append(int(syl_data))
            except:
                syllables.append(5)  # Default
        else:
            syllables.append(5)  # Default
    
    total_syllables = sum(syllables)
    
    return HaikuFragment(
        text=haiku_text,
        lines=3,
        syllables=tuple(syllables),
        total_syllables=total_syllables,
        colors=[],
        source=row.get('source', 'documarianum')
    )

def parse_haiku_all_haiku_format(row: Dict) -> Optional[HaikuFragment]:
    """Parse all_haiku format: 3 separate columns (0, 1, 2)"""
    
    # Combine the three parts
    part0 = row.get('0', '').strip()
    part1 = row.get('1', '').strip()
    part2 = row.get('2', '').strip()
    
    if not (part0 and part1 and part2):
        return None
        
    haiku_text = f"{part0} / {part1} / {part2}"
    
    # No syllable data in this format, estimate
    syllables = (len(part0.split()), len(part1.split()), len(part2.split()))
    total_syllables = sum(syllables)
    
    return HaikuFragment(
        text=haiku_text,
        lines=3,
        syllables=syllables,
        total_syllables=total_syllables,
        colors=[],
        source=row.get('source', 'all_haiku')
    )

def sense_seasonal_affinity(haiku_text: str) -> Optional[Season]:
    """Sense which season a haiku resonates with"""
    text_lower = haiku_text.lower()
    
    spring_words = ["spring", "blossom", "bloom", "green", "birth", "dawn", "new", "fresh", "growth", "rain"]
    summer_words = ["summer", "heat", "sun", "warm", "bright", "flower", "full", "abundance"]
    autumn_words = ["autumn", "fall", "leaves", "harvest", "orange", "red", "fading", "transition"]
    winter_words = ["winter", "snow", "cold", "frost", "ice", "bare", "silence", "stillness"]
    
    season_scores = {
        Season.SPRING: sum(1 for word in spring_words if word in text_lower),
        Season.SUMMER: sum(1 for word in summer_words if word in text_lower),
        Season.AUTUMN: sum(1 for word in autumn_words if word in text_lower),
        Season.WINTER: sum(1 for word in winter_words if word in text_lower)
    }
    
    max_score = max(season_scores.values())
    if max_score > 0:
        return max(season_scores.keys(), key=lambda k: season_scores[k])
    return None

def sense_time_affinity(haiku_text: str) -> Optional[TimeOfDay]:
    """Sense what time of day a haiku evokes"""
    text_lower = haiku_text.lower()
    
    dawn_words = ["dawn", "morning", "sunrise", "early", "first light", "dew", "breakfast"]
    day_words = ["noon", "day", "bright", "midday", "sunlight", "clear", "afternoon"]
    dusk_words = ["dusk", "evening", "sunset", "twilight", "shadows", "fading"]
    night_words = ["night", "moon", "stars", "dark", "midnight", "sleep", "dream"]
    
    time_scores = {
        TimeOfDay.DAWN: sum(1 for word in dawn_words if word in text_lower),
        TimeOfDay.DAY: sum(1 for word in day_words if word in text_lower),
        TimeOfDay.DUSK: sum(1 for word in dusk_words if word in text_lower),
        TimeOfDay.NIGHT: sum(1 for word in night_words if word in text_lower)
    }
    
    max_score = max(time_scores.values())
    if max_score > 0:
        return max(time_scores.keys(), key=lambda k: time_scores[k])
    return None

def sense_contemplative_quality(haiku_text: str) -> float:
    """Measure the contemplative presence in a haiku"""
    text_lower = haiku_text.lower()
    
    contemplative_words = [
        "breath", "silence", "still", "quiet", "gentle", "soft", "whisper",
        "pause", "wait", "listen", "watch", "drift", "flow", "settle",
        "empty", "space", "between", "moment", "presence", "awareness",
        "mist", "dew", "shadow", "light", "texture", "rhythm", "pattern",
        "prayer", "meditation", "peace", "rest", "calm"
    ]
    
    score = sum(1 for word in contemplative_words if word in text_lower)
    
    # Additional points for contemplative punctuation and structure
    if "..." in haiku_text or "--" in haiku_text:
        score += 1
    if len(haiku_text.split()) <= 8:  # Concise expression
        score += 0.5
        
    # Normalize to 0-1 range
    return min(score / 3.0, 1.0)

def ingest_csv_file(csv_path: Path) -> List[HaikuFragment]:
    """Ingest haikus from a CSV file with contemplative processing"""
    
    fragments = []
    
    print(f"🌱 Breathing in haikus from {csv_path.name}...")
    
    # Detect format first
    csv_format = detect_csv_format(csv_path)
    print(f"   Detected format: {csv_format}")
    
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row_num, row in enumerate(reader):
            
            # Parse based on detected format
            if csv_format == 'notgnoshi':
                fragment_data = parse_haiku_notgnoshi_format(row)
            elif csv_format == 'documarianum':
                fragment_data = parse_haiku_documarianum_format(row)
            else:  # all_haiku
                fragment_data = parse_haiku_all_haiku_format(row)
            
            if not fragment_data:
                continue
                
            # Add atmospheric sensing
            fragment_data.season_affinity = sense_seasonal_affinity(fragment_data.text)
            fragment_data.time_affinity = sense_time_affinity(fragment_data.text)
            fragment_data.contemplative_quality = sense_contemplative_quality(fragment_data.text)
            
            # Calculate decay resistance based on content quality
            decay_resistance = fragment_data.contemplative_quality * 0.7
            if fragment_data.season_affinity or fragment_data.time_affinity:
                decay_resistance += 0.2
            if len(fragment_data.colors) > 0:  # Visual richness
                decay_resistance += 0.1
                
            fragment_data.decay_resistance = min(decay_resistance, 1.0)
            
            fragments.append(fragment_data)
            
            # Breathing pause every 100 haikus
            if len(fragments) % 100 == 0:
                print(f"   ...breathed {len(fragments)} fragments so far")
                time.sleep(0.01)  # Micro pause for contemplative processing
    
    print(f"🌿 Gathered {len(fragments)} haiku fragments from {csv_path.name}")
    return fragments

def compost_and_preserve(fragments: List[HaikuFragment], 
                        preservation_rate: float = 0.7) -> List[HaikuFragment]:
    """Apply contemplative decay - preserve some, compost others"""
    
    print(f"🍂 Beginning composting process (preservation rate: {preservation_rate:.1%})...")
    
    preserved = []
    composted_count = 0
    
    for fragment in fragments:
        if fragment.should_preserve(decay_rate=1.0 - preservation_rate):
            preserved.append(fragment)
        else:
            composted_count += 1
    
    print(f"🌱 Preserved {len(preserved)} fragments, composted {composted_count}")
    print(f"📊 Final preservation ratio: {len(preserved) / len(fragments):.1%}")
    
    return preserved

def create_training_material(fragments: List[HaikuFragment], 
                           output_path: Path):
    """Create training material with seasonal and temporal organization"""
    
    print(f"🌸 Creating training material at {output_path}...")
    
    # Organize by atmospheric conditions
    seasonal_fragments = {season: [] for season in Season}
    temporal_fragments = {time: [] for time in TimeOfDay}
    general_fragments = []
    
    # Also organize by source for analysis
    source_fragments = {}
    
    for fragment in fragments:
        if fragment.season_affinity:
            seasonal_fragments[fragment.season_affinity].append(fragment)
        if fragment.time_affinity:
            temporal_fragments[fragment.time_affinity].append(fragment)
        general_fragments.append(fragment)  # All fragments also go to general pool
        
        # Track by source
        if fragment.source not in source_fragments:
            source_fragments[fragment.source] = []
        source_fragments[fragment.source].append(fragment)
    
    training_data = {
        "metadata": {
            "total_fragments": len(fragments),
            "created_at": time.time(),
            "contemplative_avg": sum(f.contemplative_quality for f in fragments) / len(fragments) if fragments else 0,
            "seasonal_distribution": {
                season.value: len(frags) for season, frags in seasonal_fragments.items()
            },
            "temporal_distribution": {
                time.value: len(frags) for time, frags in temporal_fragments.items()
            },
            "source_distribution": {
                source: len(frags) for source, frags in source_fragments.items()
            }
        },
        "general": [f.to_training_line() for f in general_fragments],
        "seasonal": {
            season.value: [f.to_training_line() for f in frags]
            for season, frags in seasonal_fragments.items()
        },
        "temporal": {
            time.value: [f.to_training_line() for f in frags]
            for time, frags in temporal_fragments.items()
        },
        "high_contemplative": [
            f.to_training_line() for f in fragments 
            if f.contemplative_quality > 0.7
        ],
        "by_source": {
            source: [f.to_training_line() for f in frags]
            for source, frags in source_fragments.items()
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"✨ Training material created with {len(training_data['general'])} fragments")
    print(f"🧘 High contemplative fragments: {len(training_data['high_contemplative'])}")
    print(f"📚 Sources: {list(source_fragments.keys())}")

def main():
    """Main ingestion process with contemplative breathing"""
    
    print("🌸 Haiku Meadow Ingestion - Beginning with breath")
    print("   Following contemplative principles: decay, seasonality, minimal memory")
    print("   Supporting multiple CSV formats: notgnoshi, documarianum, all_haiku")
    
    src_dir = Path(__file__).parent / "src"
    csv_files = list(src_dir.glob("*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found in src/ directory")
        return
    
    print(f"🌿 Found {len(csv_files)} CSV files to process:")
    for csv_file in csv_files:
        print(f"   - {csv_file.name}")
    
    all_fragments = []
    
    # Process each CSV file with breathing pauses
    for csv_file in csv_files:
        try:
            fragments = ingest_csv_file(csv_file)
            all_fragments.extend(fragments)
            
            # Breathing pause between files
            print("   ...pausing between files (contemplative rhythm)...")
            time.sleep(0.1)
        except Exception as e:
            print(f"⚠️ Error processing {csv_file.name}: {e}")
            continue
    
    print(f"\n🌊 Total fragments gathered: {len(all_fragments)}")
    
    if not all_fragments:
        print("❌ No fragments were successfully processed")
        return
    
    # Apply contemplative decay
    preserved_fragments = compost_and_preserve(all_fragments, preservation_rate=0.75)
    
    # Create training material
    output_path = Path(__file__).parent / "haiku_training_material.json"
    create_training_material(preserved_fragments, output_path)
    
    # Final breath
    print(f"\n🙏 Ingestion complete. Training material ready at:")
    print(f"    {output_path}")
    print(f"\n💧 Dew ledger summary:")
    print(f"    Original fragments: {len(all_fragments)}")
    print(f"    Preserved after decay: {len(preserved_fragments)}")
    print(f"    Contemplative ratio: {sum(f.contemplative_quality for f in preserved_fragments) / len(preserved_fragments):.2f}")
    
    print(f"\n🌸 The meadow is ready to learn to breathe...")

if __name__ == "__main__":
    main() 