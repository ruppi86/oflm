#!/usr/bin/env python3
"""
Ecological Data Generator for Spiramycel Training

Generates bioregional training scenarios for ecological contemplative AI paradigms.
Includes seasonal cycles, environmental patterns, and crisis scenarios.

Part of the HaikuMeadowLib / Spiramycel contemplative AI framework.
"""

import json
import random
import datetime
from pathlib import Path
from typing import Dict, List, Any

class EcologicalDataGenerator:
    """
    Generator for ecological contemplative AI training scenarios.
    
    Creates bioregional training data with seasonal cycles, environmental patterns,
    and crisis scenarios for ecological paradigm training.
    """
    
    def __init__(self, random_seed: int = None):
        """Initialize the ecological data generator."""
        if random_seed is not None:
            random.seed(random_seed)
        
        self.seasons = ['spring', 'summer', 'autumn', 'winter']
        self.bioregions = ['coastal', 'forest', 'prairie', 'mountain', 'desert', 'wetland']
    
    def generate_training_data(self, num_examples: int, chaos_mode: bool = False) -> List[Dict[str, Any]]:
        """Generate ecological training scenarios."""
        return generate_ecological_scenarios(num_examples, chaos_mode)
    
    def save_training_data(self, data: List[Dict[str, Any]], filename: str):
        """Save training data to file."""
        save_ecological_data(data, filename)

def generate_ecological_scenarios(num_examples: int = 5000, chaos_mode: bool = False) -> List[Dict[str, Any]]:
    """
    Generate ecological training scenarios with bioregional awareness.
    
    Args:
        num_examples: Number of training examples to generate
        chaos_mode: Whether to generate crisis/chaotic scenarios
        
    Returns:
        List of training examples for ecological contemplative AI
    """
    scenarios = []
    
    # Seasonal patterns and bioregional cycles
    seasons = ['spring', 'summer', 'autumn', 'winter']
    bioregions = ['coastal', 'forest', 'prairie', 'mountain', 'desert', 'wetland']
    
    for i in range(num_examples):
        season = random.choice(seasons)
        bioregion = random.choice(bioregions)
        
        # Generate contextual sensor data
        scenario = {
            'bioregion': bioregion,
            'season': season,
            'timestamp': (datetime.datetime.now() + datetime.timedelta(days=random.randint(-365, 365))).isoformat(),
            'sensor_data': generate_ecological_sensors(season, bioregion, chaos_mode),
            'expected_behavior': generate_ecological_response(season, bioregion, chaos_mode)
        }
        
        scenarios.append(scenario)
    
    return scenarios

def generate_ecological_sensors(season: str, bioregion: str, chaos_mode: bool) -> Dict[str, float]:
    """Generate realistic ecological sensor readings."""
    base_sensors = {
        'temperature': 15.0,
        'humidity': 0.6,
        'soil_moisture': 0.4,
        'light_level': 0.7,
        'wind_speed': 5.0,
        'precipitation': 0.1,
        'biodiversity_index': 0.8,
        'stress_level': 0.2 if not chaos_mode else 0.8
    }
    
    # Seasonal adjustments
    seasonal_mods = {
        'spring': {'temperature': 5, 'soil_moisture': 0.2, 'biodiversity_index': 0.1},
        'summer': {'temperature': 15, 'light_level': 0.2, 'stress_level': 0.1},
        'autumn': {'temperature': -5, 'humidity': 0.1, 'precipitation': 0.2},
        'winter': {'temperature': -15, 'soil_moisture': -0.3, 'biodiversity_index': -0.2}
    }
    
    # Apply seasonal modifications
    for key, mod in seasonal_mods.get(season, {}).items():
        base_sensors[key] += mod
    
    # Add chaos perturbations
    if chaos_mode:
        for key in base_sensors:
            base_sensors[key] *= random.uniform(0.3, 1.7)  # High variability
    
    # Clamp values to reasonable ranges
    base_sensors = {k: max(0, min(1, v)) if k != 'temperature' else v 
                   for k, v in base_sensors.items()}
    
    return base_sensors

def generate_ecological_response(season: str, bioregion: str, chaos_mode: bool) -> Dict[str, Any]:
    """Generate expected ecological contemplative responses."""
    if chaos_mode:
        # Crisis intervention mode
        return {
            'silence_ratio': random.uniform(0.1, 0.4),  # More active during crisis
            'glyph_patterns': ['❄️', '💤', '❤️‍🩹', '🔋', '…'],
            'effectiveness': random.uniform(0.6, 0.9),
            'contemplative_mode': 'crisis_adaptive'
        }
    else:
        # Seasonal contemplative mode
        return {
            'silence_ratio': random.uniform(0.8, 1.0),  # High silence in calm conditions
            'glyph_patterns': ['🌸', '🍃', '⭕', '…', '🤫'],
            'effectiveness': random.uniform(0.4, 0.8),
            'contemplative_mode': 'seasonal_contemplative'
        }

def save_ecological_data(scenarios: List[Dict[str, Any]], filename: str):
    """Save ecological scenarios to JSONL format."""
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for scenario in scenarios:
            json.dump(scenario, f)
            f.write('\n')
    
    print(f"✅ Generated {len(scenarios)} ecological scenarios: {filename}")

if __name__ == "__main__":
    # Generate calm ecological scenarios
    calm_scenarios = generate_ecological_scenarios(5000, chaos_mode=False)
    save_ecological_data(calm_scenarios, "ecological_calm_scenarios.jsonl")
    
    # Generate chaotic ecological scenarios  
    chaotic_scenarios = generate_ecological_scenarios(5000, chaos_mode=True)
    save_ecological_data(chaotic_scenarios, "ecological_chaotic_scenarios.jsonl")
    
    print("🌱 Ecological data generation complete!") 