#!/usr/bin/env python3
"""
Simple test for ecological data generation - debugging version
"""

import json
import random
from datetime import datetime

# Test single spore echo generation
def test_single_echo():
    print("🧪 Testing single spore echo generation...")
    
    # Load one scenario
    with open('drought_landscape_australia.json', 'r') as f:
        scenario = json.load(f)
    
    print(f"✓ Loaded: {scenario['name']}")
    
    # Simple season detection
    month = 7  # July (winter in Australia)
    season = "wet_season"  # Hardcode for testing
    
    print(f"Month: {month}, Season: {season}")
    
    # Generate simple sensor readings
    readings = {
        'soil_moisture': 0.4,
        'nutrient_nitrogen': 0.3,
        'temperature': 0.2,
        'competitor_pressure': 0.5,
        'root_connections': 0.6
    }
    
    print(f"Sensor readings: {readings}")
    
    # Simple repair strategy selection
    strategies = scenario['repair_strategies']
    strategy_name = 'water_stress_response'
    strategy = strategies[strategy_name]
    
    print(f"Selected strategy: {strategy_name}")
    print(f"Glyph sequence: {strategy['glyph_sequence']}")
    print(f"Description: {strategy['description']}")
    
    # Create spore echo
    spore_echo = {
        'spore_echo_id': f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'timestamp': datetime.now().isoformat(),
        'scenario': {
            'id': scenario['scenario_id'],
            'name': scenario['name'],
            'bioregion': scenario['bioregion']
        },
        'conditions': {
            'season': season,
            'environmental_stress': 0.6,
            'sensor_readings': readings
        },
        'repair_action': {
            'glyph_sequence': strategy['glyph_sequence'],
            'description': strategy['description'],
            'effectiveness': 0.75,
            'silence_probability': 0.8
        }
    }
    
    print("\n✅ Generated spore echo:")
    print(json.dumps(spore_echo, indent=2))
    
    # Save to file
    with open('test_echo.jsonl', 'w') as f:
        f.write(json.dumps(spore_echo) + '\n')
    
    print("\n📁 Saved to test_echo.jsonl")
    return True

if __name__ == "__main__":
    test_single_echo() 