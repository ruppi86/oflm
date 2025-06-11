#!/usr/bin/env python3
"""
Ecological Data Generator for Spiramycel Training

Generates realistic spore echoes based on actual ecological scenarios:
- Drought-stressed eucalyptus forest (Australia)
- Rice paddy ecosystem (Guangzhou, China) 
- Groundwater monitoring (Sweden)

Each scenario includes multi-generational patterns, seasonal cycles,
and real bioregional adaptation strategies.
"""

import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

@dataclass
class NetworkConditions:
    """Represents current network conditions for spore echo generation"""
    scenario_id: str
    timestamp: datetime
    season: str
    year_in_cycle: int
    sensor_readings: Dict[str, float]
    environmental_stress: float
    repair_urgency: float
    historical_context: str

class EcologicalDataGenerator:
    """Generates realistic ecological training data for Spiramycel"""
    
    def __init__(self, scenarios_dir: str = None):
        """Initialize with scenario definitions"""
        if scenarios_dir is None:
            scenarios_dir = Path(__file__).parent
        else:
            scenarios_dir = Path(scenarios_dir)
            
        self.scenarios_dir = scenarios_dir
        self.scenarios = {}
        self.load_scenarios()
        
        # Multi-generational time tracking
        self.base_year = 2024
        self.simulation_years = 50  # 50-year simulation
        
    def load_scenarios(self):
        """Load all scenario JSON files"""
        scenario_files = [
            "drought_landscape_australia.json",
            "rice_paddy_guangzhou.json", 
            "groundwater_sweden.json"
        ]
        
        for scenario_file in scenario_files:
            try:
                with open(self.scenarios_dir / scenario_file, 'r', encoding='utf-8') as f:
                    scenario = json.load(f)
                    self.scenarios[scenario['scenario_id']] = scenario
                    print(f"✓ Loaded scenario: {scenario['name']}")
            except Exception as e:
                print(f"⚠ Warning: Could not load {scenario_file}: {e}")
    
    def get_season_for_month(self, scenario: Dict, month: int) -> str:
        """Determine which season a month belongs to in the scenario"""
        for season_name, season_data in scenario['seasonal_cycles'].items():
            if season_name == "transition_periods":
                # Handle nested transition periods
                if isinstance(season_data, dict):
                    for sub_season, sub_data in season_data.items():
                        if 'months' in sub_data and isinstance(sub_data['months'], list):
                            if month in sub_data['months']:
                                return sub_season  # Return "autumn" or "spring"
            else:
                # Handle direct seasons
                if 'months' in season_data and isinstance(season_data['months'], list):
                    if month in season_data['months']:
                        return season_name
        return "transition"
    
    def simulate_sensor_readings(self, scenario: Dict, season: str, year_in_cycle: int, 
                                extreme_event: str = None) -> Dict[str, float]:
        """Generate realistic sensor readings based on scenario and conditions"""
        readings = {}
        sensor_mappings = scenario['sensor_mappings']
        season_data = scenario['seasonal_cycles'].get(season, {})
        
        for sensor_type, ranges in sensor_mappings.items():
            # Base reading from seasonal patterns
            if season_data:
                base_reading = self._get_seasonal_baseline(sensor_type, season_data)
            else:
                base_reading = 0.5  # neutral default
            
            # Apply multi-generational patterns
            reading = self._apply_generational_patterns(
                base_reading, scenario, sensor_type, year_in_cycle
            )
            
            # Apply extreme events
            if extreme_event:
                reading = self._apply_extreme_event(reading, extreme_event, sensor_type)
            
            # Add natural variation
            reading += random.gauss(0, 0.05)  # 5% standard deviation
            readings[sensor_type] = max(0.0, min(1.0, reading))
            
        return readings
    
    def _get_seasonal_baseline(self, sensor_type: str, season_data: Dict) -> float:
        """Get seasonal baseline for sensor type"""
        # Map season descriptions to sensor readings
        baselines = {
            'soil_moisture': {
                'high': 0.7, 'moderate': 0.5, 'low': 0.3, 'very_low': 0.15,
                'flooded': 0.9, 'drained': 0.2
            },
            'water_level': {
                'flooded': 0.8, 'high': 0.7, 'normal': 0.5, 'low': 0.3, 'drought': 0.1
            },
            'nutrient_nitrogen': {
                'abundant': 0.8, 'high': 0.7, 'moderate': 0.5, 'low': 0.3, 'very_low': 0.2
            },
            'temperature': {
                'optimal': 0.6, 'cold': 0.2, 'warm': 0.8, 'frozen': 0.0, 'hot': 1.0
            }
        }
        
        # Extract activity/condition indicators from season data
        for key, value in season_data.items():
            if sensor_type in baselines and value in baselines[sensor_type]:
                return baselines[sensor_type][value]
        
        return 0.5  # default neutral
    
    def _apply_generational_patterns(self, base_reading: float, scenario: Dict, 
                                   sensor_type: str, year_in_cycle: int) -> float:
        """Apply multi-generational cyclical patterns"""
        patterns = scenario.get('multi_generational_patterns', {})
        reading = base_reading
        
        try:
            # El Niño/La Niña effects (Australia)
            if 'el_nino_years' in patterns:
                el_nino_years = patterns['el_nino_years'].get('frequency_years', [])
                if isinstance(el_nino_years, list) and year_in_cycle in el_nino_years:
                    if sensor_type == 'soil_moisture':
                        reading *= patterns['el_nino_years'].get('rainfall_reduction', 0.6)
                    elif sensor_type == 'temperature':
                        reading += 0.2  # increased heat stress
            
            if 'la_nina_years' in patterns:
                la_nina_years = patterns['la_nina_years'].get('frequency_years', [])
                if isinstance(la_nina_years, list) and year_in_cycle in la_nina_years:
                    if sensor_type == 'soil_moisture':
                        reading *= patterns['la_nina_years'].get('rainfall_increase', 1.4)
                    elif sensor_type == 'nutrient_nitrogen':
                        reading += 0.1  # growth opportunities
            
            # Flood cycles (Rice paddy)
            if 'flood_cycles' in patterns:
                major_flood_years = patterns['flood_cycles'].get('major_flood_years', [])
                if isinstance(major_flood_years, list) and year_in_cycle in major_flood_years:
                    if sensor_type == 'water_level':
                        reading = min(1.0, reading + 0.3)
                    elif sensor_type == 'methane_production':
                        reading = min(1.0, reading + 0.2)
            
            # Climate oscillations (Sweden)
            if 'climate_oscillations' in patterns:
                nao_positive = patterns['climate_oscillations'].get('nao_positive_years', [])
                if isinstance(nao_positive, list) and year_in_cycle in nao_positive:
                    if sensor_type == 'temperature':
                        reading += 0.1  # warmer
                    elif sensor_type == 'water_table_level':
                        reading += 0.05  # more precipitation
        
        except (KeyError, TypeError) as e:
            # If there's any error in pattern application, just return base reading
            pass
            
        return reading
    
    def _apply_extreme_event(self, reading: float, event: str, sensor_type: str) -> float:
        """Apply extreme weather/environmental events"""
        if event == "drought":
            if sensor_type in ['soil_moisture', 'water_level', 'water_table_level']:
                return reading * 0.3  # severe reduction
            elif sensor_type == 'temperature':
                return min(1.0, reading + 0.3)  # heat stress
        
        elif event == "flood":
            if sensor_type in ['water_level', 'water_table_level']:
                return min(1.0, reading + 0.4)
            elif sensor_type == 'nitrate_concentration':
                return min(1.0, reading + 0.2)  # runoff contamination
        
        elif event == "fire":
            if sensor_type == 'temperature':
                return 1.0  # extreme heat
            elif sensor_type in ['soil_moisture', 'root_connections']:
                return reading * 0.1  # severe damage
        
        elif event == "contamination_event":
            if sensor_type in ['nitrate_concentration', 'heavy_metal_levels']:
                return min(1.0, reading + 0.4)
            elif sensor_type == 'bacterial_activity':
                return reading * 0.6  # stress response
        
        return reading
    
    def select_repair_strategy(self, scenario: Dict, conditions: NetworkConditions) -> Tuple[List[int], str, float]:
        """Select appropriate repair strategy based on conditions"""
        strategies = scenario['repair_strategies']
        
        # Evaluate conditions for each strategy
        viable_strategies = []
        for strategy_name, strategy_data in strategies.items():
            if self._evaluate_strategy_conditions(strategy_data['conditions'], conditions):
                effectiveness = random.uniform(*strategy_data['effectiveness_range'])
                viable_strategies.append((strategy_name, strategy_data, effectiveness))
        
        if not viable_strategies:
            # Fallback to most general strategy
            strategy_name = list(strategies.keys())[0]
            strategy_data = strategies[strategy_name]
            effectiveness = random.uniform(*strategy_data['effectiveness_range']) * 0.5
            viable_strategies = [(strategy_name, strategy_data, effectiveness)]
        
        # Select best strategy (highest effectiveness)
        best_strategy = max(viable_strategies, key=lambda x: x[2])
        strategy_name, strategy_data, effectiveness = best_strategy
        
        return (strategy_data['glyph_sequence'], 
                strategy_data['description'], 
                effectiveness)
    
    def _evaluate_strategy_conditions(self, conditions_str: str, conditions: NetworkConditions) -> bool:
        """Evaluate if strategy conditions are met"""
        # Simple condition evaluation - in practice this would be more sophisticated
        readings = conditions.sensor_readings
        
        if "soil_moisture < 0.15" in conditions_str:
            return readings.get('soil_moisture', 0.5) < 0.15
        elif "water_level > 0.8" in conditions_str:
            return readings.get('water_level', 0.5) > 0.8
        elif "nitrate_elevated" in conditions_str:
            return readings.get('nitrate_concentration', 0.3) > 0.5
        elif "contamination_detected" in conditions_str:
            return (readings.get('heavy_metal_levels', 0.2) > 0.6 or 
                   readings.get('nitrate_concentration', 0.3) > 0.7)
        elif "winter_conditions" in conditions_str or "temperature < 0.2" in conditions_str:
            return readings.get('temperature', 0.5) < 0.2
        elif "wet_season" in conditions_str:
            return conditions.season in ['wet_season', 'spring_snowmelt', 'early_rice_season']
        elif "bacterial_balance > 0.6" in conditions_str:
            return readings.get('bacterial_balance', 0.5) > 0.6
        
        return True  # Default to allowing strategy
    
    def generate_spore_echo(self, scenario_id: str, timestamp: datetime = None, 
                          extreme_event: str = None) -> Dict[str, Any]:
        """Generate a single realistic spore echo"""
        if scenario_id not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_id}")
        
        scenario = self.scenarios[scenario_id]
        
        if timestamp is None:
            # Random time within simulation period
            days_offset = random.randint(0, self.simulation_years * 365)
            timestamp = datetime(self.base_year, 1, 1) + timedelta(days=days_offset)
        
        # Determine seasonal and cyclical context
        month = timestamp.month
        year_in_cycle = (timestamp.year - self.base_year) % 30  # 30-year major cycle
        season = self.get_season_for_month(scenario, month)
        
        # Generate sensor readings
        sensor_readings = self.simulate_sensor_readings(
            scenario, season, year_in_cycle, extreme_event
        )
        
        # Calculate environmental stress
        stress_indicators = ['soil_moisture', 'water_level', 'nitrate_concentration', 
                           'heavy_metal_levels', 'temperature']
        stress_values = []
        for indicator in stress_indicators:
            if indicator in sensor_readings:
                # Convert readings to stress (0.5 is optimal, deviations increase stress)
                stress = abs(sensor_readings[indicator] - 0.5) * 2
                stress_values.append(stress)
        
        environmental_stress = np.mean(stress_values) if stress_values else 0.3
        repair_urgency = min(1.0, environmental_stress + random.uniform(0, 0.2))
        
        # Create network conditions
        conditions = NetworkConditions(
            scenario_id=scenario_id,
            timestamp=timestamp,
            season=season,
            year_in_cycle=year_in_cycle,
            sensor_readings=sensor_readings,
            environmental_stress=environmental_stress,
            repair_urgency=repair_urgency,
            historical_context=f"{scenario['name']} - {season} - year {year_in_cycle}"
        )
        
        # Select repair strategy
        glyph_sequence, description, effectiveness = self.select_repair_strategy(scenario, conditions)
        
        # Calculate Tystnadsmajoritet (silence probability)
        silence_probability = max(0.1, 0.875 - environmental_stress * 0.3)
        
        return {
            'spore_echo_id': f"{scenario_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
            'timestamp': timestamp.isoformat(),
            'scenario': {
                'id': scenario_id,
                'name': scenario['name'],
                'bioregion': scenario['bioregion'],
                'ecosystem_type': scenario['ecosystem_type']
            },
            'conditions': {
                'season': season,
                'year_in_cycle': year_in_cycle,
                'environmental_stress': environmental_stress,
                'repair_urgency': repair_urgency,
                'extreme_event': extreme_event,
                'sensor_readings': sensor_readings
            },
            'repair_action': {
                'glyph_sequence': glyph_sequence,
                'description': description,
                'effectiveness': effectiveness,
                'silence_probability': silence_probability
            },
            'historical_context': conditions.historical_context,
            'multi_generational_wisdom': {
                'pattern_recognition_confidence': min(1.0, year_in_cycle / 20),
                'adaptation_strength': environmental_stress * effectiveness,
                'bioregional_alignment': effectiveness * 0.8 + 0.2
            }
        }
    
    def generate_training_dataset(self, num_echoes: int = 1000, 
                                output_file: str = "ecological_training_data.jsonl") -> str:
        """Generate a complete training dataset"""
        print(f"🌱 Generating {num_echoes} ecological spore echoes...")
        
        output_path = self.scenarios_dir / output_file
        scenario_ids = list(self.scenarios.keys())
        
        if not scenario_ids:
            raise ValueError("No scenarios loaded!")
        
        extreme_events = [None, "drought", "flood", "fire", "contamination_event"]
        extreme_probability = 0.15  # 15% chance of extreme events
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i in range(num_echoes):
                # Distribute across scenarios
                scenario_id = random.choice(scenario_ids)
                
                # Occasional extreme events
                extreme_event = None
                if random.random() < extreme_probability:
                    extreme_event = random.choice(extreme_events[1:])  # exclude None
                
                try:
                    spore_echo = self.generate_spore_echo(scenario_id, extreme_event=extreme_event)
                    f.write(json.dumps(spore_echo) + '\n')
                    
                    if (i + 1) % 100 == 0:
                        print(f"  Generated {i + 1}/{num_echoes} spore echoes...")
                        
                except Exception as e:
                    print(f"⚠ Warning: Failed to generate echo {i}: {e}")
                    continue
        
        print(f"✓ Generated {num_echoes} spore echoes")
        print(f"📁 Saved to: {output_path}")
        
        # Statistics
        self.print_dataset_statistics(output_path)
        
        return str(output_path)
    
    def print_dataset_statistics(self, dataset_path: str):
        """Print statistics about the generated dataset"""
        with open(dataset_path, 'r') as f:
            echoes = [json.loads(line) for line in f if line.strip()]
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total spore echoes: {len(echoes)}")
        
        if len(echoes) == 0:
            print("   ⚠ No valid spore echoes generated!")
            return
        
        # Scenario distribution
        scenario_counts = {}
        for echo in echoes:
            scenario = echo['scenario']['id']
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        
        print(f"   Scenario distribution:")
        for scenario, count in scenario_counts.items():
            percentage = (count / len(echoes)) * 100
            print(f"     {scenario}: {count} ({percentage:.1f}%)")
        
        # Effectiveness distribution
        effectiveness_values = [echo['repair_action']['effectiveness'] for echo in echoes]
        avg_effectiveness = np.mean(effectiveness_values) if effectiveness_values else 0.0
        print(f"   Average repair effectiveness: {avg_effectiveness:.3f}")
        
        # Environmental stress distribution
        stress_values = [echo['conditions']['environmental_stress'] for echo in echoes]
        avg_stress = np.mean(stress_values) if stress_values else 0.0
        print(f"   Average environmental stress: {avg_stress:.3f}")
        
        # Extreme events
        extreme_count = sum(1 for echo in echoes if echo['conditions'].get('extreme_event'))
        extreme_pct = (extreme_count / len(echoes) * 100) if len(echoes) > 0 else 0.0
        print(f"   Extreme events: {extreme_count} ({extreme_pct:.1f}%)")

def main():
    """Main function to generate ecological training data"""
    generator = EcologicalDataGenerator()
    
    print("🌍 Ecological Spiramycel Training Data Generator")
    print("=" * 50)
    
    # Generate different dataset sizes
    datasets = [
        (500, "ecological_small.jsonl"),
        (2000, "ecological_medium.jsonl"),
        (5000, "ecological_large.jsonl")
    ]
    
    for num_echoes, filename in datasets:
        print(f"\n🎯 Generating {filename}...")
        generator.generate_training_dataset(num_echoes, filename)
    
    print("\n✅ All ecological datasets generated!")
    print("\nNext steps:")
    print("  1. Review generated data: cat training_scenarios/ecological_*.jsonl | head -5")
    print("  2. Train Spiramycel: python serious_training.py --ecological-data")
    print("  3. Compare performance vs abstract scenarios")

if __name__ == "__main__":
    main() 