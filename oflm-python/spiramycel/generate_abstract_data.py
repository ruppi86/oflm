#!/usr/bin/env python3
"""
Abstract Data Generator for Spiramycel Training

Pre-generates abstract training data to eliminate the performance bottleneck
where abstract training was generating data during training time.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

# Import existing components
from glyph_codec import SpiramycelGlyphCodec, Season
from neural_trainer import SporeMapLedger, SpiramycelTrainer

class AbstractDataGenerator:
    """Generates abstract training data for Spiramycel (pre-processed)"""
    
    def __init__(self):
        self.codec = SpiramycelGlyphCodec()
        
    def generate_training_dataset(self, num_echoes: int = 5000, 
                                output_file: str = "abstract_training_data.jsonl",
                                chaos_mode: bool = False) -> str:
        """Generate a complete abstract training dataset
        
        This extracts the data generation logic from SpiramycelTrainer.create_enhanced_training_data()
        and saves it to a JSONL file for fast loading during training.
        """
        
        print(f"🔬 Generating {num_echoes} abstract spore echoes...")
        if chaos_mode:
            print("⚡ Chaos mode: HIGH stress environment (more problems)")
        else:
            print("🧘 Calm mode: LOW stress environment (more optimal conditions)")
        
        output_path = Path("training_scenarios") / output_file
        output_path.parent.mkdir(exist_ok=True)
        
        # Abstract scenarios (extracted from neural_trainer.py)
        abstract_scenarios = {
            "urban_fiber": {
                "name": "Urban Fiber Network Infrastructure",
                "description": "High-bandwidth metro networks with thermal and congestion challenges",
                "problem_types": {
                    "thermal_overload": {"sensor_ranges": {"temperature": (35, 65)}, "repair_glyphs": [0x17, 0x03, 0x16], "effectiveness": (0.4, 0.8)},
                    "bandwidth_saturation": {"sensor_ranges": {"bandwidth": (0.0, 0.3)}, "repair_glyphs": [0x01, 0x03, 0x05], "effectiveness": (0.5, 0.85)},
                    "power_grid_fluctuation": {"sensor_ranges": {"voltage": (2.8, 3.8)}, "repair_glyphs": [0x11, 0x12, 0x16], "effectiveness": (0.6, 0.9)},
                    "optimal_conditions": {"repair_glyphs": [0x31, 0x32, 0x37], "effectiveness": (0.05, 0.3)}
                },
                "bioregions": ["downtown_core", "residential_fiber", "business_district", "metro_junction", "data_center"],
                "seasonal_patterns": {"summer": "thermal_stress", "winter": "stable_cool", "spring": "moderate", "autumn": "optimal"}
            },
            
            "satellite_remote": {
                "name": "Satellite & Remote Network",
                "description": "Long-distance wireless with latency, power, and weather challenges", 
                "problem_types": {
                    "signal_degradation": {"sensor_ranges": {"error_rate": (0.05, 0.4)}, "repair_glyphs": [0x05, 0x03, 0x04], "effectiveness": (0.6, 0.85)},
                    "power_constraints": {"sensor_ranges": {"voltage": (2.0, 2.8)}, "repair_glyphs": [0x12, 0x14, 0x18], "effectiveness": (0.4, 0.75)},
                    "weather_disruption": {"sensor_ranges": {"latency": (0.3, 1.0), "error_rate": (0.1, 0.5)}, "repair_glyphs": [0x01, 0x05, 0x25], "effectiveness": (0.3, 0.7)},
                    "optimal_conditions": {"repair_glyphs": [0x33, 0x35, 0x38], "effectiveness": (0.1, 0.4)}
                },
                "bioregions": ["mountain_station", "island_node", "polar_research", "remote_relay", "satellite_ground"],
                "seasonal_patterns": {"summer": "solar_optimal", "winter": "power_limited", "spring": "weather_variable", "autumn": "stable"}
            },
            
            "industrial_iot": {
                "name": "Industrial IoT Network",
                "description": "Harsh industrial environments with thousands of devices and reliability demands",
                "problem_types": {
                    "electromagnetic_interference": {"sensor_ranges": {"error_rate": (0.08, 0.3)}, "repair_glyphs": [0x23, 0x24, 0x2F], "effectiveness": (0.5, 0.8)},
                    "device_overload": {"sensor_ranges": {"latency": (0.2, 0.8)}, "repair_glyphs": [0x25, 0x28, 0x22], "effectiveness": (0.6, 0.85)},
                    "environmental_stress": {"sensor_ranges": {"temperature": (25, 50), "voltage": (2.5, 3.2)}, "repair_glyphs": [0x17, 0x12, 0x24], "effectiveness": (0.4, 0.75)},
                    "optimal_conditions": {"repair_glyphs": [0x36, 0x39, 0x3F], "effectiveness": (0.08, 0.35)}
                },
                "bioregions": ["factory_floor", "refinery_network", "logistics_hub", "smart_city", "mining_operation"],
                "seasonal_patterns": {"summer": "heat_stress", "winter": "heating_costs", "spring": "maintenance_season", "autumn": "production_peak"}
            }
        }
        
        # Weight scenarios based on chaos_mode
        if chaos_mode:
            scenario_weights = [0.4, 0.4, 0.4]  # Equal weight to all 3 scenarios
            problem_vs_optimal_ratio = 0.7  # 70% problems
        else:
            scenario_weights = [0.33, 0.33, 0.34]  # Equal weight to all 3 scenarios
            problem_vs_optimal_ratio = 0.4  # 40% problems
        
        scenario_names = list(abstract_scenarios.keys())
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i in range(num_echoes):
                # Select one of the 3 scenarios
                scenario_name = random.choices(scenario_names, weights=scenario_weights, k=1)[0]
                scenario = abstract_scenarios[scenario_name]
                
                # Select bioregion within this scenario
                bioregion = random.choice(scenario["bioregions"])
                
                # Choose problem type vs optimal based on chaos_mode
                problem_types = list(scenario["problem_types"].keys())
                optimal_types = [pt for pt in problem_types if "optimal" in pt]
                problem_types_only = [pt for pt in problem_types if "optimal" not in pt]
                
                if random.random() < problem_vs_optimal_ratio:
                    problem_type_name = random.choice(problem_types_only)
                else:
                    problem_type_name = random.choice(optimal_types)
                
                problem_type = scenario["problem_types"][problem_type_name]
                
                # Generate sensor conditions
                sensor_readings = {"latency": 0.1, "voltage": 3.3, "temperature": 25.0, "error_rate": 0.02, "bandwidth": 0.8}
                
                if "sensor_ranges" in problem_type:
                    for sensor, (min_val, max_val) in problem_type["sensor_ranges"].items():
                        if sensor == "voltage":
                            sensor_readings[sensor] = random.uniform(min_val, max_val)
                        elif sensor == "temperature":
                            sensor_readings[sensor] = random.uniform(min_val, max_val)
                        elif sensor == "bandwidth":
                            sensor_readings[sensor] = random.uniform(min_val, max_val)
                        else:  # latency, error_rate
                            sensor_readings[sensor] = random.uniform(min_val, max_val)
                else:  # Optimal conditions
                    sensor_readings = {
                        "latency": random.uniform(0.05, 0.11),
                        "voltage": random.uniform(3.3, 3.4),
                        "temperature": random.uniform(22.0, 26.0),
                        "error_rate": random.uniform(0.0, 0.01),
                        "bandwidth": random.uniform(0.7, 0.9)
                    }
                
                # Select repair glyphs
                primary_glyphs = random.choices(problem_type["repair_glyphs"], k=random.randint(1, 3))
                
                # Add contemplative glyphs
                contemplative_glyphs = self.codec.get_contemplative_glyphs()
                
                if "optimal" in problem_type_name:
                    silence_count = random.randint(8, 12)  # Heavy silence
                else:
                    silence_count = random.randint(4, 8)   # Moderate silence
                
                contemplative_selection = random.choices(contemplative_glyphs, k=silence_count)
                glyph_sequence = primary_glyphs + contemplative_selection
                
                # Shuffle to mix repair and contemplative naturally
                random.shuffle(glyph_sequence)
                
                # Effectiveness with seasonal variation
                effectiveness = random.uniform(*problem_type["effectiveness"])
                season = random.choice(list(Season))
                
                seasonal_pattern = scenario["seasonal_patterns"].get(season.value.lower(), "moderate")
                if seasonal_pattern in ["thermal_stress", "heat_stress", "power_limited"]:
                    effectiveness *= random.uniform(0.8, 1.0)
                elif seasonal_pattern in ["optimal", "stable", "solar_optimal"]:
                    effectiveness *= random.uniform(1.0, 1.1)
                
                effectiveness = min(1.0, max(0.0, effectiveness))
                
                # Calculate environmental stress based on scenario and problem type
                if "optimal" in problem_type_name:
                    environmental_stress = random.uniform(0.1, 0.3)
                else:
                    environmental_stress = random.uniform(0.4, 0.8)
                
                # Create spore echo in same format as ecological data
                spore_echo = {
                    "scenario": {
                        "id": scenario_name,
                        "name": scenario["name"],
                        "description": scenario["description"],
                        "bioregion": bioregion,
                        "season": season.value
                    },
                    "conditions": {
                        "sensor_readings": sensor_readings,
                        "environmental_stress": environmental_stress,
                        "extreme_event": problem_type_name if "optimal" not in problem_type_name else None
                    },
                    "repair_action": {
                        "glyph_sequence": glyph_sequence,
                        "effectiveness": effectiveness,
                        "description": f"Abstract {problem_type_name} repair in {scenario_name} {bioregion}",
                        "silence_probability": len(contemplative_selection) / len(glyph_sequence) if glyph_sequence else 0.0
                    }
                }
                
                # Adjust for calm mode
                if not chaos_mode:
                    # Reduce environmental stress
                    spore_echo['conditions']['environmental_stress'] *= 0.6
                    
                    # Improve sensor readings toward optimal
                    for sensor, value in spore_echo['conditions']['sensor_readings'].items():
                        if sensor == "voltage":
                            optimal_value = 3.3
                        elif sensor == "temperature":
                            optimal_value = 25.0
                        elif sensor == "bandwidth":
                            optimal_value = 0.8
                        elif sensor == "latency":
                            optimal_value = 0.1
                        elif sensor == "error_rate":
                            optimal_value = 0.02
                        else:
                            optimal_value = value
                        
                        # Move 30% toward optimal
                        new_value = value + (optimal_value - value) * 0.3
                        spore_echo['conditions']['sensor_readings'][sensor] = new_value
                    
                    # Increase silence probability
                    spore_echo['repair_action']['silence_probability'] = min(1.0, 
                        spore_echo['repair_action']['silence_probability'] + 0.2)
                
                f.write(json.dumps(spore_echo) + '\n')
                
                if (i + 1) % 500 == 0:
                    print(f"  Generated {i + 1}/{num_echoes} abstract spore echoes...")
        
        print(f"✓ Generated {num_echoes} abstract spore echoes")
        print(f"📁 Saved to: {output_path}")
        
        # Statistics
        self.print_dataset_statistics(output_path)
        
        return str(output_path)
    
    def print_dataset_statistics(self, dataset_path: str):
        """Print statistics about the generated dataset"""
        with open(dataset_path, 'r') as f:
            echoes = [json.loads(line) for line in f if line.strip()]
        
        print(f"\n📊 Abstract Dataset Statistics:")
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
        avg_effectiveness = sum(effectiveness_values) / len(effectiveness_values) if effectiveness_values else 0.0
        print(f"   Average repair effectiveness: {avg_effectiveness:.3f}")
        
        # Environmental stress distribution
        stress_values = [echo['conditions']['environmental_stress'] for echo in echoes]
        avg_stress = sum(stress_values) / len(stress_values) if stress_values else 0.0
        print(f"   Average environmental stress: {avg_stress:.3f}")
        
        # Problem types vs optimal
        optimal_count = sum(1 for echo in echoes if echo['conditions']['extreme_event'] is None)
        problem_count = len(echoes) - optimal_count
        optimal_pct = (optimal_count / len(echoes) * 100) if len(echoes) > 0 else 0.0
        print(f"   Optimal conditions: {optimal_count} ({optimal_pct:.1f}%)")
        print(f"   Problem scenarios: {problem_count} ({100-optimal_pct:.1f}%)")
        
        # Silence analysis
        silence_values = [echo['repair_action']['silence_probability'] for echo in echoes]
        avg_silence = sum(silence_values) / len(silence_values) if silence_values else 0.0
        high_silence = sum(1 for s in silence_values if s > 0.8)
        print(f"   Average silence probability: {avg_silence:.3f}")
        print(f"   High silence (>0.8): {high_silence} ({high_silence/len(echoes)*100:.1f}%)")

def main():
    """Main function to generate abstract training data"""
    generator = AbstractDataGenerator()
    
    print("🔬 Abstract Spiramycel Training Data Generator")
    print("=" * 50)
    
    # Generate datasets for controlled comparison
    datasets = [
        # Small datasets for quick testing
        (500, "abstract_small_chaotic.jsonl", True),
        (500, "abstract_small_calm.jsonl", False),
        
        # Medium datasets
        (2000, "abstract_medium_chaotic.jsonl", True),
        (2000, "abstract_medium_calm.jsonl", False),
        
        # Large datasets for serious training
        (5000, "abstract_large_chaotic.jsonl", True),
        (5000, "abstract_large_calm.jsonl", False)
    ]
    
    for num_echoes, filename, chaos_mode in datasets:
        print(f"\n🎯 Generating {filename}...")
        generator.generate_training_dataset(num_echoes, filename, chaos_mode)
    
    print("\n✅ All abstract datasets generated!")
    print("\nDatasets available:")
    print("  CHAOTIC: abstract_*_chaotic.jsonl")
    print("  CALM: abstract_*_calm.jsonl")
    print("\nThese can now be used with fast JSONL loading instead of")
    print("expensive runtime data generation during training!")

if __name__ == "__main__":
    main() 