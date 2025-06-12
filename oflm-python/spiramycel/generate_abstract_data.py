#!/usr/bin/env python3
"""
Abstract Data Generator for Spiramycel Training

Pre-generates abstract training data to eliminate the performance bottleneck
where abstract training was generating data during training time.

Includes o3's stability fixes for robust data generation.
"""

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Import existing components
from glyph_codec import SpiramycelGlyphCodec
from spore_map import Season
from neural_trainer import SporeMapLedger, SpiramycelTrainer

class AbstractDataGenerator:
    """Generates abstract training data for Spiramycel (pre-processed) with stability improvements"""
    
    def __init__(self, random_seed: Optional[int] = None):
        self.codec = SpiramycelGlyphCodec()
        
        # Set reproducible random seed for consistent data generation
        if random_seed is not None:
            random.seed(random_seed)
            self.random_seed = random_seed
            print(f"🌱 Random seed set to {random_seed} for reproducible generation")
        else:
            self.random_seed = None
        
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
        
        # Robust path handling
        try:
            output_path = Path("training_scenarios") / output_file
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"❌ Error creating output directory: {e}")
            return ""
        
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
        
        # Fixed scenario weights (same fix as neural_trainer.py)
        if chaos_mode:
            scenario_weights = [1.0, 1.0, 1.0]  # Equal weight (normalized by random.choices)
            problem_vs_optimal_ratio = 0.7  # 70% problems
        else:
            scenario_weights = [1.0, 1.0, 1.0]  # Equal weight (normalized by random.choices)
            problem_vs_optimal_ratio = 0.4  # 40% problems
        
        scenario_names = list(abstract_scenarios.keys())
        
        # Robust file writing with error handling
        echoes_generated = 0
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for i in range(num_echoes):
                    try:
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
                        
                        # Generate sensor deltas (consistent with spore_map expected format)
                        sensor_deltas = {"latency": 0.0, "voltage": 0.0, "temperature": 0.0, "error_rate": 0.0, "bandwidth": 0.0}
                        
                        if "sensor_ranges" in problem_type:
                            for sensor, (min_val, max_val) in problem_type["sensor_ranges"].items():
                                if sensor == "voltage":
                                    sensor_deltas[sensor] = random.uniform(min_val, max_val) - 3.3  # Delta from nominal
                                elif sensor == "temperature":
                                    sensor_deltas[sensor] = random.uniform(min_val, max_val) - 25.0  # Delta from nominal  
                                elif sensor == "bandwidth":
                                    sensor_deltas[sensor] = random.uniform(min_val, max_val) - 0.8   # Delta from nominal
                                else:  # latency, error_rate (absolute values)
                                    sensor_deltas[sensor] = random.uniform(min_val, max_val)
                        else:  # Optimal conditions
                            sensor_deltas = {
                                "latency": random.uniform(-0.05, 0.01),
                                "voltage": random.uniform(0.0, 0.1),
                                "temperature": random.uniform(-3.0, 1.0),
                                "error_rate": random.uniform(0.0, 0.01),
                                "bandwidth": random.uniform(-0.1, 0.1)
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
                        
                        # Create spore echo in format compatible with spore_map.py
                        spore_echo = {
                            "scenario": {
                                "id": scenario_name,
                                "name": scenario["name"],
                                "description": scenario["description"],
                                "bioregion": bioregion,
                                "season": season.value  # Fixed: serialize enum as string
                            },
                            "conditions": {
                                "sensor_deltas": sensor_deltas,  # Changed from sensor_readings for consistency
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
                            
                            # Improve sensor deltas toward optimal (deltas closer to 0)
                            for sensor, delta in spore_echo['conditions']['sensor_deltas'].items():
                                # Move 30% toward optimal (delta = 0)
                                new_delta = delta * 0.7  # Reduce magnitude by 30%
                                spore_echo['conditions']['sensor_deltas'][sensor] = new_delta
                            
                            # Increase silence probability
                            spore_echo['repair_action']['silence_probability'] = min(1.0, 
                                spore_echo['repair_action']['silence_probability'] + 0.2)
                        
                        # Write JSON with proper encoding
                        f.write(json.dumps(spore_echo, ensure_ascii=False) + '\n')
                        echoes_generated += 1
                        
                        # Improved progress reporting (less frequent, more informative)
                        if (i + 1) % 500 == 0:
                            progress_pct = ((i + 1) / num_echoes) * 100
                            print(f"  Generated {i + 1}/{num_echoes} abstract spore echoes ({progress_pct:.1f}%)...")
                            
                    except Exception as e:
                        print(f"⚠ Error generating echo {i+1}: {e}")
                        continue
                        
        except Exception as e:
            print(f"❌ Error writing to file {output_path}: {e}")
            return ""
        
        if echoes_generated == 0:
            print(f"❌ No spore echoes generated successfully!")
            return ""
        
        print(f"✓ Generated {echoes_generated} abstract spore echoes")
        print(f"📁 Saved to: {output_path}")
        
        # Statistics with memory-efficient processing
        self.print_dataset_statistics(output_path)
        
        return str(output_path)
    
    def print_dataset_statistics(self, dataset_path: str):
        """Print statistics about the generated dataset (memory-efficient)"""
        
        print(f"\n📊 Abstract Dataset Statistics:")
        
        try:
            # Count lines first
            with open(dataset_path, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for line in f if line.strip())
            
            print(f"   Total spore echoes: {total_lines}")
            
            if total_lines == 0:
                print("   ⚠ No valid spore echoes found!")
                return
            
            # Memory-efficient statistics processing
            scenario_counts = {}
            effectiveness_sum = 0.0
            stress_sum = 0.0
            optimal_count = 0
            silence_sum = 0.0
            high_silence_count = 0
            valid_echoes = 0
            
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        if line.strip():
                            echo = json.loads(line)
                            
                            # Scenario distribution
                            scenario = echo['scenario']['id']
                            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
                            
                            # Effectiveness accumulation
                            effectiveness_sum += echo['repair_action']['effectiveness']
                            
                            # Environmental stress accumulation
                            stress_sum += echo['conditions']['environmental_stress']
                            
                            # Optimal conditions count
                            if echo['conditions']['extreme_event'] is None:
                                optimal_count += 1
                            
                            # Silence analysis
                            silence_prob = echo['repair_action']['silence_probability']
                            silence_sum += silence_prob
                            if silence_prob > 0.8:
                                high_silence_count += 1
                            
                            valid_echoes += 1
                            
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"⚠ Corrupt echo at line {line_num+1}: {e}")
                        continue
                    except Exception as e:
                        print(f"⚠ Error processing line {line_num+1}: {e}")
                        continue
            
            if valid_echoes == 0:
                print("   ⚠ No valid spore echoes could be processed!")
                return
                
            # Print scenario distribution
            print(f"   Scenario distribution:")
            for scenario, count in scenario_counts.items():
                percentage = (count / valid_echoes) * 100
                print(f"     {scenario}: {count} ({percentage:.1f}%)")
            
            # Print averages
            avg_effectiveness = effectiveness_sum / valid_echoes
            avg_stress = stress_sum / valid_echoes
            avg_silence = silence_sum / valid_echoes
            
            print(f"   Average repair effectiveness: {avg_effectiveness:.3f}")
            print(f"   Average environmental stress: {avg_stress:.3f}")
            
            # Problem types vs optimal
            problem_count = valid_echoes - optimal_count
            optimal_pct = (optimal_count / valid_echoes * 100) if valid_echoes > 0 else 0.0
            print(f"   Optimal conditions: {optimal_count} ({optimal_pct:.1f}%)")
            print(f"   Problem scenarios: {problem_count} ({100-optimal_pct:.1f}%)")
            
            # Silence analysis
            high_silence_pct = (high_silence_count / valid_echoes * 100) if valid_echoes > 0 else 0.0
            print(f"   Average silence probability: {avg_silence:.3f}")
            print(f"   High silence (>0.8): {high_silence_count} ({high_silence_pct:.1f}%)")
            
        except FileNotFoundError:
            print(f"   ❌ Dataset file not found: {dataset_path}")
        except Exception as e:
            print(f"   ❌ Error analyzing dataset: {e}")

def main():
    """Main function to generate abstract training data"""
    
    # Create generator with reproducible seed for consistent results
    generator = AbstractDataGenerator(random_seed=42)
    
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
    
    start_time = time.time()
    successful_datasets = 0
    
    for num_echoes, filename, chaos_mode in datasets:
        print(f"\n🎯 Generating {filename}...")
        result_path = generator.generate_training_dataset(num_echoes, filename, chaos_mode)
        
        if result_path:
            successful_datasets += 1
        else:
            print(f"❌ Failed to generate {filename}")
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✅ Generated {successful_datasets}/{len(datasets)} datasets successfully!")
    print(f"⏱ Total generation time: {elapsed_time:.1f} seconds")
    print("\nDatasets available:")
    print("  CHAOTIC: abstract_*_chaotic.jsonl")
    print("  CALM: abstract_*_calm.jsonl")
    print("\nThese can now be used with fast JSONL loading instead of")
    print("expensive runtime data generation during training!")

if __name__ == "__main__":
    main() 