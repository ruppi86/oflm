#!/usr/bin/env python3
"""
Out-of-Distribution (OOD) Test Set Generator

Generates expanded OOD test sets by analyzing existing patterns and creating
statistically valid samples that maintain scenario characteristics.

Supports scaling from 40 samples (10×4) to 400 samples (100×4) or any size.
"""

import json
import numpy as np
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class ScenarioPattern:
    """Statistical pattern for a scenario extracted from existing data"""
    scenario_id: str
    inspiration: str
    stress_signature: str
    bioregion: str
    
    # Statistical distributions for sensor_deltas
    latency_range: Tuple[float, float]
    voltage_range: Tuple[float, float] 
    temperature_range: Tuple[float, float]
    effectiveness_range: Tuple[float, float]
    
    # Advanced patterns (for more realistic generation)
    latency_distribution: str = "uniform"  # "uniform", "bimodal", "normal"
    voltage_distribution: str = "uniform"
    temperature_distribution: str = "uniform"
    effectiveness_distribution: str = "normal"
    
    # Special characteristics
    special_patterns: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.special_patterns is None:
            self.special_patterns = {}

class OODTestSetGenerator:
    """Generator for expanded out-of-distribution test sets"""
    
    def __init__(self, seed: int = 42):
        """Initialize with reproducible random seed"""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Define scenario patterns for environment switching (current alien environments)
        self.switch_scenario_patterns = {
            "arctic_oscillation": ScenarioPattern(
                scenario_id="arctic_oscillation",
                inspiration="Arctic tundra thermal cycles",
                stress_signature="oscillatory", 
                bioregion="arctic_tundra",
                latency_range=(0.08, 0.25),
                voltage_range=(0.70, 0.95),
                temperature_range=(0.05, 0.95),
                effectiveness_range=(0.68, 0.76),
                temperature_distribution="bimodal",  # Hot/cold cycles
                special_patterns={
                    "thermal_oscillation": True,
                    "temp_bimodal_centers": [0.1, 0.9],  # Cold and hot peaks
                    "temp_bimodal_std": 0.03
                }
            ),
            
            "urban_jitter": ScenarioPattern(
                scenario_id="urban_jitter", 
                inspiration="5G network interference patterns",
                stress_signature="rhythmic_irregularity",
                bioregion="urban_dense",
                latency_range=(0.15, 0.95),
                voltage_range=(0.15, 0.95), 
                temperature_range=(0.35, 0.75),
                effectiveness_range=(0.32, 0.62),
                latency_distribution="uniform",  # Wide irregular patterns
                voltage_distribution="uniform",
                special_patterns={
                    "interference_spikes": True,
                    "jitter_correlation": -0.7  # Negative correlation latency/voltage
                }
            ),
            
            "voltage_undershoot": ScenarioPattern(
                scenario_id="voltage_undershoot",
                inspiration="Solar panel cloud shadow events", 
                stress_signature="recovery_lag",
                bioregion="solar_farm",
                latency_range=(0.32, 0.50),
                voltage_range=(0.12, 0.35),
                temperature_range=(0.62, 0.80),
                effectiveness_range=(0.33, 0.48),
                special_patterns={
                    "power_recovery": True,
                    "voltage_temp_correlation": 0.6  # Higher temp = lower voltage
                }
            ),
            
            "inverted_stability": ScenarioPattern(
                scenario_id="inverted_stability",
                inspiration="Quantum coherence maintenance",
                stress_signature="inverted", 
                bioregion="quantum_lab",
                latency_range=(0.03, 0.15),
                voltage_range=(0.85, 0.97),
                temperature_range=(0.12, 0.28),
                effectiveness_range=(0.79, 0.91),
                effectiveness_distribution="normal",  # Tight distribution around high effectiveness
                special_patterns={
                    "optimal_conditions": True,
                    "stability_correlation": 0.8  # All sensors positively correlated
                }
            )
        }
        
        # Define balanced cross-paradigm testing scenarios
        self.same_scenario_patterns = {
            # ================================
            # ECOLOGICAL SCENARIOS  
            # ================================
            
            # Ecological Chaotic (for testing ecological_calm models)
            "ecological_rice_crisis": ScenarioPattern(
                scenario_id="ecological_rice_crisis",
                inspiration="Rice paddy ecosystem collapse (drought + disease)",
                stress_signature="agricultural_crisis",
                bioregion="rice_paddy_guangzhou", 
                latency_range=(0.2, 0.8),      # High water stress
                voltage_range=(0.1, 0.4),      # Low nutrients
                temperature_range=(0.7, 0.95), # Heat stress
                effectiveness_range=(0.2, 0.6),
                special_patterns={
                    "agricultural_stress": True,
                    "water_shortage": True,
                    "nutrient_depletion": 0.8
                }
            ),
            
            "ecological_watershed_collapse": ScenarioPattern(
                scenario_id="ecological_watershed_collapse", 
                inspiration="Watershed contamination + aquifer depletion",
                stress_signature="hydrological_crisis",
                bioregion="groundwater_sweden",
                latency_range=(0.3, 0.9),      # Slow contamination spread
                voltage_range=(0.05, 0.3),     # Severe pollution
                temperature_range=(0.1, 0.4),  # Cold stress + pollution
                effectiveness_range=(0.15, 0.5),
                special_patterns={
                    "contamination_spread": True,
                    "aquifer_depletion": 0.9,
                    "pollution_correlation": -0.8
                }
            ),
            
            # Ecological Calm (for testing ecological_chaotic models)
            "ecological_pristine_forest": ScenarioPattern(
                scenario_id="ecological_pristine_forest",
                inspiration="Old-growth forest in perfect balance",
                stress_signature="ecological_harmony",
                bioregion="pristine_rainforest",
                latency_range=(0.01, 0.05),    # Minimal disturbance
                voltage_range=(0.85, 0.98),    # Rich nutrients
                temperature_range=(0.25, 0.35), # Perfect climate
                effectiveness_range=(0.8, 0.95),
                special_patterns={
                    "biodiversity_peak": True,
                    "nutrient_cycling": 0.95,
                    "carbon_sequestration": 0.9
                }
            ),
            
            "ecological_coral_paradise": ScenarioPattern(
                scenario_id="ecological_coral_paradise",
                inspiration="Thriving coral reef ecosystem",
                stress_signature="marine_harmony", 
                bioregion="healthy_coral_reef",
                latency_range=(0.02, 0.08),    # Gentle tidal flows
                voltage_range=(0.8, 0.95),     # Rich marine nutrients
                temperature_range=(0.3, 0.4),  # Perfect water temp
                effectiveness_range=(0.75, 0.92),
                special_patterns={
                    "coral_health": 0.95,
                    "fish_abundance": True,
                    "water_clarity": 0.9
                }
            ),
            
            # ================================
            # ABSTRACT SCENARIOS
            # ================================
            
            # Abstract Chaotic (for testing abstract_calm models)
            "abstract_network_storm": ScenarioPattern(
                scenario_id="abstract_network_storm",
                inspiration="Cascading network failures + DDoS attacks",
                stress_signature="protocol_chaos",
                bioregion="distributed_network",
                latency_range=(0.6, 0.95),     # Severe network delays
                voltage_range=(0.05, 0.3),     # Power failures
                temperature_range=(0.8, 0.99), # Server overheating
                effectiveness_range=(0.1, 0.4),
                special_patterns={
                    "ddos_attack": True,
                    "cascade_failures": 0.9,
                    "packet_loss": 0.8
                }
            ),
            
            "abstract_database_corruption": ScenarioPattern(
                scenario_id="abstract_database_corruption", 
                inspiration="Database corruption + concurrent access chaos",
                stress_signature="data_integrity_crisis",
                bioregion="enterprise_datacenter",
                latency_range=(0.4, 0.9),      # Query timeouts
                voltage_range=(0.1, 0.35),     # Storage failures  
                temperature_range=(0.7, 0.95), # Thermal throttling
                effectiveness_range=(0.15, 0.5),
                special_patterns={
                    "data_corruption": 0.8,
                    "concurrent_conflicts": True,
                    "backup_failures": 0.9
                }
            ),
            
            # Abstract Calm (for testing abstract_chaotic models)
            "abstract_optimal_cluster": ScenarioPattern(
                scenario_id="abstract_optimal_cluster",
                inspiration="Perfect distributed system performance",
                stress_signature="computational_harmony",
                bioregion="optimized_datacenter",
                latency_range=(0.01, 0.05),    # Sub-millisecond responses
                voltage_range=(0.9, 0.99),     # Stable power
                temperature_range=(0.15, 0.25), # Perfect cooling
                effectiveness_range=(0.85, 0.98),
                special_patterns={
                    "load_balancing": 0.99,
                    "cache_hits": 0.95,
                    "zero_downtime": True
                }
            ),
            
            "abstract_quantum_coherence": ScenarioPattern(
                scenario_id="abstract_quantum_coherence",
                inspiration="Quantum computing in perfect coherence state",
                stress_signature="quantum_stability",
                bioregion="quantum_laboratory", 
                latency_range=(0.001, 0.01),   # Quantum-speed operations
                voltage_range=(0.95, 0.999),   # Ultra-stable power
                temperature_range=(0.05, 0.1), # Near absolute zero
                effectiveness_range=(0.9, 0.99),
                special_patterns={
                    "quantum_coherence": 0.99,
                    "error_correction": 0.98,
                    "entanglement_fidelity": 0.95
                }
            )
        }
    
    def analyze_existing_patterns(self, existing_file: str) -> Dict[str, Any]:
        """Analyze existing OOD test set to validate our patterns"""
        if not Path(existing_file).exists():
            print(f"⚠ Existing file not found: {existing_file}")
            return {}
        
        analysis = defaultdict(list)
        
        with open(existing_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    scenario_id = data["scenario_id"]
                    analysis[scenario_id].append(data)
        
        print("📊 Existing Pattern Analysis:")
        print("=" * 50)
        
        for scenario_id, samples in analysis.items():
            print(f"\n🎯 {scenario_id.upper()} ({len(samples)} samples):")
            
            # Extract sensor values
            latencies = [s["sensor_deltas"]["latency"] for s in samples]
            voltages = [s["sensor_deltas"]["voltage"] for s in samples] 
            temperatures = [s["sensor_deltas"]["temperature"] for s in samples]
            effectivenesses = [s["effectiveness"] for s in samples]
            
            print(f"   Latency: {min(latencies):.3f} - {max(latencies):.3f} (μ={np.mean(latencies):.3f})")
            print(f"   Voltage: {min(voltages):.3f} - {max(voltages):.3f} (μ={np.mean(voltages):.3f})")
            print(f"   Temperature: {min(temperatures):.3f} - {max(temperatures):.3f} (μ={np.mean(temperatures):.3f})")
            print(f"   Effectiveness: {min(effectivenesses):.3f} - {max(effectivenesses):.3f} (μ={np.mean(effectivenesses):.3f})")
        
        return dict(analysis)
    
    def generate_sensor_values(self, pattern: ScenarioPattern) -> Dict[str, float]:
        """Generate realistic sensor values following scenario patterns"""
        sensor_deltas = {}
        
        # Generate basic sensor values
        if pattern.special_patterns and pattern.special_patterns.get("thermal_oscillation"):
            # Arctic oscillation: bimodal temperature (hot/cold cycles)
            if random.random() < 0.5:
                # Cold phase
                temperature = np.random.normal(0.1, 0.03)
                temperature = max(0.05, min(0.2, temperature))
            else:
                # Hot phase  
                temperature = np.random.normal(0.9, 0.03)
                temperature = max(0.8, min(0.95, temperature))
            sensor_deltas["temperature"] = temperature
            
            # Latency varies with thermal stress
            if temperature > 0.5:  # Hot phase
                sensor_deltas["latency"] = random.uniform(0.08, 0.15)
            else:  # Cold phase
                sensor_deltas["latency"] = random.uniform(0.15, 0.25)
                
            sensor_deltas["voltage"] = random.uniform(*pattern.voltage_range)
            
        elif pattern.special_patterns and pattern.special_patterns.get("interference_spikes"):
            # Urban jitter: correlated interference patterns
            latency = random.uniform(*pattern.latency_range)
            # Negative correlation: high latency = low voltage (interference)
            voltage_target = pattern.voltage_range[1] - (latency - pattern.latency_range[0]) / (pattern.latency_range[1] - pattern.latency_range[0]) * (pattern.voltage_range[1] - pattern.voltage_range[0])
            voltage = max(pattern.voltage_range[0], min(pattern.voltage_range[1], voltage_target + random.uniform(-0.1, 0.1)))
            
            sensor_deltas["latency"] = latency
            sensor_deltas["voltage"] = voltage
            sensor_deltas["temperature"] = random.uniform(*pattern.temperature_range)
            
        elif pattern.special_patterns and pattern.special_patterns.get("power_recovery"):
            # Voltage undershoot: temperature affects voltage recovery
            temperature = random.uniform(*pattern.temperature_range)
            # Higher temperature = lower voltage (thermal stress on panels)
            temp_normalized = (temperature - pattern.temperature_range[0]) / (pattern.temperature_range[1] - pattern.temperature_range[0])
            voltage_target = pattern.voltage_range[1] - temp_normalized * (pattern.voltage_range[1] - pattern.voltage_range[0])
            voltage = max(pattern.voltage_range[0], min(pattern.voltage_range[1], voltage_target + random.uniform(-0.02, 0.02)))
            
            sensor_deltas["temperature"] = temperature
            sensor_deltas["voltage"] = voltage
            sensor_deltas["latency"] = random.uniform(*pattern.latency_range)
            
        elif pattern.special_patterns and pattern.special_patterns.get("optimal_conditions"):
            # Inverted stability: all optimal, slightly correlated
            base_quality = random.uniform(0.0, 1.0)  # Overall system quality
            
            # All sensors benefit from high system quality
            latency = pattern.latency_range[0] + (1.0 - base_quality) * (pattern.latency_range[1] - pattern.latency_range[0])
            voltage = pattern.voltage_range[0] + base_quality * (pattern.voltage_range[1] - pattern.voltage_range[0])
            temperature = pattern.temperature_range[0] + (1.0 - base_quality) * (pattern.temperature_range[1] - pattern.temperature_range[0])
            
            sensor_deltas["latency"] = latency + random.uniform(-0.01, 0.01)
            sensor_deltas["voltage"] = voltage + random.uniform(-0.01, 0.01)
            sensor_deltas["temperature"] = temperature + random.uniform(-0.01, 0.01)
            
        elif pattern.special_patterns and pattern.special_patterns.get("agricultural_stress"):
            # Rice paddy chaotic: water shortage + nutrient depletion
            water_stress = random.uniform(0.7, 0.95)  # High water stress
            nutrient_depletion = pattern.special_patterns.get("nutrient_depletion", 0.8)
            
            sensor_deltas["latency"] = water_stress + random.uniform(-0.1, 0.1)  # Water availability
            sensor_deltas["voltage"] = (1.0 - nutrient_depletion) * random.uniform(0.1, 0.4)  # Nutrients
            sensor_deltas["temperature"] = random.uniform(*pattern.temperature_range)
            
        elif pattern.special_patterns and pattern.special_patterns.get("contamination_spread"):
            # Groundwater chaotic: contamination spreading
            contamination_level = random.uniform(0.6, 0.95)
            pollution_correlation = pattern.special_patterns.get("pollution_correlation", -0.8)
            
            sensor_deltas["latency"] = contamination_level + random.uniform(-0.05, 0.05)
            sensor_deltas["voltage"] = max(0.05, (1.0 - contamination_level) * 0.3 + random.uniform(-0.05, 0.05))
            sensor_deltas["temperature"] = random.uniform(*pattern.temperature_range)
            
        elif pattern.special_patterns and pattern.special_patterns.get("extreme_drought"):
            # Drought chaotic: ecosystem collapse
            drought_severity = random.uniform(0.9, 0.99)
            soil_degradation = pattern.special_patterns.get("soil_degradation", 0.95)
            
            sensor_deltas["latency"] = drought_severity  # Water scarcity
            sensor_deltas["voltage"] = max(0.0, (1.0 - soil_degradation) * 0.2)  # Soil nutrients
            sensor_deltas["temperature"] = random.uniform(*pattern.temperature_range)
            
        elif pattern.special_patterns and pattern.special_patterns.get("perfect_balance"):
            # Optimal stability: perfect ecosystem
            balance_quality = random.uniform(0.8, 1.0)
            stability_bonus = pattern.special_patterns.get("stability_bonus", 0.9)
            
            sensor_deltas["latency"] = (1.0 - balance_quality) * 0.1  # Minimal stress
            sensor_deltas["voltage"] = balance_quality * stability_bonus  # High resources  
            sensor_deltas["temperature"] = random.uniform(*pattern.temperature_range)
            
        elif pattern.special_patterns and pattern.special_patterns.get("biodiversity_peak"):
            # Ecological pristine forest: perfect natural balance
            biodiversity = random.uniform(0.9, 1.0)
            nutrient_cycling = pattern.special_patterns.get("nutrient_cycling", 0.95)
            
            sensor_deltas["latency"] = (1.0 - biodiversity) * 0.05  # Minimal disturbance
            sensor_deltas["voltage"] = biodiversity * nutrient_cycling  # Rich nutrients
            sensor_deltas["temperature"] = random.uniform(*pattern.temperature_range)
            
        elif pattern.special_patterns and pattern.special_patterns.get("coral_health"):
            # Ecological coral paradise: thriving marine ecosystem
            coral_health = pattern.special_patterns.get("coral_health", 0.95)
            water_clarity = pattern.special_patterns.get("water_clarity", 0.9)
            
            base_health = random.uniform(0.85, 1.0)
            sensor_deltas["latency"] = (1.0 - base_health) * 0.08  # Gentle flows
            sensor_deltas["voltage"] = base_health * coral_health  # Marine nutrients
            sensor_deltas["temperature"] = random.uniform(*pattern.temperature_range)
            
        elif pattern.special_patterns and pattern.special_patterns.get("ddos_attack"):
            # Abstract network storm: cascading failures
            attack_intensity = random.uniform(0.7, 0.95)
            cascade_failures = pattern.special_patterns.get("cascade_failures", 0.9)
            packet_loss = pattern.special_patterns.get("packet_loss", 0.8)
            
            sensor_deltas["latency"] = attack_intensity  # Severe delays
            sensor_deltas["voltage"] = max(0.05, (1.0 - cascade_failures) * 0.3)  # Power failures
            sensor_deltas["temperature"] = random.uniform(*pattern.temperature_range)
            
        elif pattern.special_patterns and pattern.special_patterns.get("data_corruption"):
            # Abstract database corruption: data integrity crisis
            corruption_level = pattern.special_patterns.get("data_corruption", 0.8)
            backup_failures = pattern.special_patterns.get("backup_failures", 0.9)
            
            sensor_deltas["latency"] = corruption_level + random.uniform(-0.1, 0.1)  # Query timeouts
            sensor_deltas["voltage"] = max(0.1, (1.0 - backup_failures) * 0.35)  # Storage failures
            sensor_deltas["temperature"] = random.uniform(*pattern.temperature_range)
            
        elif pattern.special_patterns and pattern.special_patterns.get("load_balancing"):
            # Abstract optimal cluster: perfect system performance
            system_efficiency = pattern.special_patterns.get("load_balancing", 0.99)
            cache_hits = pattern.special_patterns.get("cache_hits", 0.95)
            
            optimal_performance = random.uniform(0.95, 1.0)
            sensor_deltas["latency"] = (1.0 - optimal_performance) * 0.05  # Sub-ms responses
            sensor_deltas["voltage"] = optimal_performance * system_efficiency  # Stable power
            sensor_deltas["temperature"] = random.uniform(*pattern.temperature_range)
            
        elif pattern.special_patterns and pattern.special_patterns.get("quantum_coherence"):
            # Abstract quantum coherence: quantum computing perfection
            coherence_level = pattern.special_patterns.get("quantum_coherence", 0.99)
            error_correction = pattern.special_patterns.get("error_correction", 0.98)
            
            quantum_stability = random.uniform(0.98, 1.0)
            sensor_deltas["latency"] = (1.0 - quantum_stability) * 0.01  # Quantum-speed
            sensor_deltas["voltage"] = quantum_stability * coherence_level  # Ultra-stable
            sensor_deltas["temperature"] = random.uniform(*pattern.temperature_range)
            
        else:
            # Default: independent uniform distributions
            sensor_deltas["latency"] = random.uniform(*pattern.latency_range)
            sensor_deltas["voltage"] = random.uniform(*pattern.voltage_range)
            sensor_deltas["temperature"] = random.uniform(*pattern.temperature_range)
        
        # Clamp all values to valid ranges
        sensor_deltas["latency"] = max(0.0, min(1.0, sensor_deltas["latency"]))
        sensor_deltas["voltage"] = max(0.0, min(1.0, sensor_deltas["voltage"]))
        sensor_deltas["temperature"] = max(0.0, min(1.0, sensor_deltas["temperature"]))
        
        return sensor_deltas
    
    def generate_effectiveness(self, pattern: ScenarioPattern, sensor_deltas: Dict[str, float]) -> float:
        """Generate effectiveness based on scenario and sensor conditions"""
        base_effectiveness = random.uniform(*pattern.effectiveness_range)
        
        # Apply scenario-specific adjustments
        if pattern.scenario_id == "arctic_oscillation":
            # Thermal extremes reduce effectiveness
            temp = sensor_deltas["temperature"] 
            if temp < 0.2 or temp > 0.8:  # Extreme temperatures
                base_effectiveness *= 0.95
                
        elif pattern.scenario_id == "urban_jitter":
            # High interference reduces effectiveness
            interference_level = sensor_deltas["latency"] + (1.0 - sensor_deltas["voltage"])
            if interference_level > 1.2:
                base_effectiveness *= 0.9
                
        elif pattern.scenario_id == "voltage_undershoot":
            # Low voltage severely impacts effectiveness
            if sensor_deltas["voltage"] < 0.2:
                base_effectiveness *= 0.85
                
        elif pattern.scenario_id == "inverted_stability":
            # Optimal conditions boost effectiveness
            stability = sensor_deltas["voltage"] - sensor_deltas["latency"] + (1.0 - sensor_deltas["temperature"])
            if stability > 1.5:
                base_effectiveness *= 1.05
                
        elif "ecological_" in pattern.scenario_id:
            # Ecological scenarios: effectiveness based on ecosystem health
            if "pristine" in pattern.scenario_id or "coral_paradise" in pattern.scenario_id:
                # Calm ecological conditions: high natural effectiveness
                ecosystem_health = sensor_deltas["voltage"] + (1.0 - sensor_deltas["latency"])
                if ecosystem_health > 1.5:
                    base_effectiveness *= 1.1  # Ecosystem bonus
            else:
                # Chaotic ecological conditions: environmental stress reduces effectiveness
                environmental_stress = sensor_deltas["latency"] + (1.0 - sensor_deltas["voltage"])
                if environmental_stress > 1.2:
                    base_effectiveness *= 0.8  # Environmental penalty
                    
        elif "abstract_" in pattern.scenario_id:
            # Abstract scenarios: effectiveness based on system performance
            if "optimal_cluster" in pattern.scenario_id or "quantum_coherence" in pattern.scenario_id:
                # Calm abstract conditions: high computational effectiveness
                system_performance = sensor_deltas["voltage"] + (1.0 - sensor_deltas["latency"])
                if system_performance > 1.8:
                    base_effectiveness *= 1.15  # System optimization bonus
            else:
                # Chaotic abstract conditions: system failures reduce effectiveness
                system_stress = sensor_deltas["latency"] + (1.0 - sensor_deltas["voltage"])
                if system_stress > 1.3:
                    base_effectiveness *= 0.75  # System failure penalty
        
        return max(0.0, min(1.0, base_effectiveness))
    
    def generate_sample(self, pattern: ScenarioPattern) -> Dict[str, Any]:
        """Generate a single OOD test sample"""
        sensor_deltas = self.generate_sensor_values(pattern)
        effectiveness = self.generate_effectiveness(pattern, sensor_deltas)
        
        return {
            "scenario_id": pattern.scenario_id,
            "inspiration": pattern.inspiration,
            "stress_signature": pattern.stress_signature,
            "sensor_deltas": sensor_deltas,
            "bioregion": pattern.bioregion,
            "effectiveness": round(effectiveness, 2)
        }
    
    def generate_expanded_test_set(self, 
                                 samples_per_scenario: int = 100,
                                 output_file: str = None,
                                 analyze_existing: bool = True,
                                 environment: str = "same") -> str:
        """Generate expanded OOD test set with specified samples per scenario"""
        
        # Choose scenario patterns based on environment mode
        if environment == "same":
            scenario_patterns = self.same_scenario_patterns
            env_description = "balanced cross-paradigm testing"
            print(f"🎯 Using BALANCED CROSS-PARADIGM testing:")
            print(f"   🌿 Ecological scenarios: pristine forests ↔ ecosystem collapse") 
            print(f"   🖥️ Abstract scenarios: optimal systems ↔ network storms")
        elif environment == "switch":
            scenario_patterns = self.switch_scenario_patterns
            env_description = "environment-switching (extreme generalization)"
            print(f"🌍 Using ENVIRONMENT-SWITCHING testing: completely alien environments")
        else:
            raise ValueError(f"Invalid environment mode: {environment}. Use 'same' or 'switch'")
        
        if analyze_existing and environment == "switch":
            # Use path relative to this script's directory
            script_dir = Path(__file__).parent
            existing_file = script_dir / "data/test_sets/ood_test_set.jsonl"
            self.analyze_existing_patterns(str(existing_file))
        
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_file is None:
            # Use path relative to this script's directory
            script_dir = Path(__file__).parent
            output_file = script_dir / f"data/test_sets/ood_test_set_{environment}_{samples_per_scenario}x4_{timestamp}.jsonl"
        
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔬 Generating Expanded OOD Test Set ({env_description}):")
        print(f"   Samples per scenario: {samples_per_scenario}")
        print(f"   Total samples: {samples_per_scenario * len(scenario_patterns)}")
        print(f"   Environment mode: {environment}")
        print(f"   Output file: {output_file}")
        print()
        
        all_samples = []
        
        # Generate samples for each scenario
        for scenario_id, pattern in scenario_patterns.items():
            print(f"🎯 Generating {samples_per_scenario} samples for {scenario_id}...")
            
            scenario_samples = []
            for i in range(samples_per_scenario):
                sample = self.generate_sample(pattern)
                scenario_samples.append(sample)
                all_samples.append(sample)
            
            # Show sample statistics
            latencies = [s["sensor_deltas"]["latency"] for s in scenario_samples]
            voltages = [s["sensor_deltas"]["voltage"] for s in scenario_samples]
            temperatures = [s["sensor_deltas"]["temperature"] for s in scenario_samples]
            effectivenesses = [s["effectiveness"] for s in scenario_samples]
            
            print(f"   ✅ Generated: Latency μ={np.mean(latencies):.3f} "
                  f"Voltage μ={np.mean(voltages):.3f} "
                  f"Temperature μ={np.mean(temperatures):.3f} "
                  f"Effectiveness μ={np.mean(effectivenesses):.3f}")
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in all_samples:
                f.write(json.dumps(sample) + '\n')
        
        # Final statistics
        print(f"\n📊 Generated OOD Test Set:")
        print(f"   Total samples: {len(all_samples)}")
        print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
        print(f"   Environment mode: {environment}")
        print(f"   Scenarios: {len(scenario_patterns)}")
        print(f"   Samples per scenario: {samples_per_scenario}")
        print(f"   ✅ Saved: {output_path}")
        
        return str(output_path)
    
    def validate_generated_set(self, generated_file: str, environment: str = "same"):
        """Validate the generated test set for quality and distribution"""
        print(f"\n🔍 Validating Generated Test Set: {generated_file}")
        print("=" * 60)
        
        if not Path(generated_file).exists():
            print(f"❌ File not found: {generated_file}")
            return
        
        # Choose scenario patterns based on environment mode
        if environment == "same":
            scenario_patterns = self.same_scenario_patterns
        else:
            scenario_patterns = self.switch_scenario_patterns
        
        samples_by_scenario = defaultdict(list)
        
        with open(generated_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    samples_by_scenario[sample["scenario_id"]].append(sample)
        
        print(f"📈 Validation Results:")
        
        for scenario_id, samples in samples_by_scenario.items():
            print(f"\n🎯 {scenario_id} ({len(samples)} samples):")
            
            # Check distributions
            latencies = [s["sensor_deltas"]["latency"] for s in samples]
            voltages = [s["sensor_deltas"]["voltage"] for s in samples]
            temperatures = [s["sensor_deltas"]["temperature"] for s in samples]
            effectivenesses = [s["effectiveness"] for s in samples]
            
            print(f"   Latency: range={min(latencies):.3f}-{max(latencies):.3f}, "
                  f"μ={np.mean(latencies):.3f}, σ={np.std(latencies):.3f}")
            print(f"   Voltage: range={min(voltages):.3f}-{max(voltages):.3f}, "
                  f"μ={np.mean(voltages):.3f}, σ={np.std(voltages):.3f}")
            print(f"   Temperature: range={min(temperatures):.3f}-{max(temperatures):.3f}, "
                  f"μ={np.mean(temperatures):.3f}, σ={np.std(temperatures):.3f}")
            print(f"   Effectiveness: range={min(effectivenesses):.3f}-{max(effectivenesses):.3f}, "
                  f"μ={np.mean(effectivenesses):.3f}, σ={np.std(effectivenesses):.3f}")
            
            # Validate expected patterns (skip if scenario not in patterns - for flexibility)
            if scenario_id in scenario_patterns:
                pattern = scenario_patterns[scenario_id]
                lat_in_range = all(pattern.latency_range[0] <= lat <= pattern.latency_range[1] for lat in latencies)
                vol_in_range = all(pattern.voltage_range[0] <= vol <= pattern.voltage_range[1] for vol in voltages)
                temp_in_range = all(pattern.temperature_range[0] <= temp <= pattern.temperature_range[1] for temp in temperatures)
                eff_in_range = all(pattern.effectiveness_range[0] <= eff <= pattern.effectiveness_range[1] for eff in effectivenesses)
                
                if lat_in_range and vol_in_range and temp_in_range and eff_in_range:
                    print(f"   ✅ All values within expected ranges")
                else:
                    print(f"   ⚠ Some values outside expected ranges")
                    if not lat_in_range: print(f"      ⚠ Latency out of range")
                    if not vol_in_range: print(f"      ⚠ Voltage out of range")
                    if not temp_in_range: print(f"      ⚠ Temperature out of range")
                    if not eff_in_range: print(f"      ⚠ Effectiveness out of range")
            else:
                print(f"   ✅ Scenario patterns not defined for validation")
        
        print(f"\n✅ Validation complete")

def main():
    """Main function for generating expanded OOD test sets"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate expanded OOD test sets")
    parser.add_argument("--samples", type=int, default=100, 
                       help="Samples per scenario (default: 100)")
    parser.add_argument("--environment", choices=["same", "switch"], default="same",
                       help="Test environment mode: 'same' for balanced cross-paradigm testing (default), 'switch' for alien environments")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    print("🧪 OUT-OF-DISTRIBUTION TEST SET GENERATOR")
    print("=" * 60)
    print("Robust statistical generation of expanded OOD test scenarios")
    print(f"Environment mode: {args.environment}")
    print()
    
    # Initialize generator
    generator = OODTestSetGenerator(seed=args.seed)
    
    # Generate expanded test set
    output_file = generator.generate_expanded_test_set(
        samples_per_scenario=args.samples,
        analyze_existing=(args.environment == "switch"),  # Only analyze existing for switch mode
        environment=args.environment
    )
    
    # Validate the generated set
    generator.validate_generated_set(output_file, args.environment)
    
    print(f"\n🎉 OOD Test Set Generation Complete!")
    print(f"📁 Generated: {output_file}")
    print(f"📊 Ready for robust statistical analysis with {args.samples * 4} samples")
    print(f"🔬 Environment mode: {args.environment}")
    if args.environment == "same":
        print("🎯 Tests balanced cross-paradigm contemplative adaptation:")
        print("   🌿 Ecological models: pristine forests ↔ ecosystem collapse")
        print("   🖥️ Abstract models: optimal systems ↔ network storms")
    else:
        print("🌍 Tests extreme generalization to completely alien environments")

if __name__ == "__main__":
    main() 