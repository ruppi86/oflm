#!/usr/bin/env python3
"""
Ecological Data Generator for Spiramycel Training

Generates realistic spore echoes based on actual ecological scenarios:
- Drought‑stressed eucalyptus forest (Australia)
- Rice paddy ecosystem (Guangzhou, China)
- Groundwater monitoring (Sweden)

Each scenario includes multi‑generational patterns, seasonal cycles,
and real bioregional adaptation strategies.

Includes o3's stability fixes for robust data generation.
"""

import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
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
    """Generates realistic ecological training data for Spiramycel with stability improvements"""

    def __init__(self, scenarios_dir: str | None = None, random_seed: Optional[int] = None):
        """Initialise with scenario definitions and optional reproducible seed"""
        scenarios_dir = Path(scenarios_dir) if scenarios_dir else Path(__file__).parent
        self.scenarios_dir = scenarios_dir
        self.scenarios: Dict[str, Dict[str, Any]] = {}

        # Optional reproducibility
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            self.random_seed = random_seed
            print(f"🌱 Random seed set to {random_seed} for reproducible ecological generation")
        else:
            self.random_seed = None

        # Simulation calendar
        self.base_year = 2024
        self.simulation_years = 50  # 50‑year simulation window

        # Diagnostics
        self.unknown_seasonal_combos: set[Tuple[str, Tuple[str, ...]]] = set()

        self.load_scenarios()

    # ---------------------------------------------------------------------
    # Scenario loading
    # ---------------------------------------------------------------------
    def load_scenarios(self):
        """Load all scenario JSON files with robust error handling"""
        scenario_files = [
            "drought_landscape_australia.json",
            "rice_paddy_guangzhou.json",
            "groundwater_sweden.json",
        ]

        loaded = 0
        for fname in scenario_files:
            fpath = self.scenarios_dir / fname
            if not fpath.exists():
                print(f"⚠ Warning: Scenario file not found: {fpath}")
                continue
            try:
                with open(fpath, "r", encoding="utf‑8") as fh:
                    scenario = json.load(fh)
            except json.JSONDecodeError as e:
                print(f"⚠ Warning: Invalid JSON in {fname}: {e}")
                continue
            except Exception as e:
                print(f"⚠ Warning: Could not load {fname}: {e}")
                continue

            # Minimal schema check
            required = {"scenario_id", "name", "bioregion", "ecosystem_type"}
            if not required.issubset(scenario):
                print(f"⚠ Warning: {fname} missing required fields: {required}")
                continue

            self.scenarios[scenario["scenario_id"]] = scenario
            print(f"✓ Loaded scenario: {scenario['name']}")
            loaded += 1

        if not self.scenarios:
            raise FileNotFoundError(
                "No scenario JSON files loaded – check path and file existence."
            )
        print(f"📊 Successfully loaded {loaded} scenarios")

    # ------------------------------------------------------------------
    # Season mapping helpers
    # ------------------------------------------------------------------
    def get_season_for_month(self, scenario: Dict[str, Any], month: int) -> str:
        """Return the season string for a given month according to scenario meta."""
        for season_name, season_data in scenario["seasonal_cycles"].items():
            if season_name == "transition_periods" and isinstance(season_data, dict):
                for sub, sub_data in season_data.items():
                    if month in sub_data.get("months", []):
                        return sub  # autumn / spring
            else:
                if month in season_data.get("months", []):
                    return season_name

        # Fallback heuristics
        if month in (3, 4, 5):
            return "spring"
        if month in (6, 7, 8):
            return "summer"
        if month in (9, 10, 11):
            return "autumn"
        return "winter"  # months 12,1,2

    # ------------------------------------------------------------------
    # Sensor simulation helpers
    # ------------------------------------------------------------------
    def simulate_sensor_readings(
        self,
        scenario: Dict[str, Any],
        season: str,
        year_in_cycle: int,
        extreme_event: str | None = None,
    ) -> Dict[str, float]:
        """Generate realistic sensor readings for a scenario / season."""
        readings: Dict[str, float] = {}
        sensor_mappings = scenario["sensor_mappings"]
        season_data = scenario["seasonal_cycles"].get(season, {})

        for sensor_type in sensor_mappings:
            base = self._get_seasonal_baseline(sensor_type, season_data)
            val = self._apply_generational_patterns(base, scenario, sensor_type, year_in_cycle)
            if extreme_event:
                val = self._apply_extreme_event(val, extreme_event, sensor_type)
            # small stochasticity
            val += random.gauss(0, 0.05)
            readings[sensor_type] = max(0.0, min(1.0, val))
        return readings

     # ----------------------------------------------------------------------
    #  Seasonal baseline table – unified & exhaustively extended
    # ----------------------------------------------------------------------
    def _get_seasonal_baseline(self, sensor_type: str, season_data: Dict) -> float:
        """
        Map (sensor_type, seasonal-descriptor) → [0.0-1.0] baseline.

        • First try direct lookup (season_data[sensor_type]).
        • Then try alias-resolved key.
        • Then scan **all scalar values** in season_data for a match
          (handles patterns like  {"recharge_rate": "slow"}  etc.).
        • Falls back to 0.5 and logs once per unseen combo.
        """

        # ---------- 1. comprehensive baseline mapping table --------
        baselines: Dict[str, Dict[str, float]] = {
            # Soil moisture sensor
            "soil_moisture": {
                "very_low": 0.1, "low": 0.25, "moderate": 0.5, "high": 0.75, "very_high": 0.9,
                "drought_risk": 0.2, "drought_severe": 0.1, "drought_extreme": 0.05,
                "wet": 0.8, "dry": 0.2, "optimal": 0.6, "saturated": 0.95,
                "fungal_activity": 0.6, "rainfall_probability": 0.7
            },
            
            # Nitrogen/nutrient sensors
            "nitrogen_concentration": {
                "very_low": 0.1, "low": 0.25, "moderate": 0.5, "high": 0.75, "very_high": 0.9,
                "depleted": 0.15, "optimal": 0.6, "excessive": 0.9,
                "nutrient_flow": 0.6, "nutrient_transport": 0.7, "nutrient_availability": 0.65,
                "bacterial_activity": 0.7, "flooding_status": 0.8
            },
            
            "nutrient_nitrogen": {
                "very_low": 0.1, "low": 0.25, "moderate": 0.5, "high": 0.75, "very_high": 0.9,
                "depleted": 0.15, "optimal": 0.6, "excessive": 0.9,
                "nutrient_flow": 0.6, "nutrient_transport": 0.7, "nutrient_availability": 0.65,
                "bacterial_activity": 0.7, "flooding_status": 0.8, "fungal_activity": 0.6,
                "rainfall_probability": 0.7
            },
            
            # Phosphorus sensors
            "phosphorus_availability": {
                "very_low": 0.1, "low": 0.25, "moderate": 0.5, "high": 0.75, "very_high": 0.9,
                "depleted": 0.15, "optimal": 0.6, "excessive": 0.85,
                "nutrient_flow": 0.6, "nutrient_transport": 0.7, "nutrient_availability": 0.65,
                "bacterial_activity": 0.7, "flooding_status": 0.8
            },
            
            "nutrient_phosphorus": {
                "very_low": 0.1, "low": 0.25, "moderate": 0.5, "high": 0.75, "very_high": 0.9,
                "depleted": 0.15, "optimal": 0.6, "excessive": 0.85,
                "nutrient_flow": 0.6, "nutrient_transport": 0.7, "nutrient_availability": 0.65,
                "bacterial_activity": 0.7, "flooding_status": 0.8, "fungal_activity": 0.6,
                "rainfall_probability": 0.7
            },
            
            # Temperature sensor
            "temperature": {
                "very_low": 0.1, "low": 0.25, "moderate": 0.5, "high": 0.75, "very_high": 0.9,
                "cold": 0.2, "cool": 0.35, "warm": 0.65, "hot": 0.85, "extreme": 0.95,
                "optimal": 0.5, "stress": 0.8, "freezing": 0.05,
                "contamination_risk": 0.6, "nutrient_transport": 0.5, "recharge_rate": 0.4,
                "bacterial_activity": 0.6, "flooding_status": 0.5, "nutrient_flow": 0.5,
                "fungal_activity": 0.6, "rainfall_probability": 0.5
            },
            
            # Water level sensors
            "water_level": {
                "very_low": 0.1, "low": 0.25, "moderate": 0.5, "high": 0.75, "very_high": 0.9,
                "flood": 0.95, "drought": 0.1, "normal": 0.5, "optimal": 0.6,
                "bacterial_activity": 0.7, "flooding_status": 0.8, "nutrient_flow": 0.6
            },
            
            "water_table_level": {
                "very_low": 0.1, "low": 0.25, "moderate": 0.5, "high": 0.75, "very_high": 0.9,
                "depleted": 0.15, "recharged": 0.8, "optimal": 0.6, "saturated": 0.9,
                "contamination_risk": 0.4, "nutrient_transport": 0.6, "recharge_rate": 0.3,
                "slow": 0.3, "fast": 0.8, "normal": 0.5
            },
            
            # Contamination sensors
            "nitrate_concentration": {
                "very_low": 0.1, "low": 0.25, "moderate": 0.5, "high": 0.75, "very_high": 0.9,
                "safe": 0.2, "elevated": 0.7, "dangerous": 0.9, "optimal": 0.3,
                "contamination_risk": 0.6, "nutrient_transport": 0.7, "recharge_rate": 0.5
            },
            
            "heavy_metal_levels": {
                "very_low": 0.1, "low": 0.25, "moderate": 0.5, "high": 0.75, "very_high": 0.9,
                "safe": 0.1, "elevated": 0.6, "dangerous": 0.9, "toxic": 0.95,
                "contamination_risk": 0.7, "nutrient_transport": 0.3, "recharge_rate": 0.4
            },
            
            # Biological activity sensors
            "bacterial_activity": {
                "very_low": 0.1, "low": 0.25, "moderate": 0.5, "high": 0.75, "very_high": 0.9,
                "dormant": 0.1, "active": 0.7, "optimal": 0.6, "stressed": 0.3,
                "contamination_risk": 0.4, "nutrient_transport": 0.7, "recharge_rate": 0.5,
                "bacterial_activity": 0.6, "flooding_status": 0.7, "nutrient_flow": 0.7
            },
            
            "bacterial_balance": {
                "very_low": 0.1, "low": 0.25, "moderate": 0.5, "high": 0.75, "very_high": 0.9,
                "poor": 0.2, "good": 0.7, "optimal": 0.8, "excellent": 0.9,
                "bacterial_activity": 0.7, "flooding_status": 0.6, "nutrient_flow": 0.7
            },
            
            # pH stability
            "ph_stability": {
                "very_low": 0.1, "low": 0.25, "moderate": 0.5, "high": 0.75, "very_high": 0.9,
                "acidic": 0.2, "neutral": 0.5, "alkaline": 0.8, "optimal": 0.6,
                "contamination_risk": 0.4, "nutrient_transport": 0.5, "recharge_rate": 0.5
            },
            
            # Rice paddy specific sensors
            "methane_production": {
                "very_low": 0.1, "low": 0.25, "moderate": 0.5, "high": 0.75, "very_high": 0.9,
                "minimal": 0.2, "normal": 0.5, "elevated": 0.8, "excessive": 0.95,
                "bacterial_activity": 0.7, "flooding_status": 0.8, "nutrient_flow": 0.6
            },
            
            "rice_health": {
                "very_low": 0.1, "low": 0.25, "moderate": 0.5, "high": 0.75, "very_high": 0.9,
                "poor": 0.2, "fair": 0.4, "good": 0.7, "excellent": 0.9,
                "bacterial_activity": 0.6, "flooding_status": 0.7, "nutrient_flow": 0.7
            },
            
            # Forest ecosystem sensors
            "competitor_pressure": {
                "very_low": 0.1, "low": 0.25, "moderate": 0.5, "high": 0.75, "very_high": 0.9,
                "minimal": 0.2, "light": 0.4, "moderate": 0.6, "heavy": 0.8, "extreme": 0.95,
                "fungal_activity": 0.6, "nutrient_availability": 0.5, "rainfall_probability": 0.6
            },
            
            "root_connections": {
                "very_low": 0.1, "low": 0.25, "moderate": 0.5, "high": 0.75, "very_high": 0.9,
                "poor": 0.2, "fair": 0.4, "good": 0.7, "excellent": 0.9, "optimal": 0.8,
                "fungal_activity": 0.7, "nutrient_availability": 0.6, "rainfall_probability": 0.5
            },
            
            # Generic seasonal descriptors to baseline mappings
            "contamination_risk": {
                "contamination_risk": 0.6, "nutrient_transport": 0.5, "recharge_rate": 0.4,
                "bacterial_activity": 0.4, "flooding_status": 0.7, "nutrient_flow": 0.5
            },
            
            "nutrient_transport": {
                "contamination_risk": 0.5, "nutrient_transport": 0.7, "recharge_rate": 0.6,
                "bacterial_activity": 0.7, "flooding_status": 0.8, "nutrient_flow": 0.8
            },
            
            "recharge_rate": {
                "contamination_risk": 0.4, "nutrient_transport": 0.6, "recharge_rate": 0.5,
                "slow": 0.3, "moderate": 0.5, "fast": 0.8, "rapid": 0.9
            },
            
            "flooding_status": {
                "bacterial_activity": 0.7, "flooding_status": 0.8, "nutrient_flow": 0.6,
                "flooded": 0.9, "normal": 0.5, "dry": 0.2
            },
            
            "nutrient_flow": {
                "bacterial_activity": 0.7, "flooding_status": 0.6, "nutrient_flow": 0.7,
                "slow": 0.3, "moderate": 0.5, "fast": 0.8, "optimal": 0.6
            },
            
            "fungal_activity": {
                "fungal_activity": 0.6, "nutrient_availability": 0.6, "rainfall_probability": 0.5,
                "low": 0.3, "moderate": 0.5, "high": 0.8, "optimal": 0.7
            },
            
            "nutrient_availability": {
                "fungal_activity": 0.6, "nutrient_availability": 0.6, "rainfall_probability": 0.7,
                "low": 0.3, "moderate": 0.5, "high": 0.8, "optimal": 0.7
            },
            
            "rainfall_probability": {
                "fungal_activity": 0.5, "nutrient_availability": 0.7, "rainfall_probability": 0.6,
                "low": 0.2, "moderate": 0.5, "high": 0.8, "very_high": 0.9
            }
        }

        # ---------- 2. alias map (unchanged) -------------------------------
        alias = {
            "nutrient_nitrogen": "nitrogen_concentration",
            "nutrient_phosphorus": "phosphorus_availability",
            "water_table": "water_table_level",
        }

        # ---------- 3. direct + alias lookup -------------------------------
        lookup_key = alias.get(sensor_type, sensor_type)
        label_direct = season_data.get(sensor_type)
        label_alias  = season_data.get(lookup_key)

        if label_direct and lookup_key in baselines and label_direct in baselines[lookup_key]:
            return baselines[lookup_key][label_direct]

        if label_alias and lookup_key in baselines and label_alias in baselines[lookup_key]:
            return baselines[lookup_key][label_alias]

        # ---------- 4. fallback: scan all scalar *values* ------------------
        if lookup_key in baselines:
            for _, value in season_data.items():
                # skip nested dicts / lists
                if isinstance(value, (list, dict)):
                    continue
                if value in baselines[lookup_key]:
                    return baselines[lookup_key][value]

        # ---------- 5. enhanced fallback: try cross-seasonal descriptor matching --
        # For seasonal descriptors that appear in season_data, try to find them in other baselines
        for descriptor_key, descriptor_value in season_data.items():
            if isinstance(descriptor_value, (list, dict)):
                continue
            # Look for this descriptor in any baseline category
            for baseline_category, baseline_values in baselines.items():
                if descriptor_key in baseline_values:
                    return baseline_values[descriptor_key]
                if descriptor_value in baseline_values:
                    return baseline_values[descriptor_value]

        # ---------- 6. sensor-specific fallback for empty seasonal data --------
        # When no seasonal descriptors are found, provide reasonable defaults by sensor type
        sensor_defaults = {
            "soil_moisture": 0.5,           # moderate moisture
            "nutrient_nitrogen": 0.6,       # good nitrogen levels
            "nutrient_phosphorus": 0.55,    # good phosphorus
            "temperature": 0.5,             # moderate temperature
            "competitor_pressure": 0.4,     # moderate competition
            "root_connections": 0.7,        # good root health
            "water_level": 0.5,             # normal water level
            "water_table_level": 0.5,       # normal water table
            "nitrate_concentration": 0.3,   # safe nitrate levels
            "heavy_metal_levels": 0.1,      # low contamination
            "bacterial_activity": 0.6,      # healthy bacterial activity
            "bacterial_balance": 0.7,       # good bacterial balance
            "ph_stability": 0.5,            # neutral pH
            "methane_production": 0.5,      # normal methane
            "rice_health": 0.7,             # good rice health
            "nitrogen_concentration": 0.6,  # good nitrogen (alias)
            "phosphorus_availability": 0.55 # good phosphorus (alias)
        }
        
        if sensor_type in sensor_defaults:
            return sensor_defaults[sensor_type]
        
        # Try alias lookup for defaults
        if lookup_key in sensor_defaults:
            return sensor_defaults[lookup_key]

        # ---------- 7. still unknown → log once, return neutral ------------
        combo_key = (
            sensor_type,
            tuple(sorted(k for k, v in season_data.items()
                         if not isinstance(v, (list, dict))))
        )
        if combo_key not in self.unknown_seasonal_combos:
            self.unknown_seasonal_combos.add(combo_key)
            print(f"🔍 Unknown seasonal combo: sensor={sensor_type}, "
                  f"descriptors={combo_key[1]}")
        return 0.5



    
    def _apply_generational_patterns(self, base_reading: float, scenario: Dict, 
                                   sensor_type: str, year_in_cycle: int) -> float:
        """Apply multi-generational cyclical patterns with specific error handling"""
        patterns = scenario.get('multi_generational_patterns', {})
        reading = base_reading
        
        # Fixed: More specific exception handling (o3's issue #9)
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
            # Log specific pattern application errors for debugging
            print(f"⚠ Pattern application error for {sensor_type}: {e}")
        except Exception as e:
            # Catch unexpected errors but log them
            print(f"⚠ Unexpected error in pattern application for {sensor_type}: {e}")
            
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
    
    def select_repair_strategy(self, scenario: Dict, conditions: NetworkConditions, chaos_mode: bool = True) -> Tuple[List[int], str, float]:
        """Select appropriate repair strategy based on conditions"""
        strategies = scenario['repair_strategies']
        
        # In calm mode, add thriving ecosystem scenarios
        if not chaos_mode:
            # 70% chance of thriving ecosystem (pure contemplative)
            if random.random() < 0.7:
                # Perfect ecosystem balance - minimal intervention needed
                thriving_glyphs = [0x31, 0x32, 0x33, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3F, 0x40]  # Pure contemplative
                glyph_sequence = random.choices(thriving_glyphs, k=random.randint(8, 12))
                effectiveness = random.uniform(0.05, 0.25)  # Very low effectiveness - ecosystem handles itself
                return (glyph_sequence, "ecosystem in perfect harmony - contemplative observation only", effectiveness)
            
            # 20% chance of minor seasonal adjustments
            elif random.random() < 0.9:  # 0.7 + 0.2 = 0.9 total
                # Gentle seasonal maintenance
                maintenance_glyphs = [0x31, 0x32, 0x33, 0x35] + random.choices(list(strategies.values())[0]['glyph_sequence'], k=1)
                contemplative_glyphs = [0x36, 0x37, 0x38, 0x39, 0x3F, 0x40]
                glyph_sequence = maintenance_glyphs + random.choices(contemplative_glyphs, k=random.randint(6, 10))
                random.shuffle(glyph_sequence)
                effectiveness = random.uniform(0.2, 0.4)
                return (glyph_sequence, "gentle seasonal adjustment with contemplative monitoring", effectiveness)
            
            # 10% fall through to normal crisis scenarios below
        
        # Original strategy selection for chaos_mode=True or 10% of calm scenarios
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
                          extreme_event: str = None, chaos_mode: bool = True) -> Dict[str, Any]:
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
        
        # Fixed: Calculate environmental stress over ALL sensor readings (o3's issue #4)
        # More extensible approach that works with any sensor set
        stress_values = []
        for sensor_name, reading in sensor_readings.items():
            # Convert readings to stress (0.5 is optimal, deviations increase stress)
            stress = abs(reading - 0.5) * 2
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
        
        # Select repair strategy with chaos_mode parameter
        glyph_sequence, description, effectiveness = self.select_repair_strategy(scenario, conditions, chaos_mode)
        
        # Calculate Tystnadsmajoritet (silence probability)
        if not chaos_mode and random.random() < 0.7:  # Thriving ecosystem
            silence_probability = random.uniform(0.8, 0.95)  # High silence for thriving systems
        else:
            silence_probability = max(0.1, 0.875 - environmental_stress * 0.3)
        
        return {
            'spore_echo_id': f"{scenario_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
            'timestamp': timestamp.isoformat(),
            'scenario': {
                'id': scenario_id,
                'name': scenario['name'],
                'bioregion': scenario['bioregion'],
                'ecosystem_type': scenario['ecosystem_type'],
                'season': season  # Include season for data consistency
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
                                output_file: str = "ecological_training_data.jsonl",
                                chaos_mode: bool = True) -> str:
        """Generate a complete training dataset
        
        Args:
            num_echoes: Number of spore echoes to generate
            output_file: Output filename
            chaos_mode: If True, includes many extreme events (15% probability)
                       If False, mostly calm conditions (3% probability)
        """
        print(f"🌱 Generating {num_echoes} ecological spore echoes...")
        if chaos_mode:
            print("⚡ Chaos mode: HIGH stress environment (15% extreme events)")  # o3's issue #7: clarify percentage
        else:
            print("🧘 Calm mode: LOW stress environment (3% extreme events)")  # o3's issue #7: clarify percentage
        
        output_path = self.scenarios_dir / output_file
        scenario_ids = list(self.scenarios.keys())
        
        if not scenario_ids:
            raise ValueError("No scenarios loaded!")
        
        extreme_events = [None, "drought", "flood", "fire", "contamination_event"]
        # Fixed: Comment extreme event probability for clarity (o3's issue #7)
        extreme_probability = 0.15 if chaos_mode else 0.03  # 15% or 3% of echoes carry extreme events
        
        generated_count = 0
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for i in range(num_echoes):
                    # Distribute across scenarios
                    scenario_id = random.choice(scenario_ids)
                    
                    # Occasional extreme events based on chaos_mode
                    extreme_event = None
                    if random.random() < extreme_probability:
                        extreme_event = random.choice(extreme_events[1:])  # exclude None
                    
                    try:
                        spore_echo = self.generate_spore_echo(scenario_id, extreme_event=extreme_event, chaos_mode=chaos_mode)
                        
                        # Fixed: In calm mode, recompute stress after sensor adjustments (o3's issue #1)
                        if not chaos_mode:
                            # First adjust sensor readings toward healthier values
                            conditions = spore_echo['conditions']
                            for sensor, value in conditions['sensor_readings'].items():
                                # Move values toward 0.5 (optimal) by 30%
                                optimal_bias = 0.5
                                new_value = value + (optimal_bias - value) * 0.3
                                conditions['sensor_readings'][sensor] = max(0.0, min(1.0, new_value))
                            
                            # CRITICAL FIX: Recompute environmental stress after sensor adjustments
                            sensor_vals = list(conditions['sensor_readings'].values())
                            new_stress = np.mean([abs(v - 0.5) * 2 for v in sensor_vals]) if sensor_vals else 0.3
                            conditions['environmental_stress'] = new_stress * 0.6  # Additional 40% reduction
                            
                            # Update repair urgency based on new stress
                            conditions['repair_urgency'] = min(1.0, new_stress + random.uniform(0, 0.2))
                            
                            # Update repair action to reflect lower urgency
                            spore_echo['repair_action']['silence_probability'] = min(1.0, 
                                spore_echo['repair_action']['silence_probability'] + 0.2)
                        
                        f.write(json.dumps(spore_echo, ensure_ascii=False) + '\n')
                        generated_count += 1
                        
                        if (i + 1) % 100 == 0:
                            progress_pct = ((i + 1) / num_echoes) * 100
                            print(f"  Generated {i + 1}/{num_echoes} spore echoes ({progress_pct:.1f}%)...")
                            
                    except Exception as e:
                        print(f"⚠ Warning: Failed to generate echo {i}: {e}")
                        continue
                        
        except Exception as e:
            print(f"❌ Error writing to file {output_path}: {e}")
            return ""
        
        if generated_count == 0:
            print(f"❌ No spore echoes generated successfully!")
            return ""
        
        print(f"✓ Generated {generated_count} ecological spore echoes")
        print(f"📁 Saved to: {output_path}")
        
        # Statistics
        self.print_dataset_statistics(output_path)
        
        return str(output_path)
    
    def print_dataset_statistics(self, dataset_path: str):
        """Print statistics about the generated dataset with error handling"""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                echoes = []
                for line_num, line in enumerate(f):
                    try:
                        if line.strip():
                            echoes.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"⚠ Corrupt echo at line {line_num+1}: {e}")
                        continue
            
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
            
            # Analyze ecosystem condition types (for calm mode)
            thriving_count = sum(1 for echo in echoes if "perfect harmony" in echo['repair_action']['description'])
            maintenance_count = sum(1 for echo in echoes if "seasonal adjustment" in echo['repair_action']['description'])
            crisis_count = len(echoes) - thriving_count - maintenance_count
            
            if thriving_count > 0 or maintenance_count > 0:
                print(f"   Ecosystem states:")
                print(f"     Thriving (harmony): {thriving_count} ({thriving_count/len(echoes)*100:.1f}%)")
                print(f"     Maintenance: {maintenance_count} ({maintenance_count/len(echoes)*100:.1f}%)")
                print(f"     Crisis response: {crisis_count} ({crisis_count/len(echoes)*100:.1f}%)")
                
            # Silence analysis
            silence_values = [echo['repair_action']['silence_probability'] for echo in echoes]
            avg_silence = np.mean(silence_values) if silence_values else 0.0
            high_silence_count = sum(1 for s in silence_values if s > 0.8)
            high_silence_pct = (high_silence_count / len(echoes) * 100) if len(echoes) > 0 else 0.0
            print(f"   Average silence probability: {avg_silence:.3f}")
            print(f"   High silence (>0.8): {high_silence_count} ({high_silence_pct:.1f}%)")
            
        except FileNotFoundError:
            print(f"   ❌ Dataset file not found: {dataset_path}")
        except Exception as e:
            print(f"   ❌ Error analyzing dataset: {e}")

def main():
    """Main function to generate ecological training data"""
    
    # Fixed: Add reproducible seed option (o3's issue #8)
    generator = EcologicalDataGenerator(random_seed=42)
    
    print("🌍 Ecological Spiramycel Training Data Generator")
    print("=" * 60)
    
    # Generate datasets for controlled comparison
    datasets = [
        # Small datasets for quick testing
        (500, "ecological_small_chaotic.jsonl", True),
        (500, "ecological_small_calm.jsonl", False),
        
        # Medium datasets
        (2000, "ecological_medium_chaotic.jsonl", True),
        (2000, "ecological_medium_calm.jsonl", False),
        
        # Large datasets for serious training
        (5000, "ecological_large_chaotic.jsonl", True),
        (5000, "ecological_large_calm.jsonl", False)
    ]
    
    successful_datasets = 0
    start_time = datetime.now()
    
    for num_echoes, filename, chaos_mode in datasets:
        print(f"\n🎯 Generating {filename}...")
        try:
            result_path = generator.generate_training_dataset(num_echoes, filename, chaos_mode)
            if result_path:
                successful_datasets += 1
            else:
                print(f"❌ Failed to generate {filename}")
        except Exception as e:
            print(f"❌ Error generating {filename}: {e}")
    
    elapsed_time = datetime.now() - start_time
    
    print(f"\n✅ Generated {successful_datasets}/{len(datasets)} datasets successfully!")
    print(f"⏱ Total generation time: {elapsed_time.total_seconds():.1f} seconds")
    
    # Report any unknown seasonal combinations found
    if generator.unknown_seasonal_combos:
        print(f"\n🔍 Found {len(generator.unknown_seasonal_combos)} unknown seasonal combinations")
        print("   Consider adding these to baseline mappings for more accurate simulation")
    
    print("\nDatasets available:")
    print("  CHAOTIC: ecological_*_chaotic.jsonl")
    print("  CALM: ecological_*_calm.jsonl")
    print("\nThese ecological datasets complement the abstract training data")
    print("for comprehensive bioregional contemplative AI experiments!")

if __name__ == "__main__":
    main() 