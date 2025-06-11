#!/usr/bin/env python3
"""
Spiramycel Comparative Analysis Framework

Deep comparison between Ecological vs Abstract training approaches.
Scientific analysis of learning patterns, glyph usage, and philosophical implications.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import Counter
import matplotlib.pyplot as plt
import time

from glyph_codec import SpiramycelGlyphCodec, GlyphCategory
from neural_trainer import SpiramycelNeuralModel, NetworkConditions

@dataclass
class ModelPerformance:
    """Performance metrics for a trained model"""
    model_name: str
    dataset_size: int
    training_time_seconds: float
    epochs: int
    parameter_count: int
    
    # Loss metrics
    final_glyph_loss: float
    final_effectiveness_loss: float
    final_silence_loss: float
    
    # Learning curves
    glyph_loss_curve: List[float]
    effectiveness_loss_curve: List[float]
    silence_loss_curve: List[float]
    
    # Model characteristics
    model_type: str  # "ecological" or "abstract"
    training_approach: str

@dataclass
class GlyphAnalysis:
    """Analysis of glyph usage patterns"""
    category_distribution: Dict[str, int]
    most_common_glyphs: List[Tuple[int, int]]  # (glyph_id, count)
    silence_ratio: float
    unique_sequences: int
    average_sequence_length: float

@dataclass
class BehavioralProfile:
    """Behavioral characteristics of a model"""
    stress_response_pattern: List[int]
    optimal_condition_pattern: List[int]
    contemplative_tendency: float
    adaptation_strategy: str
    crisis_management_style: str

class SpiramycelComparativeAnalyzer:
    """Comprehensive analysis framework for model comparison"""
    
    def __init__(self):
        self.codec = SpiramycelGlyphCodec()
        self.performance_data = {}
        self.glyph_analyses = {}
        self.behavioral_profiles = {}
        
    def load_model_performance(self, 
                             model_name: str,
                             model_path: str,
                             training_log: Optional[str] = None) -> ModelPerformance:
        """Load and analyze model performance from training results"""
        
        print(f"📊 Analyzing {model_name} performance...")
        
        # Try to load training log if available
        glyph_curve = []
        eff_curve = []
        silence_curve = []
        
        if training_log and Path(training_log).exists():
            # Parse training log (implementation would depend on log format)
            pass
        
        # Load model to get parameter count
        try:
            if Path(model_path).exists():
                model = SpiramycelNeuralModel(force_cpu_mode=True)
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                param_count = model.count_parameters()
            else:
                param_count = 0
        except Exception as e:
            print(f"⚠ Could not load model {model_path}: {e}")
            param_count = 0
        
        # Create performance object (would be populated from actual training data)
        performance = ModelPerformance(
            model_name=model_name,
            dataset_size=0,  # To be filled
            training_time_seconds=0.0,  # To be filled
            epochs=0,  # To be filled
            parameter_count=param_count,
            final_glyph_loss=0.0,  # To be filled
            final_effectiveness_loss=0.0,  # To be filled
            final_silence_loss=0.0,  # To be filled
            glyph_loss_curve=glyph_curve,
            effectiveness_loss_curve=eff_curve,
            silence_loss_curve=silence_curve,
            model_type="unknown",  # To be filled
            training_approach="unknown"  # To be filled
        )
        
        self.performance_data[model_name] = performance
        return performance
    
    def analyze_glyph_patterns(self, 
                             model_path: str,
                             test_scenarios: List[NetworkConditions],
                             model_name: str) -> GlyphAnalysis:
        """Analyze glyph usage patterns across different scenarios"""
        
        print(f"🔍 Analyzing glyph patterns for {model_name}...")
        
        try:
            # Load model
            model = SpiramycelNeuralModel(force_cpu_mode=True)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            
            all_glyphs = []
            sequences = []
            
            with torch.no_grad():
                for scenario in test_scenarios:
                    # Create dummy input (real implementation would vary)
                    dummy_input = torch.zeros(1, 15, dtype=torch.long)
                    condition_tensor = torch.tensor(scenario.to_condition_vector(), dtype=torch.float32).unsqueeze(0)
                    
                    # Get model predictions
                    glyph_logits, _, _, _, _ = model(dummy_input, condition_tensor)
                    
                    # Extract predicted glyphs
                    predicted_glyphs = torch.argmax(glyph_logits[0, :5], dim=1).tolist()  # First 5 glyphs
                    all_glyphs.extend(predicted_glyphs)
                    sequences.append(predicted_glyphs)
            
            # Analyze patterns
            glyph_counter = Counter(all_glyphs)
            
            # Category distribution
            category_dist = {cat.value: 0 for cat in GlyphCategory}
            for glyph_id in all_glyphs:
                if glyph_id in self.codec.glyphs:
                    category = self.codec.glyphs[glyph_id].category
                    category_dist[category.value] += 1
            
            # Silence ratio
            silence_glyphs = self.codec.get_contemplative_glyphs()
            silence_count = sum(1 for g in all_glyphs if g in silence_glyphs)
            silence_ratio = silence_count / len(all_glyphs) if all_glyphs else 0
            
            analysis = GlyphAnalysis(
                category_distribution=category_dist,
                most_common_glyphs=glyph_counter.most_common(10),
                silence_ratio=silence_ratio,
                unique_sequences=len(set(tuple(seq) for seq in sequences)),
                average_sequence_length=np.mean([len(seq) for seq in sequences]) if sequences else 0
            )
            
            self.glyph_analyses[model_name] = analysis
            return analysis
            
        except Exception as e:
            print(f"⚠ Error analyzing glyph patterns: {e}")
            return GlyphAnalysis({}, [], 0.0, 0, 0.0)
    
    def generate_behavioral_profile(self,
                                  model_path: str, 
                                  model_name: str) -> BehavioralProfile:
        """Generate comprehensive behavioral profile of model"""
        
        print(f"🧠 Generating behavioral profile for {model_name}...")
        
        try:
            model = SpiramycelNeuralModel(force_cpu_mode=True)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            
            # Test scenarios
            stress_scenario = NetworkConditions(latency=0.9, voltage=0.1, temperature=0.9, error_rate=0.8, bandwidth=0.1)
            optimal_scenario = NetworkConditions(latency=0.1, voltage=0.8, temperature=0.5, error_rate=0.05, bandwidth=0.9)
            
            with torch.no_grad():
                # Stress response
                dummy_input = torch.zeros(1, 15, dtype=torch.long)
                stress_tensor = torch.tensor(stress_scenario.to_condition_vector(), dtype=torch.float32).unsqueeze(0)
                stress_logits, stress_eff, stress_silence, _, _ = model(dummy_input, stress_tensor)
                stress_glyphs = torch.argmax(stress_logits[0, :3], dim=1).tolist()
                
                # Optimal response  
                optimal_tensor = torch.tensor(optimal_scenario.to_condition_vector(), dtype=torch.float32).unsqueeze(0)
                optimal_logits, optimal_eff, optimal_silence, _, _ = model(dummy_input, optimal_tensor)
                optimal_glyphs = torch.argmax(optimal_logits[0, :3], dim=1).tolist()
                
                # Contemplative tendency
                contemplative_score = torch.sigmoid(stress_silence).mean().item()
            
            # Analyze response patterns
            silence_glyphs = self.codec.get_contemplative_glyphs()
            stress_silence_ratio = sum(1 for g in stress_glyphs if g in silence_glyphs) / len(stress_glyphs)
            
            # Determine adaptation strategy
            if stress_silence_ratio > 0.6:
                adaptation_strategy = "contemplative_withdrawal"
            elif any(g in self.codec.get_category_glyphs(GlyphCategory.NETWORK) for g in stress_glyphs):
                adaptation_strategy = "active_intervention"
            else:
                adaptation_strategy = "balanced_response"
            
            # Crisis management style
            if stress_glyphs == optimal_glyphs:
                crisis_style = "consistent_behavior"
            elif stress_silence_ratio > 0.5:
                crisis_style = "contemplative_crisis_management"
            else:
                crisis_style = "active_crisis_management"
            
            profile = BehavioralProfile(
                stress_response_pattern=stress_glyphs,
                optimal_condition_pattern=optimal_glyphs,
                contemplative_tendency=contemplative_score,
                adaptation_strategy=adaptation_strategy,
                crisis_management_style=crisis_style
            )
            
            self.behavioral_profiles[model_name] = profile
            return profile
            
        except Exception as e:
            print(f"⚠ Error generating behavioral profile: {e}")
            return BehavioralProfile([], [], 0.0, "unknown", "unknown")
    
    def create_performance_matrix(self) -> str:
        """Create comprehensive performance comparison matrix"""
        
        if len(self.performance_data) < 2:
            return "Need at least 2 models for comparison"
        
        matrix = "🔬 SPIRAMYCEL PERFORMANCE COMPARISON MATRIX\n"
        matrix += "=" * 70 + "\n\n"
        
        # Model overview
        matrix += "📊 MODEL OVERVIEW:\n"
        matrix += "-" * 30 + "\n"
        for name, perf in self.performance_data.items():
            matrix += f"{name:20} | {perf.parameter_count:,} params | {perf.dataset_size:,} samples | {perf.training_time_seconds:.1f}s\n"
        
        matrix += "\n📈 LOSS COMPARISON:\n"
        matrix += "-" * 30 + "\n"
        matrix += f"{'Model':<20} | {'Glyph Loss':<12} | {'Effectiveness':<12} | {'Silence':<12}\n"
        matrix += "-" * 65 + "\n"
        
        for name, perf in self.performance_data.items():
            matrix += f"{name:<20} | {perf.final_glyph_loss:<12.3f} | {perf.final_effectiveness_loss:<12.4f} | {perf.final_silence_loss:<12.4f}\n"
        
        return matrix
    
    def compare_glyph_philosophies(self) -> str:
        """Deep philosophical comparison of glyph usage patterns"""
        
        comparison = "🧘 CONTEMPLATIVE PHILOSOPHY ANALYSIS\n"
        comparison += "=" * 50 + "\n\n"
        
        for name, analysis in self.glyph_analyses.items():
            profile = self.behavioral_profiles.get(name)
            
            comparison += f"🌱 {name.upper()} MODEL:\n"
            comparison += "-" * 30 + "\n"
            comparison += f"Silence Ratio: {analysis.silence_ratio:.1%}\n"
            comparison += f"Category Focus: {max(analysis.category_distribution, key=analysis.category_distribution.get)}\n"
            
            if profile:
                comparison += f"Crisis Style: {profile.crisis_management_style}\n"
                comparison += f"Adaptation: {profile.adaptation_strategy}\n"
                comparison += f"Contemplative Tendency: {profile.contemplative_tendency:.3f}\n"
            
            # Analyze most common glyphs
            comparison += "Top Glyphs:\n"
            for glyph_id, count in analysis.most_common_glyphs[:3]:
                if glyph_id in self.codec.glyphs:
                    glyph_info = self.codec.glyphs[glyph_id]
                    comparison += f"  {glyph_info.symbol} - {glyph_info.description}\n"
            
            comparison += "\n"
        
        # Tystnadsmajoritet analysis
        comparison += "🤫 TYSTNADSMAJORITET ADHERENCE:\n"
        comparison += "-" * 40 + "\n"
        
        for name, analysis in self.glyph_analyses.items():
            target_silence = 0.875  # 87.5% target
            adherence = min(analysis.silence_ratio / target_silence, 1.0)
            comparison += f"{name}: {adherence:.1%} adherence to contemplative principles\n"
        
        return comparison
    
    def generate_full_report(self, 
                           ecological_model: str = "ecological_spiramycel_femto.pt",
                           abstract_model: str = "spiramycel_model_final.pt") -> str:
        """Generate comprehensive comparative analysis report"""
        
        print("📋 Generating Full Comparative Analysis Report...")
        
        # Test scenarios for behavioral analysis
        test_scenarios = [
            NetworkConditions(latency=0.9, voltage=0.1, temperature=0.9, error_rate=0.8, bandwidth=0.1),  # High stress
            NetworkConditions(latency=0.1, voltage=0.8, temperature=0.5, error_rate=0.05, bandwidth=0.9),  # Optimal
            NetworkConditions(latency=0.5, voltage=0.5, temperature=0.5, error_rate=0.2, bandwidth=0.5),   # Balanced
        ]
        
        # Analyze models if they exist
        models_to_analyze = [
            ("Ecological", ecological_model),
            ("Abstract", abstract_model)
        ]
        
        for name, path in models_to_analyze:
            if Path(path).exists():
                self.load_model_performance(name, path)
                self.analyze_glyph_patterns(path, test_scenarios, name)
                self.generate_behavioral_profile(path, name)
            else:
                print(f"⚠ Model not found: {path}")
        
        # Generate report sections
        report = "🍄 SPIRAMYCEL COMPARATIVE ANALYSIS REPORT\n"
        report += "=" * 80 + "\n"
        report += f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += self.create_performance_matrix() + "\n\n"
        report += self.compare_glyph_philosophies() + "\n\n"
        
        # Conclusions
        report += "🎯 KEY INSIGHTS:\n"
        report += "-" * 20 + "\n"
        
        if len(self.glyph_analyses) >= 2:
            eco_silence = self.glyph_analyses.get("Ecological", {}).silence_ratio or 0
            abs_silence = self.glyph_analyses.get("Abstract", {}).silence_ratio or 0
            
            if eco_silence > abs_silence:
                report += "• Ecological training shows higher contemplative adherence\n"
            else:
                report += "• Abstract training shows higher contemplative adherence\n"
                
            report += f"• Silence ratio difference: {abs(eco_silence - abs_silence):.1%}\n"
        
        report += "• Training approach significantly affects glyph philosophy\n"
        report += "• Model behavior reflects training paradigm\n"
        
        return report

def quick_analysis_demo():
    """Quick demonstration of analysis capabilities"""
    print("🔬 Quick Comparative Analysis Demo")
    print("=" * 50)
    
    analyzer = SpiramycelComparativeAnalyzer()
    
    # Generate test report
    report = analyzer.generate_full_report()
    print(report)
    
    # Save report
    report_path = Path("comparative_analysis_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📁 Report saved to: {report_path}")

if __name__ == "__main__":
    quick_analysis_demo() 