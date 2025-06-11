#!/usr/bin/env python3
"""
Unified Spiramycel Analysis Orchestrator

Coordinates all comparative analysis tools for comprehensive evaluation:
- Real-time performance monitoring
- Comparative analysis framework
- Philosophical implications framework
- Behavioral difference analysis
"""

import json
import asyncio
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from comparative_analysis import SpiramycelComparativeAnalyzer
from performance_monitor import SpiramycelPerformanceMonitor, TrainingMetrics
from philosophical_framework import SpiramycelPhilosophicalFramework

@dataclass
class UnifiedAnalysisResults:
    """Complete analysis results"""
    timestamp: str
    performance_summary: str
    comparative_matrix: str
    philosophical_report: str
    behavioral_analysis: Dict
    key_insights: List[str]
    recommendations: List[str]

class SpiramycelUnifiedAnalyzer:
    """Orchestrates all analysis components for comprehensive evaluation"""
    
    def __init__(self, monitoring_interval: float = 10.0):
        self.performance_monitor = SpiramycelPerformanceMonitor(monitoring_interval)
        self.comparative_analyzer = SpiramycelComparativeAnalyzer()
        self.philosophical_framework = SpiramycelPhilosophicalFramework()
        
        self.is_analyzing = False
        self.analysis_thread = None
        self.results_history = []
        
    def start_unified_analysis(self, 
                             ecological_model_path: str = "ecological_spiramycel_femto.pt",
                             abstract_model_path: str = "spiramycel_model_final.pt",
                             enable_realtime_monitoring: bool = True):
        """Start comprehensive analysis of both models"""
        
        print("🔬 STARTING UNIFIED SPIRAMYCEL ANALYSIS")
        print("=" * 60)
        print(f"Ecological Model: {ecological_model_path}")
        print(f"Abstract Model: {abstract_model_path}")
        print(f"Real-time Monitoring: {enable_realtime_monitoring}")
        print("=" * 60)
        
        self.ecological_path = ecological_model_path
        self.abstract_path = abstract_model_path
        
        # Start real-time monitoring if enabled
        if enable_realtime_monitoring:
            models_to_monitor = []
            if Path(ecological_model_path).exists():
                models_to_monitor.append("Ecological_Model")
            if Path(abstract_model_path).exists():
                models_to_monitor.append("Abstract_Model")
            
            if models_to_monitor:
                self.performance_monitor.start_monitoring(models_to_monitor)
        
        # Start analysis thread
        self.is_analyzing = True
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()
        
        print("🚀 Unified analysis started! Real-time monitoring active.")
        
    def stop_unified_analysis(self) -> UnifiedAnalysisResults:
        """Stop analysis and generate final comprehensive report"""
        
        print("🏁 Stopping unified analysis and generating final report...")
        
        self.is_analyzing = False
        if self.analysis_thread:
            self.analysis_thread.join()
        
        self.performance_monitor.stop_monitoring()
        
        # Generate comprehensive final analysis
        final_results = self._generate_comprehensive_analysis()
        
        # Save results
        self._save_unified_results(final_results)
        
        return final_results
    
    def add_training_update(self, 
                          model_name: str,
                          epoch: int,
                          glyph_loss: float,
                          effectiveness_loss: float,
                          silence_loss: float,
                          learning_rate: float = 0.001):
        """Add training update to monitoring system"""
        
        self.performance_monitor.add_training_metric(
            model_name, epoch, glyph_loss, effectiveness_loss, silence_loss, learning_rate
        )
    
    def _analysis_loop(self):
        """Main analysis loop for periodic comprehensive updates"""
        
        analysis_count = 0
        
        while self.is_analyzing:
            try:
                analysis_count += 1
                print(f"\n🔄 Running Analysis Cycle #{analysis_count}")
                print("-" * 40)
                
                # Generate intermediate analysis
                intermediate_results = self._generate_intermediate_analysis()
                
                if intermediate_results:
                    print("📊 Intermediate Analysis Complete")
                    print(f"⏰ Next analysis in {self.performance_monitor.update_interval * 3:.0f}s")
                
                # Wait for next cycle (3x monitoring interval for less frequent comprehensive analysis)
                time.sleep(self.performance_monitor.update_interval * 3)
                
            except Exception as e:
                print(f"⚠ Analysis loop error: {e}")
                time.sleep(30)  # Wait before retrying
    
    def _generate_intermediate_analysis(self) -> Optional[Dict]:
        """Generate intermediate analysis during training"""
        
        # Check if we have sufficient data
        if not self.performance_monitor.metrics_history:
            return None
        
        # Quick performance snapshot
        performance_summary = self.performance_monitor.get_training_summary()
        
        # Basic comparative insights
        current_metrics = {}
        for model_name, metrics_list in self.performance_monitor.metrics_history.items():
            if metrics_list:
                latest = metrics_list[-1]
                current_metrics[model_name] = {
                    'final_glyph_loss': latest.glyph_loss,
                    'final_effectiveness_loss': latest.effectiveness_loss,
                    'final_silence_loss': latest.silence_loss,
                    'silence_ratio': 0.875 - latest.silence_loss,  # Estimate
                    'model_type': latest.model_type
                }
        
        # Quick insights
        insights = []
        if len(current_metrics) >= 2:
            models = list(current_metrics.keys())
            if len(models) == 2:
                m1, m2 = models
                glyph_diff = abs(current_metrics[m1]['final_glyph_loss'] - current_metrics[m2]['final_glyph_loss'])
                
                if current_metrics[m1]['final_glyph_loss'] < current_metrics[m2]['final_glyph_loss']:
                    insights.append(f"🏆 {m1} currently leading with {glyph_diff:.4f} glyph loss advantage")
                else:
                    insights.append(f"🏆 {m2} currently leading with {glyph_diff:.4f} glyph loss advantage")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'performance_summary': performance_summary,
            'current_metrics': current_metrics,
            'insights': insights
        }
    
    def _generate_comprehensive_analysis(self) -> UnifiedAnalysisResults:
        """Generate complete comprehensive analysis"""
        
        print("📋 Generating comprehensive analysis...")
        
        # Collect training results from monitoring
        training_results = {}
        model_behaviors = {}
        
        for model_name, metrics_list in self.performance_monitor.metrics_history.items():
            if metrics_list:
                first_metric = metrics_list[0]
                last_metric = metrics_list[-1]
                
                # Calculate improvement
                glyph_improvement = 0
                if first_metric.glyph_loss > 0:
                    glyph_improvement = ((first_metric.glyph_loss - last_metric.glyph_loss) / first_metric.glyph_loss) * 100
                
                training_results[model_name] = {
                    'final_glyph_loss': last_metric.glyph_loss,
                    'final_effectiveness_loss': last_metric.effectiveness_loss,
                    'final_silence_loss': last_metric.silence_loss,
                    'silence_ratio': max(0, 0.875 - last_metric.silence_loss),  # Estimate based on target
                    'glyph_improvement_percent': glyph_improvement,
                    'training_epochs': last_metric.epoch,
                    'model_type': last_metric.model_type
                }
                
                # Infer behavioral characteristics
                avg_glyph = sum(m.glyph_loss for m in metrics_list) / len(metrics_list)
                avg_silence = sum(m.silence_loss for m in metrics_list) / len(metrics_list)
                
                if avg_silence < 0.1:
                    adaptation_style = "contemplative_adaptation"
                elif avg_glyph < 1.0:
                    adaptation_style = "efficient_optimization"
                else:
                    adaptation_style = "gradual_learning"
                
                model_behaviors[model_name] = {
                    'adaptation_style': adaptation_style,
                    'stability': 'stable' if len(metrics_list) > 5 else 'limited_data',
                    'learning_velocity': len(metrics_list) / ((last_metric.timestamp - first_metric.timestamp) / 60)  # epochs per minute
                }
        
        # Generate component analyses
        print("📊 Running comparative analysis...")
        comparative_matrix = self.comparative_analyzer.create_performance_matrix()
        
        print("🧘 Running philosophical analysis...")
        philosophical_insights = self.philosophical_framework.analyze_training_philosophy(
            training_results, model_behaviors
        )
        philosophical_report = self.philosophical_framework.generate_contemplative_report()
        
        print("📈 Generating performance summary...")
        performance_summary = self.performance_monitor.get_training_summary()
        
        # Generate key insights
        key_insights = self._extract_key_insights(training_results, philosophical_insights)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(training_results, model_behaviors)
        
        results = UnifiedAnalysisResults(
            timestamp=datetime.now().isoformat(),
            performance_summary=performance_summary,
            comparative_matrix=comparative_matrix,
            philosophical_report=philosophical_report,
            behavioral_analysis=model_behaviors,
            key_insights=key_insights,
            recommendations=recommendations
        )
        
        self.results_history.append(results)
        return results
    
    def _extract_key_insights(self, training_results: Dict, philosophical_insights: List) -> List[str]:
        """Extract the most important insights from all analyses"""
        
        insights = []
        
        # Performance insights
        if len(training_results) >= 2:
            models = list(training_results.keys())
            if len(models) == 2:
                m1, m2 = models
                glyph1 = training_results[m1]['final_glyph_loss']
                glyph2 = training_results[m2]['final_glyph_loss']
                
                better_model = m1 if glyph1 < glyph2 else m2
                improvement_diff = abs(glyph1 - glyph2)
                
                insights.append(f"🏆 {better_model} achieved superior performance with {improvement_diff:.4f} glyph loss advantage")
                
                # Silence comparison
                silence1 = training_results[m1]['silence_ratio']
                silence2 = training_results[m2]['silence_ratio']
                silence_diff = abs(silence1 - silence2)
                
                if silence_diff > 0.1:
                    higher_silence = m1 if silence1 > silence2 else m2
                    insights.append(f"🤫 {higher_silence} shows {silence_diff:.1%} higher contemplative adherence")
        
        # Philosophical insights
        if philosophical_insights:
            highest_insight = max(philosophical_insights, key=lambda x: x.contemplative_score)
            insights.append(f"🧘 Deepest insight: {highest_insight.insight_text[:100]}...")
        
        # Training efficiency insights
        for model, results in training_results.items():
            improvement = results['glyph_improvement_percent']
            if improvement > 80:
                insights.append(f"⚡ {model} showed exceptional {improvement:.1f}% improvement")
            elif improvement > 50:
                insights.append(f"📈 {model} demonstrated solid {improvement:.1f}% improvement")
        
        return insights[:5]  # Top 5 insights
    
    def _generate_recommendations(self, training_results: Dict, model_behaviors: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        
        recommendations = []
        
        if len(training_results) >= 2:
            # Compare approaches
            eco_models = [m for m, r in training_results.items() if r.get('model_type') == 'ecological']
            abs_models = [m for m, r in training_results.items() if r.get('model_type') == 'abstract']
            
            if eco_models and abs_models:
                eco_performance = sum(training_results[m]['final_glyph_loss'] for m in eco_models) / len(eco_models)
                abs_performance = sum(training_results[m]['final_glyph_loss'] for m in abs_models) / len(abs_models)
                
                if eco_performance < abs_performance:
                    recommendations.append("🌱 Consider prioritizing ecological training approaches for future models")
                    recommendations.append("🔄 Investigate hybrid approaches combining ecological grounding with abstract optimization")
                else:
                    recommendations.append("🔬 Abstract approaches show promise - consider scaling up symbolic training")
                    recommendations.append("🧘 Integrate more contemplative elements to improve abstract model wisdom")
        
        # Silence-based recommendations
        for model, results in training_results.items():
            silence_ratio = results['silence_ratio']
            if silence_ratio < 0.5:
                recommendations.append(f"🤫 {model} needs more contemplative space - consider silence loss weighting")
            elif silence_ratio > 0.9:
                recommendations.append(f"⚖️ {model} might benefit from more active learning balance")
        
        # Performance-based recommendations
        best_model = min(training_results.items(), key=lambda x: x[1]['final_glyph_loss'])
        recommendations.append(f"🏆 Analyze {best_model[0]} architecture for transferable insights")
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _save_unified_results(self, results: UnifiedAnalysisResults):
        """Save complete analysis results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON data
        json_path = f"unified_analysis_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(results), f, indent=2)
        
        # Save human-readable report
        report_path = f"unified_report_{timestamp}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("🍄 SPIRAMYCEL UNIFIED ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {results.timestamp}\n\n")
            
            f.write("📊 PERFORMANCE SUMMARY:\n")
            f.write("=" * 30 + "\n")
            f.write(results.performance_summary + "\n\n")
            
            f.write("🔬 COMPARATIVE ANALYSIS:\n") 
            f.write("=" * 30 + "\n")
            f.write(results.comparative_matrix + "\n\n")
            
            f.write("🧘 PHILOSOPHICAL IMPLICATIONS:\n")
            f.write("=" * 35 + "\n")
            f.write(results.philosophical_report + "\n\n")
            
            f.write("🎯 KEY INSIGHTS:\n")
            f.write("=" * 20 + "\n")
            for insight in results.key_insights:
                f.write(f"• {insight}\n")
            f.write("\n")
            
            f.write("💡 RECOMMENDATIONS:\n")
            f.write("=" * 25 + "\n")
            for rec in results.recommendations:
                f.write(f"• {rec}\n")
        
        print(f"📁 Unified analysis saved:")
        print(f"   JSON: {json_path}")
        print(f"   Report: {report_path}")
    
    def get_live_status(self) -> str:
        """Get current live analysis status"""
        
        if not self.is_analyzing:
            return "Analysis not active"
        
        status = "🔄 LIVE ANALYSIS STATUS\n"
        status += "=" * 30 + "\n"
        
        # Monitoring status
        if self.performance_monitor.monitoring:
            status += f"📊 Monitoring: {len(self.performance_monitor.active_models)} models\n"
            
            for model in self.performance_monitor.active_models:
                if model in self.performance_monitor.metrics_history:
                    metrics = self.performance_monitor.metrics_history[model]
                    if metrics:
                        latest = metrics[-1]
                        status += f"   {model}: Epoch {latest.epoch}, Loss {latest.glyph_loss:.4f}\n"
        else:
            status += "📊 Monitoring: Inactive\n"
        
        status += f"🧘 Philosophical Analysis: {'Active' if self.is_analyzing else 'Inactive'}\n"
        status += f"📈 Results History: {len(self.results_history)} complete analyses\n"
        
        return status

def unified_analysis_demo():
    """Demonstration of unified analysis system"""
    print("🔬 Unified Analysis System Demo")
    print("=" * 50)
    
    analyzer = SpiramycelUnifiedAnalyzer(monitoring_interval=5.0)
    
    # Start analysis (in demo mode, models may not exist)
    analyzer.start_unified_analysis(
        ecological_model_path="demo_ecological.pt",
        abstract_model_path="demo_abstract.pt",
        enable_realtime_monitoring=True
    )
    
    # Simulate some training updates
    print("\n📚 Simulating training updates...")
    
    for epoch in range(1, 11):
        # Ecological model (improving faster)
        eco_glyph = 3.0 - (epoch * 0.3) + (0.1 * (epoch % 3))
        eco_eff = 0.05 - (epoch * 0.003)
        eco_silence = 0.3 - (epoch * 0.025)
        
        analyzer.add_training_update("Ecological_Model", epoch, eco_glyph, eco_eff, eco_silence)
        
        # Abstract model (different pattern)
        abs_glyph = 3.0 - (epoch * 0.2) + (0.05 * (epoch % 4))
        abs_eff = 0.05 - (epoch * 0.002)
        abs_silence = 0.3 - (epoch * 0.015)
        
        analyzer.add_training_update("Abstract_Model", epoch, abs_glyph, abs_eff, abs_silence)
        
        if epoch % 3 == 0:
            print(f"   Epoch {epoch} data added...")
        
        time.sleep(1)  # Simulate training time
    
    # Show live status
    print("\n" + analyzer.get_live_status())
    
    # Let analysis run for a bit
    print("\n⏳ Letting analysis run for 15 seconds...")
    time.sleep(15)
    
    # Generate final report
    print("\n🏁 Generating final comprehensive report...")
    final_results = analyzer.stop_unified_analysis()
    
    print("\n📋 FINAL ANALYSIS COMPLETE!")
    print("=" * 40)
    print(f"Key Insights: {len(final_results.key_insights)}")
    print(f"Recommendations: {len(final_results.recommendations)}")
    
    # Show key insights
    print("\n🎯 TOP INSIGHTS:")
    for insight in final_results.key_insights[:3]:
        print(f"• {insight}")

if __name__ == "__main__":
    unified_analysis_demo() 