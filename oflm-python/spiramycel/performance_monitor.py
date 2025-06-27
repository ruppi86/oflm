#!/usr/bin/env python3
"""
Real-time Spiramycel Performance Monitor

Live dashboard for tracking training progress and comparing models.
"""

import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class TrainingMetrics:
    """Real-time training metrics"""
    timestamp: float
    epoch: int
    glyph_loss: float
    effectiveness_loss: float
    silence_loss: float
    total_loss: float
    learning_rate: float
    model_type: str

class SpiramycelPerformanceMonitor:
    """Real-time monitoring dashboard for training comparison"""
    
    def __init__(self, update_interval: float = 5.0):
        self.update_interval = update_interval
        self.metrics_history = {}
        self.active_models = set()
        self.monitoring = False
        self.monitor_thread = None
        
        # Performance thresholds
        self.excellent_glyph_loss = 0.5
        self.good_glyph_loss = 1.0
        self.target_silence_ratio = 0.875
        
    def start_monitoring(self, models_to_track: List[str]):
        """Start real-time monitoring of specified models"""
        print(f"🔍 Starting performance monitoring for: {', '.join(models_to_track)}")
        
        self.active_models = set(models_to_track)
        for model in models_to_track:
            self.metrics_history[model] = []
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring and save results"""
        print("⏹️ Stopping performance monitoring...")
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        # Save monitoring results
        self._save_monitoring_data()
        
    def add_training_metric(self, 
                          model_name: str,
                          epoch: int,
                          glyph_loss: float,
                          effectiveness_loss: float,
                          silence_loss: float,
                          learning_rate: float = 0.001):
        """Add new training metric point"""
        
        metric = TrainingMetrics(
            timestamp=time.time(),
            epoch=epoch,
            glyph_loss=glyph_loss,
            effectiveness_loss=effectiveness_loss,
            silence_loss=silence_loss,
            total_loss=glyph_loss + effectiveness_loss + silence_loss,
            learning_rate=learning_rate,
            model_type=self._infer_model_type(model_name)
        )
        
        if model_name not in self.metrics_history:
            self.metrics_history[model_name] = []
        
        self.metrics_history[model_name].append(metric)
        
    def _infer_model_type(self, model_name: str) -> str:
        """Infer model type from name"""
        name_lower = model_name.lower()
        if 'ecological' in name_lower or 'eco' in name_lower:
            return 'ecological'
        elif 'abstract' in name_lower or 'abs' in name_lower:
            return 'abstract'
        else:
            return 'unknown'
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                self._update_display()
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"⚠ Monitoring error: {e}")
    
    def _update_display(self):
        """Update real-time display"""
        print("\n" + "="*80)
        print(f"🔍 SPIRAMYCEL TRAINING MONITOR - {datetime.now().strftime('%H:%M:%S')}")
        print("="*80)
        
        if not self.metrics_history:
            print("No training data available yet...")
            return
        
        # Current status for each model
        for model_name in self.active_models:
            if model_name in self.metrics_history and self.metrics_history[model_name]:
                latest = self.metrics_history[model_name][-1]
                
                print(f"\n🤖 {model_name.upper()} ({latest.model_type})")
                print("-" * 40)
                print(f"Epoch: {latest.epoch}")
                print(f"Glyph Loss: {latest.glyph_loss:.4f} {self._get_loss_indicator(latest.glyph_loss)}")
                print(f"Effectiveness: {latest.effectiveness_loss:.4f}")
                print(f"Silence: {latest.silence_loss:.4f}")
                print(f"Total Loss: {latest.total_loss:.4f}")
                print(f"Learning Rate: {latest.learning_rate:.6f}")
                
                # Progress indicators
                if len(self.metrics_history[model_name]) > 1:
                    prev = self.metrics_history[model_name][-2]
                    glyph_trend = "📈" if latest.glyph_loss > prev.glyph_loss else "📉"
                    print(f"Trend: {glyph_trend}")
        
        # Comparative analysis if multiple models
        if len([m for m in self.active_models if m in self.metrics_history]) >= 2:
            self._show_comparative_status()
    
    def _get_loss_indicator(self, loss: float) -> str:
        """Get visual indicator for loss quality"""
        if loss < self.excellent_glyph_loss:
            return "🟢 Excellent"
        elif loss < self.good_glyph_loss:
            return "🟡 Good"
        else:
            return "🔴 Needs Work"
    
    def _show_comparative_status(self):
        """Show comparative status between models"""
        print("\n🔄 COMPARATIVE STATUS:")
        print("-" * 30)
        
        # Get latest metrics for comparison
        latest_metrics = {}
        for model in self.active_models:
            if model in self.metrics_history and self.metrics_history[model]:
                latest_metrics[model] = self.metrics_history[model][-1]
        
        if len(latest_metrics) >= 2:
            # Find best performing model
            best_glyph = min(latest_metrics.items(), key=lambda x: x[1].glyph_loss)
            best_total = min(latest_metrics.items(), key=lambda x: x[1].total_loss)
            
            print(f"🏆 Best Glyph Loss: {best_glyph[0]} ({best_glyph[1].glyph_loss:.4f})")
            print(f"🏆 Best Total Loss: {best_total[0]} ({best_total[1].total_loss:.4f})")
            
            # Show differences
            if len(latest_metrics) == 2:
                models = list(latest_metrics.keys())
                m1, m2 = models[0], models[1]
                glyph_diff = abs(latest_metrics[m1].glyph_loss - latest_metrics[m2].glyph_loss)
                print(f"📊 Glyph Loss Difference: {glyph_diff:.4f}")
    
    def _save_monitoring_data(self):
        """Save monitoring data to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"monitoring_data_{timestamp}.json"
        
        # Convert to serializable format
        data = {}
        for model, metrics in self.metrics_history.items():
            data[model] = [asdict(m) for m in metrics]
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"📁 Monitoring data saved to: {filename}")
    
    def generate_performance_plots(self, output_dir: str = "plots"):
        """Generate performance comparison plots"""
        print("📊 Generating performance plots...")
        
        Path(output_dir).mkdir(exist_ok=True)
        
        if len(self.metrics_history) < 1:
            print("No data to plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Spiramycel Training Comparison', fontsize=16)
        
        # Plot 1: Glyph Loss over time
        ax1 = axes[0, 0]
        for model, metrics in self.metrics_history.items():
            if metrics:
                epochs = [m.epoch for m in metrics]
                losses = [m.glyph_loss for m in metrics]
                ax1.plot(epochs, losses, marker='o', label=f"{model} ({metrics[0].model_type})")
        
        ax1.set_title('Glyph Loss Over Training')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Glyph Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Total Loss comparison
        ax2 = axes[0, 1]
        for model, metrics in self.metrics_history.items():
            if metrics:
                epochs = [m.epoch for m in metrics]
                total_losses = [m.total_loss for m in metrics]
                ax2.plot(epochs, total_losses, marker='s', label=f"{model}")
        
        ax2.set_title('Total Loss Over Training')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Total Loss')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Silence Loss (Contemplative adherence)
        ax3 = axes[1, 0]
        for model, metrics in self.metrics_history.items():
            if metrics:
                epochs = [m.epoch for m in metrics]
                silence_losses = [m.silence_loss for m in metrics]
                ax3.plot(epochs, silence_losses, marker='^', label=f"{model}")
        
        ax3.set_title('Silence Loss (Contemplative Adherence)')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Silence Loss')
        ax3.legend()
        ax3.grid(True)
        
        # Plot 4: Learning curves comparison
        ax4 = axes[1, 1]
        
        # Create performance score (inverse of average loss)
        for model, metrics in self.metrics_history.items():
            if metrics and len(metrics) > 1:
                epochs = [m.epoch for m in metrics]
                # Performance score = 1 / (1 + total_loss)
                performance = [1 / (1 + m.total_loss) for m in metrics]
                ax4.plot(epochs, performance, marker='D', label=f"{model}")
        
        ax4.set_title('Training Performance Score')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Performance Score (Higher = Better)')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plot_path = Path(output_dir) / f"training_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Plots saved to: {plot_path}")
        
    def get_training_summary(self) -> str:
        """Get comprehensive training summary"""
        if not self.metrics_history:
            return "No training data available"
        
        summary = "🍄 SPIRAMYCEL TRAINING SUMMARY\n"
        summary += "=" * 50 + "\n\n"
        
        for model, metrics in self.metrics_history.items():
            if not metrics:
                continue
                
            summary += f"🤖 {model.upper()} ({metrics[0].model_type})\n"
            summary += "-" * 30 + "\n"
            
            # Training progress
            first_metric = metrics[0]
            last_metric = metrics[-1]
            
            summary += f"Training Progress: {first_metric.epoch} → {last_metric.epoch} epochs\n"
            summary += f"Glyph Loss: {first_metric.glyph_loss:.4f} → {last_metric.glyph_loss:.4f} "
            
            glyph_improvement = ((first_metric.glyph_loss - last_metric.glyph_loss) / first_metric.glyph_loss) * 100
            summary += f"({glyph_improvement:+.1f}%)\n"
            
            summary += f"Silence Loss: {first_metric.silence_loss:.4f} → {last_metric.silence_loss:.4f}\n"
            summary += f"Total Loss: {first_metric.total_loss:.4f} → {last_metric.total_loss:.4f}\n"
            
            # Training velocity
            if len(metrics) > 1:
                training_time = last_metric.timestamp - first_metric.timestamp
                epochs_per_minute = (last_metric.epoch - first_metric.epoch) / (training_time / 60)
                summary += f"Training Velocity: {epochs_per_minute:.1f} epochs/minute\n"
            
            summary += "\n"
        
        # Comparative insights
        if len(self.metrics_history) >= 2:
            summary += "🔄 COMPARATIVE INSIGHTS:\n"
            summary += "-" * 25 + "\n"
            
            # Get final performance
            final_performances = {}
            for model, metrics in self.metrics_history.items():
                if metrics:
                    final_performances[model] = metrics[-1]
            
            if len(final_performances) >= 2:
                best_glyph = min(final_performances.items(), key=lambda x: x[1].glyph_loss)
                summary += f"🏆 Best Glyph Performance: {best_glyph[0]}\n"
                
                # Model type comparison
                ecological_models = [m for m, metrics in self.metrics_history.items() 
                                  if metrics and metrics[0].model_type == 'ecological']
                abstract_models = [m for m, metrics in self.metrics_history.items() 
                                if metrics and metrics[0].model_type == 'abstract']
                
                if ecological_models and abstract_models:
                    eco_avg_loss = np.mean([self.metrics_history[m][-1].glyph_loss for m in ecological_models])
                    abs_avg_loss = np.mean([self.metrics_history[m][-1].glyph_loss for m in abstract_models])
                    
                    if eco_avg_loss < abs_avg_loss:
                        summary += "🌱 Ecological training shows superior performance\n"
                    else:
                        summary += "🔬 Abstract training shows superior performance\n"
                    
                    summary += f"Performance gap: {abs(eco_avg_loss - abs_avg_loss):.4f}\n"
        
        return summary

def quick_monitor_demo():
    """Quick demonstration of monitoring capabilities"""
    print("🔍 Performance Monitor Demo")
    print("=" * 40)
    
    monitor = SpiramycelPerformanceMonitor(update_interval=2.0)
    
    # Simulate some training data
    models = ["Ecological_Model", "Abstract_Model"]
    monitor.start_monitoring(models)
    
    # Simulate training progress
    for epoch in range(1, 6):
        # Ecological model (improving faster)
        eco_glyph = 3.0 - (epoch * 0.5) + np.random.normal(0, 0.1)
        eco_eff = 0.05 - (epoch * 0.008) + np.random.normal(0, 0.005)
        eco_silence = 0.3 - (epoch * 0.05) + np.random.normal(0, 0.02)
        
        monitor.add_training_metric("Ecological_Model", epoch, eco_glyph, eco_eff, eco_silence)
        
        # Abstract model (slower improvement)
        abs_glyph = 3.0 - (epoch * 0.3) + np.random.normal(0, 0.1)
        abs_eff = 0.05 - (epoch * 0.005) + np.random.normal(0, 0.005)
        abs_silence = 0.3 - (epoch * 0.03) + np.random.normal(0, 0.02)
        
        monitor.add_training_metric("Abstract_Model", epoch, abs_glyph, abs_eff, abs_silence)
        
        time.sleep(3)  # Simulate training time
    
    # Generate final report
    summary = monitor.get_training_summary()
    print(summary)
    
    monitor.stop_monitoring()
    monitor.generate_performance_plots()

if __name__ == "__main__":
    quick_monitor_demo() 