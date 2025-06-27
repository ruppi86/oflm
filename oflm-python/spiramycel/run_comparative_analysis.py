#!/usr/bin/env python3
"""
Spiramycel Comparative Analysis Runner

Simple script to start comprehensive analysis while training is happening.
Run this alongside your training to get real-time comparative insights.
"""

import sys
import time
import signal
from pathlib import Path

# Try to import our analysis components
try:
    from unified_analysis import SpiramycelUnifiedAnalyzer
    from comparative_analysis import SpiramycelComparativeAnalyzer
    from performance_monitor import SpiramycelPerformanceMonitor
    from philosophical_framework import SpiramycelPhilosophicalFramework
    print("✅ All analysis components loaded successfully!")
except ImportError as e:
    print(f"⚠ Import error: {e}")
    print("Make sure you're running from the spiramycel directory with all components available.")
    sys.exit(1)

class ComparativeAnalysisRunner:
    """Simple runner for comparative analysis during training"""
    
    def __init__(self):
        self.analyzer = None
        self.running = False
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n\n🛑 Graceful shutdown requested...")
        self.stop_analysis()
        sys.exit(0)
        
    def find_models(self):
        """Find available Spiramycel models"""
        
        print("🔍 Searching for Spiramycel models...")
        
        # Look for ecological model
        ecological_paths = [
            "ecological_spiramycel_femto.pt",
            "oflm-python/spiramycel/ecological_spiramycel_femto.pt",
            "../ecological_spiramycel_femto.pt"
        ]
        
        ecological_model = None
        for path in ecological_paths:
            if Path(path).exists():
                ecological_model = path
                print(f"📈 Found ecological model: {path}")
                break
        
        # Look for abstract model
        abstract_paths = [
            "spiramycel_model_final.pt",
            "abstract_spiramycel_model.pt",
            "oflm-python/spiramycel/spiramycel_model_final.pt",
            "../spiramycel_model_final.pt"
        ]
        
        abstract_model = None
        for path in abstract_paths:
            if Path(path).exists():
                abstract_model = path
                print(f"🔬 Found abstract model: {path}")
                break
        
        if not ecological_model and not abstract_model:
            print("⚠ No Spiramycel models found!")
            print("Expected to find one of:")
            print("  - ecological_spiramycel_femto.pt")
            print("  - spiramycel_model_final.pt")
            return None, None
        
        return ecological_model, abstract_model
    
    def start_analysis(self, monitoring_interval: float = 15.0):
        """Start the comparative analysis"""
        
        print("🚀 STARTING SPIRAMYCEL COMPARATIVE ANALYSIS")
        print("=" * 60)
        
        # Find models
        ecological_model, abstract_model = self.find_models()
        
        if not ecological_model and not abstract_model:
            print("❌ Cannot start analysis without at least one model")
            return False
        
        # Create analyzer
        self.analyzer = SpiramycelUnifiedAnalyzer(monitoring_interval=monitoring_interval)
        
        # Start unified analysis
        try:
            self.analyzer.start_unified_analysis(
                ecological_model_path=ecological_model or "dummy_eco.pt",
                abstract_model_path=abstract_model or "dummy_abs.pt",
                enable_realtime_monitoring=True
            )
            
            self.running = True
            
            print("\n🔄 Analysis running! Key features:")
            print("• Real-time performance monitoring")
            print("• Comparative glyph pattern analysis")
            print("• Philosophical implications framework")
            print("• Behavioral difference analysis")
            print("\n💡 You can now run your training in another terminal.")
            print("📊 Analysis will automatically detect and compare training progress.")
            print("\n⌨️  Press Ctrl+C to stop analysis and generate final report")
            
            return True
            
        except Exception as e:
            print(f"❌ Error starting analysis: {e}")
            return False
    
    def run_monitoring_loop(self):
        """Main monitoring loop with status updates"""
        
        if not self.analyzer or not self.running:
            print("❌ Analysis not started")
            return
        
        try:
            update_count = 0
            
            while self.running:
                update_count += 1
                
                # Show status every few cycles
                if update_count % 3 == 0:
                    print(f"\n⏰ Status Update #{update_count}")
                    print("-" * 30)
                    status = self.analyzer.get_live_status()
                    print(status)
                    
                    # Show any current insights
                    if hasattr(self.analyzer.performance_monitor, 'metrics_history'):
                        if self.analyzer.performance_monitor.metrics_history:
                            print("\n📈 Quick Insights:")
                            
                            for model_name, metrics in self.analyzer.performance_monitor.metrics_history.items():
                                if metrics:
                                    latest = metrics[-1]
                                    trend = "📉 Improving" if len(metrics) > 1 and latest.glyph_loss < metrics[-2].glyph_loss else "📊 Stable"
                                    print(f"   {model_name}: {trend}, Loss: {latest.glyph_loss:.4f}")
                
                # Sleep for monitoring interval
                time.sleep(self.analyzer.performance_monitor.update_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring interrupted")
        
    def stop_analysis(self):
        """Stop analysis and generate final report"""
        
        if not self.analyzer:
            print("❌ No analysis to stop")
            return None
        
        self.running = False
        
        print("\n🏁 Stopping analysis and generating comprehensive report...")
        print("⏳ This may take a moment...")
        
        try:
            # Generate final report
            final_results = self.analyzer.stop_unified_analysis()
            
            print("\n✅ ANALYSIS COMPLETE!")
            print("=" * 40)
            
            # Show summary
            if final_results.key_insights:
                print("\n🎯 KEY INSIGHTS:")
                for i, insight in enumerate(final_results.key_insights[:3], 1):
                    print(f"{i}. {insight}")
            
            if final_results.recommendations:
                print("\n💡 TOP RECOMMENDATIONS:")
                for i, rec in enumerate(final_results.recommendations[:3], 1):
                    print(f"{i}. {rec}")
            
            print(f"\n📁 Complete reports saved with timestamp {final_results.timestamp}")
            print("📋 Check unified_report_*.txt for full analysis")
            
            return final_results
            
        except Exception as e:
            print(f"❌ Error generating final report: {e}")
            return None
    
    def quick_analysis(self):
        """Run a quick one-time analysis without monitoring"""
        
        print("⚡ RUNNING QUICK COMPARATIVE ANALYSIS")
        print("=" * 50)
        
        # Find models
        ecological_model, abstract_model = self.find_models()
        
        if not ecological_model and not abstract_model:
            print("❌ Cannot run analysis without models")
            return
        
        # Create separate analyzers for quick run
        comparative = SpiramycelComparativeAnalyzer()
        philosophical = SpiramycelPhilosophicalFramework()
        
        try:
            # Generate quick comparative report
            print("📊 Generating comparative analysis...")
            report = comparative.generate_full_report(
                ecological_model=ecological_model or "dummy_eco.pt",
                abstract_model=abstract_model or "dummy_abs.pt"
            )
            
            print("\n" + "="*60)
            print("QUICK COMPARATIVE ANALYSIS RESULTS")
            print("="*60)
            print(report)
            
            # Save quick report
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_path = f"quick_analysis_{timestamp}.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"\n📁 Quick analysis saved to: {report_path}")
            
        except Exception as e:
            print(f"❌ Error in quick analysis: {e}")

def main():
    """Main entry point"""
    
    print("🍄 SPIRAMYCEL COMPARATIVE ANALYSIS RUNNER")
    print("=" * 60)
    print("Choose your analysis mode:")
    print("1. Full monitoring (runs continuously during training)")
    print("2. Quick analysis (one-time snapshot)")
    print("3. Demo mode (simulation)")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return
    
    runner = ComparativeAnalysisRunner()
    
    if choice == "1":
        print("\n🔄 Starting full monitoring mode...")
        
        # Ask for monitoring interval
        try:
            interval_input = input("Monitoring interval in seconds (default 15): ").strip()
            interval = float(interval_input) if interval_input else 15.0
        except ValueError:
            interval = 15.0
        
        if runner.start_analysis(monitoring_interval=interval):
            runner.run_monitoring_loop()
        
    elif choice == "2":
        print("\n⚡ Running quick analysis...")
        runner.quick_analysis()
        
    elif choice == "3":
        print("\n🎭 Running demo mode...")
        from unified_analysis import unified_analysis_demo
        unified_analysis_demo()
        
    else:
        print("❌ Invalid choice")
    
    print("\n🙏 Analysis complete. Thank you for using Spiramycel Comparative Analysis!")

if __name__ == "__main__":
    main() 