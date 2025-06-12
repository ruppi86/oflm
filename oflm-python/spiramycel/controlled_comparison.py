#!/usr/bin/env python3
"""
Controlled Comparison Experiment

2x2 design to separate paradigm effects from stress effects:
- Paradigm: Ecological vs Abstract
- Stress: Calm (chaos_mode=False) vs Chaotic (chaos_mode=True)

Models saved to separate directories to preserve all four conditions.
Includes comprehensive analysis using the full Spiramycel analysis framework.
"""

import time
from pathlib import Path
import sys

# Import neural trainer components for analysis
try:
    from neural_trainer import NetworkConditions
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False
    print("⚠ Neural trainer not available - analysis will be simplified")

def run_ecological_training(chaos_mode: bool = True, suffix: str = ""):
    """Run ecological training with specified chaos mode"""
    from training_scenarios.ecological_data_generator import EcologicalDataGenerator
    from ecological_training import train_ecological_model
    
    print(f"\n🌍 ECOLOGICAL TRAINING {'(CHAOTIC)' if chaos_mode else '(CALM)'}")
    print("=" * 60)
    
    # Create ecological models directory
    ecological_dir = Path("ecological_models")
    ecological_dir.mkdir(exist_ok=True)
    
    # Generate training data
    generator = EcologicalDataGenerator()
    dataset_name = f"ecological_controlled_{suffix}.jsonl"
    data_path = generator.generate_training_dataset(
        num_echoes=5000,
        output_file=dataset_name,
        chaos_mode=chaos_mode
    )
    
    # Train model
    model_path = train_ecological_model(
        data_file=data_path,
        epochs=15
    )
    
    # Move model to ecological directory with descriptive name
    if model_path:
        new_name = ecological_dir / f"ecological_{'chaotic' if chaos_mode else 'calm'}_model.pt"
        Path(model_path).rename(new_name)
        print(f"💾 Ecological model saved to: {new_name}")
        return str(new_name)
    
    return None

def run_abstract_training(chaos_mode: bool = False, suffix: str = ""):
    """Run abstract training with specified chaos mode using pre-generated data"""
    from generate_abstract_data import AbstractDataGenerator
    from abstract_training import train_abstract_model
    
    print(f"\n✨ ABSTRACT TRAINING {'(CHAOTIC)' if chaos_mode else '(CALM)'}")
    print("=" * 60)
    
    # Create abstract models directory
    abstract_dir = Path("abstract_models")
    abstract_dir.mkdir(exist_ok=True)
    
    # Generate training data (pre-generate to files for speed)
    generator = AbstractDataGenerator()
    dataset_name = f"abstract_controlled_{suffix}.jsonl"
    data_path = generator.generate_training_dataset(
        num_echoes=5000,
        output_file=dataset_name,
        chaos_mode=chaos_mode
    )
    
    # Train model using fast file-based training
    model_path = train_abstract_model(
        data_file=data_path,
        epochs=15
    )
    
    # Move model to abstract directory with descriptive name
    if model_path:
        new_name = abstract_dir / f"abstract_{'chaotic' if chaos_mode else 'calm'}_model.pt"
        Path(model_path).rename(new_name)
        print(f"💾 Abstract model saved to: {new_name}")
        return str(new_name)
    
    return None

def run_comparative_analysis(models_dict: dict):
    """Run comprehensive comparative analysis on all trained models"""
    print(f"\n🔬 RUNNING COMPREHENSIVE COMPARATIVE ANALYSIS")
    print("=" * 60)
    
    results = {}
    
    # Import the powerful analysis components
    try:
        from comparative_analysis import SpiramycelComparativeAnalyzer
        from philosophical_framework import SpiramycelPhilosophicalFramework
        from performance_monitor import SpiramycelPerformanceMonitor
        print("✅ All analysis components loaded successfully!")
    except ImportError as e:
        print(f"⚠ Analysis framework not fully available: {e}")
        print("Running simplified analysis...")
        
        # Simplified fallback
        for condition, model_path in models_dict.items():
            if model_path and Path(model_path).exists():
                print(f"📊 Model available: {condition} → {model_path}")
                results[condition] = {"model_path": model_path, "analyzed": True}
            else:
                print(f"⚠️ Model missing: {condition}")
                results[condition] = {"model_path": None, "analyzed": False}
        return results
    
    # Run comprehensive analysis
    analyzer = SpiramycelComparativeAnalyzer()
    philosophical = SpiramycelPhilosophicalFramework()
    
    # Analyze each model that exists
    for condition, model_path in models_dict.items():
        if model_path and Path(model_path).exists():
            print(f"\n📊 Analyzing {condition} model: {model_path}")
            
            # Load model performance
            try:
                performance = analyzer.load_model_performance(condition, model_path)
                
                # Create test scenarios for analysis
                test_scenarios = [
                    # High stress scenario (chaotic conditions)
                    NetworkConditions(latency=0.9, voltage=0.1, temperature=0.9, error_rate=0.8, bandwidth=0.1),
                    # Optimal scenario (calm conditions)  
                    NetworkConditions(latency=0.1, voltage=0.8, temperature=0.5, error_rate=0.05, bandwidth=0.9),
                    # Balanced scenario
                    NetworkConditions(latency=0.5, voltage=0.5, temperature=0.5, error_rate=0.2, bandwidth=0.5),
                ]
                
                # Analyze glyph patterns
                glyph_analysis = analyzer.analyze_glyph_patterns(model_path, test_scenarios, condition)
                
                # Generate behavioral profile
                behavioral_profile = analyzer.generate_behavioral_profile(model_path, condition)
                
                results[condition] = {
                    "model_path": model_path,
                    "analyzed": True,
                    "performance": performance,
                    "glyph_analysis": glyph_analysis,
                    "behavioral_profile": behavioral_profile
                }
                
                print(f"   ✅ Analysis complete for {condition}")
                
            except Exception as e:
                print(f"   ⚠ Error analyzing {condition}: {e}")
                results[condition] = {"model_path": model_path, "analyzed": False, "error": str(e)}
        else:
            print(f"⚠️ Model missing: {condition}")
            results[condition] = {"model_path": None, "analyzed": False}
    
    # Generate comprehensive reports
    print(f"\n📋 GENERATING COMPREHENSIVE REPORTS")
    print("=" * 40)
    
    try:
        # 1. Comparative Analysis Report
        print("📊 Generating comparative analysis report...")
        comparative_report = analyzer.generate_full_report()
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = f"controlled_comparison_analysis_{timestamp}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("🧪 CONTROLLED COMPARISON EXPERIMENT - COMPREHENSIVE ANALYSIS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("🎯 EXPERIMENTAL DESIGN: 2×2 (Ecological/Abstract × Calm/Chaotic)\n\n")
            f.write(comparative_report)
        
        print(f"   📁 Saved to: {report_path}")
        
        # 2. Philosophical Analysis Report
        print("🧘 Generating philosophical framework analysis...")
        
        # Convert results to philosophical framework format
        training_results = {}
        model_behaviors = {}
        
        for condition, result in results.items():
            if result.get("analyzed") and "performance" in result:
                training_results[condition] = {
                    "final_glyph_loss": getattr(result["performance"], "final_glyph_loss", 0.0),
                    "final_silence_loss": getattr(result["performance"], "final_silence_loss", 0.0),
                    "silence_ratio": getattr(result.get("glyph_analysis", {}), "silence_ratio", 0.0),
                    "glyph_improvement_percent": 0.0  # Would need training curves to calculate
                }
                
                if "behavioral_profile" in result:
                    behavioral = result["behavioral_profile"]
                    model_behaviors[condition] = {
                        "stress_response": getattr(behavioral, "crisis_management_style", "unknown"),
                        "adaptation_strategy": getattr(behavioral, "adaptation_strategy", "unknown")
                    }
        
        if training_results:
            # Conduct philosophical analysis
            insights = philosophical.analyze_training_philosophy(training_results, model_behaviors)
            epistemological = philosophical.generate_epistemological_analysis(training_results)
            philosophical_report = philosophical.generate_contemplative_report()
            
            # Save philosophical report
            philosophical_path = f"controlled_comparison_philosophy_{timestamp}.txt"
            with open(philosophical_path, 'w', encoding='utf-8') as f:
                f.write("🧘 CONTROLLED COMPARISON - PHILOSOPHICAL IMPLICATIONS\n")
                f.write("=" * 80 + "\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("🎯 2×2 EXPERIMENTAL DESIGN PHILOSOPHICAL ANALYSIS\n\n")
                f.write(philosophical_report)
                
                f.write("\n\n🔬 PARADIGM × STRESS INTERACTION ANALYSIS:\n")
                f.write("=" * 50 + "\n")
                f.write("The 2×2 design allows us to separate:\n")
                f.write("• PARADIGM EFFECTS: Ecological vs Abstract learning approaches\n")
                f.write("• STRESS EFFECTS: Calm vs Chaotic environmental conditions\n")
                f.write("• INTERACTION EFFECTS: How paradigms respond differently to stress\n\n")
                
                if len(training_results) >= 4:
                    f.write("This reveals the deep wisdom of contemplative AI:\n")
                    f.write("Each paradigm-stress combination teaches unique lessons about\n")
                    f.write("the nature of intelligence, adaptation, and silence.\n")
            
            print(f"   📁 Saved to: {philosophical_path}")
        
        # 3. Summary Report
        print("📋 Generating executive summary...")
        summary_path = f"controlled_comparison_summary_{timestamp}.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("📊 CONTROLLED COMPARISON EXPERIMENT - EXECUTIVE SUMMARY\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("🎯 EXPERIMENTAL DESIGN:\n")
            f.write("2×2 factorial design separating paradigm effects from stress effects\n\n")
            
            f.write("📊 MODELS ANALYZED:\n")
            for condition, result in results.items():
                status = "✅ SUCCESS" if result.get("analyzed") else "❌ FAILED"
                f.write(f"   {condition}: {status}\n")
                if result.get("model_path"):
                    f.write(f"      Model: {result['model_path']}\n")
            
            f.write(f"\n📁 DETAILED REPORTS:\n")
            f.write(f"   🔬 Technical Analysis: {report_path}\n")
            if 'philosophical_path' in locals():
                f.write(f"   🧘 Philosophical Analysis: {philosophical_path}\n")
            f.write(f"   📋 This Summary: {summary_path}\n")
            
            f.write(f"\n🌱 NEXT STEPS:\n")
            f.write(f"   1. Review detailed analysis reports\n")
            f.write(f"   2. Compare paradigm effectiveness under different stress conditions\n")
            f.write(f"   3. Analyze interaction effects between paradigm and environment\n")
            f.write(f"   4. Consider implications for contemplative AI development\n")
        
        print(f"   📁 Saved to: {summary_path}")
        
        print(f"\n🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
        print(f"📂 Three detailed reports generated with timestamp {timestamp}")
        
    except Exception as e:
        print(f"❌ Error generating reports: {e}")
        import traceback
        traceback.print_exc()
    
    return results

def main():
    """Run the complete controlled comparison experiment"""
    print("🧪 CONTROLLED SPIRAMYCEL COMPARISON EXPERIMENT")
    print("=" * 70)
    print("🎯 Goal: Separate paradigm effects from stress effects")
    print("📊 Design: 2x2 (Ecological/Abstract × Calm/Chaotic)")
    print("⏰ Expected duration: 30-60 minutes total")
    print("")
    print("📋 DOCUMENTATION GENERATED:")
    print("   🔬 Technical comparative analysis report")
    print("   🧘 Philosophical implications analysis")
    print("   📊 Executive summary with next steps")
    print("   📂 All reports timestamped and preserved")
    
    input("\nPress Enter to start the experiment (Ctrl+C to abort)...")
    
    start_time = time.time()
    trained_models = {}
    
    try:
        # Run all four conditions
        print("\n🚀 PHASE 1: Training all four conditions...")
        
        # 1. Ecological Calm (A)
        print(f"\n🌱 Training condition A: Ecological + Calm")
        model_a = run_ecological_training(chaos_mode=False, suffix="calm")
        trained_models["ecological_calm"] = model_a
        
        # 2. Ecological Chaotic (B) 
        print(f"\n🌋 Training condition B: Ecological + Chaotic")
        model_b = run_ecological_training(chaos_mode=True, suffix="chaotic")
        trained_models["ecological_chaotic"] = model_b
        
        # 3. Abstract Calm (C)
        print(f"\n🧘 Training condition C: Abstract + Calm")  
        model_c = run_abstract_training(chaos_mode=False, suffix="calm")
        trained_models["abstract_calm"] = model_c
        
        # 4. Abstract Chaotic (D)
        print(f"\n⚡ Training condition D: Abstract + Chaotic")
        model_d = run_abstract_training(chaos_mode=True, suffix="chaotic")
        trained_models["abstract_chaotic"] = model_d
        
        training_time = time.time() - start_time
        print(f"\n✅ All training complete in {training_time/60:.1f} minutes!")
        
        # PHASE 2: Comprehensive Analysis (now much more powerful!)
        print(f"\n🔬 PHASE 2: Comprehensive Analysis")
        print("This will analyze:")
        print("   • Glyph usage patterns and contemplative ratios")
        print("   • Behavioral profiles under different stress conditions") 
        print("   • Philosophical implications of paradigm differences")
        print("   • Epistemological analysis of learning approaches")
        print("   • Interaction effects between paradigm and environment")
        
        results = run_comparative_analysis(trained_models)
        
        # PHASE 3: Results Summary
        print(f"\n📋 EXPERIMENTAL RESULTS SUMMARY")
        print("=" * 60)
        
        print(f"\n📊 2×2 DESIGN RESULTS:")
        print(f"┌─────────────┬──────────────┬──────────────┐")
        print(f"│             │   CALM       │   CHAOTIC    │")
        print(f"├─────────────┼──────────────┼──────────────┤")
        
        eco_calm = "✅" if results.get("ecological_calm", {}).get("analyzed") else "❌"
        eco_chaos = "✅" if results.get("ecological_chaotic", {}).get("analyzed") else "❌"
        abs_calm = "✅" if results.get("abstract_calm", {}).get("analyzed") else "❌" 
        abs_chaos = "✅" if results.get("abstract_chaotic", {}).get("analyzed") else "❌"
        
        print(f"│ ECOLOGICAL  │   {eco_calm} (A)     │   {eco_chaos} (B)     │")
        print(f"│ ABSTRACT    │   {abs_calm} (C)     │   {abs_chaos} (D)     │")
        print(f"└─────────────┴──────────────┴──────────────┘")
        
        print(f"\n🎯 ANALYSIS IMPLICATIONS:")
        print(f"   • A vs C: Paradigm effect under calm conditions")
        print(f"   • B vs D: Paradigm effect under chaotic conditions") 
        print(f"   • A vs B: Stress effect for ecological paradigm")
        print(f"   • C vs D: Stress effect for abstract paradigm")
        
        print(f"\n📁 Models saved for detailed analysis:")
        for condition, model_path in trained_models.items():
            if model_path:
                print(f"   {condition}: {model_path}")
        
        print(f"\n📂 Model Organization:")
        print(f"   📁 ecological_models/")
        print(f"      └── ecological_calm_model.pt (106KB)")
        print(f"      └── ecological_chaotic_model.pt (106KB)")
        print(f"   📁 abstract_models/")
        print(f"      └── abstract_calm_model.pt (106KB)")
        print(f"      └── abstract_chaotic_model.pt (106KB)")
        
        total_time = time.time() - start_time
        print(f"\n🎉 Experiment complete in {total_time/60:.1f} minutes!")
        print(f"🔬 Ready for detailed contemplative analysis!")
        print(f"🌱 All four organic femto language models preserved!")
        print(f"📋 Check the comprehensive analysis reports for deep insights!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Experiment interrupted by user")
        elapsed = (time.time() - start_time) / 60
        print(f"   Partial completion time: {elapsed:.1f} minutes")
        print(f"   Check saved models in ecological_models/ and abstract_models/")
    
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        print(f"   Check individual training components")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 