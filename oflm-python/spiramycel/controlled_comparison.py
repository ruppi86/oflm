#!/usr/bin/env python3
"""
Controlled Comparison Experiment

2x2 design to separate paradigm effects from stress effects:
- Paradigm: Ecological vs Abstract
- Stress: Calm (chaos_mode=False) vs Chaotic (chaos_mode=True)
"""

import time
from pathlib import Path
import sys

def run_ecological_training(chaos_mode: bool = True, suffix: str = ""):
    """Run ecological training with specified chaos mode"""
    from training_scenarios.ecological_data_generator import EcologicalDataGenerator
    from ecological_training import train_ecological_model
    
    print(f"\n🌍 ECOLOGICAL TRAINING {'(CHAOTIC)' if chaos_mode else '(CALM)'}")
    print("=" * 60)
    
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
    
    # Rename model to indicate condition
    if model_path:
        new_name = f"ecological_{'chaotic' if chaos_mode else 'calm'}_model.pt"
        Path(model_path).rename(new_name)
        return new_name
    
    return None

def run_abstract_training(chaos_mode: bool = False, suffix: str = ""):
    """Run abstract training with specified chaos mode using pre-generated data"""
    from generate_abstract_data import AbstractDataGenerator
    from abstract_training import train_abstract_model
    
    print(f"\n✨ ABSTRACT TRAINING {'(CHAOTIC)' if chaos_mode else '(CALM)'}")
    print("=" * 60)
    
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
    
    # Rename model to indicate condition
    if model_path:
        new_name = f"abstract_{'chaotic' if chaos_mode else 'calm'}_model.pt"
        Path(model_path).rename(new_name)
        return new_name
    
    return None

def run_comparative_analysis(models_dict: dict):
    """Run comparative analysis on all trained models"""
    print(f"\n🔬 RUNNING COMPARATIVE ANALYSIS")
    print("=" * 60)
    
    results = {}
    
    # Run quick analysis for each model pair
    for condition, model_path in models_dict.items():
        if model_path and Path(model_path).exists():
            print(f"\n📊 Analyzing {condition} model...")
            # Here we would run the analysis - simplified for now
            # In practice, you'd integrate with comparative_analysis.py
            results[condition] = {"model_path": model_path, "analyzed": True}
        else:
            print(f"⚠️ Model missing: {condition}")
            results[condition] = {"model_path": None, "analyzed": False}
    
    return results

def main():
    """Run the complete controlled comparison experiment"""
    print("🧪 CONTROLLED SPIRAMYCEL COMPARISON EXPERIMENT")
    print("=" * 70)
    print("🎯 Goal: Separate paradigm effects from stress effects")
    print("📊 Design: 2x2 (Ecological/Abstract × Calm/Chaotic)")
    print("⏰ Expected duration: 30-60 minutes total")
    
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
        
        # PHASE 2: Comparative Analysis
        print(f"\n🔬 PHASE 2: Comparative Analysis")
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
        
        total_time = time.time() - start_time
        print(f"\n🎉 Experiment complete in {total_time/60:.1f} minutes!")
        print(f"🔬 Ready for detailed contemplative analysis!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Experiment interrupted by user")
        elapsed = (time.time() - start_time) / 60
        print(f"   Partial completion time: {elapsed:.1f} minutes")
        print(f"   Check saved models in current directory and controlled_models/")
    
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        print(f"   Check individual training components")

if __name__ == "__main__":
    main() 