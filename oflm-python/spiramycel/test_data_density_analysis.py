#!/usr/bin/env python3
"""
Training Data Density Analysis

Investigates the critical insight that 600K models may have insufficient 
training examples per parameter compared to 200K models, leading to 
contemplative saturation.

Key hypothesis: 
- 200K: 40K examples / ~195K params = 0.205 examples/param
- 600K: 60K examples / ~719K params = 0.083 examples/param

Author: Claude  
Date: 2025-06-29
"""

import json
import glob
from pathlib import Path
from collections import defaultdict

def analyze_data_density():
    """Analyze training data density across different scales"""
    print("📊 TRAINING DATA DENSITY ANALYSIS")
    print("=" * 40)
    
    try:
        from neural_trainer import load_spiramycel_parameters
        
        # Analyze all scale configurations
        configs = [
            ('200k', 'ecological_200k', '~194,693'),
            ('200k', 'abstract_200k', '~194,693'),
            ('600k', 'ecological_600k', '~718,661'),
            ('600k', 'abstract_600k', '~718,661'),
        ]
        
        density_results = {}
        
        for scale, config_name, estimated_params in configs:
            print(f"\n🔬 {config_name.upper()}:")
            
            try:
                config = load_spiramycel_parameters(config_name)
                
                # Extract training parameters
                training_config = config.get('training', {})
                num_examples = training_config.get('num_training_examples', 'N/A')
                epochs = training_config.get('epochs', 'N/A')
                learning_rate = training_config.get('learning_rate', 'N/A')
                
                print(f"   📈 Training examples: {num_examples:,}")
                print(f"   🔄 Epochs: {epochs}")
                print(f"   📚 Learning rate: {learning_rate}")
                print(f"   🧠 Estimated parameters: {estimated_params}")
                
                # Calculate density metrics
                if isinstance(num_examples, int):
                    # Parse parameter count from string
                    param_count = int(estimated_params.replace('~', '').replace(',', ''))
                    
                    examples_per_param = num_examples / param_count
                    total_training_exposures = num_examples * epochs if isinstance(epochs, int) else 'N/A'
                    
                    print(f"   🎯 Examples per parameter: {examples_per_param:.4f}")
                    if isinstance(total_training_exposures, int):
                        exposures_per_param = total_training_exposures / param_count
                        print(f"   🔄 Total exposures per parameter: {exposures_per_param:.2f}")
                    
                    density_results[config_name] = {
                        'scale': scale,
                        'num_examples': num_examples,
                        'param_count': param_count,
                        'examples_per_param': examples_per_param,
                        'epochs': epochs,
                        'learning_rate': learning_rate,
                        'total_exposures': total_training_exposures
                    }
                    
                    # Assess density adequacy
                    if examples_per_param < 0.1:
                        print(f"   ⚠️  VERY LOW data density - likely underfitting")
                    elif examples_per_param < 0.2:
                        print(f"   📉 LOW data density - may cause poor generalization")
                    elif examples_per_param > 0.5:
                        print(f"   ✅ GOOD data density")
                    else:
                        print(f"   📊 MODERATE data density")
                        
            except Exception as e:
                print(f"   ❌ Error loading {config_name}: {e}")
        
        return density_results
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return {}

def compare_density_across_scales(density_results):
    """Compare data density between 200K and 600K models"""
    print("\n⚖️  CROSS-SCALE DENSITY COMPARISON")
    print("=" * 35)
    
    # Group by paradigm
    paradigms = ['ecological', 'abstract']
    
    for paradigm in paradigms:
        print(f"\n🔬 {paradigm.upper()} PARADIGM:")
        
        config_200k = f"{paradigm}_200k"
        config_600k = f"{paradigm}_600k"
        
        if config_200k in density_results and config_600k in density_results:
            data_200k = density_results[config_200k]
            data_600k = density_results[config_600k]
            
            # Compare key metrics
            examples_200k = data_200k['num_examples']
            examples_600k = data_600k['num_examples']
            examples_ratio = examples_600k / examples_200k
            
            density_200k = data_200k['examples_per_param']
            density_600k = data_600k['examples_per_param']
            density_ratio = density_600k / density_200k
            
            param_200k = data_200k['param_count']
            param_600k = data_600k['param_count']
            param_ratio = param_600k / param_200k
            
            print(f"   📊 Training examples: {examples_200k:,} → {examples_600k:,} ({examples_ratio:.2f}x)")
            print(f"   🧠 Parameters: {param_200k:,} → {param_600k:,} ({param_ratio:.2f}x)")
            print(f"   🎯 Density: {density_200k:.4f} → {density_600k:.4f} ({density_ratio:.2f}x)")
            
            # Analysis
            print(f"\n   📈 DENSITY ANALYSIS:")
            if density_ratio < 0.7:
                print(f"   🚨 CRITICAL: 600K has {1/density_ratio:.1f}x LOWER density than 200K!")
                print(f"   💡 Recommendation: Increase 600K training data or reduce model size")
            elif density_ratio < 0.9:
                print(f"   ⚠️  600K has moderately lower density than 200K")
            elif density_ratio > 1.1:
                print(f"   ✅ 600K has higher density than 200K")
            else:
                print(f"   📊 Similar density between scales")
            
            # Parameter efficiency
            param_efficiency_200k = examples_200k / param_200k
            param_efficiency_600k = examples_600k / param_600k
            
            print(f"\n   🔬 PARAMETER EFFICIENCY:")
            print(f"   200K efficiency: {param_efficiency_200k:.4f} examples/param")
            print(f"   600K efficiency: {param_efficiency_600k:.4f} examples/param")
            
            if param_efficiency_600k < param_efficiency_200k * 0.5:
                print(f"   🚨 600K is severely parameter-starved!")

def analyze_training_data_files():
    """Analyze actual training data files to understand content distribution"""
    print("\n📁 TRAINING DATA FILE ANALYSIS")
    print("=" * 35)
    
    # Look for training data files
    data_patterns = [
        "*.jsonl",
        "training_scenarios/*.jsonl", 
        "ecological_*_*.jsonl",
        "abstract_*_*.jsonl"
    ]
    
    training_files = []
    for pattern in data_patterns:
        training_files.extend(glob.glob(pattern))
    
    if not training_files:
        print("❌ No training data files found")
        return
    
    print(f"📄 Found {len(training_files)} potential training files")
    
    # Analyze files by scale pattern
    scale_files = {
        '200k': [],
        '600k': [],
        'other': []
    }
    
    for file_path in training_files:
        filename = Path(file_path).name
        if "piko" in filename or "200k" in filename:
            scale_files['200k'].append(file_path)
        elif "nano" in filename or "600k" in filename:
            scale_files['600k'].append(file_path)
        else:
            scale_files['other'].append(file_path)
    
    for scale, files in scale_files.items():
        if not files:
            continue
            
        print(f"\n🔬 {scale.upper()} SCALE FILES:")
        
        for file_path in files[:5]:  # Limit to 5 files per scale
            try:
                file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
                print(f"   📁 {Path(file_path).name}: {file_size_mb:.1f}MB")
                
                # Sample file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= 5:  # Just first 5 lines
                            break
                        if line.strip():
                            lines.append(line.strip())
                
                if lines:
                    print(f"     📊 Sample entries: {len(lines)}")
                    
                    # Check for stress patterns
                    try:
                        sample_entry = json.loads(lines[0])
                        if 'stress_signature' in sample_entry:
                            stress = sample_entry['stress_signature']
                            print(f"     🎯 Stress pattern: {stress}")
                        if 'effectiveness' in sample_entry:
                            effectiveness = sample_entry['effectiveness']
                            print(f"     📈 Sample effectiveness: {effectiveness:.3f}")
                    except:
                        pass
                        
            except Exception as e:
                print(f"   ❌ Error analyzing {file_path}: {e}")

def generate_recommendations(density_results):
    """Generate specific recommendations for fixing 600K models"""
    print("\n💡 RECOMMENDATIONS FOR 600K MODEL IMPROVEMENT")
    print("=" * 50)
    
    # Calculate overall density issues
    density_issues = []
    
    for config_name, data in density_results.items():
        if '600k' in config_name:
            density = data['examples_per_param']
            if density < 0.15:
                density_issues.append((config_name, density))
    
    if density_issues:
        print("🚨 CRITICAL ISSUES IDENTIFIED:")
        for config, density in density_issues:
            print(f"   • {config}: Only {density:.4f} examples/parameter")
        
        print("\n🔧 RECOMMENDED FIXES:")
        print("1. 📈 INCREASE TRAINING DATA:")
        for config, density in density_issues:
            target_density = 0.25  # Target similar to 200K
            current_examples = density_results[config]['num_examples']
            current_params = density_results[config]['param_count']
            needed_examples = int(target_density * current_params)
            increase_factor = needed_examples / current_examples
            
            print(f"   • {config}: Increase from {current_examples:,} to {needed_examples:,} examples ({increase_factor:.1f}x)")
        
        print("\n2. 🔄 ALTERNATIVE: INCREASE EPOCHS:")
        print("   • Double epochs from 20 to 40 for 600K models")
        print("   • This effectively doubles data density through repetition")
        
        print("\n3. 📚 ALTERNATIVE: IMPROVE LEARNING RATE:")
        print("   • Current 600K LR: 0.0008 is quite low")
        print("   • Try LR: 0.0012 (50% increase) for better convergence")
        
        print("\n4. 🧠 ALTERNATIVE: REDUCE MODEL SIZE:")
        print("   • Consider 400K parameter models instead of 600K")
        print("   • This would improve data density while maintaining capacity")

def main():
    """Run the complete data density analysis"""
    print("📊 600K DATA DENSITY INVESTIGATION") 
    print("=" * 40)
    print("Hypothesis: 600K models suffer from insufficient training data density")
    print("Expected: 200K has ~2.5x higher examples/parameter ratio than 600K")
    print()
    
    # Analyze configuration density
    density_results = analyze_data_density()
    
    # Compare across scales
    compare_density_across_scales(density_results)
    
    # Analyze training files
    analyze_training_data_files()
    
    # Generate recommendations
    generate_recommendations(density_results)
    
    print("\n🎯 DATA DENSITY ANALYSIS COMPLETE")
    print("=" * 35)
    print("This analysis should reveal if insufficient training data")
    print("density is causing the 600K contemplative saturation!")

if __name__ == "__main__":
    main() 