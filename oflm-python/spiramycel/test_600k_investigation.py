#!/usr/bin/env python3
"""
600K Model Investigation - Debugging Stress Adaptation Degradation

This script investigates why 600K models show worse stress-level adaptation 
than 200K models, which is counterintuitive.

Author: Claude
Date: 2025-06-29
"""

import torch
from pathlib import Path
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')

def test_model_architectures():
    """Test actual parameter counts and architectures of all scales"""
    print("🔍 INVESTIGATING MODEL ARCHITECTURES")
    print("=" * 50)
    
    try:
        from neural_trainer import SpiramycelNeuralModel, load_spiramycel_parameters
        
        # Test all scale configurations
        configs_to_test = [
            ('ecological_200k', 'ecological_models_200k/ecological_calm_model.pt'),
            ('ecological_600k', 'ecological_models_600k/ecological_calm_model.pt'),
            ('abstract_200k', 'abstract_models_200k/abstract_calm_model.pt'),
            ('abstract_600k', 'abstract_models_600k/abstract_calm_model.pt'),
        ]
        
        for config_name, model_path in configs_to_test:
            print(f"\n📊 Testing {config_name}:")
            
            # Test configuration loading
            try:
                config = load_spiramycel_parameters(config_name)
                print(f"   ✅ Config loaded successfully")
                print(f"   📐 embed_dim: {config.get('embed_dim')}")
                print(f"   📐 hidden_dim: {config.get('hidden_dim')}")
                print(f"   📐 num_layers: {config.get('num_layers')}")
                print(f"   📐 parameter_count: {config.get('parameter_count')}")
                
                # Create model from config
                model = SpiramycelNeuralModel(config=config, force_cpu_mode=True)
                actual_params = model.count_parameters()
                print(f"   🔢 Actual parameters: {actual_params:,}")
                print(f"   🎯 Model type: {model.model_type}")
                
                # Check if model file exists and load it
                if Path(model_path).exists():
                    file_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
                    print(f"   💾 Model file: {file_size_mb:.1f}MB")
                    
                    # Try to load the actual saved model
                    try:
                        model.load_state_dict(torch.load(model_path, map_location='cpu'))
                        print(f"   ✅ Model loaded successfully - architecture matches!")
                    except Exception as e:
                        print(f"   ❌ Model loading failed: {e}")
                        print(f"   🚨 ARCHITECTURE MISMATCH DETECTED!")
                else:
                    print(f"   ⚠️  Model file not found: {model_path}")
                    
            except Exception as e:
                print(f"   ❌ Config loading failed: {e}")
                
    except ImportError as e:
        print(f"❌ Import failed: {e}")

def test_training_convergence():
    """Check if 600K models actually trained properly"""
    print("\n🎯 INVESTIGATING TRAINING CONVERGENCE")
    print("=" * 50)
    
    # Check for training logs
    log_files = [
        "logs/controlled_comparison_*.log",
        "logs/ecological_*_*.log", 
        "logs/abstract_*_*.log"
    ]
    
    import glob
    recent_logs = []
    for pattern in log_files:
        recent_logs.extend(glob.glob(pattern))
    
    # Sort by modification time, get most recent
    if recent_logs:
        recent_logs.sort(key=lambda f: Path(f).stat().st_mtime, reverse=True)
        
        print(f"📄 Found {len(recent_logs)} log files")
        print("🔍 Checking most recent training logs for 600K...")
        
        # Look for 600K training evidence
        for log_file in recent_logs[:5]:  # Check 5 most recent
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if "600k" in content or "nano-scale" in content:
                    print(f"\n📋 {Path(log_file).name}:")
                    
                    # Look for training completion
                    if "TRAINING COMPLETED" in content:
                        print("   ✅ Training completed")
                    else:
                        print("   ⚠️  No training completion found")
                    
                    # Look for architecture errors
                    if "Could not analyze model architecture" in content:
                        print("   🚨 Architecture errors found!")
                    
                    # Look for parameter counts
                    import re
                    param_matches = re.findall(r'Parameters: ([\d,]+)', content)
                    if param_matches:
                        print(f"   📊 Parameter counts found: {param_matches}")
                        
            except Exception as e:
                print(f"   ❌ Error reading {log_file}: {e}")
    else:
        print("❌ No training logs found")

def test_600k_vs_200k_configs():
    """Compare 600K vs 200K configurations directly"""
    print("\n⚖️  COMPARING 600K vs 200K CONFIGURATIONS")
    print("=" * 50)
    
    try:
        from neural_trainer import load_spiramycel_parameters
        
        comparison_pairs = [
            ('ecological_200k', 'ecological_600k'),
            ('abstract_200k', 'abstract_600k')
        ]
        
        for small_config, large_config in comparison_pairs:
            print(f"\n🔄 {small_config} → {large_config}:")
            
            try:
                config_200k = load_spiramycel_parameters(small_config)
                config_600k = load_spiramycel_parameters(large_config)
                
                # Compare key parameters
                comparisons = [
                    'embed_dim', 'hidden_dim', 'num_layers', 
                    'parameter_count', 'paradigm'
                ]
                
                for param in comparisons:
                    val_200k = config_200k.get(param, 'N/A')
                    val_600k = config_600k.get(param, 'N/A')
                    print(f"   {param}: {val_200k} → {val_600k}")
                    
                # Check training parameters
                train_200k = config_200k.get('training', {})
                train_600k = config_600k.get('training', {})
                
                print("   Training config:")
                for param in ['epochs', 'learning_rate', 'num_training_examples']:
                    val_200k = train_200k.get(param, 'N/A')
                    val_600k = train_600k.get(param, 'N/A')
                    print(f"     {param}: {val_200k} → {val_600k}")
                    
            except Exception as e:
                print(f"   ❌ Error comparing configs: {e}")
                
    except ImportError as e:
        print(f"❌ Import failed: {e}")

def test_model_behavior_differences():
    """Test actual model outputs to see behavioral differences"""
    print("\n🧪 TESTING MODEL BEHAVIOR DIFFERENCES")
    print("=" * 50)
    
    try:
        from neural_trainer import SpiramycelNeuralModel, load_spiramycel_parameters, NetworkConditions
        from glyph_codec import SpiramycelGlyphCodec
        
        # Test scenarios
        test_conditions = [
            NetworkConditions(latency=0.1, voltage=0.8, temperature=0.3, error_rate=0.05, bandwidth=0.9),  # Calm
            NetworkConditions(latency=0.9, voltage=0.2, temperature=0.8, error_rate=0.4, bandwidth=0.2),   # Chaotic
        ]
        
        models_to_test = [
            ('ecological_200k', 'ecological_models_200k/ecological_calm_model.pt'),
            ('ecological_600k', 'ecological_models_600k/ecological_calm_model.pt'),
        ]
        
        codec = SpiramycelGlyphCodec()
        
        for config_name, model_path in models_to_test:
            if not Path(model_path).exists():
                print(f"⚠️  Model not found: {model_path}")
                continue
                
            print(f"\n🤖 Testing {config_name}:")
            
            try:
                config = load_spiramycel_parameters(config_name)
                model = SpiramycelNeuralModel(config=config, force_cpu_mode=True)
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                
                for i, conditions in enumerate(test_conditions):
                    condition_name = "Calm" if i == 0 else "Chaotic"
                    print(f"   {condition_name} conditions:")
                    
                    # Create condition vector
                    condition_vector = torch.tensor(conditions.to_condition_vector(), dtype=torch.float32).unsqueeze(0)
                    
                    # Generate predictions
                    with torch.no_grad():
                        # Test silence prediction
                        input_tokens = torch.tensor([[65]], dtype=torch.long)  # START token
                        glyph_logits, eff_logits, silence_logits, _, _, _ = model(input_tokens, condition_vector)
                        
                        silence_prob = torch.sigmoid(silence_logits[0, -1]).item()
                        effectiveness = torch.sigmoid(eff_logits[0, -1]).item()
                        
                        print(f"     Silence probability: {silence_prob:.3f}")
                        print(f"     Effectiveness: {effectiveness:.3f}")
                        
                        # Check if this matches expected behavior
                        if condition_name == "Calm" and silence_prob > 0.7:
                            print(f"     ✅ High silence in calm conditions")
                        elif condition_name == "Chaotic" and silence_prob < 0.3:
                            print(f"     ✅ Low silence in chaotic conditions")
                        else:
                            print(f"     ⚠️  Unexpected silence behavior")
                            
            except Exception as e:
                print(f"   ❌ Error testing {config_name}: {e}")
                
    except ImportError as e:
        print(f"❌ Import failed: {e}")

def main():
    """Run all investigations"""
    print("🔍 600K MODEL DEGRADATION INVESTIGATION")
    print("=" * 60)
    print("Investigating why 600K models show worse stress adaptation than 200K")
    print()
    
    # Run all tests
    test_model_architectures()
    test_training_convergence() 
    test_600k_vs_200k_configs()
    test_model_behavior_differences()
    
    print("\n🎯 INVESTIGATION COMPLETE")
    print("=" * 30)
    print("Review the output above to identify potential issues:")
    print("1. Architecture mismatches")
    print("2. Training convergence problems")
    print("3. Configuration errors")
    print("4. Behavioral anomalies")

if __name__ == "__main__":
    main() 