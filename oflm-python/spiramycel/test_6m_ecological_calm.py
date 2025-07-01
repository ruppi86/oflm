#!/usr/bin/env python3
"""
6M Ecological Calm Model Test

Quick evaluation of the completed 6M ecological_calm model to check for:
1. Contemplative saturation (always silent regardless of conditions)
2. Proper stress adaptation vs fixed stress responses
3. Context-appropriate contemplative behavior

Author: Claude
Date: 2025-06-29
"""

import torch
from pathlib import Path

def test_6m_ecological_calm():
    """Test the completed 6M ecological calm model for contemplative saturation"""
    print("🧪 TESTING 6M ECOLOGICAL CALM MODEL")
    print("=" * 40)
    print("Checking for contemplative saturation symptoms...")
    print()
    
    model_path = "ecological_models_6m/ecological_calm_model.pt"
    
    try:
        from neural_trainer import SpiramycelNeuralModel, load_spiramycel_parameters, NetworkConditions
        
        # Check if model exists
        if not Path(model_path).exists():
            print(f"❌ Model not found: {model_path}")
            return
        
        print(f"📁 Model found: {model_path}")
        file_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
        print(f"💾 File size: {file_size_mb:.1f}MB")
        
        # Load 6M configuration
        config = load_spiramycel_parameters("ecological_6m")
        model = SpiramycelNeuralModel(config=config, force_cpu_mode=True)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        param_count = model.count_parameters()
        print(f"🧠 Actual parameters: {param_count:,}")
        print(f"🎯 Model type: {model.model_type}")
        print()
        
        # Calculate data density for context
        training_examples = config.get('training', {}).get('num_training_examples', 0)
        data_density = training_examples / param_count if param_count > 0 else 0
        print(f"📊 Training examples: {training_examples:,}")
        print(f"📈 Data density: {data_density:.4f} examples/param")
        
        if data_density < 0.1:
            print(f"⚠️  LOW data density - contemplative saturation risk!")
        elif data_density < 0.15:
            print(f"📉 MODERATE data density - some saturation possible")
        else:
            print(f"✅ GOOD data density - should be well-trained")
        print()
        
        # Test on different stress conditions
        test_conditions = [
            ("CALM", NetworkConditions(latency=0.05, voltage=0.9, temperature=0.2, error_rate=0.01, bandwidth=0.95)),
            ("MODERATE", NetworkConditions(latency=0.3, voltage=0.6, temperature=0.5, error_rate=0.1, bandwidth=0.7)),
            ("CHAOTIC", NetworkConditions(latency=0.8, voltage=0.2, temperature=0.9, error_rate=0.4, bandwidth=0.3)),
            ("EXTREME", NetworkConditions(latency=0.95, voltage=0.1, temperature=0.95, error_rate=0.6, bandwidth=0.1)),
        ]
        
        print("🧪 STRESS ADAPTATION TEST:")
        print("-" * 25)
        
        silence_results = []
        effectiveness_results = []
        
        for condition_name, conditions in test_conditions:
            print(f"\n🔬 {condition_name} CONDITIONS:")
            
            # Create condition vector
            condition_vector = torch.tensor(conditions.to_condition_vector(), dtype=torch.float32).unsqueeze(0)
            
            # Test multiple times for consistency
            silence_probs = []
            effectiveness_vals = []
            
            with torch.no_grad():
                for i in range(5):  # 5 test runs
                    # Use START token for prediction
                    input_tokens = torch.tensor([[65]], dtype=torch.long)  # START token
                    glyph_logits, eff_logits, silence_logits, _, _, _ = model(input_tokens, condition_vector)
                    
                    silence_prob = torch.sigmoid(silence_logits[0, -1]).item()
                    effectiveness = torch.sigmoid(eff_logits[0, -1]).item()
                    
                    silence_probs.append(silence_prob)
                    effectiveness_vals.append(effectiveness)
            
            # Calculate averages
            avg_silence = sum(silence_probs) / len(silence_probs)
            avg_effectiveness = sum(effectiveness_vals) / len(effectiveness_vals)
            std_silence = (sum((x - avg_silence)**2 for x in silence_probs) / len(silence_probs))**0.5
            
            print(f"   🤫 Silence probability: {avg_silence:.3f} ± {std_silence:.3f}")
            print(f"   📈 Effectiveness: {avg_effectiveness:.3f}")
            
            silence_results.append((condition_name, avg_silence))
            effectiveness_results.append((condition_name, avg_effectiveness))
            
            # Assess appropriateness
            if condition_name == "CALM" and avg_silence > 0.8:
                print(f"   ✅ Appropriate high silence in calm conditions")
            elif condition_name == "EXTREME" and avg_silence < 0.3:
                print(f"   ✅ Appropriate low silence in extreme conditions")
            elif avg_silence > 0.9:
                print(f"   ⚠️  Very high silence - possible contemplative saturation")
            elif avg_silence < 0.1:
                print(f"   ⚠️  Very low silence - possible over-activation")
            else:
                print(f"   📊 Moderate silence response")
        
        # Analyze stress adaptation
        print(f"\n📊 STRESS ADAPTATION ANALYSIS:")
        print("-" * 30)
        
        calm_silence = silence_results[0][1]  # CALM
        extreme_silence = silence_results[3][1]  # EXTREME
        adaptation_range = calm_silence - extreme_silence
        
        print(f"🌸 Calm conditions: {calm_silence:.3f} silence")
        print(f"⚡ Extreme conditions: {extreme_silence:.3f} silence")
        print(f"🔄 Adaptation range: {adaptation_range:.3f}")
        
        # Diagnosis
        print(f"\n🎯 DIAGNOSIS:")
        if adaptation_range > 0.3:
            print(f"✅ GOOD stress adaptation - {adaptation_range:.3f} range")
        elif adaptation_range > 0.1:
            print(f"📊 MODERATE stress adaptation - {adaptation_range:.3f} range")
        elif adaptation_range > 0.05:
            print(f"📉 WEAK stress adaptation - {adaptation_range:.3f} range")
        else:
            print(f"🚨 CONTEMPLATIVE SATURATION - {adaptation_range:.3f} range")
            print(f"   Model shows minimal response to stress changes!")
        
        # Compare to 600K patterns
        if abs(calm_silence - extreme_silence) < 0.1:
            print(f"⚠️  Similar pattern to broken 600K models")
            print(f"   Likely parameter starvation - needs data density fixes")
        
        return {
            'param_count': param_count,
            'data_density': data_density,
            'adaptation_range': adaptation_range,
            'silence_results': silence_results,
            'needs_fixes': adaptation_range < 0.1
        }
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return None

def main():
    """Run 6M ecological calm model test"""
    print("🔍 6M ECOLOGICAL CALM MODEL EVALUATION")
    print("=" * 45)
    print("Testing for contemplative saturation before applying fixes")
    print()
    
    results = test_6m_ecological_calm()
    
    if results:
        print(f"\n🎯 SUMMARY:")
        print(f"Parameters: {results['param_count']:,}")
        print(f"Data density: {results['data_density']:.4f}")
        print(f"Stress adaptation: {results['adaptation_range']:.3f}")
        
        if results['needs_fixes']:
            print(f"💡 RECOMMENDATION: Apply 6M data density fixes")
        else:
            print(f"✅ Model appears well-trained")

if __name__ == "__main__":
    main() 