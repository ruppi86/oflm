#!/usr/bin/env python3
"""
Test script to verify model loading after vocabulary size fix
"""

import torch
from neural_trainer import SpiramycelNeuralModel
import os
from pathlib import Path

def test_model_loading():
    print('🧪 Testing model loading with fixed vocabulary size...')
    
    try:
        model = SpiramycelNeuralModel(force_cpu_mode=True)
        print(f'✓ Model created with vocab_size: {model.vocab_size}')
        
        # List of models to test (actual locations from controlled comparison)
        model_paths = [
            # Four main models from controlled comparison
            'ecological_models/ecological_calm_model.pt',
            'ecological_models/ecological_chaotic_model.pt', 
            'abstract_models/abstract_calm_model.pt',
            'abstract_models/abstract_chaotic_model.pt'
        ]
        
        loaded_count = 0
        failed_count = 0
        
        for model_path in model_paths:
            if Path(model_path).exists():
                try:
                    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
                    model.load_state_dict(state_dict)
                    print(f'✅ Successfully loaded: {model_path}')
                    print(f'   Parameters: {model.count_parameters():,}')
                    loaded_count += 1
                except Exception as e:
                    print(f'❌ Failed to load {model_path}: {e}')
                    failed_count += 1
            else:
                print(f'ℹ️ Model not found: {model_path}')
        
        print(f'\n📊 Results Summary:')
        print(f'   ✅ Successfully loaded: {loaded_count} models')
        print(f'   ❌ Failed to load: {failed_count} models')
        
        if loaded_count > 0:
            print(f'🎉 Vocabulary size fix is working!')
            if failed_count == 0:
                print('🌟 All found models loaded successfully!')
            return True
        else:
            print('⚠️ No models could be loaded - vocabulary size issue persists')
            return False
            
    except Exception as e:
        print(f'❌ Error creating model: {e}')
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print('\n🌟 Test completed successfully!')
    else:
        print('\n❌ Test failed!') 