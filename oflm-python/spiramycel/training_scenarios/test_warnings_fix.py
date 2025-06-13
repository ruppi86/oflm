#!/usr/bin/env python3
"""
Quick test to verify that the "Unknown seasonal combo" warnings are fixed
"""

from ecological_data_generator import EcologicalDataGenerator
import sys

def test_warnings_fix():
    print('🧪 Testing ecological data generator fix...')
    
    # Initialize generator with reproducible seed
    generator = EcologicalDataGenerator(random_seed=42)
    print('✓ Generator initialized')
    
    # Generate just a few echoes to test - should have minimal warnings now
    print('📝 Generating 50 test echoes (calm mode)...')
    result = generator.generate_training_dataset(50, 'test_fix_calm.jsonl', chaos_mode=False)
    print(f'Calm mode result: {result}')
    
    print('\n📝 Generating 50 test echoes (chaos mode)...')
    result = generator.generate_training_dataset(50, 'test_fix_chaos.jsonl', chaos_mode=True)
    print(f'Chaos mode result: {result}')
    
    # Report unknown combinations found
    if generator.unknown_seasonal_combos:
        print(f'\n⚠ Still found {len(generator.unknown_seasonal_combos)} unknown combinations:')
        for combo in list(generator.unknown_seasonal_combos)[:5]:  # Show first 5
            print(f'   {combo}')
        if len(generator.unknown_seasonal_combos) > 5:
            print(f'   ... and {len(generator.unknown_seasonal_combos) - 5} more')
    else:
        print('\n🎉 No unknown seasonal combinations found!')
    
    print('\n✅ Test completed successfully!')

if __name__ == "__main__":
    test_warnings_fix() 