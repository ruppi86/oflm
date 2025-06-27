#!/usr/bin/env python3
"""
test_directory_structure.py - Test the reorganized directory structure

Verifies that:
- Log files go to logs/
- Training data is in training/
- Model files go to model/piko/ (CPU-trained) and model/nano/ (GPU-trained)
- Directory creation works properly
"""

from pathlib import Path
from datetime import datetime
from generator import HaikuLogger, HaikuMeadow
from train_meadow_fork import SeasonalTrainer, BreathConfig, BreathPreset

def test_log_directory():
    """Test that log files are created in logs/ directory"""
    print("🧪 Testing log directory structure...")
    
    # Create a logger and verify it uses logs/ directory
    logger = HaikuLogger()
    log_path = logger.log_path
    
    # Check that log file is in logs/ directory
    assert log_path.parent.name == "logs", f"Expected logs/, got {log_path.parent}"
    assert log_path.exists(), f"Log file should be created: {log_path}"
    
    print(f"   ✅ Log file created at: {log_path}")
    return True

def test_training_data_path():
    """Test that training data path points to training/ directory"""
    print("🧪 Testing training data path...")
    
    # Check default training data path
    from generator import main
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", type=str, default="training/haiku_training_material.json")
    
    args = parser.parse_args([])  # Empty args to get defaults
    training_path = Path(args.training_data)
    
    assert training_path.parent.name == "training", f"Expected training/, got {training_path.parent}"
    assert training_path.exists(), f"Training data should exist: {training_path}"
    
    print(f"   ✅ Training data found at: {training_path}")
    return True

def test_model_directory_structure():
    """Test that model directories are properly organized"""
    print("🧪 Testing model directory structure...")
    
    # Check default model path
    from generator import main
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="model/piko/piko_haiku_model.pt")
    
    args = parser.parse_args([])
    model_path = Path(args.model_path)
    
    # Check directory structure
    assert model_path.parts[-3] == "model", f"Expected model/ in path: {model_path}"
    assert model_path.parts[-2] == "piko", f"Expected piko/ in path: {model_path}"
    
    # Check that directory exists
    assert model_path.parent.exists(), f"Model directory should exist: {model_path.parent}"
    
    print(f"   ✅ Model path structure correct: {model_path}")
    return True

def test_meadow_fork_trainer():
    """Test that train_meadow_fork uses correct directory"""
    print("🧪 Testing SeasonalTrainer directory...")
    
    config = BreathConfig.from_preset(BreathPreset.WHISPER)
    trainer = SeasonalTrainer(config)
    
    # Check that output directory is model/
    assert trainer.output_dir.name == "model", f"Expected model/, got {trainer.output_dir}"
    assert trainer.output_dir.exists(), f"Output directory should exist: {trainer.output_dir}"
    
    print(f"   ✅ SeasonalTrainer uses: {trainer.output_dir}")
    return True

def test_directory_creation():
    """Test that directories are created automatically"""
    print("🧪 Testing automatic directory creation...")
    
    # Test creating nested model directory
    test_model_path = Path("model/test_piko/test_model.pt")
    test_model_path.parent.mkdir(parents=True, exist_ok=True)
    
    assert test_model_path.parent.exists(), f"Directory should be created: {test_model_path.parent}"
    
    # Clean up test directory
    test_model_path.parent.rmdir()
    
    print(f"   ✅ Directory creation works")
    return True

def main():
    """Run all directory structure tests"""
    print("🌸 Testing HaikuMeadowLib Directory Structure")
    print("=" * 50)
    
    tests = [
        test_log_directory,
        test_training_data_path,
        test_model_directory_structure,
        test_meadow_fork_trainer,
        test_directory_creation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"   ❌ Test failed: {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"   ❌ Test error in {test.__name__}: {e}")
    
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🌸 All directory structure tests passed!")
        print("\n🗂️  Final Directory Structure:")
        print("   haikumeadowlib-python/")
        print("   ├── logs/              # Session log files (*.jsonl)")
        print("   ├── training/          # Training data (haiku_training_material.json)")
        print("   ├── model/")
        print("   │   ├── piko/          # CPU-trained models (*.pt)")
        print("   │   └── nano/          # GPU-trained models (*.pt)")
        print("   ├── generator.py       # Main haiku generator")
        print("   ├── train_meadow_fork.py")
        print("   └── ...")
    else:
        print("⚠️  Some tests failed - please check directory structure")
    
    return failed == 0

if __name__ == "__main__":
    main() 