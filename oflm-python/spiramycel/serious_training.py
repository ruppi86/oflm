#!/usr/bin/env python3
"""
Spiramycel Serious Training

Enhanced version for proper training with larger datasets and longer epochs.
Designed for systems that can handle 90+ minute training sessions like HaikuMeadowLib.
"""

import time
from pathlib import Path

try:
    from neural_trainer import SpiramycelTrainer, TORCH_AVAILABLE, DEVICE
    from spore_map import SporeMapLedger
except ImportError:
    print("❌ Import error - make sure you're in the spiramycel directory")
    exit(1)

def serious_spiramycel_training():
    """
    Serious training session - comparable to HaikuMeadowLib's 90-minute sessions.
    """
    print("🍄 Spiramycel Serious Training Session")
    print("=" * 70)
    print("🎯 Goal: Train a proper Spiramycel model with realistic performance")
    print("⏰ Expected duration: 15-60 minutes (depending on system)")
    print("🧠 Strategy: More data + Piko model + Longer training")
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available - cannot run serious training")
        return
    
    print(f"💻 Device: {DEVICE}")
    
    # Enhanced trainer with larger output directory
    trainer = SpiramycelTrainer(output_dir=Path("serious_models"))
    
    # PHASE 1: Generate substantial training data
    print(f"\n📊 PHASE 1: Generating Training Data")
    print("-" * 50)
    
    data_sizes = [
        ("Quick test", 500),
        ("Medium training", 2000), 
        ("Serious training", 5000),
        ("HaikuMeadowLib comparable", 10000)
    ]
    
    print("Choose training data size:")
    for i, (name, size) in enumerate(data_sizes):
        print(f"  {i+1}. {name}: {size} spore echoes")
    
    choice = input("\nEnter choice (1-4) or press Enter for serious training (3): ").strip()
    
    if choice == "1":
        num_examples = 500
    elif choice == "2":
        num_examples = 2000
    elif choice == "4":
        num_examples = 10000
    else:
        num_examples = 5000  # Default serious training
    
    print(f"\n🧪 Generating {num_examples} spore echoes...")
    start_time = time.time()
    
    # Create enhanced training data with more diversity
    spore_ledger = trainer.create_enhanced_training_data(num_examples)
    
    data_time = time.time() - start_time
    print(f"✅ Training data created in {data_time:.1f} seconds")
    
    # Show enhanced statistics
    stats = spore_ledger.get_statistics()
    print(f"\n📈 Enhanced Training Dataset:")
    print(f"   Total spores: {stats['total_spores']}")
    print(f"   Average effectiveness: {stats['avg_effectiveness']:.3f}")
    print(f"   Survival rate: {stats['survival_rate']:.1%}")
    print(f"   Bioregional diversity: {len(stats['bioregional_distribution'])} regions")
    print(f"   Seasonal distribution: {stats['seasonal_distribution']}")
    
    # PHASE 2: Model configuration
    print(f"\n🧠 PHASE 2: Model Configuration")
    print("-" * 50)
    
    model_configs = [
        ("Femto (fast)", {"force_cpu_mode": True, "epochs": 10, "batch_size": 4}),
        ("Piko (better)", {"force_cpu_mode": False, "epochs": 15, "batch_size": 8}), 
        ("Extended (best)", {"force_cpu_mode": False, "epochs": 25, "batch_size": 8})
    ]
    
    print("Choose model configuration:")
    for i, (name, config) in enumerate(model_configs):
        print(f"  {i+1}. {name}: {config['epochs']} epochs, batch_size {config['batch_size']}")
    
    model_choice = input("\nEnter choice (1-3) or press Enter for Piko (2): ").strip()
    
    if model_choice == "1":
        config = model_configs[0][1]
        config_name = "Femto"
    elif model_choice == "3":
        config = model_configs[2][1] 
        config_name = "Extended"
    else:
        config = model_configs[1][1]  # Default Piko
        config_name = "Piko"
    
    print(f"\n🚀 Selected: {config_name} configuration")
    print(f"   Force CPU mode: {config['force_cpu_mode']}")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Batch size: {config['batch_size']}")
    
    # PHASE 3: Serious training
    print(f"\n🏋️ PHASE 3: Neural Training")
    print("-" * 50)
    
    print("⚠️  Training will now begin. This may take 15-60 minutes.")
    input("Press Enter to start training (Ctrl+C to abort)...")
    
    training_start = time.time()
    
    try:
        model_path = trainer.train_on_spore_echoes(
            spore_ledger,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            learning_rate=0.0005,  # Slightly lower for stability
            save_checkpoints=True  # Save progress
        )
        
        training_time = time.time() - training_start
        
        print(f"\n🎉 TRAINING COMPLETE!")
        print(f"⏰ Total training time: {training_time/60:.1f} minutes")
        print(f"📁 Model saved to: {model_path}")
        
        # PHASE 4: Model evaluation  
        print(f"\n📊 PHASE 4: Model Evaluation")
        print("-" * 50)
        
        # Test the trained model
        evaluation_results = trainer.evaluate_model(model_path, spore_ledger)
        
        print(f"📈 Final Model Performance:")
        print(f"   Glyph loss: {evaluation_results['glyph_loss']:.3f}")
        print(f"   Effectiveness loss: {evaluation_results['effectiveness_loss']:.3f}")
        print(f"   Silence loss: {evaluation_results['silence_loss']:.3f}")
        print(f"   Overall improvement: {evaluation_results['improvement']:.1%}")
        
        # Compare with initial demo
        print(f"\n🔄 Comparison with Initial Demo:")
        print(f"   Dataset size: 100 → {num_examples} ({num_examples/100:.0f}x larger)")
        print(f"   Training time: 12 seconds → {training_time/60:.1f} minutes") 
        print(f"   Expected glyph loss improvement: Significant!")
        
        print(f"\n🌟 Success! Spiramycel now has serious neural capabilities.")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
        print(f"   Partial training time: {(time.time() - training_start)/60:.1f} minutes")
        print(f"   Check serious_models/ for any saved checkpoints")
    
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print(f"   This might be due to memory constraints or other system limits")
        print(f"   Try the Femto configuration for more conservative training")

if __name__ == "__main__":
    serious_spiramycel_training() 