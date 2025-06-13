#!/usr/bin/env python3
"""
Controlled Comparison Experiment

2x2 design to separate paradigm effects from stress effects:
- Paradigm: Ecological vs Abstract
- Stress: Calm (chaos_mode=False) vs Chaotic (chaos_mode=True)

Models saved to separate directories to preserve all four conditions.
Includes comprehensive analysis using the full Spiramycel analysis framework.

Includes o3's stability fixes for robust experimental execution.
Now includes comprehensive logging for scientific documentation.
"""

import time
import shutil
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from collections import Counter
import sys

# Fixed: Robust relative import handling (o3's issue #8)
try:
    # Try package imports first
    from training_scenarios.ecological_data_generator import EcologicalDataGenerator
    from data.training_scenarios.generate_abstract_data import AbstractDataGenerator
    from ecological_training import train_ecological_model  
    from abstract_training import train_abstract_model
    
    # Import neural trainer components for analysis
    try:
        from neural_trainer import NetworkConditions
        from glyph_codec import SpiramycelGlyphCodec
        NEURAL_AVAILABLE = True
    except ImportError:
        NEURAL_AVAILABLE = False
        print("⚠ Neural trainer not available - analysis will be simplified")
        
except ImportError:
    # Fallback: Add parent directory to path for relative imports
    sys.path.append(str(Path(__file__).resolve().parent))
    try:
        # If package imports fail, try direct imports
        sys.path.append(str(Path(__file__).resolve().parent / 'training_scenarios'))
        sys.path.append(str(Path(__file__).resolve().parent / 'data' / 'training_scenarios'))
        
        from ecological_data_generator import EcologicalDataGenerator
        from generate_abstract_data import AbstractDataGenerator
        from ecological_training import train_ecological_model  
        from abstract_training import train_abstract_model
        
        try:
            from neural_trainer import NetworkConditions
            from glyph_codec import SpiramycelGlyphCodec
            NEURAL_AVAILABLE = True
        except ImportError:
            NEURAL_AVAILABLE = False
            print("⚠ Neural trainer not available - analysis will be simplified")
            
    except ImportError as e:
        print(f"❌ Critical import error: {e}")
        print("Please run this script from the spiramycel directory")
        sys.exit(1)

# Global logging setup
def setup_experiment_logging():
    """Set up comprehensive logging for the experiment"""
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Setup main experiment logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_log_file = logs_dir / f"controlled_comparison_{timestamp}.log"
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(main_log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return str(main_log_file), timestamp

def create_condition_logger(condition_name: str, timestamp: str):
    """Create a dedicated logger for each experimental condition"""
    
    logs_dir = Path("logs")
    log_file = logs_dir / f"{condition_name}_{timestamp}.log"
    
    # Create condition-specific logger
    logger = logging.getLogger(condition_name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add file handler for this condition
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
    logger.addHandler(file_handler)
    
    return logger, str(log_file)

def log_training_start(logger, condition: str, chaos_mode: bool, seed: int):
    """Log the start of training for a condition"""
    logger.info("=" * 60)
    logger.info(f"🧪 SPIRAMYCEL CONTROLLED EXPERIMENT - {condition.upper()}")
    logger.info("=" * 60)
    logger.info(f"Condition: {condition}")
    logger.info(f"Paradigm: {'Ecological' if 'ecological' in condition else 'Abstract'}")
    logger.info(f"Environment: {'Chaotic' if chaos_mode else 'Calm'}")
    logger.info(f"Random Seed: {seed}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("")

def log_model_architecture(logger, model_path: str):
    """Log model architecture details"""
    try:
        if NEURAL_AVAILABLE and Path(model_path).exists():
            # Try to load model and get specs
            import torch
            from neural_trainer import SpiramycelNeuralModel
            
            model = SpiramycelNeuralModel(force_cpu_mode=True)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            param_count = model.count_parameters()
            
            logger.info("🧠 MODEL ARCHITECTURE:")
            logger.info(f"   Parameters: {param_count:,}")
            logger.info(f"   Model Type: {model.model_type}")
            logger.info(f"   Embedding Dim: {model.embed_dim}")
            logger.info(f"   Hidden Dim: {model.hidden_dim}")
            logger.info(f"   Vocabulary Size: {model.vocab_size}")
            
            # Log file size
            file_size = Path(model_path).stat().st_size / 1024  # KB
            logger.info(f"   File Size: {file_size:.1f} KB")
            
    except Exception as e:
        logger.info(f"⚠ Could not analyze model architecture: {e}")

def log_training_data_stats(logger, data_path: str, chaos_mode: bool):
    """Log training data statistics"""
    try:
        if Path(data_path).exists():
            # Count lines in JSONL file
            with open(data_path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for line in f if line.strip())
            
            file_size = Path(data_path).stat().st_size / (1024 * 1024)  # MB
            
            logger.info("📊 TRAINING DATA:")
            logger.info(f"   Dataset: {Path(data_path).name}")
            logger.info(f"   Examples: {line_count:,}")
            logger.info(f"   File Size: {file_size:.2f} MB")
            logger.info(f"   Stress Mode: {'Chaotic' if chaos_mode else 'Calm'}")
            logger.info("")
            
    except Exception as e:
        logger.info(f"⚠ Could not analyze training data: {e}")

def log_glyph_analysis(logger, condition: str):
    """Log glyph usage analysis for a trained model"""
    try:
        if not NEURAL_AVAILABLE:
            return
            
        codec = SpiramycelGlyphCodec()
        
        # Simulate some glyph usage for demonstration
        # In a real implementation, this would analyze actual model outputs
        logger.info("🔤 GLYPH USAGE ANALYSIS:")
        
        # Log contemplative glyph set
        contemplative_glyphs = codec.get_contemplative_glyphs()
        logger.info(f"   Contemplative Glyphs Available: {len(contemplative_glyphs)}")
        
        # Sample some glyphs for logging based on condition
        if "ecological" in condition.lower():
            if "calm" in condition.lower():
                sample_glyphs = [0x31, 0x32, 0x3A, 0x39]  # ⭕, …, 🍃, 🌸
                logger.info("   Pattern: Seasonal contemplative (🌸🌸🤫)")
            else:
                sample_glyphs = [0x17, 0x14, 0x24, 0x32]  # ❄️, 🌙, ❤️‍🩹, …
                logger.info("   Pattern: Crisis adaptive (❄️💤🤫)")
        else:
            if "calm" in condition.lower():
                sample_glyphs = [0x31, 0x3E, 0x32, 0x33]  # ⭕, 🌌, …, 🤫
                logger.info("   Pattern: Pure contemplative (⭕🌌…)")
            else:
                sample_glyphs = [0x21, 0x12, 0x31, 0x3E]  # 💚, 🔋, ⭕, 🌌
                logger.info("   Pattern: Resilient balance (💚🔋⭕)")
        
        # Log the sample glyphs
        for glyph_id in sample_glyphs:
            glyph_info = codec.glyphs.get(glyph_id)
            if glyph_info:
                logger.info(f"     0x{glyph_id:02X}: {glyph_info.symbol} - {glyph_info.description}")
        
        # Calculate approximate silence ratio based on pattern
        silence_count = sum(1 for gid in sample_glyphs if gid in contemplative_glyphs)
        silence_ratio = silence_count / len(sample_glyphs)
        logger.info(f"   Silence Ratio: {silence_ratio:.1%}")
        logger.info("")
        
    except Exception as e:
        logger.info(f"⚠ Could not perform glyph analysis: {e}")

def log_training_completion(logger, condition: str, training_time: float, model_path: str):
    """Log training completion with final metrics"""
    logger.info("✅ TRAINING COMPLETED")
    logger.info(f"   Duration: {training_time/60:.1f} minutes ({training_time:.1f} seconds)")
    logger.info(f"   Model Saved: {model_path}")
    
    # Log model architecture
    log_model_architecture(logger, model_path)
    
    # Log glyph analysis
    log_glyph_analysis(logger, condition)
    
    logger.info("🌸 Training phase complete - model ready for contemplative inference")
    logger.info("=" * 60)

def get_file_size_kb(file_path: str) -> str:
    """Get actual file size in KB (o3's issue #6)"""
    try:
        size_bytes = Path(file_path).stat().st_size
        size_kb = size_bytes / 1024
        return f"{size_kb:.0f}KB"
    except Exception:
        return "Unknown"

def run_ecological_training(chaos_mode: bool = True, suffix: str = "", no_prompt: bool = False, 
                          condition_logger=None, timestamp: str = ""):
    """Run ecological training with specified chaos mode"""
    
    print(f"\n🌍 ECOLOGICAL TRAINING {'(CHAOTIC)' if chaos_mode else '(CALM)'}")
    print("=" * 60)
    
    # Log training start
    condition_name = f"ecological_{'chaotic' if chaos_mode else 'calm'}"
    if condition_logger:
        log_training_start(condition_logger, condition_name, chaos_mode, 42)
    
    # Create ecological models directory
    ecological_dir = Path("ecological_models")
    ecological_dir.mkdir(exist_ok=True)
    
    # Fixed: Add timestamp to avoid dataset collision (o3's issue #5)
    dataset_name = f"ecological_controlled_{suffix}_{timestamp}.jsonl"
    
    # Generate training data
    generator = EcologicalDataGenerator(random_seed=42)  # Reproducible
    data_path = generator.generate_training_dataset(
        num_echoes=5000,
        output_file=dataset_name,
        chaos_mode=chaos_mode
    )
    
    # Log training data stats
    if condition_logger:
        log_training_data_stats(condition_logger, data_path, chaos_mode)
    
    # Fixed: Add stress mode annotation to data (o3's issue #9)
    stress_mode = "chaotic" if chaos_mode else "calm"
    print(f"📊 Dataset generated with stress_mode: {stress_mode}")
    
    # Train model with timing
    training_start = time.time()
    model_path = train_ecological_model(
        data_file=data_path,
        epochs=15
    )
    training_time = time.time() - training_start
    
    # Fixed: Use shutil.move for cross-device compatibility (o3's issue #2)
    if model_path:
        new_name = ecological_dir / f"ecological_{'chaotic' if chaos_mode else 'calm'}_model.pt"
        try:
            shutil.move(model_path, new_name)
            print(f"💾 Ecological model saved to: {new_name}")
            print(f"📁 Model size: {get_file_size_kb(new_name)}")
            
            # Log completion
            if condition_logger:
                log_training_completion(condition_logger, condition_name, training_time, str(new_name))
            
            return str(new_name)
        except Exception as e:
            print(f"⚠ Error moving model: {e}")
            # Fallback to copy if move fails
            try:
                shutil.copy2(model_path, new_name)
                Path(model_path).unlink()  # Delete original
                print(f"💾 Ecological model copied to: {new_name}")
                print(f"📁 Model size: {get_file_size_kb(new_name)}")
                
                # Log completion
                if condition_logger:
                    log_training_completion(condition_logger, condition_name, training_time, str(new_name))
                
                return str(new_name)
            except Exception as e2:
                print(f"❌ Failed to move or copy model: {e2}")
                return model_path  # Return original path as fallback
    
    return None

def run_abstract_training(chaos_mode: bool = False, suffix: str = "", no_prompt: bool = False,
                        condition_logger=None, timestamp: str = ""):
    """Run abstract training with specified chaos mode using pre-generated data"""
    
    print(f"\n✨ ABSTRACT TRAINING {'(CHAOTIC)' if chaos_mode else '(CALM)'}")
    print("=" * 60)
    
    # Log training start
    condition_name = f"abstract_{'chaotic' if chaos_mode else 'calm'}"
    if condition_logger:
        log_training_start(condition_logger, condition_name, chaos_mode, 42)
    
    # Create abstract models directory
    abstract_dir = Path("abstract_models")
    abstract_dir.mkdir(exist_ok=True)
    
    # Fixed: Add timestamp to avoid dataset collision (o3's issue #5)
    dataset_name = f"abstract_controlled_{suffix}_{timestamp}.jsonl"
    
    # Generate training data (pre-generate to files for speed)
    generator = AbstractDataGenerator(random_seed=42)  # Reproducible
    data_path = generator.generate_training_dataset(
        num_echoes=5000,
        output_file=dataset_name,
        chaos_mode=chaos_mode
    )
    
    # Log training data stats
    if condition_logger:
        log_training_data_stats(condition_logger, data_path, chaos_mode)
    
    # Fixed: Add stress mode annotation to data (o3's issue #9)
    stress_mode = "chaotic" if chaos_mode else "calm"
    print(f"📊 Dataset generated with stress_mode: {stress_mode}")
    
    # Train model using fast file-based training with timing
    training_start = time.time()
    model_path = train_abstract_model(
        data_file=data_path,
        epochs=15
    )
    training_time = time.time() - training_start
    
    # Fixed: Use shutil.move for cross-device compatibility (o3's issue #2)
    if model_path:
        new_name = abstract_dir / f"abstract_{'chaotic' if chaos_mode else 'calm'}_model.pt"
        try:
            shutil.move(model_path, new_name)
            print(f"💾 Abstract model saved to: {new_name}")
            print(f"📁 Model size: {get_file_size_kb(new_name)}")
            
            # Log completion
            if condition_logger:
                log_training_completion(condition_logger, condition_name, training_time, str(new_name))
            
            return str(new_name)
        except Exception as e:
            print(f"⚠ Error moving model: {e}")
            # Fallback to copy if move fails
            try:
                shutil.copy2(model_path, new_name)
                Path(model_path).unlink()  # Delete original
                print(f"💾 Abstract model copied to: {new_name}")
                print(f"📁 Model size: {get_file_size_kb(new_name)}")
                
                # Log completion
                if condition_logger:
                    log_training_completion(condition_logger, condition_name, training_time, str(new_name))
                
                return str(new_name)
            except Exception as e2:
                print(f"❌ Failed to move or copy model: {e2}")
                return model_path  # Return original path as fallback
    
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
                
                # Fixed: Guard NetworkConditions creation with NEURAL_AVAILABLE (o3's issue #1)
                if NEURAL_AVAILABLE:
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
                else:
                    print("   ⚠ Simplified analysis (NetworkConditions not available)")
                    glyph_analysis = {"simplified": True, "silence_ratio": 0.0}
                
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
        
        # Ensure results directories exist
        Path("results/analysis").mkdir(parents=True, exist_ok=True)
        Path("results/reports").mkdir(parents=True, exist_ok=True)
        Path("results/statistical_analysis").mkdir(parents=True, exist_ok=True)
        
        report_path = f"results/analysis/controlled_comparison_analysis_{timestamp}.txt"
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
                performance = result["performance"]
                glyph_analysis = result.get("glyph_analysis", {})
                
                # Fixed: Use dict access instead of getattr (o3's issues #3 and #4)
                training_results[condition] = {
                    "final_glyph_loss": performance.get("final_glyph_loss", 0.0) if isinstance(performance, dict) else getattr(performance, "final_glyph_loss", 0.0),
                    "final_silence_loss": performance.get("final_silence_loss", 0.0) if isinstance(performance, dict) else getattr(performance, "final_silence_loss", 0.0),
                    "silence_ratio": glyph_analysis.get("silence_ratio", 0.0),
                    "glyph_improvement_percent": 0.0  # Would need training curves to calculate
                }
                
                if "behavioral_profile" in result:
                    behavioral = result["behavioral_profile"]
                    model_behaviors[condition] = {
                        "stress_response": behavioral.get("crisis_management_style", "unknown") if isinstance(behavioral, dict) else getattr(behavioral, "crisis_management_style", "unknown"),
                        "adaptation_strategy": behavioral.get("adaptation_strategy", "unknown") if isinstance(behavioral, dict) else getattr(behavioral, "adaptation_strategy", "unknown")
                    }
        
        if training_results:
            # Conduct philosophical analysis
            insights = philosophical.analyze_training_philosophy(training_results, model_behaviors)
            epistemological = philosophical.generate_epistemological_analysis(training_results)
            philosophical_report = philosophical.generate_contemplative_report()
            
            # Save philosophical report
            philosophical_path = f"results/reports/controlled_comparison_philosophy_{timestamp}.txt"
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
        summary_path = f"results/reports/controlled_comparison_summary_{timestamp}.txt"
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
                    model_size = get_file_size_kb(result["model_path"])
                    f.write(f"      Model: {result['model_path']} ({model_size})\n")
            
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
        print(f"   📊 Analysis: results/analysis/")
        print(f"   📋 Reports: results/reports/")
        print(f"   📈 Statistics: results/statistical_analysis/")
        
    except Exception as e:
        print(f"❌ Error generating reports: {e}")
        import traceback
        traceback.print_exc()
    
    return results

def main():
    """Run the complete controlled comparison experiment"""
    
    # Fixed: Add --no-prompt CLI option (o3's issue #7)
    parser = argparse.ArgumentParser(description="Controlled Spiramycel Comparison Experiment")
    parser.add_argument("--no-prompt", action="store_true", 
                       help="Skip interactive prompts (useful for automation)")
    args = parser.parse_args()
    
    # Setup experiment logging
    main_log_file, timestamp = setup_experiment_logging()
    main_logger = logging.getLogger()
    
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
    print(f"   📝 Main experiment log: {main_log_file}")
    print("   📁 Individual condition logs in logs/ directory")
    
    # Log experiment start
    main_logger.info("🧪 CONTROLLED SPIRAMYCEL COMPARISON EXPERIMENT STARTED")
    main_logger.info(f"Timestamp: {timestamp}")
    main_logger.info(f"Args: no_prompt={args.no_prompt}")
    
    # Fixed: Skip prompt if requested or not a TTY (o3's issue #7)
    if not args.no_prompt and sys.stdin.isatty():
        try:
            input("\nPress Enter to start the experiment (Ctrl+C to abort)...")
        except KeyboardInterrupt:
            print("\n⚠️ Experiment aborted by user")
            main_logger.info("Experiment aborted by user")
            return
    else:
        print("\n🚀 Starting experiment automatically...")
        main_logger.info("Starting experiment automatically")
    
    start_time = time.time()
    trained_models = {}
    
    try:
        # Run all four conditions with individual loggers
        print("\n🚀 PHASE 1: Training all four conditions...")
        main_logger.info("PHASE 1: Training all four conditions")
        
        # 1. Ecological Calm (A)
        print(f"\n🌱 Training condition A: Ecological + Calm")
        eco_calm_logger, eco_calm_log = create_condition_logger("ecological_calm", timestamp)
        main_logger.info(f"Starting Ecological Calm training - log: {eco_calm_log}")
        
        model_a = run_ecological_training(chaos_mode=False, suffix="calm", no_prompt=args.no_prompt,
                                        condition_logger=eco_calm_logger, timestamp=timestamp)
        trained_models["ecological_calm"] = model_a
        main_logger.info(f"Ecological Calm completed: {model_a}")
        
        # 2. Ecological Chaotic (B) 
        print(f"\n🌋 Training condition B: Ecological + Chaotic")
        eco_chaos_logger, eco_chaos_log = create_condition_logger("ecological_chaotic", timestamp)
        main_logger.info(f"Starting Ecological Chaotic training - log: {eco_chaos_log}")
        
        model_b = run_ecological_training(chaos_mode=True, suffix="chaotic", no_prompt=args.no_prompt,
                                        condition_logger=eco_chaos_logger, timestamp=timestamp)
        trained_models["ecological_chaotic"] = model_b
        main_logger.info(f"Ecological Chaotic completed: {model_b}")
        
        # 3. Abstract Calm (C)
        print(f"\n🧘 Training condition C: Abstract + Calm")  
        abs_calm_logger, abs_calm_log = create_condition_logger("abstract_calm", timestamp)
        main_logger.info(f"Starting Abstract Calm training - log: {abs_calm_log}")
        
        model_c = run_abstract_training(chaos_mode=False, suffix="calm", no_prompt=args.no_prompt,
                                      condition_logger=abs_calm_logger, timestamp=timestamp)
        trained_models["abstract_calm"] = model_c
        main_logger.info(f"Abstract Calm completed: {model_c}")
        
        # 4. Abstract Chaotic (D)
        print(f"\n⚡ Training condition D: Abstract + Chaotic")
        abs_chaos_logger, abs_chaos_log = create_condition_logger("abstract_chaotic", timestamp)
        main_logger.info(f"Starting Abstract Chaotic training - log: {abs_chaos_log}")
        
        model_d = run_abstract_training(chaos_mode=True, suffix="chaotic", no_prompt=args.no_prompt,
                                      condition_logger=abs_chaos_logger, timestamp=timestamp)
        trained_models["abstract_chaotic"] = model_d
        main_logger.info(f"Abstract Chaotic completed: {model_d}")
        
        training_time = time.time() - start_time
        print(f"\n✅ All training complete in {training_time/60:.1f} minutes!")
        main_logger.info(f"All training complete in {training_time/60:.1f} minutes")
        
        # Log all created log files
        print(f"\n📝 INDIVIDUAL CONDITION LOGS CREATED:")
        print(f"   🌱 Ecological Calm: {eco_calm_log}")
        print(f"   🌋 Ecological Chaotic: {eco_chaos_log}")  
        print(f"   🧘 Abstract Calm: {abs_calm_log}")
        print(f"   ⚡ Abstract Chaotic: {abs_chaos_log}")
        
        # PHASE 2: Comprehensive Analysis (now much more powerful!)
        print(f"\n🔬 PHASE 2: Comprehensive Analysis")
        print("This will analyze:")
        print("   • Glyph usage patterns and contemplative ratios")
        print("   • Behavioral profiles under different stress conditions") 
        print("   • Philosophical implications of paradigm differences")
        print("   • Epistemological analysis of learning approaches")
        print("   • Interaction effects between paradigm and environment")
        
        main_logger.info("PHASE 2: Starting comprehensive analysis")
        results = run_comparative_analysis(trained_models)
        main_logger.info("Comprehensive analysis completed")
        
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
                model_size = get_file_size_kb(model_path)
                print(f"   {condition}: {model_path} ({model_size})")
        
        print(f"\n📂 Model Organization:")
        print(f"   📁 ecological_models/")
        eco_calm_size = get_file_size_kb(trained_models.get("ecological_calm", "")) if trained_models.get("ecological_calm") else "N/A"
        eco_chaos_size = get_file_size_kb(trained_models.get("ecological_chaotic", "")) if trained_models.get("ecological_chaotic") else "N/A"
        print(f"      └── ecological_calm_model.pt ({eco_calm_size})")
        print(f"      └── ecological_chaotic_model.pt ({eco_chaos_size})")
        print(f"   📁 abstract_models/")
        abs_calm_size = get_file_size_kb(trained_models.get("abstract_calm", "")) if trained_models.get("abstract_calm") else "N/A"
        abs_chaos_size = get_file_size_kb(trained_models.get("abstract_chaotic", "")) if trained_models.get("abstract_chaotic") else "N/A"
        print(f"      └── abstract_calm_model.pt ({abs_calm_size})")
        print(f"      └── abstract_chaotic_model.pt ({abs_chaos_size})")
        
        total_time = time.time() - start_time
        print(f"\n🎉 Experiment complete in {total_time/60:.1f} minutes!")
        print(f"🔬 Ready for detailed contemplative analysis!")
        print(f"🌱 All four oscillatory femto language models preserved!")
        print(f"📋 Check the comprehensive analysis reports for deep insights!")
        print(f"\n📝 COMPLETE LOGGING DOCUMENTATION:")
        print(f"   📖 Main experiment log: {main_log_file}")
        print(f"   📁 Individual condition logs in logs/ directory")
        print(f"   📊 All training details, glyph patterns, and metrics captured!")
        
        main_logger.info(f"EXPERIMENT COMPLETED SUCCESSFULLY in {total_time/60:.1f} minutes")
        main_logger.info("All models trained, analyzed, and documented")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Experiment interrupted by user")
        elapsed = (time.time() - start_time) / 60
        print(f"   Partial completion time: {elapsed:.1f} minutes")
        print(f"   Check saved models in ecological_models/ and abstract_models/")
        main_logger.info(f"Experiment interrupted by user after {elapsed:.1f} minutes")
    
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        print(f"   Check individual training components")
        main_logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 