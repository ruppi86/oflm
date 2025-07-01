#!/usr/bin/env python3
"""
600K Training Convergence Analysis

Investigates whether 600K models actually converged properly compared to 200K models.
Analyzes training logs, loss patterns, and convergence behavior.

Author: Claude
Date: 2025-06-29
"""

import re
import glob
from pathlib import Path
import json
from collections import defaultdict
import matplotlib.pyplot as plt

def parse_training_logs():
    """Parse training logs to extract convergence data"""
    print("🔍 ANALYZING TRAINING CONVERGENCE PATTERNS")
    print("=" * 50)
    
    # Find all training logs
    log_patterns = [
        "logs/controlled_comparison_*.log",
        "logs/ecological_*_*.log", 
        "logs/abstract_*_*.log"
    ]
    
    all_logs = []
    for pattern in log_patterns:
        all_logs.extend(glob.glob(pattern))
    
    # Sort by modification time
    all_logs.sort(key=lambda f: Path(f).stat().st_mtime, reverse=True)
    
    print(f"📄 Found {len(all_logs)} log files")
    
    # Data structures to store training metrics
    training_data = {
        '200k': {'ecological': {}, 'abstract': {}},
        '600k': {'ecological': {}, 'abstract': {}}
    }
    
    for log_file in all_logs:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if this log contains 200k or 600k training
            scale = None
            if "200k" in content or "piko-scale" in content:
                scale = "200k"
            elif "600k" in content or "nano-scale" in content:
                scale = "600k"
            else:
                continue  # Skip logs that don't match our scales
            
            print(f"\n📋 Analyzing {Path(log_file).name} ({scale} scale)")
            
            # Extract paradigm
            paradigm = None
            if "ecological" in log_file.lower():
                paradigm = "ecological"
            elif "abstract" in log_file.lower():
                paradigm = "abstract"
            
            if not paradigm:
                # Try to detect from content
                if "ecological" in content.lower():
                    paradigm = "ecological"
                elif "abstract" in content.lower():
                    paradigm = "abstract"
                else:
                    continue
            
            # Extract training completion status
            training_completed = "TRAINING COMPLETED" in content
            print(f"   Training completed: {'✅' if training_completed else '❌'}")
            
            # Extract training duration
            duration_match = re.search(r'Duration: ([\d.]+) minutes', content)
            if duration_match:
                duration_minutes = float(duration_match.group(1))
                print(f"   Duration: {duration_minutes:.1f} minutes")
                training_data[scale][paradigm]['duration'] = duration_minutes
            
            # Extract parameter count
            param_matches = re.findall(r'Parameters: ([\d,]+)', content)
            if param_matches:
                # Remove commas and convert to int
                param_count = int(param_matches[-1].replace(',', ''))
                print(f"   Parameters: {param_count:,}")
                training_data[scale][paradigm]['parameters'] = param_count
            
            # Extract loss patterns (look for epoch-by-epoch losses)
            glyph_losses = []
            effectiveness_losses = []
            silence_losses = []
            
            # Pattern to match loss reports
            loss_pattern = r'Glyph loss: ([\d.]+).*?Effectiveness loss: ([\d.]+).*?Silence loss: ([\d.]+)'
            loss_matches = re.findall(loss_pattern, content, re.DOTALL)
            
            for glyph_loss, eff_loss, silence_loss in loss_matches:
                glyph_losses.append(float(glyph_loss))
                effectiveness_losses.append(float(eff_loss))
                silence_losses.append(float(silence_loss))
            
            if glyph_losses:
                print(f"   Epoch data points: {len(glyph_losses)}")
                print(f"   Final glyph loss: {glyph_losses[-1]:.4f}")
                print(f"   Initial → Final glyph loss: {glyph_losses[0]:.4f} → {glyph_losses[-1]:.4f}")
                
                # Calculate convergence metrics
                if len(glyph_losses) >= 2:
                    total_improvement = glyph_losses[0] - glyph_losses[-1]
                    final_5_epochs = glyph_losses[-5:] if len(glyph_losses) >= 5 else glyph_losses
                    final_stability = max(final_5_epochs) - min(final_5_epochs)
                    
                    print(f"   Loss improvement: {total_improvement:.4f}")
                    print(f"   Final stability (last 5 epochs): ±{final_stability:.4f}")
                    
                    training_data[scale][paradigm].update({
                        'glyph_losses': glyph_losses,
                        'effectiveness_losses': effectiveness_losses,
                        'silence_losses': silence_losses,
                        'total_improvement': total_improvement,
                        'final_stability': final_stability,
                        'epochs_trained': len(glyph_losses)
                    })
            
            # Check for architecture errors
            if "Could not analyze model architecture" in content:
                print("   🚨 Architecture analysis errors found!")
                training_data[scale][paradigm]['architecture_errors'] = True
            
            # Extract model file size
            file_size_match = re.search(r'File Size: ([\d.]+) KB', content)
            if file_size_match:
                file_size_kb = float(file_size_match.group(1))
                print(f"   Model file size: {file_size_kb:.0f} KB")
                training_data[scale][paradigm]['file_size_kb'] = file_size_kb
                
        except Exception as e:
            print(f"   ❌ Error reading {log_file}: {e}")
    
    return training_data

def analyze_convergence_patterns(training_data):
    """Analyze convergence patterns between 200K and 600K models"""
    print("\n📊 CONVERGENCE PATTERN ANALYSIS")
    print("=" * 40)
    
    for paradigm in ['ecological', 'abstract']:
        print(f"\n🔬 {paradigm.upper()} PARADIGM:")
        
        # Compare 200K vs 600K
        data_200k = training_data['200k'][paradigm]
        data_600k = training_data['600k'][paradigm]
        
        # Training completion
        completed_200k = data_200k.get('epochs_trained', 0) > 0
        completed_600k = data_600k.get('epochs_trained', 0) > 0
        
        print(f"   200K training: {'✅ Completed' if completed_200k else '❌ Failed'}")
        print(f"   600K training: {'✅ Completed' if completed_600k else '❌ Failed'}")
        
        if completed_200k and completed_600k:
            # Compare convergence metrics
            print(f"\n   📈 CONVERGENCE COMPARISON:")
            
            # Loss improvement
            improvement_200k = data_200k.get('total_improvement', 0)
            improvement_600k = data_600k.get('total_improvement', 0)
            print(f"   Loss improvement: 200K={improvement_200k:.4f}, 600K={improvement_600k:.4f}")
            
            # Final stability
            stability_200k = data_200k.get('final_stability', float('inf'))
            stability_600k = data_600k.get('final_stability', float('inf'))
            print(f"   Final stability: 200K=±{stability_200k:.4f}, 600K=±{stability_600k:.4f}")
            
            # Training duration
            duration_200k = data_200k.get('duration', 0)
            duration_600k = data_600k.get('duration', 0)
            print(f"   Training time: 200K={duration_200k:.1f}min, 600K={duration_600k:.1f}min")
            
            # Epochs trained
            epochs_200k = data_200k.get('epochs_trained', 0)
            epochs_600k = data_600k.get('epochs_trained', 0)
            print(f"   Epochs completed: 200K={epochs_200k}, 600K={epochs_600k}")
            
            # Convergence assessment
            print(f"\n   🎯 CONVERGENCE ASSESSMENT:")
            
            # Check if 600K converged properly
            if improvement_600k < improvement_200k * 0.5:
                print(f"   ⚠️  600K shows poor improvement compared to 200K")
            
            if stability_600k > stability_200k * 2:
                print(f"   ⚠️  600K shows poor final stability")
            
            if epochs_600k < epochs_200k:
                print(f"   ⚠️  600K training terminated early")
            
            # Check if 600K might need more training
            if data_600k.get('glyph_losses'):
                final_losses = data_600k['glyph_losses'][-3:]  # Last 3 epochs
                if len(final_losses) >= 2:
                    recent_trend = final_losses[-1] - final_losses[0]
                    if recent_trend < -0.001:  # Still decreasing
                        print(f"   📈 600K loss still decreasing - may need more epochs")
                    elif abs(recent_trend) < 0.0001:
                        print(f"   ✅ 600K loss stabilized")
                    else:
                        print(f"   📈 600K loss increasing - possible overfitting")

def create_convergence_plots(training_data):
    """Create plots comparing 200K vs 600K convergence"""
    print("\n📊 CREATING CONVERGENCE PLOTS")
    print("=" * 30)
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Training Convergence: 200K vs 600K Models', fontsize=14)
        
        paradigms = ['ecological', 'abstract']
        
        for i, paradigm in enumerate(paradigms):
            # Glyph loss plot
            ax_glyph = axes[i, 0]
            ax_glyph.set_title(f'{paradigm.title()} - Glyph Loss')
            ax_glyph.set_xlabel('Epoch')
            ax_glyph.set_ylabel('Glyph Loss')
            
            # Effectiveness loss plot  
            ax_eff = axes[i, 1]
            ax_eff.set_title(f'{paradigm.title()} - Effectiveness Loss')
            ax_eff.set_xlabel('Epoch')
            ax_eff.set_ylabel('Effectiveness Loss')
            
            # Plot 200K data
            data_200k = training_data['200k'][paradigm]
            if 'glyph_losses' in data_200k:
                epochs_200k = range(1, len(data_200k['glyph_losses']) + 1)
                ax_glyph.plot(epochs_200k, data_200k['glyph_losses'], 
                             label='200K', color='blue', marker='o', linewidth=2)
                ax_eff.plot(epochs_200k, data_200k['effectiveness_losses'], 
                           label='200K', color='blue', marker='o', linewidth=2)
            
            # Plot 600K data
            data_600k = training_data['600k'][paradigm]
            if 'glyph_losses' in data_600k:
                epochs_600k = range(1, len(data_600k['glyph_losses']) + 1)
                ax_glyph.plot(epochs_600k, data_600k['glyph_losses'], 
                             label='600K', color='red', marker='s', linewidth=2)
                ax_eff.plot(epochs_600k, data_600k['effectiveness_losses'], 
                           label='600K', color='red', marker='s', linewidth=2)
            
            ax_glyph.legend()
            ax_glyph.grid(True, alpha=0.3)
            ax_eff.legend()
            ax_eff.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = 'results/convergence_analysis_200k_vs_600k.png'
        Path('results').mkdir(exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Convergence plot saved: {plot_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"   ❌ Error creating plots: {e}")

def main():
    """Run the complete training analysis"""
    print("🔍 600K TRAINING CONVERGENCE INVESTIGATION")
    print("=" * 50)
    print("Step 1: Analyzing if 600K models converged properly")
    print()
    
    # Parse training logs
    training_data = parse_training_logs()
    
    # Analyze convergence patterns
    analyze_convergence_patterns(training_data)
    
    # Create visualizations
    create_convergence_plots(training_data)
    
    print("\n🎯 TRAINING ANALYSIS COMPLETE")
    print("=" * 35)
    print("Key Questions Answered:")
    print("1. Did 600K models complete training?")
    print("2. Did they converge as well as 200K models?") 
    print("3. Do they need more epochs/different learning rates?")
    print("4. Are there signs of overfitting or underfitting?")

if __name__ == "__main__":
    main() 