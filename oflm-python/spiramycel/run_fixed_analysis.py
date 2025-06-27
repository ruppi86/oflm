#!/usr/bin/env python3
"""
Run Fixed Comparative Analysis

Uses the corrected analysis code to generate accurate results from the 
successful controlled comparison experiment.
"""

import time
from comparative_analysis import SpiramycelComparativeAnalyzer

def main():
    """Run the fixed analysis with correct timestamp"""
    print("🔧 RUNNING FIXED COMPARATIVE ANALYSIS")
    print("=" * 60)
    
    # Use the timestamp from our successful run
    timestamp = "20250613_054843"
    
    print(f"📅 Using timestamp: {timestamp}")
    print("🔍 Loading data from successful controlled comparison...")
    
    # Initialize analyzer
    analyzer = SpiramycelComparativeAnalyzer()
    
    # Generate corrected report
    report = analyzer.generate_full_report(timestamp=timestamp)
    
    # Display results
    print("\n" + "="*80)
    print("📋 CORRECTED ANALYSIS RESULTS:")
    print("="*80)
    print(report)
    
    # Save corrected report
    corrected_report_path = f"controlled_comparison_analysis_CORRECTED_{timestamp}.txt"
    with open(corrected_report_path, 'w', encoding='utf-8') as f:
        f.write("🧪 CONTROLLED COMPARISON EXPERIMENT - CORRECTED ANALYSIS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Original Timestamp: {timestamp}\n\n")
        f.write("🎯 EXPERIMENTAL DESIGN: 2×2 (Ecological/Abstract × Calm/Chaotic)\n\n")
        f.write(report)
    
    print(f"\n✅ CORRECTED ANALYSIS SAVED: {corrected_report_path}")
    print("\n🌟 Summary of Fixes Applied:")
    print("   • Extracted real training times from logs")
    print("   • Used correct model file paths")
    print("   • Parsed actual silence ratios from controlled comparison log")
    print("   • Generated proper glyph pattern analysis")
    print("   • Fixed paradigm and stress response analysis")

if __name__ == "__main__":
    main() 