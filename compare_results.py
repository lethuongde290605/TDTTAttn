#!/usr/bin/env python3
"""
Compare evaluation results between baseline and compressed models
"""

import json
import sys
from pathlib import Path

def load_results(json_path):
    """Load results from JSON file"""
    if not Path(json_path).exists():
        return None
    
    with open(json_path, 'r') as f:
        return json.load(f)

def print_comparison(baseline_file, compressed_files):
    """Print comparison table"""
    
    baseline = load_results(baseline_file)
    if not baseline:
        print(f"❌ Baseline results not found: {baseline_file}")
        return
    
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    print()
    
    # Header
    header = f"{'Task':<20} {'Baseline':>10}"
    compressed_results = []
    for comp_file in compressed_files:
        comp = load_results(comp_file)
        if comp:
            compressed_results.append(comp)
            label = Path(comp_file).parent.name
            header += f" {label:>12}"
    
    print(header)
    print("-" * len(header))
    
    # Per-task results
    tasks = baseline.get('tasks', [])
    for task in tasks:
        row = f"{task:<20}"
        
        # Baseline
        baseline_acc = baseline.get('task_results', {}).get(task, {}).get('acc', 0.0)
        row += f" {baseline_acc:>10.4f}"
        
        # Compressed
        for comp in compressed_results:
            comp_acc = comp.get('task_results', {}).get(task, {}).get('acc', 0.0)
            delta = comp_acc - baseline_acc
            row += f" {comp_acc:>7.4f}"
            if abs(delta) > 0.0001:
                row += f"({delta:+.2f})"
            else:
                row += "     "
        
        print(row)
    
    print("-" * len(header))
    
    # Average
    row = f"{'Average':<20}"
    baseline_avg = baseline.get('avg_acc', 0.0)
    row += f" {baseline_avg:>10.4f}"
    
    for comp in compressed_results:
        comp_avg = comp.get('avg_acc', 0.0)
        delta = comp_avg - baseline_avg
        row += f" {comp_avg:>7.4f}"
        if abs(delta) > 0.0001:
            row += f"({delta:+.2f})"
        else:
            row += "     "
    
    print(row)
    print("="*80)
    print()

def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_results.py <baseline_json> [compressed_json...]")
        print("\nExample:")
        print("  python compare_results.py logs/baseline/results_summary.json \\")
        print("                            logs/err0.01/results_summary.json \\")
        print("                            logs/err0.025/results_summary.json")
        sys.exit(1)
    
    baseline_file = sys.argv[1]
    compressed_files = sys.argv[2:] if len(sys.argv) > 2 else []
    
    if not compressed_files:
        # Auto-discover compressed results
        logs_dir = Path(baseline_file).parent.parent
        compressed_files = sorted(logs_dir.glob('err*/results_summary.json'))
        if compressed_files:
            print(f"Auto-discovered {len(compressed_files)} compressed results")
    
    print_comparison(baseline_file, compressed_files)

if __name__ == '__main__':
    main()
