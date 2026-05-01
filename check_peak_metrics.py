#!/usr/bin/env python3
"""
Extract peak F1 metrics from all 4 strategies for 1:1000 imbalance
"""
import json
from pathlib import Path

base_path = Path("dataset/models/iomt_traffic/Category/imbalanced_1_1000")
strategies = ['FedAvg', 'FedProx', 'FLAME', 'FedCRA']

print("\n" + "=" * 100)
print("PEAK METRICS COMPARISON - 1:1000 IMBALANCE RATIO")
print("=" * 100)

results = {}
for strat in strategies:
    metrics_file = base_path / strat / "DNN/metrics/server_metrics.json"
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    # Find peak F1 score
    max_f1_idx = max(range(len(metrics)), key=lambda i: metrics[i]['f1_score'])
    max_f1_round = metrics[max_f1_idx]
    
    results[strat] = {
        'peak_f1': max_f1_round['f1_score'],
        'peak_round': max_f1_round['round'],
        'final_f1': metrics[-1]['f1_score'],
        'num_rounds': len(metrics),
        'max_accuracy': max(m['accuracy'] for m in metrics),
        'max_macro_fpr': max(m['macro_fpr'] for m in metrics),
    }
    
print("\n" + "Strategy Results:")
print("-" * 100)
print(f"{'Strategy':<15} {'Peak F1':<15} {'Round':<10} {'Final F1':<15} {'#Rounds':<10}")
print("-" * 100)

for strat in strategies:
    r = results[strat]
    print(f"{strat:<15} {r['peak_f1']:<15.6f} {r['peak_round']:<10} {r['final_f1']:<15.6f} {r['num_rounds']:<10}")

print("\n" + "=" * 100)
print("ANALYSIS")
print("=" * 100)

# Check if all strategies have same peak F1
peak_f1_values = [results[s]['peak_f1'] for s in strategies]
all_same = len(set(round(v, 6) for v in peak_f1_values)) == 1

if all_same:
    print(f"\n⚠️  PROBLEM DETECTED: All 4 strategies converge to the SAME peak F1 = {peak_f1_values[0]:.6f}")
    print("    This suggests strategies are not learning differently!")
else:
    print(f"\n✓ Different peak F1 values detected:")
    for strat in strategies:
        print(f"    {strat}: {results[strat]['peak_f1']:.6f}")

# Sort by performance
print("\n" + "Ranking (by peak F1):")
print("-" * 100)
sorted_strats = sorted(strategies, key=lambda s: results[s]['peak_f1'], reverse=True)
for i, strat in enumerate(sorted_strats, 1):
    r = results[strat]
    print(f"  {i}. {strat:<15} Peak F1 = {r['peak_f1']:.6f} (round {r['peak_round']})")
