#!/usr/bin/env python3
import json
from pathlib import Path

base_path = Path("dataset/models/iomt_traffic/Category/imbalanced_1_1000")
strategies = ['FedAvg', 'FedProx', 'FLAME', 'FedCRA']

# Load metrics for all strategies
data = {}
for strat in strategies:
    metrics_file = base_path / strat / "DNN/metrics/server_metrics.json"
    with open(metrics_file) as f:
        data[strat] = json.load(f)

# Extract key metrics
print("=" * 100)
print("COMPARISON OF ALL 4 STRATEGIES FOR 1:1000 IMBALANCE")
print("=" * 100)

print("\nRound-by-round F1 Scores:")
print("-" * 100)
print(f"{'Round':<8}", end="")
for strat in strategies:
    print(f"{strat:<20}", end="")
print()
print("-" * 100)

for round_idx in range(min(15, min(len(data[s]) for s in strategies))):
    print(f"{round_idx:<8}", end="")
    for strat in strategies:
        f1 = data[strat][round_idx]['f1_score']
        print(f"{f1:<20.6f}", end="")
    print()

print("\n" + "=" * 100)
print("CONVERGENCE ANALYSIS")
print("=" * 100)

for strat in strategies:
    metrics = data[strat]
    f1_scores = [m['f1_score'] for m in metrics]
    
    print(f"\n{strat}:")
    print(f"  Total rounds: {len(f1_scores)}")
    print(f"  Max F1: {max(f1_scores):.6f}")
    print(f"  Min F1: {min(f1_scores):.6f}")
    print(f"  Final F1 (last round): {f1_scores[-1]:.6f}")
    print(f"  Last 5 F1 scores: {[f'{x:.6f}' for x in f1_scores[-5:]]}")
    
    # Check when model converges (F1 reaches 0.74+)
    for i, f1 in enumerate(f1_scores):
        if f1 >= 0.74:
            print(f"  Converges to ~0.74+ at round: {i}")
            break

print("\n" + "=" * 100)
print("ARE FINAL METRICS THE SAME?")
print("=" * 100)

final_f1_scores = {strat: data[strat][-1]['f1_score'] for strat in strategies}
print("\nFinal F1 Scores:")
for strat in strategies:
    print(f"  {strat}: {final_f1_scores[strat]:.6f}")

if len(set(round(v, 6) for v in final_f1_scores.values())) == 1:
    print("\n⚠️  ALL STRATEGIES CONVERGE TO THE SAME F1 SCORE!")
else:
    print("\n✓ Strategies have DIFFERENT final F1 scores")

# Check if training metrics are identical
print("\n" + "=" * 100)
print("CHECKING IF TRAINING ROUNDS HAVE IDENTICAL LOSS VALUES")
print("=" * 100)

for round_idx in range(min(5, min(len(data[s]) for s in strategies))):
    loss_values = {strat: data[strat][round_idx]['loss'] for strat in strategies}
    all_same = len(set(round(v, 10) for v in loss_values.values())) == 1
    
    print(f"\nRound {round_idx} Loss Values:")
    for strat in strategies:
        print(f"  {strat}: {loss_values[strat]:.10f}")
    
    if all_same and round_idx > 0:
        print("  ⚠️  ALL SAME!")
