#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

TARGET_METRICS = ['accuracy', 'f1_score', 'precision', 'recall', 'error_rate']
BASE_STRATEGIES = ['FedAvg', 'FedCRA', 'FedProx']


def load_final_metrics(results_base: Path, strategy: str, alpha: float, num_clients: int):
    alpha_str = f"{alpha:.1f}" if alpha == 1.0 else str(alpha)
    metrics_path = (
        results_base
        / strategy
        / f"num_clients_{num_clients}"
        / f"dirichlet_alpha_{alpha_str}"
        / strategy
        / "DNN"
        / "metrics"
        / "server_metrics.json"
    )
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Failed to load {metrics_path}: {exc}")
    if not data:
        return None
    return data[-1]


def format_change(current, baseline, invert=False):
    if current is None or baseline is None:
        return 'n/a'
    diff = current - baseline
    if baseline == 0:
        pct = float('inf') if diff > 0 else float('-inf') if diff < 0 else 0.0
    else:
        pct = diff / abs(baseline) * 100.0
    if invert:
        diff = -diff
        pct = -pct
    return f"{diff:+.4f} ({pct:+5.1f}%)"


def strategy_status(fedcra, baseline):
    if fedcra is None or baseline is None:
        return 'missing'
    if fedcra['accuracy'] >= baseline['accuracy'] and fedcra['f1_score'] >= baseline['f1_score'] and fedcra['error_rate'] <= baseline['error_rate']:
        return 'better'
    if fedcra['accuracy'] < baseline['accuracy'] and fedcra['f1_score'] < baseline['f1_score']:
        return 'worse'
    return 'mixed'


def compare_config(results_base: Path, alpha: float, num_clients: int):
    rows = {}
    for strategy in BASE_STRATEGIES:
        rows[strategy] = load_final_metrics(results_base, strategy, alpha, num_clients)

    if rows['FedCRA'] is None:
        print(f"\n[WARN] FedCRA metrics missing for alpha={alpha}, clients={num_clients}")
        return None

    if rows['FedAvg'] is None:
        print(f"\n[WARN] FedAvg baseline missing for alpha={alpha}, clients={num_clients}")

    print(f"\n=== alpha={alpha:.1f}, clients={num_clients} ===")
    print(f"{'Strategy':<10} {'accuracy':>10} {'f1_score':>10} {'precision':>10} {'recall':>10} {'error':>10}")
    print('-' * 66)
    for strategy in BASE_STRATEGIES:
        metrics = rows[strategy]
        if metrics is None:
            print(f"{strategy:<10} {'MISSING':>56}")
            continue
        print(
            f"{strategy:<10} "
            f"{metrics.get('accuracy', 0):>10.4f} "
            f"{metrics.get('f1_score', 0):>10.4f} "
            f"{metrics.get('precision', 0):>10.4f} "
            f"{metrics.get('recall', 0):>10.4f} "
            f"{metrics.get('error_rate', 0):>10.4f}"
        )

    baselines = rows['FedAvg']
    fedcra = rows['FedCRA']
    if baselines is not None:
        print('\nFedCRA vs FedAvg:')
        for metric in TARGET_METRICS:
            invert = metric == 'error_rate'
            print(f"  {metric:<10}: {format_change(fedcra.get(metric), baselines.get(metric), invert=invert)}")
        status = strategy_status(fedcra, baselines)
        print(f"\nSummary: FedCRA vs FedAvg => {status.upper()}")
    else:
        print("\nNo FedAvg baseline available for this config.")
    return {'strategy_metrics': rows, 'status': status if baselines is not None else 'no_baseline'}


def main():
    parser = argparse.ArgumentParser(description='Check whether FedCRA updates improved results vs FedAvg')
    parser.add_argument('--results-base', default='./dataset/models/iomt_traffic/Category', help='Base path for results')
    parser.add_argument('--alpha', type=float, default=0.1, help='Dirichlet alpha to analyze')
    parser.add_argument('--clients', nargs='+', type=int, default=[5, 10, 15], help='Client counts to check')
    args = parser.parse_args()

    results_base = Path(args.results_base)
    if not results_base.exists():
        raise FileNotFoundError(f"Results base not found: {results_base}")

    print("Checking FedCRA update status for:")
    print(f"  results base: {results_base}")
    print(f"  alpha: {args.alpha}")
    print(f"  client counts: {args.clients}")

    summary = []
    for num_clients in args.clients:
        result = compare_config(results_base, args.alpha, num_clients)
        if result is not None:
            summary.append((num_clients, result['status']))

    print("\n=== Overall summary ===")
    for num_clients, status in summary:
        print(f"clients={num_clients:2d}: {status}")

    if all(status == 'better' for _, status in summary):
        print("\nResult: FedCRA is better than FedAvg in all inspected configs.")
    elif any(status == 'worse' for _, status in summary):
        print("\nResult: FedCRA is worse than FedAvg in one or more configs.")
    else:
        print("\nResult: Some configs are mixed or missing baseline data.")


if __name__ == '__main__':
    main()
