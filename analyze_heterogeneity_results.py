#!/usr/bin/env python3
"""
Heterogeneity & Class Imbalance Experiment Analysis
Compares FedCRA vs FedAvg across different Dirichlet alpha values.

This script:
- Reads metrics from all experiments
- Compares FedCRA vs FedAvg for each alpha value
- Generates comprehensive comparison report
- Creates visualizations of results
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

CLASS_NAMES = {
    0: "Benign",
    1: "DDoS",
    2: "DOS",
    3: "MQTT",
    4: "Recon",
}
MINORITY_CLASSES = {0, 3, 4}

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    print("Warning: pandas/matplotlib/seaborn not available. Skipping visualizations.")


class HeterogeneityAnalyzer:
    """Analyzes federated learning experiments under heterogeneous conditions."""
    
    def __init__(self, results_base_path):
        self.results_base = Path(results_base_path)
        self.experiments = {}
        self.comparison_results = {}
        
    def load_metrics(self, strategy_name, alpha, num_clients):
        """Load metrics for a specific strategy, alpha value, and number of clients."""
        # Handle alpha formatting (1.0 vs 1)
        alpha_str = f"{alpha:.1f}" if alpha == 1.0 else str(alpha)
        metrics_path = (self.results_base / strategy_name / 
                       f"num_clients_{num_clients}" / f"dirichlet_alpha_{alpha_str}" /
                       strategy_name / "DNN" / "metrics" / 
                       "server_metrics.json")
        
        if not metrics_path.exists():
            return None
        
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return metrics
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {metrics_path}: {e}")
            return None
    
    def load_all_experiments(self, strategies=None, alphas=None, num_clients_list=None):
        """Load all experiment metrics."""
        if strategies is None:
            strategies = [
                "FedAvg", "FedCRA", "FedProx", "FedSCaffold",
                "FedLC", "FedFocal", "FedBB", "FedLTA", "FLAME",
            ]
        if alphas is None:
            alphas = [0.1, 0.3, 0.5, 1.0, 5.0]  # Updated to match actual directory names
        
        # Auto-discover num_clients if not specified
        if num_clients_list is None:
            num_clients_list = []
            for strategy in strategies:
                strategy_path = self.results_base / strategy
                if strategy_path.exists():
                    for item in strategy_path.iterdir():
                        if item.is_dir() and item.name.startswith("num_clients_"):
                            try:
                                num_clients = int(item.name.split("_")[2])
                                if num_clients not in num_clients_list:
                                    num_clients_list.append(num_clients)
                            except (ValueError, IndexError):
                                continue
            num_clients_list.sort()
        
        if not num_clients_list:
            # Fallback to default
            num_clients_list = [5, 10, 15]
        
        print(f"Analyzing strategies: {strategies}")
        print(f"Analyzing alphas: {alphas}")
        print(f"Analyzing num_clients: {num_clients_list}")
        print()
        
        for strategy in strategies:
            for alpha in alphas:
                for num_clients in num_clients_list:
                    key = f"{strategy}_alpha_{alpha}_clients_{num_clients}"
                    metrics = self.load_metrics(strategy, alpha, num_clients)
                    if metrics:
                        self.experiments[key] = {
                            'strategy': strategy,
                            'alpha': alpha,
                            'num_clients': num_clients,
                            'metrics': metrics
                        }
                    else:
                        print(f"⚠ No metrics found for {key}")
    
    def get_experiment_entries(self, strategy=None, alpha=None, num_clients=None):
        """Return experiment entries filtered by strategy, alpha, and num_clients."""
        entries = []
        for exp in self.experiments.values():
            if strategy is not None and exp['strategy'] != strategy:
                continue
            if alpha is not None and exp['alpha'] != alpha:
                continue
            if num_clients is not None and exp['num_clients'] != num_clients:
                continue
            entries.append(exp)
        return entries

    def get_final_metrics(self, strategy, alpha, num_clients=None):
        """Get final round metrics for a strategy and alpha, optionally filtered by num_clients."""
        entries = self.get_experiment_entries(strategy=strategy, alpha=alpha, num_clients=num_clients)
        if not entries:
            return None

        final_metrics_list = []
        for exp in entries:
            metrics = exp.get('metrics')
            if metrics:
                final_metrics_list.append(metrics[-1])

        if not final_metrics_list:
            return None

        if len(final_metrics_list) == 1:
            return final_metrics_list[0]

        aggregated = {}
        all_keys = set().union(*(m.keys() for m in final_metrics_list))

        for key in all_keys:
            values = [m[key] for m in final_metrics_list if key in m]
            if not values:
                continue

            if all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
                aggregated[key] = float(np.mean(values))
            elif all(isinstance(v, dict) for v in values):
                nested = {}
                nested_keys = set().union(*(v.keys() for v in values))
                for subkey in nested_keys:
                    subvals = [v[subkey] for v in values if subkey in v]
                    if all(isinstance(x, (int, float, np.integer, np.floating)) for x in subvals):
                        nested[subkey] = float(np.mean(subvals))
                    else:
                        nested[subkey] = subvals[0]
                aggregated[key] = nested
            else:
                aggregated[key] = values[0]

        return aggregated

    def compare_strategies_for_alpha(self, alpha, num_clients=None):
        """Compare all strategies for a specific alpha value and optional number of clients."""
        strategy_metrics = {}
        for strategy in ["FedAvg", "FedCRA", "FedProx"]:
            metrics = self.get_final_metrics(strategy, alpha, num_clients=num_clients)
            if metrics:
                strategy_metrics[strategy] = metrics

        if len(strategy_metrics) < 2:
            return None

        comparison = {
            'alpha': alpha,
            'num_clients': num_clients,
            'strategies': strategy_metrics,
            'pairwise_comparisons': {}
        }

        strategies_list = list(strategy_metrics.keys())
        for i in range(len(strategies_list)):
            for j in range(i+1, len(strategies_list)):
                strat1 = strategies_list[i]
                strat2 = strategies_list[j]

                comparison['pairwise_comparisons'][f"{strat1}_vs_{strat2}"] = {}

                for metric in ['accuracy', 'f1_score', 'precision', 'recall', 'error_rate']:
                    if (metric in strategy_metrics[strat1] and 
                        metric in strategy_metrics[strat2]):
                        val1 = strategy_metrics[strat1][metric]
                        val2 = strategy_metrics[strat2][metric]

                        if isinstance(val1, (int, float, np.integer, np.floating)) and isinstance(val2, (int, float, np.integer, np.floating)):
                            diff = val2 - val1
                            improvement = (diff / (abs(val1) + 1e-10) * 100)
                            comparison['pairwise_comparisons'][f"{strat1}_vs_{strat2}"][metric] = {
                                strat1: float(val1),
                                strat2: float(val2),
                                'difference': float(diff),
                                'improvement_percent': float(improvement)
                            }

        return comparison
                            improvement = (diff / (abs(val1) + 1e-10) * 100)
                            comparison['pairwise_comparisons'][f"{strat1}_vs_{strat2}"][metric] = {
                                strat1: val1,
                                strat2: val2,
                                'difference': diff,
                                'improvement_percent': improvement
                            }
        
        return comparison
    
    def analyze_heterogeneity_impact(self):
        """Analyze how heterogeneity (alpha) and number of clients affect each strategy."""
        impact_results = {}
        
        # Get all unique combinations
        strategies = set()
        alphas = set()
        num_clients_list = set()
        
        for exp_key, exp_data in self.experiments.items():
            strategies.add(exp_data['strategy'])
            alphas.add(exp_data['alpha'])
            num_clients_list.add(exp_data['num_clients'])
        
        strategies = sorted(strategies)
        alphas = sorted(alphas)
        num_clients_list = sorted(num_clients_list)
        
        for strategy in strategies:
            impact_results[strategy] = {}
            
            for num_clients in num_clients_list:
                impact_results[strategy][num_clients] = {
                    'alphas': [],
                    'accuracies': [],
                    'f1_scores': [],
                    'error_rates': []
                }
                
                for alpha in alphas:
                    metrics = self.get_final_metrics(strategy, alpha, num_clients)
                    if metrics:
                        impact_results[strategy][num_clients]['alphas'].append(alpha)
                        impact_results[strategy][num_clients]['accuracies'].append(metrics.get('accuracy', 0))
                        impact_results[strategy][num_clients]['f1_scores'].append(metrics.get('f1_score', 0))
                        impact_results[strategy][num_clients]['error_rates'].append(metrics.get('error_rate', 1))
        
        return impact_results
    
    def generate_text_report(self, output_file=None):
        """Generate a text-based comparison report."""
        report = []
        
        report.append("=" * 100)
        report.append("COMPREHENSIVE FEDERATED LEARNING ANALYSIS - ALL STRATEGIES & CONFIGURATIONS")
        report.append("=" * 100)
        report.append("")
        report.append(f"Total experiments analyzed: {len(self.experiments)}")
        report.append("")
        
        # Get all unique combinations
        strategies = set()
        alphas = set()
        num_clients_list = set()
        
        for exp_key, exp_data in self.experiments.items():
            strategies.add(exp_data['strategy'])
            alphas.add(exp_data['alpha'])
            num_clients_list.add(exp_data['num_clients'])
        
        strategies = sorted(strategies)
        alphas = sorted(alphas)
        num_clients_list = sorted(num_clients_list)
        
        report.append(f"Strategies analyzed: {', '.join(strategies)}")
        report.append(f"Alpha values (heterogeneity): {', '.join(f'{a:.1f}' for a in alphas)}")
        report.append(f"Number of clients: {', '.join(str(num_clients_list))}")
        report.append("")
        
        # Per-configuration comparisons
        report.append("STRATEGY COMPARISONS BY CONFIGURATION")
        report.append("-" * 100)
        report.append("")
        
        for alpha in alphas:
            for num_clients in num_clients_list:
                comparison = self.compare_strategies_for_alpha_clients(alpha, num_clients)
                if comparison and comparison['pairwise_comparisons']:
                    report.append(f"Configuration: Alpha={alpha:.1f}, Clients={num_clients}")
                    report.append("  " + "-" * 96)
                    
                    for comparison_key, metrics_comp in comparison['pairwise_comparisons'].items():
                        strat1, strat2 = comparison_key.split('_vs_')
                        report.append(f"  {strat1} vs {strat2}:")
                        
                        for metric, comp_data in metrics_comp.items():
                            val1 = comp_data[strat1]
                            val2 = comp_data[strat2]
                            diff = comp_data['difference']
                            improv = comp_data['improvement_percent']
                            
                            symbol = "↑" if improv > 0 else "↓"
                            report.append(f"    {metric:12} | "
                                        f"{strat1}: {val1:7.4f} | "
                                        f"{strat2}: {val2:7.4f} | "
                                        f"Δ: {diff:+7.4f} ({improv:+7.2f}%) {symbol}")
                        
                        report.append("")
                    
                    # Per-class F1 comparison if available
                    strategy_data = comparison['strategies']
                    if len(strategy_data) >= 2:
                        strat1, strat2 = list(strategy_data.keys())[:2]  # Compare first two strategies
                        per_class1 = strategy_data[strat1].get('per_class_f1', {})
                        per_class2 = strategy_data[strat2].get('per_class_f1', {})
                        
                        if per_class1 or per_class2:
                            report.append("    Per-class F1 comparison:")
                            class_keys = sorted(
                                {int(k) for k in per_class1.keys() | per_class2.keys()}
                            )
                            for class_id in class_keys:
                                key = str(class_id)
                                val1 = per_class1.get(key, 0.0)
                                val2 = per_class2.get(key, 0.0)
                                diff = val2 - val1
                                symbol = "↑" if diff > 0 else "↓"
                                class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
                                minority = " (minority)" if class_id in MINORITY_CLASSES else ""
                                report.append(
                                    f"      class {key}: {class_name}{minority} | "
                                    f"{strat1}={val1:5.3f} "
                                    f"{strat2}={val2:5.3f} "
                                    f"(Δ={diff:+.3f}) {symbol}"
                            )
                            report.append("")
                    
                    report.append("")
        
        # Heterogeneity impact analysis
        report.append("")
        report.append("HETEROGENEITY & SCALE IMPACT ANALYSIS")
        report.append("-" * 100)
        report.append("")
        
        impact = self.analyze_heterogeneity_impact()
        
        for strategy in strategies:
            report.append(f"{strategy} Performance Analysis:")
            report.append("  " + "-" * 96)
            
            for num_clients in num_clients_list:
                if (strategy in impact and 
                    num_clients in impact[strategy] and 
                    impact[strategy][num_clients]['alphas']):
                    
                    client_data = impact[strategy][num_clients]
                    alphas_client = client_data['alphas']
                    accs = client_data['accuracies']
                    error_rates = client_data['error_rates']
                    
                    report.append(f"  {num_clients} Clients:")
                    
                    for alpha, acc, err_rate in zip(alphas_client, accs, error_rates):
                        report.append(f"    Alpha {alpha:3.1f}: Accuracy = {acc:7.4f}, Error Rate = {err_rate:7.4f}")
                    
                    # Calculate robustness metrics
                    if len(accs) > 1:
                        acc_std = np.std(accs)
                        acc_mean = np.mean(accs)
                        cv = acc_std / acc_mean if acc_mean > 0 else 0
                        report.append(f"    Robustness (accuracy std dev): {acc_std:.4f}")
                        report.append(f"    Coefficient of variation: {cv:.4f}")
                        report.append("      (Lower values = more robust across heterogeneity)")
                    
                    report.append("")
            
            report.append("")
        
        # Summary findings
        report.append("")
        report.append("SUMMARY FINDINGS")
        report.append("-" * 80)
        
        minority_names = [CLASS_NAMES[c] for c in sorted(MINORITY_CLASSES)]
        report.append(
            f"Minority classes in this dataset: {', '.join(minority_names)} "
            f"(class IDs: {', '.join(str(c) for c in sorted(MINORITY_CLASSES))})"
        )
        report.append("")
        findings = self.get_summary_findings(impact)
        for finding in findings:
            report.append(f"• {finding}")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # Print to console
        print(report_text)
        
        # Save to file
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"\nReport saved to: {output_path}")
        
        return report_text
    
    def get_summary_findings(self, impact):
        """Generate summary findings from the analysis."""
        findings = []
        
        # Get all strategies
        strategies = list(impact.keys())
        
        if len(strategies) < 2:
            findings.append("Insufficient data for comparative analysis.")
            return findings
        
        # Compare strategies across all configurations
        for num_clients in impact[strategies[0]].keys():
            for strat1, strat2 in [(strategies[i], strategies[j]) 
                                 for i in range(len(strategies)) 
                                 for j in range(i+1, len(strategies))]:
                
                if (num_clients in impact[strat1] and num_clients in impact[strat2]):
                    data1 = impact[strat1][num_clients]
                    data2 = impact[strat2][num_clients]
                    
                    if data1['accuracies'] and data2['accuracies']:
                        mean_acc1 = np.mean(data1['accuracies'])
                        mean_acc2 = np.mean(data2['accuracies'])
                        
                        if abs(mean_acc1 - mean_acc2) > 0.01:  # Significant difference
                            better = strat2 if mean_acc2 > mean_acc1 else strat1
                            worse = strat1 if mean_acc2 > mean_acc1 else strat2
                            diff = abs(mean_acc2 - mean_acc1) * 100
                            findings.append(f"{better} outperforms {worse} by {diff:.1f}% average accuracy "
                                          f"with {num_clients} clients")
                        
                        # Check robustness (lower std dev is better)
                        std1 = np.std(data1['accuracies'])
                        std2 = np.std(data2['accuracies'])
                        
                        if abs(std1 - std2) > 0.005:
                            more_robust = strat1 if std1 < std2 else strat2
                            less_robust = strat2 if std1 < std2 else strat1
                            findings.append(f"{more_robust} shows better robustness than {less_robust} "
                                          f"with {num_clients} clients")
        
        # Heterogeneity impact
        for strategy in strategies:
            for num_clients in impact[strategy].keys():
                data = impact[strategy][num_clients]
                if len(data['accuracies']) > 1:
                    # Performance at extreme heterogeneity (alpha=0.1)
                    if 0.1 in data['alphas']:
                        idx = data['alphas'].index(0.1)
                        acc_extreme = data['accuracies'][idx]
                        findings.append(f"{strategy} with {num_clients} clients achieves "
                                      f"{acc_extreme:.4f} accuracy at maximum heterogeneity (α=0.1)")
        
        minority_names = [CLASS_NAMES[c] for c in sorted(MINORITY_CLASSES)]
        findings.append(f"Minority classes in this dataset: {', '.join(minority_names)} "
                       f"(class IDs: {', '.join(str(c) for c in sorted(MINORITY_CLASSES))})")
        
        if not findings:
            findings.append("No significant differences found in current data.")
        
        return findings
    
    def create_visualizations(self, output_dir=None):
        """Create comprehensive matplotlib visualizations of results."""
        if not HAS_VISUALIZATION:
            print("Skipping visualizations: pandas/matplotlib not available")
            return
        
        if output_dir is None:
            output_dir = self.results_base.parent / "analysis_plots"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for publication-quality plots
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18
        })
        
        # Color scheme
        colors = {'FedAvg': '#1f77b4', 'FedCRA': '#ff7f0e', 'FedProx': '#2ca02c'}
        
        self._plot_accuracy_vs_heterogeneity(output_dir, colors)
        self._plot_f1_vs_heterogeneity(output_dir, colors)
        self._plot_precision_recall_vs_heterogeneity(output_dir, colors)
        self._plot_loss_vs_heterogeneity(output_dir, colors)
        self._plot_convergence_curves(output_dir, colors)
        self._plot_final_metrics_comparison(output_dir, colors)
        self._plot_per_class_f1_comparison(output_dir, colors)
        self._plot_improvement_analysis(output_dir, colors)
        self._plot_robustness_analysis(output_dir, colors)
        
        print(f"All visualizations saved to: {output_dir}")
    
    def _plot_accuracy_vs_heterogeneity(self, output_dir, colors):
        """Plot accuracy vs heterogeneity level for different numbers of clients."""
        impact = self.analyze_heterogeneity_impact()
        
        # Get all strategies and num_clients
        strategies = list(impact.keys())
        num_clients_list = sorted(list(impact[strategies[0]].keys())) if strategies else []
        
        if not strategies or not num_clients_list:
            return
        
        n_clients = len(num_clients_list)
        fig, axes = plt.subplots(1, n_clients, figsize=(6*n_clients, 6), squeeze=False)
        axes = axes.flatten()
        
        for idx, num_clients in enumerate(num_clients_list):
            ax = axes[idx]
            
            for strategy in strategies:
                if (strategy in impact and 
                    num_clients in impact[strategy] and 
                    impact[strategy][num_clients]['alphas']):
                    
                    client_data = impact[strategy][num_clients]
                    alphas = client_data['alphas']
                    accs = client_data['accuracies']
                    
                    ax.plot(alphas, accs, marker='o', label=strategy, 
                           linewidth=3, markersize=8, color=colors.get(strategy, 'blue'))
            
            ax.set_xlabel('Dirichlet Alpha (Heterogeneity)', fontsize=12)
            ax.set_ylabel('Final Test Accuracy', fontsize=12)
            ax.set_title(f'{num_clients} Clients', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11, loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_xticks([0.1, 0.3, 0.5, 1.0])
            ax.set_xticklabels(['0.1\n(High)', '0.3', '0.5', '1.0\n(Low)'])
            ax.set_ylim(0.0, 1.0)
        
        plt.suptitle('Federated Learning Performance Under Data Heterogeneity\nAccuracy vs Heterogeneity by Number of Clients', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        plot_path = output_dir / "accuracy_vs_heterogeneity_by_clients.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {plot_path}")
        plt.close()
    
    def _plot_f1_vs_heterogeneity(self, output_dir, colors):
        """Plot F1 score vs heterogeneity level for different numbers of clients."""
        impact = self.analyze_heterogeneity_impact()
        
        # Get all strategies and num_clients
        strategies = list(impact.keys())
        num_clients_list = sorted(list(impact[strategies[0]].keys())) if strategies else []
        
        if not strategies or not num_clients_list:
            return
        
        n_clients = len(num_clients_list)
        fig, axes = plt.subplots(1, n_clients, figsize=(6*n_clients, 6), squeeze=False)
        axes = axes.flatten()
        
        for idx, num_clients in enumerate(num_clients_list):
            ax = axes[idx]
            
            for strategy in strategies:
                if (strategy in impact and 
                    num_clients in impact[strategy] and 
                    impact[strategy][num_clients]['alphas']):
                    
                    client_data = impact[strategy][num_clients]
                    alphas = client_data['alphas']
                    f1s = client_data['f1_scores']
                    
                    ax.plot(alphas, f1s, marker='s', label=strategy, 
                           linewidth=3, markersize=8, color=colors.get(strategy, 'blue'))
            
            ax.set_xlabel('Dirichlet Alpha (Heterogeneity)', fontsize=12)
            ax.set_ylabel('Final Macro F1-Score', fontsize=12)
            ax.set_title(f'{num_clients} Clients', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11, loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_xticks([0.1, 0.3, 0.5, 1.0])
            ax.set_xticklabels(['0.1\n(High)', '0.3', '0.5', '1.0\n(Low)'])
            ax.set_ylim(0.0, 1.0)
        
        plt.suptitle('F1-Score Performance Under Data Heterogeneity\nby Number of Clients', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        plot_path = output_dir / "f1_vs_heterogeneity_by_clients.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {plot_path}")
        plt.close()
    
    def _plot_precision_recall_vs_heterogeneity(self, output_dir, colors):
        """Plot precision, recall, and error rate vs heterogeneity level."""
        alphas = [0.1, 0.3, 0.5, 1.0]
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))

        for idx, metric in enumerate(['precision', 'recall', 'error_rate']):
            ax = axes[idx]
            for strategy in ['FedAvg', 'FedCRA', 'FedProx']:
                values = []
                for alpha in alphas:
                    metrics = self.get_final_metrics(strategy, alpha)
                    if metrics:
                        values.append(metrics.get(metric, 0))
                    else:
                        values.append(0)
                if values:
                    ax.plot(alphas, values, marker='o', label=strategy,
                            linewidth=3, markersize=8, color=colors.get(strategy, '#000000'))

            ax.set_xlabel('Dirichlet Alpha (Heterogeneity Level)', fontsize=14)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=14)
            ax.set_title(f'{metric.replace("_", " ").title()} vs Heterogeneity', fontsize=16, fontweight='bold')
            ax.legend(fontsize=12, loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_xticks([0.1, 0.3, 0.5, 1.0])
            ax.set_xticklabels(['0.1\n(High)', '0.3', '0.5', '1.0\n(Low)'])
            if metric != 'error_rate':
                ax.set_ylim(0.0, 1.0)
            else:
                ax.set_ylim(0.0, 1.0)

        plt.tight_layout()
        plot_path = output_dir / "precision_recall_error_rate_vs_heterogeneity.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {plot_path}")
        plt.close()
    
    def _plot_loss_vs_heterogeneity(self, output_dir, colors):
        """Plot final loss and error rate vs heterogeneity level."""
        alphas = [0.1, 0.3, 0.5, 1.0]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        for strategy in ['FedAvg', 'FedCRA', 'FedProx']:
            losses = []
            error_rates = []
            for alpha in alphas:
                metrics = self.get_final_metrics(strategy, alpha)
                if metrics:
                    losses.append(metrics.get('loss', 0))
                    error_rates.append(metrics.get('error_rate', 0))
                else:
                    losses.append(0)
                    error_rates.append(0)
            if losses:
                ax1.plot(alphas, losses, marker='d', label=strategy,
                         linewidth=3, markersize=8, color=colors.get(strategy, '#000000'))
            if error_rates:
                ax2.plot(alphas, error_rates, marker='x', label=strategy,
                         linewidth=3, markersize=8, color=colors.get(strategy, '#000000'))

        ax1.set_xlabel('Dirichlet Alpha (Heterogeneity Level)', fontsize=14)
        ax1.set_ylabel('Final Training Loss', fontsize=14)
        ax1.set_title('Training Loss vs Heterogeneity', fontsize=16, fontweight='bold')
        ax1.set_xscale('log')
        ax1.set_xticks([0.1, 0.3, 0.5, 1.0])
        ax1.set_xticklabels(['0.1\n(High)', '0.3', '0.5', '1.0\n(Low)'])
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=12)

        ax2.set_xlabel('Dirichlet Alpha (Heterogeneity Level)', fontsize=14)
        ax2.set_ylabel('Error Rate', fontsize=14)
        ax2.set_title('Error Rate vs Heterogeneity', fontsize=16, fontweight='bold')
        ax2.set_xscale('log')
        ax2.set_xticks([0.1, 0.3, 0.5, 1.0])
        ax2.set_xticklabels(['0.1\n(High)', '0.3', '0.5', '1.0\n(Low)'])
        ax2.set_ylim(0.0, 1.0)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=12)

        plt.tight_layout()
        plot_path = output_dir / "loss_error_rate_vs_heterogeneity.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {plot_path}")
        plt.close()
    
    def _plot_convergence_curves(self, output_dir, colors):
        """Plot convergence curves for accuracy and error rate over training rounds."""
        alphas_to_plot = [0.1, 0.5, 1.0]
        strategies = ['FedAvg', 'FedCRA', 'FedProx']

        for alpha in alphas_to_plot:
            num_clients_list = sorted({exp['num_clients'] for exp in self.experiments.values() if exp['alpha'] == alpha})
            if not num_clients_list:
                continue

            for num_clients in num_clients_list:
                fig, ax1 = plt.subplots(figsize=(14, 7))
                ax2 = ax1.twinx()
                rounds = None
                
                for strategy in strategies:
                    entries = self.get_experiment_entries(strategy=strategy, alpha=alpha, num_clients=num_clients)
                    if not entries:
                        continue
                    metrics_list = entries[0]['metrics']
                    if not metrics_list:
                        continue
                    rounds = [m.get('round', i) for i, m in enumerate(metrics_list)]
                    accuracies = [m.get('accuracy', 0) for m in metrics_list]
                    error_rates = [m.get('error_rate', 0) for m in metrics_list]
                    
                    ax1.plot(rounds, accuracies, label=f'{strategy} Accuracy',
                             linewidth=2, color=colors.get(strategy, '#000000'))
                    ax2.plot(rounds, error_rates, linestyle='--', label=f'{strategy} Error Rate',
                             linewidth=2, color=colors.get(strategy, '#000000'))

                if rounds is None:
                    continue

                ax1.set_xlabel('Training Round', fontsize=14)
                ax1.set_ylabel('Accuracy', fontsize=14)
                ax2.set_ylabel('Error Rate', fontsize=14)
                ax1.set_title(f'Convergence Curves (α={alpha}, {num_clients} Clients)', fontsize=16, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                ax1.set_xlim(0, max(rounds))
                ax1.set_ylim(0.0, 1.0)
                ax2.set_ylim(0.0, 1.0)

                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='lower right')

                plt.tight_layout()
                plot_path = output_dir / f"convergence_alpha_{alpha}_clients_{num_clients}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved: {plot_path}")
                plt.close()
    
    def _plot_final_metrics_comparison(self, output_dir, colors):
        """Bar chart comparing final metrics across all alphas."""
        alphas = [0.1, 0.3, 0.5, 1.0]
        metrics_to_plot = ['accuracy', 'f1_score', 'precision', 'recall', 'error_rate']
        metric_names = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'Error Rate']
        
        fig, axes = plt.subplots(3, 2, figsize=(18, 15))
        axes = axes.flatten()
        
        for i, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
            ax = axes[i]
            x = np.arange(len(alphas))
            width = 0.25
            
            fedavg_vals = []
            fedcra_vals = []
            fedprox_vals = []
            
            for alpha in alphas:
                fedavg_metrics = self.get_final_metrics("FedAvg", alpha)
                fedcra_metrics = self.get_final_metrics("FedCRA", alpha)
                fedprox_metrics = self.get_final_metrics("FedProx", alpha)
                
                fedavg_vals.append(fedavg_metrics.get(metric, 0) if fedavg_metrics else 0)
                fedcra_vals.append(fedcra_metrics.get(metric, 0) if fedcra_metrics else 0)
                fedprox_vals.append(fedprox_metrics.get(metric, 0) if fedprox_metrics else 0)
            
            ax.bar(x - width, fedavg_vals, width, label='FedAvg', color=colors['FedAvg'], alpha=0.8)
            ax.bar(x, fedcra_vals, width, label='FedCRA', color=colors['FedCRA'], alpha=0.8)
            ax.bar(x + width, fedprox_vals, width, label='FedProx', color=colors['FedProx'], alpha=0.8)
            
            ax.set_xlabel('Dirichlet Alpha', fontsize=14)
            ax.set_ylabel(name, fontsize=14)
            ax.set_title(f'{name} Comparison', fontsize=16, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([f'{a}' for a in alphas])
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            if metric != 'error_rate':
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(0, 1)
        
        # Remove unused subplot if present
        if len(axes) > len(metrics_to_plot):
            axes[-1].axis('off')
        
        plt.tight_layout()
        plot_path = output_dir / "final_metrics_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {plot_path}")
        plt.close()
    
    def _plot_per_class_f1_comparison(self, output_dir, colors):
        """Plot per-class F1 scores for each alpha."""
        alphas = [0.1, 0.3, 0.5, 1.0]
        classes = sorted(CLASS_NAMES.keys())
        
        for alpha in alphas:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x = np.arange(len(classes))
            width = 0.35
            
            fedavg_f1 = []
            fedcra_f1 = []
            
            fedavg_metrics = self.get_final_metrics("FedAvg", alpha)
            fedcra_metrics = self.get_final_metrics("FedCRA", alpha)
            
            if fedavg_metrics and fedcra_metrics:
                fedavg_per_class = fedavg_metrics.get('per_class_f1', {})
                fedcra_per_class = fedcra_metrics.get('per_class_f1', {})
                
                for class_id in classes:
                    fedavg_f1.append(fedavg_per_class.get(str(class_id), 0))
                    fedcra_f1.append(fedcra_per_class.get(str(class_id), 0))
                
                ax.bar(x - width/2, fedavg_f1, width, label='FedAvg', 
                      color=colors['FedAvg'], alpha=0.8)
                ax.bar(x + width/2, fedcra_f1, width, label='FedCRA', 
                      color=colors['FedCRA'], alpha=0.8)
                
                # Highlight minority classes
                for i, class_id in enumerate(classes):
                    if class_id in MINORITY_CLASSES:
                        ax.text(i - width/2, fedavg_f1[i] + 0.02, '*', 
                              ha='center', va='bottom', fontsize=16, color='red')
                        ax.text(i + width/2, fedcra_f1[i] + 0.02, '*', 
                              ha='center', va='bottom', fontsize=16, color='red')
            
            ax.set_xlabel('Traffic Class', fontsize=14)
            ax.set_ylabel('F1-Score', fontsize=14)
            ax.set_title(f'Per-Class F1-Score Comparison (α={alpha})\nFedCRA Better Minority Class Handling', 
                        fontsize=16, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([CLASS_NAMES[c] for c in classes])
            ax.legend(fontsize=13)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1)
            
            # Add note about minority classes
            error_annotation = ''
            if fedavg_metrics and fedcra_metrics:
                avg_error_fedavg = fedavg_metrics.get('error_rate')
                avg_error_fedcra = fedcra_metrics.get('error_rate')
                if avg_error_fedavg is not None and avg_error_fedcra is not None:
                    error_annotation = (
                        f'FedAvg Error Rate: {avg_error_fedavg:.3f} | '
                        f'FedCRA Error Rate: {avg_error_fedcra:.3f}'
                    )

            ax.text(0.02, 0.98, '* Minority classes', transform=ax.transAxes, 
                   fontsize=12, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            if error_annotation:
                ax.text(0.02, 0.90, error_annotation, transform=ax.transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            plt.tight_layout()
            plot_path = output_dir / f"per_class_f1_alpha_{alpha}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {plot_path}")
            plt.close()
    
    def _plot_improvement_analysis(self, output_dir, colors):
        """Plot improvement percentages for each metric across alphas."""
        alphas = [0.1, 0.3, 0.5, 1, 5]
        metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'error_rate']
        metric_names = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'Error Rate']
        
        fig, axes = plt.subplots(3, 2, figsize=(18, 15))
        axes = axes.flatten()
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i]
            
            improvements = []
            for alpha in alphas:
                comp = self.compare_strategies_for_alpha(alpha)
                if comp and metric in comp['pairwise_comparisons'].get('FedAvg_vs_FedCRA', {}):
                    improvements.append(comp['pairwise_comparisons']['FedAvg_vs_FedCRA'][metric]['improvement_percent'])
                else:
                    improvements.append(0)
            
            colors_bar = ['green' if x > 0 else 'red' for x in improvements]
            bars = ax.bar([str(a) for a in alphas], improvements, 
                         color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1)
            
            ax.set_xlabel('Dirichlet Alpha', fontsize=14)
            ax.set_ylabel('FedCRA Improvement (%)', fontsize=14)
            ax.set_title(f'{name} Improvement: FedCRA vs FedAvg', fontsize=16, fontweight='bold')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, imp in zip(bars, improvements):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., 
                       height + (1 if height >= 0 else -3),
                       f'{imp:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                       fontsize=11, fontweight='bold')
        
        if len(axes) > len(metrics):
            axes[-1].axis('off')
        
        plt.tight_layout()
        plot_path = output_dir / "improvement_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {plot_path}")
        plt.close()
    
    def _plot_robustness_analysis(self, output_dir, colors):
        """Plot robustness analysis showing variance across heterogeneity levels."""
        impact = self.analyze_heterogeneity_impact()
        
        strategies = sorted(list(impact.keys()))
        robustness_data = {}
        
        for strategy in strategies:
            if strategy in impact and impact[strategy]:
                accuracy_stds = []
                error_rate_stds = []
                means = []
                error_means = []

                for num_clients, client_data in impact[strategy].items():
                    if client_data['accuracies']:
                        accuracy_stds.append(np.std(client_data['accuracies']))
                        means.append(np.mean(client_data['accuracies']))
                    if client_data['error_rates']:
                        error_rate_stds.append(np.std(client_data['error_rates']))
                        error_means.append(np.mean(client_data['error_rates']))

                if accuracy_stds and error_rate_stds:
                    robustness_data[strategy] = {
                        'accuracy_std': float(np.mean(accuracy_stds)),
                        'error_rate_std': float(np.mean(error_rate_stds)),
                        'mean_accuracy': float(np.mean(means)),
                        'mean_error_rate': float(np.mean(error_means))
                    }
        
        if not robustness_data:
            return

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7))
        
        ax1.bar(strategies, [robustness_data[s]['accuracy_std'] for s in strategies],
                color=[colors.get(s, '#000000') for s in strategies], alpha=0.7,
                edgecolor='black', linewidth=1)
        ax1.set_ylabel('Std Dev of Accuracy', fontsize=14)
        ax1.set_title('Accuracy Robustness', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        for i, strategy in enumerate(strategies):
            ax1.text(i, robustness_data[strategy]['accuracy_std'] + 0.0005,
                     f"{robustness_data[strategy]['accuracy_std']:.4f}",
                     ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax2.bar(strategies, [robustness_data[s]['error_rate_std'] for s in strategies],
                color=[colors.get(s, '#000000') for s in strategies], alpha=0.7,
                edgecolor='black', linewidth=1)
        ax2.set_ylabel('Std Dev of Error Rate', fontsize=14)
        ax2.set_title('Error Rate Robustness', fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        for i, strategy in enumerate(strategies):
            ax2.text(i, robustness_data[strategy]['error_rate_std'] + 0.0005,
                     f"{robustness_data[strategy]['error_rate_std']:.4f}",
                     ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax3.scatter([robustness_data[s]['error_rate_std'] for s in strategies],
                    [robustness_data[s]['mean_accuracy'] for s in strategies],
                    s=200, color=[colors.get(s, '#000000') for s in strategies],
                    edgecolor='black', linewidth=2)
        for i, strategy in enumerate(strategies):
            ax3.annotate(strategy,
                         (robustness_data[strategy]['error_rate_std'], robustness_data[strategy]['mean_accuracy']),
                         xytext=(10, 10), textcoords='offset points',
                         fontsize=14, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax3.set_xlabel('Std Dev of Error Rate', fontsize=14)
        ax3.set_ylabel('Mean Accuracy', fontsize=14)
        ax3.set_title('Accuracy vs Error-Rate Robustness', fontsize=16, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(left=0)

        plt.tight_layout()
        plot_path = output_dir / "robustness_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {plot_path}")
        plt.close()


def main():
    """Main analysis function."""
    import sys
    
    # Get project path
    project_path = Path(__file__).parent
    results_path = project_path / "dataset" / "models" / "iomt_traffic" / "Category"
    
    print("Analyzing Heterogeneity & Class Imbalance Experiments")
    print(f"Results directory: {results_path}")
    print("")
    
    analyzer = HeterogeneityAnalyzer(results_path)
    analyzer.load_all_experiments()
    
    if not analyzer.experiments:
        print("No experiments found. Please run: bash run_heterogeneity_experiments.sh")
        sys.exit(1)
    
    print(f"Loaded {len(analyzer.experiments)} experiments")
    print("")
    
    # Generate report
    report_path = project_path / "heterogeneity_analysis_report.txt"
    analyzer.generate_text_report(output_file=report_path)
    
    # Create visualizations
    try:
        viz_dir = project_path / "analysis_plots"
        analyzer.create_visualizations(output_dir=viz_dir)
        print(f"Visualizations saved to: {viz_dir}")
    except Exception as e:
        print(f"Could not create visualizations: {e}")
    
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
