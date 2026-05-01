#!/usr/bin/env python3
"""
FedCRA vs FedAvg Comparative Analysis
Generates comprehensive comparison plots showing FedCRA vs FedAvg
across different heterogeneity levels.
"""

import json
from pathlib import Path
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


class FedCRAFedAvgComparator:
    """Compares FedCRA and FedAvg strategies across heterogeneous conditions."""

    def __init__(self, results_base_path):
        self.results_base = Path(results_base_path)
        self.experiments = {}

    def load_metrics(self, strategy_name, alpha):
        """Load metrics for a specific strategy and alpha value."""
        alpha_str = f"{alpha:.1f}" if alpha == 1.0 else str(alpha)

        metrics_path = (self.results_base / strategy_name /
                       f"num_clients_5" / f"dirichlet_alpha_{alpha_str}" /
                       strategy_name / "DNN" / "metrics" /
                       "server_metrics.json")

        if not metrics_path.exists():
            print(f"Warning: No data for {strategy_name} alpha {alpha} in num_clients_5")
            return None

        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return metrics if metrics else None
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {metrics_path}: {e}")
            return None

    def load_all_experiments(self, strategies=None, alphas=None):
        """Load all experiment metrics."""
        if strategies is None:
            strategies = ["FedCRA", "FedAvg"]
        if alphas is None:
            alphas = [0.1, 0.3, 0.5]  # Only alphas available for both in num_clients_5

        for strategy in strategies:
            for alpha in alphas:
                key = f"{strategy}_alpha_{alpha}"
                metrics = self.load_metrics(strategy, alpha)
                if metrics:
                    self.experiments[key] = {
                        'strategy': strategy,
                        'alpha': alpha,
                        'metrics': metrics
                    }
                else:
                    print(f"⚠ No metrics found for {key}")

    def get_final_metrics(self, strategy, alpha):
        """Get final round metrics for a strategy and alpha."""
        key = f"{strategy}_alpha_{alpha}"
        if key not in self.experiments:
            return None

        metrics = self.experiments[key]['metrics']
        if not metrics:
            return None

        return metrics[-1]

    def analyze_heterogeneity_impact(self):
        """Analyze how heterogeneity affects each strategy."""
        impact_results = {'FedCRA': {}, 'FedAvg': {}}

        for strategy in ['FedCRA', 'FedAvg']:
            accuracies = []
            f1_scores = []
            alphas = []

            for alpha in [0.1, 0.3, 0.5, 1.0]:
                metrics = self.get_final_metrics(strategy, alpha)
                if metrics:
                    alphas.append(alpha)
                    accuracies.append(metrics.get('accuracy', 0))
                    f1_scores.append(metrics.get('f1_score', 0))

            if alphas:
                impact_results[strategy]['alphas'] = alphas
                impact_results[strategy]['accuracies'] = accuracies
                impact_results[strategy]['f1_scores'] = f1_scores

        return impact_results

    def generate_text_report(self, output_file=None):
        """Generate a text-based comparison report."""
        report = []

        report.append("=" * 80)
        report.append("FEDCRA vs FEDAVG - COMPREHENSIVE COMPARISON ANALYSIS")
        report.append("=" * 80)
        report.append("")

        report.append("PER-ALPHA COMPARATIVE ANALYSIS")
        report.append("-" * 80)
        report.append("")

        for alpha in [0.1, 0.3, 0.5, 1.0]:
            fedcra_metrics = self.get_final_metrics("FedCRA", alpha)
            fedavg_metrics = self.get_final_metrics("FedAvg", alpha)

            if fedcra_metrics and fedavg_metrics:
                report.append(f"Dirichlet Alpha = {alpha}")
                report.append("  " + "-" * 76)

                for metric in ['accuracy', 'f1_score', 'precision', 'recall']:
                    cra_val = fedcra_metrics.get(metric, 0)
                    avg_val = fedavg_metrics.get(metric, 0)

                    if isinstance(cra_val, (int, float)):
                        improvement = ((cra_val - avg_val) / (abs(avg_val) + 1e-10) * 100)
                        symbol = "↑" if improvement > 0 else "↓"
                        report.append(f"  {metric:12} | "
                                    f"FedAvg: {avg_val:7.4f} | "
                                    f"FedCRA: {cra_val:7.4f} | "
                                    f"FedCRA Improvement: {improvement:+7.2f}% {symbol}")

                fedcra_per_class = fedcra_metrics.get('per_class_f1', {})
                fedavg_per_class = fedavg_metrics.get('per_class_f1', {})

                if fedcra_per_class or fedavg_per_class:
                    report.append("  Per-class F1 Comparison:")
                    class_keys = sorted(
                        {int(k) for k in fedcra_per_class.keys() | fedavg_per_class.keys()}
                    )
                    for class_id in class_keys:
                        key = str(class_id)
                        fedavg_val = fedavg_per_class.get(key, 0.0)
                        fedcra_val = fedcra_per_class.get(key, 0.0)
                        diff = fedcra_val - fedavg_val
                        symbol = "↑" if diff > 0 else "↓"
                        class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
                        minority = " (minority)" if class_id in MINORITY_CLASSES else ""
                        report.append(
                            f"    class {key}: {class_name}{minority} | "
                            f"FedAvg={fedavg_val:5.3f} "
                            f"FedCRA={fedcra_val:5.3f} "
                            f"(Δ={diff:+.3f}) {symbol}"
                        )

                report.append("")

        report.append("")
        report.append("ROBUSTNESS ACROSS HETEROGENEITY LEVELS")
        report.append("-" * 80)
        report.append("")

        impact = self.analyze_heterogeneity_impact()

        for strategy in ['FedAvg', 'FedCRA']:
            report.append(f"{strategy} Performance across Heterogeneity Levels:")
            if strategy in impact and impact[strategy]:
                alphas = impact[strategy].get('alphas', [])
                accs = impact[strategy].get('accuracies', [])

                for alpha, acc in zip(alphas, accs):
                    report.append(f"  Alpha {alpha:3.1f}: Accuracy = {acc:7.4f}")

                if len(accs) > 1:
                    robustness = np.std(accs)
                    report.append(f"  Robustness (std dev): {robustness:.4f}")
                    report.append(f"    (Lower = more robust across heterogeneity)")

            report.append("")

        report.append("")
        report.append("SUMMARY FINDINGS")
        report.append("-" * 80)

        minority_names = [CLASS_NAMES[c] for c in sorted(MINORITY_CLASSES)]
        report.append(
            f"Minority classes: {', '.join(minority_names)} "
            f"(class IDs: {', '.join(str(c) for c in sorted(MINORITY_CLASSES))})"
        )
        report.append("")
        findings = self.get_summary_findings(impact)
        for finding in findings:
            report.append(f"• {finding}")

        report.append("")
        report.append("=" * 80)

        report_text = "\n".join(report)

        print(report_text)

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

        if impact['FedCRA'] and impact['FedAvg']:
            fedcra_accs = impact['FedCRA'].get('accuracies', [])
            fedavg_accs = impact['FedAvg'].get('accuracies', [])

            if fedcra_accs and fedavg_accs:
                fedcra_mean = np.mean(fedcra_accs)
                fedavg_mean = np.mean(fedavg_accs)

                if fedcra_mean > fedavg_mean:
                    diff = ((fedcra_mean - fedavg_mean) / fedavg_mean * 100)
                    findings.append(f"FedCRA achieves {diff:.2f}% HIGHER average accuracy")

                fedcra_var = np.var(fedcra_accs)
                fedavg_var = np.var(fedavg_accs)

                if fedcra_var < fedavg_var:
                    findings.append("FedCRA shows BETTER ROBUSTNESS to data heterogeneity")
                elif fedcra_var > fedavg_var:
                    findings.append("FedAvg exhibits better robustness to heterogeneity")

                if len(fedcra_accs) > 0 and len(fedavg_accs) > 0:
                    diff_extreme = ((fedcra_accs[0] - fedavg_accs[0]) /
                                   (abs(fedavg_accs[0]) + 1e-10) * 100)
                    findings.append(f"At maximum heterogeneity (α=0.1), FedCRA is "
                                  f"{diff_extreme:+.2f}% vs FedAvg")

        if not findings:
            findings.append("Analysis complete. See detailed metrics above.")

        return findings

    def create_visualizations(self, output_dir=None):
        """Create comprehensive matplotlib visualizations comparing FedCRA vs FedAvg."""
        if not HAS_VISUALIZATION:
            print("Skipping visualizations: pandas/matplotlib not available")
            return

        if output_dir is None:
            output_dir = self.results_base.parent / "fedcra_vs_fedavg_plots"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

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

        colors = {'FedCRA': '#ff7f0e', 'FedAvg': '#1f77b4'}

        self._plot_accuracy_comparison(output_dir, colors)
        self._plot_f1_comparison(output_dir, colors)
        self._plot_precision_recall_comparison(output_dir, colors)
        self._plot_loss_comparison(output_dir, colors)
        self._plot_convergence_curves(output_dir, colors)
        self._plot_final_metrics_bars(output_dir, colors)
        self._plot_per_class_f1_comparison(output_dir, colors)
        self._plot_improvement_bars(output_dir, colors)
        self._plot_robustness_comparison(output_dir, colors)
        self._plot_per_class_f1_comparison(output_dir, colors)
        self._plot_improvement_bars(output_dir, colors)
        self._plot_robustness_comparison(output_dir, colors)

        print(f"All visualizations saved to: {output_dir}")

    def _plot_accuracy_comparison(self, output_dir, colors):
        impact = self.analyze_heterogeneity_impact()

        plt.figure(figsize=(12, 8))

        for strategy in ['FedAvg', 'FedCRA']:
            if strategy in impact and impact[strategy]:
                alphas = impact[strategy].get('alphas', [])
                accs = impact[strategy].get('accuracies', [])
                if alphas and accs:
                    plt.plot(alphas, accs, marker='o', label=strategy,
                           linewidth=3, markersize=10, color=colors[strategy])

        plt.xlabel('Dirichlet Alpha (Heterogeneity Level)', fontsize=14)
        plt.ylabel('Final Test Accuracy', fontsize=14)
        plt.title('FedCRA vs FedAvg: Accuracy Comparison\nPerformance Under Data Heterogeneity',
                 fontsize=16, fontweight='bold')
        plt.legend(fontsize=13, loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.xticks([0.1, 0.3, 0.5], ['0.1\n(High)', '0.3', '0.5\n(Low)'])
        plt.ylim(0.7, 1.0)

        plt.tight_layout()
        plot_path = output_dir / "01_accuracy_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
        print(f"Plot saved: {plot_path}")
        plt.close()

    def _plot_f1_comparison(self, output_dir, colors):
        impact = self.analyze_heterogeneity_impact()

        plt.figure(figsize=(12, 8))

        for strategy in ['FedAvg', 'FedCRA']:
            if strategy in impact and impact[strategy]:
                alphas = impact[strategy].get('alphas', [])
                f1s = impact[strategy].get('f1_scores', [])
                if alphas and f1s:
                    plt.plot(alphas, f1s, marker='s', label=strategy,
                           linewidth=3, markersize=10, color=colors[strategy])

        plt.xlabel('Dirichlet Alpha (Heterogeneity Level)', fontsize=14)
        plt.ylabel('Final Macro F1-Score', fontsize=14)
        plt.title('FedCRA vs FedAvg: F1-Score Comparison\nImbalanced Classification Performance',
                 fontsize=16, fontweight='bold')
        plt.legend(fontsize=13, loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.xticks([0.1, 0.3, 0.5], ['0.1\n(High)', '0.3', '0.5\n(Low)'])
        plt.ylim(0.3, 0.8)

        plt.tight_layout()
        plot_path = output_dir / "02_f1_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
        print(f"Plot saved: {plot_path}")
        plt.close()

    def _plot_precision_recall_comparison(self, output_dir, colors):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        alphas = [0.1, 0.3, 0.5]

        for strategy in ['FedAvg', 'FedCRA']:
            precisions = []
            recalls = []

            for alpha in alphas:
                metrics = self.get_final_metrics(strategy, alpha)
                if metrics:
                    precisions.append(metrics.get('precision', 0))
                    recalls.append(metrics.get('recall', 0))
                else:
                    precisions.append(0)
                    recalls.append(0)

            if precisions and recalls:
                ax1.plot(alphas, precisions, marker='^', label=strategy,
                        linewidth=3, markersize=10, color=colors[strategy])
                ax2.plot(alphas, recalls, marker='v', label=strategy,
                        linewidth=3, markersize=10, color=colors[strategy])

        for ax, title, ylabel in [(ax1, 'Precision vs Heterogeneity', 'Precision'),
                                 (ax2, 'Recall vs Heterogeneity', 'Recall')]:
            ax.set_xlabel('Dirichlet Alpha (Heterogeneity Level)', fontsize=14)
            ax.set_ylabel(ylabel, fontsize=14)
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.legend(fontsize=13, loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_xticks([0.1, 0.3, 0.5])
            ax.set_xticklabels(['0.1\n(High)', '0.3', '0.5\n(Low)'])
            ax.set_ylim(0.3, 0.8)

        plt.tight_layout()
        plt.tight_layout()
        plot_path = output_dir / "03_precision_recall_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
        print(f"Plot saved: {plot_path}")
        plt.close()

    def _plot_loss_comparison(self, output_dir, colors):
        plt.figure(figsize=(12, 8))

        alphas = [0.1, 0.3, 0.5]

        for strategy in ['FedAvg', 'FedCRA']:
            losses = []

            for alpha in alphas:
                metrics = self.get_final_metrics(strategy, alpha)
                if metrics:
                    losses.append(metrics.get('loss', 0))
                else:
                    losses.append(0)

            if losses:
                plt.plot(alphas, losses, marker='d', label=strategy,
                        linewidth=3, markersize=10, color=colors[strategy])

        plt.xlabel('Dirichlet Alpha (Heterogeneity Level)', fontsize=14)
        plt.ylabel('Final Training Loss', fontsize=14)
        plt.title('FedCRA vs FedAvg: Training Loss Comparison\nLower Loss = Better Optimization',
                 fontsize=16, fontweight='bold')
        plt.legend(fontsize=13, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.xticks([0.1, 0.3, 0.5], ['0.1\n(High)', '0.3', '0.5\n(Low)'])
        plt.yscale('log')

        plt.tight_layout()
        plot_path = output_dir / "04_loss_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
        print(f"Plot saved: {plot_path}")
        plt.close()

    def _plot_convergence_curves(self, output_dir, colors):
        alphas_to_plot = [0.1, 0.3, 0.5]

        for alpha in alphas_to_plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

            for strategy in ['FedAvg', 'FedCRA']:
                key = f"{strategy}_alpha_{alpha}"
                if key in self.experiments:
                    metrics_list = self.experiments[key]['metrics']
                    if metrics_list:
                        rounds = [m.get('round', 0) for m in metrics_list]
                        accuracies = [m.get('accuracy', 0) for m in metrics_list]
                        f1_scores = [m.get('f1_score', 0) for m in metrics_list]

                        ax1.plot(rounds, accuracies, label=strategy,
                               linewidth=2.5, color=colors[strategy])
                        ax2.plot(rounds, f1_scores, label=strategy,
                               linewidth=2.5, color=colors[strategy])

            for ax, title, ylabel in [(ax1, f'Accuracy Convergence (α={alpha})', 'Accuracy'),
                                     (ax2, f'F1-Score Convergence (α={alpha})', 'F1-Score')]:
                ax.set_xlabel('Training Round', fontsize=14)
                ax.set_ylabel(ylabel, fontsize=14)
                ax.set_title(title, fontsize=16, fontweight='bold')
                ax.legend(fontsize=13)
                ax.grid(True, alpha=0.3)
                if rounds:
                    ax.set_xlim(0, max(rounds))

            plt.tight_layout()
            plot_path = output_dir / f"05_convergence_alpha_{alpha}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
            print(f"Plot saved: {plot_path}")
            plt.close()

    def _plot_final_metrics_bars(self, output_dir, colors):
        alphas = [0.1, 0.3, 0.5]
        metrics_to_plot = ['accuracy', 'f1_score', 'precision', 'recall']
        metric_names = ['Accuracy', 'F1-Score', 'Precision', 'Recall']

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for i, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
            ax = axes[i]
            x = np.arange(len(alphas))
            width = 0.35

            fedavg_vals = []
            fedcra_vals = []

            for alpha in alphas:
                fedavg_metrics = self.get_final_metrics("FedAvg", alpha)
                fedcra_metrics = self.get_final_metrics("FedCRA", alpha)

                fedavg_vals.append(fedavg_metrics.get(metric, 0) if fedavg_metrics else 0)
                fedcra_vals.append(fedcra_metrics.get(metric, 0) if fedcra_metrics else 0)

            ax.bar(x - width/2, fedavg_vals, width, label='FedAvg',
                  color=colors['FedAvg'], alpha=0.8)
            ax.bar(x + width/2, fedcra_vals, width, label='FedCRA',
                  color=colors['FedCRA'], alpha=0.8)

            ax.set_xlabel('Dirichlet Alpha', fontsize=14)
            ax.set_ylabel(name, fontsize=14)
            ax.set_title(f'{name} Comparison: FedCRA vs FedAvg', fontsize=16, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([f'{a}' for a in alphas])
            ax.legend(fontsize=13)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1)

        plt.tight_layout()
        plot_path = output_dir / "06_final_metrics_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
        print(f"Plot saved: {plot_path}")
        plt.close()

    def _plot_per_class_f1_comparison(self, output_dir, colors):
        alphas = [0.1, 0.3, 0.5]
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

                for i, class_id in enumerate(classes):
                    if class_id in MINORITY_CLASSES:
                        ax.text(i - width/2, fedavg_f1[i] + 0.02, '*',
                              ha='center', va='bottom', fontsize=16, color='red')
                        ax.text(i + width/2, fedcra_f1[i] + 0.02, '*',
                              ha='center', va='bottom', fontsize=16, color='red')

            ax.set_xlabel('Traffic Class', fontsize=14)
            ax.set_ylabel('F1-Score', fontsize=14)
            ax.set_title(f'Per-Class F1-Score Comparison (α={alpha})\nFedCRA Minority Class Handling',
                        fontsize=16, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([CLASS_NAMES[c] for c in classes])
            ax.legend(fontsize=13)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1)

            ax.text(0.02, 0.98, '* Minority classes', transform=ax.transAxes,
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()
            plot_path = output_dir / f"07_per_class_f1_alpha_{alpha}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
            print(f"Plot saved: {plot_path}")
            plt.close()

    def _plot_improvement_bars(self, output_dir, colors):
        alphas = [0.1, 0.3, 0.5]
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        metric_names = ['Accuracy', 'F1-Score', 'Precision', 'Recall']

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i]

            improvements = []
            for alpha in alphas:
                fedavg_metrics = self.get_final_metrics("FedAvg", alpha)
                fedcra_metrics = self.get_final_metrics("FedCRA", alpha)

                if fedavg_metrics and fedcra_metrics:
                    avg_val = fedavg_metrics.get(metric, 0)
                    cra_val = fedcra_metrics.get(metric, 0)
                    improvement = ((cra_val - avg_val) / (abs(avg_val) + 1e-10) * 100)
                    improvements.append(improvement)
                else:
                    improvements.append(0)

            colors_bar = ['#ff7f0e' if x > 0 else '#1f77b4' for x in improvements]
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

        plt.tight_layout()
        plot_path = output_dir / "08_improvement_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
        print(f"Plot saved: {plot_path}")
        plt.close()

    def _plot_robustness_comparison(self, output_dir, colors):
        impact = self.analyze_heterogeneity_impact()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        strategies = ['FedAvg', 'FedCRA']
        robustness_data = {}

        for strategy in strategies:
            if strategy in impact:
                impact_data = impact.get(strategy, {})
                accs = impact_data.get('accuracies', [])
                if len(accs) > 1:
                    mean_acc = np.mean(accs)
                    std_acc = np.std(accs)
                    robustness_data[strategy] = {'mean': mean_acc, 'std': std_acc, 'accs': accs}

        if len(robustness_data) >= 2:
            strategies_present = list(robustness_data.keys())

            stds = [robustness_data[s]['std'] for s in strategies_present]
            ax1.bar(strategies_present, stds,
                   color=[colors[s] for s in strategies_present], alpha=0.7,
                   edgecolor='black', linewidth=1)
            ax1.set_ylabel('Standard Deviation of Accuracy', fontsize=14)
            ax1.set_title('Robustness to Heterogeneity\n(Lower = More Robust)',
                         fontsize=16, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')

            for i, strategy in enumerate(strategies_present):
                std_val = robustness_data[strategy]['std']
                ax1.text(i, std_val + 0.001, f'{std_val:.4f}',
                        ha='center', va='bottom', fontsize=12, fontweight='bold')

            means = [robustness_data[s]['mean'] for s in strategies_present]
            stds = [robustness_data[s]['std'] for s in strategies_present]

            ax2.scatter(stds, means, s=300, color=[colors[s] for s in strategies_present],
                       edgecolor='black', linewidth=2)

            for i, strategy in enumerate(strategies_present):
                ax2.annotate(strategy, (stds[i], means[i]),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=14, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            ax2.set_xlabel('Standard Deviation of Accuracy (Lower = More Robust)', fontsize=14)
            ax2.set_ylabel('Mean Accuracy Across Heterogeneity Levels', fontsize=14)
            ax2.set_title('Performance vs Robustness Trade-off\nFedCRA Optimal Balance',
                         fontsize=16, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(left=0)

        plt.tight_layout()
        plot_path = output_dir / "09_robustness_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
        print(f"Plot saved: {plot_path}")
        plt.close()


def main():
    import sys

    project_path = Path(__file__).parent
    results_path = project_path / "dataset" / "models" / "iomt_traffic" / "Category"

    print("Comparing FedCRA vs FedAvg Performance")
    print(f"Results directory: {results_path}")
    print("")

    comparator = FedCRAFedAvgComparator(results_path)
    comparator.load_all_experiments()

    if not comparator.experiments:
        print("No experiments found. Please ensure experiment results exist.")
        sys.exit(1)

    print(f"Loaded {len(comparator.experiments)} experiments")
    print("")

    report_path = project_path / "fedcra_vs_fedavg_report.txt"
    comparator.generate_text_report(output_file=report_path)

    try:
        viz_dir = project_path / "fedcra_vs_fedavg_plots"
        comparator.create_visualizations(output_dir=viz_dir)
        print(f"\nVisualization comparison saved to: {viz_dir}")
    except Exception as e:
        print(f"Could not create visualizations: {e}")

    print("\n✓ FedCRA vs FedAvg comparison complete!")


if __name__ == "__main__":
    main()
