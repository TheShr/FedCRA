"""
compare.py — FedAvg vs FedCRA full metric comparison
=====================================================
Usage:
  python compare.py \
      --results_root "C:/Users/.../models/iomt_traffic/Category" \
      --strategies FedAvg FedCRA \
      --model DNN \
      --target_acc 0.80
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
METRICS = {
    "accuracy":    ("Accuracy",                   False),
    "f1_score":    ("Macro F1-Score",              False),
    "precision":   ("Macro Precision",             False),
    "recall":      ("Macro Recall",                False),
    "macro_fpr":   ("Macro FPR (lower=better)",    True),
    "loss":        ("Loss (lower=better)",         True),
}

CLASS_NAMES = {
    0: "apachekiller", 1: "arpspoofing",  2: "camoverflow",
    3: "mqttmalaria",  4: "netscan",      5: "normal",
    6: "rudeadyet",    7: "slowloris",    8: "slowread",
}
MINORITY_CLASSES = {0, 2, 3, 6, 8}

COLORS = {"FedAvg": "#2196F3", "FedCRA": "#E91E63",
          "FedProx": "#4CAF50", "SCAFFOLD": "#FF9800"}
LINE_STYLES = ["solid", "dashed", "dashdot", "dotted"]


def color(name, idx):
    return COLORS.get(name, f"C{idx}")


# ── Data loading ──────────────────────────────────────────────────────────────
def load_metrics(results_root: Path, strategy: str, model: str) -> pd.DataFrame:
    path = results_root / strategy / model / "metrics" / "server_metrics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No metrics file at:\n  {path}\n"
            f"Run the simulation for '{strategy}' first.")
    raw = json.loads(path.read_text())
    # Flatten per_class_f1 out before making DataFrame
    rows = []
    for row in raw:
        flat = {k: v for k, v in row.items() if k != "per_class_f1"}
        rows.append(flat)
    df = pd.DataFrame(rows)
    return df.sort_values("round").reset_index(drop=True)


def load_per_class_f1(results_root: Path, strategy: str, model: str):
    path = results_root / strategy / model / "metrics" / "server_metrics.json"
    if not path.exists():
        return None, None
    data = json.loads(path.read_text())
    all_class_ids = set()
    for row in data:
        pcf = row.get("per_class_f1", {})
        all_class_ids.update(int(k) for k in pcf.keys())
    if not all_class_ids:
        return None, None
    rounds = [r["round"] for r in data]
    per_class = {k: [] for k in sorted(all_class_ids)}
    for row in data:
        pcf = row.get("per_class_f1", {})
        for k in sorted(all_class_ids):
            per_class[k].append(float(pcf.get(str(k), pcf.get(k, 0.0))))
    return rounds, per_class


def smooth(values, window=3):
    arr = np.array(values, dtype=float)
    if window <= 1 or len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="same")


# ── Plot 1: 2×3 metrics grid ──────────────────────────────────────────────────
def plot_grid(dfs, max_rounds, prefix, win=3):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for ax_i, (metric, (label, lower)) in enumerate(METRICS.items()):
        ax = axes[ax_i]
        ax.set_title(label, fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("Round", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        for s_i, (strat, df) in enumerate(dfs.items()):
            if metric not in df.columns:
                continue
            rounds = df["round"].values[:max_rounds]
            vals   = df[metric].values[:max_rounds]
            c  = color(strat, s_i)
            ls = LINE_STYLES[s_i % len(LINE_STYLES)]
            ax.plot(rounds, vals, color=c, alpha=0.2, linewidth=0.8, linestyle=ls)
            ax.plot(rounds, smooth(vals, win), label=strat, color=c,
                    linewidth=2.2, linestyle=ls)
            best_idx = int(np.argmin(vals) if lower else np.argmax(vals))
            ax.scatter(rounds[best_idx], vals[best_idx], marker="*",
                       color=c, s=120, zorder=5)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))
    fig.suptitle("FedAvg vs FedCRA — Server Evaluation Metrics\n(★ = best round)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = f"{prefix}_metrics_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Plot 2: Convergence speed ─────────────────────────────────────────────────
def plot_convergence(dfs, target, prefix):
    names, rounds_hit = [], []
    for strat, df in dfs.items():
        if "accuracy" not in df.columns:
            continue
        hit = df[df["accuracy"] >= target]
        rounds_hit.append(int(hit["round"].iloc[0]) if len(hit) else None)
        names.append(strat)
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [color(s, i) for i, s in enumerate(names)]
    vals = [r if r is not None else 0 for r in rounds_hit]
    bars = ax.bar(names, vals, color=colors, edgecolor="white", width=0.45)
    for bar, rtt in zip(bars, rounds_hit):
        lbl = str(int(bar.get_height())) if rtt is not None else "Never"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3, lbl,
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title(f"Rounds to First Reach {target*100:.0f}% Accuracy", fontsize=12, fontweight="bold")
    ax.set_ylabel("Round")
    ax.set_ylim(0, max([v for v in vals if v] or [1]) * 1.3)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = f"{prefix}_convergence_speed.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Plot 3: Best-round vs final-round ─────────────────────────────────────────
def plot_best_vs_final(dfs, prefix):
    key_metrics = ["accuracy", "f1_score", "macro_fpr"]
    labels_map  = {"accuracy": "Accuracy", "f1_score": "Macro F1", "macro_fpr": "Macro FPR"}
    n_metrics = len(key_metrics)
    n_strats  = len(dfs)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    for ax, metric in zip(axes, key_metrics):
        lower = METRICS[metric][1]
        x = np.arange(n_strats)
        width = 0.35
        bests, finals = [], []
        for strat, df in dfs.items():
            if metric not in df.columns:
                bests.append(0); finals.append(0); continue
            vals = df[metric].values
            bests.append(float(vals.min() if lower else vals.max()))
            finals.append(float(vals[-1]))
        bars1 = ax.bar(x - width/2, bests,  width, label="Best round",  alpha=0.85,
                       color=[color(s, i) for i, s in enumerate(dfs)])
        bars2 = ax.bar(x + width/2, finals, width, label="Final round", alpha=0.45,
                       color=[color(s, i) for i, s in enumerate(dfs)])
        ax.set_xticks(x)
        ax.set_xticklabels(list(dfs.keys()))
        ax.set_title(labels_map[metric], fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.001,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)
    fig.suptitle("Best-Round vs Final-Round Performance",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = f"{prefix}_best_vs_final.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Plot 4: Radar (final + best) ──────────────────────────────────────────────
def plot_radar(dfs, prefix):
    radar_keys = ["accuracy", "f1_score", "precision", "recall"]
    N = len(radar_keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), subplot_kw=dict(polar=True))
    for ax, use_best in zip(axes, [False, True]):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([METRICS[k][0] for k in radar_keys], fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title("Best-Round Performance" if use_best else "Final-Round Performance",
                     fontsize=12, fontweight="bold", pad=20)
        for s_i, (strat, df) in enumerate(dfs.items()):
            if use_best:
                vals = [float(np.clip(df[k].max() if k in df.columns else 0, 0, 1))
                        for k in radar_keys]
            else:
                last = df.iloc[-1]
                vals = [float(np.clip(last.get(k, 0.0), 0, 1)) for k in radar_keys]
            vals += vals[:1]
            c = color(strat, s_i)
            ax.plot(angles, vals, color=c, linewidth=2, label=strat)
            ax.fill(angles, vals, color=c, alpha=0.1)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.tight_layout()
    out = f"{prefix}_radar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Plot 5: Weighted F1 vs Macro F1 ──────────────────────────────────────────
def plot_weighted_vs_macro_f1(dfs, prefix):
    metrics_to_plot = [
        ("f1_score",    "Macro F1 (equal class weight)"),
        ("f1_weighted", "Weighted F1 (proportional to support)"),
    ]
    has_weighted = any("f1_weighted" in df.columns for df in dfs.values())
    if not has_weighted:
        print("  Skipping weighted F1 chart (f1_weighted not in metrics — retrain FedCRA)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (metric, label) in zip(axes, metrics_to_plot):
        ax.set_title(label, fontweight="bold", fontsize=11)
        ax.set_xlabel("Round"); ax.set_ylabel("F1-Score")
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.4)
        for s_i, (strat, df) in enumerate(dfs.items()):
            if metric not in df.columns:
                continue
            rounds = df["round"].values
            vals   = df[metric].values
            c  = color(strat, s_i)
            ls = LINE_STYLES[s_i % len(LINE_STYLES)]
            ax.plot(rounds, vals, color=c, alpha=0.2, linewidth=0.8, linestyle=ls)
            ax.plot(rounds, smooth(vals, 3), label=strat, color=c,
                    linewidth=2.2, linestyle=ls)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))
    fig.suptitle("Macro F1 vs Weighted F1 — Effect of Class Imbalance",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = f"{prefix}_f1_weighted_vs_macro.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Plot 6: Per-class F1 ──────────────────────────────────────────────────────
def plot_per_class_f1(results_root, strategies, model, prefix):
    all_data = {}
    all_rounds_data = {}
    for strategy in strategies:
        rounds, per_class = load_per_class_f1(results_root, strategy, model)
        if rounds is not None:
            all_data[strategy] = per_class
            all_rounds_data[strategy] = rounds

    if not all_data:
        print("  Skipping per-class F1 chart (no per_class_f1 in metrics — retrain with new code)")
        return

    all_class_ids = sorted(CLASS_NAMES.keys())
    n_strats = len(all_data)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: final-round per-class F1 bars
    ax = axes[0]
    x = np.arange(len(all_class_ids))
    bar_width = 0.8 / n_strats
    for s_i, (strategy, per_class) in enumerate(all_data.items()):
        final_f1 = [per_class.get(k, [0])[-1] for k in all_class_ids]
        offset = (s_i - (n_strats - 1) / 2) * bar_width
        bars = ax.bar(x + offset, final_f1, bar_width,
                      label=strategy, color=color(strategy, s_i),
                      alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, final_f1):
            if val > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{'*' if k in MINORITY_CLASSES else ''}{CLASS_NAMES.get(k, str(k))}"
         for k in all_class_ids],
        rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("F1-Score")
    ax.set_ylim(0, 1.15)
    ax.set_title("Per-Class F1 (final round)\n* = minority class", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Right: minority class F1 over rounds
    ax2 = axes[1]
    minority_ids = sorted(MINORITY_CLASSES & set(all_class_ids))
    line_styles_m = ["solid", "dashed", "dashdot", "dotted"]
    for s_i, (strategy, per_class) in enumerate(all_data.items()):
        rounds = all_rounds_data[strategy]
        c = color(strategy, s_i)
        for m_i, class_id in enumerate(minority_ids):
            vals = per_class.get(class_id, [0] * len(rounds))
            ls = line_styles_m[m_i % len(line_styles_m)]
            label = f"{strategy} — {CLASS_NAMES.get(class_id, str(class_id))}"
            ax2.plot(rounds, smooth(vals, 3), label=label,
                     color=c, linewidth=1.8, linestyle=ls, alpha=0.85)
    ax2.set_xlabel("Round"); ax2.set_ylabel("F1-Score")
    ax2.set_title("Minority-Class F1 Over Rounds", fontweight="bold")
    ax2.legend(fontsize=7.5, ncol=2)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.set_ylim(0, 1.05)
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))

    fig.suptitle("Per-Class F1 Analysis — FedAvg vs FedCRA",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = f"{prefix}_per_class_f1.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Summary CSV + console ─────────────────────────────────────────────────────
def build_summary(dfs, max_rounds, prefix):
    rows = []
    for strat, df in dfs.items():
        df_cut = df[df["round"] <= max_rounds]
        row = {"Strategy": strat}
        for metric, (label, lower) in METRICS.items():
            if metric not in df_cut.columns:
                continue
            final = float(df_cut[metric].iloc[-1])
            best  = float(df_cut[metric].min() if lower else df_cut[metric].max())
            best_rnd = int(df_cut.loc[
                df_cut[metric].idxmin() if lower else df_cut[metric].idxmax(), "round"])
            row[f"{metric}_final"]      = round(final, 5)
            row[f"{metric}_best"]       = round(best, 5)
            row[f"{metric}_best_round"] = best_rnd
        rows.append(row)
    summary = pd.DataFrame(rows)
    csv_path = f"{prefix}_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    print("\n" + "=" * 76)
    print("  COMPARISON SUMMARY")
    print("=" * 76)
    for _, row in summary.iterrows():
        print(f"\n  Strategy: {row['Strategy']}")
        for metric, (label, lower) in METRICS.items():
            if f"{metric}_final" not in row:
                continue
            arrow = "v (lower)" if lower else "^ (higher)"
            print(f"    {label:38s}  final={row[f'{metric}_final']:.4f}  "
                  f"best={row[f'{metric}_best']:.4f} @ rnd {row[f'{metric}_best_round']}  {arrow}")
    print("=" * 76)

    if len(dfs) == 2:
        print("\n  HEAD-TO-HEAD (best-round):")
        strats = list(summary["Strategy"])
        r0, r1 = summary.iloc[0], summary.iloc[1]
        for metric, (label, lower) in METRICS.items():
            if f"{metric}_best" not in r0.index:
                continue
            v0, v1 = r0[f"{metric}_best"], r1[f"{metric}_best"]
            winner = strats[0] if (v0 < v1 if lower else v0 > v1) else strats[1]
            diff = abs(v0 - v1)
            print(f"    {label:38s}  winner={winner:10s}  margin={diff:.4f}")
    print()
    return summary


def print_per_class_summary(results_root, strategies, model):
    all_data = {}
    for strategy in strategies:
        rounds, per_class = load_per_class_f1(results_root, strategy, model)
        if rounds:
            all_data[strategy] = {k: v[-1] for k, v in per_class.items()}
    if not all_data:
        return
    print("\n" + "=" * 76)
    print("  PER-CLASS F1 — FINAL ROUND  (* = minority class)")
    print("=" * 76)
    header = f"  {'Class':30s}"
    for s in strategies:
        header += f"  {s:12s}"
    if len(strategies) == 2:
        header += "  Winner"
    print(header)
    print("  " + "-" * 70)
    for k in sorted(CLASS_NAMES.keys()):
        name = CLASS_NAMES.get(k, str(k))
        tag = " *" if k in MINORITY_CLASSES else "  "
        row = f"  {name+tag:30s}"
        vals = []
        for s in strategies:
            v = all_data.get(s, {}).get(k, 0.0)
            vals.append(v)
            row += f"  {v:.4f}      "
        if len(vals) == 2:
            winner = strategies[0] if vals[0] >= vals[1] else strategies[1]
            diff = abs(vals[0] - vals[1])
            row += f"  {winner} (+{diff:.4f})"
        print(row)
    print("=" * 76 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Compare FL strategies from server_metrics.json")
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--strategies", nargs="+", default=["FedAvg", "FedCRA"])
    ap.add_argument("--model",      default="DNN")
    ap.add_argument("--output",     default="comparison_report")
    ap.add_argument("--rounds",     type=int, default=None)
    ap.add_argument("--target_acc", type=float, default=0.80)
    ap.add_argument("--smooth",     type=int, default=3)
    args = ap.parse_args()

    root = Path(args.results_root)
    dfs = {}
    for strat in args.strategies:
        try:
            dfs[strat] = load_metrics(root, strat, args.model)
            print(f"  Loaded {len(dfs[strat])} rounds for '{strat}'")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

    if not dfs:
        print("ERROR: no valid results found.")
        sys.exit(1)

    max_rounds = args.rounds or max(len(df) for df in dfs.values())
    out_dir = Path(args.output).parent
    if str(out_dir) not in ("", "."):
        out_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating plots...")
    plot_grid(dfs, max_rounds, args.output, win=args.smooth)
    plot_convergence(dfs, args.target_acc, args.output)
    plot_best_vs_final(dfs, args.output)
    if len(dfs) >= 2:
        plot_radar(dfs, args.output)
    plot_weighted_vs_macro_f1(dfs, args.output)
    plot_per_class_f1(root, args.strategies, args.model, args.output)
    print_per_class_summary(root, args.strategies, args.model)
    build_summary(dfs, max_rounds, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
