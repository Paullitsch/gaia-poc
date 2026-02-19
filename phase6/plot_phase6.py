#!/usr/bin/env python3
"""GAIA Phase 6: Professional plots."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import json
import os

OUT_DIR = "/root/.openclaw/workspace/gaia-poc/phase6"
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) if v.replace('-','').replace('.','').replace('e','').replace('+','').isdigit() else v for k, v in row.items()})
    return rows

def smooth(data, window=10):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

# ── Plot 1: Learning curves ──
def plot_learning_curves():
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    methods = [
        ("neuromod_v2_5sig_results.csv", "Neuromod v2 (5 signals)", "#e74c3c"),
        ("neuromod_temporal_results.csv", "Neuromod + Temporal", "#3498db"),
        ("neuromod_predcoding_results.csv", "Neuromod + PredCoding", "#2ecc71"),
    ]

    for fname, label, color in methods:
        path = os.path.join(OUT_DIR, fname)
        if not os.path.exists(path):
            continue
        rows = load_csv(path)
        gens = [r['generation'] for r in rows]
        best_ever = [r['best_ever'] for r in rows]
        means = [r['mean_fitness'] for r in rows]

        ax.plot(gens, best_ever, color=color, linewidth=2, label=f"{label} (best)")
        smoothed = smooth(means, 5)
        ax.plot(gens[:len(smoothed)], smoothed, color=color, linewidth=1, alpha=0.5, linestyle='--', label=f"{label} (mean)")

    # PPO
    ppo_path = os.path.join(OUT_DIR, "ppo_results.csv")
    if os.path.exists(ppo_path):
        rows = load_csv(ppo_path)
        steps = [r['step'] for r in rows]
        # Normalize to generation scale
        if steps and methods:
            max_gen = max(r['generation'] for r in load_csv(os.path.join(OUT_DIR, methods[0][0]))) if os.path.exists(os.path.join(OUT_DIR, methods[0][0])) else 150
            max_step = max(steps) if steps else 500000
            norm_gens = [s / max_step * max_gen for s in steps]
            evals = [r['best_eval'] for r in rows]
            ax.plot(norm_gens, evals, color='#9b59b6', linewidth=2, label='PPO (best eval)')

    ax.axhline(y=200, color='gold', linestyle=':', linewidth=2, alpha=0.7, label='Solved threshold')
    ax.axhline(y=80, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Phase 5 best (+80)')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Score')
    ax.set_title('GAIA Phase 6: Learning Curves')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "phase6_learning_curves.png"), dpi=300)
    plt.close()
    print("Saved phase6_learning_curves.png")

# ── Plot 2: Final comparison bar chart ──
def plot_comparison():
    results_path = os.path.join(OUT_DIR, "phase6_results.json")
    if not os.path.exists(results_path):
        print("No results JSON found")
        return

    with open(results_path) as f:
        results = json.load(f)

    methods = []
    scores = []
    stds = []
    colors = []
    color_map = {'neuromod_v2': '#e74c3c', 'neuromod_temporal': '#3498db',
                 'neuromod_predcod': '#2ecc71', 'ppo': '#9b59b6'}

    for key, res in results.items():
        methods.append(res['method'])
        best = res.get('best_ever', res.get('best_eval', 0))
        scores.append(float(best))
        stds.append(float(res.get('final_std', 0)))
        colors.append(color_map.get(key, '#95a5a6'))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, scores, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=200, color='gold', linestyle=':', linewidth=2, label='Solved')
    ax.axhline(y=80, color='gray', linestyle=':', linewidth=1, label='Phase 5 best')
    ax.set_ylabel('Best Score')
    ax.set_title('GAIA Phase 6: Method Comparison')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{score:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "phase6_comparison.png"), dpi=300)
    plt.close()
    print("Saved phase6_comparison.png")

# ── Plot 3: Cross-phase trajectory ──
def plot_trajectory():
    # Historical data from all phases
    phases = ['P1\nCartPole', 'P2\nRew-Hebb', 'P3\nFF+Evo', 'P4\nMeta-Plast', 'P5\nNeuromod']
    scores = [500.0, 59.7, -50, -50.4, 80.0]  # P3 approximate
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    # Add phase 6 data if available
    results_path = os.path.join(OUT_DIR, "phase6_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        # Find best evolutionary method
        best_key = max((k for k in results if k != 'ppo'),
                       key=lambda k: float(results[k].get('best_ever', -999)),
                       default=None)
        if best_key:
            p6_score = float(results[best_key].get('best_ever', 0))
            phases.append(f'P6\n{results[best_key]["method"][:10]}')
            scores.append(p6_score)
            colors.append('#e67e22')

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(phases)), scores, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels(phases, fontsize=9)
    ax.set_ylabel('Best Score')
    ax.set_title('GAIA Project: Score Trajectory Across All Phases')
    ax.axhline(y=200, color='gold', linestyle=':', linewidth=2, label='LunarLander Solved')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    for i, s in enumerate(scores):
        ax.text(i, s + (10 if s >= 0 else -20), f'{s:.1f}', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "phase6_trajectory.png"), dpi=300)
    plt.close()
    print("Saved phase6_trajectory.png")

# ── Plot 4: Mutation sigma / stagnation analysis ──
def plot_sigma():
    fig, ax = plt.subplots(figsize=(10, 5))
    methods = [
        ("neuromod_v2_5sig_results.csv", "Neuromod v2", "#e74c3c"),
        ("neuromod_temporal_results.csv", "Neuromod Temporal", "#3498db"),
        ("neuromod_predcoding_results.csv", "Neuromod PredCoding", "#2ecc71"),
    ]
    for fname, label, color in methods:
        path = os.path.join(OUT_DIR, fname)
        if not os.path.exists(path):
            continue
        rows = load_csv(path)
        gens = [r['generation'] for r in rows]
        sigmas = [r.get('sigma', 0.02) for r in rows]
        ax.plot(gens, sigmas, color=color, linewidth=1.5, label=label)

    ax.set_xlabel('Generation')
    ax.set_ylabel('Effective Mutation σ')
    ax.set_title('Adaptive Mutation Strength')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "phase6_sigma.png"), dpi=300)
    plt.close()
    print("Saved phase6_sigma.png")

if __name__ == "__main__":
    plot_learning_curves()
    plot_comparison()
    plot_trajectory()
    plot_sigma()
    print("All plots generated.")
