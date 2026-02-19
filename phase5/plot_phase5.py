#!/usr/bin/env python3
"""Phase 5 publication-quality plots."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import json
import os

matplotlib.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'legend.fontsize': 10, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'font.family': 'sans-serif',
})

OUT = "/root/.openclaw/workspace/gaia-poc/phase5"
COLORS = {'meta': '#2196F3', 'neuro': '#4CAF50', 'ppo': '#FF9800', 'ff': '#9C27B0'}

def smooth(y, window=10):
    return pd.Series(y).rolling(window, min_periods=1).mean().values

# Load data
meta = pd.read_csv(f"{OUT}/meta_plasticity_results.csv")
neuro = pd.read_csv(f"{OUT}/neuromod_results.csv")
ppo = pd.read_csv(f"{OUT}/ppo_results.csv")
ff = pd.read_csv(f"{OUT}/ff_only_results.csv")
meta_hp = pd.read_csv(f"{OUT}/meta_plasticity_hyperparams.csv")

with open(f"{OUT}/results_summary.json") as f:
    summary = json.load(f)

# ─── Plot 1: Learning Curves ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Evo methods by generation
ax = axes[0]
ax.plot(meta['gen'], smooth(meta['best_ever']), color=COLORS['meta'], linewidth=2, label='Meta-Plasticity (best ever)')
ax.plot(meta['gen'], smooth(meta['mean']), color=COLORS['meta'], linewidth=1, alpha=0.5, linestyle='--', label='Meta-Plasticity (mean)')
ax.fill_between(meta['gen'], smooth(meta['mean'] - meta['std']), smooth(meta['mean'] + meta['std']), color=COLORS['meta'], alpha=0.1)

ax.plot(neuro['gen'], smooth(neuro['best_ever']), color=COLORS['neuro'], linewidth=2, label='Neuromodulated (best ever)')
ax.plot(neuro['gen'], smooth(neuro['mean']), color=COLORS['neuro'], linewidth=1, alpha=0.5, linestyle='--', label='Neuromodulated (mean)')
ax.fill_between(neuro['gen'], smooth(neuro['mean'] - neuro['std']), smooth(neuro['mean'] + neuro['std']), color=COLORS['neuro'], alpha=0.1)

ax.axhline(y=200, color='red', linestyle=':', alpha=0.5, label='Solved threshold')
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
ax.set_xlabel('Generation')
ax.set_ylabel('Reward')
ax.set_title('Evolutionary Methods: Learning Curves')
ax.legend(loc='lower right', fontsize=8)
ax.grid(True, alpha=0.3)

# Right: All methods by total evaluations
ax = axes[1]
ax.plot(meta['total_evals'], smooth(meta['best_ever']), color=COLORS['meta'], linewidth=2, label='Meta-Plasticity')
ax.plot(neuro['total_evals'], smooth(neuro['best_ever']), color=COLORS['neuro'], linewidth=2, label='Neuromodulated')
# PPO: convert steps to approximate evals
ax.plot(ppo['step'], smooth(ppo['best_eval']), color=COLORS['ppo'], linewidth=2, label='PPO')
ax.plot(ff['episode'], smooth(ff['best_eval']), color=COLORS['ff'], linewidth=2, label='FF Only')
ax.axhline(y=200, color='red', linestyle=':', alpha=0.5, label='Solved')
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
ax.set_xlabel('Evaluations / Timesteps')
ax.set_ylabel('Best Reward')
ax.set_title('All Methods: Best Performance')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/learning_curves_phase5.png")
plt.close()
print("Saved learning_curves_phase5.png")

# ─── Plot 2: Bar Chart Final Performance ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
methods = ['Meta-Plasticity\nEvo+FF', 'Neuromodulated\nEvo+FF', 'PPO\n(baseline)', 'FF Only\n(no evo)']
best_vals = [summary['meta_plasticity']['best_ever'], summary['neuromod']['best_ever'],
             summary['ppo']['best_ever'], summary['ff_only']['best_ever']]
final_vals = [summary['meta_plasticity']['final_mean'], summary['neuromod']['final_mean'],
              summary['ppo']['final_mean'], summary['ff_only']['final_mean']]
final_stds = [summary['meta_plasticity']['final_std'], summary['neuromod']['final_std'],
              summary['ppo']['final_std'], summary['ff_only']['final_std']]
colors = [COLORS['meta'], COLORS['neuro'], COLORS['ppo'], COLORS['ff']]

x = np.arange(len(methods))
width = 0.35
bars1 = ax.bar(x - width/2, best_vals, width, color=colors, alpha=0.9, label='Best Ever', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, final_vals, width, color=colors, alpha=0.5, label='Final (30-ep avg)', 
               edgecolor='black', linewidth=0.5, yerr=final_stds, capsize=3)

ax.axhline(y=200, color='red', linestyle='--', alpha=0.7, label='Solved threshold')
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
ax.set_ylabel('Reward')
ax.set_title('Phase 5: Final Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.grid(True, alpha=0.2, axis='y')

# Add value labels
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 3, f'{h:+.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUT}/bar_chart_phase5.png")
plt.close()
print("Saved bar_chart_phase5.png")

# ─── Plot 3: Meta-learned Hyperparameters ──────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

ax = axes[0, 0]
for i in range(3):
    ax.plot(meta_hp['gen'], meta_hp[f'ff_lr_{i}'], label=f'Layer {i}', alpha=0.7)
ax.set_ylabel('FF Learning Rate')
ax.set_xlabel('Generation')
ax.set_title('Evolution of FF Learning Rates')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
for i in range(3):
    ax.plot(meta_hp['gen'], meta_hp[f'thresh_{i}'], label=f'Layer {i}', alpha=0.7)
ax.set_ylabel('Goodness Threshold')
ax.set_xlabel('Generation')
ax.set_title('Evolution of Goodness Thresholds')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
for i in range(3):
    ax.plot(meta_hp['gen'], meta_hp[f'plasticity_{i}'], label=f'Layer {i}', alpha=0.7)
ax.set_ylabel('Plasticity (mutation strength)')
ax.set_xlabel('Generation')
ax.set_title('Evolution of Plasticity Coefficients')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(meta_hp['gen'], meta_hp['activation_mix'], color='black', linewidth=2)
ax.set_ylabel('Activation Mix')
ax.set_xlabel('Generation')
ax.set_title('Evolution of Activation Mixing')
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

plt.suptitle('Meta-Learned Hyperparameters (Best Agent per Generation)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/meta_hyperparams_phase5.png")
plt.close()
print("Saved meta_hyperparams_phase5.png")

# ─── Plot 4: Phase comparison (4 vs 5) ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
phases = ['Phase 4\nMeta-Plasticity', 'Phase 4\nBackprop AC', 'Phase 5\nMeta-Plasticity', 
          'Phase 5\nNeuromodulated', 'Phase 5\nPPO', 'Phase 5\nFF Only']
values = [-50.4, -158.4, -39.8, 80.0, -54.5, -89.3]
phase_colors = ['#90CAF9', '#FFCC80', COLORS['meta'], COLORS['neuro'], COLORS['ppo'], COLORS['ff']]

bars = ax.bar(range(len(phases)), values, color=phase_colors, edgecolor='black', linewidth=0.5)
ax.axhline(y=200, color='red', linestyle='--', alpha=0.5, label='Solved')
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
ax.set_ylabel('Best Reward')
ax.set_title('Progress: Phase 4 → Phase 5')
ax.set_xticks(range(len(phases)))
ax.set_xticklabels(phases, fontsize=9)
ax.grid(True, alpha=0.2, axis='y')

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 3, 
            f'{val:+.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUT}/phase_comparison.png")
plt.close()
print("Saved phase_comparison.png")

print("\nAll plots saved!")
