#!/usr/bin/env python3
"""Generate publication-quality plots for GAIA Phase 3"""

import csv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'legend.fontsize': 10, 'figure.facecolor': 'white'
})

base = '/root/.openclaw/workspace/gaia-poc/phase3'

def load_csv(name):
    eps, evals, avgs = [], [], []
    with open(f'{base}/{name}', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            eps.append(float(row['episode']))
            evals.append(float(row['eval_reward']))
            avgs.append(float(row['avg_reward']))
    return np.array(eps), np.array(evals), np.array(avgs)

methods = [
    ('ff_results.csv', 'Forward-Forward', '#e74c3c'),
    ('pc_results.csv', 'Predictive Coding', '#3498db'),
    ('dg_results.csv', 'Decoupled Greedy', '#2ecc71'),
    ('hybrid_results.csv', 'Hybrid Evo+FF', '#f39c12'),
    ('bp_results.csv', 'Backprop (Actor-Critic)', '#9b59b6'),
]

# === Plot 1: Learning Curves ===
fig, ax = plt.subplots(figsize=(12, 7))

for fname, label, color in methods:
    eps, evals, avgs = load_csv(fname)
    # Smooth with rolling window
    window = 3
    if len(evals) >= window:
        smoothed = np.convolve(evals, np.ones(window)/window, mode='valid')
        x_smooth = eps[window-1:]
    else:
        smoothed = evals
        x_smooth = eps
    ax.plot(x_smooth, smoothed, label=label, color=color, linewidth=2.5, alpha=0.9)
    ax.fill_between(x_smooth, smoothed - 30, smoothed + 30, color=color, alpha=0.1)

ax.axhline(y=200, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Solved (200)')
ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.set_xlabel('Episodes / Evaluations', fontsize=13)
ax.set_ylabel('Evaluation Reward', fontsize=13)
ax.set_title('GAIA Phase 3: Local Learning Methods vs Backprop on LunarLander-v3', fontsize=15, fontweight='bold')
ax.legend(loc='lower right', framealpha=0.9, edgecolor='gray')
ax.grid(True, alpha=0.3)
ax.set_ylim(-400, 300)
ax.text(500, -380, '* Predictive Coding diverges to -750 (clipped)', fontsize=9, color='#3498db', style='italic')
plt.tight_layout()
plt.savefig(f'{base}/learning_curves.png', dpi=300, bbox_inches='tight')
print("Saved learning_curves.png")
plt.close()

# === Plot 2: Final Performance Bar Chart ===
fig, ax = plt.subplots(figsize=(10, 6))

names = []
finals = []
bests = []
colors = []
for fname, label, color in methods:
    eps, evals, avgs = load_csv(fname)
    names.append(label)
    finals.append(evals[-1])
    bests.append(max(evals))
    colors.append(color)

x = np.arange(len(names))
w = 0.35
bars1 = ax.bar(x - w/2, finals, w, label='Final Eval', color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
bars2 = ax.bar(x + w/2, bests, w, label='Best Eval', color=colors, alpha=0.5, edgecolor='white', linewidth=1.5, hatch='//')

ax.axhline(y=200, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(len(names)-0.5, 210, 'Solved', fontsize=10, color='gray', ha='right')
ax.set_ylabel('Reward', fontsize=13)
ax.set_title('Final & Best Performance by Method', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15, ha='right', fontsize=10)
ax.legend(loc='upper left')
ax.set_ylim(-700, 250)
ax.grid(True, axis='y', alpha=0.3)

# Add value labels
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h - 20 if h < 0 else h + 5, f'{h:.0f}', 
            ha='center', va='bottom' if h >= 0 else 'top', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{base}/final_performance.png', dpi=300, bbox_inches='tight')
print("Saved final_performance.png")
plt.close()

# === Plot 3: Sample Efficiency ===
fig, ax = plt.subplots(figsize=(10, 6))

threshold = -100
efficiencies = []
for fname, label, color in methods:
    eps, evals, avgs = load_csv(fname)
    reached = None
    for i, e in enumerate(evals):
        if e >= threshold:
            reached = eps[i]
            break
    efficiencies.append(reached if reached is not None else float('inf'))

# Replace inf with max + 100 for display
max_finite = max(e for e in efficiencies if e != float('inf'))
display_eff = [e if e != float('inf') else max_finite * 1.3 for e in efficiencies]

bars = ax.bar(range(len(names)), display_eff, color=[m[2] for m in methods], alpha=0.8, edgecolor='white', linewidth=1.5)

for i, (bar, eff) in enumerate(zip(bars, efficiencies)):
    if eff == float('inf'):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 'Not\nreached', 
                ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')
    else:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{eff:.0f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Episodes to Reach Threshold', fontsize=13)
ax.set_title(f'Sample Efficiency (episodes to reach reward ≥ {threshold})', fontsize=15, fontweight='bold')
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=20, ha='right', fontsize=11)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{base}/sample_efficiency.png', dpi=300, bbox_inches='tight')
print("Saved sample_efficiency.png")
plt.close()

print("\nAll plots generated!")
