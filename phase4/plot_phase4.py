"""
GAIA Phase 4: Publication-quality plots.
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))

def load_csv(name):
    path = os.path.join(DIR, f'{name}_results.csv')
    eps, rewards = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            eps.append(int(row['episode']))
            rewards.append(float(row['eval_reward']))
    return eps, rewards

def smooth(vals, window=3):
    if len(vals) <= window:
        return vals
    return np.convolve(vals, np.ones(window)/window, mode='valid').tolist()

def main():
    plt.rcParams.update({
        'font.size': 12, 'font.family': 'serif',
        'axes.linewidth': 1.2, 'figure.dpi': 100
    })

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = {'hybrid_fixed': '#e74c3c', 'hybrid_meta': '#2ecc71', 'backprop_ac': '#3498db'}
    labels = {'hybrid_fixed': 'Hybrid Evo+FF (fixed)',
              'hybrid_meta': 'Hybrid Evo+FF (meta-learned)',
              'backprop_ac': 'Backprop Actor-Critic'}
    markers = {'hybrid_fixed': 'o', 'hybrid_meta': 's', 'backprop_ac': '^'}

    for name in ['hybrid_fixed', 'hybrid_meta', 'backprop_ac']:
        try:
            eps, rewards = load_csv(name)
            sm = smooth(rewards, 3)
            offset = len(rewards) - len(sm)
            ax.plot(eps[offset:], sm, color=colors[name], label=labels[name],
                    linewidth=2.2, marker=markers[name], markersize=4, markevery=max(1, len(sm)//15))
            ax.fill_between(eps[offset:],
                            [s-20 for s in sm], [s+20 for s in sm],
                            alpha=0.1, color=colors[name])
        except FileNotFoundError:
            print(f"Warning: {name}_results.csv not found")

    ax.axhline(y=200, color='gold', linestyle='--', linewidth=1.5, alpha=0.8, label='Solved threshold (200)')
    ax.set_xlabel('Training Episodes', fontsize=14)
    ax.set_ylabel('Evaluation Reward', fontsize=14)
    ax.set_title('GAIA Phase 4: Hybrid Evo+FF vs Backpropagation\non LunarLander-v3', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    out = os.path.join(DIR, 'learning_curves_phase4.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved {out}")
    plt.close()

    # Meta-learned hyperparameters plot
    meta_path = os.path.join(DIR, 'meta_history.csv')
    if os.path.exists(meta_path):
        data = []
        with open(meta_path) as f:
            for row in csv.DictReader(f):
                data.append(row)

        if data:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            gens = [int(d['generation']) for d in data]

            # Learning rates
            ax = axes[0, 0]
            for j in range(3):
                key = f'ff_lr_{j}'
                if key in data[0]:
                    ax.plot(gens, [float(d[key]) for d in data], label=f'Layer {j+1}', linewidth=2)
            ax.set_ylabel('FF Learning Rate')
            ax.set_title('Meta-Learned Learning Rates')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

            # Thresholds
            ax = axes[0, 1]
            for j in range(3):
                key = f'ff_thresh_{j}'
                if key in data[0]:
                    ax.plot(gens, [float(d[key]) for d in data], label=f'Layer {j+1}', linewidth=2)
            ax.set_ylabel('FF Threshold')
            ax.set_title('Meta-Learned Goodness Thresholds')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            # Policy LR
            ax = axes[1, 0]
            ax.plot(gens, [float(d['policy_lr']) for d in data], color='purple', linewidth=2)
            ax.set_ylabel('Policy Learning Rate')
            ax.set_xlabel('Generation')
            ax.set_title('Meta-Learned Policy LR')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

            # Goodness type
            ax = axes[1, 1]
            ax.plot(gens, [int(d['goodness_type']) for d in data], 'ko-', markersize=4)
            ax.set_ylabel('Goodness Type')
            ax.set_xlabel('Generation')
            ax.set_title('Selected Goodness Function')
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(['L2', 'L1', 'Max'])
            ax.grid(True, alpha=0.3)

            fig.suptitle('Evolution of Meta-Learned Hyperparameters', fontsize=15, fontweight='bold')
            plt.tight_layout()
            out = os.path.join(DIR, 'meta_hyperparams.png')
            plt.savefig(out, dpi=300, bbox_inches='tight')
            print(f"Saved {out}")
            plt.close()

if __name__ == '__main__':
    main()
