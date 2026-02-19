#!/usr/bin/env python3
"""Plot Phase 2 results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv, os
import numpy as np

def load_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)

def plot_evo_methods():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = [
        ("Pure Evolution", "results/pure_evo.csv", "#e74c3c"),
        ("Evo+Hebbian", "results/evo_hebbian.csv", "#3498db"),
        ("Evo+Reward-Hebbian", "results/evo_reward_hebbian.csv", "#2ecc71"),
        ("Novelty+Evolution", "results/novelty_evo.csv", "#9b59b6"),
    ]
    
    for name, path, color in methods:
        if not os.path.exists(path):
            continue
        data = load_csv(path)
        gens = [int(d['generation']) for d in data]
        best = [float(d['best_fitness']) for d in data]
        mean = [float(d['mean_fitness']) for d in data]
        
        ax1.plot(gens, best, label=name, color=color, linewidth=2)
        ax2.plot(gens, mean, label=name, color=color, linewidth=2)
    
    ax1.axhline(y=200, color='gold', linestyle='--', label='Solved (200)')
    ax2.axhline(y=200, color='gold', linestyle='--', label='Solved (200)')
    
    ax1.set_title('Best Fitness per Generation', fontsize=13)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.set_title('Mean Fitness per Generation', fontsize=13)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Fitness')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.suptitle('GAIA Phase 2: LunarLander-v3 — Evolutionary Methods', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/evo_fitness_curves.png', dpi=150)
    print("Saved results/evo_fitness_curves.png")
    plt.close()

def plot_reinforce():
    path = "results/reinforce.csv"
    if not os.path.exists(path):
        print("No REINFORCE data")
        return
    data = load_csv(path)
    eps = [int(d['episode']) for d in data]
    train = [float(d['train_reward']) for d in data]
    eval_r = [float(d['eval_reward']) for d in data]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps, train, label='Train (mean 50)', color='#e74c3c', alpha=0.7)
    ax.plot(eps, eval_r, label='Eval', color='#3498db', linewidth=2)
    ax.axhline(y=200, color='gold', linestyle='--', label='Solved (200)')
    ax.set_title('REINFORCE Baseline — LunarLander-v3', fontsize=13, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/reinforce_curve.png', dpi=150)
    print("Saved results/reinforce_curve.png")
    plt.close()

if __name__ == "__main__":
    plot_evo_methods()
    plot_reinforce()
    print("Done.")
