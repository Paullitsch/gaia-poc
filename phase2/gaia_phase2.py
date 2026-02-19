#!/usr/bin/env python3
"""GAIA Phase 2: Evolutionary learning on LunarLander-v3.

Methods:
  1. Pure Evolution (tournament selection + mutation)
  2. Evo + Hebbian (lifetime Hebbian plasticity)
  3. Evo + Reward-Hebbian (Hebbian gated by reward signal)
  4. Novelty Search + Evolution (behavioral diversity + fitness)

Population: 200, Generations: 100, Elitism: top 5
Adaptive mutation, simple niching via fitness sharing.
"""

import numpy as np
import gymnasium as gym
import csv, time, os, sys
from dataclasses import dataclass, field

# ── Config ──────────────────────────────────────────────────────────────
POP_SIZE = 100
GENERATIONS = 50
ELITE_COUNT = 5
EVAL_EPISODES = 2
MAX_STEPS = 300
ENV_NAME = "LunarLander-v3"
INPUT_DIM = 8
OUTPUT_DIM = 4
HIDDEN = [64, 64, 32]  # 3 hidden layers → ~10k params
MUTATION_RATE_INIT = 0.05
MUTATION_RATE_MIN = 0.005
HEBBIAN_LR = 0.001
NOVELTY_K = 15
NOVELTY_WEIGHT = 0.5  # blend: fitness*(1-w) + novelty*w

SEED = 42
np.random.seed(SEED)

# ── Network helpers ─────────────────────────────────────────────────────
def make_shapes():
    dims = [INPUT_DIM] + HIDDEN + [OUTPUT_DIM]
    shapes = []
    for i in range(len(dims)-1):
        shapes.append((dims[i], dims[i+1]))  # weight
        shapes.append((dims[i+1],))           # bias
    return shapes

SHAPES = make_shapes()
N_PARAMS = sum(s[0]*s[1] if len(s)==2 else s[0] for s in SHAPES)
print(f"Network: {[INPUT_DIM]+HIDDEN+[OUTPUT_DIM]}, params: {N_PARAMS}")

def decode_params(flat):
    """Decode flat array into list of (W, b) tuples."""
    layers = []
    idx = 0
    for i in range(0, len(SHAPES), 2):
        ws = SHAPES[i]
        bs = SHAPES[i+1]
        wn = ws[0]*ws[1]
        W = flat[idx:idx+wn].reshape(ws)
        idx += wn
        b = flat[idx:idx+bs[0]]
        idx += bs[0]
        layers.append((W, b))
    return layers

def forward(layers, x):
    for i, (W, b) in enumerate(layers):
        x = x @ W + b
        if i < len(layers) - 1:
            x = np.maximum(0, x)  # ReLU
    # softmax for action selection
    ex = np.exp(x - x.max())
    return ex / ex.sum()

# ── Hebbian helpers ─────────────────────────────────────────────────────
def hebbian_update(layers, activations, lr):
    """Simple Hebbian: dW = lr * pre * post (outer product, clipped)."""
    new_layers = []
    for i, (W, b) in enumerate(layers):
        pre = activations[i]
        post = activations[i+1]
        dW = lr * np.outer(pre, post)
        W_new = np.clip(W + dW, -5, 5)
        new_layers.append((W_new, b.copy()))
    return new_layers

def forward_with_activations(layers, x):
    acts = [x.copy()]
    for i, (W, b) in enumerate(layers):
        x = x @ W + b
        if i < len(layers) - 1:
            x = np.maximum(0, x)
        acts.append(x.copy())
    ex = np.exp(x - x.max())
    probs = ex / ex.sum()
    return probs, acts

# ── Evaluation ──────────────────────────────────────────────────────────
def evaluate(flat, method="pure", n_episodes=EVAL_EPISODES):
    """Return (mean_reward, behavior_descriptor)."""
    layers = decode_params(flat)
    total_reward = 0
    # behavior: final x,y positions averaged
    behavior = np.zeros(2)
    
    for ep in range(n_episodes):
        env = gym.make(ENV_NAME)
        obs, _ = env.reset(seed=SEED + ep)
        ep_reward = 0
        ep_layers = [(_W.copy(), _b.copy()) for _W, _b in layers]
        
        for step in range(MAX_STEPS):
            if method == "pure" or method == "novelty":
                probs = forward(ep_layers, obs)
            elif method == "hebbian":
                probs, acts = forward_with_activations(ep_layers, obs)
                ep_layers = hebbian_update(ep_layers, acts, HEBBIAN_LR)
            elif method == "reward_hebbian":
                probs, acts = forward_with_activations(ep_layers, obs)
                # gate by recent reward sign
                gate = 1.0 if ep_reward > 0 else 0.1
                ep_layers = hebbian_update(ep_layers, acts, HEBBIAN_LR * gate)
            
            action = np.argmax(probs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        
        total_reward += ep_reward
        behavior += obs[:2]  # x,y position
        env.close()
    
    return total_reward / n_episodes, behavior / n_episodes

# ── Evolution operators ─────────────────────────────────────────────────
def mutate(genome, rate):
    child = genome.copy()
    mask = np.random.random(len(child)) < rate
    child[mask] += np.random.randn(mask.sum()) * rate * 2
    return child

def crossover(p1, p2):
    mask = np.random.random(len(p1)) < 0.5
    child = np.where(mask, p1, p2)
    return child

def tournament_select(pop, fitnesses, k=3):
    idxs = np.random.choice(len(pop), k, replace=False)
    best = idxs[np.argmax(fitnesses[idxs])]
    return pop[best]

def fitness_sharing(fitnesses, pop, sigma=5.0):
    """Simple fitness sharing to encourage diversity."""
    shared = fitnesses.copy()
    for i in range(len(pop)):
        niche_count = 0
        for j in range(len(pop)):
            dist = np.linalg.norm(pop[i][:100] - pop[j][:100])  # compare first 100 params for speed
            if dist < sigma:
                niche_count += 1
        shared[i] = fitnesses[i] / max(niche_count, 1)
    return shared

def compute_novelty(behavior, archive, pop_behaviors, k=NOVELTY_K):
    """Novelty = mean distance to k nearest neighbors in behavior space."""
    all_b = list(archive) + list(pop_behaviors)
    if len(all_b) < k:
        return 0.0
    dists = [np.linalg.norm(behavior - b) for b in all_b]
    dists.sort()
    return np.mean(dists[1:k+1]) if len(dists) > k else np.mean(dists[1:])

# ── Run one method ──────────────────────────────────────────────────────
def run_method(method_name, csv_path):
    print(f"\n{'='*60}")
    print(f"  Running: {method_name}")
    print(f"{'='*60}")
    
    # Initialize population
    pop = [np.random.randn(N_PARAMS) * 0.1 for _ in range(POP_SIZE)]
    mutation_rate = MUTATION_RATE_INIT
    novelty_archive = []
    best_ever_fitness = -999
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['generation', 'best_fitness', 'mean_fitness', 'std_fitness', 'mutation_rate'])
    
    method_key = {
        "Pure Evolution": "pure",
        "Evo+Hebbian": "hebbian",
        "Evo+Reward-Hebbian": "reward_hebbian",
        "Novelty+Evolution": "novelty",
    }[method_name]
    
    total_evals = 0
    start_time = time.time()
    
    for gen in range(GENERATIONS):
        # Evaluate all
        fitnesses = np.zeros(POP_SIZE)
        behaviors = []
        for i in range(POP_SIZE):
            fit, beh = evaluate(pop[i], method=method_key)
            fitnesses[i] = fit
            behaviors.append(beh)
        total_evals += POP_SIZE * EVAL_EPISODES
        
        # For novelty search, blend fitness with novelty
        if method_key == "novelty":
            novelties = np.array([
                compute_novelty(b, novelty_archive, behaviors) for b in behaviors
            ])
            # Normalize both
            f_range = fitnesses.max() - fitnesses.min()
            n_range = novelties.max() - novelties.min()
            f_norm = (fitnesses - fitnesses.min()) / (f_range + 1e-8)
            n_norm = (novelties - novelties.min()) / (n_range + 1e-8)
            selection_scores = f_norm * (1 - NOVELTY_WEIGHT) + n_norm * NOVELTY_WEIGHT
            # Add top behaviors to archive
            top_idxs = np.argsort(novelties)[-5:]
            for idx in top_idxs:
                novelty_archive.append(behaviors[idx])
            if len(novelty_archive) > 500:
                novelty_archive = novelty_archive[-500:]
        else:
            selection_scores = fitnesses.copy()
        
        # Fitness sharing (lightweight)
        if gen % 10 == 0:
            selection_scores = fitness_sharing(selection_scores, pop)
        
        best_fit = fitnesses.max()
        mean_fit = fitnesses.mean()
        std_fit = fitnesses.std()
        best_idx = np.argmax(fitnesses)
        
        if best_fit > best_ever_fitness:
            best_ever_fitness = best_fit
        
        # Adaptive mutation
        if best_fit > 0:
            mutation_rate = max(MUTATION_RATE_MIN, MUTATION_RATE_INIT * (1 - best_fit / 300))
        
        # Log
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([gen, best_fit, mean_fit, std_fit, mutation_rate])
        
        elapsed = time.time() - start_time
        if gen % 5 == 0:
            print(f"  Gen {gen:3d} | Best: {best_fit:8.1f} | Mean: {mean_fit:8.1f} | "
                  f"Mut: {mutation_rate:.4f} | Time: {elapsed:.0f}s")
        
        if best_ever_fitness >= 200:
            print(f"  *** SOLVED at generation {gen}! ***")
        
        # Selection + reproduction
        new_pop = []
        # Elitism
        elite_idxs = np.argsort(fitnesses)[-ELITE_COUNT:]
        for idx in elite_idxs:
            new_pop.append(pop[idx].copy())
        
        # Fill rest
        while len(new_pop) < POP_SIZE:
            p1 = tournament_select(pop, selection_scores)
            p2 = tournament_select(pop, selection_scores)
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate)
            new_pop.append(child)
        
        pop = new_pop
    
    elapsed = time.time() - start_time
    print(f"  Done. Best ever: {best_ever_fitness:.1f} | Total evals: {total_evals} | Time: {elapsed:.0f}s")
    return best_ever_fitness, total_evals, elapsed

# ── Main ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    
    methods = [
        ("Pure Evolution", "results/pure_evo.csv"),
        ("Evo+Hebbian", "results/evo_hebbian.csv"),
        ("Evo+Reward-Hebbian", "results/evo_reward_hebbian.csv"),
        ("Novelty+Evolution", "results/novelty_evo.csv"),
    ]
    
    results = {}
    for name, csv_path in methods:
        best, evals, elapsed = run_method(name, csv_path)
        results[name] = {"best": best, "evals": evals, "time": elapsed}
    
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    for name, r in results.items():
        solved = "✓ SOLVED" if r["best"] >= 200 else "✗ Not solved"
        print(f"  {name:25s} | Best: {r['best']:8.1f} | Evals: {r['evals']:,} | {solved}")
