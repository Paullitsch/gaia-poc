"""
Indirect Encoding — A small 'genome network' GENERATES the weights of the policy.

Inspired by biology: DNA doesn't store every synapse weight.
It stores a developmental program that builds the brain.

CMA-ES optimizes the genome network (~800 params).
The genome network generates the policy (~3000 params).
Massive search space compression.

Environment-agnostic.
"""

import numpy as np
import gymnasium as gym
import time
import json
import os

from experiments.cma_es import CMAES


class GenomeNetwork:
    """Small network that generates policy weights."""

    def __init__(self, obs_dim=8, act_dim=4, hidden=None, act_type="discrete"):
        hidden = hidden or [64, 32]
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_type = act_type

        # Target weight sizes (the policy we're generating)
        self.target_shapes = []
        self.target_sizes = []
        dims = [obs_dim] + hidden + [act_dim]
        for i in range(len(dims) - 1):
            self.target_shapes.append((dims[i], dims[i+1]))
            self.target_sizes.append(dims[i] * dims[i+1])
            self.target_shapes.append((dims[i+1],))
            self.target_sizes.append(dims[i+1])
        self.total_target = sum(self.target_sizes)

        # Genome network: maps position encoding → weight value
        # Input: (layer_idx, row_frac, col_frac) = 3
        # Hidden: 32 → 16
        # Output: 1 (weight value)
        self.genome_shapes = [
            (3, 32), (32,),
            (32, 16), (16,),
            (16, 1), (1,),
        ]
        self.genome_sizes = [np.prod(s) for s in self.genome_shapes]
        self.n_params = sum(self.genome_sizes)

    def generate_weights(self, genome):
        """Use genome network to generate policy weights."""
        idx = 0
        g_layers = []
        for shape, size in zip(self.genome_shapes, self.genome_sizes):
            g_layers.append(genome[idx:idx + size].reshape(shape))
            idx += size

        all_weights = np.zeros(self.total_target)
        w_idx = 0

        for layer_i, (shape, size) in enumerate(zip(self.target_shapes, self.target_sizes)):
            layer_frac = layer_i / max(len(self.target_shapes) - 1, 1)

            if len(shape) == 2:
                rows, cols = shape
                for r in range(rows):
                    for c in range(cols):
                        inp = np.array([layer_frac, r / max(rows - 1, 1), c / max(cols - 1, 1)])
                        h = np.tanh(inp @ g_layers[0] + g_layers[1])
                        h = np.tanh(h @ g_layers[2] + g_layers[3])
                        val = (h @ g_layers[4] + g_layers[5])[0]
                        all_weights[w_idx] = val
                        w_idx += 1
            else:
                for b in range(shape[0]):
                    inp = np.array([layer_frac, b / max(shape[0] - 1, 1), 0.5])
                    h = np.tanh(inp @ g_layers[0] + g_layers[1])
                    h = np.tanh(h @ g_layers[2] + g_layers[3])
                    val = (h @ g_layers[4] + g_layers[5])[0]
                    all_weights[w_idx] = val
                    w_idx += 1

        return all_weights

    def act(self, obs, policy_weights):
        idx = 0
        x = obs
        for i in range(0, len(self.target_shapes), 2):
            w = policy_weights[idx:idx + self.target_sizes[i]].reshape(self.target_shapes[i])
            idx += self.target_sizes[i]
            b = policy_weights[idx:idx + self.target_sizes[i+1]]
            idx += self.target_sizes[i+1]
            x = x @ w + b
            if i < len(self.target_shapes) - 2:
                x = np.tanh(x)
        if self.act_type == "discrete":
            return int(np.argmax(x))
        return np.clip(np.tanh(x), -1, 1)


def evaluate(genome_net, genome, env_name, n_episodes=5, max_steps=1000):
    policy_weights = genome_net.generate_weights(genome)
    env = gym.make(env_name)
    total = 0.0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 1000)
        done, ep_reward, steps = False, 0.0, 0
        while not done and steps < max_steps:
            action = genome_net.act(obs, policy_weights)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            steps += 1
        total += ep_reward
    env.close()
    return total / n_episodes


def run(params=None, device="cpu", callback=None):
    params = params or {}
    env_name = params.get("environment", "LunarLander-v3")
    max_evals = params.get("max_evals", 100000)
    eval_episodes = params.get("eval_episodes", 5)
    obs_dim = params.get("obs_dim", 8)
    act_dim = params.get("act_dim", 4)
    act_type = params.get("act_type", "discrete")
    hidden = params.get("hidden", [64, 32])
    max_steps = params.get("max_steps", 1000)
    solved_threshold = params.get("solved_threshold", 200)

    genome_net = GenomeNetwork(obs_dim, act_dim, hidden, act_type)
    print(f"📐 Indirect Encoding on {env_name}: {genome_net.n_params} genome → {genome_net.total_target} policy params")
    print(f"Compression: {genome_net.total_target / genome_net.n_params:.1f}x")

    cma = CMAES(genome_net.n_params, sigma0=1.0)

    best_ever = -float("inf")
    best_params = None
    total_evals = 0
    start_time = time.time()

    while total_evals < max_evals:
        candidates = cma.ask()
        fitnesses = np.array([evaluate(genome_net, c, env_name, eval_episodes, max_steps) for c in candidates])
        total_evals += len(candidates) * eval_episodes

        cma.tell(candidates, fitnesses)

        gen_best = np.max(fitnesses)
        if gen_best > best_ever:
            best_ever = gen_best
            best_params = candidates[np.argmax(fitnesses)].copy()

        elapsed = time.time() - start_time
        entry = {
            "generation": cma.gen,
            "best_fitness": float(gen_best),
            "best_ever": float(best_ever),
            "mean_fitness": float(np.mean(fitnesses)),
            "std_fitness": float(np.std(fitnesses)),
            "sigma": float(cma.sigma),
            "total_evals": total_evals,
            "elapsed": round(elapsed, 1),
        }
        if callback:
            callback(entry)

        solved_str = "✅ SOLVED!" if best_ever >= solved_threshold else ""
        print(f"Gen {cma.gen:4d} | Best: {gen_best:8.1f} | Ever: {best_ever:8.1f} | "
              f"Mean: {np.mean(fitnesses):8.1f} | σ: {cma.sigma:.4f} | Evals: {total_evals:6d} | {elapsed:6.1f}s {solved_str}")

        if best_ever >= solved_threshold:
            break

    if best_params is not None:
        final_scores = [evaluate(genome_net, best_params, env_name, 1, max_steps) for _ in range(20)]
        final_mean, final_std = float(np.mean(final_scores)), float(np.std(final_scores))
    else:
        final_mean = final_std = 0.0

    return {
        "method": "Indirect-Encoding",
        "environment": env_name,
        "best_ever": float(best_ever),
        "final_mean": final_mean,
        "final_std": final_std,
        "total_evals": total_evals,
        "generations": cma.gen,
        "genome_params": genome_net.n_params,
        "policy_params": genome_net.total_target,
        "compression": round(genome_net.total_target / genome_net.n_params, 1),
        "elapsed_seconds": round(time.time() - start_time, 1),
        "solved": best_ever >= solved_threshold,
    }
