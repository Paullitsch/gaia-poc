"""
Network Scaling Experiment — Phase 8.

Tests CMA-ES performance across different network sizes on LunarLander.
Key question: At what parameter count does CMA-ES degrade?

Theory predicts CMA-ES scales as O(n^2) in memory and O(n) per generation,
so we expect degradation around 10K-50K parameters. OpenAI-ES (O(n)) should
scale better for larger networks.
"""

import numpy as np
import gymnasium as gym
import time
import multiprocessing as mp
import os
import json


class ScalablePolicy:
    """Policy network with configurable hidden sizes."""
    def __init__(self, obs_dim=8, act_dim=4, hidden_sizes=(64, 32)):
        self.layers = []
        dims = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.shapes = []
        self.sizes = []
        for i in range(len(dims) - 1):
            # Weight matrix
            self.shapes.append((dims[i], dims[i+1]))
            self.sizes.append(dims[i] * dims[i+1])
            # Bias
            self.shapes.append((dims[i+1],))
            self.sizes.append(dims[i+1])
        self.n_params = sum(self.sizes)

    def act(self, obs, params):
        idx = 0
        x = obs
        for i in range(0, len(self.shapes), 2):
            w_shape = self.shapes[i]
            w_size = self.sizes[i]
            b_size = self.sizes[i+1]
            w = params[idx:idx+w_size].reshape(w_shape)
            idx += w_size
            b = params[idx:idx+b_size]
            idx += b_size
            x = x @ w + b
            # tanh for hidden layers, raw for output
            if i < len(self.shapes) - 2:
                x = np.tanh(x)
        return int(np.argmax(x))


def evaluate(policy, params, n_episodes=5):
    env = gym.make("LunarLander-v3")
    total = 0.0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 1000)
        done = False
        ep_reward = 0.0
        steps = 0
        while not done and steps < 1000:
            action = policy.act(obs, params)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            steps += 1
        total += ep_reward
    env.close()
    return total / n_episodes


# Network size configurations to test
CONFIGS = {
    "tiny":   {"hidden_sizes": (32, 16),   "label": "32-16"},
    "small":  {"hidden_sizes": (64, 32),   "label": "64-32"},
    "medium": {"hidden_sizes": (128, 64),  "label": "128-64"},
    "large":  {"hidden_sizes": (256, 128), "label": "256-128"},
    "xl":     {"hidden_sizes": (512, 256), "label": "512-256"},
}


def run(params=None, device="cpu", callback=None):
    """Run scaling experiment.
    
    params.config: which size config to use (tiny/small/medium/large/xl)
    """
    params = params or {}
    config_name = params.get("config", "medium")
    max_evals = params.get("max_evals", 100000)
    eval_episodes = params.get("eval_episodes", 5)
    n_workers = params.get("n_workers", min(os.cpu_count() or 1, 16))

    if config_name not in CONFIGS:
        print(f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}")
        config_name = "medium"

    cfg = CONFIGS[config_name]
    hidden_sizes = cfg["hidden_sizes"]

    policy = ScalablePolicy(hidden_sizes=hidden_sizes)
    print(f"Scaling Experiment: {cfg['label']} — {policy.n_params} parameters")
    print(f"Budget: {max_evals} evaluations")
    print(f"Workers: {n_workers}")

    from experiments.cma_es import CMAES
    cma = CMAES(policy.n_params, sigma0=0.5)
    print(f"CMA-ES population: {cma.lam}")
    print(f"CMA-ES diagonal mode: {cma.use_diagonal}")

    best_ever = -float("inf")
    best_params = None
    total_evals = 0
    start_time = time.time()

    while total_evals < max_evals:
        candidates = cma.ask()

        if n_workers > 1:
            with mp.Pool(n_workers) as pool:
                fitnesses = pool.starmap(evaluate, [(policy, c, eval_episodes) for c in candidates])
            fitnesses = np.array(fitnesses)
        else:
            fitnesses = np.array([evaluate(policy, c, eval_episodes) for c in candidates])

        total_evals += len(candidates) * eval_episodes
        cma.tell(candidates, fitnesses)

        gen_best = np.max(fitnesses)
        gen_mean = np.mean(fitnesses)
        if gen_best > best_ever:
            best_ever = gen_best
            best_params = candidates[np.argmax(fitnesses)].copy()

        elapsed = time.time() - start_time

        entry = {
            "generation": cma.gen,
            "best_fitness": float(gen_best),
            "best_ever": float(best_ever),
            "mean_fitness": float(gen_mean),
            "sigma": float(cma.sigma),
            "total_evals": total_evals,
            "elapsed": round(elapsed, 1),
        }
        if callback:
            callback(entry)

        solved_str = "✅ SOLVED!" if best_ever >= 200 else ""
        print(f"Gen {cma.gen:4d} | Best: {gen_best:8.1f} | Ever: {best_ever:8.1f} | "
              f"Mean: {gen_mean:8.1f} | σ: {cma.sigma:.4f} | Evals: {total_evals:6d} | "
              f"{elapsed:6.1f}s {solved_str}")

        if best_ever >= 200:
            print(f"\n🎉 SOLVED with {cfg['label']} network ({policy.n_params} params)!")
            break

    # Final evaluation
    if best_params is not None:
        final_scores = [evaluate(policy, best_params, 1) for _ in range(20)]
        final_mean = float(np.mean(final_scores))
        final_std = float(np.std(final_scores))
    else:
        final_mean = final_std = 0.0

    return {
        "method": f"Scaling-{cfg['label']}",
        "environment": "LunarLander-v3",
        "config": config_name,
        "hidden_sizes": list(hidden_sizes),
        "n_params": policy.n_params,
        "best_ever": float(best_ever),
        "final_mean": final_mean,
        "final_std": final_std,
        "total_evals": total_evals,
        "generations": cma.gen,
        "elapsed_seconds": round(time.time() - start_time, 1),
        "solved": best_ever >= 200,
        "diagonal_cma": cma.use_diagonal,
    }


if __name__ == "__main__":
    for config in ["tiny", "small", "medium", "large"]:
        result = run(params={"config": config, "max_evals": 50000})
        print(f"\n{config}: {json.dumps(result, indent=2)}\n")
