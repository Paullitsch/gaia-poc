"""
Network Scaling Experiment.

Tests CMA-ES performance across different network sizes.
Key question: At what parameter count does CMA-ES degrade?

Environment-agnostic.
"""

import numpy as np
import gymnasium as gym
import time
import multiprocessing as mp
import os
import json

from experiments.cma_es import PolicyNetwork, evaluate, CMAES


def run(params=None, device="cpu", callback=None):
    params = params or {}
    env_name = params.get("environment", "LunarLander-v3")
    config_name = params.get("config", "medium")
    max_evals = params.get("max_evals", 100000)
    eval_episodes = params.get("eval_episodes", 5)
    obs_dim = params.get("obs_dim", 8)
    act_dim = params.get("act_dim", 4)
    act_type = params.get("act_type", "discrete")
    max_steps = params.get("max_steps", 1000)
    solved_threshold = params.get("solved_threshold", 200)
    n_workers = params.get("n_workers", min(os.cpu_count() or 1, 16))

    # Network size configurations
    CONFIGS = {
        "tiny":   [32, 16],
        "small":  [64, 32],
        "medium": [128, 64],
        "large":  [256, 128],
        "xl":     [512, 256],
    }

    hidden = params.get("hidden", CONFIGS.get(config_name, [128, 64]))

    policy = PolicyNetwork(obs_dim, act_dim, hidden)
    label = "→".join(map(str, hidden))
    print(f"📐 Scaling ({label}) on {env_name}: {policy.n_params} params")
    print(f"Budget: {max_evals} evals | Workers: {n_workers}")

    cma = CMAES(policy.n_params, sigma0=0.5)
    print(f"CMA-ES pop: {cma.lam} | Diagonal: {cma.use_diagonal}")

    best_ever = -float("inf")
    best_params = None
    total_evals = 0
    start_time = time.time()

    while total_evals < max_evals:
        candidates = cma.ask()
        args = [(policy, c, env_name, eval_episodes, max_steps) for c in candidates]

        if n_workers > 1:
            with mp.Pool(n_workers) as pool:
                fitnesses = pool.starmap(evaluate, args)
            fitnesses = np.array(fitnesses)
        else:
            fitnesses = np.array([evaluate(*a) for a in args])

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
            "sigma": float(cma.sigma),
            "total_evals": total_evals,
            "elapsed": round(elapsed, 1),
        }
        if callback:
            callback(entry)

        solved_str = "✅ SOLVED!" if best_ever >= solved_threshold else ""
        print(f"Gen {cma.gen:4d} | Best: {gen_best:8.1f} | Ever: {best_ever:8.1f} | "
              f"Mean: {np.mean(fitnesses):8.1f} | σ: {cma.sigma:.4f} | Evals: {total_evals:6d} | "
              f"{elapsed:6.1f}s {solved_str}")

        if best_ever >= solved_threshold:
            break

    if best_params is not None:
        final_scores = [evaluate(policy, best_params, env_name, 1, max_steps) for _ in range(20)]
        final_mean, final_std = float(np.mean(final_scores)), float(np.std(final_scores))
    else:
        final_mean = final_std = 0.0

    return {
        "method": f"Scaling-{label}",
        "environment": env_name,
        "config": config_name,
        "hidden_sizes": hidden,
        "n_params": policy.n_params,
        "best_ever": float(best_ever),
        "final_mean": final_mean,
        "final_std": final_std,
        "total_evals": total_evals,
        "generations": cma.gen,
        "elapsed_seconds": round(time.time() - start_time, 1),
        "solved": best_ever >= solved_threshold,
        "diagonal_cma": cma.use_diagonal,
    }
