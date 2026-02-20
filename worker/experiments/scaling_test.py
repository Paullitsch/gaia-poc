"""
Scaling Test: Where do gradient-free methods break?

Systematically tests CMA-ES and OpenAI-ES at increasing network sizes
on the SAME environment. This answers the critical question:
"At what parameter count does gradient-free optimization degrade?"

Configs from 1K to 500K+ parameters.
"""

import numpy as np
import gymnasium as gym
import time
import multiprocessing as mp
import os

from experiments.cma_es import PolicyNetwork, evaluate, CMAES


# Scaling configs: same environment, different network sizes
SCALING_CONFIGS = {
    "1k":   {"hidden": [32, 16],       "label": "1K (32→16)"},
    "3k":   {"hidden": [64, 32],       "label": "3K (64→32)"},
    "10k":  {"hidden": [128, 64],      "label": "10K (128→64)"},
    "33k":  {"hidden": [256, 128],     "label": "33K (256→128)"},
    "100k": {"hidden": [512, 256],     "label": "100K (512→256)"},
    "200k": {"hidden": [512, 256, 128],"label": "200K (512→256→128)"},
    "500k": {"hidden": [1024, 512, 256],"label": "500K (1024→512→256)"},
}


def run(params=None, device="cpu", callback=None):
    """Run scaling test at specific config."""
    params = params or {}
    env_name = params.get("environment", "LunarLander-v3")
    config = params.get("config", "3k")
    max_evals = params.get("max_evals", 100000)
    eval_episodes = params.get("eval_episodes", 5)
    obs_dim = params.get("obs_dim", 8)
    act_dim = params.get("act_dim", 4)
    act_type = params.get("act_type", "discrete")
    max_steps = params.get("max_steps", 1000)
    solved_threshold = params.get("solved_threshold", 200)
    n_workers = params.get("n_workers", min(os.cpu_count() or 1, 16))
    
    cfg = SCALING_CONFIGS.get(config, SCALING_CONFIGS["3k"])
    hidden = params.get("hidden", cfg["hidden"])
    
    policy = PolicyNetwork(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden, act_type=act_type)
    
    print(f"📏 Scaling Test: {cfg['label']} on {env_name}")
    print(f"Network: {obs_dim}→{'→'.join(map(str, hidden))}→{act_dim} = {policy.n_params:,} params")
    
    cma = CMAES(policy.n_params, sigma0=0.5)
    print(f"CMA-ES: pop={cma.lam} | diagonal={cma.use_diagonal} | Workers: {n_workers}")
    print(f"Budget: {max_evals:,} evals")
    
    best_ever = -float("inf")
    best_params = None
    total_evals = 0
    start_time = time.time()
    
    while total_evals < max_evals:
        candidates = cma.ask()
        
        if n_workers > 1:
            with mp.Pool(n_workers) as pool:
                fitnesses = pool.starmap(evaluate, [(policy, c, env_name, eval_episodes, max_steps) for c in candidates])
            fitnesses = np.array(fitnesses)
        else:
            fitnesses = np.array([evaluate(policy, c, env_name, eval_episodes, max_steps) for c in candidates])
        total_evals += len(candidates) * eval_episodes
        
        cma.tell(candidates, fitnesses)
        
        gen_best = float(np.max(fitnesses))
        if gen_best > best_ever:
            best_ever = gen_best
            best_params = candidates[np.argmax(fitnesses)].copy()
        
        elapsed = time.time() - start_time
        
        entry = {
            "generation": cma.gen,
            "best_fitness": gen_best,
            "best_ever": float(best_ever),
            "mean_fitness": float(np.mean(fitnesses)),
            "sigma": float(cma.sigma),
            "total_evals": total_evals,
            "elapsed": round(elapsed, 1),
        }
        if callback:
            callback(entry)
        
        solved = "✅ SOLVED!" if best_ever >= solved_threshold else ""
        print(f"Gen {cma.gen:4d} | Best: {gen_best:8.1f} | Ever: {best_ever:8.1f} | "
              f"Mean: {np.mean(fitnesses):8.1f} | σ: {cma.sigma:.4f} | Evals: {total_evals:>8,} | {elapsed:6.1f}s {solved}")
        
        if best_ever >= solved_threshold:
            break
    
    if best_params is not None:
        final_scores = [evaluate(policy, best_params, env_name, 1, max_steps) for _ in range(20)]
        final_mean, final_std = float(np.mean(final_scores)), float(np.std(final_scores))
    else:
        final_mean = final_std = 0.0
    
    return {
        "method": f"Scaling-{config}",
        "environment": env_name,
        "config": config,
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
