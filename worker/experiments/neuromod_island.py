"""
Neuromodulated Island Model — The GAIA Vision.

Combines two key innovations:
1. Neuromodulatory plasticity (biologically plausible within-episode learning)
2. Island Model (decentralized population evolution)

Works on any environment.
"""

import numpy as np
import gymnasium as gym
import time
import multiprocessing as mp
import os

from experiments.neuromod import NeuromodPolicy, evaluate_neuromod
from experiments.cma_es import CMAES


def run(params=None, device="cpu", callback=None):
    """Run Neuromod Island Model on any environment."""
    params = params or {}
    env_name = params.get("environment", "LunarLander-v3")
    max_evals = params.get("max_evals", 200000)
    n_islands = params.get("n_islands", 4)
    migration_interval = params.get("migration_interval", 10)
    eval_episodes = params.get("eval_episodes", 5)
    obs_dim = params.get("obs_dim", 8)
    act_dim = params.get("act_dim", 4)
    act_type = params.get("act_type", "discrete")
    hidden = params.get("hidden", [64, 32])
    max_steps = params.get("max_steps", 1000)
    solved_threshold = params.get("solved_threshold", 200)
    plateau_patience = params.get("plateau_patience", 300)
    n_workers = params.get("n_workers", min(os.cpu_count() or 1, 16))

    policy = NeuromodPolicy(obs_dim=obs_dim, act_dim=act_dim, act_type=act_type, hidden=hidden)

    configs = [
        {"sigma0": 0.3}, {"sigma0": 0.5}, {"sigma0": 0.8}, {"sigma0": 1.2},
    ]
    islands = [CMAES(policy.n_params, sigma0=configs[i % len(configs)]["sigma0"]) for i in range(n_islands)]

    print(f"🧬🏝️ Neuromod Island on {env_name}: {n_islands} islands")
    print(f"Params: {policy.n_params} (main: {policy.n_main}, mod: {policy.n_mod}) | Budget: {max_evals:,}")

    best_ever = -float("inf")
    best_params = None
    total_evals = 0
    total_migrations = 0
    start_time = time.time()
    gen = 0
    plateau_counter = 0
    last_best = -float("inf")

    while total_evals < max_evals:
        gen += 1
        all_candidates = []
        island_sizes = []
        for island in islands:
            candidates = island.ask()
            all_candidates.extend([(policy, c, env_name, eval_episodes, max_steps) for c in candidates])
            island_sizes.append(len(candidates))

        if n_workers > 1:
            with mp.Pool(n_workers) as pool:
                all_fitnesses = pool.starmap(evaluate_neuromod, all_candidates)
            all_fitnesses = np.array(all_fitnesses)
        else:
            all_fitnesses = np.array([evaluate_neuromod(*c) for c in all_candidates])
        total_evals += len(all_candidates) * eval_episodes

        idx = 0
        island_bests = []
        for i, island in enumerate(islands):
            size = island_sizes[i]
            candidates = [c for _, c, *_ in all_candidates[idx:idx + size]]
            fitnesses = all_fitnesses[idx:idx + size]
            island.tell(candidates, fitnesses)
            island_bests.append(np.max(fitnesses))
            if np.max(fitnesses) > best_ever:
                best_ever = float(np.max(fitnesses))
                best_params = candidates[np.argmax(fitnesses)].copy()
            idx += size

        # Ring migration
        if gen % migration_interval == 0:
            for i in range(n_islands):
                target = (i + 1) % n_islands
                alpha = 0.2
                islands[target].mean = (1 - alpha) * islands[target].mean + alpha * islands[i].mean
                total_migrations += 1

        # Plateau detection
        if best_ever > last_best + 1.0:
            last_best = best_ever
            plateau_counter = 0
        else:
            plateau_counter += 1

        elapsed = time.time() - start_time
        avg_sigma = np.mean([isl.sigma for isl in islands])
        entry = {
            "generation": gen,
            "best_fitness": float(max(island_bests)),
            "best_ever": float(best_ever),
            "mean_fitness": float(np.mean(all_fitnesses)),
            "std_fitness": float(np.std(all_fitnesses)),
            "sigma": float(avg_sigma),
            "total_evals": total_evals,
            "elapsed": round(elapsed, 1),
        }
        if callback:
            callback(entry)

        solved = "✅ SOLVED!" if best_ever >= solved_threshold else ""
        print(f"Gen {gen:4d} | Best: {max(island_bests):8.1f} | Ever: {best_ever:8.1f} | "
              f"Mean: {np.mean(all_fitnesses):8.1f} | σ̄: {avg_sigma:.4f} | Evals: {total_evals:>8,} | {elapsed:6.1f}s {solved}")

        if best_ever >= solved_threshold:
            print(f"\n🎉 SOLVED {env_name} with Neuromod Islands! Score: {best_ever:.1f}")
            break
        if plateau_counter >= plateau_patience:
            print(f"\n⏹️ Plateau after {plateau_patience} gens without improvement.")
            break

    if best_params is not None:
        final_scores = [evaluate_neuromod(policy, best_params, env_name, 1, max_steps) for _ in range(20)]
        final_mean, final_std = float(np.mean(final_scores)), float(np.std(final_scores))
    else:
        final_mean = final_std = 0.0

    return {
        "method": "Neuromod Island",
        "environment": env_name,
        "best_ever": float(best_ever),
        "final_mean": final_mean,
        "final_std": final_std,
        "total_evals": total_evals,
        "generations": gen,
        "n_params": policy.n_params,
        "n_islands": n_islands,
        "total_migrations": total_migrations,
        "elapsed_seconds": round(time.time() - start_time, 1),
        "solved": best_ever >= solved_threshold,
    }
