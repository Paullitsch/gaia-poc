"""
OpenAI Evolution Strategies — Environment-agnostic.

Estimates gradients via finite differences. Based on Salimans et al. (2017).
"""

import numpy as np
import multiprocessing as mp
import os
import gymnasium as gym
import time

from experiments.cma_es import PolicyNetwork, evaluate


def run(params=None, device="cpu", callback=None):
    """Run OpenAI-ES on any environment."""
    params = params or {}
    env_name = params.get("environment", "LunarLander-v3")
    max_evals = params.get("max_evals", 100000)
    pop_size = params.get("pop_size", 50)
    lr = params.get("lr", 0.01)
    noise_std = params.get("noise_std", 0.02)
    eval_episodes = params.get("eval_episodes", 5)
    obs_dim = params.get("obs_dim", 8)
    act_dim = params.get("act_dim", 4)
    act_type = params.get("act_type", "discrete")
    hidden = params.get("hidden", [64, 32])
    max_steps = params.get("max_steps", 1000)
    solved_threshold = params.get("solved_threshold", 200)
    n_workers = params.get("n_workers", min(os.cpu_count() or 1, 16))

    obs_type = params.get("obs_type", "vector")
    is_atari = obs_type == "pixel"

    if is_atari:
        from experiments.atari_eval import AtariCNN, evaluate_atari, evaluate_population_gpu
        n_frames = params.get("n_frames", 4)
        _model = AtariCNN(n_frames=n_frames, n_actions=act_dim)
        n_params = _model.n_params
        print(f"🎮 OpenAI-ES on {env_name} (CNN, {n_params} params)")
        def _eval(params_vec):
            return evaluate_atari(params_vec, env_name, act_dim, eval_episodes, max_steps, device)
        def _eval_batch(params_list):
            return evaluate_population_gpu(params_list, env_name, act_dim,
                                           eval_episodes, max_steps, device, n_parallel=pop_size)
    else:
        policy = PolicyNetwork(obs_dim=obs_dim, act_dim=act_dim, act_type=act_type, hidden=hidden)
        n_params = n_params
        print(f"🔀 OpenAI-ES on {env_name}: {n_params} params")
        _eval = None  # use multiprocessing path

    theta = np.random.randn(n_params) * 0.1
    print(f"Pop: {pop_size} | lr: {lr} | σ: {noise_std} | Workers: {n_workers}")

    best_ever = -float("inf")
    best_params = None
    total_evals = 0
    start_time = time.time()
    gen = 0

    while total_evals < max_evals:
        noise = np.random.randn(pop_size, n_params)

        if is_atari:
            # Batch evaluate entire population on GPU with vectorized envs
            all_params = []
            for i in range(pop_size):
                all_params.append(theta + noise_std * noise[i])
                all_params.append(theta - noise_std * noise[i])
            all_fitnesses = _eval_batch(all_params)
            rewards_pos = np.array([all_fitnesses[2*i] for i in range(pop_size)])
            rewards_neg = np.array([all_fitnesses[2*i+1] for i in range(pop_size)])
        elif n_workers > 1:
            args = []
            for i in range(pop_size):
                args.append((policy, theta + noise_std * noise[i], env_name, eval_episodes, max_steps))
                args.append((policy, theta - noise_std * noise[i], env_name, eval_episodes, max_steps))
            with mp.Pool(n_workers) as pool:
                results = pool.starmap(evaluate, args)
            rewards_pos = np.array([results[2*i] for i in range(pop_size)])
            rewards_neg = np.array([results[2*i+1] for i in range(pop_size)])
        else:
            rewards_pos = np.array([evaluate(policy, theta + noise_std * noise[i], env_name, eval_episodes, max_steps) for i in range(pop_size)])
            rewards_neg = np.array([evaluate(policy, theta - noise_std * noise[i], env_name, eval_episodes, max_steps) for i in range(pop_size)])
        total_evals += 2 * pop_size * eval_episodes

        all_rewards = np.concatenate([rewards_pos, rewards_neg])
        ranks = np.zeros_like(all_rewards)
        sorted_idx = np.argsort(all_rewards)
        for i, idx in enumerate(sorted_idx):
            ranks[idx] = i
        ranks = ranks / (2 * pop_size - 1) - 0.5

        gradient = np.zeros(n_params)
        for i in range(pop_size):
            gradient += ranks[i] * noise[i]
            gradient -= ranks[pop_size + i] * noise[i]
        gradient /= (2 * pop_size * noise_std)
        theta += lr * gradient

        gen_best = float(np.max(all_rewards))
        if gen_best > best_ever:
            best_ever = gen_best
            best_idx = np.argmax(all_rewards)
            if best_idx < pop_size:
                best_params = (theta + noise_std * noise[best_idx]).copy()
            else:
                best_params = (theta - noise_std * noise[best_idx - pop_size]).copy()

        elapsed = time.time() - start_time
        gen += 1

        entry = {
            "generation": gen,
            "best_fitness": gen_best,
            "best_ever": float(best_ever),
            "mean_fitness": float(np.mean(all_rewards)),
            "std_fitness": float(np.std(all_rewards)),
            "total_evals": total_evals,
            "elapsed": round(elapsed, 1),
        }
        if callback:
            callback(entry)

        solved = "✅ SOLVED!" if best_ever >= solved_threshold else ""
        print(f"Gen {gen:4d} | Best: {gen_best:8.1f} | Ever: {best_ever:8.1f} | "
              f"Mean: {np.mean(all_rewards):8.1f} | Evals: {total_evals:>8,} | {elapsed:6.1f}s {solved}")

        if best_ever >= solved_threshold:
            print(f"\n🎉 SOLVED {env_name} with OpenAI-ES! Score: {best_ever:.1f}")
            break

    if best_params is not None:
        if is_atari:
            final_scores = evaluate_population_gpu(
                [best_params] * 10, env_name, act_dim, 1, max_steps, device, n_parallel=10)
        else:
            final_scores = [evaluate(policy, best_params, env_name, 1, max_steps) for _ in range(20)]
        final_mean, final_std = float(np.mean(final_scores)), float(np.std(final_scores))
    else:
        final_mean = final_std = 0.0

    return {
        "method": "OpenAI-ES",
        "environment": env_name,
        "best_ever": float(best_ever),
        "final_mean": final_mean,
        "final_std": final_std,
        "total_evals": total_evals,
        "generations": gen,
        "n_params": n_params,
        "elapsed_seconds": round(time.time() - start_time, 1),
        "solved": best_ever >= solved_threshold,
    }
