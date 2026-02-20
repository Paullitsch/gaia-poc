"""
CMA-ES with Reward Shaping + Curriculum Learning.

Shaped rewards guide the agent with intermediate signals.
Curriculum starts easy, gets harder — like biological learning.

Environment-agnostic: works on any Gymnasium environment.
"""

import numpy as np
import gymnasium as gym
import time
import json
import multiprocessing as mp
import os

from experiments.cma_es import PolicyNetwork, evaluate, CMAES


def _shaped_reward_lunar(obs, next_obs, reward, difficulty):
    """LunarLander-specific reward shaping."""
    shaped = reward
    shaped += (1.0 - difficulty) * 0.1  # survival bonus

    height = next_obs[1]
    vel_y = next_obs[3]
    vel_x = next_obs[2]
    angle = abs(next_obs[4])

    if height < 0.5:
        shaped += max(0, -vel_y * 0.5) * difficulty
        shaped += max(0, 0.5 - abs(vel_x)) * difficulty
        shaped += max(0, 0.3 - angle) * difficulty
    return shaped


def _shaped_reward_bipedal(obs, next_obs, reward, difficulty):
    """BipedalWalker-specific reward shaping."""
    shaped = reward
    # Survival bonus early on
    shaped += (1.0 - difficulty) * 0.2
    # Reward forward velocity
    hull_vel_x = next_obs[2] if len(next_obs) > 2 else 0
    shaped += max(0, hull_vel_x) * 0.1 * difficulty
    # Penalize excessive hull angle
    hull_angle = abs(next_obs[0]) if len(next_obs) > 0 else 0
    shaped -= hull_angle * 0.1 * difficulty
    return shaped


def _shaped_reward_generic(obs, next_obs, reward, difficulty):
    """Fallback: just use raw reward + small survival bonus."""
    return reward + (1.0 - difficulty) * 0.1


def _get_shaper(env_name):
    if "LunarLander" in env_name:
        return _shaped_reward_lunar
    elif "BipedalWalker" in env_name:
        return _shaped_reward_bipedal
    return _shaped_reward_generic


def evaluate_shaped(policy, params, env_name, n_episodes, difficulty, max_steps):
    """Evaluate with shaped rewards on any environment."""
    env = gym.make(env_name)
    shaper = _get_shaper(env_name)
    total = 0.0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 1000)
        done = False
        ep_reward = 0.0
        steps = 0

        while not done and steps < max_steps:
            action = policy.act(obs, params)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += shaper(obs, next_obs, reward, difficulty)
            obs = next_obs
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
    n_workers = params.get("n_workers", min(os.cpu_count() or 1, 16))

    policy = PolicyNetwork(obs_dim, act_dim, hidden)
    print(f"📐 Curriculum CMA-ES on {env_name}: {policy.n_params} params")
    print(f"Network: {obs_dim}→{'→'.join(map(str, hidden))}→{act_dim} ({act_type})")

    cma = CMAES(policy.n_params, sigma0=0.5)

    best_ever = -float("inf")
    best_ever_raw = -float("inf")
    best_params = None
    total_evals = 0
    start_time = time.time()

    max_gens = max_evals // (cma.lam * eval_episodes)

    while total_evals < max_evals:
        progress = min(1.0, cma.gen / max(max_gens * 0.7, 1))
        difficulty = 0.3 + 0.7 * progress

        candidates = cma.ask()
        args = [(policy, c, env_name, eval_episodes, difficulty, max_steps) for c in candidates]

        if n_workers > 1:
            with mp.Pool(n_workers) as pool:
                fitnesses = pool.starmap(evaluate_shaped, args)
            fitnesses = np.array(fitnesses)
        else:
            fitnesses = np.array([evaluate_shaped(*a) for a in args])

        total_evals += len(candidates) * eval_episodes
        cma.tell(candidates, fitnesses)

        gen_best = np.max(fitnesses)
        if gen_best > best_ever:
            best_ever = gen_best
            best_params = candidates[np.argmax(fitnesses)].copy()

        elapsed = time.time() - start_time

        # Evaluate best on raw reward periodically
        raw_best = -999
        if cma.gen % 10 == 0 and best_params is not None:
            raw_best = evaluate(policy, best_params, env_name, 5, max_steps)
            if raw_best > best_ever_raw:
                best_ever_raw = raw_best

        entry = {
            "generation": cma.gen,
            "best_fitness": float(gen_best),
            "best_ever": float(best_ever),
            "best_ever_raw": float(best_ever_raw),
            "mean_fitness": float(np.mean(fitnesses)),
            "std_fitness": float(np.std(fitnesses)),
            "difficulty": round(difficulty, 3),
            "total_evals": total_evals,
            "elapsed": round(elapsed, 1),
        }
        if callback:
            callback(entry)

        raw_str = f" | Raw: {raw_best:8.1f}" if raw_best > -999 else ""
        solved_str = "✅ SOLVED!" if best_ever_raw >= solved_threshold else ""
        print(f"Gen {cma.gen:4d} | Best: {gen_best:8.1f} | Ever: {best_ever:8.1f} | "
              f"Mean: {np.mean(fitnesses):8.1f} | σ: {cma.sigma:.4f} | "
              f"Diff: {difficulty:.2f} | Evals: {total_evals:6d} | {elapsed:6.1f}s {solved_str}")

        if best_ever_raw >= solved_threshold:
            break

    # Final raw evaluation
    if best_params is not None:
        final_scores = [evaluate(policy, best_params, env_name, 1, max_steps) for _ in range(20)]
        final_mean = float(np.mean(final_scores))
        final_std = float(np.std(final_scores))
    else:
        final_mean = final_std = 0.0

    return {
        "method": "Curriculum-CMA-ES",
        "environment": env_name,
        "best_ever_shaped": float(best_ever),
        "best_ever_raw": float(best_ever_raw),
        "final_mean": final_mean,
        "final_std": final_std,
        "total_evals": total_evals,
        "generations": cma.gen,
        "n_params": policy.n_params,
        "elapsed_seconds": round(time.time() - start_time, 1),
        "solved": best_ever_raw >= solved_threshold,
    }


if __name__ == "__main__":
    result = run(params={"max_evals": 50000})
    print(f"\nResult: {json.dumps(result, indent=2)}")
