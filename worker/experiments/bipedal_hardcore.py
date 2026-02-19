"""
CMA-ES for BipedalWalkerHardcore-v3 — Phase 8 Stretch Goal.

Hardcore adds: stumps, pitfalls, rough terrain, stairs.
Solved threshold: 300 (same as regular).
Much harder — most RL methods struggle here.

If gradient-free methods can solve Hardcore, it would be
a significant result for the field.
"""

import numpy as np
import gymnasium as gym
import time
import multiprocessing as mp
import os
import json

OBS_DIM = 24
ACT_DIM = 4
SOLVED_THRESHOLD = 300


class PolicyNetwork:
    """Larger network for Hardcore: 24→256→128→4 (37,380 params)."""
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden1=256, hidden2=128):
        self.shapes = [
            (obs_dim, hidden1), (hidden1,),
            (hidden1, hidden2), (hidden2,),
            (hidden2, act_dim), (act_dim,),
        ]
        self.sizes = [np.prod(s) for s in self.shapes]
        self.n_params = sum(self.sizes)

    def act(self, obs, params):
        idx = 0
        layers = []
        for shape, size in zip(self.shapes, self.sizes):
            layers.append(params[idx:idx + size].reshape(shape))
            idx += size
        h = np.tanh(obs @ layers[0] + layers[1])
        h = np.tanh(h @ layers[2] + layers[3])
        return np.tanh(h @ layers[4] + layers[5])


def evaluate(policy, params, n_episodes=3):
    env = gym.make("BipedalWalker-v3", hardcore=True)
    total = 0.0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 1000)
        done = False
        ep_reward = 0.0
        steps = 0
        while not done and steps < 2000:
            action = policy.act(obs, params)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            steps += 1
        total += ep_reward
    env.close()
    return total / n_episodes


def evaluate_shaped(policy, params, n_episodes=3, difficulty=1.0):
    """Shaped rewards for curriculum."""
    env = gym.make("BipedalWalker-v3", hardcore=True)
    total = 0.0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 1000)
        done = False
        ep_reward = 0.0
        steps = 0
        while not done and steps < 2000:
            action = policy.act(obs, params)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            shaped = reward
            hull_x_vel = next_obs[2]
            if hull_x_vel > 0:
                shaped += hull_x_vel * 0.2 * (1.0 - difficulty * 0.5)
            hull_angle = abs(next_obs[0])
            if hull_angle < 0.4:
                shaped += (0.4 - hull_angle) * 0.3 * (1.0 - difficulty * 0.5)
            shaped += (1.0 - difficulty) * 0.03
            ep_reward += shaped
            obs = next_obs
            done = terminated or truncated
            steps += 1
        total += ep_reward
    env.close()
    return total / n_episodes


def run(params=None, device="cpu", callback=None):
    params = params or {}
    max_evals = params.get("max_evals", 1000000)
    eval_episodes = params.get("eval_episodes", 3)
    n_workers = params.get("n_workers", min(os.cpu_count() or 1, 16))

    policy = PolicyNetwork()
    print(f"BipedalWalker Hardcore: {policy.n_params} params")
    print(f"Budget: {max_evals} evals, Workers: {n_workers}")

    from experiments.cma_es import CMAES
    cma = CMAES(policy.n_params, sigma0=0.3)
    print(f"CMA-ES population: {cma.lam}")

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
        if n_workers > 1:
            with mp.Pool(n_workers) as pool:
                fitnesses = pool.starmap(evaluate_shaped, [(policy, c, eval_episodes, difficulty) for c in candidates])
            fitnesses = np.array(fitnesses)
        else:
            fitnesses = np.array([evaluate_shaped(policy, c, eval_episodes, difficulty) for c in candidates])

        total_evals += len(candidates) * eval_episodes
        cma.tell(candidates, fitnesses)

        gen_best = np.max(fitnesses)
        if gen_best > best_ever:
            best_ever = gen_best
            best_params = candidates[np.argmax(fitnesses)].copy()

        elapsed = time.time() - start_time

        raw_best = -999
        if cma.gen % 20 == 0 and best_params is not None:
            raw_best = evaluate(policy, best_params, 5)
            if raw_best > best_ever_raw:
                best_ever_raw = raw_best

        entry = {
            "generation": cma.gen,
            "best_fitness": float(gen_best),
            "best_ever": float(best_ever),
            "best_ever_raw": float(best_ever_raw),
            "mean_fitness": float(np.mean(fitnesses)),
            "sigma": float(cma.sigma),
            "difficulty": round(difficulty, 3),
            "total_evals": total_evals,
            "elapsed": round(elapsed, 1),
        }
        if callback:
            callback(entry)

        raw_str = f" | Raw: {raw_best:8.1f}" if raw_best > -999 else ""
        solved_str = "✅ SOLVED!" if best_ever_raw >= SOLVED_THRESHOLD else ""
        print(f"Gen {cma.gen:4d} | Best: {gen_best:8.1f} | Ever: {best_ever:8.1f}{raw_str} | "
              f"σ: {cma.sigma:.4f} | Evals: {total_evals:6d} | {elapsed:6.1f}s {solved_str}")

        if best_ever_raw >= SOLVED_THRESHOLD:
            break

    if best_params is not None:
        final_scores = [evaluate(policy, best_params, 1) for _ in range(20)]
        final_mean = float(np.mean(final_scores))
        final_std = float(np.std(final_scores))
    else:
        final_mean = final_std = 0.0

    return {
        "method": "BipedalHardcore-CMA-ES",
        "environment": "BipedalWalker-v3-Hardcore",
        "best_ever_shaped": float(best_ever),
        "best_ever_raw": float(best_ever_raw),
        "final_mean": final_mean,
        "final_std": final_std,
        "total_evals": total_evals,
        "generations": cma.gen,
        "n_params": policy.n_params,
        "elapsed_seconds": round(time.time() - start_time, 1),
        "solved": best_ever_raw >= SOLVED_THRESHOLD,
    }


if __name__ == "__main__":
    result = run(params={"max_evals": 500000})
    print(f"\nResult: {json.dumps(result, indent=2)}")
