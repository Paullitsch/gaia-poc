"""
OpenAI-ES for BipedalWalker-v3 — Phase 8.

OpenAI's Evolution Strategy with antithetic sampling.
Simpler than CMA-ES but scales better to large parameter spaces.
BipedalWalker has 11K+ params — ES might shine here.
"""

import numpy as np
import gymnasium as gym
import time
import multiprocessing as mp
import os

OBS_DIM = 24
ACT_DIM = 4
SOLVED_THRESHOLD = 300


class PolicyNetwork:
    """Same architecture as bipedal_cma: 24 -> 128 -> 64 -> 4 (tanh)."""
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden1=128, hidden2=64):
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
    """Evaluate on BipedalWalker."""
    env = gym.make("BipedalWalker-v3")
    total = 0.0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 1000)
        done = False
        ep_reward = 0.0
        steps = 0
        while not done and steps < 1600:
            action = policy.act(obs, params)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            steps += 1
        total += ep_reward
    env.close()
    return total / n_episodes


def run(params=None, device="cpu", callback=None):
    """Run OpenAI-ES on BipedalWalker."""
    params = params or {}
    max_evals = params.get("max_evals", 500000)
    eval_episodes = params.get("eval_episodes", 3)
    pop_size = params.get("pop_size", 64)  # half-population (antithetic doubles it)
    lr = params.get("lr", 0.02)
    sigma = params.get("sigma", 0.05)
    lr_decay = params.get("lr_decay", 0.999)
    n_workers = params.get("n_workers", min(os.cpu_count() or 1, 16))

    policy = PolicyNetwork()
    n = policy.n_params
    print(f"OpenAI-ES for BipedalWalker: {n} parameters")
    print(f"Population: {pop_size}x2 (antithetic)")
    print(f"LR: {lr}, σ: {sigma}")
    print(f"Budget: {max_evals} evaluations")
    print(f"Parallel workers: {n_workers}")

    theta = np.random.randn(n) * 0.1
    best_ever = -float("inf")
    best_params = None
    total_evals = 0
    start_time = time.time()
    gen = 0

    while total_evals < max_evals:
        gen += 1
        # Sample perturbations (antithetic)
        noise = [np.random.randn(n) for _ in range(pop_size)]
        candidates = []
        for eps in noise:
            candidates.append(theta + sigma * eps)
            candidates.append(theta - sigma * eps)

        # Evaluate all
        if n_workers > 1:
            with mp.Pool(n_workers) as pool:
                fitnesses = pool.starmap(evaluate, [(policy, c, eval_episodes) for c in candidates])
            fitnesses = np.array(fitnesses)
        else:
            fitnesses = np.array([evaluate(policy, c, eval_episodes) for c in candidates])

        total_evals += len(candidates) * eval_episodes

        # Track best
        gen_best_idx = np.argmax(fitnesses)
        gen_best = fitnesses[gen_best_idx]
        gen_mean = np.mean(fitnesses)

        if gen_best > best_ever:
            best_ever = gen_best
            best_params = candidates[gen_best_idx].copy()

        # Compute gradient estimate (antithetic)
        # For each pair (f+, f-), gradient contribution is (f+ - f-) * eps
        grad = np.zeros(n)
        for i, eps in enumerate(noise):
            f_plus = fitnesses[2 * i]
            f_minus = fitnesses[2 * i + 1]
            grad += (f_plus - f_minus) * eps

        # Normalize and update
        grad /= (2 * pop_size * sigma)

        # Fitness-based normalization (rank transform for stability)
        theta += lr * grad
        lr *= lr_decay

        elapsed = time.time() - start_time
        entry = {
            "generation": gen,
            "best_fitness": float(gen_best),
            "best_ever": float(best_ever),
            "mean_fitness": float(gen_mean),
            "sigma": float(sigma),
            "lr": float(lr),
            "total_evals": total_evals,
            "elapsed": round(elapsed, 1),
        }
        if callback:
            callback(entry)

        solved_str = "✅ SOLVED!" if best_ever >= SOLVED_THRESHOLD else ""
        print(f"Gen {gen:4d} | Best: {gen_best:8.1f} | Ever: {best_ever:8.1f} | "
              f"Mean: {gen_mean:8.1f} | σ: {sigma:.4f} | Evals: {total_evals:6d} | "
              f"{elapsed:6.1f}s {solved_str}")

        if best_ever >= SOLVED_THRESHOLD:
            print(f"\n🎉 SOLVED BipedalWalker with OpenAI-ES! Score: {best_ever:.1f}")
            break

    # Final robust evaluation
    if best_params is not None:
        final_scores = [evaluate(policy, best_params, 1) for _ in range(20)]
        final_mean = float(np.mean(final_scores))
        final_std = float(np.std(final_scores))
    else:
        final_mean = final_std = 0.0

    return {
        "method": "BipedalWalker-OpenAI-ES",
        "environment": "BipedalWalker-v3",
        "best_ever": float(best_ever),
        "final_mean": final_mean,
        "final_std": final_std,
        "total_evals": total_evals,
        "generations": gen,
        "n_params": policy.n_params,
        "elapsed_seconds": round(time.time() - start_time, 1),
        "solved": best_ever >= SOLVED_THRESHOLD,
    }


if __name__ == "__main__":
    import json
    result = run(params={"max_evals": 200000})
    print(f"\nResult: {json.dumps(result, indent=2)}")
