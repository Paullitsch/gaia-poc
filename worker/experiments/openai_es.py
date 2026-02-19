"""
OpenAI Evolution Strategies for LunarLander.

Estimates gradients via finite differences — perturb parameters with noise,
evaluate, use reward-weighted noise as update direction.

NOT backpropagation: no chain rule, no computational graph.
But DIRECTED updates, not random mutation.

Based on: Salimans et al. (2017) "Evolution Strategies as a Scalable Alternative to RL"
"""

import numpy as np
import multiprocessing as mp
import os
import gymnasium as gym
import time
import json


class PolicyNetwork:
    """Numpy policy network."""

    def __init__(self, obs_dim=8, act_dim=4, hidden1=64, hidden2=32):
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
        out = h @ layers[4] + layers[5]
        return int(np.argmax(out))


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


def run(params=None, device="cpu", callback=None):
    params = params or {}
    max_evals = params.get("max_evals", 100000)
    pop_size = params.get("pop_size", 50)
    lr = params.get("lr", 0.01)
    noise_std = params.get("noise_std", 0.02)
    eval_episodes = params.get("eval_episodes", 5)
    n_workers = params.get("n_workers", min(os.cpu_count() or 1, 16))
    hidden1 = params.get("hidden1", 64)
    hidden2 = params.get("hidden2", 32)

    policy = PolicyNetwork(hidden1=hidden1, hidden2=hidden2)
    theta = np.random.randn(policy.n_params) * 0.1

    print(f"OpenAI ES: {policy.n_params} params, pop={pop_size}, lr={lr}, σ={noise_std}")

    best_ever = -float("inf")
    best_params = None
    total_evals = 0
    start_time = time.time()
    gen = 0

    while total_evals < max_evals:
        # Generate perturbations
        noise = np.random.randn(pop_size, policy.n_params)
        rewards_pos = np.zeros(pop_size)
        rewards_neg = np.zeros(pop_size)

        # Evaluate mirrored perturbations (antithetic sampling) in parallel
        if n_workers > 1:
            args = []
            for i in range(pop_size):
                args.append((policy, theta + noise_std * noise[i], eval_episodes))
                args.append((policy, theta - noise_std * noise[i], eval_episodes))
            with mp.Pool(n_workers) as pool:
                results = pool.starmap(evaluate, args)
            for i in range(pop_size):
                rewards_pos[i] = results[2*i]
                rewards_neg[i] = results[2*i+1]
        else:
            for i in range(pop_size):
                rewards_pos[i] = evaluate(policy, theta + noise_std * noise[i], eval_episodes)
                rewards_neg[i] = evaluate(policy, theta - noise_std * noise[i], eval_episodes)
        total_evals += 2 * pop_size * eval_episodes

        # Combine rewards
        all_rewards = np.concatenate([rewards_pos, rewards_neg])

        # Fitness shaping (rank-based)
        ranks = np.zeros_like(all_rewards)
        sorted_idx = np.argsort(all_rewards)
        for i, idx in enumerate(sorted_idx):
            ranks[idx] = i
        ranks = ranks / (2 * pop_size - 1) - 0.5

        # Compute gradient estimate
        gradient = np.zeros(policy.n_params)
        for i in range(pop_size):
            gradient += ranks[i] * noise[i]              # positive perturbation
            gradient -= ranks[pop_size + i] * noise[i]   # negative perturbation
        gradient /= (2 * pop_size * noise_std)

        # Update
        theta += lr * gradient

        # Stats
        gen_best = np.max(all_rewards)
        gen_mean = np.mean(all_rewards)
        gen_std = np.std(all_rewards)

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
            "best_fitness": float(gen_best),
            "best_ever": float(best_ever),
            "mean_fitness": float(gen_mean),
            "std_fitness": float(gen_std),
            "total_evals": total_evals,
            "elapsed": round(elapsed, 1),
        }

        if callback:
            callback(entry)

        solved_str = "✅ SOLVED!" if best_ever >= 200 else ""
        print(f"Gen {gen:4d} | Best: {gen_best:8.1f} | Ever: {best_ever:8.1f} | "
              f"Mean: {gen_mean:8.1f} | Evals: {total_evals:6d} | {elapsed:6.1f}s {solved_str}")

        if best_ever >= 200:
            print(f"\n🎉 SOLVED with OpenAI ES! Score: {best_ever:.1f}")
            break

    # Final evaluation
    if best_params is not None:
        final_scores = [evaluate(policy, best_params, 1) for _ in range(20)]
        final_mean = float(np.mean(final_scores))
        final_std = float(np.std(final_scores))
    else:
        final_mean = final_std = 0.0

    return {
        "method": "OpenAI-ES",
        "best_ever": float(best_ever),
        "final_mean": final_mean,
        "final_std": final_std,
        "total_evals": total_evals,
        "generations": gen,
        "n_params": policy.n_params,
        "elapsed_seconds": round(time.time() - start_time, 1),
        "solved": best_ever >= 200,
    }


if __name__ == "__main__":
    result = run(params={"max_evals": 100000})
    print(f"\nResult: {json.dumps(result, indent=2)}")
