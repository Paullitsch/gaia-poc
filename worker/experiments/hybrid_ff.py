"""
Hybrid CMA-ES + Forward-Forward.

CMA-ES optimizes hyperparameters and initial weights.
Forward-Forward does within-lifetime representation learning.

Best of both: directed search + local learning.
Environment-agnostic.
"""

import numpy as np
import gymnasium as gym
import time
import json
import multiprocessing as mp
import os


class FFPolicyNetwork:
    """Policy network with Forward-Forward local learning."""

    def __init__(self, obs_dim=8, act_dim=4, hidden=None, act_type="discrete"):
        hidden = hidden or [64, 32]
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_type = act_type
        self.hidden = hidden

        # Genome: initial weights + FF hyperparameters
        self.weight_shapes = []
        self.weight_sizes = []
        dims = [obs_dim] + hidden + [act_dim]
        for i in range(len(dims) - 1):
            self.weight_shapes.append((dims[i], dims[i+1]))
            self.weight_sizes.append(dims[i] * dims[i+1])
            self.weight_shapes.append((dims[i+1],))
            self.weight_sizes.append(dims[i+1])
        self.n_weight_params = sum(self.weight_sizes)

        # FF hyperparams: lr_per_layer + goodness_thresh + plasticity
        n_hidden = len(hidden)
        self.n_hyper = n_hidden * 3 + 1  # lr(n) + thresh(n) + plast(n) + 1
        self.n_params = self.n_weight_params + self.n_hyper

    def decode(self, genome):
        weights = genome[:self.n_weight_params]
        hyper = genome[self.n_weight_params:]
        n_h = len(self.hidden)
        ff_lrs = np.abs(hyper[:n_h]) * 0.01
        goodness_thresh = np.abs(hyper[n_h:2*n_h]) + 1.0
        plasticity = np.abs(hyper[2*n_h:3*n_h]) * 0.1
        return weights, ff_lrs, goodness_thresh, plasticity

    def forward(self, x, weights):
        idx = 0
        for i in range(0, len(self.weight_shapes), 2):
            w = weights[idx:idx + self.weight_sizes[i]].reshape(self.weight_shapes[i])
            idx += self.weight_sizes[i]
            b = weights[idx:idx + self.weight_sizes[i+1]]
            idx += self.weight_sizes[i+1]
            x = x @ w + b
            if i < len(self.weight_shapes) - 2:
                x = np.tanh(x)
        return x

    def act(self, obs, weights):
        out = self.forward(obs, weights)
        if self.act_type == "discrete":
            return int(np.argmax(out))
        return np.clip(np.tanh(out), -1, 1)


def evaluate_with_ff_learning(policy, genome, env_name, n_episodes=5, max_steps=1000):
    """Evaluate with within-lifetime Forward-Forward learning."""
    weights, ff_lrs, goodness_thresh, plasticity = policy.decode(genome)
    working_weights = weights.copy()

    env = gym.make(env_name)
    total_reward = 0.0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 1000)
        done = False
        ep_reward = 0.0
        ep_experiences = []
        steps = 0

        while not done and steps < max_steps:
            action = policy.act(obs, working_weights)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_experiences.append((obs.copy(), reward))
            ep_reward += reward
            obs = next_obs
            done = terminated or truncated
            steps += 1

        total_reward += ep_reward

        # FF learning after each episode
        if len(ep_experiences) > 10:
            rewards = np.array([e[1] for e in ep_experiences])
            threshold = np.median(rewards)

            positive = [e[0] for e in ep_experiences if e[1] > threshold]
            negative = [e[0] for e in ep_experiences if e[1] <= threshold]

            if positive and negative:
                pos_batch = np.array(positive[:50])
                neg_batch = np.array(negative[:50])

                # Apply FF update to hidden layers
                idx = 0
                prev_pos = pos_batch
                prev_neg = neg_batch
                for layer_i in range(len(policy.hidden)):
                    w_size = policy.weight_sizes[layer_i * 2]
                    b_size = policy.weight_sizes[layer_i * 2 + 1]
                    W = working_weights[idx:idx + w_size].reshape(policy.weight_shapes[layer_i * 2])
                    b = working_weights[idx + w_size:idx + w_size + b_size]

                    pos_act = np.tanh(prev_pos @ W + b)
                    neg_act = np.tanh(prev_neg @ W + b)

                    pos_good = np.mean(np.sum(pos_act ** 2, axis=1))
                    neg_good = np.mean(np.sum(neg_act ** 2, axis=1))

                    li = min(layer_i, len(ff_lrs) - 1)
                    lr = ff_lrs[li] * plasticity[li]
                    if pos_good < goodness_thresh[li]:
                        W += lr * (prev_pos.T @ pos_act) / len(pos_batch)
                    if neg_good > goodness_thresh[li] * 0.5:
                        W -= lr * (prev_neg.T @ neg_act) / len(neg_batch)

                    working_weights[idx:idx + w_size] = W.flatten()
                    prev_pos = pos_act
                    prev_neg = neg_act
                    idx += w_size + b_size

    env.close()
    return total_reward / n_episodes


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

    policy = FFPolicyNetwork(obs_dim, act_dim, hidden, act_type)
    print(f"📐 Hybrid CMA+FF on {env_name}: {policy.n_params} params ({policy.n_weight_params} weights + {policy.n_hyper} FF hyper)")

    from experiments.cma_es import CMAES
    cma = CMAES(policy.n_params, sigma0=0.5)

    best_ever = -float("inf")
    best_params = None
    total_evals = 0
    start_time = time.time()

    while total_evals < max_evals:
        candidates = cma.ask()
        args = [(policy, c, env_name, eval_episodes, max_steps) for c in candidates]
        if n_workers > 1:
            with mp.Pool(n_workers) as pool:
                fitnesses = pool.starmap(evaluate_with_ff_learning, args)
            fitnesses = np.array(fitnesses)
        else:
            fitnesses = np.array([evaluate_with_ff_learning(*a) for a in args])
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
        final_scores = [evaluate_with_ff_learning(policy, best_params, env_name, 1, max_steps) for _ in range(20)]
        final_mean, final_std = float(np.mean(final_scores)), float(np.std(final_scores))
    else:
        final_mean = final_std = 0.0

    return {
        "method": "Hybrid-CMA+FF",
        "environment": env_name,
        "best_ever": float(best_ever),
        "final_mean": final_mean,
        "final_std": final_std,
        "total_evals": total_evals,
        "generations": cma.gen,
        "n_params": policy.n_params,
        "elapsed_seconds": round(time.time() - start_time, 1),
        "solved": best_ever >= solved_threshold,
    }
