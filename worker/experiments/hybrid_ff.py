"""
Hybrid CMA-ES + Forward-Forward for LunarLander.

CMA-ES optimizes hyperparameters and initial weights.
Forward-Forward does within-lifetime representation learning.

Best of both: directed search + local learning.
"""

import numpy as np
import gymnasium as gym
import time
import json
import multiprocessing as mp
import os


class FFPolicyNetwork:
    """Policy network with Forward-Forward local learning."""

    def __init__(self, obs_dim=8, act_dim=4, hidden1=64, hidden2=32):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.h1 = hidden1
        self.h2 = hidden2

        # Genome: initial weights + FF hyperparameters
        self.weight_shapes = [
            (obs_dim, hidden1), (hidden1,),
            (hidden1, hidden2), (hidden2,),
            (hidden2, act_dim), (act_dim,),
        ]
        self.weight_sizes = [np.prod(s) for s in self.weight_shapes]
        self.n_weight_params = sum(self.weight_sizes)

        # FF hyperparams: lr_per_layer(3) + goodness_thresh(2) + plasticity(2) = 7
        self.n_hyper = 7
        self.n_params = self.n_weight_params + self.n_hyper

    def decode(self, genome):
        """Decode genome into weights + hyperparams."""
        weights = genome[:self.n_weight_params]
        hyper = genome[self.n_weight_params:]

        # Decode hyperparams with sigmoid/softplus for valid ranges
        ff_lrs = np.abs(hyper[:3]) * 0.01       # learning rates [0, ~0.05]
        goodness_thresh = np.abs(hyper[3:5]) + 1.0  # thresholds [1, ~5]
        plasticity = np.abs(hyper[5:7]) * 0.1     # plasticity [0, ~0.5]

        return weights, ff_lrs, goodness_thresh, plasticity

    def forward(self, x, weights):
        idx = 0
        layers = []
        for shape, size in zip(self.weight_shapes, self.weight_sizes):
            layers.append(weights[idx:idx + size].reshape(shape))
            idx += size
        h = np.tanh(x @ layers[0] + layers[1])
        h = np.tanh(h @ layers[2] + layers[3])
        out = h @ layers[4] + layers[5]
        return out

    def act(self, obs, weights):
        logits = self.forward(obs, weights)
        return int(np.argmax(logits))


def evaluate_with_ff_learning(policy, genome, n_episodes=5):
    """Evaluate with within-lifetime Forward-Forward learning."""
    weights, ff_lrs, goodness_thresh, plasticity = policy.decode(genome)
    working_weights = weights.copy()

    env = gym.make("LunarLander-v3")
    total_reward = 0.0
    experience_buffer = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 1000)
        done = False
        ep_reward = 0.0
        ep_experiences = []
        steps = 0

        while not done and steps < 1000:
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

                # Apply FF update to first two layers
                idx = 0
                for layer_i in range(2):
                    w_shape = policy.weight_shapes[layer_i * 2]
                    b_shape = policy.weight_shapes[layer_i * 2 + 1]
                    w_size = policy.weight_sizes[layer_i * 2]
                    b_size = policy.weight_sizes[layer_i * 2 + 1]

                    W = working_weights[idx:idx + w_size].reshape(w_shape)
                    b = working_weights[idx + w_size:idx + w_size + b_size]

                    # Compute activations
                    if layer_i == 0:
                        pos_act = np.tanh(pos_batch @ W + b)
                        neg_act = np.tanh(neg_batch @ W + b)
                    else:
                        # Get prev layer activations
                        prev_w = working_weights[:policy.weight_sizes[0]].reshape(policy.weight_shapes[0])
                        prev_b = working_weights[policy.weight_sizes[0]:policy.weight_sizes[0] + policy.weight_sizes[1]]
                        pos_prev = np.tanh(pos_batch @ prev_w + prev_b)
                        neg_prev = np.tanh(neg_batch @ prev_w + prev_b)
                        pos_act = np.tanh(pos_prev @ W + b)
                        neg_act = np.tanh(neg_prev @ W + b)

                    # Goodness = sum of squared activations
                    pos_good = np.mean(np.sum(pos_act ** 2, axis=1))
                    neg_good = np.mean(np.sum(neg_act ** 2, axis=1))

                    # Get layer input (observations for layer 0, prev activations for layer 1)
                    if layer_i == 0:
                        pos_input = pos_batch
                        neg_input = neg_batch
                    else:
                        pos_input = pos_prev
                        neg_input = neg_prev

                    # FF update: increase goodness for positive, decrease for negative
                    lr = ff_lrs[layer_i] * plasticity[min(layer_i, len(plasticity) - 1)]
                    if pos_good < goodness_thresh[min(layer_i, len(goodness_thresh) - 1)]:
                        W += lr * (pos_input.T @ pos_act) / len(pos_batch)
                    if neg_good > goodness_thresh[min(layer_i, len(goodness_thresh) - 1)] * 0.5:
                        W -= lr * (neg_input.T @ neg_act) / len(neg_batch)

                    working_weights[idx:idx + w_size] = W.flatten()
                    idx += w_size + b_size

    env.close()
    return total_reward / n_episodes


def run(params=None, device="cpu", callback=None):
    params = params or {}
    max_evals = params.get("max_evals", 100000)
    eval_episodes = params.get("eval_episodes", 5)
    n_workers = params.get("n_workers", min(os.cpu_count() or 1, 16))

    policy = FFPolicyNetwork()
    print(f"Hybrid CMA+FF: {policy.n_params} params ({policy.n_weight_params} weights + {policy.n_hyper} FF hyper)")

    # Import CMA-ES from sibling
    from experiments.cma_es import CMAES
    cma = CMAES(policy.n_params, sigma0=0.5)
    print(f"CMA-ES population: {cma.lam}")

    best_ever = -float("inf")
    best_params = None
    total_evals = 0
    start_time = time.time()

    while total_evals < max_evals:
        candidates = cma.ask()
        if n_workers > 1:
            with mp.Pool(n_workers) as pool:
                fitnesses = pool.starmap(evaluate_with_ff_learning, [(policy, c, eval_episodes) for c in candidates])
            fitnesses = np.array(fitnesses)
        else:
            fitnesses = np.array([evaluate_with_ff_learning(policy, c, eval_episodes) for c in candidates])
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
            "total_evals": total_evals,
            "elapsed": round(elapsed, 1),
        }
        if callback:
            callback(entry)

        solved_str = "✅ SOLVED!" if best_ever >= 200 else ""
        print(f"Gen {cma.gen:4d} | Best: {gen_best:8.1f} | Ever: {best_ever:8.1f} | "
              f"Mean: {np.mean(fitnesses):8.1f} | Evals: {total_evals:6d} | {elapsed:6.1f}s {solved_str}")

        if best_ever >= 200:
            break

    if best_params is not None:
        final_scores = [evaluate_with_ff_learning(policy, best_params, 1) for _ in range(20)]
        final_mean = float(np.mean(final_scores))
        final_std = float(np.std(final_scores))
    else:
        final_mean = final_std = 0.0

    return {
        "method": "Hybrid-CMA+FF",
        "best_ever": float(best_ever),
        "final_mean": final_mean,
        "final_std": final_std,
        "total_evals": total_evals,
        "generations": cma.gen,
        "n_params": policy.n_params,
        "elapsed_seconds": round(time.time() - start_time, 1),
        "solved": best_ever >= 200,
    }


if __name__ == "__main__":
    result = run(params={"max_evals": 50000})
    print(f"\nResult: {json.dumps(result, indent=2)}")
