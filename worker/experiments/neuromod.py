"""
Neuromodulated CMA-ES — Biologically plausible within-episode learning.

Hebbian plasticity + neuromodulatory signal. Agents learn DURING episodes
through local rules. Works on any environment.
"""

import numpy as np
import gymnasium as gym
import time
import multiprocessing as mp
import os


class NeuromodPolicy:
    """Neuromodulated policy for any environment."""

    def __init__(self, obs_dim=8, act_dim=4, act_type="discrete", hidden=None):
        hidden = hidden or [64, 32]
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_type = act_type

        # Main network
        dims = [obs_dim] + hidden + [act_dim]
        self.main_shapes = []
        self.main_sizes = []
        for i in range(len(dims) - 1):
            w_shape = (dims[i], dims[i+1])
            b_shape = (dims[i+1],)
            self.main_shapes.extend([w_shape, b_shape])
            self.main_sizes.extend([np.prod(w_shape), np.prod(b_shape)])
        self.n_main = sum(self.main_sizes)
        self.n_layers = len(dims) - 1

        # Modulation network (small)
        mod_hidden = 16
        self.mod_shapes = [
            (obs_dim, mod_hidden), (mod_hidden,),
            (mod_hidden, 1), (1,),
        ]
        self.mod_sizes = [np.prod(s) for s in self.mod_shapes]
        self.n_mod = sum(self.mod_sizes)

        self.n_plasticity = self.n_layers
        self.n_params = self.n_main + self.n_mod + self.n_plasticity

    def decode(self, genome):
        main_w = genome[:self.n_main]
        mod_w = genome[self.n_main:self.n_main + self.n_mod]
        plast = np.abs(genome[self.n_main + self.n_mod:]) * 0.01
        return main_w, mod_w, plast

    def forward_main(self, x, weights):
        idx = 0
        activations = [x]
        for i in range(0, len(self.main_shapes), 2):
            w = weights[idx:idx + self.main_sizes[i]].reshape(self.main_shapes[i])
            idx += self.main_sizes[i]
            b = weights[idx:idx + self.main_sizes[i+1]]
            idx += self.main_sizes[i+1]
            x = x @ w + b
            x = np.tanh(x)
            activations.append(x)
        return x, activations

    def forward_mod(self, x, mod_weights):
        idx = 0
        for i in range(0, len(self.mod_shapes), 2):
            w = mod_weights[idx:idx + self.mod_sizes[i]].reshape(self.mod_shapes[i])
            idx += self.mod_sizes[i]
            b = mod_weights[idx:idx + self.mod_sizes[i+1]]
            idx += self.mod_sizes[i+1]
            x = x @ w + b
            if i < len(self.mod_shapes) - 2:
                x = np.tanh(x)
        return float(np.tanh(x[0]))


def evaluate_neuromod(policy, genome, env_name, n_episodes=5, max_steps=1000):
    """Evaluate neuromodulated agent on any environment."""
    main_w, mod_w, plast = policy.decode(genome)
    working_w = main_w.copy()
    env = gym.make(env_name)
    total_reward = 0.0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 1000)
        done, ep_reward, steps = False, 0.0, 0
        traces = np.zeros_like(working_w)

        while not done and steps < max_steps:
            output, activations = policy.forward_main(obs, working_w)
            if policy.act_type == "discrete":
                action = int(np.argmax(output))
            else:
                action = output

            mod_signal = policy.forward_mod(obs, mod_w)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            # Hebbian trace update
            idx = 0
            for layer_i in range(policy.n_layers):
                w_size = policy.main_sizes[layer_i * 2]
                b_size = policy.main_sizes[layer_i * 2 + 1]
                pre = activations[layer_i]
                post = activations[layer_i + 1]
                if pre.ndim == 1 and post.ndim == 1:
                    hebbian = np.outer(pre, post).flatten()
                    traces[idx:idx + w_size] += hebbian * mod_signal * reward * 0.001
                idx += w_size + b_size

            ep_reward += reward
            obs = next_obs
            done = terminated or truncated
            steps += 1

        # Apply plasticity at end of episode
        idx = 0
        for layer_i in range(policy.n_layers):
            w_size = policy.main_sizes[layer_i * 2]
            b_size = policy.main_sizes[layer_i * 2 + 1]
            lr = plast[layer_i] if layer_i < len(plast) else 0.001
            working_w[idx:idx + w_size] += lr * traces[idx:idx + w_size]
            idx += w_size + b_size

        total_reward += ep_reward
    env.close()
    return total_reward / n_episodes


def run(params=None, device="cpu", callback=None):
    """Run Neuromodulated CMA-ES on any environment."""
    params = params or {}
    env_name = params.get("environment", "LunarLander-v3")
    max_evals = params.get("max_evals", 200000)
    eval_episodes = params.get("eval_episodes", 5)
    obs_dim = params.get("obs_dim", 8)
    act_dim = params.get("act_dim", 4)
    act_type = params.get("act_type", "discrete")
    hidden = params.get("hidden", [64, 32])
    max_steps = params.get("max_steps", 1000)
    solved_threshold = params.get("solved_threshold", 200)
    n_workers = params.get("n_workers", min(os.cpu_count() or 1, 16))

    policy = NeuromodPolicy(obs_dim=obs_dim, act_dim=act_dim, act_type=act_type, hidden=hidden)

    print(f"🧬 Neuromod CMA-ES on {env_name}")
    print(f"Params: {policy.n_params} (main: {policy.n_main}, mod: {policy.n_mod}, plast: {policy.n_plasticity})")
    print(f"Budget: {max_evals:,} evals | Workers: {n_workers}")

    # Use diagonal CMA-ES (neuromod has many params)
    from experiments.cma_es import CMAES
    cma = CMAES(policy.n_params, sigma0=0.5)

    best_ever = -float("inf")
    best_params = None
    total_evals = 0
    start_time = time.time()

    while total_evals < max_evals:
        candidates = cma.ask()

        if n_workers > 1:
            with mp.Pool(n_workers) as pool:
                fitnesses = pool.starmap(evaluate_neuromod, [(policy, c, env_name, eval_episodes, max_steps) for c in candidates])
            fitnesses = np.array(fitnesses)
        else:
            fitnesses = np.array([evaluate_neuromod(policy, c, env_name, eval_episodes, max_steps) for c in candidates])
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
            "std_fitness": float(np.std(fitnesses)),
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
            print(f"\n🎉 SOLVED {env_name} with Neuromod! Score: {best_ever:.1f}")
            break

    if best_params is not None:
        final_scores = [evaluate_neuromod(policy, best_params, env_name, 1, max_steps) for _ in range(20)]
        final_mean, final_std = float(np.mean(final_scores)), float(np.std(final_scores))
    else:
        final_mean = final_std = 0.0

    return {
        "method": "Neuromod CMA-ES",
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
