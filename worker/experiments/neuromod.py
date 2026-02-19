"""
Neuromodulated CMA-ES for LunarLander — Phase 8 Revisit.

Phase 5 showed neuromodulation reached +80 on LunarLander.
Phase 7 showed compute is key: CMA-ES hit +274 with 100K evals.

Question: Can neuromodulation + CMA-ES + high compute beat plain CMA-ES?

Architecture:
- CMA-ES optimizes all parameters (weights + modulation params)
- Network has a neuromodulatory signal that gates learning
- Dopamine-like reward signal modulates weight plasticity
- Hebbian traces accumulate during episodes
- At episode end, traces are applied scaled by reward signal

This is biologically plausible AND uses evolutionary search.
"""

import numpy as np
import gymnasium as gym
import time
import multiprocessing as mp
import os
import json


class NeuromodPolicy:
    """Policy with neuromodulatory plasticity.
    
    Standard feedforward + modulation pathway:
    - Main path: obs → h1 → h2 → action
    - Mod path: obs → mod_h → modulation_signal (scalar)
    - Plasticity: Hebbian traces gated by modulation signal
    """
    def __init__(self, obs_dim=8, act_dim=4, hidden1=64, hidden2=32, mod_hidden=16):
        # Main network weights
        self.main_shapes = [
            (obs_dim, hidden1), (hidden1,),
            (hidden1, hidden2), (hidden2,),
            (hidden2, act_dim), (act_dim,),
        ]
        self.main_sizes = [np.prod(s) for s in self.main_shapes]
        self.n_main = sum(self.main_sizes)

        # Modulation pathway
        self.mod_shapes = [
            (obs_dim, mod_hidden), (mod_hidden,),
            (mod_hidden, 1), (1,),
        ]
        self.mod_sizes = [np.prod(s) for s in self.mod_shapes]
        self.n_mod = sum(self.mod_sizes)

        # Plasticity rates per layer (3 layers × 1 rate = 3 params)
        self.n_plasticity = 3

        self.n_params = self.n_main + self.n_mod + self.n_plasticity

        # For Hebbian traces
        self.hidden1 = hidden1
        self.hidden2 = hidden2

    def decode(self, genome):
        """Split genome into main weights, mod weights, plasticity rates."""
        main_w = genome[:self.n_main]
        mod_w = genome[self.n_main:self.n_main + self.n_mod]
        plast = np.abs(genome[self.n_main + self.n_mod:]) * 0.01  # small learning rates
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
            if i < len(self.main_shapes) - 2:
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
        return float(np.tanh(x[0]))  # modulation signal in [-1, 1]


def evaluate_neuromod(policy, genome, n_episodes=5):
    """Evaluate with within-episode neuromodulatory plasticity."""
    main_w, mod_w, plast = policy.decode(genome)
    working_w = main_w.copy()

    env = gym.make("LunarLander-v3")
    total_reward = 0.0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 1000)
        done = False
        ep_reward = 0.0
        steps = 0

        # Hebbian traces (accumulated during episode)
        traces = np.zeros_like(working_w)

        while not done and steps < 1000:
            # Forward pass
            logits, activations = policy.forward_main(obs, working_w)
            action = int(np.argmax(logits))

            # Modulation signal (dopamine-like)
            mod_signal = policy.forward_mod(obs, mod_w)

            next_obs, reward, terminated, truncated, _ = env.step(action)

            # Accumulate Hebbian traces
            # For each layer: trace += pre * post * mod_signal
            idx = 0
            for layer_i in range(3):
                w_size = policy.main_sizes[layer_i * 2]
                b_size = policy.main_sizes[layer_i * 2 + 1]

                pre = activations[layer_i]
                post = activations[layer_i + 1]

                # Outer product trace
                if pre.ndim == 1 and post.ndim == 1:
                    hebbian = np.outer(pre, post).flatten()
                    # Scale by reward signal and modulation
                    traces[idx:idx + w_size] += hebbian * mod_signal * reward * 0.001

                idx += w_size + b_size

            ep_reward += reward
            obs = next_obs
            done = terminated or truncated
            steps += 1

        # Apply accumulated traces at episode end
        idx = 0
        for layer_i in range(3):
            w_size = policy.main_sizes[layer_i * 2]
            b_size = policy.main_sizes[layer_i * 2 + 1]
            lr = plast[layer_i] if layer_i < len(plast) else 0.001
            working_w[idx:idx + w_size] += lr * traces[idx:idx + w_size]
            idx += w_size + b_size

        total_reward += ep_reward

    env.close()
    return total_reward / n_episodes


def run(params=None, device="cpu", callback=None):
    """Run Neuromodulated CMA-ES on LunarLander."""
    params = params or {}
    max_evals = params.get("max_evals", 100000)
    eval_episodes = params.get("eval_episodes", 5)
    n_workers = params.get("n_workers", min(os.cpu_count() or 1, 16))

    policy = NeuromodPolicy()
    print(f"Neuromodulated CMA-ES: {policy.n_params} params")
    print(f"  Main network: {policy.n_main}")
    print(f"  Modulation pathway: {policy.n_mod}")
    print(f"  Plasticity rates: {policy.n_plasticity}")
    print(f"Workers: {n_workers}")

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
                fitnesses = pool.starmap(evaluate_neuromod, [(policy, c, eval_episodes) for c in candidates])
            fitnesses = np.array(fitnesses)
        else:
            fitnesses = np.array([evaluate_neuromod(policy, c, eval_episodes) for c in candidates])

        total_evals += len(candidates) * eval_episodes
        cma.tell(candidates, fitnesses)

        gen_best = np.max(fitnesses)
        gen_mean = np.mean(fitnesses)
        if gen_best > best_ever:
            best_ever = gen_best
            best_params = candidates[np.argmax(fitnesses)].copy()

        elapsed = time.time() - start_time
        entry = {
            "generation": cma.gen,
            "best_fitness": float(gen_best),
            "best_ever": float(best_ever),
            "mean_fitness": float(gen_mean),
            "sigma": float(cma.sigma),
            "total_evals": total_evals,
            "elapsed": round(elapsed, 1),
        }
        if callback:
            callback(entry)

        solved_str = "✅ SOLVED!" if best_ever >= 200 else ""
        print(f"Gen {cma.gen:4d} | Best: {gen_best:8.1f} | Ever: {best_ever:8.1f} | "
              f"Mean: {gen_mean:8.1f} | σ: {cma.sigma:.4f} | Evals: {total_evals:6d} | "
              f"{elapsed:6.1f}s {solved_str}")

        if best_ever >= 200:
            print(f"\n🎉 SOLVED with Neuromodulation! Score: {best_ever:.1f}")
            break

    if best_params is not None:
        from experiments.cma_es import evaluate as raw_eval
        final_scores = [raw_eval(policy, best_params, 1) for _ in range(20)]
        final_mean = float(np.mean(final_scores))
        final_std = float(np.std(final_scores))
    else:
        final_mean = final_std = 0.0

    return {
        "method": "Neuromod-CMA-ES",
        "best_ever": float(best_ever),
        "final_mean": final_mean,
        "final_std": final_std,
        "total_evals": total_evals,
        "generations": cma.gen,
        "n_params": policy.n_params,
        "n_main": policy.n_main,
        "n_mod": policy.n_mod,
        "elapsed_seconds": round(time.time() - start_time, 1),
        "solved": best_ever >= 200,
    }


if __name__ == "__main__":
    result = run(params={"max_evals": 100000})
    print(f"\nResult: {json.dumps(result, indent=2)}")
