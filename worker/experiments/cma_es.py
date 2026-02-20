"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

The gold standard of gradient-free optimization. Works on any environment.
NO BACKPROPAGATION. NO GRADIENTS. Pure search with learned structure.
"""

import numpy as np
import gymnasium as gym
import time
import multiprocessing as mp
import os


class PolicyNetwork:
    """Environment-agnostic policy network."""

    def __init__(self, obs_dim=8, act_dim=4, act_type="discrete", hidden=None):
        hidden = hidden or [64, 32]
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_type = act_type

        dims = [obs_dim] + hidden + [act_dim]
        self.shapes = []
        self.sizes = []
        for i in range(len(dims) - 1):
            w_shape = (dims[i], dims[i+1])
            b_shape = (dims[i+1],)
            self.shapes.extend([w_shape, b_shape])
            self.sizes.extend([np.prod(w_shape), np.prod(b_shape)])
        self.n_params = sum(self.sizes)

    def forward(self, x, params):
        idx = 0
        for i in range(0, len(self.shapes), 2):
            w = params[idx:idx + self.sizes[i]].reshape(self.shapes[i])
            idx += self.sizes[i]
            b = params[idx:idx + self.sizes[i+1]]
            idx += self.sizes[i+1]
            x = x @ w + b
            if i < len(self.shapes) - 2:  # hidden layers
                x = np.tanh(x)
        if self.act_type == "continuous":
            x = np.tanh(x)  # bound to [-1, 1]
        return x

    def act(self, obs, params):
        out = self.forward(obs, params)
        if self.act_type == "discrete":
            return int(np.argmax(out))
        return out  # continuous


def evaluate(policy, params, env_name, n_episodes=5, max_steps=1000):
    """Evaluate a parameter vector on any environment.
    
    For pixel-based envs (Atari), uses CNN + frame stacking via atari_eval.
    For vector envs, uses direct numpy forward pass.
    """
    # Detect Atari environments
    if hasattr(policy, '_is_atari') and policy._is_atari:
        from experiments.atari_eval import evaluate_atari
        device = getattr(policy, 'device', 'cpu')
        return evaluate_atari(params, env_name, policy.act_dim,
                              n_episodes, max_steps, device)

    env = gym.make(env_name)
    total = 0.0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 1000)
        done, ep_reward, steps = False, 0.0, 0
        while not done and steps < max_steps:
            action = policy.act(obs, params)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            steps += 1
        total += ep_reward
    env.close()
    return total / n_episodes


class CMAES:
    """CMA-ES with automatic diagonal/full covariance."""

    def __init__(self, n_params, sigma0=0.5, pop_size=None):
        self.n = n_params
        self.sigma = sigma0
        self.mean = np.random.randn(n_params) * 0.1
        self.lam = pop_size or (4 + int(3 * np.log(n_params)))
        self.mu = self.lam // 2

        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / weights.sum()
        self.mueff = 1.0 / np.sum(self.weights ** 2)

        self.cs = (self.mueff + 2) / (n_params + self.mueff + 5)
        self.ds = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (n_params + 1)) - 1) + self.cs
        self.ps = np.zeros(n_params)

        self.cc = (4 + self.mueff / n_params) / (n_params + 4 + 2 * self.mueff / n_params)
        self.c1 = 2 / ((n_params + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n_params + 2) ** 2 + self.mueff))
        self.pc = np.zeros(n_params)

        self.use_diagonal = n_params > 2000
        if self.use_diagonal:
            self.C_diag = np.ones(n_params)
        else:
            self.C = np.eye(n_params)

        self.chi_n = np.sqrt(n_params) * (1 - 1 / (4 * n_params) + 1 / (21 * n_params ** 2))
        self.gen = 0

    def ask(self):
        if self.use_diagonal:
            std = self.sigma * np.sqrt(self.C_diag)
            return [self.mean + std * np.random.randn(self.n) for _ in range(self.lam)]
        else:
            try:
                A = np.linalg.cholesky(self.C)
            except np.linalg.LinAlgError:
                self.C = np.eye(self.n)
                A = np.eye(self.n)
            return [self.mean + self.sigma * (A @ np.random.randn(self.n)) for _ in range(self.lam)]

    def tell(self, solutions, fitnesses):
        idx = np.argsort(fitnesses)[::-1]
        selected = np.array([solutions[i] for i in idx[:self.mu]])
        old_mean = self.mean.copy()
        self.mean = self.weights @ selected

        if self.use_diagonal:
            invsqrt = 1.0 / np.sqrt(self.C_diag)
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * invsqrt * (self.mean - old_mean) / self.sigma
        else:
            try:
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(self.C))
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(self.n)
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * C_inv_sqrt @ (self.mean - old_mean) / self.sigma

        hs = int(np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) < (1.4 + 2 / (self.n + 1)) * self.chi_n)
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (self.mean - old_mean) / self.sigma

        artmp = (selected - old_mean) / self.sigma
        if self.use_diagonal:
            self.C_diag = (1 - self.c1 - self.cmu) * self.C_diag + self.c1 * (self.pc ** 2 + (1 - hs) * self.cc * (2 - self.cc) * self.C_diag) + self.cmu * np.sum(self.weights[:, None] * artmp ** 2, axis=0)
            self.C_diag = np.maximum(self.C_diag, 1e-20)
        else:
            self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hs) * self.cc * (2 - self.cc) * self.C) + self.cmu * sum(w * np.outer(a, a) for w, a in zip(self.weights, artmp))

        self.sigma *= np.exp((self.cs / self.ds) * (np.linalg.norm(self.ps) / self.chi_n - 1))
        self.sigma = np.clip(self.sigma, 1e-20, 10.0)
        self.gen += 1


def run(params=None, device="cpu", callback=None):
    """Run CMA-ES on any environment."""
    params = params or {}
    env_name = params.get("environment", "LunarLander-v3")
    max_evals = params.get("max_evals", 100000)
    sigma0 = params.get("sigma0", 0.5)
    eval_episodes = params.get("eval_episodes", 5)
    obs_dim = params.get("obs_dim", 8)
    act_dim = params.get("act_dim", 4)
    act_type = params.get("act_type", "discrete")
    hidden = params.get("hidden", [64, 32])
    max_steps = params.get("max_steps", 1000)
    solved_threshold = params.get("solved_threshold", 200)
    n_workers = params.get("n_workers", min(os.cpu_count() or 1, 16))

    obs_type = params.get("obs_type", "vector")

    if obs_type == "pixel":
        # Atari: use CNN policy via PyTorch
        from experiments.atari_eval import AtariCNN
        import torch
        n_frames = params.get("n_frames", 4)
        _model = AtariCNN(n_frames=n_frames, n_actions=act_dim)
        # Create a lightweight adapter so evaluate() knows it's Atari
        class _AtariPolicy:
            def __init__(self, model):
                self.n_params = model.n_params
                self.act_dim = act_dim
                self.act_type = "discrete"
                self._is_atari = True
                self.device = device
        policy = _AtariPolicy(_model)
        print(f"🎮 CMA-ES on {env_name} (CNN, {policy.n_params} params)")
    else:
        policy = PolicyNetwork(obs_dim=obs_dim, act_dim=act_dim, act_type=act_type, hidden=hidden)
        print(f"🧬 CMA-ES on {env_name}")
        print(f"Network: {obs_dim}→{'→'.join(map(str,hidden))}→{act_dim} ({policy.n_params} params, {act_type})")

    cma = CMAES(policy.n_params, sigma0=sigma0)
    print(f"Budget: {max_evals:,} evals | Pop: {cma.lam} | Workers: {n_workers} | Diagonal: {cma.use_diagonal}")

    best_ever = -float("inf")
    best_params = None
    total_evals = 0
    start_time = time.time()

    while total_evals < max_evals:
        candidates = cma.ask()

        if n_workers > 1:
            with mp.Pool(n_workers) as pool:
                fitnesses = pool.starmap(evaluate, [(policy, c, env_name, eval_episodes, max_steps) for c in candidates])
            fitnesses = np.array(fitnesses)
        else:
            fitnesses = np.array([evaluate(policy, c, env_name, eval_episodes, max_steps) for c in candidates])
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
            print(f"\n🎉 SOLVED {env_name}! Score: {best_ever:.1f}")
            break

    if best_params is not None:
        final_scores = [evaluate(policy, best_params, env_name, 1, max_steps) for _ in range(20)]
        final_mean, final_std = float(np.mean(final_scores)), float(np.std(final_scores))
    else:
        final_mean = final_std = 0.0

    return {
        "method": "CMA-ES",
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
