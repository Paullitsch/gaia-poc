"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for LunarLander.

The gold standard of gradient-free optimization. Learns the covariance
structure of the parameter space — discovers which parameters should
change together.

NO BACKPROPAGATION. NO GRADIENTS. Pure search with learned structure.
"""

import numpy as np
import gymnasium as gym
import time

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class PolicyNetwork:
    """Simple numpy-based policy network (no autograd needed)."""

    def __init__(self, obs_dim=8, act_dim=4, hidden1=64, hidden2=32):
        self.shapes = [
            (obs_dim, hidden1), (hidden1,),      # Layer 1
            (hidden1, hidden2), (hidden2,),       # Layer 2
            (hidden2, act_dim), (act_dim,),       # Output
        ]
        self.sizes = [np.prod(s) for s in self.shapes]
        self.n_params = sum(self.sizes)

    def forward(self, x, params):
        """Forward pass with flat parameter vector."""
        idx = 0
        layers = []
        for shape, size in zip(self.shapes, self.sizes):
            layers.append(params[idx:idx + size].reshape(shape))
            idx += size

        # Layer 1
        h = np.tanh(x @ layers[0] + layers[1])
        # Layer 2
        h = np.tanh(h @ layers[2] + layers[3])
        # Output
        out = h @ layers[4] + layers[5]
        return out

    def act(self, obs, params):
        logits = self.forward(obs, params)
        return int(np.argmax(logits))


def evaluate(policy, params, n_episodes=5, render=False):
    """Evaluate a parameter vector on LunarLander."""
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


class CMAES:
    """CMA-ES implementation optimized for neural network policy search."""

    def __init__(self, n_params, sigma0=0.5, pop_size=None):
        self.n = n_params
        self.sigma = sigma0
        self.mean = np.random.randn(n_params) * 0.1

        # Population size
        self.lam = pop_size or (4 + int(3 * np.log(n_params)))
        self.mu = self.lam // 2

        # Weights
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / weights.sum()
        self.mueff = 1.0 / np.sum(self.weights ** 2)

        # Step-size adaptation
        self.cs = (self.mueff + 2) / (n_params + self.mueff + 5)
        self.ds = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (n_params + 1)) - 1) + self.cs
        self.ps = np.zeros(n_params)

        # Covariance adaptation
        self.cc = (4 + self.mueff / n_params) / (n_params + 4 + 2 * self.mueff / n_params)
        self.c1 = 2 / ((n_params + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1 - self.c1,
            2 * (self.mueff - 2 + 1 / self.mueff) / ((n_params + 2) ** 2 + self.mueff)
        )
        self.pc = np.zeros(n_params)

        # Use diagonal covariance for large parameter spaces (memory efficient)
        self.use_diagonal = n_params > 2000
        if self.use_diagonal:
            self.C_diag = np.ones(n_params)
        else:
            self.C = np.eye(n_params)

        self.chi_n = np.sqrt(n_params) * (1 - 1 / (4 * n_params) + 1 / (21 * n_params ** 2))
        self.gen = 0

    def ask(self):
        """Generate candidate solutions."""
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
        """Update distribution based on evaluated solutions."""
        # Sort by fitness (descending — we maximize)
        idx = np.argsort(fitnesses)[::-1]
        selected = np.array([solutions[i] for i in idx[:self.mu]])

        old_mean = self.mean.copy()
        self.mean = self.weights @ selected

        # Cumulation: step-size
        if self.use_diagonal:
            invsqrt_diag = 1.0 / np.sqrt(self.C_diag)
            self.ps = (1 - self.cs) * self.ps + \
                np.sqrt(self.cs * (2 - self.cs) * self.mueff) * invsqrt_diag * (self.mean - old_mean) / self.sigma
        else:
            try:
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(self.C))
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(self.n)
            self.ps = (1 - self.cs) * self.ps + \
                np.sqrt(self.cs * (2 - self.cs) * self.mueff) * C_inv_sqrt @ (self.mean - old_mean) / self.sigma

        # Cumulation: rank-one
        hs = int(np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) < (1.4 + 2 / (self.n + 1)) * self.chi_n)
        self.pc = (1 - self.cc) * self.pc + \
            hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (self.mean - old_mean) / self.sigma

        # Covariance update
        artmp = (selected - old_mean) / self.sigma
        if self.use_diagonal:
            self.C_diag = (1 - self.c1 - self.cmu) * self.C_diag + \
                self.c1 * (self.pc ** 2 + (1 - hs) * self.cc * (2 - self.cc) * self.C_diag) + \
                self.cmu * np.sum(self.weights[:, None] * artmp ** 2, axis=0)
            self.C_diag = np.maximum(self.C_diag, 1e-20)
        else:
            self.C = (1 - self.c1 - self.cmu) * self.C + \
                self.c1 * (np.outer(self.pc, self.pc) + (1 - hs) * self.cc * (2 - self.cc) * self.C) + \
                self.cmu * sum(w * np.outer(a, a) for w, a in zip(self.weights, artmp))

        # Step-size update
        self.sigma *= np.exp((self.cs / self.ds) * (np.linalg.norm(self.ps) / self.chi_n - 1))
        self.sigma = np.clip(self.sigma, 1e-20, 10.0)

        self.gen += 1


def run(params=None, device="cpu", callback=None):
    """Run CMA-ES on LunarLander.
    
    Args:
        params: dict with optional keys:
            - max_evals: max evaluations (default 100000)
            - sigma0: initial step size (default 0.5)
            - eval_episodes: episodes per evaluation (default 5)
            - hidden1: first hidden layer size (default 64)
            - hidden2: second hidden layer size (default 32)
        device: compute device (unused for numpy, but checked for torch speedup)
        callback: function called with progress data each generation
    
    Returns:
        dict with results
    """
    params = params or {}
    max_evals = params.get("max_evals", 100000)
    sigma0 = params.get("sigma0", 0.5)
    eval_episodes = params.get("eval_episodes", 5)
    hidden1 = params.get("hidden1", 64)
    hidden2 = params.get("hidden2", 32)

    policy = PolicyNetwork(hidden1=hidden1, hidden2=hidden2)
    print(f"Network: {policy.n_params} parameters")
    print(f"Budget: {max_evals} evaluations")

    cma = CMAES(policy.n_params, sigma0=sigma0)
    print(f"CMA-ES population: {cma.lam}")

    best_ever = -float("inf")
    best_params = None
    total_evals = 0
    start_time = time.time()
    history = []

    while total_evals < max_evals:
        candidates = cma.ask()

        # Evaluate all candidates
        fitnesses = []
        for c in candidates:
            f = evaluate(policy, c, n_episodes=eval_episodes)
            fitnesses.append(f)
        fitnesses = np.array(fitnesses)
        total_evals += len(candidates) * eval_episodes

        cma.tell(candidates, fitnesses)

        gen_best = np.max(fitnesses)
        gen_mean = np.mean(fitnesses)
        gen_std = np.std(fitnesses)

        if gen_best > best_ever:
            best_ever = gen_best
            best_params = candidates[np.argmax(fitnesses)].copy()

        elapsed = time.time() - start_time

        entry = {
            "generation": cma.gen,
            "best_fitness": float(gen_best),
            "best_ever": float(best_ever),
            "mean_fitness": float(gen_mean),
            "std_fitness": float(gen_std),
            "sigma": float(cma.sigma),
            "total_evals": total_evals,
            "elapsed": round(elapsed, 1),
        }
        history.append(entry)

        if callback:
            callback(entry)

        solved_str = "✅ SOLVED!" if best_ever >= 200 else ""
        print(f"Gen {cma.gen:4d} | Best: {gen_best:8.1f} | Ever: {best_ever:8.1f} | "
              f"Mean: {gen_mean:8.1f} | σ: {cma.sigma:.4f} | Evals: {total_evals:6d} | "
              f"{elapsed:6.1f}s {solved_str}")

        if best_ever >= 200:
            print(f"\n🎉 SOLVED LunarLander with CMA-ES! Score: {best_ever:.1f}")
            break

    # Final robust evaluation of best
    if best_params is not None:
        final_scores = [evaluate(policy, best_params, n_episodes=1) for _ in range(20)]
        final_mean = np.mean(final_scores)
        final_std = np.std(final_scores)
    else:
        final_mean = final_std = 0.0

    elapsed = time.time() - start_time

    result = {
        "method": "CMA-ES",
        "best_ever": float(best_ever),
        "final_mean": float(final_mean),
        "final_std": float(final_std),
        "total_evals": total_evals,
        "generations": cma.gen,
        "n_params": policy.n_params,
        "elapsed_seconds": round(elapsed, 1),
        "solved": best_ever >= 200,
    }

    # Save best params
    if best_params is not None:
        np.save("best_cma_es_params.npy", best_params)

    return result


if __name__ == "__main__":
    result = run(
        params={"max_evals": 100000, "eval_episodes": 5},
        callback=lambda d: None
    )
    print(f"\n{'='*60}")
    print(f"Final Result: {json.dumps(result, indent=2)}")
