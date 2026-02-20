"""
GPU-Accelerated CMA-ES using multiprocessing + optional GPU inference.

Key insight: Box2D physics is CPU-bound. GPU doesn't help for small networks.
The real speedup comes from:
1. Multiprocessing for parallel environment evaluation (CPU cores)
2. GPU for batched inference ONLY when networks are large enough (>10K params)
3. Aggressive parallelism: use ALL CPU cores

For truly GPU-native evolution, we'd need JAX+Brax (physics on GPU).
This module provides a hybrid approach that works with standard gymnasium.
"""

import numpy as np
import gymnasium as gym
import time
import multiprocessing as mp
import os

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class PolicyNetwork:
    """Numpy policy — fast for small networks on CPU."""
    def __init__(self, obs_dim=8, act_dim=4, hidden1=64, hidden2=32):
        self.shapes = [
            (obs_dim, hidden1), (hidden1,),
            (hidden1, hidden2), (hidden2,),
            (hidden2, act_dim), (act_dim,),
        ]
        self.sizes = [np.prod(s) for s in self.shapes]
        self.n_params = sum(self.sizes)
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def forward(self, x, params):
        idx = 0
        for i in range(0, len(self.shapes), 2):
            w = params[idx:idx+self.sizes[i]].reshape(self.shapes[i])
            idx += self.sizes[i]
            b = params[idx:idx+self.sizes[i+1]]
            idx += self.sizes[i+1]
            x = x @ w + b
            if i < len(self.shapes) - 2:
                x = np.tanh(x)
        return x

    def act(self, obs, params):
        return int(np.argmax(self.forward(obs, params)))


def evaluate(policy, params, n_episodes=5, env_name="LunarLander-v3"):
    """Evaluate a candidate — runs in subprocess via multiprocessing."""
    env = gym.make(env_name)
    total = 0.0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 1000)
        done, steps, ep_r = False, 0, 0.0
        while not done and steps < 1000:
            action = policy.act(obs, params)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_r += reward
            done = terminated or truncated
            steps += 1
        total += ep_r
    env.close()
    return total / n_episodes


class CMAES:
    """CMA-ES optimizer."""
    def __init__(self, n_params, sigma0=0.5, pop_size=None):
        self.n = n_params
        self.sigma = sigma0
        self.mean = np.random.randn(n_params) * 0.1
        self.lam = pop_size or (4 + int(3 * np.log(n_params)))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = w / w.sum()
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
        try:
            A = np.linalg.cholesky(self.C)
        except np.linalg.LinAlgError:
            self.C = np.eye(self.n); A = np.eye(self.n)
        return [self.mean + self.sigma * (A @ np.random.randn(self.n)) for _ in range(self.lam)]

    def tell(self, solutions, fitnesses):
        idx = np.argsort(fitnesses)[::-1]
        selected = np.array([solutions[i] for i in idx[:self.mu]])
        old_mean = self.mean.copy()
        self.mean = self.weights @ selected
        if self.use_diagonal:
            invsqrt = 1.0 / np.sqrt(self.C_diag)
            self.ps = (1-self.cs)*self.ps + np.sqrt(self.cs*(2-self.cs)*self.mueff)*invsqrt*(self.mean-old_mean)/self.sigma
        else:
            try: Cinv = np.linalg.inv(np.linalg.cholesky(self.C))
            except: Cinv = np.eye(self.n)
            self.ps = (1-self.cs)*self.ps + np.sqrt(self.cs*(2-self.cs)*self.mueff)*Cinv@(self.mean-old_mean)/self.sigma
        hs = int(np.linalg.norm(self.ps)/np.sqrt(1-(1-self.cs)**(2*(self.gen+1))) < (1.4+2/(self.n+1))*self.chi_n)
        self.pc = (1-self.cc)*self.pc + hs*np.sqrt(self.cc*(2-self.cc)*self.mueff)*(self.mean-old_mean)/self.sigma
        artmp = (selected - old_mean) / self.sigma
        if self.use_diagonal:
            self.C_diag = (1-self.c1-self.cmu)*self.C_diag + self.c1*(self.pc**2+(1-hs)*self.cc*(2-self.cc)*self.C_diag) + self.cmu*np.sum(self.weights[:,None]*artmp**2, axis=0)
            self.C_diag = np.maximum(self.C_diag, 1e-20)
        else:
            self.C = (1-self.c1-self.cmu)*self.C + self.c1*(np.outer(self.pc,self.pc)+(1-hs)*self.cc*(2-self.cc)*self.C) + self.cmu*sum(w*np.outer(a,a) for w,a in zip(self.weights,artmp))
        self.sigma *= np.exp((self.cs/self.ds)*(np.linalg.norm(self.ps)/self.chi_n-1))
        self.sigma = np.clip(self.sigma, 1e-20, 10.0)
        self.gen += 1


def run(params=None, device="cpu", callback=None):
    """Run high-performance CMA-ES with maximum CPU parallelism.
    
    Uses ALL available CPU cores for parallel evaluation.
    GPU is noted but not used — Box2D is CPU-bound.
    """
    params = params or {}
    max_evals = params.get("max_evals", 200000)
    sigma0 = params.get("sigma0", 0.5)
    eval_episodes = params.get("eval_episodes", 5)
    hidden1 = params.get("hidden1", 64)
    hidden2 = params.get("hidden2", 32)
    env_name = params.get("env_name", "LunarLander-v3")
    n_workers = params.get("n_workers", min(os.cpu_count() or 1, 32))
    
    # Detect GPU (for reporting, not used for Box2D)
    gpu_name = "none"
    if HAS_TORCH and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    
    print(f"🖥️  High-Performance CMA-ES")
    print(f"Environment: {env_name}")
    print(f"CPU workers: {n_workers} | GPU: {gpu_name} (not used — Box2D is CPU-bound)")
    
    policy = PolicyNetwork(hidden1=hidden1, hidden2=hidden2)
    print(f"Network: {policy.n_params} params | Budget: {max_evals} evals")
    
    cma = CMAES(policy.n_params, sigma0=sigma0)
    print(f"CMA-ES population: {cma.lam}")
    
    best_ever = -float("inf")
    best_params = None
    total_evals = 0
    start = time.time()
    
    while total_evals < max_evals:
        candidates = cma.ask()
        
        with mp.Pool(n_workers) as pool:
            fitnesses = pool.starmap(evaluate, 
                [(policy, c, eval_episodes, env_name) for c in candidates])
        fitnesses = np.array(fitnesses)
        total_evals += len(candidates) * eval_episodes
        
        cma.tell(candidates, fitnesses)
        
        gb = np.max(fitnesses)
        gm = np.mean(fitnesses)
        if gb > best_ever:
            best_ever = gb
            best_params = candidates[np.argmax(fitnesses)].copy()
        
        elapsed = time.time() - start
        eps = total_evals / elapsed if elapsed > 0 else 0
        
        entry = {
            "generation": cma.gen,
            "best_fitness": float(gb),
            "best_ever": float(best_ever),
            "mean_fitness": float(gm),
            "std_fitness": float(np.std(fitnesses)),
            "sigma": float(cma.sigma),
            "total_evals": total_evals,
            "elapsed": round(elapsed, 1),
        }
        if callback:
            callback(entry)
        
        solved = "✅ SOLVED!" if best_ever >= 200 else ""
        print(f"Gen {cma.gen:4d} | Best: {gb:8.1f} | Ever: {best_ever:8.1f} | "
              f"Mean: {gm:8.1f} | σ: {cma.sigma:.4f} | Evals: {total_evals:6d} | "
              f"{eps:.0f} e/s | {elapsed:6.1f}s {solved}")
        
        if best_ever >= 200:
            print(f"\n🎉 SOLVED! Score: {best_ever:.1f} at {eps:.0f} evals/sec")
            break
    
    if best_params is not None:
        final_scores = [evaluate(policy, best_params, 1, env_name) for _ in range(20)]
        final_mean, final_std = float(np.mean(final_scores)), float(np.std(final_scores))
    else:
        final_mean = final_std = 0.0
    
    elapsed = time.time() - start
    return {
        "method": "GPU CMA-ES (CPU-parallel)",
        "gpu": gpu_name,
        "cpu_workers": n_workers,
        "best_ever": float(best_ever),
        "final_mean": final_mean, "final_std": final_std,
        "total_evals": total_evals,
        "generations": cma.gen,
        "n_params": policy.n_params,
        "evals_per_second": round(total_evals / elapsed, 1),
        "elapsed_seconds": round(elapsed, 1),
        "solved": best_ever >= 200,
    }
