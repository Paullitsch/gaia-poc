"""
GPU-Accelerated CMA-ES for BipedalWalker — continuous control on GPU.

Same GPU optimization as gpu_cma.py but with:
- Continuous action space (4D tanh output)
- Larger network (11.5K params)
- Vectorized BipedalWalker environments
"""

import numpy as np
import gymnasium as gym
import time
import torch
import torch.nn as nn


class ContinuousPolicy(nn.Module):
    """Policy for continuous action spaces."""
    def __init__(self, obs_dim=24, act_dim=4, hidden1=128, hidden2=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden1),
            nn.Tanh(),
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, act_dim),
            nn.Tanh(),  # Actions bounded to [-1, 1]
        )
        self.n_params = sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        return self.net(x)
    
    def load_flat_params(self, flat_params):
        idx = 0
        for p in self.parameters():
            size = p.numel()
            p.data = torch.from_numpy(flat_params[idx:idx+size].reshape(p.shape)).float().to(p.device)
            idx += size


class GPUBipedalEvaluator:
    def __init__(self, n_envs, device, hidden1=128, hidden2=64, hardcore=False):
        env_name = "BipedalWalkerHardcore-v3" if hardcore else "BipedalWalker-v3"
        self.envs = gym.vector.SyncVectorEnv([lambda e=env_name: gym.make(e) for _ in range(n_envs)])
        self.n_envs = n_envs
        self.device = device
        self.policy = ContinuousPolicy(24, 4, hidden1, hidden2).to(device)
        self.policy.eval()
    
    def evaluate_batch(self, candidates, n_episodes=3):
        fitnesses = []
        for params in candidates:
            self.policy.load_flat_params(params)
            fitness = self._eval(n_episodes)
            fitnesses.append(fitness)
        return np.array(fitnesses)
    
    def _eval(self, n_episodes):
        n = min(n_episodes, self.n_envs)
        obs_np, _ = self.envs.reset()
        rewards = np.zeros(self.n_envs)
        done_flags = np.zeros(self.n_envs, dtype=bool)
        
        for _ in range(1600):
            if done_flags[:n].all():
                break
            with torch.no_grad():
                obs_t = torch.from_numpy(obs_np).float().to(self.device)
                actions = self.policy(obs_t).cpu().numpy()
            obs_np, r, term, trunc, _ = self.envs.step(actions)
            for i in range(n):
                if not done_flags[i]:
                    rewards[i] += r[i]
                    if term[i] or trunc[i]:
                        done_flags[i] = True
        
        return float(np.mean(rewards[:n]))
    
    def close(self):
        self.envs.close()


class CMAES:
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
        self.C_diag = np.ones(n_params)
        self.chi_n = np.sqrt(n_params) * (1 - 1 / (4 * n_params) + 1 / (21 * n_params ** 2))
        self.gen = 0

    def ask(self):
        std = self.sigma * np.sqrt(self.C_diag)
        return [self.mean + std * np.random.randn(self.n) for _ in range(self.lam)]

    def tell(self, solutions, fitnesses):
        idx = np.argsort(fitnesses)[::-1]
        selected = np.array([solutions[i] for i in idx[:self.mu]])
        old_mean = self.mean.copy()
        self.mean = self.weights @ selected
        invsqrt = 1.0 / np.sqrt(self.C_diag)
        self.ps = (1-self.cs)*self.ps + np.sqrt(self.cs*(2-self.cs)*self.mueff)*invsqrt*(self.mean-old_mean)/self.sigma
        hs = int(np.linalg.norm(self.ps)/np.sqrt(1-(1-self.cs)**(2*(self.gen+1))) < (1.4+2/(self.n+1))*self.chi_n)
        self.pc = (1-self.cc)*self.pc + hs*np.sqrt(self.cc*(2-self.cc)*self.mueff)*(self.mean-old_mean)/self.sigma
        artmp = (selected - old_mean) / self.sigma
        self.C_diag = (1-self.c1-self.cmu)*self.C_diag + self.c1*(self.pc**2+(1-hs)*self.cc*(2-self.cc)*self.C_diag) + self.cmu*np.sum(self.weights[:,None]*artmp**2, axis=0)
        self.C_diag = np.maximum(self.C_diag, 1e-20)
        self.sigma *= np.exp((self.cs/self.ds)*(np.linalg.norm(self.ps)/self.chi_n-1))
        self.sigma = np.clip(self.sigma, 1e-20, 10.0)
        self.gen += 1


def run(params=None, device="cpu", callback=None):
    params = params or {}
    max_evals = params.get("max_evals", 500000)
    eval_episodes = params.get("eval_episodes", 3)
    n_envs = params.get("n_envs", 16)
    hidden1 = params.get("hidden1", 128)
    hidden2 = params.get("hidden2", 64)
    hardcore = params.get("hardcore", False)
    
    if device == "cpu" and torch.cuda.is_available():
        device = "cuda"
    dev = torch.device(device)
    
    env_label = "BipedalWalker-Hardcore" if hardcore else "BipedalWalker"
    print(f"🖥️  GPU CMA-ES for {env_label} on {dev}")
    print(f"Parallel envs: {n_envs}")
    
    evaluator = GPUBipedalEvaluator(n_envs, dev, hidden1, hidden2, hardcore)
    n_params = evaluator.policy.n_params
    print(f"Network: {n_params} params | Budget: {max_evals} evals")
    
    cma = CMAES(n_params, sigma0=0.5)
    print(f"CMA-ES population: {cma.lam}")
    
    best_ever = -float("inf")
    best_params = None
    total_evals = 0
    start = time.time()
    threshold = 300 if not hardcore else 200
    
    while total_evals < max_evals:
        candidates = cma.ask()
        fitnesses = evaluator.evaluate_batch(candidates, eval_episodes)
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
        
        solved = "✅ SOLVED!" if best_ever >= threshold else ""
        print(f"Gen {cma.gen:4d} | Best: {gb:8.1f} | Ever: {best_ever:8.1f} | "
              f"Mean: {gm:8.1f} | σ: {cma.sigma:.4f} | Evals: {total_evals:6d} | "
              f"{eps:.0f} e/s | {elapsed:6.1f}s {solved}")
        
        if best_ever >= threshold:
            print(f"\n🎉🖥️  SOLVED {env_label} on GPU! Score: {best_ever:.1f} @ {eps:.0f} e/s")
            break
    
    evaluator.close()
    elapsed = time.time() - start
    
    return {
        "method": f"GPU CMA-ES ({env_label})",
        "device": str(dev),
        "best_ever": float(best_ever),
        "total_evals": total_evals,
        "generations": cma.gen,
        "n_params": n_params,
        "evals_per_second": round(total_evals / elapsed, 1),
        "elapsed_seconds": round(elapsed, 1),
        "solved": best_ever >= threshold,
        "hardcore": hardcore,
    }
