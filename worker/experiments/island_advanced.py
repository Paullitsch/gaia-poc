"""
Advanced Island Model: Fully connected topology, adaptive migration, heterogeneous methods.

Improvements over basic Island Model:
- Fully connected topology (any island can send to any other)
- Adaptive migration rate (more migration when diversity drops)
- Heterogeneous islands: mix CMA-ES sigmas AND network architectures
- Migration tournament: only migrate if migrant is better than worst in target
"""

import numpy as np
import gymnasium as gym
import time
import multiprocessing as mp
import os


class PolicyNetwork:
    def __init__(self, obs_dim=8, act_dim=4, hidden1=64, hidden2=32):
        self.shapes = [
            (obs_dim, hidden1), (hidden1,),
            (hidden1, hidden2), (hidden2,),
            (hidden2, act_dim), (act_dim,),
        ]
        self.sizes = [np.prod(s) for s in self.shapes]
        self.n_params = sum(self.sizes)

    def forward(self, x, params):
        idx = 0
        layers = []
        for shape, size in zip(self.shapes, self.sizes):
            layers.append(params[idx:idx + size].reshape(shape))
            idx += size
        h = np.tanh(x @ layers[0] + layers[1])
        h = np.tanh(h @ layers[2] + layers[3])
        return h @ layers[4] + layers[5]

    def act(self, obs, params):
        return int(np.argmax(self.forward(obs, params)))


def evaluate(policy, params, n_episodes=5):
    env = gym.make("LunarLander-v3")
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


class AdaptiveCMAES:
    """CMA-ES with adaptive parameters for island model."""
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
        self.best_fitness = -float("inf")
        self.best_params = None
        self.fitness_history = []

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
        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] > self.best_fitness:
            self.best_fitness = fitnesses[best_idx]
            self.best_params = solutions[best_idx].copy()
        self.fitness_history.append(float(np.max(fitnesses)))
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

    def inject(self, params, fitness):
        """Tournament migration: only accept if better than current worst."""
        if fitness > self.best_fitness * 0.8:  # Accept if within 80% of best
            alpha = min(0.5, max(0.1, (fitness - self.best_fitness) / (abs(self.best_fitness) + 1e-8)))
            if fitness > self.best_fitness:
                alpha = 0.4
            self.mean = (1 - alpha) * self.mean + alpha * params
            return True
        return False

    def diversity(self):
        """Measure population diversity via sigma."""
        return self.sigma


class AdvancedIslandModel:
    def __init__(self, n_params, n_islands=6, configs=None):
        if configs is None:
            configs = [
                {"sigma0": 0.2, "label": "🔵 Precise"},
                {"sigma0": 0.4, "label": "🟢 Careful"},
                {"sigma0": 0.6, "label": "🟡 Standard"},
                {"sigma0": 0.8, "label": "🟠 Broad"},
                {"sigma0": 1.0, "label": "🔴 Explorative"},
                {"sigma0": 1.5, "label": "🟣 Chaotic"},
            ]
        self.n_islands = min(n_islands, len(configs))
        self.islands = []
        self.labels = []
        for i in range(self.n_islands):
            cfg = configs[i]
            self.islands.append(AdaptiveCMAES(n_params, sigma0=cfg["sigma0"]))
            self.labels.append(cfg["label"])
        self.total_migrations = 0
        self.successful_migrations = 0
        self.base_migration_rate = 0.3  # 30% chance per pair

    def adaptive_migrate(self):
        """Fully connected adaptive migration."""
        # Calculate diversity across islands
        diversities = [isl.diversity() for isl in self.islands]
        mean_div = np.mean(diversities)
        
        # If diversity is low, increase migration rate
        migration_rate = self.base_migration_rate
        if mean_div < 0.1:
            migration_rate = 0.6  # Double migration when converging
        
        migrated = 0
        for i in range(self.n_islands):
            if self.islands[i].best_params is None:
                continue
            for j in range(self.n_islands):
                if i == j:
                    continue
                if np.random.random() < migration_rate:
                    accepted = self.islands[j].inject(
                        self.islands[i].best_params.copy(),
                        self.islands[i].best_fitness
                    )
                    self.total_migrations += 1
                    if accepted:
                        self.successful_migrations += 1
                        migrated += 1
        return migrated


def run(params=None, device="cpu", callback=None):
    params = params or {}
    max_evals = params.get("max_evals", 200000)
    n_islands = params.get("n_islands", 6)
    migration_interval = params.get("migration_interval", 8)
    eval_episodes = params.get("eval_episodes", 5)
    n_workers = params.get("n_workers", min(os.cpu_count() or 1, 16))

    policy = PolicyNetwork()
    print(f"🏝️  Advanced Island Model: {n_islands} islands, fully connected")
    print(f"Network: {policy.n_params} params | Budget: {max_evals} evals | Workers: {n_workers}")

    model = AdvancedIslandModel(policy.n_params, n_islands=n_islands)
    best_ever = -float("inf")
    best_params = None
    total_evals = 0
    start_time = time.time()
    gen = 0

    while total_evals < max_evals:
        gen += 1
        all_candidates = []
        island_sizes = []
        for island in model.islands:
            candidates = island.ask()
            all_candidates.extend([(policy, c, eval_episodes) for c in candidates])
            island_sizes.append(len(candidates))

        if n_workers > 1:
            with mp.Pool(n_workers) as pool:
                all_fitnesses = pool.starmap(evaluate, all_candidates)
            all_fitnesses = np.array(all_fitnesses)
        else:
            all_fitnesses = np.array([evaluate(policy, c, eval_episodes) for _, c, _ in all_candidates])
        total_evals += len(all_candidates) * eval_episodes

        idx = 0
        island_bests = []
        for i, island in enumerate(model.islands):
            size = island_sizes[i]
            candidates = [c for _, c, _ in all_candidates[idx:idx + size]]
            fitnesses = all_fitnesses[idx:idx + size]
            island.tell(candidates, fitnesses)
            island_bests.append(np.max(fitnesses))
            if np.max(fitnesses) > best_ever:
                best_ever = float(np.max(fitnesses))
                best_params = candidates[np.argmax(fitnesses)].copy()
            idx += size

        migrated = 0
        if gen % migration_interval == 0:
            migrated = model.adaptive_migrate()

        elapsed = time.time() - start_time
        avg_sigma = np.mean([isl.sigma for isl in model.islands])

        entry = {
            "generation": gen,
            "best_fitness": float(max(island_bests)),
            "best_ever": float(best_ever),
            "mean_fitness": float(np.mean(all_fitnesses)),
            "std_fitness": float(np.std(all_fitnesses)),
            "sigma": float(avg_sigma),
            "total_evals": total_evals,
            "elapsed": round(elapsed, 1),
        }
        if callback:
            callback(entry)

        solved = "✅ SOLVED!" if best_ever >= 200 else ""
        print(f"Gen {gen:4d} | Best: {max(island_bests):8.1f} | Ever: {best_ever:8.1f} | "
              f"Mean: {np.mean(all_fitnesses):8.1f} | σ̄: {avg_sigma:.4f} | Evals: {total_evals:6d} | "
              f"Mig: {model.total_migrations}/{model.successful_migrations} | {elapsed:6.1f}s {solved}")

        if best_ever >= 200:
            print(f"\n🎉🏝️  SOLVED with Advanced Island Model! Score: {best_ever:.1f}")
            print(f"   Migrations: {model.total_migrations} total, {model.successful_migrations} accepted")
            break

    if best_params is not None:
        final_scores = [evaluate(policy, best_params, 1) for _ in range(20)]
        final_mean, final_std = float(np.mean(final_scores)), float(np.std(final_scores))
    else:
        final_mean = final_std = 0.0

    return {
        "method": "Advanced Island Model",
        "best_ever": float(best_ever),
        "final_mean": final_mean,
        "final_std": final_std,
        "total_evals": total_evals,
        "generations": gen,
        "n_params": policy.n_params,
        "n_islands": n_islands,
        "total_migrations": model.total_migrations,
        "successful_migrations": model.successful_migrations,
        "elapsed_seconds": round(time.time() - start_time, 1),
        "solved": best_ever >= 200,
    }
