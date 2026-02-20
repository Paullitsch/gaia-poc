"""
Island Model: Multiple CMA-ES populations evolving in parallel with periodic migration.

This is the evolutionary computing analog of GAIA's decentralized vision:
- Each "island" runs an independent CMA-ES with different hyperparameters
- Periodically, the best individuals migrate between islands
- This maintains diversity while allowing good solutions to spread
- Directly maps to distributed GAIA nodes

NO BACKPROPAGATION. Pure evolutionary dynamics with population structure.
"""

import numpy as np
import gymnasium as gym
import time
import multiprocessing as mp
import os


class PolicyNetwork:
    """Simple numpy-based policy network."""

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
        out = h @ layers[4] + layers[5]
        return out

    def act(self, obs, params):
        logits = self.forward(obs, params)
        return int(np.argmax(logits))


def evaluate(policy, params, n_episodes=5):
    """Evaluate a parameter vector."""
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


class IslandCMAES:
    """CMA-ES instance for one island."""

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
        self.cmu = min(
            1 - self.c1,
            2 * (self.mueff - 2 + 1 / self.mueff) / ((n_params + 2) ** 2 + self.mueff)
        )
        self.pc = np.zeros(n_params)

        self.use_diagonal = n_params > 2000
        if self.use_diagonal:
            self.C_diag = np.ones(n_params)
        else:
            self.C = np.eye(n_params)

        self.chi_n = np.sqrt(n_params) * (1 - 1 / (4 * n_params) + 1 / (21 * n_params ** 2))
        self.gen = 0
        
        # Track best for this island
        self.best_fitness = -float("inf")
        self.best_params = None

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

        # Track best
        gen_best_idx = np.argmax(fitnesses)
        if fitnesses[gen_best_idx] > self.best_fitness:
            self.best_fitness = fitnesses[gen_best_idx]
            self.best_params = solutions[gen_best_idx].copy()

        old_mean = self.mean.copy()
        self.mean = self.weights @ selected

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

        hs = int(np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) < (1.4 + 2 / (self.n + 1)) * self.chi_n)
        self.pc = (1 - self.cc) * self.pc + \
            hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (self.mean - old_mean) / self.sigma

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

        self.sigma *= np.exp((self.cs / self.ds) * (np.linalg.norm(self.ps) / self.chi_n - 1))
        self.sigma = np.clip(self.sigma, 1e-20, 10.0)
        self.gen += 1

    def inject(self, params, fitness):
        """Inject a migrant into this island's distribution.
        
        Blends the migrant into the mean with weight proportional to how much
        better it is than the current island's best.
        """
        if fitness > self.best_fitness:
            # Strong migration: shift mean 30% toward migrant
            alpha = 0.3
            self.mean = (1 - alpha) * self.mean + alpha * params
            print(f"    💫 Strong migration: fitness {fitness:.1f} > island best {self.best_fitness:.1f}")
        else:
            # Weak migration: just add diversity (10% blend)
            alpha = 0.1
            self.mean = (1 - alpha) * self.mean + alpha * params
            print(f"    🌊 Weak migration: fitness {fitness:.1f}")


class IslandModel:
    """
    Multiple CMA-ES islands with periodic migration.
    
    Migration topology: Ring (each island sends to the next)
    Migration policy: Best individual from each island
    Migration interval: Every N generations
    """

    def __init__(self, n_params, n_islands=4, migration_interval=10, configs=None):
        self.n_islands = n_islands
        self.migration_interval = migration_interval
        
        # Each island can have different hyperparameters
        if configs is None:
            configs = [
                {"sigma0": 0.3, "label": "Conservative"},   # Low sigma, careful search
                {"sigma0": 0.5, "label": "Standard"},        # Default
                {"sigma0": 0.8, "label": "Explorative"},     # High sigma, broad search
                {"sigma0": 1.0, "label": "Wild"},             # Very high sigma, maximum diversity
            ]
        
        self.islands = []
        self.labels = []
        for i in range(n_islands):
            cfg = configs[i % len(configs)]
            island = IslandCMAES(n_params, sigma0=cfg["sigma0"])
            self.islands.append(island)
            self.labels.append(cfg.get("label", f"Island {i}"))
        
        self.total_migrations = 0
        self.migration_history = []

    def migrate(self):
        """Ring migration: each island sends its best to the next."""
        migrants = []
        for island in self.islands:
            if island.best_params is not None:
                migrants.append((island.best_params.copy(), island.best_fitness))
            else:
                migrants.append(None)
        
        n_migrated = 0
        for i in range(self.n_islands):
            # Send to next island (ring topology)
            target = (i + 1) % self.n_islands
            if migrants[i] is not None:
                params, fitness = migrants[i]
                self.islands[target].inject(params, fitness)
                n_migrated += 1
        
        self.total_migrations += n_migrated
        return n_migrated


def run(params=None, device="cpu", callback=None):
    """Run Island Model CMA-ES on LunarLander.
    
    Args:
        params: dict with optional keys:
            - max_evals: max evaluations (default 100000)
            - n_islands: number of islands (default 4)
            - migration_interval: generations between migrations (default 10)
            - eval_episodes: episodes per evaluation (default 5)
        callback: progress callback
    
    Returns:
        dict with results
    """
    params = params or {}
    max_evals = params.get("max_evals", 100000)
    n_islands = params.get("n_islands", 4)
    migration_interval = params.get("migration_interval", 10)
    eval_episodes = params.get("eval_episodes", 5)
    n_workers = params.get("n_workers", min(os.cpu_count() or 1, 16))

    policy = PolicyNetwork()
    print(f"🏝️  Island Model: {n_islands} islands, migration every {migration_interval} gens")
    print(f"Network: {policy.n_params} parameters")
    print(f"Budget: {max_evals} evaluations")
    print(f"Workers: {n_workers}")

    island_configs = [
        {"sigma0": 0.3, "label": "🟢 Conservative"},
        {"sigma0": 0.5, "label": "🔵 Standard"},
        {"sigma0": 0.8, "label": "🟡 Explorative"},
        {"sigma0": 1.2, "label": "🔴 Wild"},
    ][:n_islands]

    model = IslandModel(policy.n_params, n_islands=n_islands, 
                         migration_interval=migration_interval, configs=island_configs)

    best_ever = -float("inf")
    best_params = None
    total_evals = 0
    start_time = time.time()
    gen = 0

    while total_evals < max_evals:
        gen += 1
        
        # Each island does one generation
        all_candidates = []
        island_sizes = []
        for island in model.islands:
            candidates = island.ask()
            all_candidates.extend([(policy, c, eval_episodes) for c in candidates])
            island_sizes.append(len(candidates))

        # Evaluate ALL candidates in parallel (across all islands)
        if n_workers > 1:
            with mp.Pool(n_workers) as pool:
                all_fitnesses = pool.starmap(evaluate, all_candidates)
            all_fitnesses = np.array(all_fitnesses)
        else:
            all_fitnesses = np.array([evaluate(policy, c, eval_episodes) for _, c, _ in all_candidates])

        total_evals += len(all_candidates) * eval_episodes

        # Distribute fitnesses back to islands
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

        # Migration
        migrated = 0
        if gen % migration_interval == 0:
            migrated = model.migrate()
            print(f"  🔄 Migration round {gen // migration_interval}: {migrated} migrants sent")

        elapsed = time.time() - start_time
        island_status = " | ".join([f"{model.labels[i]}: {island_bests[i]:.0f}" for i in range(n_islands)])

        entry = {
            "generation": gen,
            "best_fitness": float(max(island_bests)),
            "best_ever": float(best_ever),
            "mean_fitness": float(np.mean(all_fitnesses)),
            "std_fitness": float(np.std(all_fitnesses)),
            "sigma": float(np.mean([isl.sigma for isl in model.islands])),
            "total_evals": total_evals,
            "elapsed": round(elapsed, 1),
            "migrations": model.total_migrations,
        }

        if callback:
            callback(entry)

        solved_str = "✅ SOLVED!" if best_ever >= 200 else ""
        print(f"Gen {gen:4d} | Best: {max(island_bests):8.1f} | Ever: {best_ever:8.1f} | "
              f"Mean: {np.mean(all_fitnesses):8.1f} | Evals: {total_evals:6d} | "
              f"Migrations: {model.total_migrations} | {elapsed:6.1f}s {solved_str}")
        print(f"         {island_status}")

        if best_ever >= 200:
            print(f"\n🎉🏝️  SOLVED with Island Model! Score: {best_ever:.1f}")
            print(f"   Total migrations: {model.total_migrations}")
            break

    # Final evaluation
    if best_params is not None:
        final_scores = [evaluate(policy, best_params, n_episodes=1) for _ in range(20)]
        final_mean = np.mean(final_scores)
        final_std = np.std(final_scores)
    else:
        final_mean = final_std = 0.0

    elapsed = time.time() - start_time

    result = {
        "method": "Island Model CMA-ES",
        "best_ever": float(best_ever),
        "final_mean": float(final_mean),
        "final_std": float(final_std),
        "total_evals": total_evals,
        "generations": gen,
        "n_params": policy.n_params,
        "n_islands": n_islands,
        "total_migrations": model.total_migrations,
        "elapsed_seconds": round(elapsed, 1),
        "solved": best_ever >= 200,
        "island_configs": [{"sigma0": isl.sigma, "best": isl.best_fitness} 
                          for isl in model.islands],
    }

    if best_params is not None:
        np.save("best_island_model_params.npy", best_params)

    return result


if __name__ == "__main__":
    result = run(
        params={"max_evals": 100000, "n_islands": 4, "migration_interval": 10},
        callback=lambda d: None
    )
    import json
    print(f"\n{'='*60}")
    print(f"Final Result: {json.dumps(result, indent=2)}")
