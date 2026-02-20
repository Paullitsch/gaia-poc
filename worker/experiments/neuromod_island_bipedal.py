"""
Neuromodulated Island Model for BipedalWalker-v3.

The GAIA vision applied to locomotion:
- Neuromodulatory plasticity (biologically plausible within-episode learning)
- Island Model (decentralized population evolution)
- Continuous action space (4D hip/knee torques)

Each island evolves neuromodulated agents for walking. Migration shares
the best biologically-plausible walkers across the network.
"""

import numpy as np
import gymnasium as gym
import time
import multiprocessing as mp
import os

# BipedalWalker constants
OBS_DIM = 24
ACT_DIM = 4  # continuous: hip1, knee1, hip2, knee2 torques in [-1, 1]
SOLVED_THRESHOLD = 300


class NeuromodBipedalPolicy:
    """Neuromodulated policy for BipedalWalker.
    
    Main network: 24 -> 128 -> 64 -> 4 (tanh output)
    Modulation network: 24 -> 16 -> 1 (produces learning rate signal)
    Plasticity params: 3 (one per layer pair)
    Total params: ~12,000
    """
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden1=128, hidden2=64, mod_hidden=16):
        self.main_shapes = [
            (obs_dim, hidden1), (hidden1,),
            (hidden1, hidden2), (hidden2,),
            (hidden2, act_dim), (act_dim,),
        ]
        self.main_sizes = [np.prod(s) for s in self.main_shapes]
        self.n_main = sum(self.main_sizes)
        
        self.mod_shapes = [
            (obs_dim, mod_hidden), (mod_hidden,),
            (mod_hidden, 1), (1,),
        ]
        self.mod_sizes = [np.prod(s) for s in self.mod_shapes]
        self.n_mod = sum(self.mod_sizes)
        
        self.n_plasticity = 3
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
            x = np.tanh(x)  # tanh for all layers including output (continuous actions in [-1,1])
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


def evaluate_neuromod_bipedal(policy, genome, n_episodes=3):
    """Evaluate a neuromodulated agent on BipedalWalker."""
    main_w, mod_w, plast = policy.decode(genome)
    working_w = main_w.copy()
    env = gym.make("BipedalWalker-v3")
    total_reward = 0.0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 1000)
        done, ep_reward, steps = False, 0.0, 0
        traces = np.zeros_like(working_w)

        while not done and steps < 1600:  # BipedalWalker has longer episodes
            action, activations = policy.forward_main(obs, working_w)
            mod_signal = policy.forward_mod(obs, mod_w)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            # Hebbian trace update
            idx = 0
            for layer_i in range(3):
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
        for layer_i in range(3):
            w_size = policy.main_sizes[layer_i * 2]
            b_size = policy.main_sizes[layer_i * 2 + 1]
            lr = plast[layer_i] if layer_i < len(plast) else 0.001
            working_w[idx:idx + w_size] += lr * traces[idx:idx + w_size]
            idx += w_size + b_size

        total_reward += ep_reward
    env.close()
    return total_reward / n_episodes


class IslandCMAES:
    """Diagonal CMA-ES for one island."""
    def __init__(self, n_params, sigma0=0.5):
        self.n = n_params
        self.sigma = sigma0
        self.mean = np.random.randn(n_params) * 0.1
        self.lam = 4 + int(3 * np.log(n_params))
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
        self.C_diag = np.ones(n_params)
        self.chi_n = np.sqrt(n_params) * (1 - 1 / (4 * n_params) + 1 / (21 * n_params ** 2))
        self.gen = 0
        self.best_fitness = -float("inf")
        self.best_params = None

    def ask(self):
        std = self.sigma * np.sqrt(self.C_diag)
        return [self.mean + std * np.random.randn(self.n) for _ in range(self.lam)]

    def tell(self, solutions, fitnesses):
        idx = np.argsort(fitnesses)[::-1]
        selected = np.array([solutions[i] for i in idx[:self.mu]])
        bi = np.argmax(fitnesses)
        if fitnesses[bi] > self.best_fitness:
            self.best_fitness = fitnesses[bi]
            self.best_params = solutions[bi].copy()
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

    def inject(self, params, fitness):
        if fitness > self.best_fitness * 0.8:
            alpha = 0.3 if fitness > self.best_fitness else 0.1
            self.mean = (1 - alpha) * self.mean + alpha * params
            return True
        return False


def run(params=None, device="cpu", callback=None):
    params = params or {}
    max_evals = params.get("max_evals", 500000)
    n_islands = params.get("n_islands", 4)
    migration_interval = params.get("migration_interval", 10)
    eval_episodes = params.get("eval_episodes", 3)
    n_workers = params.get("n_workers", min(os.cpu_count() or 1, 16))
    plateau_patience = params.get("plateau_patience", 300)

    policy = NeuromodBipedalPolicy()
    print(f"🧬🦿 Neuromod Island BipedalWalker: {n_islands} islands")
    print(f"Params: {policy.n_params} (main: {policy.n_main}, mod: {policy.n_mod}, plast: {policy.n_plasticity})")
    print(f"Budget: {max_evals} evals | Workers: {n_workers} | Plateau patience: {plateau_patience}")

    configs = [
        {"sigma0": 0.3, "label": "🟢 Conservative"},
        {"sigma0": 0.5, "label": "🔵 Standard"},
        {"sigma0": 0.8, "label": "🟡 Explorative"},
        {"sigma0": 1.2, "label": "🔴 Wild"},
    ]
    islands = [IslandCMAES(policy.n_params, sigma0=configs[i % len(configs)]["sigma0"]) for i in range(n_islands)]

    best_ever = -float("inf")
    best_params = None
    total_evals = 0
    total_migrations = 0
    start_time = time.time()
    gen = 0
    plateau_counter = 0
    last_best = -float("inf")

    while total_evals < max_evals:
        gen += 1
        all_candidates = []
        island_sizes = []
        for island in islands:
            candidates = island.ask()
            all_candidates.extend([(policy, c, eval_episodes) for c in candidates])
            island_sizes.append(len(candidates))

        if n_workers > 1:
            with mp.Pool(n_workers) as pool:
                all_fitnesses = pool.starmap(evaluate_neuromod_bipedal, all_candidates)
            all_fitnesses = np.array(all_fitnesses)
        else:
            all_fitnesses = np.array([evaluate_neuromod_bipedal(policy, c, eval_episodes) for _, c, _ in all_candidates])
        total_evals += len(all_candidates) * eval_episodes

        idx = 0
        island_bests = []
        for i, island in enumerate(islands):
            size = island_sizes[i]
            candidates = [c for _, c, _ in all_candidates[idx:idx + size]]
            fitnesses = all_fitnesses[idx:idx + size]
            island.tell(candidates, fitnesses)
            island_bests.append(np.max(fitnesses))
            if np.max(fitnesses) > best_ever:
                best_ever = float(np.max(fitnesses))
                best_params = candidates[np.argmax(fitnesses)].copy()
            idx += size

        # Ring migration
        if gen % migration_interval == 0:
            for i in range(n_islands):
                target = (i + 1) % n_islands
                if islands[i].best_params is not None:
                    islands[target].inject(islands[i].best_params.copy(), islands[i].best_fitness)
                    total_migrations += 1

        # Plateau detection
        if best_ever > last_best + 1.0:
            last_best = best_ever
            plateau_counter = 0
        else:
            plateau_counter += 1

        elapsed = time.time() - start_time
        avg_sigma = np.mean([isl.sigma for isl in islands])
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

        solved = "✅ SOLVED!" if best_ever >= SOLVED_THRESHOLD else ""
        print(f"Gen {gen:4d} | Best: {max(island_bests):8.1f} | Ever: {best_ever:8.1f} | "
              f"Mean: {np.mean(all_fitnesses):8.1f} | σ̄: {avg_sigma:.4f} | Evals: {total_evals:>8d} | {elapsed:6.1f}s {solved}")

        if best_ever >= SOLVED_THRESHOLD:
            print(f"\n🎉🧬🦿 SOLVED BipedalWalker with Neuromod Islands! Score: {best_ever:.1f}")
            break

        if plateau_counter >= plateau_patience:
            print(f"\n⏹️ Plateau detected after {plateau_patience} generations without improvement.")
            break

    if best_params is not None:
        final_scores = [evaluate_neuromod_bipedal(policy, best_params, 1) for _ in range(20)]
        final_mean, final_std = float(np.mean(final_scores)), float(np.std(final_scores))
    else:
        final_mean = final_std = 0.0

    return {
        "method": "Neuromod Island BipedalWalker",
        "environment": "BipedalWalker-v3",
        "best_ever": float(best_ever),
        "final_mean": final_mean, "final_std": final_std,
        "total_evals": total_evals, "generations": gen,
        "n_params": policy.n_params, "n_islands": n_islands,
        "total_migrations": total_migrations,
        "elapsed_seconds": round(time.time() - start_time, 1),
        "solved": best_ever >= SOLVED_THRESHOLD,
    }
