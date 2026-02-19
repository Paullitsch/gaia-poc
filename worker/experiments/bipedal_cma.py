"""
CMA-ES for BipedalWalker-v3 — Phase 8.

BipedalWalker is a MUCH harder challenge than LunarLander:
- Continuous action space (4D: hip/knee torques for 2 legs)
- 24D observation (hull angle, velocities, joint angles, leg contact, lidar)
- Solved threshold: 300 (over 100 episodes)
- Requires coordinated locomotion — a real test for gradient-free methods

Uses PyTorch for GPU-accelerated policy evaluation when available.
Falls back to numpy on CPU.
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

# BipedalWalker constants
OBS_DIM = 24
ACT_DIM = 4  # continuous: hip1, knee1, hip2, knee2 torques in [-1, 1]
SOLVED_THRESHOLD = 300


class PolicyNetwork:
    """Larger network for BipedalWalker's harder task.
    
    Architecture: 24 -> 128 -> 64 -> 4 (tanh output for continuous actions)
    Total params: 24*128 + 128 + 128*64 + 64 + 64*4 + 4 = 11,588
    """
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden1=128, hidden2=64):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.shapes = [
            (obs_dim, hidden1), (hidden1,),
            (hidden1, hidden2), (hidden2,),
            (hidden2, act_dim), (act_dim,),
        ]
        self.sizes = [np.prod(s) for s in self.shapes]
        self.n_params = sum(self.sizes)

    def act(self, obs, params):
        """Forward pass -> tanh for continuous actions in [-1, 1]."""
        idx = 0
        layers = []
        for shape, size in zip(self.shapes, self.sizes):
            layers.append(params[idx:idx + size].reshape(shape))
            idx += size
        h = np.tanh(obs @ layers[0] + layers[1])
        h = np.tanh(h @ layers[2] + layers[3])
        out = np.tanh(h @ layers[4] + layers[5])  # tanh for [-1, 1] actions
        return out


class TorchPolicy:
    """GPU-accelerated policy for batch evaluation."""
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden1=128, hidden2=64, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.shapes = [
            (obs_dim, hidden1), (hidden1,),
            (hidden1, hidden2), (hidden2,),
            (hidden2, act_dim), (act_dim,),
        ]
        self.sizes = [np.prod(s) for s in self.shapes]
        self.n_params = sum(self.sizes)

    def act(self, obs, params):
        """Single observation -> action (numpy, for env interaction)."""
        idx = 0
        layers = []
        for shape, size in zip(self.shapes, self.sizes):
            layers.append(params[idx:idx + size].reshape(shape))
            idx += size
        h = np.tanh(obs @ layers[0] + layers[1])
        h = np.tanh(h @ layers[2] + layers[3])
        out = np.tanh(h @ layers[4] + layers[5])
        return out

    def batch_forward(self, obs_batch, params_batch):
        """Batch forward pass on GPU.
        
        obs_batch: (batch, obs_dim) numpy
        params_batch: list of flat numpy param vectors
        Returns: (batch, act_dim) numpy actions
        """
        with torch.no_grad():
            obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
            results = []
            for params in params_batch:
                idx = 0
                layers = []
                for shape, size in zip(self.shapes, self.sizes):
                    layers.append(torch.tensor(
                        params[idx:idx + size].reshape(shape),
                        dtype=torch.float32, device=self.device
                    ))
                    idx += size
                h = torch.tanh(obs_t @ layers[0] + layers[1])
                h = torch.tanh(h @ layers[2] + layers[3])
                out = torch.tanh(h @ layers[4] + layers[5])
                results.append(out.cpu().numpy())
            return results


def evaluate(policy, params, n_episodes=3, render=False):
    """Evaluate a parameter vector on BipedalWalker."""
    env = gym.make("BipedalWalker-v3")
    total = 0.0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 1000)
        done = False
        ep_reward = 0.0
        steps = 0
        while not done and steps < 1600:
            action = policy.act(obs, params)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            steps += 1
        total += ep_reward
    env.close()
    return total / n_episodes


def evaluate_shaped(policy, params, n_episodes=3, difficulty=1.0):
    """Evaluate with reward shaping for curriculum learning."""
    env = gym.make("BipedalWalker-v3")
    total = 0.0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 1000)
        done = False
        ep_reward = 0.0
        steps = 0
        max_x = 0.0

        while not done and steps < 1600:
            action = policy.act(obs, params)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            shaped = reward

            # Forward progress bonus (encourage moving right)
            hull_x_vel = next_obs[2]
            if hull_x_vel > 0:
                shaped += hull_x_vel * 0.3 * (1.0 - difficulty * 0.5)

            # Upright bonus (don't fall over)
            hull_angle = abs(next_obs[0])
            if hull_angle < 0.4:
                shaped += (0.4 - hull_angle) * 0.5 * (1.0 - difficulty * 0.5)

            # Survival bonus (decreases with difficulty)
            shaped += (1.0 - difficulty) * 0.05

            ep_reward += shaped
            obs = next_obs
            done = terminated or truncated
            steps += 1

        total += ep_reward
    env.close()
    return total / n_episodes


def run(params=None, device="cpu", callback=None):
    """Run CMA-ES on BipedalWalker-v3.
    
    Args:
        params: dict with optional keys:
            - max_evals: max evaluations (default 500000)
            - eval_episodes: episodes per evaluation (default 3)
            - hidden1/hidden2: network size (default 128/64)
            - use_curriculum: use shaped rewards (default True)
            - sigma0: initial step size (default 0.3)
        device: 'cpu' or 'cuda'
        callback: progress callback
    """
    params = params or {}
    max_evals = params.get("max_evals", 500000)
    eval_episodes = params.get("eval_episodes", 3)
    hidden1 = params.get("hidden1", 128)
    hidden2 = params.get("hidden2", 64)
    use_curriculum = params.get("use_curriculum", True)
    sigma0 = params.get("sigma0", 0.3)
    n_workers = params.get("n_workers", min(os.cpu_count() or 1, 16))

    # Use torch policy if available, else numpy
    if HAS_TORCH and device != "cpu":
        policy = TorchPolicy(hidden1=hidden1, hidden2=hidden2, device=device)
        print(f"🔥 Using PyTorch on {policy.device}")
    else:
        policy = PolicyNetwork(hidden1=hidden1, hidden2=hidden2)
        print(f"💻 Using NumPy (CPU)")

    print(f"Network: {policy.n_params} parameters")
    print(f"Budget: {max_evals} evaluations")
    print(f"Curriculum: {'ON' if use_curriculum else 'OFF'}")
    print(f"Parallel workers: {n_workers}")

    # Import CMA-ES from our cma_es module
    from experiments.cma_es import CMAES
    cma = CMAES(policy.n_params, sigma0=sigma0)
    print(f"CMA-ES population: {cma.lam}")

    best_ever = -float("inf")
    best_ever_raw = -float("inf")
    best_params = None
    total_evals = 0
    start_time = time.time()
    max_gens = max_evals // (cma.lam * eval_episodes)

    eval_fn = evaluate_shaped if use_curriculum else evaluate

    while total_evals < max_evals:
        # Curriculum difficulty ramps over training
        progress = min(1.0, cma.gen / max(max_gens * 0.7, 1))
        difficulty = 0.3 + 0.7 * progress

        candidates = cma.ask()

        # Parallel evaluation
        if n_workers > 1:
            if use_curriculum:
                args = [(policy, c, eval_episodes, difficulty) for c in candidates]
                with mp.Pool(n_workers) as pool:
                    fitnesses = pool.starmap(evaluate_shaped, args)
            else:
                args = [(policy, c, eval_episodes) for c in candidates]
                with mp.Pool(n_workers) as pool:
                    fitnesses = pool.starmap(evaluate, args)
            fitnesses = np.array(fitnesses)
        else:
            if use_curriculum:
                fitnesses = np.array([evaluate_shaped(policy, c, eval_episodes, difficulty) for c in candidates])
            else:
                fitnesses = np.array([evaluate(policy, c, eval_episodes) for c in candidates])

        total_evals += len(candidates) * eval_episodes
        cma.tell(candidates, fitnesses)

        gen_best = np.max(fitnesses)
        gen_mean = np.mean(fitnesses)

        if gen_best > best_ever:
            best_ever = gen_best
            best_params = candidates[np.argmax(fitnesses)].copy()

        elapsed = time.time() - start_time

        # Raw evaluation every 20 gens
        raw_best = -999
        if cma.gen % 20 == 0 and best_params is not None:
            raw_best = evaluate(policy, best_params, 5)
            if raw_best > best_ever_raw:
                best_ever_raw = raw_best

        entry = {
            "generation": cma.gen,
            "best_fitness": float(gen_best),
            "best_ever": float(best_ever),
            "best_ever_raw": float(best_ever_raw),
            "mean_fitness": float(gen_mean),
            "sigma": float(cma.sigma),
            "difficulty": round(difficulty, 3) if use_curriculum else 1.0,
            "total_evals": total_evals,
            "elapsed": round(elapsed, 1),
        }
        if callback:
            callback(entry)

        raw_str = f" | Raw: {raw_best:8.1f}" if raw_best > -999 else ""
        solved_str = "✅ SOLVED!" if best_ever_raw >= SOLVED_THRESHOLD else ""
        print(f"Gen {cma.gen:4d} | Best: {gen_best:8.1f} | Ever: {best_ever:8.1f}{raw_str} | "
              f"Mean: {gen_mean:8.1f} | σ: {cma.sigma:.4f} | Evals: {total_evals:6d} | "
              f"{elapsed:6.1f}s {solved_str}")

        if best_ever_raw >= SOLVED_THRESHOLD:
            print(f"\n🎉 SOLVED BipedalWalker with CMA-ES! Score: {best_ever_raw:.1f}")
            break

    # Final robust evaluation
    if best_params is not None:
        final_scores = [evaluate(policy, best_params, 1) for _ in range(20)]
        final_mean = float(np.mean(final_scores))
        final_std = float(np.std(final_scores))
    else:
        final_mean = final_std = 0.0

    return {
        "method": "BipedalWalker-CMA-ES",
        "environment": "BipedalWalker-v3",
        "best_ever": float(best_ever),
        "best_ever_raw": float(best_ever_raw),
        "final_mean": final_mean,
        "final_std": final_std,
        "total_evals": total_evals,
        "generations": cma.gen,
        "n_params": policy.n_params,
        "elapsed_seconds": round(time.time() - start_time, 1),
        "solved": best_ever_raw >= SOLVED_THRESHOLD,
        "curriculum": use_curriculum,
    }


if __name__ == "__main__":
    result = run(params={"max_evals": 200000, "eval_episodes": 3})
    import json
    print(f"\nResult: {json.dumps(result, indent=2)}")
