"""
Population-Based Training for BipedalWalker-v3 — Phase 8.

PBT runs multiple CMA-ES instances with different hyperparameters
simultaneously. Periodically, the worst-performing instances copy
the weights from the best and mutate their hyperparameters.

This is uniquely suited to GAIA's distributed architecture:
each PBT member could run on a different worker.
"""

import numpy as np
import gymnasium as gym
import time
import multiprocessing as mp
import os
import json


OBS_DIM = 24
ACT_DIM = 4
SOLVED_THRESHOLD = 300


class PolicyNetwork:
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden1=128, hidden2=64):
        self.shapes = [
            (obs_dim, hidden1), (hidden1,),
            (hidden1, hidden2), (hidden2,),
            (hidden2, act_dim), (act_dim,),
        ]
        self.sizes = [np.prod(s) for s in self.shapes]
        self.n_params = sum(self.sizes)

    def act(self, obs, params):
        idx = 0
        layers = []
        for shape, size in zip(self.shapes, self.sizes):
            layers.append(params[idx:idx + size].reshape(shape))
            idx += size
        h = np.tanh(obs @ layers[0] + layers[1])
        h = np.tanh(h @ layers[2] + layers[3])
        return np.tanh(h @ layers[4] + layers[5])


def evaluate(policy, params, n_episodes=3):
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


class PBTMember:
    """One member of the PBT population — runs its own CMA-ES."""
    def __init__(self, n_params, sigma0, pop_size=None):
        from experiments.cma_es import CMAES
        self.cma = CMAES(n_params, sigma0=sigma0, pop_size=pop_size)
        self.sigma0 = sigma0
        self.best_ever = -float("inf")
        self.best_params = None
        self.recent_score = -float("inf")  # rolling average of last 5 gens

    def step(self, policy, eval_fn, n_workers=1, eval_episodes=3):
        """Run one CMA-ES generation."""
        candidates = self.cma.ask()
        if n_workers > 1:
            with mp.Pool(n_workers) as pool:
                fitnesses = pool.starmap(eval_fn, [(policy, c, eval_episodes) for c in candidates])
            fitnesses = np.array(fitnesses)
        else:
            fitnesses = np.array([eval_fn(policy, c, eval_episodes) for c in candidates])

        self.cma.tell(candidates, fitnesses)

        gen_best = np.max(fitnesses)
        if gen_best > self.best_ever:
            self.best_ever = gen_best
            self.best_params = candidates[np.argmax(fitnesses)].copy()

        # Rolling average
        self.recent_score = 0.8 * self.recent_score + 0.2 * np.mean(fitnesses)
        return fitnesses


def run(params=None, device="cpu", callback=None):
    """Run PBT on BipedalWalker."""
    params = params or {}
    max_evals = params.get("max_evals", 500000)
    eval_episodes = params.get("eval_episodes", 3)
    n_members = params.get("n_members", 4)
    exploit_interval = params.get("exploit_interval", 20)  # gens between exploit/explore
    n_workers = params.get("n_workers", min(os.cpu_count() or 1, 16))

    policy = PolicyNetwork()
    print(f"PBT for BipedalWalker: {policy.n_params} params")
    print(f"PBT members: {n_members}")
    print(f"Exploit interval: every {exploit_interval} generations")
    print(f"Workers per member: {max(1, n_workers // n_members)}")

    # Initialize members with different hyperparameters
    sigmas = np.linspace(0.1, 0.8, n_members)
    members = [PBTMember(policy.n_params, sigma0=s) for s in sigmas]
    print(f"Initial sigmas: {[f'{s:.2f}' for s in sigmas]}")

    total_evals = 0
    start_time = time.time()
    gen = 0
    workers_per_member = max(1, n_workers // n_members)

    while total_evals < max_evals:
        gen += 1

        # Run one generation for each member
        for i, member in enumerate(members):
            fitnesses = member.step(policy, evaluate, workers_per_member, eval_episodes)
            total_evals += len(fitnesses) * eval_episodes

        # Best across all members
        global_best = max(m.best_ever for m in members)
        best_member = max(range(len(members)), key=lambda i: members[i].best_ever)

        elapsed = time.time() - start_time
        entry = {
            "generation": gen,
            "best_ever": float(global_best),
            "member_scores": [float(m.recent_score) for m in members],
            "member_sigmas": [float(m.cma.sigma) for m in members],
            "best_member": best_member,
            "total_evals": total_evals,
            "elapsed": round(elapsed, 1),
        }
        if callback:
            callback(entry)

        scores_str = " | ".join([f"M{i}:{m.recent_score:+.0f}" for i, m in enumerate(members)])
        solved_str = "✅ SOLVED!" if global_best >= SOLVED_THRESHOLD else ""
        print(f"Gen {gen:4d} | Best: {global_best:8.1f} | {scores_str} | "
              f"Evals: {total_evals:6d} | {elapsed:6.1f}s {solved_str}")

        # PBT Exploit/Explore
        if gen % exploit_interval == 0:
            scores = [m.recent_score for m in members]
            worst_idx = np.argmin(scores)
            best_idx = np.argmax(scores)

            if worst_idx != best_idx and scores[best_idx] > scores[worst_idx] + 10:
                # Exploit: copy best → worst
                members[worst_idx].cma.mean = members[best_idx].cma.mean.copy()
                # Explore: mutate sigma
                members[worst_idx].cma.sigma = members[best_idx].cma.sigma * np.random.uniform(0.8, 1.2)
                print(f"  🔄 PBT: M{worst_idx} ← M{best_idx} (σ: {members[worst_idx].cma.sigma:.4f})")

        if global_best >= SOLVED_THRESHOLD:
            print(f"\n🎉 SOLVED BipedalWalker with PBT! Score: {global_best:.1f}")
            break

    # Final evaluation
    best_member = max(members, key=lambda m: m.best_ever)
    if best_member.best_params is not None:
        final_scores = [evaluate(policy, best_member.best_params, 1) for _ in range(20)]
        final_mean = float(np.mean(final_scores))
        final_std = float(np.std(final_scores))
    else:
        final_mean = final_std = 0.0

    return {
        "method": "BipedalWalker-PBT",
        "environment": "BipedalWalker-v3",
        "best_ever": float(max(m.best_ever for m in members)),
        "final_mean": final_mean,
        "final_std": final_std,
        "total_evals": total_evals,
        "generations": gen,
        "n_params": policy.n_params,
        "n_members": n_members,
        "elapsed_seconds": round(time.time() - start_time, 1),
        "solved": max(m.best_ever for m in members) >= SOLVED_THRESHOLD,
    }


if __name__ == "__main__":
    result = run(params={"max_evals": 200000, "n_members": 3})
    print(f"\nResult: {json.dumps(result, indent=2)}")
