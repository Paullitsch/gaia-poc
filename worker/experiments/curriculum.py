"""
CMA-ES with Reward Shaping + Curriculum Learning.

Shaped rewards guide the agent with intermediate signals.
Curriculum starts easy, gets harder — like biological learning.
"""

import numpy as np
import gymnasium as gym
import time
import json


class PolicyNetwork:
    def __init__(self, obs_dim=8, act_dim=4, hidden1=64, hidden2=32):
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
        out = h @ layers[4] + layers[5]
        return int(np.argmax(out))


def evaluate_shaped(policy, params, n_episodes=5, difficulty=1.0):
    """Evaluate with shaped rewards."""
    env = gym.make("LunarLander-v3")
    total = 0.0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 1000)
        done = False
        ep_reward = 0.0
        steps = 0
        max_height = -float("inf")

        while not done and steps < 1000:
            action = policy.act(obs, params)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            # Shaped rewards (supplement, don't replace)
            shaped = reward

            # Bonus for being alive (survival bonus, decreases with difficulty)
            shaped += (1.0 - difficulty) * 0.1

            # Bonus for low velocity near ground (smooth landing)
            height = next_obs[1]
            vel_y = next_obs[3]
            vel_x = next_obs[2]
            angle = abs(next_obs[4])

            if height < 0.5:  # Near ground
                # Reward slow descent
                shaped += max(0, -vel_y * 0.5) * difficulty
                # Reward low horizontal velocity
                shaped += max(0, 0.5 - abs(vel_x)) * difficulty
                # Reward being upright
                shaped += max(0, 0.3 - angle) * difficulty

            ep_reward += shaped
            obs = next_obs
            done = terminated or truncated
            steps += 1

        total += ep_reward
    env.close()
    return total / n_episodes


def run(params=None, device="cpu", callback=None):
    params = params or {}
    max_evals = params.get("max_evals", 100000)
    eval_episodes = params.get("eval_episodes", 5)

    policy = PolicyNetwork()
    print(f"Curriculum CMA-ES: {policy.n_params} params")

    from experiments.cma_es import CMAES
    cma = CMAES(policy.n_params, sigma0=0.5)

    best_ever = -float("inf")
    best_ever_raw = -float("inf")
    best_params = None
    total_evals = 0
    start_time = time.time()

    # Curriculum: difficulty ramps from 0.3 to 1.0
    max_gens = max_evals // (cma.lam * eval_episodes)

    while total_evals < max_evals:
        # Curriculum difficulty
        progress = min(1.0, cma.gen / max(max_gens * 0.7, 1))
        difficulty = 0.3 + 0.7 * progress

        candidates = cma.ask()
        fitnesses = []
        for c in candidates:
            f = evaluate_shaped(policy, c, eval_episodes, difficulty)
            fitnesses.append(f)
        fitnesses = np.array(fitnesses)
        total_evals += len(candidates) * eval_episodes

        cma.tell(candidates, fitnesses)

        gen_best = np.max(fitnesses)
        if gen_best > best_ever:
            best_ever = gen_best
            best_params = candidates[np.argmax(fitnesses)].copy()

        elapsed = time.time() - start_time

        # Also evaluate best on raw reward (no shaping) periodically
        raw_best = -999
        if cma.gen % 10 == 0 and best_params is not None:
            from experiments.cma_es import evaluate as raw_eval
            raw_best = raw_eval(policy, best_params, 5)
            if raw_best > best_ever_raw:
                best_ever_raw = raw_best

        entry = {
            "generation": cma.gen,
            "best_fitness": float(gen_best),
            "best_ever": float(best_ever),
            "best_ever_raw": float(best_ever_raw),
            "mean_fitness": float(np.mean(fitnesses)),
            "std_fitness": float(np.std(fitnesses)),
            "difficulty": round(difficulty, 3),
            "total_evals": total_evals,
            "elapsed": round(elapsed, 1),
        }
        if callback:
            callback(entry)

        raw_str = f" | Raw: {raw_best:8.1f}" if raw_best > -999 else ""
        solved_str = "✅ SOLVED!" if best_ever_raw >= 200 else ""
        print(f"Gen {cma.gen:4d} | Best: {gen_best:8.1f} | Ever: {best_ever:8.1f}{raw_str} | "
              f"Diff: {difficulty:.2f} | Evals: {total_evals:6d} | {elapsed:6.1f}s {solved_str}")

        if best_ever_raw >= 200:
            break

    # Final raw evaluation
    if best_params is not None:
        from experiments.cma_es import evaluate as raw_eval
        final_scores = [raw_eval(policy, best_params, 1) for _ in range(20)]
        final_mean = float(np.mean(final_scores))
        final_std = float(np.std(final_scores))
    else:
        final_mean = final_std = 0.0

    return {
        "method": "Curriculum-CMA-ES",
        "best_ever_shaped": float(best_ever),
        "best_ever_raw": float(best_ever_raw),
        "final_mean": final_mean,
        "final_std": final_std,
        "total_evals": total_evals,
        "generations": cma.gen,
        "n_params": policy.n_params,
        "elapsed_seconds": round(time.time() - start_time, 1),
        "solved": best_ever_raw >= 200,
    }


if __name__ == "__main__":
    result = run(params={"max_evals": 50000})
    print(f"\nResult: {json.dumps(result, indent=2)}")
