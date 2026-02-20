"""
PPO Baseline — Backpropagation reference for comparison.

This is the CONTROL GROUP. Same environments, same network sizes,
but trained with PPO (gradient-based). Proves that gradient-free
methods can match backprop performance.

Requires: pip install torch
"""

import numpy as np
import gymnasium as gym
import time
import os

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical, Normal
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class PPOPolicy(nn.Module):
    """Same architecture as gradient-free methods for fair comparison."""

    def __init__(self, obs_dim, act_dim, act_type="discrete", hidden=None):
        super().__init__()
        hidden = hidden or [64, 32]
        self.act_type = act_type

        # Build network with same architecture as CMA-ES PolicyNetwork
        layers = []
        dims = [obs_dim] + hidden
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.Tanh())
        self.shared = nn.Sequential(*layers)

        self.actor = nn.Linear(hidden[-1], act_dim)
        self.critic = nn.Linear(hidden[-1], 1)

        if act_type == "continuous":
            self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Count params (actor only, for fair comparison)
        self.n_actor_params = sum(p.numel() for p in self.shared.parameters()) + \
                              sum(p.numel() for p in self.actor.parameters())

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

    def get_action(self, obs):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            logits, value = self(obs_t)
            if self.act_type == "discrete":
                dist = Categorical(logits=logits)
                action = dist.sample()
                return action.item(), dist.log_prob(action).item(), value.item()
            else:
                std = torch.exp(self.log_std)
                dist = Normal(torch.tanh(logits), std)
                action = dist.sample()
                action = torch.clamp(action, -1, 1)
                log_prob = dist.log_prob(action).sum(-1)
                return action.squeeze(0).numpy(), log_prob.item(), value.item()

    def evaluate(self, obs, actions):
        logits, values = self(obs)
        if self.act_type == "discrete":
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
        else:
            std = torch.exp(self.log_std)
            dist = Normal(torch.tanh(logits), std)
            log_probs = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1)
        return log_probs, values.squeeze(-1), entropy


def collect_rollout(policy, env_name, max_steps=1000, n_steps=2048):
    """Collect experience for PPO update."""
    env = gym.make(env_name)
    obs_list, act_list, rew_list, val_list, logp_list, done_list = [], [], [], [], [], []

    obs, _ = env.reset()
    ep_reward = 0
    ep_rewards = []

    for _ in range(n_steps):
        action, log_prob, value = policy.get_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        obs_list.append(obs)
        act_list.append(action)
        rew_list.append(reward)
        val_list.append(value)
        logp_list.append(log_prob)
        done_list.append(done)

        ep_reward += reward
        obs = next_obs

        if done:
            ep_rewards.append(ep_reward)
            ep_reward = 0
            obs, _ = env.reset()

    env.close()

    # Compute advantages (GAE)
    gamma, lam = 0.99, 0.95
    advantages = np.zeros(n_steps)
    last_gae = 0
    _, _, last_val = policy.get_action(obs)

    for t in reversed(range(n_steps)):
        next_val = last_val if t == n_steps - 1 else val_list[t + 1]
        next_done = 0 if t == n_steps - 1 else done_list[t + 1]
        delta = rew_list[t] + gamma * next_val * (1 - done_list[t]) - val_list[t]
        advantages[t] = last_gae = delta + gamma * lam * (1 - done_list[t]) * last_gae

    returns = advantages + np.array(val_list)

    return {
        "obs": np.array(obs_list),
        "actions": np.array(act_list),
        "log_probs": np.array(logp_list),
        "advantages": advantages,
        "returns": returns,
        "ep_rewards": ep_rewards,
    }


def ppo_update(policy, optimizer, rollout, act_type, clip_eps=0.2, epochs=10, batch_size=64):
    """PPO clipped objective update."""
    obs = torch.FloatTensor(rollout["obs"])
    if act_type == "discrete":
        actions = torch.LongTensor(rollout["actions"])
    else:
        actions = torch.FloatTensor(rollout["actions"])
    old_log_probs = torch.FloatTensor(rollout["log_probs"])
    advantages = torch.FloatTensor(rollout["advantages"])
    returns = torch.FloatTensor(rollout["returns"])

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = len(obs)
    for _ in range(epochs):
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = idx[start:end]

            log_probs, values, entropy = policy.evaluate(
                obs[batch_idx], actions[batch_idx]
            )

            ratio = torch.exp(log_probs - old_log_probs[batch_idx])
            surr1 = ratio * advantages[batch_idx]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages[batch_idx]

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * (returns[batch_idx] - values).pow(2).mean()
            entropy_loss = -0.01 * entropy.mean()

            loss = actor_loss + critic_loss + entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()


def evaluate_policy(policy, env_name, n_episodes=20, max_steps=1000):
    """Evaluate without gradient computation."""
    env = gym.make(env_name)
    scores = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 1000)
        done, total = False, 0.0
        steps = 0
        while not done and steps < max_steps:
            action, _, _ = policy.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
            steps += 1
        scores.append(total)
    env.close()
    return scores


def run(params=None, device="cpu", callback=None):
    """Run PPO baseline on any environment."""
    if not HAS_TORCH:
        return {
            "method": "PPO (Backprop)",
            "error": "PyTorch not installed",
            "solved": False,
        }

    params = params or {}
    env_name = params.get("environment", "LunarLander-v3")
    max_evals = params.get("max_evals", 100000)
    obs_dim = params.get("obs_dim", 8)
    act_dim = params.get("act_dim", 4)
    act_type = params.get("act_type", "discrete")
    hidden = params.get("hidden", [64, 32])
    max_steps = params.get("max_steps", 1000)
    solved_threshold = params.get("solved_threshold", 200)

    obs_type = params.get("obs_type", "vector")
    if obs_type == "pixel":
        return {
            "method": "PPO (Backprop)",
            "environment": env_name,
            "error": "PPO Atari CNN not yet implemented — use ES methods for pixel envs",
            "uses_backprop": True,
            "solved": False,
        }

    policy = PPOPolicy(obs_dim, act_dim, act_type, hidden)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    print(f"📐 PPO (Backprop Baseline) on {env_name}")
    print(f"Network: {obs_dim}→{'→'.join(map(str,hidden))}→{act_dim} ({policy.n_actor_params} actor params)")
    print(f"⚠️  Uses BACKPROPAGATION — this is the control group")

    best_ever = -float("inf")
    total_evals = 0
    start_time = time.time()
    gen = 0
    n_steps = 2048

    while total_evals < max_evals:
        gen += 1
        rollout = collect_rollout(policy, env_name, max_steps, n_steps)
        ppo_update(policy, optimizer, rollout, act_type)

        # Count evaluations (steps in environment)
        total_evals += n_steps

        ep_rewards = rollout["ep_rewards"]
        if ep_rewards:
            gen_best = max(ep_rewards)
            gen_mean = np.mean(ep_rewards)
            if gen_best > best_ever:
                best_ever = gen_best
        else:
            gen_best = gen_mean = 0

        elapsed = time.time() - start_time
        entry = {
            "generation": gen,
            "best_fitness": float(gen_best),
            "best_ever": float(best_ever),
            "mean_fitness": float(gen_mean),
            "total_evals": total_evals,
            "elapsed": round(elapsed, 1),
        }
        if callback:
            callback(entry)

        solved = "✅ SOLVED!" if best_ever >= solved_threshold else ""
        print(f"Gen {gen:4d} | Best: {gen_best:8.1f} | Ever: {best_ever:8.1f} | "
              f"Mean: {gen_mean:8.1f} | Evals: {total_evals:>8,} | {elapsed:6.1f}s {solved}")

        if best_ever >= solved_threshold:
            # Verify with robust eval
            eval_scores = evaluate_policy(policy, env_name, 20, max_steps)
            eval_mean = np.mean(eval_scores)
            if eval_mean >= solved_threshold:
                print(f"\n🎉 SOLVED {env_name} with PPO! Eval mean: {eval_mean:.1f}")
                best_ever = max(best_ever, max(eval_scores))
                break

    final_scores = evaluate_policy(policy, env_name, 20, max_steps)
    final_mean, final_std = float(np.mean(final_scores)), float(np.std(final_scores))

    return {
        "method": "PPO (Backprop)",
        "environment": env_name,
        "uses_backprop": True,
        "best_ever": float(best_ever),
        "final_mean": final_mean,
        "final_std": final_std,
        "total_evals": total_evals,
        "generations": gen,
        "n_params": policy.n_actor_params,
        "elapsed_seconds": round(time.time() - start_time, 1),
        "solved": best_ever >= solved_threshold,
    }
