"""Backprop baseline: REINFORCE on CartPole-v1 with same-sized network."""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import time

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32), nn.Tanh(),
            nn.Linear(32, 16), nn.Tanh(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

def train_reinforce(n_episodes=2000):
    env = gym.make('CartPole-v1')
    policy = Policy()
    print(f"Params: {sum(p.numel() for p in policy.parameters())}")
    optimizer = optim.Adam(policy.parameters(), lr=0.01)

    history = {'reward': [], 'time': []}
    running_reward = 0

    for ep in range(n_episodes):
        t0 = time.time()
        obs, _ = env.reset()
        log_probs, rewards = [], []
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs)
            probs = policy(obs_t)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            obs, r, term, trunc, _ = env.step(action.item())
            rewards.append(r)
            done = term or trunc

        ep_reward = sum(rewards)
        running_reward = 0.95 * running_reward + 0.05 * ep_reward

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = sum(-lp * R for lp, R in zip(log_probs, returns))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dt = time.time() - t0
        history['reward'].append(ep_reward)
        history['time'].append(dt)

        if ep % 100 == 0:
            print(f"Ep {ep:4d} | Reward: {ep_reward:6.1f} | Running: {running_reward:6.1f}")

        if running_reward >= 475:
            print(f"\nSolved at episode {ep}! Running reward: {running_reward:.1f}")
            break

    env.close()

    # Final eval
    env = gym.make('CartPole-v1')
    eval_rewards = []
    for _ in range(20):
        obs, _ = env.reset()
        total = 0; done = False
        while not done:
            with torch.no_grad():
                probs = policy(torch.FloatTensor(obs))
            action = probs.argmax().item()
            obs, r, term, trunc, _ = env.step(action)
            total += r; done = term or trunc
        eval_rewards.append(total)
    env.close()

    print(f"\nFinal eval (20 eps): mean={np.mean(eval_rewards):.1f}, std={np.std(eval_rewards):.1f}")
    return history, eval_rewards

if __name__ == '__main__':
    history, eval_rewards = train_reinforce()
    import json
    with open('/root/.openclaw/workspace/gaia-poc/backprop_results.json', 'w') as f:
        json.dump({
            'episodes': len(history['reward']),
            'total_time': sum(history['time']),
            'final_eval_mean': float(np.mean(eval_rewards)),
            'final_eval_std': float(np.std(eval_rewards)),
            'solved_at': next((i for i, r in enumerate(history['reward'])
                             if np.mean(history['reward'][max(0,i-100):i+1]) > 475), None)
        }, f)
    print("Results saved to backprop_results.json")
