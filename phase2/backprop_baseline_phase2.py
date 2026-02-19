#!/usr/bin/env python3
"""REINFORCE baseline for LunarLander-v3 using numpy only (no PyTorch).

Simple policy gradient with baseline (mean reward).
Same network architecture as evolutionary methods for fair comparison.
"""

import numpy as np
import gymnasium as gym
import csv, time, os

ENV_NAME = "LunarLander-v3"
INPUT_DIM = 8
OUTPUT_DIM = 4
HIDDEN = [64, 64, 32]
LR = 0.001
GAMMA = 0.99
EPISODES = 2000
EVAL_EVERY = 50
EVAL_EPISODES = 10
MAX_STEPS = 500
SEED = 42

np.random.seed(SEED)

class PolicyNetwork:
    def __init__(self):
        dims = [INPUT_DIM] + HIDDEN + [OUTPUT_DIM]
        self.weights = []
        self.biases = []
        for i in range(len(dims)-1):
            # Xavier init
            scale = np.sqrt(2.0 / (dims[i] + dims[i+1]))
            self.weights.append(np.random.randn(dims[i], dims[i+1]) * scale)
            self.biases.append(np.zeros(dims[i+1]))
        self.n_params = sum(w.size + b.size for w, b in zip(self.weights, self.biases))
    
    def forward(self, x):
        """Returns (action_probs, cached_activations_for_backprop)."""
        activations = [x.copy()]
        pre_activations = []
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = x @ W + b
            pre_activations.append(z)
            if i < len(self.weights) - 1:
                x = np.maximum(0, z)  # ReLU
            else:
                # Softmax
                ex = np.exp(z - z.max())
                x = ex / ex.sum()
            activations.append(x)
        return x, activations, pre_activations
    
    def backward(self, activations, pre_activations, action, advantage):
        """Compute REINFORCE gradients."""
        probs = activations[-1]
        # d_softmax: dL/dz = probs - one_hot(action) (cross-entropy gradient, scaled by advantage)
        dz = probs.copy()
        dz[action] -= 1.0
        dz *= -advantage  # negative because we want to maximize
        
        grad_w = []
        grad_b = []
        
        for i in range(len(self.weights) - 1, -1, -1):
            a = activations[i]
            gw = np.outer(a, dz)
            gb = dz.copy()
            grad_w.insert(0, gw)
            grad_b.insert(0, gb)
            
            if i > 0:
                dz = dz @ self.weights[i].T
                # ReLU derivative
                dz *= (pre_activations[i-1] > 0).astype(float)
        
        return grad_w, grad_b
    
    def update(self, grad_w, grad_b, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * grad_w[i]
            self.biases[i] -= lr * grad_b[i]

def compute_returns(rewards, gamma=GAMMA):
    returns = np.zeros_like(rewards)
    G = 0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns

def evaluate_policy(policy, n_episodes=EVAL_EPISODES):
    total = 0
    for ep in range(n_episodes):
        env = gym.make(ENV_NAME)
        obs, _ = env.reset(seed=SEED + 10000 + ep)
        ep_r = 0
        for _ in range(MAX_STEPS):
            probs, _, _ = policy.forward(obs)
            action = np.argmax(probs)
            obs, r, term, trunc, _ = env.step(action)
            ep_r += r
            if term or trunc:
                break
        total += ep_r
        env.close()
    return total / n_episodes

def main():
    os.makedirs("results", exist_ok=True)
    policy = PolicyNetwork()
    print(f"REINFORCE baseline | Params: {policy.n_params}")
    
    csv_path = "results/reinforce.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'train_reward', 'eval_reward'])
    
    reward_history = []
    start = time.time()
    best_eval = -999
    
    for ep in range(EPISODES):
        env = gym.make(ENV_NAME)
        obs, _ = env.reset(seed=SEED + ep)
        
        saved = []  # (activations, pre_activations, action, reward)
        ep_reward = 0
        
        for step in range(MAX_STEPS):
            probs, acts, pre_acts = policy.forward(obs)
            # Sample action
            action = np.random.choice(OUTPUT_DIM, p=np.clip(probs, 1e-8, 1.0))
            next_obs, reward, term, trunc, _ = env.step(action)
            saved.append((acts, pre_acts, action, reward))
            obs = next_obs
            ep_reward += reward
            if term or trunc:
                break
        env.close()
        
        reward_history.append(ep_reward)
        
        # Compute returns and update
        rewards = np.array([s[3] for s in saved])
        returns = compute_returns(rewards)
        baseline = returns.mean()
        advantages = returns - baseline
        # Normalize
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Accumulate gradients
        acc_gw = [np.zeros_like(w) for w in policy.weights]
        acc_gb = [np.zeros_like(b) for b in policy.biases]
        
        for t, (acts, pre_acts, action, _) in enumerate(saved):
            gw, gb = policy.backward(acts, pre_acts, action, advantages[t])
            for i in range(len(acc_gw)):
                acc_gw[i] += gw[i]
                acc_gb[i] += gb[i]
        
        # Average and apply
        n = len(saved)
        for i in range(len(acc_gw)):
            acc_gw[i] /= n
            acc_gb[i] /= n
        policy.update(acc_gw, acc_gb, LR)
        
        # Evaluate periodically
        if (ep + 1) % EVAL_EVERY == 0:
            eval_r = evaluate_policy(policy)
            if eval_r > best_eval:
                best_eval = eval_r
            elapsed = time.time() - start
            mean_train = np.mean(reward_history[-EVAL_EVERY:])
            print(f"  Ep {ep+1:4d} | Train(50): {mean_train:8.1f} | Eval: {eval_r:8.1f} | Best: {best_eval:8.1f} | Time: {elapsed:.0f}s")
            
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([ep+1, mean_train, eval_r])
    
    elapsed = time.time() - start
    solved = "✓ SOLVED" if best_eval >= 200 else "✗ Not solved"
    print(f"\nRESULT: Best eval: {best_eval:.1f} | {solved} | Time: {elapsed:.0f}s | Episodes: {EPISODES}")

if __name__ == "__main__":
    main()
