#!/usr/bin/env python3
"""GAIA Phase 3: Local Learning Methods vs Backprop on LunarLander-v3"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import csv
import os
import time
import copy
from collections import deque

DEVICE = 'cpu'
ENV_NAME = 'LunarLander-v3'
OBS_DIM = 8
ACT_DIM = 4
HIDDEN = 64  # ~10k params with 2 hidden layers
NUM_EPISODES = 600
EVAL_EVERY = 10
SEED = 42

def make_env():
    return gym.make(ENV_NAME)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

# ============================================================
# Method 1: Forward-Forward for RL
# ============================================================
class FFLayer(nn.Module):
    def __init__(self, in_dim, out_dim, lr=0.01):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.opt = torch.optim.Adam(self.linear.parameters(), lr=lr)
        self.threshold = 2.0
    
    def forward(self, x):
        return F.relu(self.linear(x))
    
    def goodness(self, x):
        h = self.forward(x)
        return h.pow(2).mean(dim=-1)  # scalar per sample
    
    def train_step(self, pos_x, neg_x):
        g_pos = self.goodness(pos_x)
        g_neg = self.goodness(neg_x)
        # Loss: want g_pos > threshold, g_neg < threshold
        loss = torch.log(1 + torch.exp(-(g_pos - self.threshold))).mean() + \
               torch.log(1 + torch.exp(g_neg - self.threshold)).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return self.forward(pos_x).detach(), self.forward(neg_x).detach()

class ForwardForwardAgent:
    def __init__(self):
        self.layers = nn.ModuleList([
            FFLayer(OBS_DIM + ACT_DIM, HIDDEN, lr=0.005),
            FFLayer(HIDDEN, HIDDEN, lr=0.005),
        ])
        # Separate feature encoder for policy (obs only)
        self.feat = nn.Sequential(nn.Linear(OBS_DIM, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU())
        self.policy = nn.Linear(HIDDEN, ACT_DIM)
        self.policy_opt = torch.optim.Adam(list(self.feat.parameters()) + list(self.policy.parameters()), lr=0.003)
        self.buffer = deque(maxlen=5000)
        self.param_count = count_params(self.layers) + count_params(self.feat) + count_params(self.policy)
    
    def get_features(self, obs):
        x = torch.FloatTensor(obs)
        return self.feat(x)
    
    def act(self, obs):
        with torch.no_grad():
            feat = self.get_features(obs)
            logits = self.policy(feat)
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1).item()
    
    def store(self, obs, action, reward):
        self.buffer.append((obs, action, reward))
    
    def train(self):
        if len(self.buffer) < 200:
            return
        data = list(self.buffer)
        rewards = [d[2] for d in data]
        median_r = np.median(rewards)
        
        pos_data = [(o, a) for o, a, r in data if r > median_r]
        neg_data = [(o, a) for o, a, r in data if r <= median_r]
        
        if len(pos_data) < 10 or len(neg_data) < 10:
            return
        
        # Create input vectors (obs concat one-hot action)
        def make_input(samples):
            xs = []
            for obs, act in samples:
                one_hot = np.zeros(ACT_DIM)
                one_hot[act] = 1.0
                xs.append(np.concatenate([obs, one_hot]))
            return torch.FloatTensor(np.array(xs))
        
        pos_x = make_input(pos_data)
        neg_x = make_input(neg_data[:len(pos_data)])
        
        # Train FF layers
        px, nx = pos_x, neg_x
        for layer in self.layers:
            px, nx = layer.train_step(px, nx)
        
        # Train policy with REINFORCE using goodness as signal
        batch_idx = np.random.choice(len(data), min(128, len(data)), replace=False)
        batch = [data[i] for i in batch_idx]
        obs_batch = torch.FloatTensor(np.array([b[0] for b in batch]))
        act_batch = torch.LongTensor([b[1] for b in batch])
        rew_batch = torch.FloatTensor([b[2] for b in batch])
        rew_batch = (rew_batch - rew_batch.mean()) / (rew_batch.std() + 1e-8)
        
        feats = self.feat(obs_batch)
        
        logits = self.policy(feats)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs.gather(1, act_batch.unsqueeze(1)).squeeze()
        loss = -(selected * rew_batch).mean()
        self.policy_opt.zero_grad()
        loss.backward()
        self.policy_opt.step()


# ============================================================
# Method 2: Predictive Coding Network
# ============================================================
class PredictiveCodingAgent:
    def __init__(self):
        self.W1 = nn.Linear(OBS_DIM, HIDDEN)
        self.W2 = nn.Linear(HIDDEN, HIDDEN)
        self.policy = nn.Linear(HIDDEN, ACT_DIM)
        
        # Top-down prediction weights
        self.P2 = nn.Linear(HIDDEN, HIDDEN)  # predicts layer 1 from layer 2
        
        self.params = list(self.W1.parameters()) + list(self.W2.parameters()) + \
                      list(self.P2.parameters()) + list(self.policy.parameters())
        self.opt = torch.optim.Adam(self.params, lr=0.003)
        self.buffer = deque(maxlen=5000)
        self.param_count = sum(p.numel() for p in self.params)
    
    def forward(self, obs_t):
        h1 = F.relu(self.W1(obs_t))
        h2 = F.relu(self.W2(h1))
        return h1, h2
    
    def act(self, obs):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs)
            _, h2 = self.forward(obs_t)
            logits = self.policy(h2)
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1).item()
    
    def store(self, obs, action, reward):
        self.buffer.append((obs, action, reward))
    
    def train(self):
        if len(self.buffer) < 200:
            return
        data = list(self.buffer)
        batch_idx = np.random.choice(len(data), min(128, len(data)), replace=False)
        batch = [data[i] for i in batch_idx]
        
        obs_batch = torch.FloatTensor(np.array([b[0] for b in batch]))
        act_batch = torch.LongTensor([b[1] for b in batch])
        rew_batch = torch.FloatTensor([b[2] for b in batch])
        rew_batch = (rew_batch - rew_batch.mean()) / (rew_batch.std() + 1e-8)
        
        h1, h2 = self.forward(obs_batch)
        
        # Predictive coding loss: prediction error at each layer
        h1_pred = F.relu(self.P2(h2))
        pc_loss = F.mse_loss(h1_pred, h1.detach())
        
        # Policy loss (REINFORCE)
        logits = self.policy(h2)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs.gather(1, act_batch.unsqueeze(1)).squeeze()
        policy_loss = -(selected * rew_batch).mean()
        
        loss = policy_loss + 0.1 * pc_loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


# ============================================================
# Method 3: Decoupled Greedy Learning
# ============================================================
class DecoupledGreedyAgent:
    def __init__(self):
        # Each layer has its own local classifier
        self.W1 = nn.Linear(OBS_DIM, HIDDEN)
        self.head1 = nn.Linear(HIDDEN, ACT_DIM)  # local objective for layer 1
        self.W2 = nn.Linear(HIDDEN, HIDDEN)
        self.head2 = nn.Linear(HIDDEN, ACT_DIM)  # local objective for layer 2
        self.policy = nn.Linear(HIDDEN, ACT_DIM)
        
        # Separate optimizers for each layer
        self.opt1 = torch.optim.Adam(list(self.W1.parameters()) + list(self.head1.parameters()), lr=0.003)
        self.opt2 = torch.optim.Adam(list(self.W2.parameters()) + list(self.head2.parameters()), lr=0.003)
        self.opt_policy = torch.optim.Adam(self.policy.parameters(), lr=0.003)
        
        self.buffer = deque(maxlen=5000)
        self.param_count = sum(p.numel() for p in [self.W1, self.head1, self.W2, self.head2, self.policy] 
                              for p in p.parameters())
    
    def act(self, obs):
        with torch.no_grad():
            x = torch.FloatTensor(obs)
            h1 = F.relu(self.W1(x))
            h2 = F.relu(self.W2(h1.detach()))
            logits = self.policy(h2.detach())
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1).item()
    
    def store(self, obs, action, reward):
        self.buffer.append((obs, action, reward))
    
    def train(self):
        if len(self.buffer) < 200:
            return
        data = list(self.buffer)
        batch_idx = np.random.choice(len(data), min(128, len(data)), replace=False)
        batch = [data[i] for i in batch_idx]
        
        obs_batch = torch.FloatTensor(np.array([b[0] for b in batch]))
        act_batch = torch.LongTensor([b[1] for b in batch])
        rew_batch = torch.FloatTensor([b[2] for b in batch])
        rew_batch = (rew_batch - rew_batch.mean()) / (rew_batch.std() + 1e-8)
        
        # Layer 1: local greedy training
        h1 = F.relu(self.W1(obs_batch))
        local_logits1 = self.head1(h1)
        local_lp1 = F.log_softmax(local_logits1, dim=-1).gather(1, act_batch.unsqueeze(1)).squeeze()
        loss1 = -(local_lp1 * rew_batch).mean()
        self.opt1.zero_grad()
        loss1.backward()
        self.opt1.step()
        
        # Layer 2: local greedy training (detached from layer 1)
        h1_det = F.relu(self.W1(obs_batch)).detach()
        h2 = F.relu(self.W2(h1_det))
        local_logits2 = self.head2(h2)
        local_lp2 = F.log_softmax(local_logits2, dim=-1).gather(1, act_batch.unsqueeze(1)).squeeze()
        loss2 = -(local_lp2 * rew_batch).mean()
        self.opt2.zero_grad()
        loss2.backward()
        self.opt2.step()
        
        # Policy head
        h2_det = F.relu(self.W2(h1_det)).detach()
        logits = self.policy(h2_det)
        lp = F.log_softmax(logits, dim=-1).gather(1, act_batch.unsqueeze(1)).squeeze()
        loss_p = -(lp * rew_batch).mean()
        self.opt_policy.zero_grad()
        loss_p.backward()
        self.opt_policy.step()


# ============================================================
# Method 4: Hybrid Evolution + Forward-Forward
# ============================================================
class HybridEvoFFAgent:
    def __init__(self):
        self.pop_size = 20
        self.population = []
        for _ in range(self.pop_size):
            agent = ForwardForwardAgent()
            self.population.append(agent)
        self.best_agent = self.population[0]
        self.generation = 0
        self.param_count = self.population[0].param_count
    
    def act(self, obs):
        return self.best_agent.act(obs)
    
    def evolve(self, fitness_scores):
        # Sort by fitness
        paired = list(zip(fitness_scores, self.population))
        paired.sort(key=lambda x: x[0], reverse=True)
        
        self.best_agent = paired[0][1]
        
        # Keep top 25%
        elite_n = max(2, self.pop_size // 4)
        elites = [p[1] for p in paired[:elite_n]]
        
        new_pop = list(elites)  # keep elites
        
        # Fill rest with mutated elites
        while len(new_pop) < self.pop_size:
            parent = elites[np.random.randint(len(elites))]
            child = copy.deepcopy(parent)
            # Mutate weights
            with torch.no_grad():
                for p in child.layers.parameters():
                    p.add_(torch.randn_like(p) * 0.05)
                for p in child.policy.parameters():
                    p.add_(torch.randn_like(p) * 0.05)
            child.buffer = deque(maxlen=5000)
            # Reset optimizers
            for layer in child.layers:
                layer.opt = torch.optim.Adam(layer.linear.parameters(), lr=0.005)
            child.policy_opt = torch.optim.Adam(child.policy.parameters(), lr=0.003)
            new_pop.append(child)
        
        self.population = new_pop
        self.generation += 1


# ============================================================
# Method 5: Backprop Baseline (Actor-Critic / REINFORCE with baseline)
# ============================================================
class BackpropAgent:
    def __init__(self):
        self.actor = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, ACT_DIM)
        )
        self.critic = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 1)
        )
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=0.001)
        self.param_count = count_params(self.actor) + count_params(self.critic)
        self.trajectories = []
    
    def act(self, obs):
        with torch.no_grad():
            logits = self.actor(torch.FloatTensor(obs))
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1).item()
    
    def store_trajectory(self, trajectory):
        self.trajectories.append(trajectory)
    
    def train(self):
        if len(self.trajectories) < 4:
            return
        
        all_obs, all_acts, all_returns = [], [], []
        gamma = 0.99
        
        for traj in self.trajectories:
            obs_list, act_list, rew_list = zip(*traj)
            # Compute returns
            returns = []
            G = 0
            for r in reversed(rew_list):
                G = r + gamma * G
                returns.insert(0, G)
            all_obs.extend(obs_list)
            all_acts.extend(act_list)
            all_returns.extend(returns)
        
        obs_t = torch.FloatTensor(np.array(all_obs))
        act_t = torch.LongTensor(all_acts)
        ret_t = torch.FloatTensor(all_returns)
        ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)
        
        # Actor loss
        logits = self.actor(obs_t)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs.gather(1, act_t.unsqueeze(1)).squeeze()
        
        values = self.critic(obs_t).squeeze()
        advantage = ret_t - values.detach()
        
        actor_loss = -(selected * advantage).mean()
        critic_loss = F.mse_loss(values, ret_t)
        
        loss = actor_loss + 0.5 * critic_loss
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 0.5)
        self.opt.step()
        
        self.trajectories = []


# ============================================================
# Training loops
# ============================================================

def run_episode(env, agent_act):
    obs, _ = env.reset()
    total_reward = 0
    trajectory = []
    done = False
    steps = 0
    while not done and steps < 1000:
        action = agent_act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        trajectory.append((obs, action, reward))
        total_reward += reward
        obs = next_obs
        done = terminated or truncated
        steps += 1
    return total_reward, trajectory

def evaluate(env, agent_act, n=5):
    rewards = []
    for _ in range(n):
        r, _ = run_episode(env, agent_act)
        rewards.append(r)
    return np.mean(rewards)

def train_simple_agent(agent_cls, name, num_episodes=NUM_EPISODES):
    """For FF, PC, Decoupled agents that use store() and train()"""
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    agent = agent_cls()
    print(f"Parameters: {agent.param_count}")
    
    env = make_env()
    results = []
    reward_history = deque(maxlen=100)
    
    for ep in range(num_episodes):
        total_reward, trajectory = run_episode(env, agent.act)
        reward_history.append(total_reward)
        
        # Store experience
        for obs, action, reward in trajectory:
            agent.store(obs, action, total_reward / len(trajectory))  # per-step reward signal
        
        # Train every episode
        agent.train()
        
        if ep % EVAL_EVERY == 0:
            eval_r = evaluate(env, agent.act)
            avg_r = np.mean(reward_history) if reward_history else 0
            results.append((ep, eval_r, avg_r))
            if ep % 50 == 0:
                print(f"  Episode {ep}: eval={eval_r:.1f}, avg100={avg_r:.1f}")
    
    env.close()
    return results

def train_hybrid(num_episodes=NUM_EPISODES):
    name = "Hybrid Evo+FF"
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    agent = HybridEvoFFAgent()
    print(f"Parameters per individual: {agent.param_count}")
    
    env = make_env()
    results = []
    episodes_per_gen = 10
    episodes_per_individual = episodes_per_gen // agent.pop_size or 1
    
    total_ep = 0
    best_eval = -999
    
    while total_ep < num_episodes:
        # Evaluate each individual
        fitness = []
        for ind in agent.population:
            ind_rewards = []
            for _ in range(2):  # 2 episodes per individual
                r, traj = run_episode(env, ind.act)
                ind_rewards.append(r)
                # FF learning within lifetime
                for obs, action, reward in traj:
                    ind.store(obs, action, r / len(traj))
                ind.train()
                total_ep += 1
            fitness.append(np.mean(ind_rewards))
        
        agent.evolve(fitness)
        
        eval_r = evaluate(env, agent.act)
        best_eval = max(best_eval, eval_r)
        results.append((total_ep, eval_r, max(fitness)))
        
        if agent.generation % 5 == 0:
            print(f"  Gen {agent.generation}, Episodes={total_ep}: eval={eval_r:.1f}, best_fit={max(fitness):.1f}")
    
    env.close()
    return results

def train_backprop(num_episodes=NUM_EPISODES):
    name = "Backprop (Actor-Critic)"
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    agent = BackpropAgent()
    print(f"Parameters: {agent.param_count}")
    
    env = make_env()
    results = []
    reward_history = deque(maxlen=100)
    
    for ep in range(num_episodes):
        total_reward, trajectory = run_episode(env, agent.act)
        reward_history.append(total_reward)
        agent.store_trajectory(trajectory)
        
        # Train every 4 episodes
        if (ep + 1) % 4 == 0:
            agent.train()
        
        if ep % EVAL_EVERY == 0:
            eval_r = evaluate(env, agent.act)
            avg_r = np.mean(reward_history) if reward_history else 0
            results.append((ep, eval_r, avg_r))
            if ep % 50 == 0:
                print(f"  Episode {ep}: eval={eval_r:.1f}, avg100={avg_r:.1f}")
    
    env.close()
    return results


# ============================================================
# Main
# ============================================================
def save_results(results, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'eval_reward', 'avg_reward'])
        for row in results:
            writer.writerow(row)

if __name__ == '__main__':
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    os.makedirs('/root/.openclaw/workspace/gaia-poc/phase3', exist_ok=True)
    base = '/root/.openclaw/workspace/gaia-poc/phase3'
    
    t0 = time.time()
    
    # 1. Forward-Forward
    ff_results = train_simple_agent(ForwardForwardAgent, "Forward-Forward")
    save_results(ff_results, f'{base}/ff_results.csv')
    
    # 2. Predictive Coding
    pc_results = train_simple_agent(PredictiveCodingAgent, "Predictive Coding")
    save_results(pc_results, f'{base}/pc_results.csv')
    
    # 3. Decoupled Greedy
    dg_results = train_simple_agent(DecoupledGreedyAgent, "Decoupled Greedy")
    save_results(dg_results, f'{base}/dg_results.csv')
    
    # 4. Hybrid Evo + FF
    hybrid_results = train_hybrid()
    save_results(hybrid_results, f'{base}/hybrid_results.csv')
    
    # 5. Backprop baseline
    bp_results = train_backprop()
    save_results(bp_results, f'{base}/bp_results.csv')
    
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"All training complete in {elapsed:.0f}s")
    
    # Print final scores
    for name, res in [("Forward-Forward", ff_results), ("Predictive Coding", pc_results),
                       ("Decoupled Greedy", dg_results), ("Hybrid Evo+FF", hybrid_results),
                       ("Backprop (AC)", bp_results)]:
        final = res[-1][1] if res else 0
        best = max(r[1] for r in res) if res else 0
        print(f"  {name:25s}: final={final:.1f}, best={best:.1f}")
