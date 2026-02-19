#!/usr/bin/env python3
"""GAIA Phase 5: Maximum compute push. Four methods on LunarLander."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import csv
import json
import time
import os
from collections import deque
from copy import deepcopy

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

ENV_NAME = "LunarLander-v3"
SOLVED_THRESHOLD = 200
OUT_DIR = "/root/.openclaw/workspace/gaia-poc/phase5"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Utility ───────────────────────────────────────────────────────────

def make_env():
    return gym.make(ENV_NAME)

def evaluate_agent(agent, n_episodes=10, max_steps=1000):
    """Evaluate agent over n episodes, return mean reward."""
    env = make_env()
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0
        for _ in range(max_steps):
            action = agent.act(obs)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            if term or trunc:
                break
        rewards.append(total)
    env.close()
    return np.mean(rewards), np.std(rewards)

# ─── Network for Evo+FF Methods ───────────────────────────────────────

class FFLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
        return F.relu(self.linear(x))
    
    def goodness(self, x):
        h = self.forward(x)
        return h.pow(2).mean(dim=-1)

class EvoFFAgent:
    """Agent with FF layers + linear policy head, used by evolutionary methods."""
    def __init__(self, obs_dim=8, act_dim=4, hidden_sizes=(128, 64, 32)):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_sizes = hidden_sizes
        
        # FF layers
        dims = [obs_dim] + list(hidden_sizes)
        self.ff_layers = [FFLayer(dims[i], dims[i+1]) for i in range(len(hidden_sizes))]
        
        # Policy head
        self.policy = nn.Linear(hidden_sizes[-1], act_dim)
        nn.init.zeros_(self.policy.weight)
        nn.init.zeros_(self.policy.bias)
        
        # Meta-parameters (evolved)
        self.meta = {
            'ff_lr': [0.005] * len(hidden_sizes),
            'goodness_thresh': [2.0] * len(hidden_sizes),
            'plasticity': [0.1] * len(hidden_sizes),
            'activation_mix': 0.5,  # mix between FF features and raw
        }
    
    def get_features(self, obs_tensor):
        x = obs_tensor
        for layer in self.ff_layers:
            x = layer.forward(x)
        return x
    
    def act(self, obs):
        with torch.no_grad():
            x = torch.FloatTensor(obs).unsqueeze(0)
            features = self.get_features(x)
            logits = self.policy(features)
            return logits.argmax(dim=-1).item()
    
    def get_flat_params(self):
        params = []
        for layer in self.ff_layers:
            for p in layer.parameters():
                params.append(p.data.flatten())
        for p in self.policy.parameters():
            params.append(p.data.flatten())
        return torch.cat(params)
    
    def set_flat_params(self, flat):
        idx = 0
        for layer in self.ff_layers:
            for p in layer.parameters():
                n = p.numel()
                p.data.copy_(flat[idx:idx+n].reshape(p.shape))
                idx += n
        for p in self.policy.parameters():
            n = p.numel()
            p.data.copy_(flat[idx:idx+n].reshape(p.shape))
            idx += n
    
    def param_count(self):
        return sum(p.numel() for layer in self.ff_layers for p in layer.parameters()) + \
               sum(p.numel() for p in self.policy.parameters())
    
    def ff_learn_step(self, good_obs, bad_obs):
        """One FF learning step on good/bad observations."""
        for i, layer in enumerate(self.ff_layers):
            lr = self.meta['ff_lr'][i]
            thresh = self.meta['goodness_thresh'][i]
            
            g_good = layer.goodness(good_obs)
            g_bad = layer.goodness(bad_obs)
            
            # FF objective: goodness(good) > thresh > goodness(bad)
            loss = torch.log(1 + torch.exp(-(g_good - thresh))).mean() + \
                   torch.log(1 + torch.exp(g_bad - thresh)).mean()
            
            # Manual gradient step (no autograd graph needed beyond this)
            layer.zero_grad()
            loss.backward(retain_graph=False)
            with torch.no_grad():
                for p in layer.parameters():
                    if p.grad is not None:
                        p.data -= lr * p.grad
            
            # Update inputs for next layer
            with torch.no_grad():
                good_obs = layer.forward(good_obs)
                bad_obs = layer.forward(bad_obs)
    
    def clone(self):
        new = EvoFFAgent(self.obs_dim, self.act_dim, self.hidden_sizes)
        new.set_flat_params(self.get_flat_params().clone())
        new.meta = deepcopy(self.meta)
        return new


# ─── METHOD A: Hybrid Evo+FF with Meta-Plasticity ─────────────────────

def run_meta_plasticity(pop_size=300, generations=200, elite_size=10, 
                        tournament_size=5, eval_episodes=10, ff_episodes_per_gen=5):
    print(f"\n{'='*60}")
    print("METHOD A: Hybrid Evo+FF with Meta-Plasticity")
    print(f"Population: {pop_size}, Generations: {generations}")
    print(f"{'='*60}")
    
    csv_path = os.path.join(OUT_DIR, "meta_plasticity_results.csv")
    meta_csv_path = os.path.join(OUT_DIR, "meta_plasticity_hyperparams.csv")
    
    # Initialize population
    population = [EvoFFAgent() for _ in range(pop_size)]
    print(f"Params per agent: {population[0].param_count()}")
    
    # Randomize meta-parameters
    for agent in population:
        for i in range(len(agent.meta['ff_lr'])):
            agent.meta['ff_lr'][i] = 10 ** np.random.uniform(-4, -1)
            agent.meta['goodness_thresh'][i] = np.random.uniform(0.5, 5.0)
            agent.meta['plasticity'][i] = 10 ** np.random.uniform(-3, 0)
        agent.meta['activation_mix'] = np.random.uniform(0, 1)
    
    # Species tracking (simple: based on meta-parameter similarity)
    best_ever_fitness = -float('inf')
    best_ever_agent = None
    
    all_rows = []
    meta_rows = []
    t0 = time.time()
    total_evals = 0
    
    for gen in range(generations):
        # FF learning phase: each agent does a few episodes of FF learning
        env = make_env()
        for agent in population:
            good_obs_list = []
            bad_obs_list = []
            for _ in range(ff_episodes_per_gen):
                obs, _ = env.reset()
                episode_obs = []
                episode_rewards = []
                total_r = 0
                for _ in range(300):  # shorter episodes for speed
                    action = agent.act(obs)
                    next_obs, r, term, trunc, _ = env.step(action)
                    episode_obs.append(obs)
                    episode_rewards.append(r)
                    total_r += r
                    obs = next_obs
                    if term or trunc:
                        break
                
                if len(episode_obs) > 1:
                    obs_t = torch.FloatTensor(np.array(episode_obs))
                    rewards = np.array(episode_rewards)
                    median_r = np.median(rewards)
                    good_mask = rewards >= median_r
                    bad_mask = rewards < median_r
                    if good_mask.sum() > 0 and bad_mask.sum() > 0:
                        good_obs_list.append(obs_t[good_mask])
                        bad_obs_list.append(obs_t[bad_mask])
            
            if good_obs_list and bad_obs_list:
                good = torch.cat(good_obs_list)
                bad = torch.cat(bad_obs_list)
                # Subsample for speed
                if len(good) > 200:
                    idx = np.random.choice(len(good), 200, replace=False)
                    good = good[idx]
                if len(bad) > 200:
                    idx = np.random.choice(len(bad), 200, replace=False)
                    bad = bad[idx]
                agent.ff_learn_step(good, bad)
        env.close()
        
        # Evaluate fitness (reduced episodes for most, full for elites later)
        fitnesses = []
        for agent in population:
            f, _ = evaluate_agent(agent, n_episodes=3, max_steps=500)
            fitnesses.append(f)
            total_evals += 3
        
        fitnesses = np.array(fitnesses)
        
        # Re-evaluate top candidates more thoroughly
        top_indices = np.argsort(fitnesses)[-10:]
        for idx in top_indices:
            f, _ = evaluate_agent(population[idx], n_episodes=eval_episodes, max_steps=1000)
            fitnesses[idx] = f
            total_evals += eval_episodes
        
        best_idx = np.argmax(fitnesses)
        gen_best = fitnesses[best_idx]
        gen_mean = np.mean(fitnesses)
        gen_std = np.std(fitnesses)
        
        if gen_best > best_ever_fitness:
            best_ever_fitness = gen_best
            best_ever_agent = population[best_idx].clone()
        
        elapsed = time.time() - t0
        
        # Log meta-params of best agent
        best_agent = population[best_idx]
        meta_row = {
            'gen': gen, 
            'best_fitness': gen_best,
            'ff_lr_0': best_agent.meta['ff_lr'][0],
            'ff_lr_1': best_agent.meta['ff_lr'][1],
            'ff_lr_2': best_agent.meta['ff_lr'][2],
            'thresh_0': best_agent.meta['goodness_thresh'][0],
            'thresh_1': best_agent.meta['goodness_thresh'][1],
            'thresh_2': best_agent.meta['goodness_thresh'][2],
            'plasticity_0': best_agent.meta['plasticity'][0],
            'plasticity_1': best_agent.meta['plasticity'][1],
            'plasticity_2': best_agent.meta['plasticity'][2],
            'activation_mix': best_agent.meta['activation_mix'],
        }
        meta_rows.append(meta_row)
        
        row = {'gen': gen, 'best': gen_best, 'mean': gen_mean, 'std': gen_std, 
               'best_ever': best_ever_fitness, 'total_evals': total_evals, 'time': elapsed}
        all_rows.append(row)
        
        if gen % 10 == 0 or gen == generations - 1:
            print(f"  Gen {gen:3d}: best={gen_best:+.1f} mean={gen_mean:+.1f} best_ever={best_ever_fitness:+.1f} evals={total_evals} t={elapsed:.0f}s")
        
        # Selection + reproduction
        new_pop = []
        
        # Elitism
        elite_indices = np.argsort(fitnesses)[-elite_size:]
        for idx in elite_indices:
            new_pop.append(population[idx].clone())
        
        # Tournament selection + mutation for rest
        while len(new_pop) < pop_size:
            # Tournament
            contestants = np.random.choice(pop_size, tournament_size, replace=False)
            winner_idx = contestants[np.argmax(fitnesses[contestants])]
            child = population[winner_idx].clone()
            
            # Mutate weights
            params = child.get_flat_params()
            mutation_strength = child.meta['plasticity'][0]
            noise = torch.randn_like(params) * mutation_strength
            child.set_flat_params(params + noise)
            
            # Mutate meta-parameters
            for i in range(len(child.meta['ff_lr'])):
                child.meta['ff_lr'][i] *= np.exp(np.random.randn() * 0.3)
                child.meta['ff_lr'][i] = np.clip(child.meta['ff_lr'][i], 1e-5, 0.5)
                child.meta['goodness_thresh'][i] += np.random.randn() * 0.3
                child.meta['goodness_thresh'][i] = np.clip(child.meta['goodness_thresh'][i], 0.1, 10.0)
                child.meta['plasticity'][i] *= np.exp(np.random.randn() * 0.3)
                child.meta['plasticity'][i] = np.clip(child.meta['plasticity'][i], 1e-4, 1.0)
            child.meta['activation_mix'] += np.random.randn() * 0.1
            child.meta['activation_mix'] = np.clip(child.meta['activation_mix'], 0, 1)
            
            new_pop.append(child)
        
        population = new_pop
    
    # Write CSVs
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)
    
    with open(meta_csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=meta_rows[0].keys())
        w.writeheader()
        w.writerows(meta_rows)
    
    # Final thorough evaluation of best
    final_mean, final_std = evaluate_agent(best_ever_agent, n_episodes=30, max_steps=1000)
    print(f"\n  FINAL: best_ever={best_ever_fitness:+.1f}, thorough_eval={final_mean:+.1f}±{final_std:.1f}")
    
    return {
        'method': 'Meta-Plasticity Evo+FF',
        'best_ever': best_ever_fitness,
        'final_mean': final_mean,
        'final_std': final_std,
        'total_evals': total_evals,
        'time': time.time() - t0,
        'best_meta': best_ever_agent.meta if best_ever_agent else None,
    }


# ─── METHOD B: Hybrid Evo+FF with Neuromodulation ─────────────────────

class NeuromodulatedAgent(EvoFFAgent):
    """Adds multiple reward signals that modulate learning differently per layer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_layers = len(self.hidden_sizes)
        # Modulation weights: 3 signals x n_layers
        self.mod_weights = np.random.randn(3, n_layers) * 0.1
        self.prev_reward = 0
        self.reward_history = deque(maxlen=100)
    
    def compute_modulation(self, reward):
        """Compute 3 neuromodulatory signals."""
        # Signal 1: immediate reward (dopamine)
        sig1 = np.tanh(reward / 100.0)
        
        # Signal 2: reward prediction error (TD-like)
        sig2 = np.tanh((reward - self.prev_reward) / 50.0)
        self.prev_reward = reward
        
        # Signal 3: novelty/surprise
        self.reward_history.append(reward)
        if len(self.reward_history) > 5:
            expected = np.mean(self.reward_history)
            sig3 = np.tanh((reward - expected) / 50.0)
        else:
            sig3 = 0.0
        
        return np.array([sig1, sig2, sig3])
    
    def modulated_ff_learn(self, good_obs, bad_obs, signals):
        """FF learning with neuromodulatory gating."""
        for i, layer in enumerate(self.ff_layers):
            # Compute effective lr from base lr * modulation
            mod = np.dot(signals, self.mod_weights[:, i])
            effective_lr = self.meta['ff_lr'][i] * (1.0 + np.tanh(mod))
            effective_lr = max(effective_lr, 1e-6)
            
            thresh = self.meta['goodness_thresh'][i]
            g_good = layer.goodness(good_obs)
            g_bad = layer.goodness(bad_obs)
            
            loss = torch.log(1 + torch.exp(-(g_good - thresh))).mean() + \
                   torch.log(1 + torch.exp(g_bad - thresh)).mean()
            
            layer.zero_grad()
            loss.backward(retain_graph=False)
            with torch.no_grad():
                for p in layer.parameters():
                    if p.grad is not None:
                        p.data -= effective_lr * p.grad
            
            with torch.no_grad():
                good_obs = layer.forward(good_obs)
                bad_obs = layer.forward(bad_obs)
    
    def clone(self):
        new = NeuromodulatedAgent(self.obs_dim, self.act_dim, self.hidden_sizes)
        new.set_flat_params(self.get_flat_params().clone())
        new.meta = deepcopy(self.meta)
        new.mod_weights = self.mod_weights.copy()
        return new


def run_neuromodulation(pop_size=200, generations=150, elite_size=8,
                        tournament_size=5, eval_episodes=10, ff_episodes_per_gen=5):
    print(f"\n{'='*60}")
    print("METHOD B: Hybrid Evo+FF with Neuromodulation")
    print(f"Population: {pop_size}, Generations: {generations}")
    print(f"{'='*60}")
    
    csv_path = os.path.join(OUT_DIR, "neuromod_results.csv")
    
    population = [NeuromodulatedAgent() for _ in range(pop_size)]
    
    for agent in population:
        for i in range(len(agent.meta['ff_lr'])):
            agent.meta['ff_lr'][i] = 10 ** np.random.uniform(-4, -1)
            agent.meta['goodness_thresh'][i] = np.random.uniform(0.5, 5.0)
            agent.meta['plasticity'][i] = 10 ** np.random.uniform(-3, 0)
    
    best_ever_fitness = -float('inf')
    best_ever_agent = None
    all_rows = []
    t0 = time.time()
    total_evals = 0
    
    for gen in range(generations):
        # FF learning with neuromodulation
        env = make_env()
        for agent in population:
            good_obs_list, bad_obs_list = [], []
            episode_reward = 0
            for _ in range(ff_episodes_per_gen):
                obs, _ = env.reset()
                ep_obs, ep_rewards = [], []
                for _ in range(300):
                    action = agent.act(obs)
                    next_obs, r, term, trunc, _ = env.step(action)
                    ep_obs.append(obs)
                    ep_rewards.append(r)
                    episode_reward += r
                    obs = next_obs
                    if term or trunc:
                        break
                
                if len(ep_obs) > 1:
                    obs_t = torch.FloatTensor(np.array(ep_obs))
                    rewards = np.array(ep_rewards)
                    median_r = np.median(rewards)
                    good_mask = rewards >= median_r
                    bad_mask = rewards < median_r
                    if good_mask.sum() > 0 and bad_mask.sum() > 0:
                        good_obs_list.append(obs_t[good_mask])
                        bad_obs_list.append(obs_t[bad_mask])
            
            if good_obs_list and bad_obs_list:
                good = torch.cat(good_obs_list)[:200]
                bad = torch.cat(bad_obs_list)[:200]
                signals = agent.compute_modulation(episode_reward / max(ff_episodes_per_gen, 1))
                agent.modulated_ff_learn(good, bad, signals)
        env.close()
        
        # Evaluate
        fitnesses = []
        for agent in population:
            f, _ = evaluate_agent(agent, n_episodes=3, max_steps=500)
            fitnesses.append(f)
            total_evals += 3
        fitnesses = np.array(fitnesses)
        
        # Re-eval top
        top_indices = np.argsort(fitnesses)[-15:]
        for idx in top_indices:
            f, _ = evaluate_agent(population[idx], n_episodes=eval_episodes, max_steps=1000)
            fitnesses[idx] = f
            total_evals += eval_episodes
        
        best_idx = np.argmax(fitnesses)
        gen_best = fitnesses[best_idx]
        
        if gen_best > best_ever_fitness:
            best_ever_fitness = gen_best
            best_ever_agent = population[best_idx].clone()
        
        elapsed = time.time() - t0
        row = {'gen': gen, 'best': gen_best, 'mean': np.mean(fitnesses), 
               'std': np.std(fitnesses), 'best_ever': best_ever_fitness,
               'total_evals': total_evals, 'time': elapsed}
        all_rows.append(row)
        
        if gen % 10 == 0 or gen == generations - 1:
            print(f"  Gen {gen:3d}: best={gen_best:+.1f} mean={np.mean(fitnesses):+.1f} best_ever={best_ever_fitness:+.1f} t={elapsed:.0f}s")
        
        # Selection
        new_pop = []
        elite_indices = np.argsort(fitnesses)[-elite_size:]
        for idx in elite_indices:
            new_pop.append(population[idx].clone())
        
        while len(new_pop) < pop_size:
            contestants = np.random.choice(pop_size, tournament_size, replace=False)
            winner_idx = contestants[np.argmax(fitnesses[contestants])]
            child = population[winner_idx].clone()
            
            params = child.get_flat_params()
            noise = torch.randn_like(params) * child.meta['plasticity'][0]
            child.set_flat_params(params + noise)
            
            # Mutate meta + modulation weights
            for i in range(len(child.meta['ff_lr'])):
                child.meta['ff_lr'][i] *= np.exp(np.random.randn() * 0.3)
                child.meta['ff_lr'][i] = np.clip(child.meta['ff_lr'][i], 1e-5, 0.5)
                child.meta['goodness_thresh'][i] += np.random.randn() * 0.3
                child.meta['goodness_thresh'][i] = np.clip(child.meta['goodness_thresh'][i], 0.1, 10.0)
                child.meta['plasticity'][i] *= np.exp(np.random.randn() * 0.3)
                child.meta['plasticity'][i] = np.clip(child.meta['plasticity'][i], 1e-4, 1.0)
            child.mod_weights += np.random.randn(*child.mod_weights.shape) * 0.1
            
            new_pop.append(child)
        population = new_pop
    
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)
    
    final_mean, final_std = evaluate_agent(best_ever_agent, n_episodes=30, max_steps=1000)
    print(f"\n  FINAL: best_ever={best_ever_fitness:+.1f}, thorough_eval={final_mean:+.1f}±{final_std:.1f}")
    
    return {
        'method': 'Neuromodulated Evo+FF',
        'best_ever': best_ever_fitness,
        'final_mean': final_mean,
        'final_std': final_std,
        'total_evals': total_evals,
        'time': time.time() - t0,
    }


# ─── METHOD C: PPO Baseline ───────────────────────────────────────────

class PPONet(nn.Module):
    def __init__(self, obs_dim=8, act_dim=4, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)
    
    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)
    
    def act(self, obs):
        with torch.no_grad():
            x = torch.FloatTensor(obs).unsqueeze(0)
            logits, _ = self.forward(x)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            return action.item(), dist.log_prob(action).item(), logits

def run_ppo(total_timesteps=500_000, lr=3e-4, gamma=0.99, lam=0.95,
            clip_eps=0.2, epochs=10, batch_size=64, ent_coef=0.01,
            rollout_len=2048):
    print(f"\n{'='*60}")
    print("METHOD C: PPO Baseline")
    print(f"Timesteps: {total_timesteps}")
    print(f"{'='*60}")
    
    csv_path = os.path.join(OUT_DIR, "ppo_results.csv")
    
    net = PPONet()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    print(f"Params: {sum(p.numel() for p in net.parameters())}")
    
    env = make_env()
    obs, _ = env.reset()
    
    all_rows = []
    episode_rewards = []
    current_ep_reward = 0
    t0 = time.time()
    total_evals = 0
    step = 0
    eval_interval = 10000
    next_eval = eval_interval
    
    best_eval = -float('inf')
    
    while step < total_timesteps:
        # Collect rollout
        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []
        
        for _ in range(rollout_len):
            action, logp, _ = net.act(obs)
            with torch.no_grad():
                _, val = net.forward(torch.FloatTensor(obs).unsqueeze(0))
            
            next_obs, r, term, trunc, _ = env.step(action)
            
            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(logp)
            rew_buf.append(r)
            done_buf.append(term or trunc)
            val_buf.append(val.item())
            
            current_ep_reward += r
            obs = next_obs
            step += 1
            
            if term or trunc:
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0
                obs, _ = env.reset()
        
        # GAE
        with torch.no_grad():
            _, last_val = net.forward(torch.FloatTensor(obs).unsqueeze(0))
            last_val = last_val.item()
        
        advantages = np.zeros(len(rew_buf))
        gae = 0
        for t in reversed(range(len(rew_buf))):
            next_val = last_val if t == len(rew_buf) - 1 else val_buf[t + 1]
            next_nonterminal = 0.0 if done_buf[t] else 1.0
            delta = rew_buf[t] + gamma * next_val * next_nonterminal - val_buf[t]
            gae = delta + gamma * lam * next_nonterminal * gae
            advantages[t] = gae
        
        returns = advantages + np.array(val_buf)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        obs_t = torch.FloatTensor(np.array(obs_buf))
        act_t = torch.LongTensor(act_buf)
        logp_old = torch.FloatTensor(logp_buf)
        adv_t = torch.FloatTensor(advantages)
        ret_t = torch.FloatTensor(returns)
        
        # PPO update
        for _ in range(epochs):
            indices = np.random.permutation(len(obs_buf))
            for start in range(0, len(obs_buf), batch_size):
                end = start + batch_size
                idx = indices[start:end]
                
                logits, vals = net.forward(obs_t[idx])
                dist = torch.distributions.Categorical(logits=logits)
                logp_new = dist.log_prob(act_t[idx])
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(logp_new - logp_old[idx])
                surr1 = ratio * adv_t[idx]
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t[idx]
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * (ret_t[idx] - vals.squeeze()).pow(2).mean()
                loss = actor_loss + critic_loss - ent_coef * entropy
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()
        
        # Periodic evaluation
        if step >= next_eval:
            # Evaluate
            eval_rewards = []
            for _ in range(20):
                eval_obs, _ = make_env().reset()
                eval_env = make_env()
                eval_obs, _ = eval_env.reset()
                er = 0
                for _ in range(1000):
                    with torch.no_grad():
                        logits, _ = net.forward(torch.FloatTensor(eval_obs).unsqueeze(0))
                        a = logits.argmax(dim=-1).item()
                    eval_obs, r, term, trunc, _ = eval_env.step(a)
                    er += r
                    if term or trunc:
                        break
                eval_rewards.append(er)
                eval_env.close()
            
            eval_mean = np.mean(eval_rewards)
            eval_std = np.std(eval_rewards)
            total_evals += 20
            
            if eval_mean > best_eval:
                best_eval = eval_mean
            
            elapsed = time.time() - t0
            recent = np.mean(episode_rewards[-50:]) if episode_rewards else 0
            row = {'step': step, 'eval_mean': eval_mean, 'eval_std': eval_std,
                   'train_mean': recent, 'best_eval': best_eval, 'time': elapsed}
            all_rows.append(row)
            
            print(f"  Step {step:6d}: eval={eval_mean:+.1f}±{eval_std:.1f} train={recent:+.1f} best={best_eval:+.1f} t={elapsed:.0f}s")
            
            next_eval += eval_interval
    
    env.close()
    
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)
    
    # Final eval
    final_rewards = []
    for _ in range(30):
        eval_env = make_env()
        eval_obs, _ = eval_env.reset()
        er = 0
        for _ in range(1000):
            with torch.no_grad():
                logits, _ = net.forward(torch.FloatTensor(eval_obs).unsqueeze(0))
                a = logits.argmax(dim=-1).item()
            eval_obs, r, term, trunc, _ = eval_env.step(a)
            er += r
            if term or trunc:
                break
        final_rewards.append(er)
        eval_env.close()
    
    final_mean = np.mean(final_rewards)
    final_std = np.std(final_rewards)
    print(f"\n  FINAL: best_eval={best_eval:+.1f}, final={final_mean:+.1f}±{final_std:.1f}")
    
    return {
        'method': 'PPO',
        'best_ever': best_eval,
        'final_mean': final_mean,
        'final_std': final_std,
        'total_evals': total_evals,
        'time': time.time() - t0,
    }


# ─── METHOD D: Forward-Forward Only ───────────────────────────────────

class FFOnlyAgent:
    """Pure FF learning, no evolution."""
    def __init__(self, obs_dim=8, act_dim=4, hidden_sizes=(128, 64, 32)):
        dims = [obs_dim] + list(hidden_sizes)
        self.ff_layers = [FFLayer(dims[i], dims[i+1]) for i in range(len(hidden_sizes))]
        self.policy = nn.Linear(hidden_sizes[-1], act_dim)
        nn.init.zeros_(self.policy.weight)
        nn.init.zeros_(self.policy.bias)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.001)
        self.ff_lr = 0.005
        self.goodness_thresh = 2.0
        self.obs_dim = obs_dim
        self.act_dim = act_dim
    
    def act(self, obs):
        with torch.no_grad():
            x = torch.FloatTensor(obs).unsqueeze(0)
            for layer in self.ff_layers:
                x = layer.forward(x)
            logits = self.policy(x)
            return logits.argmax(dim=-1).item()
    
    def act_stochastic(self, obs):
        x = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            for layer in self.ff_layers:
                x = layer.forward(x)
        # Policy head with gradient
        logits = self.policy(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def ff_learn(self, good_obs, bad_obs):
        for layer in self.ff_layers:
            g_good = layer.goodness(good_obs)
            g_bad = layer.goodness(bad_obs)
            loss = torch.log(1 + torch.exp(-(g_good - self.goodness_thresh))).mean() + \
                   torch.log(1 + torch.exp(g_bad - self.goodness_thresh)).mean()
            layer.zero_grad()
            loss.backward(retain_graph=False)
            with torch.no_grad():
                for p in layer.parameters():
                    if p.grad is not None:
                        p.data -= self.ff_lr * p.grad
            with torch.no_grad():
                good_obs = layer.forward(good_obs)
                bad_obs = layer.forward(bad_obs)
    
    def policy_learn(self, obs, actions, rewards):
        """REINFORCE update on policy head only."""
        x = torch.FloatTensor(np.array(obs))
        with torch.no_grad():
            for layer in self.ff_layers:
                x = layer.forward(x)
        
        logits = self.policy(x)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(torch.LongTensor(actions))
        
        # Normalize rewards
        rewards_t = torch.FloatTensor(rewards)
        rewards_t = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
        
        loss = -(logp * rewards_t).mean()
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()


def run_ff_only(n_episodes=5000, eval_interval=100):
    print(f"\n{'='*60}")
    print("METHOD D: Forward-Forward Only (no evolution)")
    print(f"Episodes: {n_episodes}")
    print(f"{'='*60}")
    
    csv_path = os.path.join(OUT_DIR, "ff_only_results.csv")
    
    agent = FFOnlyAgent()
    env = make_env()
    
    all_rows = []
    recent_rewards = deque(maxlen=100)
    best_eval = -float('inf')
    t0 = time.time()
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_obs, ep_actions, ep_rewards = [], [], []
        total_r = 0
        
        for _ in range(1000):
            action = agent.act(obs)  # greedy with some exploration
            # Epsilon-greedy
            if np.random.random() < max(0.01, 0.3 * (1 - ep / n_episodes)):
                action = np.random.randint(agent.act_dim)
            
            next_obs, r, term, trunc, _ = env.step(action)
            ep_obs.append(obs)
            ep_actions.append(action)
            ep_rewards.append(r)
            total_r += r
            obs = next_obs
            if term or trunc:
                break
        
        recent_rewards.append(total_r)
        
        # FF learning
        if len(ep_obs) > 1:
            obs_t = torch.FloatTensor(np.array(ep_obs))
            rewards = np.array(ep_rewards)
            median_r = np.median(rewards)
            good_mask = rewards >= median_r
            bad_mask = rewards < median_r
            if good_mask.sum() > 0 and bad_mask.sum() > 0:
                agent.ff_learn(obs_t[good_mask], obs_t[bad_mask])
        
        # Policy learning
        if len(ep_obs) > 1:
            # Compute returns
            returns = []
            G = 0
            for r in reversed(ep_rewards):
                G = r + 0.99 * G
                returns.insert(0, G)
            agent.policy_learn(ep_obs, ep_actions, returns)
        
        # Evaluate
        if (ep + 1) % eval_interval == 0:
            eval_mean, eval_std = evaluate_agent(agent, n_episodes=10)
            if eval_mean > best_eval:
                best_eval = eval_mean
            elapsed = time.time() - t0
            row = {'episode': ep + 1, 'eval_mean': eval_mean, 'eval_std': eval_std,
                   'train_mean': np.mean(recent_rewards), 'best_eval': best_eval, 'time': elapsed}
            all_rows.append(row)
            
            if (ep + 1) % 500 == 0:
                print(f"  Ep {ep+1:5d}: eval={eval_mean:+.1f}±{eval_std:.1f} train={np.mean(recent_rewards):+.1f} best={best_eval:+.1f} t={elapsed:.0f}s")
    
    env.close()
    
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)
    
    final_mean, final_std = evaluate_agent(agent, n_episodes=30)
    print(f"\n  FINAL: best_eval={best_eval:+.1f}, final={final_mean:+.1f}±{final_std:.1f}")
    
    return {
        'method': 'FF Only',
        'best_ever': best_eval,
        'final_mean': final_mean,
        'final_std': final_std,
        'total_evals': n_episodes,
        'time': time.time() - t0,
    }


# ─── MAIN ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {}
    
    # Run all methods
    results['meta_plasticity'] = run_meta_plasticity(
        pop_size=100, generations=100, elite_size=8,
        tournament_size=5, eval_episodes=5, ff_episodes_per_gen=2
    )
    
    results['neuromod'] = run_neuromodulation(
        pop_size=80, generations=80, elite_size=6,
        tournament_size=5, eval_episodes=5, ff_episodes_per_gen=2
    )
    
    results['ppo'] = run_ppo(total_timesteps=300_000)
    
    results['ff_only'] = run_ff_only(n_episodes=3000)
    
    # Save summary
    with open(os.path.join(OUT_DIR, "results_summary.json"), 'w') as f:
        # Convert non-serializable items
        clean = {}
        for k, v in results.items():
            clean[k] = {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv) 
                        for kk, vv in v.items() if kk != 'best_meta'}
            if 'best_meta' in v and v['best_meta'] is not None:
                clean[k]['best_meta'] = {mk: [float(x) for x in mv] if isinstance(mv, list) else float(mv)
                                          for mk, mv in v['best_meta'].items()}
        json.dump(clean, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {v['method']:30s}: best={v['best_ever']:+.1f} final={v['final_mean']:+.1f}±{v['final_std']:.1f} time={v['time']:.0f}s")
    print(f"\nAll results saved to {OUT_DIR}/")
