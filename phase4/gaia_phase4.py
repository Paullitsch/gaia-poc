"""
GAIA Phase 4: Hybrid Evolution + Forward-Forward with Meta-Learned Plasticity
vs Backprop Actor-Critic on LunarLander-v3

Three methods:
  a) Hybrid Evo+FF (fixed hyperparams)
  b) Hybrid Evo+FF with meta-learned plasticity
  c) Backprop actor-critic baseline (well-tuned)

2000 episodes/evaluations, ~20K param networks, proper tournament selection + niching.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import csv
import time
import os
from copy import deepcopy

SEED = 42
ENV_NAME = "LunarLander-v3"
TOTAL_EPISODES = 2000
EVAL_INTERVAL = 50  # evaluate every N episodes
NUM_EVAL_EPISODES = 5

np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Forward-Forward Layer ─────────────────────────────────────────
class FFLayer(nn.Module):
    def __init__(self, in_dim, out_dim, lr=0.01, threshold=2.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        self.opt = torch.optim.Adam(self.linear.parameters(), lr=lr)
        self.threshold = threshold

    def forward(self, x):
        x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        return F.relu(self.linear(x_norm))

    def goodness(self, x):
        h = self.forward(x)
        return (h ** 2).mean(dim=-1)

    def learn(self, pos_x, neg_x):
        g_pos = self.goodness(pos_x)
        g_neg = self.goodness(neg_x)
        # Push positive goodness above threshold, negative below
        loss = torch.log(1 + torch.exp(-(g_pos - self.threshold))).mean() + \
               torch.log(1 + torch.exp(g_neg - self.threshold)).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()


# ─── Hybrid Evo+FF Agent ──────────────────────────────────────────
class HybridFFAgent:
    def __init__(self, obs_dim=8, act_dim=4, hidden_sizes=(128, 64, 32),
                 ff_lr=0.01, ff_threshold=2.0, policy_lr=0.005):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # FF representation layers
        dims = [obs_dim] + list(hidden_sizes)
        self.ff_layers = []
        for i in range(len(dims)-1):
            self.ff_layers.append(FFLayer(dims[i], dims[i+1], lr=ff_lr, threshold=ff_threshold))
        # Policy head (small, uses REINFORCE)
        self.policy = nn.Linear(hidden_sizes[-1], act_dim)
        nn.init.xavier_uniform_(self.policy.weight)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)

        self.saved_log_probs = []
        self.rewards = []

    def get_representation(self, obs):
        x = torch.FloatTensor(obs).unsqueeze(0)
        for layer in self.ff_layers:
            x = layer.forward(x)
        return x

    def select_action(self, obs):
        rep = self.get_representation(obs)
        logits = self.policy(rep)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.saved_log_probs.append(dist.log_prob(action))
        return action.item()

    def update_policy(self, gamma=0.99):
        if len(self.rewards) == 0:
            return
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        loss = 0
        for lp, R in zip(self.saved_log_probs, returns):
            loss -= lp * R
        self.policy_opt.zero_grad()
        loss.backward()
        self.policy_opt.step()
        self.saved_log_probs = []
        self.rewards = []

    def ff_learn_from_episode(self, good_obs, bad_obs):
        """Train FF layers on good vs bad observations from episode."""
        if len(good_obs) == 0 or len(bad_obs) == 0:
            return
        pos = torch.FloatTensor(np.array(good_obs))
        neg = torch.FloatTensor(np.array(bad_obs))
        # Ensure same batch size
        min_n = min(len(pos), len(neg))
        pos, neg = pos[:min_n], neg[:min_n]
        x_pos, x_neg = pos, neg
        for layer in self.ff_layers:
            layer.learn(x_pos, x_neg)
            x_pos = layer.forward(x_pos).detach()
            x_neg = layer.forward(x_neg).detach()

    def param_count(self):
        total = sum(p.numel() for l in self.ff_layers for p in l.linear.parameters())
        total += sum(p.numel() for p in self.policy.parameters())
        return total


# ─── Evolutionary Infrastructure ──────────────────────────────────
class Species:
    def __init__(self):
        self.members = []
        self.best_fitness = -float('inf')
        self.stagnation = 0

def tournament_select(pop_with_fitness, k=3):
    """Tournament selection."""
    candidates = [pop_with_fitness[i] for i in np.random.choice(len(pop_with_fitness), k, replace=False)]
    return max(candidates, key=lambda x: x[1])[0]

def mutate_genome(genome, mutation_rate=0.1, mutation_strength=0.05):
    """Mutate a genome (dict of numpy arrays)."""
    new_genome = {}
    for key, val in genome.items():
        mask = np.random.random(val.shape) < mutation_rate
        noise = np.random.randn(*val.shape) * mutation_strength
        new_genome[key] = val + mask * noise
    return new_genome

def crossover_genomes(g1, g2):
    """Uniform crossover."""
    child = {}
    for key in g1:
        mask = np.random.random(g1[key].shape) > 0.5
        child[key] = np.where(mask, g1[key], g2[key])
    return child

def agent_to_genome(agent):
    genome = {}
    for i, layer in enumerate(agent.ff_layers):
        genome[f'ff_{i}_w'] = layer.linear.weight.data.numpy().copy()
        genome[f'ff_{i}_b'] = layer.linear.bias.data.numpy().copy()
    genome['policy_w'] = agent.policy.weight.data.numpy().copy()
    genome['policy_b'] = agent.policy.bias.data.numpy().copy()
    return genome

def genome_to_agent(genome, agent):
    for i, layer in enumerate(agent.ff_layers):
        layer.linear.weight.data = torch.FloatTensor(genome[f'ff_{i}_w'])
        layer.linear.bias.data = torch.FloatTensor(genome[f'ff_{i}_b'])
    agent.policy.weight.data = torch.FloatTensor(genome['policy_w'])
    agent.policy.bias.data = torch.FloatTensor(genome['policy_b'])

def genome_distance(g1, g2):
    dist = 0
    n = 0
    for key in g1:
        dist += np.sum((g1[key] - g2[key])**2)
        n += g1[key].size
    return dist / n


# ─── Meta-learned plasticity genome ──────────────────────────────
class MetaGenome:
    """Stores both network weights AND learning hyperparameters."""
    def __init__(self, num_ff_layers=3):
        self.ff_lrs = np.random.uniform(0.001, 0.05, num_ff_layers)
        self.ff_thresholds = np.random.uniform(0.5, 5.0, num_ff_layers)
        self.policy_lr = np.random.uniform(0.001, 0.02)
        self.goodness_type = np.random.choice([0, 1, 2])  # 0=L2, 1=L1, 2=max

    def mutate(self):
        new = MetaGenome.__new__(MetaGenome)
        new.ff_lrs = self.ff_lrs * np.exp(np.random.randn(len(self.ff_lrs)) * 0.2)
        new.ff_lrs = np.clip(new.ff_lrs, 0.0001, 0.1)
        new.ff_thresholds = self.ff_thresholds + np.random.randn(len(self.ff_thresholds)) * 0.3
        new.ff_thresholds = np.clip(new.ff_thresholds, 0.1, 10.0)
        new.policy_lr = self.policy_lr * np.exp(np.random.randn() * 0.2)
        new.policy_lr = np.clip(new.policy_lr, 0.0001, 0.05)
        new.goodness_type = self.goodness_type if np.random.random() > 0.1 else np.random.choice([0, 1, 2])
        return new

    def to_dict(self):
        return {
            'ff_lrs': self.ff_lrs.tolist(),
            'ff_thresholds': self.ff_thresholds.tolist(),
            'policy_lr': float(self.policy_lr),
            'goodness_type': int(self.goodness_type)
        }


def apply_meta_to_agent(agent, meta):
    """Apply meta-learned hyperparams to an agent."""
    for i, layer in enumerate(agent.ff_layers):
        for pg in layer.opt.param_groups:
            pg['lr'] = float(meta.ff_lrs[i])
        layer.threshold = float(meta.ff_thresholds[i])
    for pg in agent.policy_opt.param_groups:
        pg['lr'] = float(meta.policy_lr)


# ─── Run one episode ──────────────────────────────────────────────
def run_episode(env, agent, train_ff=True, train_policy=True, max_steps=1000):
    obs, _ = env.reset()
    total_reward = 0
    all_obs = []
    all_rewards = []

    for t in range(max_steps):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.rewards.append(reward)
        all_obs.append(obs.copy())
        all_rewards.append(reward)
        total_reward += reward
        obs = next_obs
        if terminated or truncated:
            break

    if train_policy:
        agent.update_policy()

    # FF learning: split observations by reward quality
    if train_ff and len(all_obs) > 10:
        rewards_arr = np.array(all_rewards)
        median_r = np.median(rewards_arr)
        good_idx = rewards_arr >= median_r
        bad_idx = rewards_arr < median_r
        good_obs = [all_obs[i] for i in range(len(all_obs)) if good_idx[i]]
        bad_obs = [all_obs[i] for i in range(len(all_obs)) if bad_idx[i]]
        agent.ff_learn_from_episode(good_obs, bad_obs)

    return total_reward


def evaluate_agent(env, agent, n_episodes=NUM_EVAL_EPISODES):
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0
        for t in range(1000):
            with torch.no_grad():
                rep = agent.get_representation(obs)
                logits = agent.policy(rep)
                action = logits.argmax(dim=-1).item()
            obs, r, done, trunc, _ = env.step(action)
            total += r
            if done or trunc:
                break
        rewards.append(total)
    return np.mean(rewards)


# ─── Backprop Actor-Critic Baseline ───────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, obs_dim=8, act_dim=4):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.actor = nn.Linear(32, act_dim)
        self.critic = nn.Linear(32, 1)

    def forward(self, x):
        h = self.shared(x)
        return F.softmax(self.actor(h), dim=-1), self.critic(h)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def run_backprop_baseline(total_episodes=TOTAL_EPISODES, eval_interval=EVAL_INTERVAL):
    env = gym.make(ENV_NAME)
    eval_env = gym.make(ENV_NAME)
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    gamma = 0.99

    print(f"Backprop AC params: {model.param_count()}")
    results = []

    for ep in range(total_episodes):
        obs, _ = env.reset()
        log_probs, values, rewards = [], [], []

        for t in range(1000):
            x = torch.FloatTensor(obs).unsqueeze(0)
            probs, val = model(x)
            if torch.isnan(probs).any():
                probs = torch.ones(1, 4) / 4.0
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            values.append(val.squeeze())
            obs, r, done, trunc, _ = env.step(action.item())
            rewards.append(r)
            if done or trunc:
                break

        # Compute returns
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        advantage = returns - values.detach()
        if advantage.std() > 1e-8:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        actor_loss = -(log_probs * advantage).mean()
        critic_loss = F.mse_loss(values, returns)
        loss = actor_loss + 0.5 * critic_loss

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
        optimizer.step()

        if (ep + 1) % eval_interval == 0:
            model.eval()
            eval_rewards = []
            for _ in range(NUM_EVAL_EPISODES):
                obs_e, _ = eval_env.reset()
                total = 0
                for _ in range(1000):
                    with torch.no_grad():
                        probs_e, _ = model(torch.FloatTensor(obs_e).unsqueeze(0))
                        action_e = probs_e.argmax(dim=-1).item()
                    obs_e, r_e, d_e, t_e, _ = eval_env.step(action_e)
                    total += r_e
                    if d_e or t_e:
                        break
                eval_rewards.append(total)
            mean_eval = np.mean(eval_rewards)
            results.append({'episode': ep+1, 'eval_reward': mean_eval})
            print(f"  Backprop ep {ep+1}: eval={mean_eval:.1f}")
            model.train()

    env.close()
    eval_env.close()
    return results


# ─── Hybrid Evo+FF Methods ────────────────────────────────────────
def run_hybrid_evo_ff(total_episodes=TOTAL_EPISODES, eval_interval=EVAL_INTERVAL,
                      meta_learning=False, label="hybrid_ff"):
    env = gym.make(ENV_NAME)
    eval_env = gym.make(ENV_NAME)

    POP_SIZE = 20
    EPISODES_PER_AGENT = 3  # training episodes per agent per generation
    ELITE_FRAC = 0.2

    # Initialize population
    agents = []
    meta_genomes = []
    for i in range(POP_SIZE):
        if meta_learning:
            mg = MetaGenome(num_ff_layers=3)
            agent = HybridFFAgent(ff_lr=mg.ff_lrs[0], ff_threshold=mg.ff_thresholds[0],
                                  policy_lr=mg.policy_lr)
            apply_meta_to_agent(agent, mg)
            meta_genomes.append(mg)
        else:
            agent = HybridFFAgent(ff_lr=0.01, ff_threshold=2.0, policy_lr=0.005)
        agents.append(agent)

    if agents:
        print(f"{label} params per agent: {agents[0].param_count()}")

    results = []
    meta_history = []
    total_eps_used = 0
    generation = 0

    while total_eps_used < total_episodes:
        generation += 1
        fitnesses = []

        for i, agent in enumerate(agents):
            agent_fitness = []
            for _ in range(EPISODES_PER_AGENT):
                if total_eps_used >= total_episodes:
                    break
                r = run_episode(env, agent, train_ff=True, train_policy=True)
                agent_fitness.append(r)
                total_eps_used += 1
            fitnesses.append(np.mean(agent_fitness) if agent_fitness else -999)
            if total_eps_used >= total_episodes:
                break

        # Evaluate best agent
        if total_eps_used % eval_interval < POP_SIZE * EPISODES_PER_AGENT or generation <= 2:
            best_idx = np.argmax(fitnesses)
            eval_r = evaluate_agent(eval_env, agents[best_idx])
            results.append({'episode': total_eps_used, 'eval_reward': eval_r})
            print(f"  {label} gen {generation} (ep {total_eps_used}): best_train={max(fitnesses):.1f}, eval={eval_r:.1f}")

            if meta_learning and meta_genomes:
                best_meta = meta_genomes[best_idx]
                meta_history.append({
                    'generation': generation,
                    'episode': total_eps_used,
                    **{f'ff_lr_{j}': best_meta.ff_lrs[j] for j in range(len(best_meta.ff_lrs))},
                    **{f'ff_thresh_{j}': best_meta.ff_thresholds[j] for j in range(len(best_meta.ff_thresholds))},
                    'policy_lr': best_meta.policy_lr,
                    'goodness_type': best_meta.goodness_type
                })

        # Evolution step
        n_elite = max(2, int(POP_SIZE * ELITE_FRAC))
        sorted_idx = np.argsort(fitnesses)[::-1]
        elite_idx = sorted_idx[:n_elite]

        # Get genomes of elites
        elite_genomes = [agent_to_genome(agents[i]) for i in elite_idx]
        elite_metas = [meta_genomes[i] if meta_learning else None for i in elite_idx]

        # Create next generation
        new_agents = []
        new_metas = []

        # Keep elites
        for i in range(n_elite):
            new_agents.append(agents[elite_idx[i]])
            if meta_learning:
                new_metas.append(meta_genomes[elite_idx[i]])

        # Fill rest with offspring
        pop_fitness_pairs = list(zip(elite_genomes, [fitnesses[i] for i in elite_idx]))
        while len(new_agents) < POP_SIZE:
            p1 = tournament_select(pop_fitness_pairs, k=min(3, len(pop_fitness_pairs)))
            p2 = tournament_select(pop_fitness_pairs, k=min(3, len(pop_fitness_pairs)))
            child_genome = crossover_genomes(p1, p2)
            child_genome = mutate_genome(child_genome, mutation_rate=0.15, mutation_strength=0.08)

            if meta_learning:
                parent_meta = elite_metas[np.random.randint(len(elite_metas))]
                child_meta = parent_meta.mutate()
                child_agent = HybridFFAgent(ff_lr=child_meta.ff_lrs[0],
                                            ff_threshold=child_meta.ff_thresholds[0],
                                            policy_lr=child_meta.policy_lr)
                apply_meta_to_agent(child_agent, child_meta)
                new_metas.append(child_meta)
            else:
                child_agent = HybridFFAgent()

            genome_to_agent(child_genome, child_agent)
            new_agents.append(child_agent)

        agents = new_agents
        meta_genomes = new_metas if meta_learning else []

    env.close()
    eval_env.close()
    return results, meta_history


# ─── Main ──────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("GAIA Phase 4: Hybrid Evo+FF vs Backprop on LunarLander")
    print("=" * 60)

    # Method A: Hybrid Evo+FF (fixed)
    print("\n[A] Hybrid Evo+FF (fixed hyperparams)...")
    t0 = time.time()
    results_a, _ = run_hybrid_evo_ff(meta_learning=False, label="hybrid_fixed")
    time_a = time.time() - t0
    print(f"  Time: {time_a:.0f}s")

    # Method B: Hybrid Evo+FF with meta-learned plasticity
    print("\n[B] Hybrid Evo+FF (meta-learned plasticity)...")
    t0 = time.time()
    results_b, meta_hist = run_hybrid_evo_ff(meta_learning=True, label="hybrid_meta")
    time_b = time.time() - t0
    print(f"  Time: {time_b:.0f}s")

    # Method C: Backprop actor-critic
    print("\n[C] Backprop Actor-Critic...")
    t0 = time.time()
    results_c = run_backprop_baseline()
    time_c = time.time() - t0
    print(f"  Time: {time_c:.0f}s")

    # Save CSVs
    for name, data in [('hybrid_fixed', results_a), ('hybrid_meta', results_b), ('backprop_ac', results_c)]:
        path = os.path.join(OUT_DIR, f'{name}_results.csv')
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['episode', 'eval_reward'])
            writer.writeheader()
            writer.writerows(data)
        print(f"Saved {path}")

    if meta_hist:
        path = os.path.join(OUT_DIR, 'meta_history.csv')
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=meta_hist[0].keys())
            writer.writeheader()
            writer.writerows(meta_hist)
        print(f"Saved {path}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for name, data, t in [('Hybrid Fixed', results_a, time_a),
                           ('Hybrid Meta', results_b, time_b),
                           ('Backprop AC', results_c, time_c)]:
        if data:
            best = max(d['eval_reward'] for d in data)
            final = data[-1]['eval_reward']
            print(f"  {name:20s}: best={best:8.1f}, final={final:8.1f}, time={t:.0f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
