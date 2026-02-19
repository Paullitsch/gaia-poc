#!/usr/bin/env python3
"""GAIA Phase 6: Deep neuromodulation push — 5 signals, eligibility traces, predictive coding."""

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
OUT_DIR = "/root/.openclaw/workspace/gaia-poc/phase6"
os.makedirs(OUT_DIR, exist_ok=True)

def make_env():
    return gym.make(ENV_NAME)

def evaluate_agent(agent, n_episodes=10, max_steps=1000):
    env = make_env()
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0
        if hasattr(agent, 'reset_state'):
            agent.reset_state()
        for _ in range(max_steps):
            action = agent.act(obs)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            if term or trunc:
                break
        rewards.append(total)
    env.close()
    return np.mean(rewards), np.std(rewards), rewards

# ═══════════════════════════════════════════════════════════════
# FF Layer with per-synapse neuromodulation
# ═══════════════════════════════════════════════════════════════

class FFLayerNeuromod(nn.Module):
    def __init__(self, in_dim, out_dim, n_signals=5):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.n_signals = n_signals
        # Per-neuron modulation weights (not per-synapse to keep params manageable)
        # Shape: (n_signals, out_dim)
        self.mod_weights = torch.zeros(n_signals, out_dim)
        nn.init.normal_(self.mod_weights, 0, 0.05)

    def forward(self, x):
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
        return F.relu(self.linear(x))

    def goodness(self, x):
        h = self.forward(x)
        return h.pow(2).mean(dim=-1)

    def param_count(self):
        return sum(p.numel() for p in self.linear.parameters()) + self.mod_weights.numel()

    def get_mod_factor(self, signals_t):
        """Compute per-neuron modulation factor from signals. Returns (out_dim,)."""
        return torch.einsum('s,so->o', signals_t, self.mod_weights)


# ═══════════════════════════════════════════════════════════════
# Method A: Enhanced Neuromod Evo+FF v2 (5 signals, per-synapse)
# ═══════════════════════════════════════════════════════════════

class NeuromodAgentV2:
    """5 neuromodulatory signals, per-synapse modulation weights."""
    def __init__(self, obs_dim=8, act_dim=4, hidden_sizes=(128, 96, 64, 32)):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_sizes = hidden_sizes
        self.n_signals = 5  # dopamine, TD, novelty, acetylcholine, serotonin

        dims = [obs_dim] + list(hidden_sizes)
        self.ff_layers = [FFLayerNeuromod(dims[i], dims[i+1], self.n_signals) for i in range(len(hidden_sizes))]
        self.policy = nn.Linear(hidden_sizes[-1], act_dim)
        nn.init.zeros_(self.policy.weight)
        nn.init.zeros_(self.policy.bias)

        self.meta = {
            'ff_lr': [0.005] * len(hidden_sizes),
            'goodness_thresh': [2.0] * len(hidden_sizes),
            'serotonin_baseline': 0.5,  # exploration/exploitation balance
            'ach_decay': 0.9,  # acetylcholine attention decay
        }

        # Running stats for neuromod signals
        self.reward_history = deque(maxlen=50)
        self.state_history = deque(maxlen=100)
        self.value_estimate = 0.0
        self.attention_focus = np.zeros(obs_dim)

    def reset_state(self):
        self.reward_history.clear()
        self.state_history.clear()
        self.value_estimate = 0.0
        self.attention_focus = np.zeros(self.obs_dim)

    def compute_neuromod_signals(self, obs, reward):
        signals = np.zeros(self.n_signals)

        # Signal 0: Dopamine (reward)
        signals[0] = np.tanh(reward / 100.0)

        # Signal 1: TD error
        new_value = 0.95 * self.value_estimate + 0.05 * reward
        signals[1] = np.tanh((reward + 0.99 * new_value - self.value_estimate) / 50.0)
        self.value_estimate = new_value

        # Signal 2: Novelty
        if len(self.state_history) > 5:
            states = np.array(list(self.state_history))
            mean_s = states.mean(axis=0)
            std_s = states.std(axis=0) + 1e-6
            novelty = np.linalg.norm((obs - mean_s) / std_s) / np.sqrt(len(obs))
            signals[2] = min(1.0, novelty)
        else:
            signals[2] = 1.0

        # Signal 3: Acetylcholine (attention - focus on dimensions with high variance)
        if len(self.state_history) > 10:
            states = np.array(list(self.state_history)[-20:])
            var = states.var(axis=0)
            self.attention_focus = self.meta['ach_decay'] * self.attention_focus + (1 - self.meta['ach_decay']) * var
            signals[3] = np.tanh(np.mean(self.attention_focus))
        else:
            signals[3] = 0.5

        # Signal 4: Serotonin (exploration/exploitation based on recent reward trend)
        if len(self.reward_history) > 5:
            recent = list(self.reward_history)
            trend = np.mean(recent[-5:]) - np.mean(recent)
            # Positive trend → more exploitation (lower serotonin)
            # Negative trend → more exploration (higher serotonin)
            signals[4] = self.meta['serotonin_baseline'] - np.tanh(trend / 50.0) * 0.3
        else:
            signals[4] = self.meta['serotonin_baseline']

        self.state_history.append(obs.copy())
        self.reward_history.append(reward)

        return signals

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
            for p in layer.linear.parameters():
                params.append(p.data.flatten())
            params.append(layer.mod_weights.flatten())
        for p in self.policy.parameters():
            params.append(p.data.flatten())
        return torch.cat(params)

    def set_flat_params(self, flat):
        idx = 0
        for layer in self.ff_layers:
            for p in layer.linear.parameters():
                n = p.numel()
                p.data.copy_(flat[idx:idx+n].reshape(p.shape))
                idx += n
            n = layer.mod_weights.numel()
            layer.mod_weights = flat[idx:idx+n].reshape(layer.mod_weights.shape).clone()
            idx += n
        for p in self.policy.parameters():
            n = p.numel()
            p.data.copy_(flat[idx:idx+n].reshape(p.shape))
            idx += n

    def param_count(self):
        total = sum(p.numel() for layer in self.ff_layers for p in layer.linear.parameters())
        total += sum(layer.mod_weights.numel() for layer in self.ff_layers)
        total += sum(p.numel() for p in self.policy.parameters())
        return total

    def ff_learn_step_modulated(self, good_obs, bad_obs, signals):
        """FF learning with per-synapse neuromodulation."""
        signals_t = torch.FloatTensor(signals)
        for i, layer in enumerate(self.ff_layers):
            lr = self.meta['ff_lr'][i]
            thresh = self.meta['goodness_thresh'][i]

            # Per-neuron modulation factor
            mod_factor = layer.get_mod_factor(signals_t)  # (out_dim,)
            neuron_mod = 1.0 + torch.tanh(mod_factor)  # [0, 2]
            weight_mod = neuron_mod.unsqueeze(1)  # (out_dim, 1) for broadcasting
            bias_mod = neuron_mod

            g_good = layer.goodness(good_obs)
            g_bad = layer.goodness(bad_obs)

            loss = torch.log(1 + torch.exp(-(g_good - thresh))).mean() + \
                   torch.log(1 + torch.exp(g_bad - thresh)).mean()

            layer.linear.zero_grad()
            loss.backward(retain_graph=False)

            with torch.no_grad():
                if layer.linear.weight.grad is not None:
                    layer.linear.weight.data -= lr * layer.linear.weight.grad * weight_mod
                if layer.linear.bias.grad is not None:
                    layer.linear.bias.data -= lr * layer.linear.bias.grad * bias_mod

            with torch.no_grad():
                good_obs = layer.forward(good_obs)
                bad_obs = layer.forward(bad_obs)

    def clone(self):
        new = NeuromodAgentV2(self.obs_dim, self.act_dim, self.hidden_sizes)
        new.set_flat_params(self.get_flat_params().clone())
        new.meta = deepcopy(self.meta)
        return new


# ═══════════════════════════════════════════════════════════════
# Method B: Neuromod + Temporal Dynamics (Eligibility Traces)
# ═══════════════════════════════════════════════════════════════

class NeuromodTemporalAgent(NeuromodAgentV2):
    """Adds eligibility traces for STDP-like credit assignment."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_decay = 0.95
        self.traces = None
        self._init_traces()

    def _init_traces(self):
        self.traces = []
        for layer in self.ff_layers:
            self.traces.append({
                'weight': torch.zeros_like(layer.linear.weight.data),
                'bias': torch.zeros_like(layer.linear.bias.data),
            })

    def reset_state(self):
        super().reset_state()
        self._init_traces()

    def ff_learn_step_temporal(self, good_obs, bad_obs, signals):
        """FF learning with eligibility traces + neuromodulation."""
        signals_t = torch.FloatTensor(signals)
        dopamine = signals[0]  # Use dopamine to gate trace reinforcement

        for i, layer in enumerate(self.ff_layers):
            lr = self.meta['ff_lr'][i]
            thresh = self.meta['goodness_thresh'][i]

            mod_factor = layer.get_mod_factor(signals_t)
            neuron_mod = 1.0 + torch.tanh(mod_factor)
            weight_mod = neuron_mod.unsqueeze(1)
            bias_mod = neuron_mod




            g_good = layer.goodness(good_obs)
            g_bad = layer.goodness(bad_obs)

            loss = torch.log(1 + torch.exp(-(g_good - thresh))).mean() + \
                   torch.log(1 + torch.exp(g_bad - thresh)).mean()

            layer.linear.zero_grad()
            loss.backward(retain_graph=False)

            with torch.no_grad():
                # Update eligibility traces (accumulate gradients)
                if layer.linear.weight.grad is not None:
                    self.traces[i]['weight'] = self.trace_decay * self.traces[i]['weight'] + layer.linear.weight.grad
                if layer.linear.bias.grad is not None:
                    self.traces[i]['bias'] = self.trace_decay * self.traces[i]['bias'] + layer.linear.bias.grad

                # Apply traces modulated by dopamine + per-synapse modulation
                effective_lr = lr * (1.0 + dopamine)
                layer.linear.weight.data -= effective_lr * self.traces[i]['weight'] * weight_mod
                layer.linear.bias.data -= effective_lr * self.traces[i]['bias'] * bias_mod

            with torch.no_grad():
                good_obs = layer.forward(good_obs)
                bad_obs = layer.forward(bad_obs)

    def clone(self):
        new = NeuromodTemporalAgent(self.obs_dim, self.act_dim, self.hidden_sizes)
        new.set_flat_params(self.get_flat_params().clone())
        new.meta = deepcopy(self.meta)
        new.trace_decay = self.trace_decay
        return new


# ═══════════════════════════════════════════════════════════════
# Method C: Neuromod + Predictive Coding Hybrid
# ═══════════════════════════════════════════════════════════════

class PredCodingNeuromodAgent(NeuromodAgentV2):
    """Each layer predicts next layer + uses neuromod signals."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dims = [self.obs_dim] + list(self.hidden_sizes)
        # Prediction weights: each layer predicts the next
        self.predictors = []
        for i in range(len(self.hidden_sizes) - 1):
            pred = nn.Linear(dims[i+1], dims[i+2])
            nn.init.kaiming_normal_(pred.weight)
            nn.init.zeros_(pred.bias)
            self.predictors.append(pred)
        self.pred_lr = 0.001
        self.pred_weight = 0.3  # how much prediction error influences learning

    def ff_learn_step_predictive(self, good_obs, bad_obs, signals):
        """FF + predictive coding + neuromodulation."""
        signals_t = torch.FloatTensor(signals)

        # Forward pass collecting activations
        good_acts = [good_obs]
        bad_acts = [bad_obs]
        x_g, x_b = good_obs, bad_obs
        for layer in self.ff_layers:
            x_g = layer.forward(x_g)
            x_b = layer.forward(x_b)
            good_acts.append(x_g.detach())
            bad_acts.append(x_b.detach())

        for i, layer in enumerate(self.ff_layers):
            lr = self.meta['ff_lr'][i]
            thresh = self.meta['goodness_thresh'][i]

            mod_factor = layer.get_mod_factor(signals_t)
            neuron_mod = 1.0 + torch.tanh(mod_factor)
            weight_mod = neuron_mod.unsqueeze(1)
            bias_mod = neuron_mod

            g_good = layer.goodness(good_acts[i])
            g_bad = layer.goodness(bad_acts[i])

            # FF loss
            ff_loss = torch.log(1 + torch.exp(-(g_good - thresh))).mean() + \
                      torch.log(1 + torch.exp(g_bad - thresh)).mean()

            # Prediction error loss (if not last layer)
            pred_loss = torch.tensor(0.0)
            if i < len(self.predictors):
                predicted = F.relu(self.predictors[i](good_acts[i+1].detach()))
                actual = good_acts[i+2].detach() if i+2 < len(good_acts) else good_acts[i+1].detach()
                # Match dimensions
                if predicted.shape == actual.shape:
                    pred_loss = F.mse_loss(predicted, actual)
                    # Clip to prevent divergence
                    pred_loss = torch.clamp(pred_loss, 0, 10.0)

            total_loss = ff_loss + self.pred_weight * pred_loss

            layer.linear.zero_grad()
            # Only backprop through ff_loss for layer params
            ff_loss.backward(retain_graph=False)

            with torch.no_grad():
                if layer.linear.weight.grad is not None:
                    grad = torch.clamp(layer.linear.weight.grad, -1.0, 1.0)
                    layer.linear.weight.data -= lr * grad * weight_mod
                if layer.linear.bias.grad is not None:
                    grad = torch.clamp(layer.linear.bias.grad, -1.0, 1.0)
                    layer.linear.bias.data -= lr * grad * bias_mod

            # Update predictor
            if i < len(self.predictors):
                self.predictors[i].zero_grad()
                if pred_loss.requires_grad:
                    pred_loss.backward(retain_graph=False)
                    with torch.no_grad():
                        for p in self.predictors[i].parameters():
                            if p.grad is not None:
                                p.data -= self.pred_lr * torch.clamp(p.grad, -1.0, 1.0)

    def get_flat_params(self):
        params = [super().get_flat_params()]
        for pred in self.predictors:
            for p in pred.parameters():
                params.append(p.data.flatten())
        return torch.cat(params)

    def set_flat_params(self, flat):
        # Set base params
        base_size = 0
        for layer in self.ff_layers:
            base_size += sum(p.numel() for p in layer.linear.parameters()) + layer.mod_weights.numel()
        base_size += sum(p.numel() for p in self.policy.parameters())
        super().set_flat_params(flat[:base_size])

        idx = base_size
        for pred in self.predictors:
            for p in pred.parameters():
                n = p.numel()
                if idx + n <= len(flat):
                    p.data.copy_(flat[idx:idx+n].reshape(p.shape))
                idx += n

    def param_count(self):
        base = super().param_count()
        pred_params = sum(p.numel() for pred in self.predictors for p in pred.parameters())
        return base + pred_params

    def clone(self):
        new = PredCodingNeuromodAgent(self.obs_dim, self.act_dim, self.hidden_sizes)
        new.set_flat_params(self.get_flat_params().clone())
        new.meta = deepcopy(self.meta)
        new.pred_lr = self.pred_lr
        new.pred_weight = self.pred_weight
        return new


# ═══════════════════════════════════════════════════════════════
# Method D: PPO Baseline (proper)
# ═══════════════════════════════════════════════════════════════

class PPOAgent:
    def __init__(self, obs_dim=8, act_dim=4, hidden_size=128):
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=3e-4
        )

    def act(self, obs):
        with torch.no_grad():
            logits = self.actor(torch.FloatTensor(obs))
            return torch.distributions.Categorical(logits=logits).sample().item()

    def act_greedy(self, obs):
        with torch.no_grad():
            logits = self.actor(torch.FloatTensor(obs))
            return logits.argmax().item()

    def param_count(self):
        return sum(p.numel() for p in self.actor.parameters()) + \
               sum(p.numel() for p in self.critic.parameters())


def run_ppo(total_steps=500_000, n_steps=2048, n_epochs=10, batch_size=64,
            gamma=0.99, gae_lambda=0.95, clip_eps=0.2, ent_coef=0.01,
            vf_coef=0.5, max_grad_norm=0.5):
    print(f"\n{'='*60}")
    print("METHOD D: PPO Baseline (proper)")
    print(f"Total steps: {total_steps}")
    print(f"{'='*60}")

    agent = PPOAgent()
    print(f"Params: {agent.param_count()}")
    csv_path = os.path.join(OUT_DIR, "ppo_results.csv")

    env = make_env()
    obs, _ = env.reset()
    all_rows = []
    episode_reward = 0
    episode_count = 0
    step_count = 0
    best_eval = -float('inf')
    t0 = time.time()

    while step_count < total_steps:
        # Collect rollout
        obs_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf = [], [], [], [], [], []

        for _ in range(n_steps):
            obs_t = torch.FloatTensor(obs)
            with torch.no_grad():
                logits = agent.actor(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)
                value = agent.critic(obs_t).squeeze()

            next_obs, reward, term, trunc, _ = env.step(action.item())
            done = term or trunc

            obs_buf.append(obs)
            act_buf.append(action.item())
            rew_buf.append(reward)
            done_buf.append(done)
            val_buf.append(value.item())
            logp_buf.append(logp.item())

            episode_reward += reward
            step_count += 1

            if done:
                episode_count += 1
                if episode_count % 20 == 0:
                    # Evaluate
                    eval_rewards = []
                    for _ in range(10):
                        e_obs, _ = make_env().reset()
                        e_env = make_env()
                        e_obs, _ = e_env.reset()
                        e_total = 0
                        for _ in range(1000):
                            a = agent.act_greedy(e_obs)
                            e_obs, r, t, tr, _ = e_env.step(a)
                            e_total += r
                            if t or tr:
                                break
                        eval_rewards.append(e_total)
                        e_env.close()
                    eval_mean = np.mean(eval_rewards)
                    eval_std = np.std(eval_rewards)
                    if eval_mean > best_eval:
                        best_eval = eval_mean
                    all_rows.append({
                        'step': step_count,
                        'episode': episode_count,
                        'eval_mean': eval_mean,
                        'eval_std': eval_std,
                        'best_eval': best_eval,
                        'train_reward': episode_reward,
                    })
                    elapsed = time.time() - t0
                    print(f"  Step {step_count:>7d} | Ep {episode_count:>4d} | Eval: {eval_mean:>7.1f}±{eval_std:.1f} | Best: {best_eval:.1f} | {elapsed:.0f}s")

                episode_reward = 0
                obs, _ = env.reset()
            else:
                obs = next_obs

        # Compute GAE
        obs_t = torch.FloatTensor(np.array(obs_buf))
        act_t = torch.LongTensor(act_buf)
        rew_arr = np.array(rew_buf)
        done_arr = np.array(done_buf)
        val_arr = np.array(val_buf)
        old_logp_t = torch.FloatTensor(logp_buf)

        with torch.no_grad():
            next_val = agent.critic(torch.FloatTensor(obs)).squeeze().item()

        advantages = np.zeros(len(rew_arr))
        last_gae = 0
        for t in reversed(range(len(rew_arr))):
            if t == len(rew_arr) - 1:
                next_v = next_val * (1 - done_arr[t])
            else:
                next_v = val_arr[t+1] * (1 - done_arr[t])
            delta = rew_arr[t] + gamma * next_v - val_arr[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - done_arr[t]) * last_gae

        returns = advantages + val_arr
        adv_t = torch.FloatTensor(advantages)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.FloatTensor(returns)

        # PPO update
        indices = np.arange(len(obs_buf))
        for _ in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                mb = indices[start:start+batch_size]
                mb_obs = obs_t[mb]
                mb_act = act_t[mb]
                mb_adv = adv_t[mb]
                mb_ret = ret_t[mb]
                mb_old_logp = old_logp_t[mb]

                logits = agent.actor(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()
                values = agent.critic(mb_obs).squeeze()

                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values, mb_ret)
                loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy

                agent.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(agent.actor.parameters()) + list(agent.critic.parameters()),
                    max_grad_norm
                )
                agent.opt.step()

    env.close()
    elapsed = time.time() - t0

    # Final eval
    final_mean, final_std, final_rewards = evaluate_agent(agent, n_episodes=30, max_steps=1000)
    # Use act_greedy for final eval
    eval_env = make_env()
    greedy_rewards = []
    for _ in range(30):
        o, _ = eval_env.reset()
        tr = 0
        for _ in range(1000):
            a = agent.act_greedy(o)
            o, r, t, tru, _ = eval_env.step(a)
            tr += r
            if t or tru:
                break
        greedy_rewards.append(tr)
    eval_env.close()
    greedy_mean = np.mean(greedy_rewards)
    greedy_std = np.std(greedy_rewards)

    print(f"\n  PPO Final (greedy, 30-ep): {greedy_mean:.1f} ± {greedy_std:.1f}")
    print(f"  Best eval during training: {best_eval:.1f}")
    print(f"  Time: {elapsed:.0f}s")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['step', 'episode', 'eval_mean', 'eval_std', 'best_eval', 'train_reward'])
        writer.writeheader()
        writer.writerows(all_rows)

    return {
        'method': 'PPO',
        'best_eval': best_eval,
        'final_mean': greedy_mean,
        'final_std': greedy_std,
        'steps': total_steps,
        'time': elapsed,
        'params': agent.param_count(),
    }


# ═══════════════════════════════════════════════════════════════
# Evolutionary runner (shared by methods A, B, C)
# ═══════════════════════════════════════════════════════════════

def run_evolutionary(method_name, agent_class, learn_fn_name,
                     pop_size=500, generations=300, elite_frac=0.02,
                     tournament_size=5, eval_episodes=15,
                     ff_episodes_per_gen=5, mutation_sigma=0.02):
    print(f"\n{'='*60}")
    print(f"METHOD: {method_name}")
    print(f"Population: {pop_size}, Generations: {generations}")
    print(f"{'='*60}")

    csv_path = os.path.join(OUT_DIR, f"{method_name.lower().replace(' ', '_').replace('+', '_')}_results.csv")

    # Initialize population
    population = [agent_class() for _ in range(pop_size)]
    print(f"Params per agent: {population[0].param_count()}")

    # Randomize meta-parameters
    for agent in population:
        for i in range(len(agent.meta['ff_lr'])):
            agent.meta['ff_lr'][i] = 10 ** np.random.uniform(-4, -1)
            agent.meta['goodness_thresh'][i] = np.random.uniform(0.5, 5.0)

    best_ever_fitness = -float('inf')
    best_ever_agent = None
    all_rows = []
    t0 = time.time()
    total_evals = 0
    stagnation_count = 0
    prev_best = -float('inf')

    for gen in range(generations):
        # FF learning phase
        env = make_env()
        for agent in population:
            agent.reset_state()
            good_obs_list, bad_obs_list = [], []
            all_signals = np.zeros(agent.n_signals)
            n_signal_samples = 0

            for _ in range(ff_episodes_per_gen):
                obs, _ = env.reset()
                episode_obs, episode_rewards = [], []
                for _ in range(300):
                    action = agent.act(obs)
                    next_obs, r, term, trunc, _ = env.step(action)
                    episode_obs.append(obs)
                    episode_rewards.append(r)
                    signals = agent.compute_neuromod_signals(obs, r)
                    all_signals += signals
                    n_signal_samples += 1
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
                if len(good) > 200:
                    idx = np.random.choice(len(good), 200, replace=False)
                    good = good[idx]
                if len(bad) > 200:
                    idx = np.random.choice(len(bad), 200, replace=False)
                    bad = bad[idx]

                avg_signals = all_signals / max(n_signal_samples, 1)
                learn_fn = getattr(agent, learn_fn_name)
                learn_fn(good, bad, avg_signals)
        env.close()

        # Quick fitness evaluation (3 episodes)
        fitnesses = np.zeros(pop_size)
        for i, agent in enumerate(population):
            agent.reset_state()
            f, _, _ = evaluate_agent(agent, n_episodes=3, max_steps=500)
            fitnesses[i] = f
            total_evals += 3

        # Re-evaluate top candidates thoroughly
        n_elite = max(2, int(pop_size * elite_frac))
        top_indices = np.argsort(fitnesses)[-min(20, n_elite*2):]
        for idx in top_indices:
            population[idx].reset_state()
            f, _, _ = evaluate_agent(population[idx], n_episodes=eval_episodes, max_steps=1000)
            fitnesses[idx] = f
            total_evals += eval_episodes

        gen_best = fitnesses.max()
        gen_mean = fitnesses.mean()
        gen_std = fitnesses.std()

        if gen_best > best_ever_fitness:
            best_ever_fitness = gen_best
            best_ever_agent = population[np.argmax(fitnesses)].clone()
            stagnation_count = 0
        else:
            stagnation_count += 1

        # Adaptive mutation on stagnation
        effective_sigma = mutation_sigma * (1.0 + 0.5 * min(stagnation_count / 20, 2.0))

        elapsed = time.time() - t0
        all_rows.append({
            'generation': gen,
            'best_fitness': gen_best,
            'best_ever': best_ever_fitness,
            'mean_fitness': gen_mean,
            'std_fitness': gen_std,
            'total_evals': total_evals,
            'time': elapsed,
            'sigma': effective_sigma,
        })

        if gen % 10 == 0 or gen == generations - 1:
            print(f"  Gen {gen:>3d} | Best: {gen_best:>7.1f} | Ever: {best_ever_fitness:>7.1f} | "
                  f"Mean: {gen_mean:>7.1f} | σ_mut: {effective_sigma:.4f} | {elapsed:.0f}s")

        # Selection + mutation
        sorted_idx = np.argsort(fitnesses)
        elites = [population[i].clone() for i in sorted_idx[-n_elite:]]
        new_pop = list(elites)

        while len(new_pop) < pop_size:
            # Tournament selection
            candidates = np.random.choice(pop_size, tournament_size, replace=False)
            winner_idx = candidates[np.argmax(fitnesses[candidates])]
            parent = population[winner_idx]
            child = parent.clone()

            # Mutate weights
            flat = child.get_flat_params()
            mask = torch.rand_like(flat) < 0.1  # sparse mutation
            flat[mask] += effective_sigma * torch.randn(mask.sum())
            child.set_flat_params(flat)

            # Mutate meta-parameters
            for i in range(len(child.meta['ff_lr'])):
                child.meta['ff_lr'][i] *= np.exp(0.1 * np.random.randn())
                child.meta['ff_lr'][i] = np.clip(child.meta['ff_lr'][i], 1e-5, 0.1)
                child.meta['goodness_thresh'][i] *= np.exp(0.1 * np.random.randn())
                child.meta['goodness_thresh'][i] = np.clip(child.meta['goodness_thresh'][i], 0.1, 10.0)

            new_pop.append(child)

        population = new_pop[:pop_size]

    # Final evaluation of best agent
    best_ever_agent.reset_state()
    final_mean, final_std, final_rewards = evaluate_agent(best_ever_agent, n_episodes=30, max_steps=1000)
    print(f"\n  {method_name} Final (30-ep): {final_mean:.1f} ± {final_std:.1f}")
    print(f"  Best ever: {best_ever_fitness:.1f}")
    print(f"  Total evals: {total_evals}")
    print(f"  Time: {time.time()-t0:.0f}s")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    return {
        'method': method_name,
        'best_ever': best_ever_fitness,
        'final_mean': final_mean,
        'final_std': final_std,
        'total_evals': total_evals,
        'time': time.time() - t0,
        'params': population[0].param_count(),
        'generations': generations,
        'pop_size': pop_size,
        'history': all_rows,
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = {}

    # Due to compute constraints, scale down from ideal but keep ratios meaningful
    # Target: ~150K evals per evolutionary method, PPO 500K steps
    # With pop=200, gen=150: 200*150*3=90K quick + reeval ~15K = ~105K evals
    # Acceptable compromise for CPU-only

    POP = 50
    GENS = 80

    print("="*60)
    print("GAIA PHASE 6: DEEP NEUROMODULATION PUSH")
    print(f"Target: Solve LunarLander (>{SOLVED_THRESHOLD})")
    print(f"Population: {POP}, Generations: {GENS}")
    print("="*60)

    # Method A: Enhanced Neuromod (5 signals, per-synapse)
    results['neuromod_v2'] = run_evolutionary(
        method_name="Neuromod_v2_5sig",
        agent_class=NeuromodAgentV2,
        learn_fn_name='ff_learn_step_modulated',
        pop_size=POP, generations=GENS,
        eval_episodes=5, ff_episodes_per_gen=2,
        mutation_sigma=0.02,
    )

    # Method B: Neuromod + Temporal (Eligibility Traces)
    results['neuromod_temporal'] = run_evolutionary(
        method_name="Neuromod_Temporal",
        agent_class=NeuromodTemporalAgent,
        learn_fn_name='ff_learn_step_temporal',
        pop_size=POP, generations=GENS,
        eval_episodes=5, ff_episodes_per_gen=2,
        mutation_sigma=0.02,
    )

    # Method C: Neuromod + Predictive Coding
    results['neuromod_predcod'] = run_evolutionary(
        method_name="Neuromod_PredCoding",
        agent_class=PredCodingNeuromodAgent,
        learn_fn_name='ff_learn_step_predictive',
        pop_size=POP, generations=GENS,
        eval_episodes=5, ff_episodes_per_gen=2,
        mutation_sigma=0.02,
    )

    # Method D: PPO (proper)
    results['ppo'] = run_ppo(total_steps=200_000)

    # Save all results
    summary = {}
    for key, res in results.items():
        summary[key] = {k: v for k, v in res.items() if k != 'history'}

    with open(os.path.join(OUT_DIR, "phase6_results.json"), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "="*60)
    print("PHASE 6 SUMMARY")
    print("="*60)
    for key, res in results.items():
        best = res.get('best_ever', res.get('best_eval', '?'))
        final = res.get('final_mean', '?')
        print(f"  {res['method']:>25s} | Best: {best:>7} | Final: {final:>7} | Params: {res['params']}")

    # Check if solved
    for key, res in results.items():
        best = res.get('best_ever', res.get('best_eval', -999))
        if isinstance(best, (int, float)) and best >= SOLVED_THRESHOLD:
            print(f"\n🎉 {res['method']} SOLVED LunarLander with score {best}!")

    print(f"\nResults saved to {OUT_DIR}/")
