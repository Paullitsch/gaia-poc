"""
Meta-Learning: Evolution of Learning Rules.

The biological approach: DON'T evolve weights, evolve the RULES that learn weights.
Like DNA — genes don't encode synapse weights, they encode developmental programs.

Architecture:
- Inner loop: Learning rule updates weights during lifetime (episodes)
- Outer loop: CMA-ES evolves the learning rule parameters

This is the path to scalable gradient-free AI:
- Small genome (learning rule params) → CMA-ES handles this
- Large network (actual weights) → learning rule handles this
- Separation of concerns mirrors biology
"""

import numpy as np
import gymnasium as gym
import time
import multiprocessing as mp
import os

from experiments.cma_es import PolicyNetwork, CMAES


class HebbianLearningRule:
    """Parameterized Hebbian learning rule.
    
    Each synapse updates based on:
    Δw = η * (A * pre * post + B * pre + C * post + D)
    
    Where A, B, C, D, η are evolved parameters per layer.
    This is a generalized Hebbian rule that includes:
    - Pure Hebbian (A>0, others≈0)
    - Anti-Hebbian (A<0)
    - Presynaptic (B≠0)
    - Postsynaptic (C≠0)
    - Decay (D<0)
    """
    
    def __init__(self, n_layers, params_per_layer=5):
        self.n_layers = n_layers
        self.params_per_layer = params_per_layer  # A, B, C, D, eta
        self.n_params = n_layers * params_per_layer
    
    def decode(self, genome):
        """Decode flat genome into per-layer learning rule parameters."""
        rules = []
        for i in range(self.n_layers):
            offset = i * self.params_per_layer
            raw = genome[offset:offset + self.params_per_layer]
            # Constrain parameters to reasonable ranges
            A = np.tanh(raw[0]) * 0.1      # Hebbian coefficient [-0.1, 0.1]
            B = np.tanh(raw[1]) * 0.05     # Presynaptic [-0.05, 0.05]
            C = np.tanh(raw[2]) * 0.05     # Postsynaptic [-0.05, 0.05]
            D = -np.abs(raw[3]) * 0.01     # Decay (always negative) [-0.01, 0]
            eta = np.abs(raw[4]) * 0.01    # Learning rate [0, 0.01]
            rules.append((A, B, C, D, eta))
        return rules


def evaluate_with_learning(policy, init_weights, rule_genome, rule, env_name,
                           n_episodes=5, max_steps=1000, n_lifetime_episodes=3):
    """Evaluate: start with init_weights, apply learning rule across episodes.
    
    Key insight: the learning rule gets multiple episodes to improve the weights.
    First episodes may perform badly, later ones should improve — that's LEARNING.
    """
    rules = rule.decode(rule_genome)
    weights = init_weights.copy()
    
    env = gym.make(env_name)
    total_reward = 0.0
    
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 1000)
        done = False
        ep_reward = 0.0
        steps = 0
        
        # Track activations for Hebbian update
        pre_activations = []
        post_activations = []
        
        while not done and steps < max_steps:
            # Forward pass with activation tracking
            x = obs.astype(np.float32)
            layer_pre = []
            layer_post = []
            
            idx = 0
            for layer_i in range(0, len(policy.shapes), 2):
                pre = x.copy()
                w = weights[idx:idx + policy.sizes[layer_i]].reshape(policy.shapes[layer_i])
                idx += policy.sizes[layer_i]
                b = weights[idx:idx + policy.sizes[layer_i + 1]]
                idx += policy.sizes[layer_i + 1]
                x = x @ w + b
                if layer_i < len(policy.shapes) - 2:
                    x = np.tanh(x)
                post = x.copy()
                layer_pre.append(pre)
                layer_post.append(post)
            
            # Action selection
            if policy.act_type == "discrete":
                action = int(np.argmax(x))
            else:
                action = np.tanh(x)
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            
            # Store activations for batch update
            pre_activations.append(layer_pre)
            post_activations.append(layer_post)
            
            obs = next_obs
            done = terminated or truncated
            steps += 1
        
        total_reward += ep_reward
        
        # Apply Hebbian learning rule after each episode
        # (reward-modulated: scale updates by episode reward)
        reward_signal = max(0, ep_reward / 100.0)  # Normalize, positive only
        
        if len(pre_activations) > 0 and ep < n_episodes - 1:  # Don't update after last episode
            # Sample a batch of timesteps
            n_samples = min(50, len(pre_activations))
            sample_idx = np.random.choice(len(pre_activations), n_samples, replace=False)
            
            idx = 0
            for layer_i in range(len(rules)):
                if layer_i >= len(pre_activations[0]):
                    break
                A, B, C, D, eta = rules[layer_i]
                w_size = policy.sizes[layer_i * 2]
                w_shape = policy.shapes[layer_i * 2]
                
                w = weights[idx:idx + w_size].reshape(w_shape)
                
                # Compute average Hebbian update across sampled timesteps
                dw = np.zeros_like(w)
                for si in sample_idx:
                    pre = pre_activations[si][layer_i]
                    post = post_activations[si][layer_i]
                    # Generalized Hebbian: Δw = A*pre*post + B*pre + C*post + D
                    dw += A * np.outer(pre, post) + B * pre[:, None] + C * post[None, :] + D
                dw /= n_samples
                
                # Reward-modulated update
                w += eta * reward_signal * dw
                # Weight clipping for stability
                w = np.clip(w, -5.0, 5.0)
                
                weights[idx:idx + w_size] = w.flatten()
                idx += w_size + policy.sizes[layer_i * 2 + 1]
    
    env.close()
    return total_reward / n_episodes


def run(params=None, device="cpu", callback=None):
    """Meta-Learning: CMA-ES evolves learning rules, not weights."""
    params = params or {}
    env_name = params.get("environment", "LunarLander-v3")
    max_evals = params.get("max_evals", 100000)
    eval_episodes = params.get("eval_episodes", 5)
    obs_dim = params.get("obs_dim", 8)
    act_dim = params.get("act_dim", 4)
    act_type = params.get("act_type", "discrete")
    hidden = params.get("hidden", [64, 32])
    max_steps = params.get("max_steps", 1000)
    solved_threshold = params.get("solved_threshold", 200)
    n_workers = params.get("n_workers", min(os.cpu_count() or 1, 16))
    n_lifetime_eps = params.get("n_lifetime_episodes", 3)
    
    policy = PolicyNetwork(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden, act_type=act_type)
    n_layers = len(hidden) + 1  # hidden layers + output
    rule = HebbianLearningRule(n_layers)
    
    # Genome = initial weights + learning rule params
    n_total = policy.n_params + rule.n_params
    
    print(f"🧬 Meta-Learning on {env_name}")
    print(f"Network: {obs_dim}→{'→'.join(map(str, hidden))}→{act_dim} ({policy.n_params} weight params)")
    print(f"Learning rule: {rule.n_params} params ({n_layers} layers × {rule.params_per_layer} Hebbian coefficients)")
    print(f"Total genome: {n_total} params | Lifetime episodes: {n_lifetime_eps}")
    
    cma = CMAES(n_total, sigma0=0.5)
    print(f"Budget: {max_evals:,} evals | Pop: {cma.lam} | Workers: {n_workers}")
    
    best_ever = -float("inf")
    best_params = None
    total_evals = 0
    start_time = time.time()
    
    while total_evals < max_evals:
        candidates = cma.ask()
        
        # Split genome into weights + rule
        args = []
        for c in candidates:
            init_w = c[:policy.n_params]
            rule_g = c[policy.n_params:]
            args.append((policy, init_w, rule_g, rule, env_name,
                         eval_episodes, max_steps, n_lifetime_eps))
        
        if n_workers > 1:
            with mp.Pool(n_workers) as pool:
                fitnesses = pool.starmap(evaluate_with_learning, args)
            fitnesses = np.array(fitnesses)
        else:
            fitnesses = np.array([evaluate_with_learning(*a) for a in args])
        total_evals += len(candidates) * eval_episodes
        
        cma.tell(candidates, fitnesses)
        
        gen_best = float(np.max(fitnesses))
        if gen_best > best_ever:
            best_ever = gen_best
            best_params = candidates[np.argmax(fitnesses)].copy()
        
        elapsed = time.time() - start_time
        
        # Decode best learning rule for logging
        if best_params is not None:
            best_rules = rule.decode(best_params[policy.n_params:])
            rule_str = " | ".join([f"A={r[0]:.3f} η={r[4]:.4f}" for r in best_rules[:2]])
        else:
            rule_str = ""
        
        entry = {
            "generation": cma.gen,
            "best_fitness": gen_best,
            "best_ever": float(best_ever),
            "mean_fitness": float(np.mean(fitnesses)),
            "std_fitness": float(np.std(fitnesses)),
            "sigma": float(cma.sigma),
            "total_evals": total_evals,
            "elapsed": round(elapsed, 1),
        }
        if callback:
            callback(entry)
        
        solved = "✅ SOLVED!" if best_ever >= solved_threshold else ""
        print(f"Gen {cma.gen:4d} | Best: {gen_best:8.1f} | Ever: {best_ever:8.1f} | "
              f"Mean: {np.mean(fitnesses):8.1f} | σ: {cma.sigma:.4f} | Evals: {total_evals:>8,} | {elapsed:6.1f}s {solved}")
        
        if best_ever >= solved_threshold:
            print(f"\n🎉 SOLVED with Meta-Learning! Learning rules evolved successfully!")
            if best_params is not None:
                print(f"Evolved rules: {rule_str}")
            break
    
    # Final eval
    if best_params is not None:
        init_w = best_params[:policy.n_params]
        rule_g = best_params[policy.n_params:]
        final_scores = [evaluate_with_learning(policy, init_w, rule_g, rule, env_name,
                                                1, max_steps, n_lifetime_eps)
                        for _ in range(20)]
        final_mean, final_std = float(np.mean(final_scores)), float(np.std(final_scores))
        
        # Log the evolved learning rules
        evolved_rules = rule.decode(rule_g)
        print(f"\n📊 Evolved Learning Rules:")
        for i, (A, B, C, D, eta) in enumerate(evolved_rules):
            print(f"  Layer {i}: A={A:.4f} B={B:.4f} C={C:.4f} D={D:.4f} η={eta:.5f}")
    else:
        final_mean = final_std = 0.0
    
    return {
        "method": "Meta-Learning",
        "environment": env_name,
        "best_ever": float(best_ever),
        "final_mean": final_mean,
        "final_std": final_std,
        "total_evals": total_evals,
        "generations": cma.gen,
        "n_weight_params": policy.n_params,
        "n_rule_params": rule.n_params,
        "n_total_params": n_total,
        "elapsed_seconds": round(time.time() - start_time, 1),
        "solved": best_ever >= solved_threshold,
    }
