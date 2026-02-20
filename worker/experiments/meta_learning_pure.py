"""
Pure Meta-Learning: Evolve ONLY learning rules. Weights start random every time.

THIS is the biological approach:
- DNA encodes learning rules (small, evolvable)
- Synapses start random at birth
- Learning rules shape the brain during lifetime
- Evolution selects rules that produce good learners, not good weights

Key difference from meta_learning.py:
- Genome = ONLY learning rule params (15-25 params for 3-5 layers)
- Weights are randomly initialized each evaluation
- The learning rule must be GENERAL enough to train any random init
- This tests: can evolution find learning rules that reliably produce intelligence?

This is fundamentally harder but more powerful:
- If it works: the rules scale to ANY network size (genome stays small)
- The genome size is O(layers), not O(params)
- Truly separates "what to learn" from "how to learn"
"""

import numpy as np
import gymnasium as gym
import time
import multiprocessing as mp
import os

from experiments.cma_es import PolicyNetwork, CMAES


class AdvancedHebbianRule:
    """Extended Hebbian learning rule with more expressive power.
    
    Per-layer parameters:
    - A: Hebbian coefficient (pre * post correlation)
    - B: Presynaptic term
    - C: Postsynaptic term  
    - D: Weight decay
    - eta: Learning rate
    - mod: Reward modulation strength
    - gate: Eligibility trace decay (how much past matters)
    
    Update rule:
    trace = gate * trace + (A * pre⊗post + B * pre + C * post + D * w)
    Δw = eta * (reward_signal * mod + (1-mod)) * trace
    
    This allows:
    - Pure Hebbian (mod≈0): learns correlations regardless of reward
    - Reward-modulated (mod≈1): only updates when rewarded (like dopamine)
    - Mixed: baseline Hebbian + reward bonus
    """
    
    def __init__(self, n_layers, params_per_layer=7):
        self.n_layers = n_layers
        self.params_per_layer = params_per_layer
        self.n_params = n_layers * params_per_layer
    
    def decode(self, genome):
        """Decode flat genome into per-layer rule parameters."""
        rules = []
        for i in range(self.n_layers):
            offset = i * self.params_per_layer
            raw = genome[offset:offset + self.params_per_layer]
            A = np.tanh(raw[0]) * 0.1       # Hebbian [-0.1, 0.1]
            B = np.tanh(raw[1]) * 0.05      # Presynaptic [-0.05, 0.05]
            C = np.tanh(raw[2]) * 0.05      # Postsynaptic [-0.05, 0.05]
            D = np.tanh(raw[3]) * 0.01      # Weight decay [-0.01, 0.01]
            eta = np.abs(raw[4]) * 0.02     # Learning rate [0, 0.02]
            mod = 1.0 / (1.0 + np.exp(-raw[5]))  # Reward modulation [0, 1] (sigmoid)
            gate = 1.0 / (1.0 + np.exp(-raw[6]))  # Eligibility trace [0, 1]
            rules.append((A, B, C, D, eta, mod, gate))
        return rules


def evaluate_pure(policy, rule_genome, rule, env_name, seed,
                  n_eval_episodes=5, max_steps=1000, n_lifetime_episodes=10):
    """Evaluate learning rules starting from RANDOM weights.
    
    Lifecycle:
    1. Random weight initialization (different each eval via seed)
    2. N "training" episodes where learning rule modifies weights
    3. Final episodes scored (only last episodes count — measures what was LEARNED)
    
    The seed ensures reproducibility within a generation but different
    random inits across generations (diversity).
    """
    rng = np.random.RandomState(seed)
    
    # Random weight init — Xavier/He-like scaling
    weights = np.zeros(policy.n_params, dtype=np.float32)
    idx = 0
    for i in range(0, len(policy.shapes), 2):
        fan_in = policy.shapes[i][0]
        fan_out = policy.shapes[i][1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        w_size = policy.sizes[i]
        weights[idx:idx + w_size] = rng.randn(w_size).astype(np.float32) * std
        idx += w_size
        b_size = policy.sizes[i + 1]
        weights[idx:idx + b_size] = 0.0  # Biases start at 0
        idx += b_size
    
    rules = rule.decode(rule_genome)
    env = gym.make(env_name)
    
    # Eligibility traces (one per layer)
    traces = []
    for i in range(0, len(policy.shapes), 2):
        traces.append(np.zeros(policy.shapes[i], dtype=np.float32))
    
    total_episodes = n_lifetime_episodes + n_eval_episodes
    training_reward = 0.0
    eval_reward = 0.0
    
    for ep in range(total_episodes):
        is_eval = (ep >= n_lifetime_episodes)
        obs, _ = env.reset(seed=seed * 1000 + ep)
        done = False
        ep_reward = 0.0
        steps = 0
        
        # Collect activations for Hebbian update
        ep_pre = []
        ep_post = []
        
        while not done and steps < max_steps:
            x = obs.astype(np.float32)
            layer_pre = []
            layer_post = []
            
            w_idx = 0
            for layer_i in range(0, len(policy.shapes), 2):
                pre = x.copy()
                w = weights[w_idx:w_idx + policy.sizes[layer_i]].reshape(policy.shapes[layer_i])
                w_idx += policy.sizes[layer_i]
                b = weights[w_idx:w_idx + policy.sizes[layer_i + 1]]
                w_idx += policy.sizes[layer_i + 1]
                x = x @ w + b
                if layer_i < len(policy.shapes) - 2:
                    x = np.tanh(x)
                layer_pre.append(pre)
                layer_post.append(x.copy())
            
            if policy.act_type == "discrete":
                action = int(np.argmax(x))
            else:
                action = np.tanh(x)
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            ep_pre.append(layer_pre)
            ep_post.append(layer_post)
            
            obs = next_obs
            done = terminated or truncated
            steps += 1
        
        if is_eval:
            eval_reward += ep_reward
        else:
            training_reward += ep_reward
        
        # Apply learning rule after each TRAINING episode (not eval episodes)
        if not is_eval and len(ep_pre) > 0:
            # Reward signal: normalize to roughly [-1, 1]
            reward_signal = np.tanh(ep_reward / 200.0)
            
            n_samples = min(100, len(ep_pre))
            sample_idx = rng.choice(len(ep_pre), n_samples, replace=False)
            
            w_idx = 0
            for li, rule_li in enumerate(range(min(len(rules), len(ep_pre[0])))):
                A, B, C, D, eta, mod, gate = rules[rule_li]
                w_size = policy.sizes[li * 2]
                w_shape = policy.shapes[li * 2]
                
                w = weights[w_idx:w_idx + w_size].reshape(w_shape)
                
                # Compute Hebbian update from sampled timesteps
                dw = np.zeros_like(w)
                for si in sample_idx:
                    pre = ep_pre[si][rule_li]
                    post = ep_post[si][rule_li]
                    dw += A * np.outer(pre, post) + B * pre[:, None] + C * post[None, :] + D * w
                dw /= n_samples
                
                # Update eligibility trace
                traces[li] = gate * traces[li] + dw
                
                # Reward-modulated update: mix autonomous + reward-gated learning
                effective_signal = reward_signal * mod + (1.0 - mod)
                w += eta * effective_signal * traces[li]
                
                # Weight clipping
                w = np.clip(w, -5.0, 5.0)
                
                weights[w_idx:w_idx + w_size] = w.flatten()
                w_idx += w_size + policy.sizes[li * 2 + 1]
    
    env.close()
    return eval_reward / n_eval_episodes


def run(params=None, device="cpu", callback=None):
    """Pure Meta-Learning: CMA-ES evolves ONLY learning rules."""
    params = params or {}
    env_name = params.get("environment", "LunarLander-v3")
    max_evals = params.get("max_evals", 100000)
    n_eval_episodes = params.get("eval_episodes", 5)
    obs_dim = params.get("obs_dim", 8)
    act_dim = params.get("act_dim", 4)
    act_type = params.get("act_type", "discrete")
    hidden = params.get("hidden", [64, 32])
    max_steps = params.get("max_steps", 1000)
    solved_threshold = params.get("solved_threshold", 200)
    n_workers = params.get("n_workers", min(os.cpu_count() or 1, 16))
    n_lifetime_eps = params.get("n_lifetime_episodes", 10)
    n_random_inits = params.get("n_random_inits", 3)  # Test each rule on N random inits
    
    policy = PolicyNetwork(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden, act_type=act_type)
    n_layers = len(hidden) + 1
    rule = AdvancedHebbianRule(n_layers)
    
    print(f"🧬 PURE Meta-Learning on {env_name}")
    print(f"Network: {obs_dim}→{'→'.join(map(str, hidden))}→{act_dim} ({policy.n_params:,} weight params)")
    print(f"Learning rule genome: ONLY {rule.n_params} params ({n_layers} layers × {rule.params_per_layer} coefficients)")
    print(f"Weights: RANDOM init every evaluation (not evolved!)")
    print(f"Lifetime: {n_lifetime_eps} training eps → {n_eval_episodes} eval eps | {n_random_inits} random inits per candidate")
    print(f"Budget: {max_evals:,} evals | Workers: {n_workers}")
    
    cma = CMAES(rule.n_params, sigma0=1.0)  # Higher sigma for small genome
    print(f"CMA-ES: pop={cma.lam} | {rule.n_params} dimensions (tiny genome!)")
    
    best_ever = -float("inf")
    best_params = None
    total_evals = 0
    start_time = time.time()
    
    while total_evals < max_evals:
        candidates = cma.ask()
        
        # Evaluate each candidate on multiple random weight inits
        # This ensures the rule is GENERAL, not lucky with one init
        all_args = []
        candidate_map = []  # Track which args belong to which candidate
        for ci, c in enumerate(candidates):
            for ri in range(n_random_inits):
                seed = cma.gen * 10000 + ci * 100 + ri
                all_args.append((policy, c, rule, env_name, seed,
                                n_eval_episodes, max_steps, n_lifetime_eps))
                candidate_map.append(ci)
        
        if n_workers > 1:
            with mp.Pool(n_workers) as pool:
                all_scores = pool.starmap(evaluate_pure, all_args)
        else:
            all_scores = [evaluate_pure(*a) for a in all_args]
        
        # Average score per candidate across random inits
        fitnesses = np.zeros(len(candidates))
        for i, score in enumerate(all_scores):
            fitnesses[candidate_map[i]] += score / n_random_inits
        
        total_evals += len(all_args) * (n_lifetime_eps + n_eval_episodes)
        
        cma.tell(candidates, fitnesses)
        
        gen_best = float(np.max(fitnesses))
        if gen_best > best_ever:
            best_ever = gen_best
            best_params = candidates[np.argmax(fitnesses)].copy()
        
        elapsed = time.time() - start_time
        
        # Log evolved rules
        rule_str = ""
        if best_params is not None:
            best_rules = rule.decode(best_params)
            parts = []
            for i, (A, B, C, D, eta, mod, gate) in enumerate(best_rules):
                parts.append(f"L{i}:A={A:.3f},η={eta:.4f},mod={mod:.2f}")
            rule_str = " | ".join(parts)
        
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
        if cma.gen % 10 == 0 and rule_str:
            print(f"  Rules: {rule_str}")
        
        if best_ever >= solved_threshold:
            print(f"\n🎉 SOLVED with PURE Meta-Learning!")
            print(f"Learning rules alone (no evolved weights) can train a network from random!")
            break
    
    # Final eval: test best rule on MANY random inits
    if best_params is not None:
        final_scores = []
        for ri in range(20):
            seed = 99999 + ri
            s = evaluate_pure(policy, best_params, rule, env_name, seed,
                             n_eval_episodes, max_steps, n_lifetime_eps)
            final_scores.append(s)
        final_mean = float(np.mean(final_scores))
        final_std = float(np.std(final_scores))
        
        evolved_rules = rule.decode(best_params)
        print(f"\n📊 Evolved Learning Rules (genome = {rule.n_params} params):")
        for i, (A, B, C, D, eta, mod, gate) in enumerate(evolved_rules):
            print(f"  Layer {i}: A={A:.4f} B={B:.4f} C={C:.4f} D={D:.4f} η={eta:.5f} mod={mod:.3f} gate={gate:.3f}")
        print(f"\nFinal: {final_mean:.1f} ± {final_std:.1f} (over 20 random inits)")
    else:
        final_mean = final_std = 0.0
    
    return {
        "method": "Pure-Meta-Learning",
        "environment": env_name,
        "best_ever": float(best_ever),
        "final_mean": final_mean,
        "final_std": final_std,
        "total_evals": total_evals,
        "generations": cma.gen,
        "genome_size": rule.n_params,
        "network_params": policy.n_params,
        "n_lifetime_episodes": n_lifetime_eps,
        "n_random_inits": n_random_inits,
        "elapsed_seconds": round(time.time() - start_time, 1),
        "solved": best_ever >= solved_threshold,
    }
