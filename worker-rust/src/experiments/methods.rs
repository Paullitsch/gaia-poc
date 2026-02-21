//! All gradient-free methods in pure Rust.
//!
//! Each method: fn run(env_name, params, on_gen) -> RunResult
//! Validated against reference implementations:
//! - CMA-ES: Hansen's purecma.py (github.com/CMA-ES/pycma)
//! - OpenAI-ES: Salimans et al. 2017 (github.com/openai/evolution-strategies-starter)
//! - Neuromod: Our Python implementation (Hebbian + modulation network)
//! - Meta-Learning: Najarro & Risi 2020 (github.com/enajx/HebbianMetaLearning)

use super::env::{self};
use super::policy::Policy;
use super::optim::{CmaEs, Rng as OptRng, compute_centered_ranks};
use super::native_runner::{GenResult, RunResult};
use rayon::prelude::*;
use serde_json::Value;
use std::time::Instant;

// ─── Shared evaluate function ─────────────────────────────────────────

fn evaluate(env_name: &str, policy: &Policy, params: &[f32], n_episodes: usize, max_steps: usize) -> f64 {
    let mut total = 0.0;
    for ep in 0..n_episodes {
        let mut env = env::make(env_name, Some(ep as u64 * 1000))
            .unwrap_or_else(|| panic!("Unknown env: {}", env_name));
        let obs = env.reset(Some(ep as u64 * 1000));
        let mut obs = obs;
        let mut ep_reward = 0.0;
        for _ in 0..max_steps {
            let action = policy.forward(&obs, params);
            let result = env.step(&action);
            ep_reward += result.reward;
            if result.done() { break; }
            obs = result.observation;
        }
        total += ep_reward;
    }
    total / n_episodes as f64
}

// ─── GPU-aware batch evaluation ───────────────────────────────────────

/// Check if GPU full-episode evaluation is available for this environment.
#[cfg(feature = "cuda")]
fn try_gpu_evaluator(env_name: &str, policy: &Policy, n_episodes: usize) -> Option<super::gpu_eval::GpuEvaluator> {
    match env_name {
        "CartPole-v1" | "LunarLander-v3" | "Swimmer-v1" => {
            super::gpu_eval::GpuEvaluator::new(env_name, policy, n_episodes)
        }
        _ if env_name.starts_with("Pendulum-") && env_name.ends_with("Link") => {
            super::gpu_eval::GpuEvaluator::new(env_name, policy, n_episodes)
        }
        _ => None, // BipedalWalker: Box2D too complex for GPU
    }
}

/// Wrapper that creates and caches GPU evaluator for batch evaluation.
struct SmartEvaluator {
    #[cfg(feature = "cuda")]
    gpu: Option<super::gpu_eval::GpuEvaluator>,
    env_name: String,
    n_episodes: usize,
    max_steps: usize,
}

impl SmartEvaluator {
    fn new(env_name: &str, _policy: &Policy, n_episodes: usize, max_steps: usize) -> Self {
        #[cfg(feature = "cuda")]
        let gpu = try_gpu_evaluator(env_name, _policy, n_episodes);
        SmartEvaluator {
            #[cfg(feature = "cuda")]
            gpu,
            env_name: env_name.to_string(),
            n_episodes,
            max_steps,
        }
    }

    fn is_gpu(&self) -> bool {
        #[cfg(feature = "cuda")]
        { self.gpu.is_some() }
        #[cfg(not(feature = "cuda"))]
        { false }
    }

    fn evaluate_batch(&self, candidates: &[Vec<f64>], policy: &Policy) -> Vec<f64> {
        #[cfg(feature = "cuda")]
        {
            if let Some(ref eval) = self.gpu {
                let mut fits = eval.evaluate_batch(candidates);
                // NaN guard: replace NaN/Inf with large negative
                for f in &mut fits {
                    if f.is_nan() || f.is_infinite() { *f = -1e6; }
                }
                return fits;
            }
        }
        candidates.par_iter().map(|c| {
            let pf32: Vec<f32> = c.iter().map(|&v| v as f32).collect();
            let r = evaluate(&self.env_name, policy, &pf32, self.n_episodes, self.max_steps);
            if r.is_nan() || r.is_infinite() { -1e6 } else { r }
        }).collect()
    }
}

/// Evaluate with Hebbian neuromodulation (within-episode learning).
fn evaluate_neuromod(
    env_name: &str, policy: &Policy, genome: &[f64],
    mod_net: &ModNetwork, n_episodes: usize, max_steps: usize,
) -> f64 {
    let n_main = policy.n_params;
    let n_mod = mod_net.n_params;

    // Decode genome
    let main_weights: Vec<f32> = genome[..n_main].iter().map(|&v| v as f32).collect();
    let mod_weights: Vec<f32> = genome[n_main..n_main + n_mod].iter().map(|&v| v as f32).collect();
    let plast_raw = &genome[n_main + n_mod..];
    let n_layers = policy.layer_dims.len();
    let plasticity: Vec<f64> = (0..n_layers).map(|i| {
        if i < plast_raw.len() { plast_raw[i].abs() * 0.01 } else { 0.001 }
    }).collect();

    let mut working_w = main_weights.clone();
    let mut total_reward = 0.0;

    for ep in 0..n_episodes {
        let mut env = env::make(env_name, Some(ep as u64 * 1000))
            .unwrap_or_else(|| panic!("Unknown env: {}", env_name));
        let mut obs = env.reset(Some(ep as u64 * 1000));
        let mut ep_reward = 0.0;

        // Hebbian traces (per weight)
        let mut traces = vec![0.0f32; n_main];

        for _ in 0..max_steps {
            let (_output, pre_acts, post_acts) = policy.forward_with_activations(&obs, &working_w);

            // Action selection
            let action = policy.forward(&obs, &working_w);

            // Modulation signal
            let mod_signal = mod_net.forward(&obs, &mod_weights);

            let result = env.step(&action);
            let reward = result.reward;
            ep_reward += reward;

            // Hebbian trace update: trace += pre * post * mod_signal * reward * 0.001
            let mut offset = 0;
            for layer_i in 0..n_layers.min(pre_acts.len()) {
                let (fan_in, fan_out) = policy.layer_dims[layer_i];
                let w_size = fan_in * fan_out;
                let pre = &pre_acts[layer_i];
                let post = &post_acts[layer_i];

                for i in 0..fan_in.min(pre.len()) {
                    for j in 0..fan_out.min(post.len()) {
                        traces[offset + i * fan_out + j] +=
                            pre[i] * post[j] * mod_signal * reward as f32 * 0.001;
                    }
                }
                offset += w_size + fan_out; // skip biases
            }

            if result.done() { break; }
            obs = result.observation;
        }

        // Apply plasticity at end of episode (except last)
        if ep < n_episodes - 1 {
            let mut offset = 0;
            for layer_i in 0..n_layers {
                let (fan_in, fan_out) = policy.layer_dims[layer_i];
                let w_size = fan_in * fan_out;
                let lr = plasticity[layer_i] as f32;
                for k in 0..w_size {
                    working_w[offset + k] += lr * traces[offset + k];
                    working_w[offset + k] = working_w[offset + k].clamp(-5.0, 5.0);
                }
                offset += w_size + fan_out;
            }
        }

        total_reward += ep_reward;
    }
    total_reward / n_episodes as f64
}

/// Evaluate with Hebbian meta-learning (ABCD+η learning rules).
fn evaluate_with_learning(
    env_name: &str, policy: &Policy, genome: &[f64],
    n_episodes: usize, max_steps: usize,
) -> f64 {
    let n_main = policy.n_params;
    let n_layers = policy.layer_dims.len();
    let rule_params_per_layer = 5; // A, B, C, D, eta

    // Decode: weights + per-layer learning rules
    let mut working_w: Vec<f32> = genome[..n_main].iter().map(|&v| v as f32).collect();
    let rule_genome = &genome[n_main..];

    // Decode rules per layer
    let rules: Vec<(f64, f64, f64, f64, f64)> = (0..n_layers).map(|i| {
        let off = i * rule_params_per_layer;
        if off + 4 < rule_genome.len() {
            let a = rule_genome[off].tanh() * 0.1;       // Hebbian coeff
            let b = rule_genome[off+1].tanh() * 0.05;    // Presynaptic
            let c = rule_genome[off+2].tanh() * 0.05;    // Postsynaptic
            let d = -rule_genome[off+3].abs() * 0.01;    // Decay (always negative)
            let eta = rule_genome[off+4].abs() * 0.01;   // Learning rate
            (a, b, c, d, eta)
        } else {
            (0.0, 0.0, 0.0, 0.0, 0.0)
        }
    }).collect();

    let mut total_reward = 0.0;

    for ep in 0..n_episodes {
        let mut env = env::make(env_name, Some(ep as u64 * 1000))
            .unwrap_or_else(|| panic!("Unknown env: {}", env_name));
        let mut obs = env.reset(Some(ep as u64 * 1000));
        let mut ep_reward = 0.0;

        // Store activations for Hebbian update
        let mut all_pre_acts: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut all_post_acts: Vec<Vec<Vec<f32>>> = Vec::new();

        for _ in 0..max_steps {
            let (_output, pre_acts, post_acts) = policy.forward_with_activations(&obs, &working_w);
            let action = policy.forward(&obs, &working_w);

            let result = env.step(&action);
            ep_reward += result.reward;

            all_pre_acts.push(pre_acts);
            all_post_acts.push(post_acts);

            if result.done() { break; }
            obs = result.observation;
        }

        total_reward += ep_reward;

        // Apply Hebbian learning after each episode (except last)
        if !all_pre_acts.is_empty() && ep < n_episodes - 1 {
            let reward_signal = (ep_reward / 100.0).max(0.0);

            // Sample timesteps (up to 50)
            let n_samples = all_pre_acts.len().min(50);
            let step = if all_pre_acts.len() <= n_samples { 1 } else { all_pre_acts.len() / n_samples };

            let mut offset = 0;
            for layer_i in 0..n_layers {
                let (fan_in, fan_out) = policy.layer_dims[layer_i];
                let w_size = fan_in * fan_out;
                let (a, b, c, d, eta) = rules[layer_i];

                if eta.abs() < 1e-12 { offset += w_size + fan_out; continue; }

                // Average Hebbian update across sampled timesteps
                let mut dw = vec![0.0f64; w_size];
                let mut count = 0;
                for si in (0..all_pre_acts.len()).step_by(step.max(1)).take(n_samples) {
                    if layer_i >= all_pre_acts[si].len() { break; }
                    let pre = &all_pre_acts[si][layer_i];
                    let post = &all_post_acts[si][layer_i];

                    for i in 0..fan_in.min(pre.len()) {
                        for j in 0..fan_out.min(post.len()) {
                            // Generalized Hebbian: Δw = A*pre*post + B*pre + C*post + D
                            dw[i * fan_out + j] +=
                                a * pre[i] as f64 * post[j] as f64
                                + b * pre[i] as f64
                                + c * post[j] as f64
                                + d;
                        }
                    }
                    count += 1;
                }

                if count > 0 {
                    for k in 0..w_size {
                        dw[k] /= count as f64;
                        working_w[offset + k] += (eta * reward_signal * dw[k]) as f32;
                        working_w[offset + k] = working_w[offset + k].clamp(-5.0, 5.0);
                    }
                }

                offset += w_size + fan_out;
            }
        }
    }

    total_reward / n_episodes as f64
}

// ─── Modulation Network (for Neuromod) ────────────────────────────────

/// Small feed-forward network: obs → hidden(16) → 1 scalar (tanh).
struct ModNetwork {
    obs_dim: usize,
    hidden: usize,
    n_params: usize,
}

impl ModNetwork {
    fn new(obs_dim: usize) -> Self {
        let hidden = 16;
        let n_params = obs_dim * hidden + hidden + hidden * 1 + 1;
        ModNetwork { obs_dim, hidden, n_params }
    }

    fn forward(&self, obs: &[f32], params: &[f32]) -> f32 {
        let mut offset = 0;

        // Layer 1: obs → hidden (tanh)
        let w1_size = self.obs_dim * self.hidden;
        let mut h = vec![0.0f32; self.hidden];
        for j in 0..self.hidden {
            let mut sum = params[offset + w1_size + j]; // bias
            for i in 0..self.obs_dim.min(obs.len()) {
                sum += obs[i] * params[offset + i * self.hidden + j];
            }
            h[j] = sum.tanh();
        }
        offset += w1_size + self.hidden;

        // Layer 2: hidden → 1 (tanh)
        let mut out = params[offset + self.hidden]; // bias
        for j in 0..self.hidden {
            out += h[j] * params[offset + j];
        }
        out.tanh()
    }
}

// ─── Parameter extraction helpers ─────────────────────────────────────

fn extract_params(env_name: &str, params: &Value) -> (usize, usize, f64, Vec<usize>) {
    let max_evals = params.get("max_evals").and_then(|v| v.as_u64()).unwrap_or(100000) as usize;
    let eval_episodes = params.get("eval_episodes").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
    let sigma0 = params.get("sigma0").and_then(|v| v.as_f64()).unwrap_or(0.5);
    let hidden = if let Some(h) = params.get("hidden").and_then(|v| v.as_array()) {
        h.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect()
    } else if let Some(cfg) = params.get("config").and_then(|v| v.as_str()) {
        match cfg {
            "1k" => vec![32, 16], "3k" => vec![64, 32], "10k" => vec![128, 64],
            "33k" => vec![256, 128], "100k" => vec![512, 256],
            "200k" => vec![512, 256, 128], "500k" => vec![1024, 512, 256],
            _ => env::default_hidden(env_name),
        }
    } else {
        env::default_hidden(env_name)
    };
    (max_evals, eval_episodes, sigma0, hidden)
}

/// Optimal hidden sizes for CMA-ES methods (keep params < 5K for full covariance).
fn cma_optimal_hidden(env_name: &str) -> Vec<usize> {
    match env_name {
        "CartPole-v1" => vec![32, 16],
        "LunarLander-v3" => vec![64, 32],
        "BipedalWalker-v3" => vec![64, 32],
        _ => vec![64, 32],
    }
}

fn extract_cma_params(env_name: &str, params: &Value) -> (usize, usize, f64, Vec<usize>) {
    let max_evals = params.get("max_evals").and_then(|v| v.as_u64()).unwrap_or(100000) as usize;
    let eval_episodes = params.get("eval_episodes").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
    let sigma0 = params.get("sigma0").and_then(|v| v.as_f64()).unwrap_or(0.5);
    let hidden = if let Some(h) = params.get("hidden").and_then(|v| v.as_array()) {
        h.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect()
    } else if params.get("config").is_some() {
        return extract_params(env_name, params);
    } else {
        cma_optimal_hidden(env_name)
    };
    (max_evals, eval_episodes, sigma0, hidden)
}

fn final_eval(env_name: &str, policy: &Policy, best_params: &Option<Vec<f64>>, max_steps: usize) -> (f64, f64) {
    if let Some(ref bp) = best_params {
        let pf32: Vec<f32> = bp.iter().map(|&v| v as f32).collect();
        let scores: Vec<f64> = (0..20).map(|_| evaluate(env_name, policy, &pf32, 1, max_steps)).collect();
        let m = scores.iter().sum::<f64>() / 20.0;
        let s = (scores.iter().map(|x| (x - m).powi(2)).sum::<f64>() / 20.0).sqrt();
        (m, s)
    } else { (0.0, 0.0) }
}

// ─── CMA-ES ──────────────────────────────────────────────────────────

pub fn run_cma_es(env_name: &str, params: &Value, mut on_gen: impl FnMut(GenResult)) -> RunResult {
    let env_cfg = env::get_env_config(env_name).expect("Unknown env");
    let (max_evals, eval_episodes, sigma0, hidden) = extract_cma_params(env_name, params);
    let policy = Policy::new(env_cfg.obs_dim, env_cfg.action_space.size(), &hidden, env_cfg.action_space);
    let mut cma = CmaEs::new(policy.n_params, sigma0, None);
    let max_steps = env_cfg.max_steps;
    let solved = env_cfg.solved_threshold;

    let patience = params.get("patience").and_then(|v| v.as_u64()).unwrap_or(200) as usize;
    let evaluator = SmartEvaluator::new(env_name, &policy, eval_episodes, max_steps);

    eprintln!("🦀{} CMA-ES on {} | {} params | pop={} | patience={} | full_cov={} | GPU={}",
        if evaluator.is_gpu() { "🚀" } else { "" },
        env_name, policy.n_params, cma.pop_size, patience, !cma.use_diagonal, evaluator.is_gpu());

    let mut best_ever = f64::NEG_INFINITY;
    let mut best_params: Option<Vec<f64>> = None;
    let mut total_evals = 0usize;
    let mut stale_gens = 0usize;
    let mut restarts = 0usize;
    let start = Instant::now();

    while total_evals < max_evals {
        let candidates = cma.ask();
        let fitnesses = evaluator.evaluate_batch(&candidates, &policy);

        total_evals += candidates.len() * eval_episodes;
        cma.tell(&candidates, &fitnesses);

        let gen_best = fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if gen_best > best_ever {
            best_ever = gen_best;
            stale_gens = 0;
            let idx = fitnesses.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            best_params = Some(candidates[idx].clone());
        } else {
            stale_gens += 1;
        }

        on_gen(GenResult {
            generation: cma.gen, best: gen_best, best_ever, mean: fitnesses.iter().sum::<f64>() / fitnesses.len() as f64,
            sigma: cma.sigma, evals: total_evals, time: start.elapsed().as_secs_f64(),
        });

        if best_ever >= solved { break; }

        if stale_gens >= patience || cma.sigma < 1e-8 {
            restarts += 1;
            eprintln!("  🔄 Restart #{} at gen {} (stale={}, σ={:.2e})", restarts, cma.gen, stale_gens, cma.sigma);
            let new_sigma = sigma0 * (1.0 + 0.2 * restarts as f64);
            cma = CmaEs::new(policy.n_params, new_sigma, None);
            if let Some(ref bp) = best_params {
                for (i, v) in bp.iter().enumerate() { cma.mean[i] = *v; }
            }
            stale_gens = 0;
        }
    }

    let (fm, fs) = final_eval(env_name, &policy, &best_params, max_steps);
    RunResult { method: "CMA-ES".into(), environment: env_name.into(), best_ever, final_mean: fm, final_std: fs,
        total_evals, generations: cma.gen, elapsed: start.elapsed().as_secs_f64(), solved: best_ever >= solved }
}

// ─── OpenAI-ES (with rank-based fitness shaping) ─────────────────────

pub fn run_openai_es(env_name: &str, params: &Value, mut on_gen: impl FnMut(GenResult)) -> RunResult {
    let env_cfg = env::get_env_config(env_name).expect("Unknown env");
    let (max_evals, eval_episodes, _, hidden) = extract_params(env_name, params);
    let policy = Policy::new(env_cfg.obs_dim, env_cfg.action_space.size(), &hidden, env_cfg.action_space);
    let n = policy.n_params;
    let max_steps = env_cfg.max_steps;
    let solved = env_cfg.solved_threshold;

    // Defaults: OpenAI paper used noise=0.02 with 720 workers and 1M+ params.
    // For our smaller networks (722-2788 params) and fewer workers (50),
    // noise_std must be larger to produce behavioral diversity.
    // Scale noise inversely with sqrt(pop_size) to maintain gradient quality.
    let pop_size = params.get("pop_size").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
    let lr = params.get("lr").and_then(|v| v.as_f64()).unwrap_or(0.02);
    let noise_std = params.get("noise_std").and_then(|v| v.as_f64()).unwrap_or(0.1);
    let weight_decay = params.get("weight_decay").and_then(|v| v.as_f64()).unwrap_or(0.0);

    let evaluator = SmartEvaluator::new(env_name, &policy, eval_episodes, max_steps);

    eprintln!("🦀{} OpenAI-ES on {} | {} params | pop={} | lr={} | σ={} | wd={} | GPU={}",
        if evaluator.is_gpu() { "🚀" } else { "" },
        env_name, n, pop_size, lr, noise_std, weight_decay, evaluator.is_gpu());

    // Initialize theta with small random values (matching Python: randn * 0.1)
    let mut rng = OptRng::new(42);
    let mut theta: Vec<f64> = (0..n).map(|_| rng.randn() * 0.1).collect();

    let mut best_ever = f64::NEG_INFINITY;
    let mut best_params: Option<Vec<f64>> = None;
    let mut total_evals = 0usize;
    let mut gen = 0usize;
    let start = Instant::now();

    while total_evals < max_evals {
        gen += 1;

        // Generate perturbations
        let epsilons: Vec<Vec<f64>> = (0..pop_size).map(|_| rng.randn_vec(n)).collect();

        // Build ALL candidates (positive + negative perturbations)
        let mut all_candidates: Vec<Vec<f64>> = Vec::with_capacity(pop_size * 2);
        for eps in &epsilons {
            let plus: Vec<f64> = (0..n).map(|i| theta[i] + noise_std * eps[i]).collect();
            let minus: Vec<f64> = (0..n).map(|i| theta[i] - noise_std * eps[i]).collect();
            all_candidates.push(plus);
            all_candidates.push(minus);
        }

        // Evaluate ALL at once (GPU batch or rayon)
        let all_rewards = evaluator.evaluate_batch(&all_candidates, &policy);

        total_evals += pop_size * 2 * eval_episodes;

        let ranks = compute_centered_ranks(&all_rewards);
        // ranks[2*i] = rank of plus perturbation i
        // ranks[2*i+1] = rank of minus perturbation i

        // Compute gradient using ranks
        let mut grad = vec![0.0f64; n];
        for (i, eps) in epsilons.iter().enumerate() {
            let rank_diff = ranks[2 * i] - ranks[2 * i + 1];
            for j in 0..n {
                grad[j] += rank_diff * eps[j];
            }
        }

        // Update theta: gradient ascent with optional weight decay
        // FIX: normalize by pop_size (not 2*pop_size) — matches OpenAI reference es.py
        for j in 0..n {
            theta[j] = theta[j] * (1.0 - weight_decay)
                + lr / (pop_size as f64 * noise_std) * grad[j];
        }

        // Track best
        let gen_best = all_rewards.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let gen_mean = all_rewards.iter().sum::<f64>() / all_rewards.len() as f64;
        if gen_best > best_ever {
            best_ever = gen_best;
            best_params = Some(theta.clone());
        }

        on_gen(GenResult {
            generation: gen, best: gen_best, best_ever, mean: gen_mean,
            sigma: noise_std, evals: total_evals, time: start.elapsed().as_secs_f64(),
        });

        if best_ever >= solved { break; }
    }

    let (fm, fs) = final_eval(env_name, &policy, &best_params, max_steps);
    RunResult { method: "OpenAI-ES".into(), environment: env_name.into(), best_ever, final_mean: fm, final_std: fs,
        total_evals, generations: gen, elapsed: start.elapsed().as_secs_f64(), solved: best_ever >= solved }
}

// ─── Curriculum CMA-ES ───────────────────────────────────────────────

pub fn run_curriculum(env_name: &str, params: &Value, mut on_gen: impl FnMut(GenResult)) -> RunResult {
    let env_cfg = env::get_env_config(env_name).expect("Unknown env");
    let (max_evals, eval_episodes, sigma0, hidden) = extract_cma_params(env_name, params);
    let policy = Policy::new(env_cfg.obs_dim, env_cfg.action_space.size(), &hidden, env_cfg.action_space);
    let mut cma = CmaEs::new(policy.n_params, sigma0, None);
    let max_steps = env_cfg.max_steps;
    let solved = env_cfg.solved_threshold;

    // Note: Curriculum uses varying max_steps, so GPU evaluator uses fixed max_steps
    // and curriculum adjusts episode length. GPU eval runs full episodes anyway,
    // so we use CPU for curriculum to get proper step limiting.
    // TODO: Add max_steps parameter to GpuEvaluator for curriculum support.

    eprintln!("🦀 Curriculum CMA-ES on {} | {} params | full_cov={}", env_name, policy.n_params, !cma.use_diagonal);

    let max_gens = max_evals / (cma.pop_size * eval_episodes);
    let mut best_ever = f64::NEG_INFINITY;
    let mut best_params: Option<Vec<f64>> = None;
    let mut total_evals = 0usize;
    let start = Instant::now();

    while total_evals < max_evals {
        let progress = (cma.gen as f64 / (max_gens as f64 * 0.7).max(1.0)).min(1.0);
        let difficulty = 0.3 + 0.7 * progress;
        let curr_max_steps = (max_steps as f64 * difficulty) as usize;

        let candidates = cma.ask();
        let fitnesses: Vec<f64> = candidates.par_iter()
            .map(|c| {
                let pf32: Vec<f32> = c.iter().map(|&v| v as f32).collect();
                evaluate(env_name, &policy, &pf32, eval_episodes, curr_max_steps)
            }).collect();

        total_evals += candidates.len() * eval_episodes;
        cma.tell(&candidates, &fitnesses);

        let gen_best = fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if gen_best > best_ever {
            best_ever = gen_best;
            let idx = fitnesses.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            best_params = Some(candidates[idx].clone());
        }

        // Raw eval on full steps periodically
        if cma.gen % 10 == 0 {
            if let Some(ref bp) = best_params {
                let pf32: Vec<f32> = bp.iter().map(|&v| v as f32).collect();
                let raw = evaluate(env_name, &policy, &pf32, eval_episodes, max_steps);
                if raw > best_ever { best_ever = raw; }
            }
        }

        on_gen(GenResult {
            generation: cma.gen, best: gen_best, best_ever,
            mean: fitnesses.iter().sum::<f64>() / fitnesses.len() as f64,
            sigma: cma.sigma, evals: total_evals, time: start.elapsed().as_secs_f64(),
        });

        if best_ever >= solved { break; }
    }

    let (fm, fs) = final_eval(env_name, &policy, &best_params, max_steps);
    RunResult { method: "Curriculum".into(), environment: env_name.into(), best_ever, final_mean: fm, final_std: fs,
        total_evals, generations: cma.gen, elapsed: start.elapsed().as_secs_f64(), solved: best_ever >= solved }
}

// ─── Neuromod CMA-ES (with actual Hebbian modulation) ────────────────

pub fn run_neuromod(env_name: &str, params: &Value, mut on_gen: impl FnMut(GenResult)) -> RunResult {
    let env_cfg = env::get_env_config(env_name).expect("Unknown env");
    let (max_evals, eval_episodes, sigma0, hidden) = extract_cma_params(env_name, params);
    let policy = Policy::new(env_cfg.obs_dim, env_cfg.action_space.size(), &hidden, env_cfg.action_space);
    let n_layers = hidden.len() + 1;

    // Modulation network: obs → hidden(16) → 1
    let mod_net = ModNetwork::new(env_cfg.obs_dim);

    // Genome: main_weights + mod_network + plasticity_per_layer
    let total_params = policy.n_params + mod_net.n_params + n_layers;

    let mut cma = CmaEs::new(total_params, sigma0, None);
    let max_steps = env_cfg.max_steps;
    let solved = env_cfg.solved_threshold;

    // TODO: Neuromod uses within-episode Hebbian plasticity. Can't batch on GPU.
    eprintln!("🦀 Neuromod CMA-ES on {} | {} weight + {} mod + {} plast = {} total | full_cov={} | GPU=false (Hebbian)",
        env_name, policy.n_params, mod_net.n_params, n_layers, total_params, !cma.use_diagonal);

    let mut best_ever = f64::NEG_INFINITY;
    let mut best_params_raw: Option<Vec<f64>> = None;
    let mut total_evals = 0usize;
    let start = Instant::now();

    while total_evals < max_evals {
        let candidates = cma.ask();
        let fitnesses: Vec<f64> = candidates.par_iter()
            .map(|c| {
                evaluate_neuromod(env_name, &policy, c, &mod_net, eval_episodes, max_steps)
            }).collect();

        total_evals += candidates.len() * eval_episodes;
        cma.tell(&candidates, &fitnesses);

        let gen_best = fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if gen_best > best_ever {
            best_ever = gen_best;
            let idx = fitnesses.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            best_params_raw = Some(candidates[idx].clone());
        }

        on_gen(GenResult {
            generation: cma.gen, best: gen_best, best_ever,
            mean: fitnesses.iter().sum::<f64>() / fitnesses.len() as f64,
            sigma: cma.sigma, evals: total_evals, time: start.elapsed().as_secs_f64(),
        });

        if best_ever >= solved { break; }
    }

    // Final eval uses the neuromod evaluation (with learning)
    let (fm, fs) = if let Some(ref bp) = best_params_raw {
        let scores: Vec<f64> = (0..20).map(|_| {
            evaluate_neuromod(env_name, &policy, bp, &mod_net, 1, max_steps)
        }).collect();
        let m = scores.iter().sum::<f64>() / 20.0;
        let s = (scores.iter().map(|x| (x - m).powi(2)).sum::<f64>() / 20.0).sqrt();
        (m, s)
    } else { (0.0, 0.0) };

    RunResult { method: "Neuromod".into(), environment: env_name.into(), best_ever, final_mean: fm, final_std: fs,
        total_evals, generations: cma.gen, elapsed: start.elapsed().as_secs_f64(), solved: best_ever >= solved }
}

// ─── Island Model ────────────────────────────────────────────────────

pub fn run_island_model(env_name: &str, params: &Value, mut on_gen: impl FnMut(GenResult)) -> RunResult {
    let env_cfg = env::get_env_config(env_name).expect("Unknown env");
    let (max_evals, eval_episodes, sigma0, hidden) = extract_cma_params(env_name, params);
    let policy = Policy::new(env_cfg.obs_dim, env_cfg.action_space.size(), &hidden, env_cfg.action_space);
    let max_steps = env_cfg.max_steps;
    let solved = env_cfg.solved_threshold;

    let n_islands = params.get("n_islands").and_then(|v| v.as_u64()).unwrap_or(4) as usize;
    let migration_interval = params.get("migration_interval").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

    let evaluator = SmartEvaluator::new(env_name, &policy, eval_episodes, max_steps);

    eprintln!("🦀{} Island Model on {} | {} islands | migrate every {} gens | GPU={}",
        if evaluator.is_gpu() { "🚀" } else { "" },
        env_name, n_islands, migration_interval, evaluator.is_gpu());

    let mut islands: Vec<CmaEs> = (0..n_islands)
        .map(|i| {
            let mut c = CmaEs::new(policy.n_params, sigma0 * (1.0 + 0.2 * i as f64), None);
            let mut rng = OptRng::new(i as u64 * 12345);
            for j in 0..policy.n_params {
                c.mean[j] = rng.randn() * 0.1;
            }
            c
        })
        .collect();

    let mut global_best_ever = f64::NEG_INFINITY;
    let mut global_best_params: Option<Vec<f64>> = None;
    let mut total_evals = 0usize;
    let mut gen = 0usize;
    let start = Instant::now();

    while total_evals < max_evals {
        gen += 1;
        let mut island_results: Vec<(f64, Vec<f64>, f64)> = Vec::new();

        // Collect ALL candidates from ALL islands for one big batch eval
        let mut all_candidates: Vec<Vec<f64>> = Vec::new();
        let mut island_pop_sizes: Vec<usize> = Vec::new();
        let mut island_candidates: Vec<Vec<Vec<f64>>> = Vec::new();

        for island in &mut islands {
            let candidates = island.ask();
            island_pop_sizes.push(candidates.len());
            all_candidates.extend(candidates.iter().cloned());
            island_candidates.push(candidates);
        }

        // Single batch eval for ALL islands
        let all_fitnesses = evaluator.evaluate_batch(&all_candidates, &policy);
        total_evals += all_candidates.len() * eval_episodes;

        // Split results back to islands
        let mut offset = 0;
        for (island_idx, island) in islands.iter_mut().enumerate() {
            let pop = island_pop_sizes[island_idx];
            let fitnesses = &all_fitnesses[offset..offset + pop];
            let candidates = &island_candidates[island_idx];

            island.tell(candidates, fitnesses);

            let gen_best = fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let idx = fitnesses.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            let gen_mean = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;
            island_results.push((gen_best, candidates[idx].clone(), gen_mean));

            offset += pop;
        }

        for (score, params_vec, _) in &island_results {
            if *score > global_best_ever {
                global_best_ever = *score;
                global_best_params = Some(params_vec.clone());
            }
        }

        if gen % migration_interval == 0 && n_islands > 1 {
            let mut scores: Vec<(usize, f64)> = island_results.iter().enumerate()
                .map(|(i, (s, _, _))| (i, *s)).collect();
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let best_island = scores[0].0;
            let worst_island = scores[scores.len() - 1].0;
            let best_mean = islands[best_island].mean.clone();
            for j in 0..policy.n_params {
                islands[worst_island].mean[j] = 0.5 * islands[worst_island].mean[j] + 0.5 * best_mean[j];
            }
        }

        let overall_mean = island_results.iter().map(|(_, _, m)| m).sum::<f64>() / n_islands as f64;
        on_gen(GenResult {
            generation: gen, best: island_results.iter().map(|(s, _, _)| *s).fold(f64::NEG_INFINITY, f64::max),
            best_ever: global_best_ever, mean: overall_mean,
            sigma: islands[0].sigma, evals: total_evals, time: start.elapsed().as_secs_f64(),
        });

        if global_best_ever >= solved { break; }
    }

    let (fm, fs) = final_eval(env_name, &policy, &global_best_params, max_steps);
    RunResult { method: "Island Model".into(), environment: env_name.into(), best_ever: global_best_ever,
        final_mean: fm, final_std: fs, total_evals, generations: gen,
        elapsed: start.elapsed().as_secs_f64(), solved: global_best_ever >= solved }
}

// ─── Meta-Learning (evolve weights + Hebbian learning rules) ─────────

pub fn run_meta_learning(env_name: &str, params: &Value, mut on_gen: impl FnMut(GenResult)) -> RunResult {
    let env_cfg = env::get_env_config(env_name).expect("Unknown env");
    let (max_evals, eval_episodes, sigma0, hidden) = extract_cma_params(env_name, params);
    let policy = Policy::new(env_cfg.obs_dim, env_cfg.action_space.size(), &hidden, env_cfg.action_space);
    let n_layers = hidden.len() + 1;
    let rule_params = n_layers * 5; // A, B, C, D, eta per layer
    let total_params = policy.n_params + rule_params;

    let mut cma = CmaEs::new(total_params, sigma0, None);
    let max_steps = env_cfg.max_steps;
    let solved = env_cfg.solved_threshold;

    // TODO: Meta-Learning uses within-episode weight modification (Hebbian learning),
    // which requires per-step weight updates. This can't be batched on GPU.
    // Future: consider GPU kernel that includes Hebbian updates in the episode loop.

    eprintln!("🦀 Meta-Learning on {} | {} weights + {} rule = {} genome | full_cov={} | GPU=false (Hebbian)",
        env_name, policy.n_params, rule_params, total_params, !cma.use_diagonal);

    let mut best_ever = f64::NEG_INFINITY;
    let mut best_params: Option<Vec<f64>> = None;
    let mut total_evals = 0usize;
    let start = Instant::now();

    while total_evals < max_evals {
        let candidates = cma.ask();
        let fitnesses: Vec<f64> = candidates.par_iter()
            .map(|c| {
                evaluate_with_learning(env_name, &policy, c, eval_episodes, max_steps)
            }).collect();

        total_evals += candidates.len() * eval_episodes;
        cma.tell(&candidates, &fitnesses);

        let gen_best = fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if gen_best > best_ever {
            best_ever = gen_best;
            let idx = fitnesses.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            best_params = Some(candidates[idx].clone());
        }

        on_gen(GenResult {
            generation: cma.gen, best: gen_best, best_ever,
            mean: fitnesses.iter().sum::<f64>() / fitnesses.len() as f64,
            sigma: cma.sigma, evals: total_evals, time: start.elapsed().as_secs_f64(),
        });

        if best_ever >= solved { break; }
    }

    // Final eval with learning
    let (fm, fs) = if let Some(ref bp) = best_params {
        let scores: Vec<f64> = (0..20).map(|_| {
            evaluate_with_learning(env_name, &policy, bp, 1, max_steps)
        }).collect();
        let m = scores.iter().sum::<f64>() / 20.0;
        let s = (scores.iter().map(|x| (x - m).powi(2)).sum::<f64>() / 20.0).sqrt();
        (m, s)
    } else { (0.0, 0.0) };

    RunResult { method: "Meta-Learning".into(), environment: env_name.into(), best_ever, final_mean: fm, final_std: fs,
        total_evals, generations: cma.gen, elapsed: start.elapsed().as_secs_f64(), solved: best_ever >= solved }
}
