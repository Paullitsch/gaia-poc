//! GPU-accelerated experiment methods.
//!
//! Uses GpuVecEnv to evaluate many candidates in parallel:
//! - Each candidate gets N episodes running simultaneously
//! - Forward pass batched across all environments
//! - Massive speedup over sequential evaluation
//!
//! Architecture for CMA-ES with pop_size=28, eval_episodes=5:
//!   → 28 × 5 = 140 environments running in parallel
//!   → Each step: 140 forward passes + 140 env.step() in one batch
//!   → On GPU: single kernel launch for all 140

use super::env::{self, ActionSpace};
use super::gpu_env::{GpuVecEnv, batch_forward, batch_argmax};
use super::optim::{CmaEs, Rng as OptRng, compute_centered_ranks};
use super::policy::Policy;
use super::native_runner::{GenResult, RunResult};
use serde_json::Value;
use std::time::Instant;

// ─── GPU-accelerated Vectorized Evaluate ────────────────────────────

/// Evaluate multiple candidates in parallel using vectorized environments.
/// Each candidate gets `n_episodes` environments → total = candidates.len() × n_episodes.
/// All environments step simultaneously.
fn batch_evaluate(
    env_name: &str,
    candidates: &[Vec<f64>],
    policy: &Policy,
    n_episodes: usize,
    max_steps: usize,
) -> Vec<f64> {
    let n_candidates = candidates.len();
    let total_envs = n_candidates * n_episodes;
    let obs_dim = policy.config.obs_dim;
    let act_dim = policy.config.act_dim;
    let discrete = matches!(policy.config.action_space, ActionSpace::Discrete(_));

    // Create vectorized environment for ALL evaluations at once
    let mut env = GpuVecEnv::new(env_name, total_envs);
    env.reset_all(42);

    // Track per-env rewards and done status
    let mut total_rewards = vec![0.0f64; total_envs];
    let mut active = vec![true; total_envs];

    for _step in 0..max_steps {
        // Check if all done
        if active.iter().all(|&a| !a) { break; }

        // Build observations for batch forward pass
        // We need to run forward pass per candidate (shared weights) across its episodes
        let obs = env.get_observations();

        // For each candidate, compute actions for its episodes
        let mut all_actions = vec![0.0f32; if discrete { total_envs } else { total_envs * act_dim }];

        for c_idx in 0..n_candidates {
            let params: Vec<f32> = candidates[c_idx].iter().map(|&v| v as f32).collect();
            let env_start = c_idx * n_episodes;
            let env_end = env_start + n_episodes;

            // Get observations for this candidate's episodes
            let n_active: usize = (env_start..env_end).filter(|&i| active[i]).count();
            if n_active == 0 { continue; }

            // Batch forward pass for this candidate's episodes
            let obs_slice_start = env_start * obs_dim;
            let obs_slice_end = env_end * obs_dim;
            let logits = batch_forward(
                &obs[obs_slice_start..obs_slice_end],
                &params,
                &policy.layer_dims,
                n_episodes,
                obs_dim,
                act_dim,
                discrete,
            );

            if discrete {
                let actions = batch_argmax(&logits, n_episodes, act_dim);
                for ep in 0..n_episodes {
                    all_actions[env_start + ep] = actions[ep];
                }
            } else {
                for ep in 0..n_episodes {
                    for a in 0..act_dim {
                        all_actions[(env_start + ep) * act_dim + a] = logits[ep * act_dim + a];
                    }
                }
            }
        }

        // Step ALL environments at once
        let (_, rewards, dones) = env.step_all(&all_actions);

        // Accumulate rewards
        for i in 0..total_envs {
            if active[i] {
                total_rewards[i] += rewards[i] as f64;
                if dones[i] != 0 {
                    active[i] = false;
                }
            }
        }
    }

    // Aggregate per-candidate scores (mean over episodes)
    let mut fitnesses = vec![0.0f64; n_candidates];
    for c_idx in 0..n_candidates {
        let env_start = c_idx * n_episodes;
        let sum: f64 = (0..n_episodes).map(|ep| total_rewards[env_start + ep]).sum();
        fitnesses[c_idx] = sum / n_episodes as f64;
    }
    fitnesses
}

// ─── GPU CMA-ES ─────────────────────────────────────────────────────

pub fn run_gpu_cma_es(env_name: &str, params: &Value, mut on_gen: impl FnMut(GenResult)) -> RunResult {
    let env_cfg = env::get_env_config(env_name).expect("Unknown env");

    let hidden = if let Some(h) = params.get("hidden").and_then(|v| v.as_array()) {
        h.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect()
    } else {
        match env_name {
            "CartPole-v1" => vec![32, 16],
            _ => vec![64, 32],
        }
    };
    let max_evals = params.get("max_evals").and_then(|v| v.as_u64()).unwrap_or(100000) as usize;
    let eval_episodes = params.get("eval_episodes").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
    let sigma0 = params.get("sigma0").and_then(|v| v.as_f64()).unwrap_or(0.5);
    let patience = params.get("patience").and_then(|v| v.as_u64()).unwrap_or(200) as usize;

    let policy = Policy::new(env_cfg.obs_dim, env_cfg.action_space.size(), &hidden, env_cfg.action_space);
    let mut cma = CmaEs::new(policy.n_params, sigma0, None);
    let max_steps = env_cfg.max_steps;
    let solved = env_cfg.solved_threshold;

    let total_envs_per_gen = cma.pop_size * eval_episodes;

    eprintln!("🦀🚀 GPU CMA-ES on {} | {} params | pop={} | {}×{} = {} parallel envs",
        env_name, policy.n_params, cma.pop_size, cma.pop_size, eval_episodes, total_envs_per_gen);

    let mut best_ever = f64::NEG_INFINITY;
    let mut best_params: Option<Vec<f64>> = None;
    let mut total_evals = 0usize;
    let mut stale_gens = 0usize;
    let mut restarts = 0usize;
    let start = Instant::now();

    while total_evals < max_evals {
        let candidates = cma.ask();

        // ALL candidates evaluated simultaneously via vectorized envs
        let fitnesses = batch_evaluate(env_name, &candidates, &policy, eval_episodes, max_steps);

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
            generation: cma.gen, best: gen_best, best_ever,
            mean: fitnesses.iter().sum::<f64>() / fitnesses.len() as f64,
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

    // Final evaluation
    let (fm, fs) = if let Some(ref bp) = best_params {
        let pf32: Vec<f32> = bp.iter().map(|&v| v as f32).collect();
        let scores: Vec<f64> = (0..20).map(|_| {
            super::gpu_env::vec_evaluate(
                env_name, &pf32, &policy.layer_dims, 1, max_steps,
                policy.config.obs_dim, policy.config.act_dim,
                matches!(policy.config.action_space, ActionSpace::Discrete(_)),
            )
        }).collect();
        let m = scores.iter().sum::<f64>() / 20.0;
        let s = (scores.iter().map(|x| (x - m).powi(2)).sum::<f64>() / 20.0).sqrt();
        (m, s)
    } else { (0.0, 0.0) };

    RunResult {
        method: "GPU-CMA-ES".into(), environment: env_name.into(), best_ever,
        final_mean: fm, final_std: fs, total_evals, generations: cma.gen,
        elapsed: start.elapsed().as_secs_f64(), solved: best_ever >= solved,
    }
}

// ─── GPU OpenAI-ES ──────────────────────────────────────────────────

pub fn run_gpu_openai_es(env_name: &str, params: &Value, mut on_gen: impl FnMut(GenResult)) -> RunResult {
    let env_cfg = env::get_env_config(env_name).expect("Unknown env");
    let hidden = if let Some(h) = params.get("hidden").and_then(|v| v.as_array()) {
        h.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect()
    } else {
        env::default_hidden(env_name)
    };
    let max_evals = params.get("max_evals").and_then(|v| v.as_u64()).unwrap_or(100000) as usize;
    let eval_episodes = params.get("eval_episodes").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
    let pop_size = params.get("pop_size").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
    let lr = params.get("lr").and_then(|v| v.as_f64()).unwrap_or(0.02);
    let noise_std = params.get("noise_std").and_then(|v| v.as_f64()).unwrap_or(0.1);
    let weight_decay = params.get("weight_decay").and_then(|v| v.as_f64()).unwrap_or(0.0);

    let policy = Policy::new(env_cfg.obs_dim, env_cfg.action_space.size(), &hidden, env_cfg.action_space);
    let n = policy.n_params;
    let max_steps = env_cfg.max_steps;
    let solved = env_cfg.solved_threshold;

    let total_envs_per_gen = pop_size * 2 * eval_episodes; // + and - perturbations

    eprintln!("🦀🚀 GPU OpenAI-ES on {} | {} params | pop={} | {} parallel envs/gen",
        env_name, n, pop_size, total_envs_per_gen);

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

        // Build ALL candidates: pop_size positive + pop_size negative
        let mut all_candidates: Vec<Vec<f64>> = Vec::with_capacity(pop_size * 2);
        for eps in &epsilons {
            let plus: Vec<f64> = (0..n).map(|i| theta[i] + noise_std * eps[i]).collect();
            let minus: Vec<f64> = (0..n).map(|i| theta[i] - noise_std * eps[i]).collect();
            all_candidates.push(plus);
            all_candidates.push(minus);
        }

        // Evaluate ALL 2*pop_size candidates simultaneously
        let all_fitnesses = batch_evaluate(env_name, &all_candidates, &policy, eval_episodes, max_steps);

        total_evals += pop_size * 2 * eval_episodes;

        // Centered ranks
        let ranks = compute_centered_ranks(&all_fitnesses);

        // Gradient: ranks are interleaved [plus_0, minus_0, plus_1, minus_1, ...]
        let mut grad = vec![0.0f64; n];
        for (i, eps) in epsilons.iter().enumerate() {
            let rank_diff = ranks[2 * i] - ranks[2 * i + 1];
            for j in 0..n {
                grad[j] += rank_diff * eps[j];
            }
        }

        // Update theta
        for j in 0..n {
            theta[j] = theta[j] * (1.0 - weight_decay)
                + lr / (pop_size as f64 * noise_std) * grad[j];
        }

        let gen_best = all_fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let gen_mean = all_fitnesses.iter().sum::<f64>() / all_fitnesses.len() as f64;
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

    // Final evaluation
    let (fm, fs) = if let Some(ref bp) = best_params {
        let pf32: Vec<f32> = bp.iter().map(|&v| v as f32).collect();
        let scores: Vec<f64> = (0..20).map(|_| {
            super::gpu_env::vec_evaluate(
                env_name, &pf32, &policy.layer_dims, 1, max_steps,
                policy.config.obs_dim, policy.config.act_dim,
                matches!(policy.config.action_space, ActionSpace::Discrete(_)),
            )
        }).collect();
        let m = scores.iter().sum::<f64>() / 20.0;
        let s = (scores.iter().map(|x| (x - m).powi(2)).sum::<f64>() / 20.0).sqrt();
        (m, s)
    } else { (0.0, 0.0) };

    RunResult {
        method: "GPU-OpenAI-ES".into(), environment: env_name.into(), best_ever,
        final_mean: fm, final_std: fs, total_evals, generations: gen,
        elapsed: start.elapsed().as_secs_f64(), solved: best_ever >= solved,
    }
}

// ─── CPU vs GPU Benchmark ───────────────────────────────────────────

/// Benchmark: compare sequential vs vectorized evaluation speed.
pub fn run_gpu_benchmark(env_name: &str, params: &Value, mut on_gen: impl FnMut(GenResult)) -> RunResult {
    let env_cfg = env::get_env_config(env_name).expect("Unknown env");
    let hidden = vec![64, 32];
    let policy = Policy::new(env_cfg.obs_dim, env_cfg.action_space.size(), &hidden, env_cfg.action_space);
    let max_steps = env_cfg.max_steps;
    let _discrete = matches!(env_cfg.action_space, ActionSpace::Discrete(_));

    let n_candidates = params.get("n_candidates").and_then(|v| v.as_u64()).unwrap_or(28) as usize;
    let eval_episodes = params.get("eval_episodes").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

    // Generate random candidates
    let mut rng = OptRng::new(42);
    let candidates: Vec<Vec<f64>> = (0..n_candidates)
        .map(|_| (0..policy.n_params).map(|_| rng.randn() * 0.1).collect())
        .collect();

    // ── Sequential evaluation (current method) ──
    let seq_start = Instant::now();
    let _seq_fits: Vec<f64> = candidates.iter().map(|c| {
        let pf32: Vec<f32> = c.iter().map(|&v| v as f32).collect();
        let mut total = 0.0;
        for ep in 0..eval_episodes {
            let mut env = env::make(env_name, Some(ep as u64 * 1000)).unwrap();
            let mut obs = env.reset(Some(ep as u64 * 1000));
            let mut ep_reward = 0.0;
            for _ in 0..max_steps {
                let action = policy.forward(&obs, &pf32);
                let result = env.step(&action);
                ep_reward += result.reward;
                if result.done() { break; }
                obs = result.observation;
            }
            total += ep_reward;
        }
        total / eval_episodes as f64
    }).collect();
    let seq_time = seq_start.elapsed().as_secs_f64();

    // ── Rayon parallel evaluation (current method with threading) ──
    use rayon::prelude::*;
    let par_start = Instant::now();
    let _par_fits: Vec<f64> = candidates.par_iter().map(|c| {
        let pf32: Vec<f32> = c.iter().map(|&v| v as f32).collect();
        let mut total = 0.0;
        for ep in 0..eval_episodes {
            let mut env = env::make(env_name, Some(ep as u64 * 1000)).unwrap();
            let mut obs = env.reset(Some(ep as u64 * 1000));
            let mut ep_reward = 0.0;
            for _ in 0..max_steps {
                let action = policy.forward(&obs, &pf32);
                let result = env.step(&action);
                ep_reward += result.reward;
                if result.done() { break; }
                obs = result.observation;
            }
            total += ep_reward;
        }
        total / eval_episodes as f64
    }).collect();
    let par_time = par_start.elapsed().as_secs_f64();

    // ── Vectorized evaluation (GPU-ready, CPU fallback) ──
    let vec_start = Instant::now();
    let _vec_fits = batch_evaluate(env_name, &candidates, &policy, eval_episodes, max_steps);
    let vec_time = vec_start.elapsed().as_secs_f64();

    let speedup_par = seq_time / par_time;
    let speedup_vec = seq_time / vec_time;

    eprintln!("🏁 Benchmark: {} candidates × {} episodes on {}",
        n_candidates, eval_episodes, env_name);
    eprintln!("  Sequential:  {:.3}s", seq_time);
    eprintln!("  Rayon:       {:.3}s ({:.1}x speedup)", par_time, speedup_par);
    eprintln!("  Vectorized:  {:.3}s ({:.1}x speedup)", vec_time, speedup_vec);

    on_gen(GenResult {
        generation: 1,
        best: speedup_vec,
        best_ever: speedup_vec,
        mean: speedup_par,
        sigma: 0.0,
        evals: n_candidates * eval_episodes,
        time: seq_time + par_time + vec_time,
    });

    RunResult {
        method: "GPU-Benchmark".into(),
        environment: env_name.into(),
        best_ever: speedup_vec,
        final_mean: speedup_par,
        final_std: 0.0,
        total_evals: n_candidates * eval_episodes * 3,
        generations: 1,
        elapsed: seq_time + par_time + vec_time,
        solved: false,
    }
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_evaluate_cartpole() {
        let env_cfg = env::get_env_config("CartPole-v1").unwrap();
        let policy = Policy::new(env_cfg.obs_dim, env_cfg.action_space.size(), &[32, 16], env_cfg.action_space);

        // Create some candidates
        let mut rng = OptRng::new(42);
        let candidates: Vec<Vec<f64>> = (0..4)
            .map(|_| (0..policy.n_params).map(|_| rng.randn() * 0.1).collect())
            .collect();

        let fitnesses = batch_evaluate("CartPole-v1", &candidates, &policy, 3, 500);
        assert_eq!(fitnesses.len(), 4);

        // All should be positive (CartPole gives reward 1.0 per step)
        for (i, &f) in fitnesses.iter().enumerate() {
            assert!(f > 0.0, "Candidate {} fitness {} should be > 0", i, f);
        }
    }

    #[test]
    fn test_batch_evaluate_matches_sequential() {
        // Verify batch evaluation gives same results as sequential
        let env_cfg = env::get_env_config("CartPole-v1").unwrap();
        let hidden = vec![32, 16];
        let policy = Policy::new(env_cfg.obs_dim, env_cfg.action_space.size(), &hidden, env_cfg.action_space);
        let discrete = matches!(env_cfg.action_space, ActionSpace::Discrete(_));

        let mut rng = OptRng::new(123);
        let candidates: Vec<Vec<f64>> = (0..3)
            .map(|_| (0..policy.n_params).map(|_| rng.randn() * 0.1).collect())
            .collect();

        // Batch evaluation
        let batch_fits = batch_evaluate("CartPole-v1", &candidates, &policy, 2, 500);

        // Sequential evaluation using vec_evaluate
        let seq_fits: Vec<f64> = candidates.iter().map(|c| {
            let pf32: Vec<f32> = c.iter().map(|&v| v as f32).collect();
            super::super::gpu_env::vec_evaluate(
                "CartPole-v1", &pf32, &policy.layer_dims, 2, 500,
                env_cfg.obs_dim, env_cfg.action_space.size(), discrete,
            )
        }).collect();

        // Should be very close (same seeds, same math)
        for i in 0..3 {
            let diff = (batch_fits[i] - seq_fits[i]).abs();
            assert!(diff < 1e-3, "Candidate {} batch={:.2} seq={:.2} diff={:.6}",
                i, batch_fits[i], seq_fits[i], diff);
        }
    }

    #[test]
    fn test_gpu_cma_es_cartpole() {
        // Quick test: GPU CMA-ES should solve CartPole easily
        let params = serde_json::json!({
            "max_evals": 5000,
            "eval_episodes": 3,
        });
        let mut last_gen = None;
        let result = run_gpu_cma_es("CartPole-v1", &params, |g| {
            last_gen = Some(g);
        });

        assert!(result.best_ever > 50.0, "GPU CMA-ES should get >50 on CartPole, got {}", result.best_ever);
        assert!(last_gen.is_some());
    }
}
