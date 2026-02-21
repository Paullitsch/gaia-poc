//! Generic CMA-ES benchmark runner with Rayon parallel evaluation.

use super::env::{self};
use super::policy::Policy;
use super::optim::CmaEs;
use rayon::prelude::*;
use std::time::Instant;

/// Evaluate a parameter vector on any environment (single-threaded per call).
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

/// Run CMA-ES on CartPole.
pub fn run(max_evals: usize) -> BenchResult {
    run_env("CartPole-v1", max_evals, true)
}

/// Run CMA-ES on any environment with optional parallelism.
pub fn run_env(env_name: &str, max_evals: usize, parallel: bool) -> BenchResult {
    let env_cfg = env::get_env_config(env_name).unwrap();
    let hidden = env::default_hidden(env_name);
    let policy = Policy::new(
        env_cfg.obs_dim,
        env_cfg.action_space.size(),
        &hidden,
        env_cfg.action_space,
    );

    let n_threads = if parallel { rayon::current_num_threads() } else { 1 };
    println!("🦀 Rust CMA-ES on {}", env_name);
    println!("Network: {} ({} params)", policy.arch_string(), policy.n_params);

    let mut cma = CmaEs::new(policy.n_params, 0.5, None);
    println!("Pop: {} | Budget: {} | Threads: {}", cma.pop_size, max_evals, n_threads);

    let eval_episodes = 5;
    let max_steps = env_cfg.max_steps;
    let solved = env_cfg.solved_threshold;

    let mut best_ever: f64 = f64::NEG_INFINITY;
    let mut best_params: Option<Vec<f64>> = None;
    let mut total_evals: usize = 0;
    let start = Instant::now();

    while total_evals < max_evals {
        let candidates = cma.ask();

        let fitnesses: Vec<f64> = if parallel {
            candidates.par_iter()
                .map(|c| {
                    let params_f32: Vec<f32> = c.iter().map(|&v| v as f32).collect();
                    evaluate(env_name, &policy, &params_f32, eval_episodes, max_steps)
                })
                .collect()
        } else {
            candidates.iter()
                .map(|c| {
                    let params_f32: Vec<f32> = c.iter().map(|&v| v as f32).collect();
                    evaluate(env_name, &policy, &params_f32, eval_episodes, max_steps)
                })
                .collect()
        };

        total_evals += candidates.len() * eval_episodes;
        cma.tell(&candidates, &fitnesses);

        let gen_best = fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let gen_mean: f64 = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;

        if gen_best > best_ever {
            best_ever = gen_best;
            let idx = fitnesses.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap();
            best_params = Some(candidates[idx].clone());
        }

        let elapsed = start.elapsed().as_secs_f64();
        let solved_str = if best_ever >= solved { " ✅ SOLVED!" } else { "" };
        println!("Gen {:4} | Best: {:8.1} | Ever: {:8.1} | Mean: {:8.1} | σ: {:.4} | Evals: {:>8} | {:.1}s{}",
            cma.gen, gen_best, best_ever, gen_mean, cma.sigma, total_evals, elapsed, solved_str);

        if best_ever >= solved {
            println!("\n🎉 {} SOLVED in {:.2}s!", env_name, elapsed);
            break;
        }
    }

    let elapsed = start.elapsed().as_secs_f64();

    // Final eval
    let (final_mean, final_std) = if let Some(ref bp) = best_params {
        let params_f32: Vec<f32> = bp.iter().map(|&v| v as f32).collect();
        let scores: Vec<f64> = (0..20)
            .map(|_| evaluate(env_name, &policy, &params_f32, 1, max_steps))
            .collect();
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let std = (scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64).sqrt();
        (mean, std)
    } else {
        (0.0, 0.0)
    };

    println!("\n📊 Final: {:.1} ± {:.1} (20 episodes)", final_mean, final_std);
    println!("⏱️  Total: {:.3}s | {:.0} evals/sec | {} threads", elapsed, total_evals as f64 / elapsed, n_threads);

    BenchResult {
        env_name: env_name.to_string(),
        best_ever,
        final_mean,
        final_std,
        total_evals,
        generations: cma.gen,
        elapsed_secs: elapsed,
        evals_per_sec: total_evals as f64 / elapsed,
        n_threads,
        solved: best_ever >= solved,
    }
}

#[derive(Debug)]
pub struct BenchResult {
    pub env_name: String,
    pub best_ever: f64,
    pub final_mean: f64,
    pub final_std: f64,
    pub total_evals: usize,
    pub generations: usize,
    pub elapsed_secs: f64,
    pub evals_per_sec: f64,
    pub n_threads: usize,
    pub solved: bool,
}
