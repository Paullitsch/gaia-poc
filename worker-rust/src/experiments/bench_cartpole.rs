//! CMA-ES on CartPole benchmark — pure Rust, no Python.
//!
//! Run with: cargo run --release -- --bench-cartpole

use super::env::{self, Action, Environment};
use super::policy::Policy;
use super::optim::CmaEs;
use std::time::Instant;

/// Evaluate a parameter vector on CartPole.
fn evaluate(policy: &Policy, params: &[f32], n_episodes: usize, max_steps: usize) -> f64 {
    let mut total = 0.0;
    for ep in 0..n_episodes {
        let mut env = env::cartpole::CartPole::new(Some(ep as u64 * 1000));
        let obs = env.reset(Some(ep as u64 * 1000));
        let mut obs = obs;
        let mut ep_reward = 0.0;

        for _ in 0..max_steps {
            let action = policy.forward(&obs, params);
            let result = env.step(&action);
            ep_reward += result.reward;
            if result.done() {
                break;
            }
            obs = result.observation;
        }
        total += ep_reward;
    }
    total / n_episodes as f64
}

/// Run CMA-ES on CartPole and return results.
pub fn run(max_evals: usize) -> BenchResult {
    let env_cfg = env::get_env_config("CartPole-v1").unwrap();
    let hidden = env::default_hidden("CartPole-v1");
    let policy = Policy::new(
        env_cfg.obs_dim,
        env_cfg.action_space.size(),
        &hidden,
        env_cfg.action_space,
    );

    println!("🦀 Rust CMA-ES on CartPole-v1");
    println!("Network: {} ({} params)", policy.arch_string(), policy.n_params);

    let mut cma = CmaEs::new(policy.n_params, 0.5, None);
    println!("Pop: {} | Budget: {} evals", cma.pop_size, max_evals);

    let eval_episodes = 5;
    let max_steps = env_cfg.max_steps;
    let solved = env_cfg.solved_threshold;

    let mut best_ever: f64 = f64::NEG_INFINITY;
    let mut best_params: Option<Vec<f64>> = None;
    let mut total_evals: usize = 0;
    let start = Instant::now();

    while total_evals < max_evals {
        let candidates = cma.ask();

        let fitnesses: Vec<f64> = candidates.iter()
            .map(|c| {
                let params_f32: Vec<f32> = c.iter().map(|&v| v as f32).collect();
                evaluate(&policy, &params_f32, eval_episodes, max_steps)
            })
            .collect();

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
            println!("\n🎉 CartPole SOLVED in {:.2}s!", elapsed);
            break;
        }
    }

    let elapsed = start.elapsed().as_secs_f64();

    // Final eval
    let (final_mean, final_std) = if let Some(ref bp) = best_params {
        let params_f32: Vec<f32> = bp.iter().map(|&v| v as f32).collect();
        let scores: Vec<f64> = (0..20)
            .map(|i| {
                let mut env = env::cartpole::CartPole::new(Some(99999 + i));
                let obs = env.reset(Some(99999 + i));
                let mut obs = obs;
                let mut r = 0.0;
                for _ in 0..max_steps {
                    let action = policy.forward(&obs, &params_f32);
                    let result = env.step(&action);
                    r += result.reward;
                    if result.done() { break; }
                    obs = result.observation;
                }
                r
            })
            .collect();
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let std = (scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64).sqrt();
        (mean, std)
    } else {
        (0.0, 0.0)
    };

    println!("\n📊 Final: {:.1} ± {:.1} (20 episodes)", final_mean, final_std);
    println!("⏱️  Total: {:.3}s | {:.0} evals/sec", elapsed, total_evals as f64 / elapsed);

    BenchResult {
        best_ever,
        final_mean,
        final_std,
        total_evals,
        generations: cma.gen,
        elapsed_secs: elapsed,
        evals_per_sec: total_evals as f64 / elapsed,
        solved: best_ever >= solved,
    }
}

#[derive(Debug)]
pub struct BenchResult {
    pub best_ever: f64,
    pub final_mean: f64,
    pub final_std: f64,
    pub total_evals: usize,
    pub generations: usize,
    pub elapsed_secs: f64,
    pub evals_per_sec: f64,
    pub solved: bool,
}
