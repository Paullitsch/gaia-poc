//! Native Rust experiment runner — replaces Python subprocess.
//!
//! When a job arrives for a supported env+method, runs it in native Rust.
//! Falls back to Python for unsupported combinations.

use super::env::{self, Environment, ActionSpace};
use super::policy::Policy;
use super::optim::CmaEs;
use rayon::prelude::*;
use serde_json::Value;
use std::time::Instant;

/// Result streamed per generation.
#[derive(Debug, Clone)]
pub struct GenResult {
    pub generation: usize,
    pub best: f64,
    pub best_ever: f64,
    pub mean: f64,
    pub sigma: f64,
    pub evals: usize,
    pub time: f64,
}

/// Final result.
#[derive(Debug)]
pub struct RunResult {
    pub method: String,
    pub environment: String,
    pub best_ever: f64,
    pub final_mean: f64,
    pub final_std: f64,
    pub total_evals: usize,
    pub generations: usize,
    pub elapsed: f64,
    pub solved: bool,
}

/// Check if we can run this job natively.
pub fn can_run_native(method: &str, environment: &str) -> bool {
    let supported_envs = ["CartPole-v1", "LunarLander-v3", "BipedalWalker-v3"];
    let supported_methods = ["cma_es", "openai_es", "scaling_test"];
    supported_envs.contains(&environment) && supported_methods.contains(&method)
}

/// Evaluate one candidate.
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

/// Run a CMA-ES job natively.
pub fn run_cma_es(
    env_name: &str,
    params: &Value,
    mut on_gen: impl FnMut(GenResult),
) -> RunResult {
    let env_cfg = env::get_env_config(env_name).expect("Unknown env");
    
    // Extract params from job
    let max_evals = params.get("max_evals").and_then(|v| v.as_u64()).unwrap_or(100000) as usize;
    let eval_episodes = params.get("eval_episodes").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
    let sigma0 = params.get("sigma0").and_then(|v| v.as_f64()).unwrap_or(0.5);
    
    // Network size: from params or env default
    let hidden: Vec<usize> = if let Some(h) = params.get("hidden").and_then(|v| v.as_array()) {
        h.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect()
    } else if let Some(cfg) = params.get("config").and_then(|v| v.as_str()) {
        // Scaling test configs
        match cfg {
            "1k" => vec![32, 16],
            "3k" => vec![64, 32],
            "10k" => vec![128, 64],
            "33k" => vec![256, 128],
            "100k" => vec![512, 256],
            _ => env::default_hidden(env_name),
        }
    } else {
        env::default_hidden(env_name)
    };

    let policy = Policy::new(env_cfg.obs_dim, env_cfg.action_space.size(), &hidden, env_cfg.action_space);
    let mut cma = CmaEs::new(policy.n_params, sigma0, None);
    
    let max_steps = env_cfg.max_steps;
    let solved = env_cfg.solved_threshold;
    
    eprintln!("🦀 Native CMA-ES on {} | {} params | pop={} | budget={}",
        env_name, policy.n_params, cma.pop_size, max_evals);

    let mut best_ever = f64::NEG_INFINITY;
    let mut best_params: Option<Vec<f64>> = None;
    let mut total_evals = 0usize;
    let start = Instant::now();

    while total_evals < max_evals {
        let candidates = cma.ask();

        let fitnesses: Vec<f64> = candidates.par_iter()
            .map(|c| {
                let pf32: Vec<f32> = c.iter().map(|&v| v as f32).collect();
                evaluate(env_name, &policy, &pf32, eval_episodes, max_steps)
            })
            .collect();

        total_evals += candidates.len() * eval_episodes;
        cma.tell(&candidates, &fitnesses);

        let gen_best = fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let gen_mean = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;

        if gen_best > best_ever {
            best_ever = gen_best;
            let idx = fitnesses.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap();
            best_params = Some(candidates[idx].clone());
        }

        let elapsed = start.elapsed().as_secs_f64();
        
        let gr = GenResult {
            generation: cma.gen,
            best: gen_best,
            best_ever,
            mean: gen_mean,
            sigma: cma.sigma,
            evals: total_evals,
            time: elapsed,
        };
        on_gen(gr);

        if best_ever >= solved {
            eprintln!("🎉 {} SOLVED! Score: {:.1} in {:.1}s", env_name, best_ever, elapsed);
            break;
        }
    }

    // Final eval
    let (final_mean, final_std) = if let Some(ref bp) = best_params {
        let pf32: Vec<f32> = bp.iter().map(|&v| v as f32).collect();
        let scores: Vec<f64> = (0..20)
            .map(|_| evaluate(env_name, &policy, &pf32, 1, max_steps))
            .collect();
        let m = scores.iter().sum::<f64>() / 20.0;
        let s = (scores.iter().map(|x| (x - m).powi(2)).sum::<f64>() / 20.0).sqrt();
        (m, s)
    } else {
        (0.0, 0.0)
    };

    RunResult {
        method: "CMA-ES".into(),
        environment: env_name.into(),
        best_ever,
        final_mean,
        final_std,
        total_evals,
        generations: cma.gen,
        elapsed: start.elapsed().as_secs_f64(),
        solved: best_ever >= solved,
    }
}
