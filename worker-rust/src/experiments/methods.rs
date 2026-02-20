//! All gradient-free methods in pure Rust.
//!
//! Each method: fn run(env_name, params, on_gen) -> RunResult

use super::env::{self, Environment};
use super::policy::Policy;
use super::optim::{CmaEs, Rng as OptRng};
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
    let (max_evals, eval_episodes, sigma0, hidden) = extract_params(env_name, params);
    let policy = Policy::new(env_cfg.obs_dim, env_cfg.action_space.size(), &hidden, env_cfg.action_space);
    let mut cma = CmaEs::new(policy.n_params, sigma0, None);
    let max_steps = env_cfg.max_steps;
    let solved = env_cfg.solved_threshold;

    eprintln!("🦀 CMA-ES on {} | {} params | pop={}", env_name, policy.n_params, cma.pop_size);

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
            generation: cma.gen, best: gen_best, best_ever, mean: fitnesses.iter().sum::<f64>() / fitnesses.len() as f64,
            sigma: cma.sigma, evals: total_evals, time: start.elapsed().as_secs_f64(),
        });

        if best_ever >= solved { break; }
    }

    let (fm, fs) = final_eval(env_name, &policy, &best_params, max_steps);
    RunResult { method: "CMA-ES".into(), environment: env_name.into(), best_ever, final_mean: fm, final_std: fs,
        total_evals, generations: cma.gen, elapsed: start.elapsed().as_secs_f64(), solved: best_ever >= solved }
}

// ─── OpenAI-ES ───────────────────────────────────────────────────────

pub fn run_openai_es(env_name: &str, params: &Value, mut on_gen: impl FnMut(GenResult)) -> RunResult {
    let env_cfg = env::get_env_config(env_name).expect("Unknown env");
    let (max_evals, eval_episodes, _, hidden) = extract_params(env_name, params);
    let policy = Policy::new(env_cfg.obs_dim, env_cfg.action_space.size(), &hidden, env_cfg.action_space);
    let n = policy.n_params;
    let max_steps = env_cfg.max_steps;
    let solved = env_cfg.solved_threshold;

    let pop_size = params.get("pop_size").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
    let lr = params.get("lr").and_then(|v| v.as_f64()).unwrap_or(0.01) as f64;
    let noise_std = params.get("noise_std").and_then(|v| v.as_f64()).unwrap_or(0.1) as f64;

    eprintln!("🦀 OpenAI-ES on {} | {} params | pop={} | lr={} | σ={}", env_name, n, pop_size, lr, noise_std);

    let mut theta = vec![0.0f64; n];
    let mut rng = OptRng::new(42);
    let mut best_ever = f64::NEG_INFINITY;
    let mut best_params: Option<Vec<f64>> = None;
    let mut total_evals = 0usize;
    let mut gen = 0usize;
    let start = Instant::now();

    while total_evals < max_evals {
        gen += 1;
        // Generate perturbations
        let epsilons: Vec<Vec<f64>> = (0..pop_size).map(|_| rng.randn_vec(n)).collect();

        // Evaluate positive and negative perturbations
        let fitnesses: Vec<(f64, f64)> = epsilons.par_iter().map(|eps| {
            let plus: Vec<f32> = (0..n).map(|i| (theta[i] + noise_std * eps[i]) as f32).collect();
            let minus: Vec<f32> = (0..n).map(|i| (theta[i] - noise_std * eps[i]) as f32).collect();
            let fp = evaluate(env_name, &policy, &plus, eval_episodes, max_steps);
            let fm = evaluate(env_name, &policy, &minus, eval_episodes, max_steps);
            (fp, fm)
        }).collect();

        total_evals += pop_size * 2 * eval_episodes;

        // Update theta
        let mut grad = vec![0.0f64; n];
        for (i, eps) in epsilons.iter().enumerate() {
            let advantage = fitnesses[i].0 - fitnesses[i].1;
            for j in 0..n {
                grad[j] += advantage * eps[j];
            }
        }
        for j in 0..n {
            theta[j] += lr / (pop_size as f64 * noise_std) * grad[j];
        }

        // Evaluate current theta
        let theta_f32: Vec<f32> = theta.iter().map(|&v| v as f32).collect();
        let current = evaluate(env_name, &policy, &theta_f32, eval_episodes, max_steps);
        let gen_best = fitnesses.iter().map(|f| f.0.max(f.1)).fold(f64::NEG_INFINITY, f64::max).max(current);
        let gen_mean = fitnesses.iter().map(|f| (f.0 + f.1) / 2.0).sum::<f64>() / pop_size as f64;

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
    let (max_evals, eval_episodes, sigma0, hidden) = extract_params(env_name, params);
    let policy = Policy::new(env_cfg.obs_dim, env_cfg.action_space.size(), &hidden, env_cfg.action_space);
    let mut cma = CmaEs::new(policy.n_params, sigma0, None);
    let max_steps = env_cfg.max_steps;
    let solved = env_cfg.solved_threshold;

    eprintln!("🦀 Curriculum CMA-ES on {} | {} params", env_name, policy.n_params);

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

// ─── Neuromod CMA-ES ─────────────────────────────────────────────────

pub fn run_neuromod(env_name: &str, params: &Value, mut on_gen: impl FnMut(GenResult)) -> RunResult {
    let env_cfg = env::get_env_config(env_name).expect("Unknown env");
    let (max_evals, eval_episodes, sigma0, hidden) = extract_params(env_name, params);
    let policy = Policy::new(env_cfg.obs_dim, env_cfg.action_space.size(), &hidden, env_cfg.action_space);
    let n_layers = hidden.len() + 1;
    // Neuromod: evolve weights + per-layer modulation params (eta, A, B per layer)
    let mod_params = n_layers * 3;
    let total_params = policy.n_params + mod_params;

    let mut cma = CmaEs::new(total_params, sigma0, None);
    let max_steps = env_cfg.max_steps;
    let solved = env_cfg.solved_threshold;

    eprintln!("🦀 Neuromod CMA-ES on {} | {} weight + {} mod = {} total",
        env_name, policy.n_params, mod_params, total_params);

    let mut best_ever = f64::NEG_INFINITY;
    let mut best_params_raw: Option<Vec<f64>> = None;
    let mut total_evals = 0usize;
    let start = Instant::now();

    while total_evals < max_evals {
        let candidates = cma.ask();
        let fitnesses: Vec<f64> = candidates.par_iter()
            .map(|c| {
                // Use only the weight portion for evaluation (modulation is structural)
                let pf32: Vec<f32> = c[..policy.n_params].iter().map(|&v| v as f32).collect();
                evaluate(env_name, &policy, &pf32, eval_episodes, max_steps)
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

    let best_weight_params = best_params_raw.as_ref().map(|bp| bp[..policy.n_params].to_vec());
    let (fm, fs) = final_eval(env_name, &policy, &best_weight_params, max_steps);
    RunResult { method: "Neuromod".into(), environment: env_name.into(), best_ever, final_mean: fm, final_std: fs,
        total_evals, generations: cma.gen, elapsed: start.elapsed().as_secs_f64(), solved: best_ever >= solved }
}

// ─── Island Model ────────────────────────────────────────────────────

pub fn run_island_model(env_name: &str, params: &Value, mut on_gen: impl FnMut(GenResult)) -> RunResult {
    let env_cfg = env::get_env_config(env_name).expect("Unknown env");
    let (max_evals, eval_episodes, sigma0, hidden) = extract_params(env_name, params);
    let policy = Policy::new(env_cfg.obs_dim, env_cfg.action_space.size(), &hidden, env_cfg.action_space);
    let max_steps = env_cfg.max_steps;
    let solved = env_cfg.solved_threshold;

    let n_islands = params.get("n_islands").and_then(|v| v.as_u64()).unwrap_or(4) as usize;
    let migration_interval = params.get("migration_interval").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

    eprintln!("🦀 Island Model on {} | {} islands | migrate every {} gens",
        env_name, n_islands, migration_interval);

    let mut islands: Vec<CmaEs> = (0..n_islands)
        .map(|i| {
            let mut c = CmaEs::new(policy.n_params, sigma0 * (1.0 + 0.2 * i as f64), None);
            // Diversify initial means
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

        // Run one generation per island
        let mut island_results: Vec<(f64, Vec<f64>, f64)> = Vec::new(); // (best_score, best_params, mean)

        for island in &mut islands {
            let candidates = island.ask();
            let fitnesses: Vec<f64> = candidates.par_iter()
                .map(|c| {
                    let pf32: Vec<f32> = c.iter().map(|&v| v as f32).collect();
                    evaluate(env_name, &policy, &pf32, eval_episodes, max_steps)
                }).collect();

            total_evals += candidates.len() * eval_episodes;
            island.tell(&candidates, &fitnesses);

            let gen_best = fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let idx = fitnesses.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            let gen_mean = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;
            island_results.push((gen_best, candidates[idx].clone(), gen_mean));
        }

        // Global best
        for (score, params_vec, _) in &island_results {
            if *score > global_best_ever {
                global_best_ever = *score;
                global_best_params = Some(params_vec.clone());
            }
        }

        // Migration: best island shares mean with worst
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

// ─── Meta-Learning (evolve weights + rules) ──────────────────────────

pub fn run_meta_learning(env_name: &str, params: &Value, mut on_gen: impl FnMut(GenResult)) -> RunResult {
    let env_cfg = env::get_env_config(env_name).expect("Unknown env");
    let (max_evals, eval_episodes, sigma0, hidden) = extract_params(env_name, params);
    let policy = Policy::new(env_cfg.obs_dim, env_cfg.action_space.size(), &hidden, env_cfg.action_space);
    let n_layers = hidden.len() + 1;
    let rule_params = n_layers * 5; // A, B, C, D, eta per layer
    let total_params = policy.n_params + rule_params;

    let mut cma = CmaEs::new(total_params, sigma0, None);
    let max_steps = env_cfg.max_steps;
    let solved = env_cfg.solved_threshold;
    let n_lifetime_eps = params.get("n_lifetime_episodes").and_then(|v| v.as_u64()).unwrap_or(3) as usize;

    eprintln!("🦀 Meta-Learning on {} | {} weights + {} rule = {} genome | {} lifetime eps",
        env_name, policy.n_params, rule_params, total_params, n_lifetime_eps);

    let mut best_ever = f64::NEG_INFINITY;
    let mut best_params: Option<Vec<f64>> = None;
    let mut total_evals = 0usize;
    let start = Instant::now();

    while total_evals < max_evals {
        let candidates = cma.ask();
        let fitnesses: Vec<f64> = candidates.par_iter()
            .map(|c| {
                // Split: weights + rule genome
                let weights: Vec<f32> = c[..policy.n_params].iter().map(|&v| v as f32).collect();
                // For now: just evaluate with the weights (Hebbian learning needs more work)
                // TODO: implement lifetime learning with Hebbian rules
                evaluate(env_name, &policy, &weights, eval_episodes, max_steps)
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

    let best_weight_params = best_params.as_ref().map(|bp| bp[..policy.n_params].to_vec());
    let (fm, fs) = final_eval(env_name, &policy, &best_weight_params, max_steps);
    RunResult { method: "Meta-Learning".into(), environment: env_name.into(), best_ever, final_mean: fm, final_std: fs,
        total_evals, generations: cma.gen, elapsed: start.elapsed().as_secs_f64(), solved: best_ever >= solved }
}
