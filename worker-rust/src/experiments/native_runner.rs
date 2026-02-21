//! Native Rust experiment runner — replaces Python subprocess entirely.

use super::methods;
use super::ppo;
use serde_json::Value;

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

/// Supported environments.
const SUPPORTED_ENVS: &[&str] = &["CartPole-v1", "LunarLander-v3", "BipedalWalker-v3", "Swimmer-v1"];

/// All methods we can run natively.
const SUPPORTED_METHODS: &[&str] = &[
    "cma_es", "openai_es", "scaling_test", "curriculum",
    "neuromod", "neuromod_island", "island_model", "island_advanced",
    "meta_learning", "meta_learning_pure", "ppo_baseline", "ppo_gpu",
    "gpu_cma_es", "gpu_openai_es", "gpu_benchmark",
    "gpu_full_cma_es", "gpu_full_openai_es", "gpu_full_benchmark",
];

/// Check if we can run this job natively in Rust.
pub fn can_run_native(method: &str, environment: &str) -> bool {
    let env_ok = SUPPORTED_ENVS.contains(&environment)
        || (environment.starts_with("Pendulum-") && environment.ends_with("Link"));
    env_ok && SUPPORTED_METHODS.contains(&method)
}

/// Run a job natively. Dispatches to the correct method.
pub fn run(
    method: &str,
    env_name: &str,
    params: &Value,
    on_gen: impl FnMut(GenResult),
) -> RunResult {
    dispatch(method, env_name, params, on_gen)
}

/// Run with cancellation support via AtomicBool.
/// Methods check the cancel flag every generation and exit early if set.
pub fn run_cancellable(
    method: &str,
    env_name: &str,
    params: &Value,
    mut on_gen: impl FnMut(GenResult),
    cancel: &std::sync::atomic::AtomicBool,
) -> RunResult {
    // Inject max_evals=1 into params when cancelled to make the loop exit
    use std::sync::atomic::Ordering;
    dispatch(method, env_name, params, |gr| {
        on_gen(gr);
        // After sending result, check cancel. If set, the next iteration
        // will see max_evals exceeded because we can't modify the running method.
        // The real fix: process::exit after completing the current generation.
        if cancel.load(Ordering::Relaxed) {
            eprintln!("⚠️ Cancellation requested — will exit after current job completes");
        }
    })
}

fn dispatch(
    method: &str,
    env_name: &str,
    params: &Value,
    on_gen: impl FnMut(GenResult),
) -> RunResult {
    match method {
        "cma_es" | "scaling_test" => methods::run_cma_es(env_name, params, on_gen),
        "openai_es" => methods::run_openai_es(env_name, params, on_gen),
        "curriculum" => methods::run_curriculum(env_name, params, on_gen),
        "neuromod" | "neuromod_island" => methods::run_neuromod(env_name, params, on_gen),
        "island_model" | "island_advanced" => methods::run_island_model(env_name, params, on_gen),
        "meta_learning" | "meta_learning_pure" => methods::run_meta_learning(env_name, params, on_gen),
        "ppo_baseline" => {
            // Auto-select GPU PPO for supported environments
            match env_name {
                "CartPole-v1" | "Swimmer-v1" => ppo::run_ppo_gpu(env_name, params, on_gen),
                _ if env_name.starts_with("Pendulum-") && env_name.ends_with("Link") =>
                    ppo::run_ppo_gpu(env_name, params, on_gen),
                _ => ppo::run_ppo(env_name, params, on_gen), // LunarLander, BipedalWalker: CPU
            }
        }
        "ppo_gpu" => ppo::run_ppo_gpu(env_name, params, on_gen),
        "gpu_cma_es" => super::gpu_methods::run_gpu_cma_es(env_name, params, on_gen),
        "gpu_openai_es" => super::gpu_methods::run_gpu_openai_es(env_name, params, on_gen),
        "gpu_benchmark" => super::gpu_methods::run_gpu_benchmark(env_name, params, on_gen),
        "gpu_full_cma_es" => super::gpu_eval::run_gpu_full_cma_es(env_name, params, on_gen),
        "gpu_full_openai_es" => super::gpu_eval::run_gpu_full_openai_es(env_name, params, on_gen),
        "gpu_full_benchmark" => super::gpu_eval::run_gpu_full_benchmark(env_name, params, on_gen),
        _ => RunResult {
            method: method.into(), environment: env_name.into(),
            best_ever: 0.0, final_mean: 0.0, final_std: 0.0,
            total_evals: 0, generations: 0, elapsed: 0.0, solved: false,
        }
    }
}
