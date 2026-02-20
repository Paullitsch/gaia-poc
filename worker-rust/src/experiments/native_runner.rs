//! Native Rust experiment runner — replaces Python subprocess entirely.

use super::methods;
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
const SUPPORTED_ENVS: &[&str] = &["CartPole-v1", "LunarLander-v3", "BipedalWalker-v3"];

/// All methods we can run natively.
const SUPPORTED_METHODS: &[&str] = &[
    "cma_es", "openai_es", "scaling_test", "curriculum",
    "neuromod", "neuromod_island", "island_model", "island_advanced",
    "meta_learning", "meta_learning_pure",
];

/// Check if we can run this job natively in Rust.
pub fn can_run_native(method: &str, environment: &str) -> bool {
    SUPPORTED_ENVS.contains(&environment) && SUPPORTED_METHODS.contains(&method)
}

/// Run a job natively. Dispatches to the correct method.
pub fn run(
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
        _ => RunResult {
            method: method.into(), environment: env_name.into(),
            best_ever: 0.0, final_mean: 0.0, final_std: 0.0,
            total_evals: 0, generations: 0, elapsed: 0.0, solved: false,
        }
    }
}
