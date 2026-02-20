//! GAIA Native Experiments — gradient-free RL in pure Rust.
//!
//! Architecture:
//! - `env`: Environment trait + implementations
//! - `policy`: Neural network forward pass (CPU, no backprop)
//! - `optim`: CMA-ES, OpenAI-ES optimizers
//! - `methods`: Experiment runners (cma, meta_learning, etc.)

pub mod env;
pub mod policy;
pub mod optim;
pub mod bench_cartpole;
