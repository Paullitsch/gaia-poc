//! Environment abstraction for GAIA experiments.
//!
//! Mirrors gymnasium's API but in pure Rust.
//! Each environment defines its observation/action spaces
//! and step/reset dynamics.

use std::fmt;

/// Action space type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActionSpace {
    /// Discrete actions: 0..n
    Discrete(usize),
    /// Continuous actions: n-dimensional vector in [-1, 1]
    Continuous(usize),
}

impl ActionSpace {
    pub fn size(&self) -> usize {
        match self {
            ActionSpace::Discrete(n) => *n,
            ActionSpace::Continuous(n) => *n,
        }
    }

    pub fn is_discrete(&self) -> bool {
        matches!(self, ActionSpace::Discrete(_))
    }
}

/// Action passed to environment.
#[derive(Debug, Clone)]
pub enum Action {
    Discrete(usize),
    Continuous(Vec<f32>),
}

/// Result of a step.
#[derive(Debug, Clone)]
pub struct StepResult {
    pub observation: Vec<f32>,
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
}

impl StepResult {
    pub fn done(&self) -> bool {
        self.terminated || self.truncated
    }
}

/// Environment configuration — enough to construct any environment.
#[derive(Debug, Clone)]
pub struct EnvConfig {
    pub name: String,
    pub obs_dim: usize,
    pub action_space: ActionSpace,
    pub max_steps: usize,
    pub solved_threshold: f64,
}

/// The core Environment trait.
/// Note: not Send because Box2D worlds contain raw pointers.
/// For parallelism, create one environment per thread.
pub trait Environment {
    /// Reset the environment to initial state, returns observation.
    fn reset(&mut self, seed: Option<u64>) -> Vec<f32>;

    /// Take an action, returns step result.
    fn step(&mut self, action: &Action) -> StepResult;

    /// Environment configuration.
    fn config(&self) -> &EnvConfig;

    /// Current step count.
    fn steps(&self) -> usize;
}

/// Registry of known environments.
pub fn get_env_config(name: &str) -> Option<EnvConfig> {
    match name {
        "CartPole-v1" => Some(EnvConfig {
            name: name.to_string(),
            obs_dim: 4,
            action_space: ActionSpace::Discrete(2),
            max_steps: 500,
            solved_threshold: 475.0,
        }),
        "LunarLander-v3" => Some(EnvConfig {
            name: name.to_string(),
            obs_dim: 8,
            action_space: ActionSpace::Discrete(4),
            max_steps: 1000,
            solved_threshold: 200.0,
        }),
        "BipedalWalker-v3" => Some(EnvConfig {
            name: name.to_string(),
            obs_dim: 24,
            action_space: ActionSpace::Continuous(4),
            max_steps: 1600,
            solved_threshold: 300.0,
        }),
        _ => None,
    }
}

/// Hidden layer configs for each environment.
pub fn default_hidden(name: &str) -> Vec<usize> {
    match name {
        "CartPole-v1" => vec![32, 16],
        "LunarLander-v3" => vec![64, 32],
        "BipedalWalker-v3" | "BipedalWalkerHardcore-v3" => vec![128, 64],
        _ => vec![64, 32],
    }
}

/// Factory: create an environment by name.
pub fn make(name: &str, seed: Option<u64>) -> Option<Box<dyn Environment>> {
    match name {
        "CartPole-v1" => Some(Box::new(cartpole::CartPole::new(seed))),
        "LunarLander-v3" => Some(Box::new(super::lunar_lander::LunarLander::new(seed))),
        _ => None,
    }
}

// ─── CartPole (pure Rust, no dependencies) ────────────────────────────

pub mod cartpole {
    use super::*;

    const GRAVITY: f64 = 9.8;
    const CART_MASS: f64 = 1.0;
    const POLE_MASS: f64 = 0.1;
    const TOTAL_MASS: f64 = CART_MASS + POLE_MASS;
    const POLE_HALF_LENGTH: f64 = 0.5;
    const FORCE_MAG: f64 = 10.0;
    const TAU: f64 = 0.02; // timestep
    const X_THRESHOLD: f64 = 2.4;
    const THETA_THRESHOLD: f64 = 12.0 * std::f64::consts::PI / 180.0;

    pub struct CartPole {
        config: EnvConfig,
        state: [f64; 4], // x, x_dot, theta, theta_dot
        step_count: usize,
        rng_seed: u64,
    }

    impl CartPole {
        pub fn new(seed: Option<u64>) -> Self {
            let config = get_env_config("CartPole-v1").unwrap();
            let mut env = CartPole {
                config,
                state: [0.0; 4],
                step_count: 0,
                rng_seed: seed.unwrap_or(42),
            };
            env.reset(seed);
            env
        }

        /// Simple LCG random in [-0.05, 0.05]
        fn rand_small(&mut self) -> f64 {
            self.rng_seed = self.rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = ((self.rng_seed >> 33) as f64) / (u32::MAX as f64); // [0, 1)
            val * 0.1 - 0.05 // [-0.05, 0.05)
        }
    }

    impl Environment for CartPole {
        fn reset(&mut self, seed: Option<u64>) -> Vec<f32> {
            if let Some(s) = seed {
                self.rng_seed = s;
            }
            self.state = [
                self.rand_small(),
                self.rand_small(),
                self.rand_small(),
                self.rand_small(),
            ];
            self.step_count = 0;
            self.state.iter().map(|&v| v as f32).collect()
        }

        fn step(&mut self, action: &Action) -> StepResult {
            let force = match action {
                Action::Discrete(a) => if *a == 1 { FORCE_MAG } else { -FORCE_MAG },
                Action::Continuous(v) => v[0] as f64 * FORCE_MAG,
            };

            let [x, x_dot, theta, theta_dot] = self.state;
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();

            let temp = (force + POLE_MASS * POLE_HALF_LENGTH * theta_dot * theta_dot * sin_theta) / TOTAL_MASS;
            let theta_acc = (GRAVITY * sin_theta - cos_theta * temp)
                / (POLE_HALF_LENGTH * (4.0 / 3.0 - POLE_MASS * cos_theta * cos_theta / TOTAL_MASS));
            let x_acc = temp - POLE_MASS * POLE_HALF_LENGTH * theta_acc * cos_theta / TOTAL_MASS;

            // Euler integration
            let new_x = x + TAU * x_dot;
            let new_x_dot = x_dot + TAU * x_acc;
            let new_theta = theta + TAU * theta_dot;
            let new_theta_dot = theta_dot + TAU * theta_acc;

            self.state = [new_x, new_x_dot, new_theta, new_theta_dot];
            self.step_count += 1;

            let terminated = new_x.abs() > X_THRESHOLD || new_theta.abs() > THETA_THRESHOLD;
            let truncated = self.step_count >= self.config.max_steps;

            StepResult {
                observation: self.state.iter().map(|&v| v as f32).collect(),
                reward: if terminated { 0.0 } else { 1.0 },
                terminated,
                truncated,
            }
        }

        fn config(&self) -> &EnvConfig {
            &self.config
        }

        fn steps(&self) -> usize {
            self.step_count
        }
    }
}

impl fmt::Display for EnvConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (obs={}, act={:?}, solved≥{})",
            self.name, self.obs_dim, self.action_space, self.solved_threshold)
    }
}
