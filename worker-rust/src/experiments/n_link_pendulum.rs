//! N-Link Pendulum — GPU-native, scalable difficulty.
//!
//! Generalization of inverted pendulum / Acrobot to N links.
//! All joints are actuated (continuous control).
//!
//! Physics: N rigid links connected end-to-end, hanging from a fixed point.
//! Gravity acts on each link. Goal: balance all links upright.
//!
//! State: [cos(θ₁), sin(θ₁), ..., cos(θₙ), sin(θₙ), ω₁, ..., ωₙ] = 3N dims
//! Action: [τ₁, ..., τₙ] torques on each joint, in [-1, 1]
//! Reward: height of tip + stability bonus - control cost
//!
//! Difficulty scales with N:
//! - N=2: like Acrobot (easy-medium)
//! - N=3: significantly harder
//! - N=5: very hard for gradient-free
//! - N=10: likely impossible without local learning
//!
//! This is a GPU-friendly environment: pure ODE physics, no contacts.

use super::env::*;

// ─── Constants ────────────────────────────────────────────────────────
const GRAVITY: f32 = 9.81;
const LINK_LENGTH: f32 = 1.0;
const LINK_MASS: f32 = 1.0;
const DT: f32 = 0.05;  // 20 Hz
const MAX_TORQUE: f32 = 1.0;
const CTRL_COST_WEIGHT: f32 = 0.01;
const MAX_VEL: f32 = 8.0; // velocity clamp

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng { state: seed.wrapping_add(1) }
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (self.state >> 11) as f64 / (1u64 << 53) as f64;
        lo + (hi - lo) * u as f32
    }
}

pub struct NLinkPendulum {
    config: EnvConfig,
    n_links: usize,
    // State
    angles: Vec<f32>,     // θ_i: angle from vertical (0 = upright)
    velocities: Vec<f32>, // ω_i: angular velocity
    step_count: usize,
    max_steps: usize,
    rng: Rng,
}

impl NLinkPendulum {
    pub fn new(n_links: usize, seed: Option<u64>) -> Self {
        let obs_dim = 3 * n_links; // cos, sin, vel per link
        let max_steps = match n_links {
            2 => 500,
            3 => 500,
            5 => 1000,
            _ => 1000,
        };
        // Solved threshold: average height > 0.8 * max_height
        // Max height = n_links * LINK_LENGTH
        let max_height = n_links as f64 * LINK_LENGTH as f64;
        let solved_threshold = max_height * 200.0; // reward ~200 per step × steps

        let config = EnvConfig {
            name: format!("Pendulum-{}Link", n_links),
            obs_dim,
            action_space: ActionSpace::Continuous(n_links),
            max_steps,
            solved_threshold,
        };

        let mut env = NLinkPendulum {
            config,
            n_links,
            angles: vec![0.0; n_links],
            velocities: vec![0.0; n_links],
            step_count: 0,
            max_steps,
            rng: Rng::new(seed.unwrap_or(42)),
        };
        env.do_reset();
        env
    }

    fn do_reset(&mut self) -> Vec<f32> {
        // Start hanging down (π) with small perturbation
        for i in 0..self.n_links {
            self.angles[i] = std::f32::consts::PI + self.rng.uniform(-0.1, 0.1);
            self.velocities[i] = self.rng.uniform(-0.1, 0.1);
        }
        self.step_count = 0;
        self.get_obs()
    }

    fn get_obs(&self) -> Vec<f32> {
        let mut obs = Vec::with_capacity(3 * self.n_links);
        for i in 0..self.n_links {
            obs.push(self.angles[i].cos());
            obs.push(self.angles[i].sin());
            obs.push(self.velocities[i]);
        }
        obs
    }

    /// Compute tip height (sum of vertical projections of all links).
    /// Maximum = n_links * LINK_LENGTH (all pointing up).
    /// Minimum = -n_links * LINK_LENGTH (all pointing down).
    fn tip_height(&self) -> f32 {
        let mut height = 0.0f32;
        let mut cumulative_angle = 0.0f32;
        for i in 0..self.n_links {
            cumulative_angle += self.angles[i];
            // Each link contributes its vertical projection
            // angle=0 means pointing up → cos(0) = 1 → positive height
            height += LINK_LENGTH * cumulative_angle.cos();
        }
        height
    }

    fn do_step(&mut self, actions: &[f32]) -> StepResult {
        // ── Physics: N-link pendulum dynamics ──
        // Simplified model using independent joint dynamics with coupling through gravity.
        // Full Lagrangian dynamics would require computing the mass matrix,
        // but for gradient-free optimization the simplified model is sufficient.

        let n = self.n_links;

        // Compute gravitational torques for each joint
        // Torque on joint i from gravity acting on links i..N
        let mut grav_torques = vec![0.0f32; n];
        for i in 0..n {
            let mut torque = 0.0f32;
            // Gravity torque: sum of mass * g * distance * sin(angle) for links below
            for j in i..n {
                let mass_below = LINK_MASS * (n - j) as f32;
                let lever = LINK_LENGTH;
                let mut angle_to_j = 0.0f32;
                for k in 0..=j {
                    angle_to_j += self.angles[k];
                }
                torque += -mass_below * GRAVITY * lever * angle_to_j.sin();
            }
            grav_torques[i] = torque;
        }

        // Compute effective inertia for each joint
        // Joint i has inertia from all links below it
        let mut inertias = vec![0.0f32; n];
        for i in 0..n {
            let n_below = (n - i) as f32;
            // Simplified: I = sum of m*L² for links below
            inertias[i] = LINK_MASS * LINK_LENGTH * LINK_LENGTH * n_below;
        }

        // Update velocities and angles
        for i in 0..n {
            let applied_torque = actions[i.min(actions.len() - 1)].clamp(-1.0, 1.0) * MAX_TORQUE;

            // Damping
            let damping = -0.1 * self.velocities[i];

            let alpha = (grav_torques[i] + applied_torque + damping) / inertias[i];
            self.velocities[i] += alpha * DT;
            self.velocities[i] = self.velocities[i].clamp(-MAX_VEL, MAX_VEL);
            self.angles[i] += self.velocities[i] * DT;
        }

        // ── Reward ──
        // 1. Height of tip (normalized to [0, 1])
        let height = self.tip_height();
        let max_height = self.n_links as f32 * LINK_LENGTH;
        let height_reward = (height + max_height) / (2.0 * max_height); // [0, 1]

        // 2. Stability bonus (low velocity = more stable)
        let vel_penalty: f32 = self.velocities.iter().map(|v| v * v).sum::<f32>() * 0.01;

        // 3. Control cost
        let ctrl_cost: f32 = actions.iter().map(|a| a * a).sum::<f32>() * CTRL_COST_WEIGHT;

        let reward = height_reward - vel_penalty - ctrl_cost;

        self.step_count += 1;
        let truncated = self.step_count >= self.max_steps;

        StepResult {
            observation: self.get_obs(),
            reward: reward as f64,
            terminated: false, // Never terminates
            truncated,
        }
    }
}

impl Environment for NLinkPendulum {
    fn reset(&mut self, seed: Option<u64>) -> Vec<f32> {
        if let Some(s) = seed {
            self.rng = Rng::new(s);
        }
        self.do_reset()
    }

    fn step(&mut self, action: &Action) -> StepResult {
        match action {
            Action::Continuous(v) => self.do_step(v),
            Action::Discrete(_) => {
                // Map discrete to continuous: 0=no torque on all joints
                let actions = vec![0.0f32; self.n_links];
                self.do_step(&actions)
            }
        }
    }

    fn config(&self) -> &EnvConfig { &self.config }
    fn steps(&self) -> usize { self.step_count }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pendulum_2link_reset() {
        let mut env = NLinkPendulum::new(2, Some(42));
        let obs = env.reset(Some(42));
        assert_eq!(obs.len(), 6); // 3 * 2 links
        // Should have cos, sin, vel for each link
        // Starting near PI (hanging down): cos(PI) ≈ -1, sin(PI) ≈ 0
        assert!(obs[0] < -0.9, "cos(PI) should be near -1, got {}", obs[0]);
    }

    #[test]
    fn test_pendulum_5link_reset() {
        let mut env = NLinkPendulum::new(5, Some(42));
        let obs = env.reset(Some(42));
        assert_eq!(obs.len(), 15); // 3 * 5 links
    }

    #[test]
    fn test_pendulum_step() {
        let mut env = NLinkPendulum::new(3, Some(42));
        env.reset(Some(42));
        
        let result = env.step(&Action::Continuous(vec![0.5, -0.5, 0.2]));
        assert_eq!(result.observation.len(), 9); // 3 * 3
        assert!(!result.terminated);
    }

    #[test]
    fn test_pendulum_via_factory() {
        let mut env = super::super::env::make("Pendulum-3Link", Some(42)).unwrap();
        let obs = env.reset(Some(42));
        assert_eq!(obs.len(), 9);
    }

    #[test]
    fn test_pendulum_config() {
        let cfg = super::super::env::get_env_config("Pendulum-5Link").unwrap();
        assert_eq!(cfg.obs_dim, 15);
        assert_eq!(cfg.action_space, ActionSpace::Continuous(5));
    }
}
