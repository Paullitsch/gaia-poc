//! Feed-forward neural network policy — pure Rust, no backprop needed.
//!
//! Weights are stored as flat f32 vectors, manipulated by ES optimizers.
//! Forward pass only — this is what evolution optimizes.

use super::env::{Action, ActionSpace};

/// Network architecture description.
#[derive(Debug, Clone)]
pub struct PolicyConfig {
    pub obs_dim: usize,
    pub act_dim: usize,
    pub hidden: Vec<usize>,
    pub action_space: ActionSpace,
}

/// Feed-forward policy network.
#[derive(Debug, Clone)]
pub struct Policy {
    pub config: PolicyConfig,
    /// Layer shapes: [(in, out), ...] for weights, [out, ...] for biases
    pub layer_dims: Vec<(usize, usize)>,
    /// Total number of parameters (weights + biases)
    pub n_params: usize,
}

impl Policy {
    pub fn new(obs_dim: usize, act_dim: usize, hidden: &[usize], action_space: ActionSpace) -> Self {
        let mut dims = Vec::new();
        let mut prev = obs_dim;
        for &h in hidden {
            dims.push((prev, h));
            prev = h;
        }
        dims.push((prev, act_dim));

        let n_params: usize = dims.iter().map(|(i, o)| i * o + o).sum();

        Policy {
            config: PolicyConfig {
                obs_dim,
                act_dim,
                hidden: hidden.to_vec(),
                action_space,
            },
            layer_dims: dims,
            n_params,
        }
    }

    /// Forward pass: observation → action.
    /// `params` is a flat f32 slice of length `n_params`.
    pub fn forward(&self, obs: &[f32], params: &[f32]) -> Action {
        debug_assert_eq!(params.len(), self.n_params);
        debug_assert_eq!(obs.len(), self.config.obs_dim);

        let mut x: Vec<f32> = obs.to_vec();
        let mut offset = 0;

        for (layer_idx, &(fan_in, fan_out)) in self.layer_dims.iter().enumerate() {
            let w_size = fan_in * fan_out;
            let weights = &params[offset..offset + w_size];
            offset += w_size;
            let biases = &params[offset..offset + fan_out];
            offset += fan_out;

            // Matrix multiply: x (1×fan_in) @ W (fan_in×fan_out) + b
            let mut out = vec![0.0f32; fan_out];
            for j in 0..fan_out {
                let mut sum = biases[j];
                for i in 0..fan_in {
                    sum += x[i] * weights[i * fan_out + j];
                }
                out[j] = sum;
            }

            // Activation: tanh for hidden layers, none for output
            let is_last = layer_idx == self.layer_dims.len() - 1;
            if !is_last {
                for v in &mut out {
                    *v = v.tanh();
                }
            }

            x = out;
        }

        // Convert output to action
        match self.config.action_space {
            ActionSpace::Discrete(_) => {
                let best = x.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                Action::Discrete(best)
            }
            ActionSpace::Continuous(_) => {
                // Tanh squash to [-1, 1]
                let actions: Vec<f32> = x.iter().map(|v| v.tanh()).collect();
                Action::Continuous(actions)
            }
        }
    }

    /// Forward pass returning raw output (no action conversion).
    /// Useful for Hebbian learning where we need activations.
    pub fn forward_with_activations(&self, obs: &[f32], params: &[f32])
        -> (Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>)
    {
        let mut x: Vec<f32> = obs.to_vec();
        let mut offset = 0;
        let mut pre_acts = Vec::new();
        let mut post_acts = Vec::new();

        for (layer_idx, &(fan_in, fan_out)) in self.layer_dims.iter().enumerate() {
            pre_acts.push(x.clone());

            let w_size = fan_in * fan_out;
            let weights = &params[offset..offset + w_size];
            offset += w_size;
            let biases = &params[offset..offset + fan_out];
            offset += fan_out;

            let mut out = vec![0.0f32; fan_out];
            for j in 0..fan_out {
                let mut sum = biases[j];
                for i in 0..fan_in {
                    sum += x[i] * weights[i * fan_out + j];
                }
                out[j] = sum;
            }

            let is_last = layer_idx == self.layer_dims.len() - 1;
            if !is_last {
                for v in &mut out {
                    *v = v.tanh();
                }
            }

            post_acts.push(out.clone());
            x = out;
        }

        (x, pre_acts, post_acts)
    }

    /// Xavier/He initialization for a parameter vector.
    pub fn random_init(&self, seed: u64) -> Vec<f32> {
        let mut params = vec![0.0f32; self.n_params];
        let mut rng = SimpleRng::new(seed);
        let mut offset = 0;

        for &(fan_in, fan_out) in &self.layer_dims {
            let std = (2.0 / (fan_in + fan_out) as f64).sqrt() as f32;
            let w_size = fan_in * fan_out;
            for i in 0..w_size {
                params[offset + i] = rng.randn() * std;
            }
            offset += w_size;
            // Biases = 0
            offset += fan_out;
        }

        params
    }

    /// Format network architecture as string.
    pub fn arch_string(&self) -> String {
        let mut parts = vec![self.config.obs_dim.to_string()];
        for &h in &self.config.hidden {
            parts.push(h.to_string());
        }
        parts.push(self.config.act_dim.to_string());
        parts.join("→")
    }
}

/// Minimal PRNG for reproducible initialization.
/// Xoshiro256** — fast, good quality.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        SimpleRng { state: seed.wrapping_add(1) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    /// Box-Muller transform for Gaussian samples.
    fn randn(&mut self) -> f32 {
        let u1 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let u1 = u1.max(1e-10); // avoid log(0)
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_param_count() {
        // 8→64→32→4 = (8*64+64) + (64*32+32) + (32*4+4) = 576+2080+132 = 2788
        let p = Policy::new(8, 4, &[64, 32], ActionSpace::Discrete(4));
        assert_eq!(p.n_params, 2788);
    }

    #[test]
    fn test_forward_discrete() {
        let p = Policy::new(4, 2, &[8], ActionSpace::Discrete(2));
        let params = vec![0.0f32; p.n_params];
        let obs = vec![1.0, 0.0, 0.0, 0.0];
        let action = p.forward(&obs, &params);
        match action {
            Action::Discrete(a) => assert!(a < 2),
            _ => panic!("Expected discrete action"),
        }
    }

    #[test]
    fn test_forward_continuous() {
        let p = Policy::new(4, 2, &[8], ActionSpace::Continuous(2));
        let params = vec![0.1f32; p.n_params];
        let obs = vec![1.0, 0.5, -0.5, 0.0];
        let action = p.forward(&obs, &params);
        match action {
            Action::Continuous(v) => {
                assert_eq!(v.len(), 2);
                for a in &v {
                    assert!(*a >= -1.0 && *a <= 1.0);
                }
            }
            _ => panic!("Expected continuous action"),
        }
    }

    #[test]
    fn test_random_init() {
        let p = Policy::new(8, 4, &[64, 32], ActionSpace::Discrete(4));
        let params = p.random_init(42);
        assert_eq!(params.len(), p.n_params);
        // Should not be all zeros
        assert!(params.iter().any(|&v| v != 0.0));
    }
}
