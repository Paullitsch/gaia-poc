//! PPO (Proximal Policy Optimization) in pure Rust — backpropagation baseline.
//!
//! Manual forward + backward pass for small FF networks. No autograd framework needed.
//! This is the BACKPROP CONTROL GROUP — proves that our gradient-free methods compete.

use super::env::{self, ActionSpace, Action};
use super::native_runner::{GenResult, RunResult};
use super::optim::Rng;
use serde_json::Value;
use std::time::Instant;

// ─── Differentiable Policy Network ──────────────────────────────────

/// A small FF network with manual forward + backward pass.
/// Stores both weights and their gradients.
struct DiffPolicy {
    layer_dims: Vec<(usize, usize)>,
    weights: Vec<Vec<f64>>,    // [layer][fan_in * fan_out]  (row-major)
    biases: Vec<Vec<f64>>,     // [layer][fan_out]
    action_space: ActionSpace,
    /// Learnable log standard deviation for continuous actions (per action dim).
    /// FIX: was hardcoded to -0.5, now learned via gradient descent.
    log_std: Option<Vec<f64>>,
}

impl DiffPolicy {
    fn new(obs_dim: usize, act_dim: usize, hidden: &[usize], action_space: ActionSpace) -> Self {
        let mut dims = Vec::new();
        let mut prev = obs_dim;
        for &h in hidden {
            dims.push((prev, h));
            prev = h;
        }
        dims.push((prev, act_dim));

        let mut rng = Rng::new(42);
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for &(fan_in, fan_out) in &dims {
            let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
            let w: Vec<f64> = (0..fan_in * fan_out).map(|_| rng.randn() * std).collect();
            let b = vec![0.0; fan_out];
            weights.push(w);
            biases.push(b);
        }

        let log_std = match &action_space {
            ActionSpace::Continuous(n) => Some(vec![-0.5; *n]), // initial std ≈ 0.6
            _ => None,
        };

        DiffPolicy { layer_dims: dims, weights, biases, action_space, log_std }
    }

    fn n_params(&self) -> usize {
        let net: usize = self.layer_dims.iter().map(|(i, o)| i * o + o).sum();
        net + self.log_std.as_ref().map_or(0, |v| v.len())
    }

    /// Forward pass returning logits/means + all intermediate activations for backprop.
    fn forward(&self, obs: &[f64]) -> ForwardResult {
        let mut x = obs.to_vec();
        let mut pre_activations = Vec::new();  // z = Wx + b (before activation)
        let mut post_activations = Vec::new(); // a = activation(z)

        post_activations.push(x.clone()); // input layer

        for (layer_idx, &(fan_in, fan_out)) in self.layer_dims.iter().enumerate() {
            let w = &self.weights[layer_idx];
            let b = &self.biases[layer_idx];

            // z = x @ W + b
            let mut z = vec![0.0; fan_out];
            for j in 0..fan_out {
                let mut sum = b[j];
                for i in 0..fan_in {
                    sum += x[i] * w[i * fan_out + j];
                }
                z[j] = sum;
            }

            pre_activations.push(z.clone());

            // Activation
            let is_last = layer_idx == self.layer_dims.len() - 1;
            let a = if !is_last {
                z.iter().map(|&v| v.tanh()).collect()
            } else {
                z.clone() // raw logits for output
            };

            post_activations.push(a.clone());
            x = a;
        }

        ForwardResult {
            output: x,
            pre_activations,
            post_activations,
        }
    }

    /// Backward pass: compute gradients of loss w.r.t. all weights/biases.
    /// `d_output` is dL/d(output).
    fn backward(&self, fwd: &ForwardResult, d_output: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let n_layers = self.layer_dims.len();
        let mut d_weights = Vec::new();
        let mut d_biases = Vec::new();

        for _ in 0..n_layers {
            d_weights.push(Vec::new());
            d_biases.push(Vec::new());
        }

        let mut delta = d_output.to_vec();

        for layer_idx in (0..n_layers).rev() {
            let (fan_in, fan_out) = self.layer_dims[layer_idx];
            let is_last = layer_idx == n_layers - 1;

            // Apply activation derivative for non-last layers
            if !is_last {
                let z = &fwd.pre_activations[layer_idx];
                for j in 0..fan_out {
                    let tanh_z = z[j].tanh();
                    delta[j] *= 1.0 - tanh_z * tanh_z; // dtanh/dz = 1 - tanh²
                }
            }

            // dL/db = delta
            d_biases[layer_idx] = delta.clone();

            // dL/dW = a_prev^T @ delta
            let a_prev = &fwd.post_activations[layer_idx]; // activation of previous layer
            let mut dw = vec![0.0; fan_in * fan_out];
            for i in 0..fan_in {
                for j in 0..fan_out {
                    dw[i * fan_out + j] = a_prev[i] * delta[j];
                }
            }
            d_weights[layer_idx] = dw;

            // Propagate delta to previous layer: delta_prev = delta @ W^T
            if layer_idx > 0 {
                let w = &self.weights[layer_idx];
                let mut delta_prev = vec![0.0; fan_in];
                for i in 0..fan_in {
                    for j in 0..fan_out {
                        delta_prev[i] += w[i * fan_out + j] * delta[j];
                    }
                }
                delta = delta_prev;
            }
        }

        (d_weights, d_biases)
    }

    /// Apply gradients with Adam optimizer step.
    /// `d_log_std` is optional gradient for learnable log_std (continuous only).
    fn apply_gradients(&mut self, d_weights: &[Vec<f64>], d_biases: &[Vec<f64>],
                       d_log_std: Option<&[f64]>, adam: &mut Adam) {
        let mut flat_grad = Vec::new();
        for layer_idx in 0..self.layer_dims.len() {
            flat_grad.extend_from_slice(&d_weights[layer_idx]);
            flat_grad.extend_from_slice(&d_biases[layer_idx]);
        }
        // Append log_std gradients
        if let Some(dls) = d_log_std {
            flat_grad.extend_from_slice(dls);
        }

        let updates = adam.step(&flat_grad);

        let mut offset = 0;
        for layer_idx in 0..self.layer_dims.len() {
            let (fan_in, fan_out) = self.layer_dims[layer_idx];
            let w_size = fan_in * fan_out;
            for i in 0..w_size {
                self.weights[layer_idx][i] -= updates[offset + i];
            }
            offset += w_size;
            for j in 0..fan_out {
                self.biases[layer_idx][j] -= updates[offset + j];
            }
            offset += fan_out;
        }
        // Update log_std
        if let Some(ref mut ls) = self.log_std {
            for i in 0..ls.len() {
                ls[i] -= updates[offset + i];
                ls[i] = ls[i].clamp(-2.0, 0.5); // keep std in reasonable range
            }
        }
    }

    /// Sample action from policy (continuous: Gaussian, discrete: categorical).
    fn sample_action(&self, output: &[f64], rng: &mut Rng) -> (Action, f64) {
        match self.action_space {
            ActionSpace::Discrete(n) => {
                // Softmax
                let max_v = output.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp: Vec<f64> = output.iter().map(|v| (v - max_v).exp()).collect();
                let sum: f64 = exp.iter().sum();
                let probs: Vec<f64> = exp.iter().map(|v| v / sum).collect();

                // Sample
                let u = rng.uniform(0.0, 1.0);
                let mut cumsum = 0.0;
                let mut action = n - 1;
                for i in 0..n {
                    cumsum += probs[i];
                    if u < cumsum {
                        action = i;
                        break;
                    }
                }
                let log_prob = probs[action].max(1e-10).ln();
                (Action::Discrete(action), log_prob)
            }
            ActionSpace::Continuous(n) => {
                // FIX: use learnable log_std per action dimension
                let log_std_vec = self.log_std.as_ref().unwrap();
                let mut actions = Vec::with_capacity(n);
                let mut log_prob = 0.0;
                for i in 0..n {
                    let log_s = log_std_vec[i].clamp(-2.0, 0.5); // clamp for stability
                    let std = log_s.exp();
                    let mean = output[i]; // raw output, tanh applied below
                    let noise = rng.randn() * std;
                    let raw_a = mean + noise;
                    let a = raw_a.tanh(); // squash to [-1, 1]
                    // FIX: Squashed Gaussian log-prob (tanh correction)
                    // log π(a|s) = log N(raw_a; mean, std) - log(1 - tanh²(raw_a))
                    let gauss_lp = -0.5 * ((raw_a - mean) / std).powi(2) - log_s - 0.5 * (2.0 * std::f64::consts::PI).ln();
                    let tanh_correction = (1.0 - a * a + 1e-6).ln();
                    log_prob += gauss_lp - tanh_correction;
                    actions.push(a as f32);
                }
                (Action::Continuous(actions), log_prob)
            }
        }
    }

    /// Compute log probability of an action given output.
    fn log_prob(&self, output: &[f64], action: &Action) -> f64 {
        match (&self.action_space, action) {
            (ActionSpace::Discrete(_), Action::Discrete(a)) => {
                let max_v = output.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp: Vec<f64> = output.iter().map(|v| (v - max_v).exp()).collect();
                let sum: f64 = exp.iter().sum();
                let probs: Vec<f64> = exp.iter().map(|v| v / sum).collect();
                probs[*a].max(1e-10).ln()
            }
            (ActionSpace::Continuous(_n), Action::Continuous(actions)) => {
                let log_std_vec = self.log_std.as_ref().unwrap();
                let mut lp = 0.0;
                for (i, &a) in actions.iter().enumerate() {
                    let log_s = log_std_vec[i].clamp(-2.0, 0.5);
                    let std = log_s.exp();
                    let mean = output[i];
                    // Inverse tanh to recover raw_a from squashed action
                    let a64 = (a as f64).clamp(-0.999, 0.999);
                    let raw_a = 0.5 * ((1.0 + a64) / (1.0 - a64)).ln(); // atanh
                    let gauss_lp = -0.5 * ((raw_a - mean) / std).powi(2) - log_s - 0.5 * (2.0 * std::f64::consts::PI).ln();
                    let tanh_correction = (1.0 - a64 * a64 + 1e-6).ln();
                    lp += gauss_lp - tanh_correction;
                }
                lp
            }
            _ => 0.0,
        }
    }

    /// Compute gradient of log_prob w.r.t. network output.
    fn d_log_prob(&self, output: &[f64], action: &Action) -> Vec<f64> {
        match (&self.action_space, action) {
            (ActionSpace::Discrete(_n), Action::Discrete(a)) => {
                let max_v = output.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp: Vec<f64> = output.iter().map(|v| (v - max_v).exp()).collect();
                let sum: f64 = exp.iter().sum();
                let probs: Vec<f64> = exp.iter().map(|v| v / sum).collect();
                // d log_prob / d logit_j = (1{j==a} - probs[j])
                let mut d = vec![0.0; output.len()];
                for j in 0..output.len() {
                    d[j] = if j == *a { 1.0 - probs[j] } else { -probs[j] };
                }
                d
            }
            (ActionSpace::Continuous(n), Action::Continuous(actions)) => {
                let log_std_vec = self.log_std.as_ref().unwrap();
                let mut d = vec![0.0; *n];
                for i in 0..*n {
                    let log_s = log_std_vec[i].clamp(-2.0, 0.5);
                    let std = log_s.exp();
                    let mean = output[i];
                    let a64 = (actions[i] as f64).clamp(-0.999, 0.999);
                    let raw_a = 0.5 * ((1.0 + a64) / (1.0 - a64)).ln(); // atanh
                    // d log_prob / d mean = (raw_a - mean) / std²
                    d[i] = (raw_a - mean) / (std * std);
                }
                d
            }
            _ => vec![0.0; output.len()],
        }
    }
}

struct ForwardResult {
    output: Vec<f64>,
    pre_activations: Vec<Vec<f64>>,
    post_activations: Vec<Vec<f64>>,
}

// ─── Value Network (same architecture) ──────────────────────────────

struct ValueNet {
    inner: DiffPolicy,
}

impl ValueNet {
    fn new(obs_dim: usize, hidden: &[usize]) -> Self {
        // Value net outputs single scalar
        ValueNet {
            inner: DiffPolicy::new(obs_dim, 1, hidden, ActionSpace::Discrete(1)),
        }
    }

    fn predict(&self, obs: &[f64]) -> f64 {
        self.inner.forward(obs).output[0]
    }

    #[allow(dead_code)]
    fn train_step(&mut self, obs: &[f64], target: f64, adam: &mut Adam) -> f64 {
        let fwd = self.inner.forward(obs);
        let prediction = fwd.output[0];
        let error = prediction - target;
        let loss = error * error;

        // d_loss/d_output = 2 * error
        let d_output = vec![2.0 * error];
        let (d_weights, d_biases) = self.inner.backward(&fwd, &d_output);
        self.inner.apply_gradients(&d_weights, &d_biases, None, adam);

        loss
    }
}

// ─── Adam Optimizer ─────────────────────────────────────────────────

struct Adam {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    m: Vec<f64>,
    v: Vec<f64>,
    t: usize,
}

impl Adam {
    fn new(n_params: usize, lr: f64) -> Self {
        Adam {
            lr, beta1: 0.9, beta2: 0.999, eps: 1e-8,
            m: vec![0.0; n_params],
            v: vec![0.0; n_params],
            t: 0,
        }
    }

    fn step(&mut self, grads: &[f64]) -> Vec<f64> {
        self.t += 1;
        let mut updates = vec![0.0; grads.len()];
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..grads.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grads[i] * grads[i];
            let m_hat = self.m[i] / bias_correction1;
            let v_hat = self.v[i] / bias_correction2;
            updates[i] = self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
        updates
    }
}

// ─── Rollout Buffer ─────────────────────────────────────────────────

struct Transition {
    obs: Vec<f64>,
    action: Action,
    log_prob: f64,
    reward: f64,
    value: f64,
    done: bool,
}

fn compute_gae(transitions: &[Transition], gamma: f64, lambda: f64, last_value: f64) -> (Vec<f64>, Vec<f64>) {
    let n = transitions.len();
    let mut advantages = vec![0.0; n];
    let mut returns = vec![0.0; n];
    let mut gae = 0.0;

    for t in (0..n).rev() {
        let next_value = if t + 1 < n && !transitions[t].done {
            transitions[t + 1].value
        } else if !transitions[t].done {
            last_value
        } else {
            0.0
        };
        let delta = transitions[t].reward + gamma * next_value - transitions[t].value;
        let mask = if transitions[t].done { 0.0 } else { 1.0 };
        gae = delta + gamma * lambda * mask * gae;
        advantages[t] = gae;
        returns[t] = advantages[t] + transitions[t].value;
    }

    // Normalize advantages
    let mean = advantages.iter().sum::<f64>() / n as f64;
    let std = (advantages.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / n as f64).sqrt().max(1e-8);
    for a in &mut advantages {
        *a = (*a - mean) / std;
    }

    (advantages, returns)
}

// ─── PPO Training ───────────────────────────────────────────────────

pub fn run_ppo(env_name: &str, params: &Value, mut on_gen: impl FnMut(GenResult)) -> RunResult {
    let env_cfg = env::get_env_config(env_name).expect("Unknown env");
    let hidden = if let Some(h) = params.get("hidden").and_then(|v| v.as_array()) {
        h.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect()
    } else {
        env::default_hidden(env_name)
    };
    let max_evals = params.get("max_evals").and_then(|v| v.as_u64()).unwrap_or(100000) as usize;
    let max_steps = env_cfg.max_steps;
    let solved = env_cfg.solved_threshold;

    // PPO hyperparameters
    let rollout_steps = params.get("rollout_steps").and_then(|v| v.as_u64()).unwrap_or(2048) as usize;
    let n_epochs = params.get("n_epochs").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
    let minibatch_size = params.get("minibatch_size").and_then(|v| v.as_u64()).unwrap_or(64) as usize;
    let clip_eps = params.get("clip_eps").and_then(|v| v.as_f64()).unwrap_or(0.2);
    let gamma = params.get("gamma").and_then(|v| v.as_f64()).unwrap_or(0.99);
    let lambda = params.get("lambda").and_then(|v| v.as_f64()).unwrap_or(0.95);
    let lr = params.get("lr").and_then(|v| v.as_f64()).unwrap_or(3e-4);
    let vf_coeff = params.get("vf_coeff").and_then(|v| v.as_f64()).unwrap_or(0.5);
    let ent_coeff = params.get("ent_coeff").and_then(|v| v.as_f64()).unwrap_or(0.01);

    let obs_dim = env_cfg.obs_dim;
    let act_dim = env_cfg.action_space.size();

    let mut policy = DiffPolicy::new(obs_dim, act_dim, &hidden, env_cfg.action_space);
    let mut value_net = ValueNet::new(obs_dim, &hidden);
    let mut policy_adam = Adam::new(policy.n_params(), lr);
    let mut value_adam = Adam::new(value_net.inner.n_params(), lr);
    let mut rng = Rng::new(42);

    eprintln!("🦀⚡ PPO on {} | {} policy params | {} value params | rollout={} | lr={}",
        env_name, policy.n_params(), value_net.inner.n_params(), rollout_steps, lr);

    let mut env = env::make(env_name, Some(0)).expect("Unknown env");
    let mut obs = env.reset(Some(0)).iter().map(|&v| v as f64).collect::<Vec<_>>();
    let mut best_ever = f64::NEG_INFINITY;
    let mut total_evals = 0usize;
    let mut gen = 0usize;
    let mut ep_reward = 0.0;
    let mut ep_step = 0usize;
    let mut recent_rewards: Vec<f64> = Vec::new();
    let start = Instant::now();

    while total_evals < max_evals {
        gen += 1;

        // ── Collect rollout ──
        let mut transitions = Vec::with_capacity(rollout_steps);
        for _ in 0..rollout_steps {
            let fwd = policy.forward(&obs);
            let value = value_net.predict(&obs);
            let (action, log_prob) = policy.sample_action(&fwd.output, &mut rng);

            let result = env.step(&action);
            let reward = result.reward as f64;
            let done = result.done();
            ep_reward += reward;
            ep_step += 1;

            transitions.push(Transition {
                obs: obs.clone(),
                action,
                log_prob,
                reward,
                value,
                done,
            });

            total_evals += 1;

            if done || ep_step >= max_steps {
                recent_rewards.push(ep_reward);
                if recent_rewards.len() > 20 { recent_rewards.remove(0); }
                if ep_reward > best_ever { best_ever = ep_reward; }
                ep_reward = 0.0;
                ep_step = 0;
                obs = env.reset(Some(total_evals as u64)).iter().map(|&v| v as f64).collect();
            } else {
                obs = result.observation.iter().map(|&v| v as f64).collect();
            }
        }

        // ── Compute GAE ──
        let last_value = value_net.predict(&obs);
        let (advantages, returns) = compute_gae(&transitions, gamma, lambda, last_value);

        // ── PPO update epochs ──
        let n = transitions.len();
        for _epoch in 0..n_epochs {
            // Shuffle indices
            let mut indices: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = (rng.next_u64() as usize) % (i + 1);
                indices.swap(i, j);
            }

            // Mini-batches
            for batch_start in (0..n).step_by(minibatch_size) {
                let batch_end = (batch_start + minibatch_size).min(n);
                let batch_indices = &indices[batch_start..batch_end];

                // Accumulate policy gradients
                let mut total_d_weights: Vec<Vec<f64>> = policy.weights.iter()
                    .map(|w| vec![0.0; w.len()]).collect();
                let mut total_d_biases: Vec<Vec<f64>> = policy.biases.iter()
                    .map(|b| vec![0.0; b.len()]).collect();
                let n_log_std = policy.log_std.as_ref().map_or(0, |v| v.len());
                let mut total_d_log_std = vec![0.0; n_log_std];
                let batch_size = batch_indices.len() as f64;

                for &idx in batch_indices {
                    let t = &transitions[idx];

                    // New log prob
                    let fwd = policy.forward(&t.obs);
                    let new_log_prob = policy.log_prob(&fwd.output, &t.action);

                    // PPO ratio and clipped objective
                    let ratio = (new_log_prob - t.log_prob).exp();
                    let surr1 = ratio * advantages[idx];
                    let surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages[idx];

                    let use_surr1 = surr1 <= surr2;
                    let clip_grad = if use_surr1 {
                        ratio * advantages[idx]
                    } else {
                        0.0
                    };

                    // Gradient of log_prob w.r.t. output (for network weights)
                    let d_log_prob = policy.d_log_prob(&fwd.output, &t.action);

                    // Entropy gradient
                    let d_entropy = compute_d_entropy(&fwd.output, &policy.action_space, &policy.log_std);

                    // Combined gradient for network output
                    let d_output: Vec<f64> = (0..fwd.output.len()).map(|i| {
                        (-clip_grad * d_log_prob[i] - ent_coeff * d_entropy[i]) / batch_size
                    }).collect();

                    let (dw, db) = policy.backward(&fwd, &d_output);
                    for l in 0..dw.len() {
                        for i in 0..dw[l].len() { total_d_weights[l][i] += dw[l][i]; }
                        for i in 0..db[l].len() { total_d_biases[l][i] += db[l][i]; }
                    }

                    // FIX: Gradient for learnable log_std (continuous only)
                    if n_log_std > 0 {
                        if let (ActionSpace::Continuous(_), Action::Continuous(actions)) = (&policy.action_space, &t.action) {
                            let log_std_vec = policy.log_std.as_ref().unwrap();
                            for i in 0..n_log_std {
                                let log_s = log_std_vec[i].clamp(-2.0, 0.5);
                                let std = log_s.exp();
                                let mean = fwd.output[i];
                                let a64 = (actions[i] as f64).clamp(-0.999, 0.999);
                                let raw_a = 0.5 * ((1.0 + a64) / (1.0 - a64)).ln();
                                // d log_prob / d log_std = (raw_a - mean)²/std² - 1
                                let d_lp_d_ls = ((raw_a - mean) / std).powi(2) - 1.0;
                                // d entropy / d log_std = 1 (Gaussian entropy = log_std + const)
                                let d_ent_d_ls = 1.0;
                                total_d_log_std[i] += (-clip_grad * d_lp_d_ls - ent_coeff * d_ent_d_ls) / batch_size;
                            }
                        }
                    }

                    // FIX: Value function update with vf_coeff
                    let fwd_v = value_net.inner.forward(&t.obs);
                    let prediction = fwd_v.output[0];
                    let error = prediction - returns[idx];
                    let d_output_v = vec![2.0 * vf_coeff * error];
                    let (dw_v, db_v) = value_net.inner.backward(&fwd_v, &d_output_v);
                    value_net.inner.apply_gradients(&dw_v, &db_v, None, &mut value_adam);
                }

                let d_ls = if n_log_std > 0 { Some(total_d_log_std.as_slice()) } else { None };
                policy.apply_gradients(&total_d_weights, &total_d_biases, d_ls, &mut policy_adam);
            }
        }

        // ── Report ──
        let mean_reward = if recent_rewards.is_empty() { 0.0 }
            else { recent_rewards.iter().sum::<f64>() / recent_rewards.len() as f64 };

        on_gen(GenResult {
            generation: gen,
            best: recent_rewards.last().copied().unwrap_or(0.0),
            best_ever,
            mean: mean_reward,
            sigma: policy.log_std.as_ref().map_or(0.0, |ls| ls.iter().map(|v| v.exp()).sum::<f64>() / ls.len() as f64),
            evals: total_evals,
            time: start.elapsed().as_secs_f64(),
        });

        if best_ever >= solved { break; }
    }

    // Final evaluation (deterministic — use mean action)
    let mut final_scores = Vec::new();
    for ep in 0..20 {
        let mut env = env::make(env_name, Some(ep as u64 * 9999)).unwrap();
        let mut obs = env.reset(Some(ep as u64 * 9999)).iter().map(|&v| v as f64).collect::<Vec<_>>();
        let mut ep_reward = 0.0;
        for _ in 0..max_steps {
            let fwd = policy.forward(&obs);
            // Deterministic: argmax for discrete, tanh(mean) for continuous
            let action = match policy.action_space {
                ActionSpace::Discrete(_) => {
                    let best = fwd.output.iter().enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i).unwrap_or(0);
                    Action::Discrete(best)
                }
                ActionSpace::Continuous(n) => {
                    let actions: Vec<f32> = (0..n).map(|i| fwd.output[i].tanh() as f32).collect();
                    Action::Continuous(actions)
                }
            };
            let result = env.step(&action);
            ep_reward += result.reward as f64;
            if result.done() { break; }
            obs = result.observation.iter().map(|&v| v as f64).collect();
        }
        final_scores.push(ep_reward);
    }
    let fm = final_scores.iter().sum::<f64>() / 20.0;
    let fs = (final_scores.iter().map(|x| (x - fm).powi(2)).sum::<f64>() / 20.0).sqrt();

    RunResult {
        method: "PPO".into(),
        environment: env_name.into(),
        best_ever,
        final_mean: fm,
        final_std: fs,
        total_evals,
        generations: gen,
        elapsed: start.elapsed().as_secs_f64(),
        solved: best_ever >= solved,
    }
}

/// Gradient of entropy w.r.t. logits (for backpropagation).
/// Returns d(H)/d(output) — positive direction increases entropy.
fn compute_d_entropy(output: &[f64], action_space: &ActionSpace, _log_std: &Option<Vec<f64>>) -> Vec<f64> {
    match action_space {
        ActionSpace::Discrete(_) => {
            let max_v = output.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp: Vec<f64> = output.iter().map(|v| (v - max_v).exp()).collect();
            let sum: f64 = exp.iter().sum();
            let probs: Vec<f64> = exp.iter().map(|v| v / sum).collect();
            let weighted_sum: f64 = probs.iter()
                .map(|p| p * (p.max(1e-10).ln() + 1.0))
                .sum();
            probs.iter().map(|p| {
                -p * (p.max(1e-10).ln() + 1.0) + p * weighted_sum
            }).collect()
        }
        ActionSpace::Continuous(n) => {
            // FIX: Gaussian entropy w.r.t. output (mean).
            // H = 0.5 * ln(2πe) + log_std — does NOT depend on mean (output).
            // The mean doesn't affect entropy; entropy gradient for log_std is
            // handled separately in the training loop.
            // So d(H)/d(output) = 0 is actually correct for Gaussian!
            // But we include squash correction: effective entropy includes tanh,
            // which depends on mean. d(entropy_correction)/d(mean) ≈ 0 in expectation.
            vec![0.0; *n]
        }
    }
}
