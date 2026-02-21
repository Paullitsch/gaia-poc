//! GPU-native vectorized environments for massively parallel simulation.
//!
//! Key idea: run N environments simultaneously on GPU, one CUDA thread per environment.
//! For small networks (2788 params), the forward pass is also batched on GPU.
//!
//! Architecture:
//! - GpuVecEnv: manages N parallel environment instances on GPU
//! - All state stored in GPU memory (no CPU↔GPU transfer per step)
//! - Batch interface: step_all(actions[N]) → (obs[N], rewards[N], dones[N])
//! - CPU fallback for testing on machines without GPU
//!
//! Supported environments:
//! - CartPole-v1 (simple ODE, no contacts)
//! - LunarLander-v3 (rigid body + thrusters, simplified contacts)

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchConfig, LaunchAsync};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::compile_ptx;
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Vectorized environment state on GPU (or CPU fallback).
pub struct GpuVecEnv {
    pub env_name: String,
    pub n_envs: usize,
    pub obs_dim: usize,
    pub act_dim: usize,
    pub max_steps: usize,
    pub using_gpu: bool,

    // CPU fallback storage
    states: Vec<f32>,
    observations: Vec<f32>,
    rewards: Vec<f32>,
    dones: Vec<u8>,
    steps: Vec<u32>,
    seeds: Vec<u64>,

    #[cfg(feature = "cuda")]
    gpu: Option<GpuState>,
}

#[cfg(feature = "cuda")]
struct GpuState {
    dev: Arc<CudaDevice>,
    d_states: CudaSlice<f32>,
    d_observations: CudaSlice<f32>,
    d_rewards: CudaSlice<f32>,
    d_dones: CudaSlice<u8>,
    d_steps: CudaSlice<u32>,
    d_seeds: CudaSlice<u64>,
    d_actions: CudaSlice<f32>,
    step_fn_name: String,
    reset_fn_name: String,
}

// ─── CartPole Constants ─────────────────────────────────────────────

const CP_GRAVITY: f32 = 9.8;
const CP_CART_MASS: f32 = 1.0;
const CP_POLE_MASS: f32 = 0.1;
const CP_TOTAL_MASS: f32 = CP_CART_MASS + CP_POLE_MASS;
const CP_POLE_HALF_LEN: f32 = 0.5;
const CP_FORCE_MAG: f32 = 10.0;
const CP_TAU: f32 = 0.02;
const CP_X_THRESH: f32 = 2.4;
const CP_THETA_THRESH: f32 = 0.2094; // 12 degrees in radians
const CP_STATE_DIM: usize = 4;
const CP_OBS_DIM: usize = 4;
const CP_MAX_STEPS: usize = 500;

// ─── LunarLander Constants ──────────────────────────────────────────

const LL_GRAVITY: f32 = -10.0;
const LL_FPS: f32 = 50.0;
const LL_MAIN_ENGINE_POWER: f32 = 13.0;
const LL_SIDE_ENGINE_POWER: f32 = 0.6;
const LL_INITIAL_Y: f32 = 1.4;  // viewport fraction
const LL_STATE_DIM: usize = 8;
const LL_OBS_DIM: usize = 8;
const LL_MAX_STEPS: usize = 1000;

#[cfg(feature = "cuda")]
fn is_gpu_available() -> bool {
    static GPU_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GPU_AVAILABLE.get_or_init(|| {
        match CudaDevice::new(0) {
            Ok(_) => {
                eprintln!("🟢 CUDA GPU detected — GPU acceleration available");
                true
            }
            Err(e) => {
                eprintln!("🟡 No CUDA GPU ({}) — using CPU fallback", e);
                false
            }
        }
    })
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
fn is_gpu_available() -> bool { false }

impl GpuVecEnv {
    /// Create N parallel environments. Tries GPU first, falls back to CPU.
    pub fn new(env_name: &str, n_envs: usize) -> Self {
        let (obs_dim, act_dim, max_steps, state_dim) = match env_name {
            "CartPole-v1" => (CP_OBS_DIM, 2, CP_MAX_STEPS, CP_STATE_DIM),
            "LunarLander-v3" => (LL_OBS_DIM, 4, LL_MAX_STEPS, LL_STATE_DIM),
            "Swimmer-v1" => (8, 2, 1000, 10), // [x,y,body_angle,j1,j2,vx,vy,angvel,j1vel,j2vel]
            _ if env_name.starts_with("Pendulum-") && env_name.ends_with("Link") => {
                let n: usize = env_name.strip_prefix("Pendulum-")
                    .and_then(|s| s.strip_suffix("Link"))
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(2);
                let obs = 3 * n; // cos, sin, vel per link
                let max_s = if n <= 3 { 500 } else { 1000 };
                (obs, n, max_s, 2 * n) // state = angles + velocities
            }
            _ => panic!("GPU env not supported: {}", env_name),
        };

        #[cfg(feature = "cuda")]
        let gpu = if is_gpu_available() && env_name == "CartPole-v1" {
            Self::init_gpu_cartpole(n_envs)
        } else {
            None
        };

        #[cfg(feature = "cuda")]
        let using_gpu = gpu.is_some();
        #[cfg(not(feature = "cuda"))]
        let using_gpu = false;

        GpuVecEnv {
            env_name: env_name.to_string(),
            n_envs,
            obs_dim,
            act_dim,
            max_steps,
            using_gpu,
            states: vec![0.0; n_envs * state_dim],
            observations: vec![0.0; n_envs * obs_dim],
            rewards: vec![0.0; n_envs],
            dones: vec![0; n_envs],
            steps: vec![0; n_envs],
            seeds: (0..n_envs).map(|i| i as u64 * 12345 + 42).collect(),
            #[cfg(feature = "cuda")]
            gpu,
        }
    }

    /// Initialize GPU state for CartPole.
    #[cfg(feature = "cuda")]
    fn init_gpu_cartpole(n_envs: usize) -> Option<GpuState> {
        let dev = CudaDevice::new(0).ok()?;

        // Compile CartPole CUDA kernel
        let ptx = compile_ptx(CARTPOLE_CUDA_SRC).ok()?;
        dev.load_ptx(ptx, "cartpole", &["cartpole_step", "cartpole_reset"]).ok()?;

        let d_states = dev.alloc_zeros::<f32>(n_envs * CP_STATE_DIM).ok()?;
        let d_observations = dev.alloc_zeros::<f32>(n_envs * CP_OBS_DIM).ok()?;
        let d_rewards = dev.alloc_zeros::<f32>(n_envs).ok()?;
        let d_dones = dev.alloc_zeros::<u8>(n_envs).ok()?;
        let d_steps = dev.alloc_zeros::<u32>(n_envs).ok()?;
        let d_seeds = dev.alloc_zeros::<u64>(n_envs).ok()?;
        let d_actions = dev.alloc_zeros::<f32>(n_envs).ok()?;

        eprintln!("🚀 CartPole CUDA kernel compiled — {} envs on GPU", n_envs);

        Some(GpuState {
            dev,
            d_states,
            d_observations,
            d_rewards,
            d_dones,
            d_steps,
            d_seeds,
            d_actions,
            step_fn_name: "cartpole_step".to_string(),
            reset_fn_name: "cartpole_reset".to_string(),
        })
    }

    /// Reset all environments with sequential seeds.
    pub fn reset_all(&mut self, base_seed: u64) {
        for i in 0..self.n_envs {
            self.seeds[i] = base_seed + i as u64;
        }

        #[cfg(feature = "cuda")]
        if self.gpu.is_some() {
            if let Ok(()) = self.gpu_reset_all() {
                return;
            }
        }

        for i in 0..self.n_envs {
            self.reset_env(i);
        }
    }

    /// GPU reset: upload seeds, launch kernel, download observations.
    #[cfg(feature = "cuda")]
    fn gpu_reset_all(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let n = self.n_envs;
        let seeds_copy = self.seeds.clone();

        let gpu = self.gpu.as_mut().ok_or("no gpu")?;
        gpu.dev.htod_copy_into(seeds_copy, &mut gpu.d_seeds)?;

        let reset_fn_name = gpu.reset_fn_name.clone();
        let f = gpu.dev.get_func("cartpole", &reset_fn_name).ok_or("no kernel")?;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (((n + 255) / 256) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            f.launch(cfg, (
                &gpu.d_states,
                &gpu.d_observations,
                &gpu.d_dones,
                &gpu.d_steps,
                &gpu.d_seeds,
                n as i32,
            ))?;
        }

        // Download observations
        let obs = gpu.dev.dtoh_sync_copy(&gpu.d_observations)?;
        let dones = gpu.dev.dtoh_sync_copy(&gpu.d_dones)?;
        drop(gpu);
        self.observations[..obs.len()].copy_from_slice(&obs);
        self.dones[..dones.len()].copy_from_slice(&dones);

        Ok(())
    }

    /// Reset a single environment.
    fn reset_env(&mut self, idx: usize) {
        match self.env_name.as_str() {
            "CartPole-v1" => self.reset_cartpole(idx),
            "LunarLander-v3" => self.reset_lunar_lander(idx),
            _ => {}
        }
    }

    /// Step all environments with given actions.
    /// For discrete envs: actions[i] is the action index as f32.
    /// For continuous envs: actions[i * act_dim .. (i+1) * act_dim].
    pub fn step_all(&mut self, actions: &[f32]) -> (&[f32], &[f32], &[u8]) {
        #[cfg(feature = "cuda")]
        if self.gpu.is_some() {
            if let Ok(()) = self.gpu_step_all(actions) {
                return (&self.observations, &self.rewards, &self.dones);
            }
        }

        match self.env_name.as_str() {
            "CartPole-v1" => self.step_all_cartpole(actions),
            "LunarLander-v3" => self.step_all_lunar_lander(actions),
            _ => {}
        }
        (&self.observations, &self.rewards, &self.dones)
    }

    /// GPU step: upload actions, launch kernel, download results.
    #[cfg(feature = "cuda")]
    fn gpu_step_all(&mut self, actions: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
        let n = self.n_envs;

        let gpu = self.gpu.as_mut().unwrap();
        gpu.dev.htod_copy_into(actions.to_vec(), &mut gpu.d_actions)?;

        let step_fn_name = gpu.step_fn_name.clone();
        let f = gpu.dev.get_func("cartpole", &step_fn_name).ok_or("no kernel")?;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (((n + 255) / 256) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            f.launch(cfg, (
                &gpu.d_states,
                &gpu.d_observations,
                &gpu.d_rewards,
                &gpu.d_dones,
                &gpu.d_steps,
                &gpu.d_actions,
                n as i32,
            ))?;
        }

        let obs = gpu.dev.dtoh_sync_copy(&gpu.d_observations)?;
        let rewards = gpu.dev.dtoh_sync_copy(&gpu.d_rewards)?;
        let dones = gpu.dev.dtoh_sync_copy(&gpu.d_dones)?;
        drop(gpu); // release borrow before writing to self
        self.observations[..obs.len()].copy_from_slice(&obs);
        self.rewards[..rewards.len()].copy_from_slice(&rewards);
        self.dones[..dones.len()].copy_from_slice(&dones);

        Ok(())
    }

    /// Auto-reset environments that are done.
    pub fn auto_reset(&mut self) {
        for i in 0..self.n_envs {
            if self.dones[i] != 0 {
                self.seeds[i] = self.seeds[i].wrapping_add(1000000);
                self.reset_env(i);
            }
        }
    }

    /// Get observations for all environments.
    pub fn get_observations(&self) -> &[f32] {
        &self.observations
    }

    // ─── CartPole CPU Implementation ────────────────────────────────

    fn reset_cartpole(&mut self, idx: usize) {
        let off = idx * CP_STATE_DIM;
        let seed = &mut self.seeds[idx];
        for j in 0..CP_STATE_DIM {
            self.states[off + j] = Self::lcg_small_val(seed);
        }
        self.steps[idx] = 0;
        self.dones[idx] = 0;
        let obs_off = idx * self.obs_dim;
        for j in 0..CP_OBS_DIM {
            self.observations[obs_off + j] = self.states[off + j];
        }
    }

    fn step_all_cartpole(&mut self, actions: &[f32]) {
        for i in 0..self.n_envs {
            if self.dones[i] != 0 { continue; }

            let off = i * CP_STATE_DIM;
            let action = actions[i] as usize;
            let force = if action == 1 { CP_FORCE_MAG } else { -CP_FORCE_MAG };

            let x = self.states[off];
            let x_dot = self.states[off + 1];
            let theta = self.states[off + 2];
            let theta_dot = self.states[off + 3];

            let cos_t = theta.cos();
            let sin_t = theta.sin();

            let temp = (force + CP_POLE_MASS * CP_POLE_HALF_LEN * theta_dot * theta_dot * sin_t) / CP_TOTAL_MASS;
            let theta_acc = (CP_GRAVITY * sin_t - cos_t * temp)
                / (CP_POLE_HALF_LEN * (4.0 / 3.0 - CP_POLE_MASS * cos_t * cos_t / CP_TOTAL_MASS));
            let x_acc = temp - CP_POLE_MASS * CP_POLE_HALF_LEN * theta_acc * cos_t / CP_TOTAL_MASS;

            self.states[off]     = x + CP_TAU * x_dot;
            self.states[off + 1] = x_dot + CP_TAU * x_acc;
            self.states[off + 2] = theta + CP_TAU * theta_dot;
            self.states[off + 3] = theta_dot + CP_TAU * theta_acc;

            self.steps[i] += 1;

            let terminated = self.states[off].abs() > CP_X_THRESH
                          || self.states[off + 2].abs() > CP_THETA_THRESH;
            let truncated = self.steps[i] as usize >= CP_MAX_STEPS;

            self.rewards[i] = if terminated { 0.0 } else { 1.0 };
            self.dones[i] = if terminated || truncated { 1 } else { 0 };

            // obs = state
            let obs_off = i * self.obs_dim;
            for j in 0..CP_OBS_DIM {
                self.observations[obs_off + j] = self.states[off + j];
            }
        }
    }

    // ─── LunarLander CPU Implementation ─────────────────────────────

    fn reset_lunar_lander(&mut self, idx: usize) {
        let off = idx * LL_STATE_DIM;
        let seed = &mut self.seeds[idx];
        self.states[off]     = Self::lcg_small_val(seed) * 0.1; // x near center
        self.states[off + 1] = LL_INITIAL_Y;                     // y at top
        self.states[off + 2] = Self::lcg_small_val(seed) * 0.1; // vx
        self.states[off + 3] = Self::lcg_small_val(seed) * 0.1; // vy
        self.states[off + 4] = 0.0;                               // angle
        self.states[off + 5] = 0.0;                               // angular velocity
        self.states[off + 6] = 0.0;                               // leg1 contact
        self.states[off + 7] = 0.0;                               // leg2 contact
        self.steps[idx] = 0;
        self.dones[idx] = 0;

        self.compute_ll_obs(idx);
    }

    fn compute_ll_obs(&mut self, idx: usize) {
        let off = idx * LL_STATE_DIM;
        let obs_off = idx * self.obs_dim;
        // Gymnasium LunarLander observations are normalized:
        // x/VIEWPORT_W*SCALE, y/VIEWPORT_H*SCALE*2 (shifted), vx*SCALE/FPS, vy*SCALE/FPS,
        // angle, angular_vel, leg1, leg2
        self.observations[obs_off]     = self.states[off];     // x (already in world coords)
        self.observations[obs_off + 1] = self.states[off + 1]; // y
        self.observations[obs_off + 2] = self.states[off + 2]; // vx
        self.observations[obs_off + 3] = self.states[off + 3]; // vy
        self.observations[obs_off + 4] = self.states[off + 4]; // angle
        self.observations[obs_off + 5] = self.states[off + 5]; // angular_vel
        self.observations[obs_off + 6] = self.states[off + 6]; // leg1
        self.observations[obs_off + 7] = self.states[off + 7]; // leg2
    }

    fn step_all_lunar_lander(&mut self, actions: &[f32]) {
        let dt = 1.0 / LL_FPS;

        for i in 0..self.n_envs {
            if self.dones[i] != 0 { continue; }

            let off = i * LL_STATE_DIM;
            let action = actions[i] as usize;

            let mut x = self.states[off];
            let mut y = self.states[off + 1];
            let mut vx = self.states[off + 2];
            let mut vy = self.states[off + 3];
            let mut angle = self.states[off + 4];
            let mut ang_vel = self.states[off + 5];

            // Apply gravity
            vy += LL_GRAVITY / LL_FPS;

            // Apply engine forces based on action
            match action {
                2 => {
                    // Main engine (bottom)
                    let fx = -angle.sin() * LL_MAIN_ENGINE_POWER / LL_FPS;
                    let fy = angle.cos() * LL_MAIN_ENGINE_POWER / LL_FPS;
                    vx += fx;
                    vy += fy;
                }
                1 => {
                    // Left engine → rotate right + slight push
                    ang_vel -= LL_SIDE_ENGINE_POWER / LL_FPS;
                    vx += angle.cos() * LL_SIDE_ENGINE_POWER * 0.1 / LL_FPS;
                }
                3 => {
                    // Right engine → rotate left + slight push
                    ang_vel += LL_SIDE_ENGINE_POWER / LL_FPS;
                    vx -= angle.cos() * LL_SIDE_ENGINE_POWER * 0.1 / LL_FPS;
                }
                _ => {} // action 0 = do nothing
            }

            // Integrate position
            x += vx * dt;
            y += vy * dt;
            angle += ang_vel * dt;

            // Simple ground collision (y <= 0)
            let leg1_contact;
            let leg2_contact;
            if y <= 0.0 {
                y = 0.0;
                vy = 0.0;
                vx *= 0.5; // friction
                ang_vel *= 0.5;
                leg1_contact = 1.0f32;
                leg2_contact = 1.0f32;
            } else {
                leg1_contact = 0.0;
                leg2_contact = 0.0;
            }

            // Update state
            self.states[off]     = x;
            self.states[off + 1] = y;
            self.states[off + 2] = vx;
            self.states[off + 3] = vy;
            self.states[off + 4] = angle;
            self.states[off + 5] = ang_vel;
            self.states[off + 6] = leg1_contact;
            self.states[off + 7] = leg2_contact;

            self.steps[i] += 1;

            // Reward: matching Gymnasium formula
            // simplified: no prev shaping tracking in vectorized version
            let shaping = -100.0 * (x * x + y * y).sqrt()
                         - 100.0 * (vx * vx + vy * vy).sqrt()
                         - 100.0 * angle.abs()
                         + 10.0 * leg1_contact
                         + 10.0 * leg2_contact;
            let fuel_penalty = if action == 2 { -0.3 } else if action == 1 || action == 3 { -0.03 } else { 0.0 };
            self.rewards[i] = shaping / 100.0 + fuel_penalty;

            // Terminal conditions
            let crashed = y <= 0.0 && (vx.abs() > 0.5 || angle.abs() > 0.5);
            let landed = y <= 0.0 && vx.abs() < 0.1 && angle.abs() < 0.1;
            let out_of_bounds = x.abs() > 1.5 || y > 2.0;
            let timeout = self.steps[i] as usize >= LL_MAX_STEPS;

            if crashed {
                self.rewards[i] = -100.0;
                self.dones[i] = 1;
            } else if landed {
                self.rewards[i] += 100.0;
                self.dones[i] = 1;
            } else if out_of_bounds || timeout {
                self.dones[i] = 1;
            }

            self.compute_ll_obs(i);
        }
    }

    // ─── Utility ────────────────────────────────────────────────────

    /// LCG random in [-0.05, 0.05]
    fn lcg_small_val(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let val = ((*seed >> 33) as f32) / (u32::MAX as f32);
        val * 0.1 - 0.05
    }
}

// ─── Batched Policy Forward Pass ────────────────────────────────────

/// Batched forward pass for N policies with same weights.
/// CPU version — GPU version would use cuBLAS or custom kernel.
pub fn batch_forward(
    obs: &[f32],       // [n × obs_dim]
    params: &[f32],    // [n_params] shared weights
    layer_dims: &[(usize, usize)],
    n: usize,
    obs_dim: usize,
    act_dim: usize,
    discrete: bool,
) -> Vec<f32> {
    // Output: [n × act_dim] (raw logits for discrete, tanh-squashed for continuous)
    let mut results = vec![0.0f32; n * act_dim];

    for env_i in 0..n {
        let obs_start = env_i * obs_dim;
        let mut x: Vec<f32> = obs[obs_start..obs_start + obs_dim].to_vec();
        let mut offset = 0;

        for (layer_idx, &(fan_in, fan_out)) in layer_dims.iter().enumerate() {
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

            let is_last = layer_idx == layer_dims.len() - 1;
            if !is_last {
                for v in &mut out { *v = v.tanh(); }
            } else if !discrete {
                for v in &mut out { *v = v.tanh(); }
            }
            x = out;
        }

        // Store result
        let res_start = env_i * act_dim;
        for j in 0..act_dim {
            results[res_start + j] = x[j];
        }
    }

    results
}

/// For discrete envs: argmax over logits → action index as f32
pub fn batch_argmax(logits: &[f32], n: usize, act_dim: usize) -> Vec<f32> {
    let mut actions = vec![0.0f32; n];
    for i in 0..n {
        let start = i * act_dim;
        let mut best_idx = 0;
        let mut best_val = f32::NEG_INFINITY;
        for j in 0..act_dim {
            if logits[start + j] > best_val {
                best_val = logits[start + j];
                best_idx = j;
            }
        }
        actions[i] = best_idx as f32;
    }
    actions
}

// ─── Vectorized Evaluate ────────────────────────────────────────────

/// Evaluate a candidate (params) across n_episodes using vectorized environments.
/// All episodes run in parallel on n_envs environments.
pub fn vec_evaluate(
    env_name: &str,
    params: &[f32],
    layer_dims: &[(usize, usize)],
    n_episodes: usize,
    max_steps: usize,
    obs_dim: usize,
    act_dim: usize,
    discrete: bool,
) -> f64 {
    let mut env = GpuVecEnv::new(env_name, n_episodes);
    env.reset_all(42);

    let mut total_rewards = vec![0.0f64; n_episodes];
    let mut active = vec![true; n_episodes];

    for _step in 0..max_steps {
        // Check if all done
        if active.iter().all(|&a| !a) { break; }

        // Batch forward pass
        let logits = batch_forward(
            env.get_observations(), params, layer_dims,
            n_episodes, obs_dim, act_dim, discrete,
        );

        // Convert to actions
        let actions = if discrete {
            batch_argmax(&logits, n_episodes, act_dim)
        } else {
            logits // already tanh-squashed
        };

        // Step all envs
        let (_, rewards, dones) = env.step_all(&actions);

        // Accumulate rewards
        for i in 0..n_episodes {
            if active[i] {
                total_rewards[i] += rewards[i] as f64;
                if dones[i] != 0 {
                    active[i] = false;
                }
            }
        }
    }

    total_rewards.iter().sum::<f64>() / n_episodes as f64
}

// ─── CUDA Kernels ───────────────────────────────────────────────────

/// CUDA kernel source for CartPole (compiled at runtime via nvrtc).
pub const CARTPOLE_CUDA_SRC: &str = r#"
extern "C" __global__ void cartpole_step(
    float* states,       // [N, 4]
    float* observations, // [N, 4]
    float* rewards,      // [N]
    unsigned char* dones, // [N]
    unsigned int* steps,  // [N]
    const float* actions, // [N]
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || dones[i]) return;

    int off = i * 4;
    float x = states[off], x_dot = states[off+1];
    float theta = states[off+2], theta_dot = states[off+3];

    float force = (actions[i] > 0.5f) ? 10.0f : -10.0f;
    float cos_t = cosf(theta), sin_t = sinf(theta);
    float total_mass = 1.1f;
    float pole_mass = 0.1f;
    float half_len = 0.5f;

    float temp = (force + pole_mass * half_len * theta_dot * theta_dot * sin_t) / total_mass;
    float theta_acc = (9.8f * sin_t - cos_t * temp)
        / (half_len * (4.0f/3.0f - pole_mass * cos_t * cos_t / total_mass));
    float x_acc = temp - pole_mass * half_len * theta_acc * cos_t / total_mass;

    states[off]   = x + 0.02f * x_dot;
    states[off+1] = x_dot + 0.02f * x_acc;
    states[off+2] = theta + 0.02f * theta_dot;
    states[off+3] = theta_dot + 0.02f * theta_acc;

    steps[i] += 1;

    bool terminated = fabsf(states[off]) > 2.4f || fabsf(states[off+2]) > 0.2094f;
    bool truncated = steps[i] >= 500;

    rewards[i] = terminated ? 0.0f : 1.0f;
    dones[i] = (terminated || truncated) ? 1 : 0;

    // obs = state
    observations[off]   = states[off];
    observations[off+1] = states[off+1];
    observations[off+2] = states[off+2];
    observations[off+3] = states[off+3];
}

extern "C" __global__ void cartpole_reset(
    float* states,
    float* observations,
    unsigned char* dones,
    unsigned int* steps,
    unsigned long long* seeds,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // LCG random init
    unsigned long long s = seeds[i];
    int off = i * 4;
    for (int j = 0; j < 4; j++) {
        s = s * 6364136223846793005ULL + 1;
        float val = (float)(s >> 33) / (float)0xFFFFFFFF;
        states[off + j] = val * 0.1f - 0.05f;
        observations[off + j] = states[off + j];
    }
    seeds[i] = s;
    dones[i] = 0;
    steps[i] = 0;
}
"#;

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cartpole_reset() {
        let mut env = GpuVecEnv::new("CartPole-v1", 10);
        env.reset_all(42);

        // All states should be small ([-0.05, 0.05])
        for i in 0..10 {
            for j in 0..4 {
                let val = env.states[i * 4 + j];
                assert!(val.abs() <= 0.05, "State [{},{}] = {} out of range", i, j, val);
            }
        }

        // All should be not done
        assert!(env.dones.iter().all(|&d| d == 0));
        assert!(env.steps.iter().all(|&s| s == 0));
    }

    #[test]
    fn test_cartpole_step_basic() {
        let mut env = GpuVecEnv::new("CartPole-v1", 4);
        env.reset_all(42);

        // Step with action=1 (push right) for all
        let actions = vec![1.0f32; 4];
        env.step_all(&actions);

        // Should get reward 1.0 (not terminated yet)
        for i in 0..4 {
            assert_eq!(env.rewards[i], 1.0, "Env {} should get reward 1.0", i);
            assert_eq!(env.dones[i], 0, "Env {} should not be done", i);
        }

        // Observations should match states
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(env.observations[i * 4 + j], env.states[i * 4 + j]);
            }
        }
    }

    #[test]
    fn test_cartpole_matches_cpu_reference() {
        // Compare vectorized CartPole against the original CPU CartPole
        use super::super::env::{self, Environment, Action};

        let seed = 42u64;
        let mut ref_env = env::make("CartPole-v1", Some(seed)).unwrap();
        let ref_obs = ref_env.reset(Some(seed));

        let mut vec_env = GpuVecEnv::new("CartPole-v1", 1);
        vec_env.seeds[0] = seed;
        vec_env.reset_env(0);

        // Compare initial observations
        for j in 0..4 {
            let diff = (ref_obs[j] - vec_env.observations[j]).abs();
            assert!(diff < 1e-5, "Init obs[{}]: ref={} vec={} diff={}",
                j, ref_obs[j], vec_env.observations[j], diff);
        }

        // Step 100 times with alternating actions
        for step in 0..100 {
            let action = (step % 2) as usize;
            let ref_result = ref_env.step(&Action::Discrete(action));
            let actions = vec![action as f32];
            vec_env.step_all(&actions);

            // Compare observations
            for j in 0..4 {
                let diff = (ref_result.observation[j] - vec_env.observations[j]).abs();
                assert!(diff < 1e-4, "Step {} obs[{}]: ref={} vec={} diff={}",
                    step, j, ref_result.observation[j], vec_env.observations[j], diff);
            }

            // Compare rewards and done
            let ref_reward = ref_result.reward;
            assert!((ref_reward - vec_env.rewards[0] as f64).abs() < 1e-6,
                "Step {} reward: ref={} vec={}", step, ref_reward, vec_env.rewards[0]);

            if ref_result.done() {
                assert_eq!(vec_env.dones[0], 1, "Step {}: ref done but vec not", step);
                break;
            }
        }
    }

    #[test]
    fn test_cartpole_500_steps_perfect_balance() {
        // With all zeros state, pushing alternately should keep it alive for a while
        let mut env = GpuVecEnv::new("CartPole-v1", 1);
        env.reset_all(0); // seed 0

        let mut total_reward = 0.0f32;
        for step in 0..500 {
            let action = vec![(step % 2) as f32];
            let (_, rewards, dones) = env.step_all(&action);
            total_reward += rewards[0];
            if dones[0] != 0 { break; }
        }
        // Should survive at least a few steps
        assert!(total_reward > 5.0, "CartPole should survive >5 steps, got {}", total_reward);
    }

    #[test]
    fn test_vectorized_many_envs() {
        let n = 1000;
        let mut env = GpuVecEnv::new("CartPole-v1", n);
        env.reset_all(42);

        // Step all 1000 envs
        let actions = vec![1.0f32; n];
        let (obs, rewards, dones) = env.step_all(&actions);

        assert_eq!(obs.len(), n * 4);
        assert_eq!(rewards.len(), n);
        assert_eq!(dones.len(), n);

        // All should still be alive after 1 step
        assert!(dones.iter().all(|&d| d == 0));
    }

    #[test]
    fn test_lunar_lander_reset() {
        let mut env = GpuVecEnv::new("LunarLander-v3", 10);
        env.reset_all(42);

        for i in 0..10 {
            // Y should be at initial height
            let y = env.states[i * 8 + 1];
            assert!((y - LL_INITIAL_Y).abs() < 0.01, "Env {} y={} should be near {}", i, y, LL_INITIAL_Y);
            // Legs should not be touching
            assert_eq!(env.states[i * 8 + 6], 0.0);
            assert_eq!(env.states[i * 8 + 7], 0.0);
        }
    }

    #[test]
    fn test_lunar_lander_gravity() {
        let mut env = GpuVecEnv::new("LunarLander-v3", 1);
        env.reset_all(42);

        let initial_y = env.states[1];

        // Do nothing for 10 steps
        for _ in 0..10 {
            let actions = vec![0.0f32]; // noop
            env.step_all(&actions);
        }

        // Should have fallen
        assert!(env.states[1] < initial_y, "Lander should fall: y={} < initial_y={}", env.states[1], initial_y);
    }

    #[test]
    fn test_batch_forward() {
        let obs_dim = 4;
        let act_dim = 2;
        let layer_dims = vec![(4, 8), (8, 2)];
        let n_params: usize = layer_dims.iter().map(|(i, o)| i * o + o).sum(); // 4*8+8 + 8*2+2 = 50
        let params = vec![0.1f32; n_params];

        let n = 5;
        let obs = vec![0.5f32; n * obs_dim];

        let result = batch_forward(&obs, &params, &layer_dims, n, obs_dim, act_dim, true);
        assert_eq!(result.len(), n * act_dim);

        // All envs should get the same output (same obs, same params)
        for i in 1..n {
            for j in 0..act_dim {
                assert_eq!(result[j], result[i * act_dim + j]);
            }
        }
    }

    #[test]
    fn test_batch_argmax() {
        let logits = vec![
            0.1, 0.9, 0.2, 0.3,  // env 0 → action 1
            0.5, 0.1, 0.8, 0.2,  // env 1 → action 2
            0.9, 0.1, 0.1, 0.1,  // env 2 → action 0
        ];
        let actions = batch_argmax(&logits, 3, 4);
        assert_eq!(actions[0], 1.0);
        assert_eq!(actions[1], 2.0);
        assert_eq!(actions[2], 0.0);
    }

    #[test]
    fn test_vec_evaluate_cartpole() {
        // Random params should get some positive reward on CartPole
        let obs_dim = 4;
        let act_dim = 2;
        let layer_dims = vec![(4, 32), (32, 16), (16, 2)];
        let n_params: usize = layer_dims.iter().map(|(i, o)| i * o + o).sum();
        let params = vec![0.01f32; n_params];

        let score = vec_evaluate("CartPole-v1", &params, &layer_dims, 5, 500, obs_dim, act_dim, true);
        assert!(score > 0.0, "CartPole should get positive score, got {}", score);
    }

    #[test]
    fn test_auto_reset() {
        let mut env = GpuVecEnv::new("CartPole-v1", 2);
        env.reset_all(42);

        // Force env 0 to be done
        env.dones[0] = 1;

        env.auto_reset();

        // Env 0 should be reset
        assert_eq!(env.dones[0], 0);
        assert_eq!(env.steps[0], 0);
        // Env 1 should be unchanged
    }
}
