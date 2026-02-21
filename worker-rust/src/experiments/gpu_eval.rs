//! Full on-device GPU evaluation: forward pass + env step in a single kernel.
//!
//! Zero PCIe transfers during episode loop.
//! Only transfer: params (once at start) → rewards (once at end).
//!
//! One CUDA thread per environment runs the ENTIRE episode:
//!   loop { obs → forward_pass → argmax → env_step → accumulate_reward }
//! All on GPU registers/shared memory.

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchConfig, LaunchAsync};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::compile_ptx;
#[cfg(feature = "cuda")]
use std::sync::Arc;

use super::env::{self, ActionSpace};
use super::policy::Policy;
use super::native_runner::{GenResult, RunResult};
use super::optim::{CmaEs, Rng as OptRng, compute_centered_ranks};
use serde_json::Value;
use std::time::Instant;

/// CUDA kernel: full episode evaluation on GPU.
/// Each thread runs one environment for one complete episode.
/// Network: 2-hidden-layer FF with tanh activations.
///
/// Template params compiled into kernel via string substitution:
/// - OBS_DIM, ACT_DIM, H1, H2, N_PARAMS, MAX_STEPS
/// - ENV_TYPE: 0=CartPole, 1=LunarLander, 2=Swimmer, 3=NLinkPendulum
const FULL_EVAL_KERNEL_TEMPLATE: &str = r#"
// ─── Network constants (substituted at compile time) ────────────
#define OBS_DIM {OBS_DIM}
#define ACT_DIM {ACT_DIM}
#define H1 {H1}
#define H2 {H2}
#define N_PARAMS {N_PARAMS}
#define MAX_STEPS {MAX_STEPS}
#define ENV_TYPE {ENV_TYPE}

// ─── CartPole constants ─────────────────────────────────────────
#define CP_GRAVITY 9.8f
#define CP_CART_MASS 1.0f
#define CP_POLE_MASS 0.1f
#define CP_TOTAL_MASS 1.1f
#define CP_HALF_LEN 0.5f
#define CP_FORCE 10.0f
#define CP_TAU 0.02f
#define CP_X_THRESH 2.4f
#define CP_THETA_THRESH 0.2094f

// ─── Forward pass: obs → action index ───────────────────────────
__device__ int forward_pass_discrete(
    const float* obs,
    const float* params
) {
    // Layer 1: obs(OBS_DIM) → h1(H1), tanh
    float h1[H1];
    int off = 0;
    for (int j = 0; j < H1; j++) {
        float sum = params[off + OBS_DIM * H1 + j]; // bias
        for (int i = 0; i < OBS_DIM; i++) {
            sum += obs[i] * params[off + i * H1 + j];
        }
        h1[j] = tanhf(sum);
    }
    off += OBS_DIM * H1 + H1;

    // Layer 2: h1(H1) → h2(H2), tanh
    float h2[H2];
    for (int j = 0; j < H2; j++) {
        float sum = params[off + H1 * H2 + j]; // bias
        for (int i = 0; i < H1; i++) {
            sum += h1[i] * params[off + i * H2 + j];
        }
        h2[j] = tanhf(sum);
    }
    off += H1 * H2 + H2;

    // Output layer: h2(H2) → out(ACT_DIM), no activation
    float out[ACT_DIM];
    for (int j = 0; j < ACT_DIM; j++) {
        float sum = params[off + H2 * ACT_DIM + j]; // bias
        for (int i = 0; i < H2; i++) {
            sum += h2[i] * params[off + i * ACT_DIM + j];
        }
        out[j] = sum;
    }

    // Argmax
    int best = 0;
    float best_val = out[0];
    for (int j = 1; j < ACT_DIM; j++) {
        if (out[j] > best_val) {
            best_val = out[j];
            best = j;
        }
    }
    return best;
}

// ─── Forward pass: obs → continuous action (tanh squashed) ──────
__device__ void forward_pass_continuous(
    const float* obs,
    const float* params,
    float* actions
) {
    float h1[H1];
    int off = 0;
    for (int j = 0; j < H1; j++) {
        float sum = params[off + OBS_DIM * H1 + j];
        for (int i = 0; i < OBS_DIM; i++) {
            sum += obs[i] * params[off + i * H1 + j];
        }
        h1[j] = tanhf(sum);
    }
    off += OBS_DIM * H1 + H1;

    float h2[H2];
    for (int j = 0; j < H2; j++) {
        float sum = params[off + H1 * H2 + j];
        for (int i = 0; i < H1; i++) {
            sum += h1[i] * params[off + i * H2 + j];
        }
        h2[j] = tanhf(sum);
    }
    off += H1 * H2 + H2;

    for (int j = 0; j < ACT_DIM; j++) {
        float sum = params[off + H2 * ACT_DIM + j];
        for (int i = 0; i < H2; i++) {
            sum += h2[i] * params[off + i * ACT_DIM + j];
        }
        actions[j] = tanhf(sum); // squash to [-1, 1]
    }
}

// ─── CartPole step ──────────────────────────────────────────────
__device__ float cartpole_step_device(float* state, int action) {
    float x = state[0], x_dot = state[1];
    float theta = state[2], theta_dot = state[3];

    float force = (action == 1) ? CP_FORCE : -CP_FORCE;
    float cos_t = cosf(theta), sin_t = sinf(theta);

    float temp = (force + CP_POLE_MASS * CP_HALF_LEN * theta_dot * theta_dot * sin_t) / CP_TOTAL_MASS;
    float theta_acc = (CP_GRAVITY * sin_t - cos_t * temp)
        / (CP_HALF_LEN * (4.0f/3.0f - CP_POLE_MASS * cos_t * cos_t / CP_TOTAL_MASS));
    float x_acc = temp - CP_POLE_MASS * CP_HALF_LEN * theta_acc * cos_t / CP_TOTAL_MASS;

    state[0] = x + CP_TAU * x_dot;
    state[1] = x_dot + CP_TAU * x_acc;
    state[2] = theta + CP_TAU * theta_dot;
    state[3] = theta_dot + CP_TAU * theta_acc;

    int terminated = (fabsf(state[0]) > CP_X_THRESH) || (fabsf(state[2]) > CP_THETA_THRESH);
    return terminated ? -1.0f : 1.0f; // -1 = done, reward = 1.0 per step
}

// ─── LunarLander constants ──────────────────────────────────────
#define LL_GRAVITY -10.0f
#define LL_MAIN_POWER 13.0f
#define LL_SIDE_POWER 0.6f
#define LL_FPS 50.0f
#define LL_INITIAL_Y 1.4f

// ─── LunarLander step ──────────────────────────────────────────
// state: [x, y, vx, vy, angle, ang_vel, leg1, leg2]
// Returns: reward (negative = done with crash penalty)
__device__ float lunarlander_step_device(float* state, int action, int step) {
    float x = state[0], y = state[1];
    float vx = state[2], vy = state[3];
    float angle = state[4], ang_vel = state[5];
    float dt = 1.0f / LL_FPS;

    // Gravity
    vy += LL_GRAVITY / LL_FPS;

    // Engine forces
    if (action == 2) { // main engine
        vx += -sinf(angle) * LL_MAIN_POWER / LL_FPS;
        vy +=  cosf(angle) * LL_MAIN_POWER / LL_FPS;
    } else if (action == 1) { // left engine
        ang_vel -= LL_SIDE_POWER / LL_FPS;
        vx += cosf(angle) * LL_SIDE_POWER * 0.1f / LL_FPS;
    } else if (action == 3) { // right engine
        ang_vel += LL_SIDE_POWER / LL_FPS;
        vx -= cosf(angle) * LL_SIDE_POWER * 0.1f / LL_FPS;
    }

    // Integrate
    x += vx * dt;
    y += vy * dt;
    angle += ang_vel * dt;

    // Ground collision
    float leg1 = 0.0f, leg2 = 0.0f;
    if (y <= 0.0f) {
        y = 0.0f; vy = 0.0f;
        vx *= 0.5f; ang_vel *= 0.5f;
        leg1 = 1.0f; leg2 = 1.0f;
    }

    state[0] = x; state[1] = y; state[2] = vx; state[3] = vy;
    state[4] = angle; state[5] = ang_vel; state[6] = leg1; state[7] = leg2;

    // Reward (Gymnasium-style shaping)
    float shaping = -100.0f * sqrtf(x*x + y*y)
                   - 100.0f * sqrtf(vx*vx + vy*vy)
                   - 100.0f * fabsf(angle)
                   + 10.0f * leg1 + 10.0f * leg2;
    float fuel = (action == 2) ? -0.3f : ((action == 1 || action == 3) ? -0.03f : 0.0f);
    float reward = shaping / 100.0f + fuel;

    // Terminal conditions
    int crashed = (y <= 0.0f) && (fabsf(vx) > 0.5f || fabsf(angle) > 0.5f);
    int landed = (y <= 0.0f) && (fabsf(vx) < 0.1f) && (fabsf(angle) < 0.1f);
    int oob = (fabsf(x) > 1.5f) || (y > 2.0f);

    if (crashed) return -10000.0f; // signal: done + -100 penalty
    if (landed) return 10000.0f + reward; // signal: done + +100 bonus
    if (oob) return -20000.0f; // signal: done, no extra penalty

    return reward;
}

// ─── Swimmer constants ──────────────────────────────────────────
#define SW_VISCOSITY 0.1f
#define SW_SEG_LEN 0.1f
#define SW_SEG_MASS 1.0f
#define SW_DT 0.01f
#define SW_FRAME_SKIP 4
#define SW_CTRL_COST 0.0001f
#define SW_SEG_INERTIA (SW_SEG_MASS * SW_SEG_LEN * SW_SEG_LEN / 12.0f)

// state layout for Swimmer: [x, y, body_angle, j1_angle, j2_angle, vx, vy, body_angvel, j1_vel, j2_vel]
#define SW_STATE_DIM 10

__device__ float swimmer_step_device(float* state, float* actions) {
    float x = state[0], y = state[1];
    float body_angle = state[2];
    float j1 = state[3], j2 = state[4];
    float vx = state[5], vy = state[6];
    float body_angvel = state[7];
    float j1_vel = state[8], j2_vel = state[9];

    for (int sub = 0; sub < SW_FRAME_SKIP; sub++) {
        // Viscous drag (anisotropic: 5x more normal to segments)
        float drag_fx = 0.0f, drag_fy = 0.0f, drag_torque = 0.0f;
        float angles[3] = { body_angle, body_angle + j1, body_angle + j1 + j2 };
        float ang_vels[3] = { body_angvel, body_angvel + j1_vel, body_angvel + j1_vel + j2_vel };

        for (int s = 0; s < 3; s++) {
            float ca = cosf(angles[s]), sa = sinf(angles[s]);
            float vt = vx * ca + vy * sa;
            float vn = -vx * sa + vy * ca;
            float dt_f = -SW_VISCOSITY * vt * SW_SEG_LEN;
            float dn_f = -SW_VISCOSITY * 5.0f * vn * SW_SEG_LEN;
            drag_fx += dt_f * ca - dn_f * sa;
            drag_fy += dt_f * sa + dn_f * ca;
            drag_torque += -SW_VISCOSITY * 2.0f * ang_vels[s] * SW_SEG_LEN * SW_SEG_LEN;
        }

        float ax = drag_fx / (SW_SEG_MASS * 3.0f);
        float ay = drag_fy / (SW_SEG_MASS * 3.0f);
        float body_alpha = (drag_torque - actions[0] - actions[1]) / (SW_SEG_INERTIA * 3.0f);
        float j1_alpha = (actions[0] + (-SW_VISCOSITY * j1_vel * SW_SEG_LEN)) / SW_SEG_INERTIA;
        float j2_alpha = (actions[1] + (-SW_VISCOSITY * j2_vel * SW_SEG_LEN)) / SW_SEG_INERTIA;

        vx += ax * SW_DT; vy += ay * SW_DT;
        x += vx * SW_DT; y += vy * SW_DT;
        body_angvel += body_alpha * SW_DT;
        body_angle += body_angvel * SW_DT;
        j1_vel += j1_alpha * SW_DT;
        j1 += j1_vel * SW_DT;
        j1 = fmaxf(-1.5f, fminf(1.5f, j1));
        j2_vel += j2_alpha * SW_DT;
        j2 += j2_vel * SW_DT;
        j2 = fmaxf(-1.5f, fminf(1.5f, j2));
    }

    float x_before = state[0];
    state[0] = x; state[1] = y; state[2] = body_angle;
    state[3] = j1; state[4] = j2;
    state[5] = vx; state[6] = vy; state[7] = body_angvel;
    state[8] = j1_vel; state[9] = j2_vel;

    float dt_ctrl = SW_DT * SW_FRAME_SKIP;
    float forward_reward = (x - x_before) / dt_ctrl;
    float ctrl_cost = SW_CTRL_COST * (actions[0]*actions[0] + actions[1]*actions[1]);
    return forward_reward - ctrl_cost;
}

// ─── N-Link Pendulum step ───────────────────────────────────────
// N_LINKS is defined via {N_LINKS} substitution (default 2)
#ifndef N_LINKS
#define N_LINKS 2
#endif
#define PEND_GRAVITY 9.81f
#define PEND_LINK_LEN 1.0f
#define PEND_LINK_MASS 1.0f
#define PEND_DT 0.05f
#define PEND_MAX_VEL 8.0f
#define PEND_CTRL_COST 0.01f

// state layout: [angles[N_LINKS], velocities[N_LINKS]]
__device__ float pendulum_step_device(float* angles, float* vels, float* actions, int n_links) {
    for (int i = 0; i < n_links; i++) {
        // Gravity torque on joint i
        float grav_torque = 0.0f;
        for (int j = i; j < n_links; j++) {
            float mass_below = PEND_LINK_MASS * (float)(n_links - j);
            float angle_sum = 0.0f;
            for (int k = 0; k <= j; k++) angle_sum += angles[k];
            grav_torque += -mass_below * PEND_GRAVITY * PEND_LINK_LEN * sinf(angle_sum);
        }

        float inertia = PEND_LINK_MASS * PEND_LINK_LEN * PEND_LINK_LEN * (float)(n_links - i);
        float applied = fmaxf(-1.0f, fminf(1.0f, actions[i]));
        float damping = -0.1f * vels[i];
        float alpha = (grav_torque + applied + damping) / inertia;

        vels[i] += alpha * PEND_DT;
        vels[i] = fmaxf(-PEND_MAX_VEL, fminf(PEND_MAX_VEL, vels[i]));
        angles[i] += vels[i] * PEND_DT;
    }

    // Height reward
    float height = 0.0f;
    float cum_angle = 0.0f;
    for (int i = 0; i < n_links; i++) {
        cum_angle += angles[i];
        height += PEND_LINK_LEN * cosf(cum_angle);
    }
    float max_h = (float)n_links * PEND_LINK_LEN;
    float height_reward = (height + max_h) / (2.0f * max_h);
    float vel_penalty = 0.0f;
    float ctrl_cost = 0.0f;
    for (int i = 0; i < n_links; i++) {
        vel_penalty += vels[i] * vels[i] * 0.01f;
        ctrl_cost += actions[i] * actions[i] * PEND_CTRL_COST;
    }
    return height_reward - vel_penalty - ctrl_cost;
}

// ─── Main kernel: one thread = one full episode ─────────────────
extern "C" __global__ void evaluate_episodes(
    const float* __restrict__ all_params,  // [n_candidates × N_PARAMS]
    float* rewards_out,                     // [n_candidates × n_episodes]
    int n_candidates,
    int n_episodes,
    unsigned long long base_seed
) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_candidates * n_episodes;
    if (global_idx >= total) return;

    int cand_idx = global_idx / n_episodes;

    const float* params = &all_params[cand_idx * N_PARAMS];

    // Init state with LCG random
    float state[OBS_DIM];
    unsigned long long seed = base_seed + (unsigned long long)global_idx * 12345ULL + 1ULL;

    #if ENV_TYPE == 0
        // CartPole: small random init
        for (int i = 0; i < OBS_DIM; i++) {
            seed = seed * 6364136223846793005ULL + 1ULL;
            float val = (float)(seed >> 33) / (float)0xFFFFFFFF;
            state[i] = val * 0.1f - 0.05f;
        }
    #elif ENV_TYPE == 1
        // LunarLander: specific init
        seed = seed * 6364136223846793005ULL + 1ULL;
        state[0] = ((float)(seed >> 33) / (float)0xFFFFFFFF * 0.1f - 0.05f) * 0.1f; // x
        state[1] = LL_INITIAL_Y;  // y
        seed = seed * 6364136223846793005ULL + 1ULL;
        state[2] = ((float)(seed >> 33) / (float)0xFFFFFFFF * 0.1f - 0.05f) * 0.1f; // vx
        seed = seed * 6364136223846793005ULL + 1ULL;
        state[3] = ((float)(seed >> 33) / (float)0xFFFFFFFF * 0.1f - 0.05f) * 0.1f; // vy
        state[4] = 0.0f; state[5] = 0.0f; state[6] = 0.0f; state[7] = 0.0f;
    #elif ENV_TYPE == 2
        // Swimmer: [x, y, body_angle, j1, j2, vx, vy, body_angvel, j1_vel, j2_vel]
        state[0] = 0.0f; state[1] = 0.0f;
        seed = seed * 6364136223846793005ULL + 1ULL;
        state[2] = ((float)(seed >> 33) / (float)0xFFFFFFFF * 0.2f - 0.1f); // body_angle
        seed = seed * 6364136223846793005ULL + 1ULL;
        state[3] = ((float)(seed >> 33) / (float)0xFFFFFFFF * 0.2f - 0.1f); // j1
        seed = seed * 6364136223846793005ULL + 1ULL;
        state[4] = ((float)(seed >> 33) / (float)0xFFFFFFFF * 0.2f - 0.1f); // j2
        state[5] = 0.0f; state[6] = 0.0f; state[7] = 0.0f; state[8] = 0.0f; state[9] = 0.0f;
    #elif ENV_TYPE == 3
        // N-Link Pendulum: angles start at PI (hanging down) + small perturbation
        for (int i = 0; i < N_LINKS; i++) {
            seed = seed * 6364136223846793005ULL + 1ULL;
            state[i] = 3.14159265f + ((float)(seed >> 33) / (float)0xFFFFFFFF * 0.2f - 0.1f);
            seed = seed * 6364136223846793005ULL + 1ULL;
            state[N_LINKS + i] = ((float)(seed >> 33) / (float)0xFFFFFFFF * 0.2f - 0.1f);
        }
    #endif

    float total_reward = 0.0f;

    for (int step = 0; step < MAX_STEPS; step++) {
        #if ENV_TYPE == 0
            // CartPole: discrete action
            int action = forward_pass_discrete(state, params);
            float r = cartpole_step_device(state, action);
            if (r < 0.0f) break; // terminated
            total_reward += 1.0f;
        #elif ENV_TYPE == 1
            // LunarLander: discrete action
            int action = forward_pass_discrete(state, params);
            float r = lunarlander_step_device(state, action, step);
            if (r <= -10000.0f) { total_reward -= 100.0f; break; } // crashed
            if (r >= 10000.0f) { total_reward += 100.0f + (r - 10000.0f); break; } // landed
            if (r <= -20000.0f) { break; } // out of bounds
            total_reward += r;
        #elif ENV_TYPE == 2
            // Swimmer: continuous action, obs = [body_angle, j1, j2, vx, vy, angvel, j1vel, j2vel]
            float obs_sw[8] = { state[2], state[3], state[4], state[5], state[6], state[7], state[8], state[9] };
            float actions_sw[2];
            forward_pass_continuous(obs_sw, params, actions_sw);
            float r = swimmer_step_device(state, actions_sw);
            total_reward += r;
        #elif ENV_TYPE == 3
            // N-Link Pendulum: continuous action, obs = [cos, sin, vel per link]
            float obs_p[OBS_DIM];
            for (int i = 0; i < N_LINKS; i++) {
                obs_p[3*i] = cosf(state[i]);
                obs_p[3*i+1] = sinf(state[i]);
                obs_p[3*i+2] = state[N_LINKS + i];
            }
            float actions_p[ACT_DIM];
            forward_pass_continuous(obs_p, params, actions_p);
            float r = pendulum_step_device(state, &state[N_LINKS], actions_p, N_LINKS);
            total_reward += r;
        #endif
    }

    rewards_out[global_idx] = total_reward;
}
"#;

// ─── GPU Evaluator ──────────────────────────────────────────────

#[cfg(feature = "cuda")]
pub struct GpuEvaluator {
    dev: Arc<CudaDevice>,
    n_params: usize,
    n_episodes: usize,
    env_name: String,
}

#[cfg(feature = "cuda")]
impl GpuEvaluator {
    /// Create a GPU evaluator. Compiles the CUDA kernel for the given env/network.
    pub fn new(env_name: &str, policy: &Policy, n_episodes: usize) -> Option<Self> {
        let dev = CudaDevice::new(0).ok()?;

        let (h1, h2) = if policy.config.hidden.len() == 2 {
            (policy.config.hidden[0], policy.config.hidden[1])
        } else {
            eprintln!("⚠️ GPU eval only supports 2-hidden-layer networks");
            return None;
        };

        let env_type = match env_name {
            "CartPole-v1" => 0,
            "LunarLander-v3" => 1,
            "Swimmer-v1" => 2,
            _ if env_name.starts_with("Pendulum-") && env_name.ends_with("Link") => 3,
            _ => {
                eprintln!("⚠️ GPU eval not supported for {}", env_name);
                return None;
            }
        };

        // For N-Link Pendulum, extract N
        let n_links: usize = if env_type == 3 {
            env_name.strip_prefix("Pendulum-")
                .and_then(|s| s.strip_suffix("Link"))
                .and_then(|s| s.parse().ok())
                .unwrap_or(2)
        } else { 0 };

        // Compile kernel with concrete dimensions
        let max_steps_str = match env_name {
            "CartPole-v1" => "500",
            "LunarLander-v3" | "Swimmer-v1" => "1000",
            _ if env_name.starts_with("Pendulum-") => {
                if n_links <= 3 { "500" } else { "1000" }
            }
            _ => "1000",
        };
        let src = FULL_EVAL_KERNEL_TEMPLATE
            .replace("{OBS_DIM}", &policy.config.obs_dim.to_string())
            .replace("{ACT_DIM}", &policy.config.act_dim.to_string())
            .replace("{H1}", &h1.to_string())
            .replace("{H2}", &h2.to_string())
            .replace("{N_PARAMS}", &policy.n_params.to_string())
            .replace("{MAX_STEPS}", max_steps_str)
            .replace("{ENV_TYPE}", &env_type.to_string())
            .replace("N_LINKS 2", &format!("N_LINKS {}", if n_links > 0 { n_links } else { 2 }));

        let ptx = compile_ptx(&src).ok()?;
        dev.load_ptx(ptx, "eval", &["evaluate_episodes"]).ok()?;

        eprintln!("🚀 GPU Full-Episode kernel compiled | {}→{}→{}→{} | {} params",
            policy.config.obs_dim, h1, h2, policy.config.act_dim, policy.n_params);

        Some(GpuEvaluator {
            dev,
            n_params: policy.n_params,
            n_episodes,
            env_name: env_name.to_string(),
        })
    }

    /// Evaluate multiple candidates. Returns fitness for each candidate.
    /// ALL computation happens on GPU — zero transfers during episodes.
    pub fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        let n_candidates = candidates.len();
        let total_episodes = n_candidates * self.n_episodes;

        // Flatten all params into one buffer
        let mut all_params = vec![0.0f32; n_candidates * self.n_params];
        for (c, cand) in candidates.iter().enumerate() {
            for (i, &v) in cand.iter().enumerate() {
                all_params[c * self.n_params + i] = v as f32;
            }
        }

        // Upload params (ONLY transfer to GPU)
        let d_params = self.dev.htod_copy(all_params).unwrap();
        let d_rewards = self.dev.alloc_zeros::<f32>(total_episodes).unwrap();

        // Launch: one thread per episode
        let f = self.dev.get_func("eval", "evaluate_episodes").unwrap();
        let block = 256;
        let grid = ((total_episodes + block - 1) / block) as u32;
        let cfg = LaunchConfig {
            block_dim: (block as u32, 1, 1),
            grid_dim: (grid, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            f.launch(cfg, (
                &d_params,
                &d_rewards,
                n_candidates as i32,
                self.n_episodes as i32,
                42u64, // base_seed
            )).unwrap();
        }

        // Download rewards (ONLY transfer from GPU)
        let rewards: Vec<f32> = self.dev.dtoh_sync_copy(&d_rewards).unwrap();

        // Aggregate per candidate
        let mut fitnesses = vec![0.0f64; n_candidates];
        for c in 0..n_candidates {
            let sum: f64 = (0..self.n_episodes)
                .map(|ep| rewards[c * self.n_episodes + ep] as f64)
                .sum();
            fitnesses[c] = sum / self.n_episodes as f64;
        }
        fitnesses
    }
}

// ─── GPU-accelerated CMA-ES (full on-device) ───────────────────

pub fn run_gpu_full_cma_es(env_name: &str, params: &Value, mut on_gen: impl FnMut(GenResult)) -> RunResult {
    let env_cfg = env::get_env_config(env_name).expect("Unknown env");
    let hidden = super::env::default_hidden(env_name);
    let max_evals = params.get("max_evals").and_then(|v| v.as_u64()).unwrap_or(100000) as usize;
    let eval_episodes = params.get("eval_episodes").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
    let sigma0 = params.get("sigma0").and_then(|v| v.as_f64()).unwrap_or(0.5);
    let patience = params.get("patience").and_then(|v| v.as_u64()).unwrap_or(200) as usize;

    let policy = Policy::new(env_cfg.obs_dim, env_cfg.action_space.size(), &hidden, env_cfg.action_space);
    let mut cma = CmaEs::new(policy.n_params, sigma0, None);
    let solved = env_cfg.solved_threshold;

    // Try GPU evaluator
    #[cfg(feature = "cuda")]
    let gpu_eval = super::gpu_eval::GpuEvaluator::new(env_name, &policy, eval_episodes);
    #[cfg(not(feature = "cuda"))]
    let gpu_eval: Option<()> = None;

    let using_gpu = gpu_eval.is_some();
    eprintln!("🦀{} Full CMA-ES on {} | {} params | pop={} | {} episodes | GPU={}",
        if using_gpu { "🚀" } else { "💻" },
        env_name, policy.n_params, cma.pop_size, eval_episodes, using_gpu);

    let mut best_ever = f64::NEG_INFINITY;
    let mut best_params: Option<Vec<f64>> = None;
    let mut total_evals = 0usize;
    let mut stale_gens = 0usize;
    let mut restarts = 0usize;
    let start = Instant::now();

    while total_evals < max_evals {
        let candidates = cma.ask();

        let fitnesses = {
            #[cfg(feature = "cuda")]
            {
                if let Some(ref eval) = gpu_eval {
                    eval.evaluate_batch(&candidates)
                } else {
                    cpu_evaluate_batch(env_name, &candidates, &policy, eval_episodes, env_cfg.max_steps)
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                cpu_evaluate_batch(env_name, &candidates, &policy, eval_episodes, env_cfg.max_steps)
            }
        };

        total_evals += candidates.len() * eval_episodes;
        cma.tell(&candidates, &fitnesses);

        let gen_best = fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if gen_best > best_ever {
            best_ever = gen_best;
            stale_gens = 0;
            let idx = fitnesses.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            best_params = Some(candidates[idx].clone());
        } else {
            stale_gens += 1;
        }

        on_gen(GenResult {
            generation: cma.gen, best: gen_best, best_ever,
            mean: fitnesses.iter().sum::<f64>() / fitnesses.len() as f64,
            sigma: cma.sigma, evals: total_evals, time: start.elapsed().as_secs_f64(),
        });

        if best_ever >= solved { break; }

        if stale_gens >= patience || cma.sigma < 1e-8 {
            restarts += 1;
            let new_sigma = sigma0 * (1.0 + 0.2 * restarts as f64);
            cma = CmaEs::new(policy.n_params, new_sigma, None);
            if let Some(ref bp) = best_params {
                for (i, v) in bp.iter().enumerate() { cma.mean[i] = *v; }
            }
            stale_gens = 0;
        }
    }

    RunResult {
        method: if using_gpu { "GPU-Full-CMA-ES" } else { "CPU-Vec-CMA-ES" }.into(),
        environment: env_name.into(), best_ever,
        final_mean: best_ever, final_std: 0.0,
        total_evals, generations: cma.gen,
        elapsed: start.elapsed().as_secs_f64(), solved: best_ever >= solved,
    }
}

// ─── GPU OpenAI-ES (full on-device) ─────────────────────────────

pub fn run_gpu_full_openai_es(env_name: &str, params: &Value, mut on_gen: impl FnMut(GenResult)) -> RunResult {
    let env_cfg = env::get_env_config(env_name).expect("Unknown env");
    let hidden = if let Some(h) = params.get("hidden").and_then(|v| v.as_array()) {
        h.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect()
    } else {
        env::default_hidden(env_name)
    };
    let max_evals = params.get("max_evals").and_then(|v| v.as_u64()).unwrap_or(100000) as usize;
    let eval_episodes = params.get("eval_episodes").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
    let pop_size = params.get("pop_size").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
    let lr = params.get("lr").and_then(|v| v.as_f64()).unwrap_or(0.02);
    let noise_std = params.get("noise_std").and_then(|v| v.as_f64()).unwrap_or(0.1);
    let weight_decay = params.get("weight_decay").and_then(|v| v.as_f64()).unwrap_or(0.0);

    let policy = Policy::new(env_cfg.obs_dim, env_cfg.action_space.size(), &hidden, env_cfg.action_space);
    let n = policy.n_params;
    let solved = env_cfg.solved_threshold;

    #[cfg(feature = "cuda")]
    let gpu_eval = super::gpu_eval::GpuEvaluator::new(env_name, &policy, eval_episodes);
    #[cfg(not(feature = "cuda"))]
    let gpu_eval: Option<()> = None;

    let using_gpu = gpu_eval.is_some();
    eprintln!("🦀{} Full OpenAI-ES on {} | {} params | pop={} | GPU={}",
        if using_gpu { "🚀" } else { "💻" }, env_name, n, pop_size, using_gpu);

    let mut rng = OptRng::new(42);
    let mut theta: Vec<f64> = (0..n).map(|_| rng.randn() * 0.1).collect();

    let mut best_ever = f64::NEG_INFINITY;
    let mut best_params: Option<Vec<f64>> = None;
    let mut total_evals = 0usize;
    let mut gen = 0usize;
    let start = Instant::now();

    while total_evals < max_evals {
        gen += 1;

        let epsilons: Vec<Vec<f64>> = (0..pop_size).map(|_| rng.randn_vec(n)).collect();

        // Build all candidates
        let mut all_candidates: Vec<Vec<f64>> = Vec::with_capacity(pop_size * 2);
        for eps in &epsilons {
            let plus: Vec<f64> = (0..n).map(|i| theta[i] + noise_std * eps[i]).collect();
            let minus: Vec<f64> = (0..n).map(|i| theta[i] - noise_std * eps[i]).collect();
            all_candidates.push(plus);
            all_candidates.push(minus);
        }

        let all_fitnesses = {
            #[cfg(feature = "cuda")]
            {
                if let Some(ref eval) = gpu_eval {
                    eval.evaluate_batch(&all_candidates)
                } else {
                    cpu_evaluate_batch(env_name, &all_candidates, &policy, eval_episodes, env_cfg.max_steps)
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                cpu_evaluate_batch(env_name, &all_candidates, &policy, eval_episodes, env_cfg.max_steps)
            }
        };

        total_evals += pop_size * 2 * eval_episodes;

        let ranks = compute_centered_ranks(&all_fitnesses);
        let mut grad = vec![0.0f64; n];
        for (i, eps) in epsilons.iter().enumerate() {
            let rank_diff = ranks[2 * i] - ranks[2 * i + 1];
            for j in 0..n {
                grad[j] += rank_diff * eps[j];
            }
        }

        for j in 0..n {
            theta[j] = theta[j] * (1.0 - weight_decay)
                + lr / (pop_size as f64 * noise_std) * grad[j];
        }

        let gen_best = all_fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let gen_mean = all_fitnesses.iter().sum::<f64>() / all_fitnesses.len() as f64;
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

    RunResult {
        method: if using_gpu { "GPU-Full-OpenAI-ES" } else { "CPU-OpenAI-ES" }.into(),
        environment: env_name.into(), best_ever,
        final_mean: best_ever, final_std: 0.0,
        total_evals, generations: gen,
        elapsed: start.elapsed().as_secs_f64(), solved: best_ever >= solved,
    }
}

/// CPU fallback: evaluate candidates using rayon
fn cpu_evaluate_batch(
    env_name: &str,
    candidates: &[Vec<f64>],
    policy: &Policy,
    n_episodes: usize,
    max_steps: usize,
) -> Vec<f64> {
    use rayon::prelude::*;
    candidates.par_iter().map(|c| {
        let pf32: Vec<f32> = c.iter().map(|&v| v as f32).collect();
        let mut total = 0.0;
        for ep in 0..n_episodes {
            let mut env = env::make(env_name, Some(ep as u64 * 1000)).unwrap();
            let mut obs = env.reset(Some(ep as u64 * 1000));
            let mut ep_reward = 0.0;
            for _ in 0..max_steps {
                let action = policy.forward(&obs, &pf32);
                let result = env.step(&action);
                ep_reward += result.reward;
                if result.done() { break; }
                obs = result.observation;
            }
            total += ep_reward;
        }
        total / n_episodes as f64
    }).collect()
}

/// Benchmark: CPU (rayon) vs GPU (full on-device)
pub fn run_gpu_full_benchmark(env_name: &str, params: &Value, mut on_gen: impl FnMut(GenResult)) -> RunResult {
    let env_cfg = env::get_env_config(env_name).expect("Unknown env");
    let hidden = super::env::default_hidden(env_name);
    let n_candidates = params.get("n_candidates").and_then(|v| v.as_u64()).unwrap_or(28) as usize;
    let eval_episodes = params.get("eval_episodes").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

    let policy = Policy::new(env_cfg.obs_dim, env_cfg.action_space.size(), &hidden, env_cfg.action_space);

    let mut rng = OptRng::new(42);
    let candidates: Vec<Vec<f64>> = (0..n_candidates)
        .map(|_| (0..policy.n_params).map(|_| rng.randn() * 0.1).collect())
        .collect();

    // CPU baseline (rayon)
    let cpu_start = Instant::now();
    let _cpu_fits = cpu_evaluate_batch(env_name, &candidates, &policy, eval_episodes, env_cfg.max_steps);
    let cpu_time = cpu_start.elapsed().as_secs_f64();

    // GPU full on-device
    let mut gpu_time = f64::MAX;
    let mut speedup = 0.0;

    #[cfg(feature = "cuda")]
    {
        if let Some(eval) = GpuEvaluator::new(env_name, &policy, eval_episodes) {
            // Warmup
            let _ = eval.evaluate_batch(&candidates);

            let gpu_start = Instant::now();
            let _gpu_fits = eval.evaluate_batch(&candidates);
            gpu_time = gpu_start.elapsed().as_secs_f64();
            speedup = cpu_time / gpu_time;
        }
    }

    eprintln!("🏁 Full On-Device Benchmark: {} candidates × {} episodes on {}",
        n_candidates, eval_episodes, env_name);
    eprintln!("  CPU (rayon):     {:.4}s", cpu_time);
    eprintln!("  GPU (on-device): {:.4}s ({:.1}x speedup)", gpu_time, speedup);

    on_gen(GenResult {
        generation: 1, best: speedup, best_ever: speedup, mean: cpu_time,
        sigma: gpu_time, evals: n_candidates * eval_episodes * 2,
        time: cpu_time + gpu_time,
    });

    RunResult {
        method: "GPU-Full-Benchmark".into(), environment: env_name.into(),
        best_ever: speedup, final_mean: cpu_time, final_std: gpu_time,
        total_evals: n_candidates * eval_episodes * 2, generations: 1,
        elapsed: cpu_time + gpu_time, solved: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_evaluate_batch() {
        let env_cfg = env::get_env_config("CartPole-v1").unwrap();
        let policy = Policy::new(env_cfg.obs_dim, env_cfg.action_space.size(), &[32, 16], env_cfg.action_space);

        let mut rng = OptRng::new(42);
        let candidates: Vec<Vec<f64>> = (0..4)
            .map(|_| (0..policy.n_params).map(|_| rng.randn() * 0.1).collect())
            .collect();

        let fits = cpu_evaluate_batch("CartPole-v1", &candidates, &policy, 3, 500);
        assert_eq!(fits.len(), 4);
        for f in &fits {
            assert!(*f > 0.0, "Should get positive CartPole score");
        }
    }

    #[test]
    fn test_kernel_template_substitution() {
        let src = FULL_EVAL_KERNEL_TEMPLATE
            .replace("{OBS_DIM}", "4")
            .replace("{ACT_DIM}", "2")
            .replace("{H1}", "32")
            .replace("{H2}", "16")
            .replace("{N_PARAMS}", "722")
            .replace("{MAX_STEPS}", "500")
            .replace("{ENV_TYPE}", "0");

        assert!(src.contains("#define OBS_DIM 4"));
        assert!(src.contains("#define H1 32"));
        assert!(src.contains("#define N_PARAMS 722"));
        assert!(!src.contains("{OBS_DIM}"));
    }
}
