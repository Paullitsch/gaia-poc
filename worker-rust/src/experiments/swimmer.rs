//! Swimmer environment — GPU-native, no Box2D/MuJoCo needed.
//!
//! Based on Coulom's PhD thesis and MuJoCo Swimmer-v5.
//! 3-segment swimmer in 2D viscous fluid.
//!
//! Physics: Simple 2D rigid body chain with viscous drag.
//! - 3 segments connected by 2 rotational joints
//! - Viscous fluid drag proportional to velocity
//! - Actions: torques on 2 joints
//!
//! Observation (8): [body_angle, joint1_angle, joint2_angle,
//!                   vx, vy, body_angular_vel, joint1_vel, joint2_vel]
//! Action (2): [torque1, torque2] in [-1, 1]
//! Reward: forward_velocity - 0.0001 * control_cost
//! Solved: >360 (based on MuJoCo reference, no strict threshold)

use super::env::*;

// ─── Constants ────────────────────────────────────────────────────────
const N_SEGMENTS: usize = 3;
const N_JOINTS: usize = N_SEGMENTS - 1; // 2
const SEGMENT_LENGTH: f32 = 0.1;
const SEGMENT_MASS: f32 = 1.0;
const VISCOSITY: f32 = 0.1;
const DT: f32 = 0.01; // 100 Hz physics
const FRAME_SKIP: usize = 4; // 25 Hz control
const MAX_STEPS: usize = 1000;
const CTRL_COST_WEIGHT: f32 = 0.0001;

// Moment of inertia for thin rod: I = (1/12) * m * L^2
const SEGMENT_INERTIA: f32 = SEGMENT_MASS * SEGMENT_LENGTH * SEGMENT_LENGTH / 12.0;

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

pub struct Swimmer {
    config: EnvConfig,
    // State: position and velocity of the system
    // Generalized coordinates: x, y, body_angle, joint_angles[2]
    // Generalized velocities: vx, vy, body_angular_vel, joint_vels[2]
    x: f32,
    y: f32,
    body_angle: f32,      // angle of first segment
    joint_angles: [f32; N_JOINTS],  // relative angles between segments
    vx: f32,
    vy: f32,
    body_angular_vel: f32,
    joint_vels: [f32; N_JOINTS],
    step_count: usize,
    rng: Rng,
}

impl Swimmer {
    pub fn new(seed: Option<u64>) -> Self {
        let config = EnvConfig {
            name: "Swimmer-v1".to_string(),
            obs_dim: 8,
            action_space: ActionSpace::Continuous(2),
            max_steps: MAX_STEPS,
            solved_threshold: 360.0,
        };
        let mut env = Swimmer {
            config,
            x: 0.0, y: 0.0,
            body_angle: 0.0,
            joint_angles: [0.0; N_JOINTS],
            vx: 0.0, vy: 0.0,
            body_angular_vel: 0.0,
            joint_vels: [0.0; N_JOINTS],
            step_count: 0,
            rng: Rng::new(seed.unwrap_or(42)),
        };
        env.do_reset();
        env
    }

    fn do_reset(&mut self) -> Vec<f32> {
        // Small random initial state (matching MuJoCo init)
        self.x = 0.0;
        self.y = 0.0;
        self.body_angle = self.rng.uniform(-0.1, 0.1);
        for i in 0..N_JOINTS {
            self.joint_angles[i] = self.rng.uniform(-0.1, 0.1);
        }
        self.vx = 0.0;
        self.vy = 0.0;
        self.body_angular_vel = 0.0;
        self.joint_vels = [0.0; N_JOINTS];
        self.step_count = 0;
        self.get_obs()
    }

    fn get_obs(&self) -> Vec<f32> {
        vec![
            self.body_angle,
            self.joint_angles[0],
            self.joint_angles[1],
            self.vx,
            self.vy,
            self.body_angular_vel,
            self.joint_vels[0],
            self.joint_vels[1],
        ]
    }

    /// Compute segment positions and orientations for drag calculation.
    fn get_segment_info(&self) -> [(f32, f32, f32); N_SEGMENTS] {
        let mut segments = [(0.0f32, 0.0f32, 0.0f32); N_SEGMENTS];

        // First segment
        let angle0 = self.body_angle;
        let cx0 = self.x;
        let cy0 = self.y;
        segments[0] = (cx0, cy0, angle0);

        // Subsequent segments
        let mut prev_angle = angle0;
        let mut tip_x = cx0 + SEGMENT_LENGTH * 0.5 * prev_angle.cos();
        let mut tip_y = cy0 + SEGMENT_LENGTH * 0.5 * prev_angle.sin();

        for i in 0..N_JOINTS {
            let angle = prev_angle + self.joint_angles[i];
            let cx = tip_x + SEGMENT_LENGTH * 0.5 * angle.cos();
            let cy = tip_y + SEGMENT_LENGTH * 0.5 * angle.sin();
            segments[i + 1] = (cx, cy, angle);

            tip_x = cx + SEGMENT_LENGTH * 0.5 * angle.cos();
            tip_y = cy + SEGMENT_LENGTH * 0.5 * angle.sin();
            prev_angle = angle;
        }

        segments
    }

    fn do_step(&mut self, actions: &[f32; 2]) -> StepResult {
        let x_before = self.x;

        // Sub-steps for numerical stability
        for _ in 0..FRAME_SKIP {
            self.physics_step(actions);
        }

        let x_after = self.x;
        let dt_ctrl = DT * FRAME_SKIP as f32;

        // Reward: forward velocity - control cost
        let forward_reward = (x_after - x_before) / dt_ctrl;
        let ctrl_cost = CTRL_COST_WEIGHT * (actions[0] * actions[0] + actions[1] * actions[1]);
        let reward = forward_reward - ctrl_cost;

        self.step_count += 1;
        let truncated = self.step_count >= MAX_STEPS;

        StepResult {
            observation: self.get_obs(),
            reward: reward as f64,
            terminated: false, // Swimmer never terminates
            truncated,
        }
    }

    fn physics_step(&mut self, actions: &[f32; 2]) {
        let segments = self.get_segment_info();

        // ── Viscous drag forces on each segment ──
        // Drag is anisotropic: higher normal to segment, lower along it
        let mut drag_fx = 0.0f32;
        let mut drag_fy = 0.0f32;
        let mut drag_torque = 0.0f32;

        for i in 0..N_SEGMENTS {
            let (_, _, angle) = segments[i];

            // Segment velocity (approximate: using center of mass velocity)
            // For a more accurate model, we'd track per-segment velocities
            let seg_vx = self.vx;
            let seg_vy = self.vy;

            // Decompose velocity into tangential and normal components
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let v_tangential = seg_vx * cos_a + seg_vy * sin_a;
            let v_normal = -seg_vx * sin_a + seg_vy * cos_a;

            // Drag: higher for normal motion (like a flat plate)
            let drag_t = -VISCOSITY * v_tangential * SEGMENT_LENGTH;
            let drag_n = -VISCOSITY * 5.0 * v_normal * SEGMENT_LENGTH; // 5x more drag normal

            // Convert back to world frame
            drag_fx += drag_t * cos_a - drag_n * sin_a;
            drag_fy += drag_t * sin_a + drag_n * cos_a;

            // Angular drag
            let total_ang_vel = self.body_angular_vel
                + if i > 0 { self.joint_vels[..i].iter().sum::<f32>() } else { 0.0 };
            drag_torque += -VISCOSITY * 2.0 * total_ang_vel * SEGMENT_LENGTH * SEGMENT_LENGTH;
        }

        let total_mass = SEGMENT_MASS * N_SEGMENTS as f32;

        // ── Linear acceleration ──
        let ax = drag_fx / total_mass;
        let ay = drag_fy / total_mass;

        // ── Angular acceleration of body ──
        let total_inertia = SEGMENT_INERTIA * N_SEGMENTS as f32;
        // Joint torques create reaction on body
        let body_torque = drag_torque - actions[0] - actions[1]; // reaction from joint torques
        let body_alpha = body_torque / total_inertia;

        // ── Joint accelerations ──
        // Each joint responds to its applied torque minus drag
        let mut joint_alphas = [0.0f32; N_JOINTS];
        for i in 0..N_JOINTS {
            let joint_drag = -VISCOSITY * self.joint_vels[i] * SEGMENT_LENGTH;
            joint_alphas[i] = (actions[i] + joint_drag) / SEGMENT_INERTIA;
        }

        // ── Euler integration ──
        self.vx += ax * DT;
        self.vy += ay * DT;
        self.x += self.vx * DT;
        self.y += self.vy * DT;

        self.body_angular_vel += body_alpha * DT;
        self.body_angle += self.body_angular_vel * DT;

        for i in 0..N_JOINTS {
            self.joint_vels[i] += joint_alphas[i] * DT;
            self.joint_angles[i] += self.joint_vels[i] * DT;
            // Clamp joint angles to prevent instability
            self.joint_angles[i] = self.joint_angles[i].clamp(-1.5, 1.5);
        }
    }
}

impl Environment for Swimmer {
    fn reset(&mut self, seed: Option<u64>) -> Vec<f32> {
        if let Some(s) = seed {
            self.rng = Rng::new(s);
        }
        self.do_reset()
    }

    fn step(&mut self, action: &Action) -> StepResult {
        match action {
            Action::Continuous(v) => {
                let mut a = [0.0f32; 2];
                for i in 0..2.min(v.len()) {
                    a[i] = v[i].clamp(-1.0, 1.0);
                }
                self.do_step(&a)
            }
            Action::Discrete(_) => self.do_step(&[0.0; 2]),
        }
    }

    fn config(&self) -> &EnvConfig { &self.config }
    fn steps(&self) -> usize { self.step_count }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swimmer_reset() {
        let mut env = Swimmer::new(Some(42));
        let obs = env.reset(Some(42));
        assert_eq!(obs.len(), 8);
        // Velocities should be zero initially
        assert_eq!(obs[3], 0.0); // vx
        assert_eq!(obs[4], 0.0); // vy
    }

    #[test]
    fn test_swimmer_step() {
        let mut env = Swimmer::new(Some(42));
        env.reset(Some(42));
        
        // Apply torque and step
        let result = env.step(&Action::Continuous(vec![1.0, -1.0]));
        assert_eq!(result.observation.len(), 8);
        assert!(!result.terminated);
        assert!(!result.truncated);
    }

    #[test]
    fn test_swimmer_never_terminates() {
        let mut env = Swimmer::new(Some(42));
        env.reset(Some(42));
        
        for _ in 0..100 {
            let result = env.step(&Action::Continuous(vec![1.0, 1.0]));
            assert!(!result.terminated, "Swimmer should never terminate");
        }
    }

    #[test]
    fn test_swimmer_via_factory() {
        let mut env = super::super::env::make("Swimmer-v1", Some(42)).unwrap();
        let obs = env.reset(Some(42));
        assert_eq!(obs.len(), 8);
        let result = env.step(&Action::Continuous(vec![0.5, -0.5]));
        assert_eq!(result.observation.len(), 8);
    }
}
