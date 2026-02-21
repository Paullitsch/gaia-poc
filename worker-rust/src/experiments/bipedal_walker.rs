//! BipedalWalker-v3 in pure Rust using wrapped2d (Box2D).
//!
//! Port of gymnasium's BipedalWalker-v3.
//! Continuous actions: [hip1, knee1, hip2, knee2] in [-1, 1]
//! Observation: 24-dim (hull angle/vel, joint angles/speeds, leg contacts, 10 LIDAR)

use wrapped2d::b2;
use wrapped2d::b2::UnknownJoint;
use wrapped2d::user_data::NoUserData;
use super::env::*;

// ─── Constants ────────────────────────────────────────────────────────
const FPS: f32 = 50.0;
const SCALE: f32 = 30.0;
const MOTORS_TORQUE: f32 = 80.0;
const SPEED_HIP: f32 = 4.0;
const SPEED_KNEE: f32 = 6.0;
const LIDAR_RANGE: f32 = 160.0 / SCALE;

const VIEWPORT_W: f32 = 600.0;
const VIEWPORT_H: f32 = 400.0;

const TERRAIN_STEP: f32 = 14.0 / SCALE;
const TERRAIN_LENGTH: usize = 200;
const TERRAIN_HEIGHT: f32 = VIEWPORT_H / SCALE / 4.0;
const TERRAIN_GRASS: usize = 10;
const TERRAIN_STARTPAD: usize = 20;
const FRICTION: f32 = 2.5;
const INITIAL_RANDOM: f32 = 5.0;

const LEG_DOWN: f32 = -8.0 / SCALE;
const LEG_W: f32 = 8.0 / SCALE;
const LEG_H: f32 = 34.0 / SCALE;

const HULL_POLY: [(f32, f32); 5] = [
    (-30.0, 9.0), (6.0, 9.0), (34.0, 1.0), (34.0, -8.0), (-30.0, -8.0),
];

struct Rng { state: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { state: seed.wrapping_add(1) } }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + (hi - lo) * u as f32
    }
    #[allow(dead_code)]
    fn random(&mut self) -> f32 { self.uniform(0.0, 1.0) }
    fn integers(&mut self, lo: i32, hi: i32) -> i32 {
        lo + ((self.next_u64() % (hi - lo) as u64) as i32)
    }
}

pub struct BipedalWalker {
    config: EnvConfig,
    world: b2::World<NoUserData>,
    hull: Option<b2::BodyHandle>,
    legs: Vec<b2::BodyHandle>,       // 4 bodies: upper_left, lower_left, upper_right, lower_right
    joints: Vec<b2::JointHandle>,    // 4 joints: hip_left, knee_left, hip_right, knee_right
    terrain_bodies: Vec<b2::BodyHandle>,
    terrain_x: Vec<f32>,
    terrain_y: Vec<f32>,
    leg_ground_contact: [bool; 4],   // lower legs only: index 1 and 3
    prev_shaping: Option<f32>,
    game_over: bool,
    step_count: usize,
    rng: Rng,
}

impl BipedalWalker {
    pub fn new(seed: Option<u64>) -> Self {
        let config = get_env_config("BipedalWalker-v3").unwrap();
        let gravity = b2::Vec2 { x: 0.0, y: -10.0 };
        let world = b2::World::new(&gravity);

        let mut env = BipedalWalker {
            config,
            world,
            hull: None,
            legs: Vec::new(),
            joints: Vec::new(),
            terrain_bodies: Vec::new(),
            terrain_x: Vec::new(),
            terrain_y: Vec::new(),
            leg_ground_contact: [false; 4],
            prev_shaping: None,
            game_over: false,
            step_count: 0,
            rng: Rng::new(seed.unwrap_or(42)),
        };
        env.do_reset();
        env
    }

    fn destroy(&mut self) {
        self.joints.clear(); // Joints destroyed with bodies
        for leg in self.legs.drain(..) {
            self.world.destroy_body(leg);
        }
        if let Some(h) = self.hull.take() {
            self.world.destroy_body(h);
        }
        for t in self.terrain_bodies.drain(..) {
            self.world.destroy_body(t);
        }
    }

    fn generate_terrain(&mut self) {
        let mut velocity = 0.0f32;
        let mut y = TERRAIN_HEIGHT;
        self.terrain_x.clear();
        self.terrain_y.clear();

        // Generate height profile (non-hardcore: just smooth terrain)
        let mut counter = TERRAIN_STARTPAD as i32;
        for i in 0..TERRAIN_LENGTH {
            let x = i as f32 * TERRAIN_STEP;
            self.terrain_x.push(x);

            velocity = 0.8 * velocity + 0.01 * (TERRAIN_HEIGHT - y).signum();
            if i > TERRAIN_STARTPAD {
                velocity += self.rng.uniform(-1.0, 1.0) / SCALE;
            }
            y += velocity;

            self.terrain_y.push(y);
            counter -= 1;
            if counter == 0 {
                counter = self.rng.integers(TERRAIN_GRASS as i32 / 2, TERRAIN_GRASS as i32);
            }
        }

        // Create terrain edge bodies
        for i in 0..TERRAIN_LENGTH - 1 {
            let edge = b2::EdgeShape::new_with(
                &b2::Vec2 { x: self.terrain_x[i], y: self.terrain_y[i] },
                &b2::Vec2 { x: self.terrain_x[i + 1], y: self.terrain_y[i + 1] },
            );
            let body = self.world.create_body(&b2::BodyDef {
                body_type: b2::BodyType::Static,
                ..b2::BodyDef::new()
            });
            self.world.body_mut(body).create_fixture(&edge, &mut b2::FixtureDef {
                friction: FRICTION,
                filter: b2::Filter {
                    category_bits: 0x0001,
                    mask_bits: 0xFFFF,
                    group_index: 0,
                },
                ..b2::FixtureDef::new()
            });
            self.terrain_bodies.push(body);
        }
    }

    fn do_reset(&mut self) -> Vec<f32> {
        self.destroy();

        let gravity = b2::Vec2 { x: 0.0, y: -10.0 };
        self.world = b2::World::new(&gravity);
        self.game_over = false;
        self.prev_shaping = None;
        self.step_count = 0;
        self.leg_ground_contact = [false; 4];

        self.generate_terrain();

        let init_x = TERRAIN_STEP * TERRAIN_STARTPAD as f32 / 2.0;
        let init_y = TERRAIN_HEIGHT + 2.0 * LEG_H;

        // Hull
        let hull_verts: Vec<b2::Vec2> = HULL_POLY.iter()
            .map(|&(x, y)| b2::Vec2 { x: x / SCALE, y: y / SCALE })
            .collect();
        let hull_shape = b2::PolygonShape::new_with(&hull_verts);

        let hull = self.world.create_body(&b2::BodyDef {
            body_type: b2::BodyType::Dynamic,
            position: b2::Vec2 { x: init_x, y: init_y },
            ..b2::BodyDef::new()
        });
        self.world.body_mut(hull).create_fixture(&hull_shape, &mut b2::FixtureDef {
            density: 5.0,
            friction: 0.1,
            filter: b2::Filter {
                category_bits: 0x0020,
                mask_bits: 0x0001,
                group_index: -1,
            },
            ..b2::FixtureDef::new()
        });

        // Initial random push
        let fx = self.rng.uniform(-INITIAL_RANDOM, INITIAL_RANDOM);
        self.world.body_mut(hull).apply_force_to_center(&b2::Vec2 { x: fx, y: 0.0 }, true);
        self.hull = Some(hull);

        // Create 2 legs, each with upper + lower segments
        self.legs.clear();
        self.joints.clear();

        for &dir in &[-1.0f32, 1.0] {
            // Upper leg
            let upper_shape = b2::PolygonShape::new_box(LEG_W / 2.0, LEG_H / 2.0);
            let upper = self.world.create_body(&b2::BodyDef {
                body_type: b2::BodyType::Dynamic,
                position: b2::Vec2 {
                    x: init_x,
                    y: init_y - LEG_H / 2.0 - LEG_DOWN,
                },
                angle: dir * 0.05,
                ..b2::BodyDef::new()
            });
            self.world.body_mut(upper).create_fixture(&upper_shape, &mut b2::FixtureDef {
                density: 1.0,
                restitution: 0.0,
                filter: b2::Filter {
                    category_bits: 0x0020,
                    mask_bits: 0x0001,
                    group_index: -1,
                },
                ..b2::FixtureDef::new()
            });

            // Hip joint
            let mut hip_jd = b2::RevoluteJointDef::new(hull, upper);
            hip_jd.local_anchor_a = b2::Vec2 { x: 0.0, y: LEG_DOWN };
            hip_jd.local_anchor_b = b2::Vec2 { x: 0.0, y: LEG_H / 2.0 };
            hip_jd.enable_motor = true;
            hip_jd.enable_limit = true;
            hip_jd.max_motor_torque = MOTORS_TORQUE;
            hip_jd.motor_speed = dir;
            hip_jd.lower_angle = -0.8;
            hip_jd.upper_angle = 1.1;
            let hip_joint = self.world.create_joint(&hip_jd);

            self.legs.push(upper);
            self.joints.push(hip_joint);

            // Lower leg
            let lower_shape = b2::PolygonShape::new_box(0.8 * LEG_W / 2.0, LEG_H / 2.0);
            let lower = self.world.create_body(&b2::BodyDef {
                body_type: b2::BodyType::Dynamic,
                position: b2::Vec2 {
                    x: init_x,
                    y: init_y - LEG_H * 3.0 / 2.0 - LEG_DOWN,
                },
                angle: dir * 0.05,
                ..b2::BodyDef::new()
            });
            self.world.body_mut(lower).create_fixture(&lower_shape, &mut b2::FixtureDef {
                density: 1.0,
                restitution: 0.0,
                filter: b2::Filter {
                    category_bits: 0x0020,
                    mask_bits: 0x0001,
                    group_index: -1,
                },
                ..b2::FixtureDef::new()
            });

            // Knee joint
            let mut knee_jd = b2::RevoluteJointDef::new(upper, lower);
            knee_jd.local_anchor_a = b2::Vec2 { x: 0.0, y: -LEG_H / 2.0 };
            knee_jd.local_anchor_b = b2::Vec2 { x: 0.0, y: LEG_H / 2.0 };
            knee_jd.enable_motor = true;
            knee_jd.enable_limit = true;
            knee_jd.max_motor_torque = MOTORS_TORQUE;
            knee_jd.motor_speed = 1.0;
            knee_jd.lower_angle = -1.6;
            knee_jd.upper_angle = -0.1;
            let knee_joint = self.world.create_joint(&knee_jd);

            self.legs.push(lower);
            self.joints.push(knee_joint);
        }

        // Noop step to get initial obs
        let result = self.do_step(&[0.0, 0.0, 0.0, 0.0]);
        result.observation
    }

    fn do_step(&mut self, action: &[f32; 4]) -> StepResult {
        let hull = self.hull.expect("reset() not called");

        // Apply motor controls (torque mode, not speed mode)
        for i in 0..4 {
            let speed = if i % 2 == 0 { SPEED_HIP } else { SPEED_KNEE };
            let a = action[i].clamp(-1.0, 1.0);
            let mut jref = self.world.joint_mut(self.joints[i]);
            if let UnknownJoint::Revolute(ref mut rj) = **jref {
                rj.set_motor_speed(speed * a.signum());
                rj.set_max_motor_torque(MOTORS_TORQUE * a.abs().min(1.0));
            }
        }

        // Physics step
        self.world.step(1.0 / FPS, 6 * 30, 2 * 30);

        // Check ground contacts
        self.check_contacts();

        // LIDAR
        let pos = *self.world.body(hull).position();
        let vel = *self.world.body(hull).linear_velocity();
        let hull_angle = self.world.body(hull).angle();
        let hull_angular_vel = self.world.body(hull).angular_velocity();

        let mut lidar_fractions = [1.0f32; 10];
        for i in 0..10 {
            let angle = 1.5 * i as f32 / 10.0;
            let p2 = b2::Vec2 {
                x: pos.x + angle.sin() * LIDAR_RANGE,
                y: pos.y - angle.cos() * LIDAR_RANGE,
            };
            // Simple raycast: find closest intersection with terrain
            let mut closest = 1.0f32;
            for ti in 0..self.terrain_x.len().saturating_sub(1) {
                let t1 = b2::Vec2 { x: self.terrain_x[ti], y: self.terrain_y[ti] };
                let t2 = b2::Vec2 { x: self.terrain_x[ti + 1], y: self.terrain_y[ti + 1] };
                if let Some(frac) = ray_segment_intersect(pos, p2, t1, t2) {
                    if frac < closest { closest = frac; }
                }
            }
            lidar_fractions[i] = closest;
        }

        // Get joint angles and speeds
        // IMPORTANT: joint_speed() = actual angular velocity (GetJointSpeed)
        // NOT motor_speed() which is the target speed we set (GetMotorSpeed)
        let mut joint_angles = [0.0f32; 4];
        let mut joint_speeds = [0.0f32; 4];
        for i in 0..4 {
            let jref = self.world.joint(self.joints[i]);
            if let UnknownJoint::Revolute(ref rj) = **jref {
                joint_angles[i] = rj.joint_angle();
                joint_speeds[i] = rj.joint_speed(); // actual velocity, not motor target!
            }
        }

        // Build 24-dim state
        let state: Vec<f32> = vec![
            hull_angle,
            2.0 * hull_angular_vel / FPS,
            0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,
            0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
            joint_angles[0],                            // hip left angle
            joint_speeds[0] / SPEED_HIP,                // hip left speed
            joint_angles[1] + 1.0,                      // knee left angle
            joint_speeds[1] / SPEED_KNEE,               // knee left speed
            if self.leg_ground_contact[1] { 1.0 } else { 0.0 }, // left lower contact
            joint_angles[2],                            // hip right angle
            joint_speeds[2] / SPEED_HIP,
            joint_angles[3] + 1.0,
            joint_speeds[3] / SPEED_KNEE,
            if self.leg_ground_contact[3] { 1.0 } else { 0.0 }, // right lower contact
            lidar_fractions[0], lidar_fractions[1], lidar_fractions[2],
            lidar_fractions[3], lidar_fractions[4], lidar_fractions[5],
            lidar_fractions[6], lidar_fractions[7], lidar_fractions[8],
            lidar_fractions[9],
        ];

        // Reward
        let shaping = 130.0 * pos.x / SCALE - 5.0 * state[0].abs();
        let mut reward = match self.prev_shaping {
            Some(prev) => shaping - prev,
            None => 0.0,
        };
        self.prev_shaping = Some(shaping);

        for a in action {
            reward -= 0.00035 * MOTORS_TORQUE * a.abs().min(1.0);
        }

        let mut terminated = false;
        if self.game_over || pos.x < 0.0 {
            reward = -100.0;
            terminated = true;
        }
        if pos.x > (TERRAIN_LENGTH - TERRAIN_GRASS) as f32 * TERRAIN_STEP {
            terminated = true;
        }

        self.step_count += 1;
        let truncated = self.step_count >= self.config.max_steps;

        StepResult {
            observation: state,
            reward: reward as f64,
            terminated,
            truncated,
        }
    }

    fn check_contacts(&mut self) {
        self.leg_ground_contact = [false; 4];
        let hull = match self.hull { Some(h) => h, None => return };

        for contact in self.world.contacts() {
            if !contact.is_touching() { continue; }
            let (body_a, _) = contact.fixture_a();
            let (body_b, _) = contact.fixture_b();

            // Check lower legs (indices 1 and 3)
            for &li in &[1usize, 3] {
                if li < self.legs.len() {
                    let leg = self.legs[li];
                    if body_a == leg || body_b == leg {
                        self.leg_ground_contact[li] = true;
                    }
                }
            }

            // Hull touching ground = game over
            if (body_a == hull || body_b == hull)
                && !self.legs.iter().any(|&l| l == body_a || l == body_b)
            {
                self.game_over = true;
            }
        }
    }
}

/// Ray-segment intersection. Returns fraction [0,1] or None.
fn ray_segment_intersect(p: b2::Vec2, p2: b2::Vec2, a: b2::Vec2, b: b2::Vec2) -> Option<f32> {
    let dx = p2.x - p.x;
    let dy = p2.y - p.y;
    let ex = b.x - a.x;
    let ey = b.y - a.y;

    let denom = dx * ey - dy * ex;
    if denom.abs() < 1e-10 { return None; }

    let t = ((a.x - p.x) * ey - (a.y - p.y) * ex) / denom;
    let u = ((a.x - p.x) * dy - (a.y - p.y) * dx) / denom;

    if t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0 {
        Some(t)
    } else {
        None
    }
}

impl Environment for BipedalWalker {
    fn reset(&mut self, seed: Option<u64>) -> Vec<f32> {
        if let Some(s) = seed {
            self.rng = Rng::new(s);
        }
        self.do_reset()
    }

    fn step(&mut self, action: &Action) -> StepResult {
        match action {
            Action::Continuous(v) => {
                let mut a = [0.0f32; 4];
                for i in 0..4.min(v.len()) { a[i] = v[i]; }
                self.do_step(&a)
            }
            Action::Discrete(_) => self.do_step(&[0.0; 4]),
        }
    }

    fn config(&self) -> &EnvConfig { &self.config }
    fn steps(&self) -> usize { self.step_count }
}
