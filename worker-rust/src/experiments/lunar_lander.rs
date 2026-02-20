//! LunarLander-v3 in pure Rust using wrapped2d (Box2D).
//!
//! Faithful port of gymnasium's LunarLander-v2/v3.
//! Discrete actions: 0=noop, 1=left, 2=main, 3=right

use wrapped2d::b2;
use wrapped2d::user_data::NoUserData;
use super::env::*;

// ─── Constants (matching gymnasium exactly) ───────────────────────────
const FPS: f32 = 50.0;
const SCALE: f32 = 30.0;
const MAIN_ENGINE_POWER: f32 = 13.0;
const SIDE_ENGINE_POWER: f32 = 0.6;
const INITIAL_RANDOM: f32 = 1000.0;

const VIEWPORT_W: f32 = 600.0;
const VIEWPORT_H: f32 = 400.0;

const LEG_AWAY: f32 = 20.0;
const LEG_DOWN: f32 = 18.0;
const LEG_W: f32 = 2.0;
const LEG_H: f32 = 8.0;
const LEG_SPRING_TORQUE: f32 = 40.0;

const SIDE_ENGINE_HEIGHT: f32 = 14.0;
const SIDE_ENGINE_AWAY: f32 = 12.0;
const MAIN_ENGINE_Y_LOCATION: f32 = 4.0;

const LANDER_POLY: [(f32, f32); 6] = [
    (-14.0, 17.0), (-17.0, 0.0), (-17.0, -10.0),
    (17.0, -10.0), (17.0, 0.0), (14.0, 17.0),
];

const CHUNKS: usize = 11;

/// Simple LCG PRNG.
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
}

pub struct LunarLander {
    config: EnvConfig,
    world: b2::World<NoUserData>,
    lander: Option<b2::BodyHandle>,
    legs: [Option<b2::BodyHandle>; 2],
    ground: Option<b2::BodyHandle>,
    leg_ground_contact: [bool; 2],
    helipad_y: f32,
    prev_shaping: Option<f32>,
    game_over: bool,
    step_count: usize,
    rng: Rng,
}

impl LunarLander {
    pub fn new(seed: Option<u64>) -> Self {
        let config = get_env_config("LunarLander-v3").unwrap();
        let gravity = b2::Vec2 { x: 0.0, y: -10.0 };
        let world = b2::World::new(&gravity);
        
        let mut env = LunarLander {
            config,
            world,
            lander: None,
            legs: [None; 2],
            ground: None,
            leg_ground_contact: [false; 2],
            helipad_y: 0.0,
            prev_shaping: None,
            game_over: false,
            step_count: 0,
            rng: Rng::new(seed.unwrap_or(42)),
        };
        env.do_reset();
        env
    }

    fn destroy(&mut self) {
        // Destroy in reverse order: legs, lander, ground
        for leg in &mut self.legs {
            if let Some(h) = leg.take() {
                self.world.destroy_body(h);
            }
        }
        if let Some(h) = self.lander.take() {
            self.world.destroy_body(h);
        }
        if let Some(h) = self.ground.take() {
            self.world.destroy_body(h);
        }
    }

    fn do_reset(&mut self) -> Vec<f32> {
        self.destroy();
        
        let gravity = b2::Vec2 { x: 0.0, y: -10.0 };
        self.world = b2::World::new(&gravity);
        self.game_over = false;
        self.prev_shaping = None;
        self.step_count = 0;
        self.leg_ground_contact = [false; 2];
        
        let w = VIEWPORT_W / SCALE;
        let h = VIEWPORT_H / SCALE;
        
        // Terrain heights
        let mut height = vec![0.0f32; CHUNKS + 1];
        for v in &mut height { *v = self.rng.uniform(0.0, h / 2.0); }
        let chunk_x: Vec<f32> = (0..CHUNKS).map(|i| w / (CHUNKS as f32 - 1.0) * i as f32).collect();
        self.helipad_y = h / 4.0;
        for i in (CHUNKS/2).saturating_sub(2)..=(CHUNKS/2 + 2).min(CHUNKS) {
            height[i] = self.helipad_y;
        }
        let smooth_y: Vec<f32> = (0..CHUNKS).map(|i| {
            let prev = if i > 0 { height[i-1] } else { height[0] };
            0.33 * (prev + height[i] + height[i + 1])
        }).collect();
        
        // Ground body
        let ground = self.world.create_body(&b2::BodyDef {
            body_type: b2::BodyType::Static,
            ..b2::BodyDef::new()
        });
        
        // Bottom edge
        let edge = b2::EdgeShape::new_with(
            &b2::Vec2 { x: 0.0, y: 0.0 },
            &b2::Vec2 { x: w, y: 0.0 },
        );
        self.world.body_mut(ground).create_fixture(&edge, &mut b2::FixtureDef {
            density: 0.0, friction: 0.1, ..b2::FixtureDef::new()
        });
        
        // Terrain edges
        for i in 0..CHUNKS - 1 {
            let edge = b2::EdgeShape::new_with(
                &b2::Vec2 { x: chunk_x[i], y: smooth_y[i] },
                &b2::Vec2 { x: chunk_x[i + 1], y: smooth_y[i + 1] },
            );
            self.world.body_mut(ground).create_fixture(&edge, &mut b2::FixtureDef {
                density: 0.0, friction: 0.1, ..b2::FixtureDef::new()
            });
        }
        self.ground = Some(ground);
        
        // Lander
        let initial_x = w / 2.0;
        let initial_y = h;
        
        let vertices: Vec<b2::Vec2> = LANDER_POLY.iter()
            .map(|&(x, y)| b2::Vec2 { x: x / SCALE, y: y / SCALE })
            .collect();
        let lander_shape = b2::PolygonShape::new_with(&vertices);
        
        let lander = self.world.create_body(&b2::BodyDef {
            body_type: b2::BodyType::Dynamic,
            position: b2::Vec2 { x: initial_x, y: initial_y },
            angle: 0.0,
            ..b2::BodyDef::new()
        });
        
        self.world.body_mut(lander).create_fixture(&lander_shape, &mut b2::FixtureDef {
            density: 5.0,
            friction: 0.1,
            restitution: 0.0,
            filter: b2::Filter {
                category_bits: 0x0010,
                mask_bits: 0x001,
                group_index: 0,
            },
            ..b2::FixtureDef::new()
        });
        
        // Initial random force
        let fx = self.rng.uniform(-INITIAL_RANDOM, INITIAL_RANDOM);
        let fy = self.rng.uniform(-INITIAL_RANDOM, INITIAL_RANDOM);
        self.world.body_mut(lander).apply_force_to_center(&b2::Vec2 { x: fx, y: fy }, true);
        self.lander = Some(lander);
        
        // Legs
        for idx in 0..2usize {
            let dir = if idx == 0 { -1.0f32 } else { 1.0 };
            
            let leg = self.world.create_body(&b2::BodyDef {
                body_type: b2::BodyType::Dynamic,
                position: b2::Vec2 {
                    x: initial_x - dir * LEG_AWAY / SCALE,
                    y: initial_y,
                },
                angle: dir * 0.05,
                ..b2::BodyDef::new()
            });
            
            let leg_shape = b2::PolygonShape::new_box(LEG_W / SCALE, LEG_H / SCALE);
            self.world.body_mut(leg).create_fixture(&leg_shape, &mut b2::FixtureDef {
                density: 1.0,
                restitution: 0.0,
                filter: b2::Filter {
                    category_bits: 0x0020,
                    mask_bits: 0x001,
                    group_index: 0,
                },
                ..b2::FixtureDef::new()
            });
            
            // Joint
            let mut jd = b2::RevoluteJointDef::new(lander, leg);
            jd.local_anchor_a = b2::Vec2 { x: 0.0, y: 0.0 };
            jd.local_anchor_b = b2::Vec2 {
                x: dir * LEG_AWAY / SCALE,
                y: LEG_DOWN / SCALE,
            };
            jd.enable_motor = true;
            jd.enable_limit = true;
            jd.max_motor_torque = LEG_SPRING_TORQUE;
            jd.motor_speed = 0.3 * dir;
            
            if dir < 0.0 {
                jd.lower_angle = 0.9 - 0.5;
                jd.upper_angle = 0.9;
            } else {
                jd.lower_angle = -0.9;
                jd.upper_angle = -0.9 + 0.5;
            }
            
            self.world.create_joint(&jd);
            self.legs[idx] = Some(leg);
        }
        
        // Initial noop step
        let result = self.do_step(0);
        result.observation
    }

    fn do_step(&mut self, action: usize) -> StepResult {
        let lander = self.lander.expect("reset() not called");
        
        let angle = self.world.body(lander).angle();
        let tip = (angle.sin(), angle.cos());
        let side = (-tip.1, tip.0);
        
        let d0 = self.rng.uniform(-1.0, 1.0) / SCALE;
        let d1 = self.rng.uniform(-1.0, 1.0) / SCALE;
        
        // Main engine (action == 2)
        let mut m_power = 0.0f32;
        if action == 2 {
            m_power = 1.0;
            let ox = tip.0 * (MAIN_ENGINE_Y_LOCATION / SCALE + 2.0 * d0) + side.0 * d1;
            let oy = -tip.1 * (MAIN_ENGINE_Y_LOCATION / SCALE + 2.0 * d0) - side.1 * d1;
            
            let pos = *self.world.body(lander).position();
            let impulse_pos = b2::Vec2 { x: pos.x + ox, y: pos.y + oy };
            
            self.world.body_mut(lander).apply_linear_impulse(
                &b2::Vec2 { x: -ox * MAIN_ENGINE_POWER, y: -oy * MAIN_ENGINE_POWER },
                &impulse_pos, true,
            );
        }
        
        // Side engines
        let mut s_power = 0.0f32;
        if action == 1 || action == 3 {
            let direction = action as f32 - 2.0;
            s_power = 1.0;
            
            let ox = tip.0 * d0 + side.0 * (3.0 * d1 + direction * SIDE_ENGINE_AWAY / SCALE);
            let oy = -tip.1 * d0 - side.1 * (3.0 * d1 + direction * SIDE_ENGINE_AWAY / SCALE);
            
            let pos = *self.world.body(lander).position();
            let impulse_pos = b2::Vec2 {
                x: pos.x + ox - tip.0 * 17.0 / SCALE,
                y: pos.y + oy + tip.1 * SIDE_ENGINE_HEIGHT / SCALE,
            };
            
            self.world.body_mut(lander).apply_linear_impulse(
                &b2::Vec2 { x: -ox * SIDE_ENGINE_POWER, y: -oy * SIDE_ENGINE_POWER },
                &impulse_pos, true,
            );
        }
        
        // Physics step
        self.world.step(1.0 / FPS, 6 * 30, 2 * 30);
        
        // Check ground contacts
        self.check_contacts();
        
        // Build observation
        let pos = *self.world.body(lander).position();
        let vel = *self.world.body(lander).linear_velocity();
        let angle = self.world.body(lander).angle();
        let angular_vel = self.world.body(lander).angular_velocity();
        
        let w = VIEWPORT_W / SCALE;
        let h = VIEWPORT_H / SCALE;
        
        let state = [
            (pos.x - w / 2.0) / (w / 2.0),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (h / 2.0),
            vel.x * (w / 2.0) / FPS,
            vel.y * (h / 2.0) / FPS,
            angle,
            20.0 * angular_vel / FPS,
            if self.leg_ground_contact[0] { 1.0 } else { 0.0 },
            if self.leg_ground_contact[1] { 1.0 } else { 0.0 },
        ];
        
        // Reward
        let shaping = -100.0 * (state[0] * state[0] + state[1] * state[1]).sqrt()
            - 100.0 * (state[2] * state[2] + state[3] * state[3]).sqrt()
            - 100.0 * state[4].abs()
            + 10.0 * state[6]
            + 10.0 * state[7];
        
        let mut reward = match self.prev_shaping {
            Some(prev) => shaping - prev,
            None => 0.0,
        };
        self.prev_shaping = Some(shaping);
        
        reward -= m_power * 0.30;
        reward -= s_power * 0.03;
        
        let mut terminated = false;
        if self.game_over || state[0].abs() >= 1.0 {
            terminated = true;
            reward = -100.0;
        }
        if !self.world.body(lander).is_awake() {
            terminated = true;
            reward = 100.0;
        }
        
        self.step_count += 1;
        let truncated = self.step_count >= self.config.max_steps;
        
        StepResult {
            observation: state.iter().map(|&v| v as f32).collect(),
            reward: reward as f64,
            terminated,
            truncated,
        }
    }
    
    fn check_contacts(&mut self) {
        self.leg_ground_contact = [false; 2];
        let lander = match self.lander { Some(h) => h, None => return };
        let leg0 = match self.legs[0] { Some(h) => h, None => return };
        let leg1 = match self.legs[1] { Some(h) => h, None => return };
        
        for contact in self.world.contacts() {
            if !contact.is_touching() { continue; }
            let (body_a, _) = contact.fixture_a();
            let (body_b, _) = contact.fixture_b();
            
            // Leg 0 contact
            if body_a == leg0 || body_b == leg0 {
                self.leg_ground_contact[0] = true;
            }
            // Leg 1 contact
            if body_a == leg1 || body_b == leg1 {
                self.leg_ground_contact[1] = true;
            }
            // Lander body touching ground (crash)
            if (body_a == lander || body_b == lander)
                && body_a != leg0 && body_b != leg0
                && body_a != leg1 && body_b != leg1
            {
                self.game_over = true;
            }
        }
    }
}

impl Environment for LunarLander {
    fn reset(&mut self, seed: Option<u64>) -> Vec<f32> {
        if let Some(s) = seed {
            self.rng = Rng::new(s);
        }
        self.do_reset()
    }

    fn step(&mut self, action: &Action) -> StepResult {
        let a = match action {
            Action::Discrete(a) => *a,
            Action::Continuous(_) => 0,
        };
        self.do_step(a)
    }

    fn config(&self) -> &EnvConfig { &self.config }
    fn steps(&self) -> usize { self.step_count }
}
