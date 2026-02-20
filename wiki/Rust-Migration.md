# Rust-Migration 🦀

> Worker v0.7.0: Alle Experimente nativ in Rust — kein Python mehr nötig

## Motivation

Python ist langsam für Neuroevolution. Environment-Simulation (`env.step()`) und Population-Evaluation sind CPU-bound — genau da wo Rust glänzt.

## Speedups

| Environment | Python | Rust (1 Thread) | Rust (4 Threads) | Speedup |
|-------------|--------|-----------------|-------------------|---------|
| CartPole | 152 evals/s | 2.073 evals/s | — | **13.6×** |
| LunarLander | ~150 evals/s | ~550 evals/s | solved in 7.1s | **3.6-10.4×** |
| BipedalWalker | ~80 evals/s | ~640 evals/s | — | **~8×** |

## Architektur

### Environments (`env.rs`)

```rust
pub trait Environment {
    fn reset(&mut self) -> Vec<f32>;
    fn step(&mut self, action: &[f32]) -> (Vec<f32>, f32, bool);
    fn obs_dim(&self) -> usize;
    fn act_dim(&self) -> usize;
    fn max_steps(&self) -> usize;
}
```

3 Environments implementiert:
- **CartPole** — Pure Rust, keine Dependencies
- **LunarLander** — Box2D via `wrapped2d` crate (cmake + g++ Build-Deps)
- **BipedalWalker** — Box2D mit Terrain, LIDAR, 4 Revolute Joints

### Policy Network (`policy.rs`)
- Feedforward-Netz mit konfigurierbaren Hidden Layers
- Tanh-Aktivierung (hidden), je nach Env: Softmax (diskret) oder Tanh (kontinuierlich)
- `from_params()` / `to_params()` für CMA-ES Integration

### Optimizer (`optim.rs`)
- CMA-ES mit voller Kovarianzmatrix
- Eigendekomposition für Sampling
- Rank-μ Update

### Methoden
7 Methoden portiert:
1. **CMA-ES** — Standard
2. **OpenAI-ES** — Antithetisches Sampling
3. **Curriculum** — Reward Shaping + Difficulty Ramp
4. **Neuromod** — Plastische Gewichte
5. **Island Model** — 4 Populationen + Migration
6. **Meta-Learning** — Evolve Gewichte + Lernregeln
7. **Scaling Test** — Verschiedene Netzgrößen

### Parallelisierung (Rayon)
- `rayon::par_iter()` für Population-Evaluation
- Ein Environment pro Thread (Box2D nicht thread-safe)
- Automatisch auf alle CPU-Kerne verteilt

## CLI

```bash
# Benchmark einzelner Environments
gaia-worker --bench cartpole
gaia-worker --bench lunarlander --bench-evals 50000
gaia-worker --bench bipedal

# Normaler Worker-Modus (Jobs vom Server)
gaia-worker --server https://gaia.kndl.at --token gaia2026
```

Worker erkennt automatisch `env + method` → native Rust. Unsupported Kombinationen → Fehlermeldung.

## Build

```bash
# Linux
cargo build --release

# Windows Cross-Compile (auf Linux)
cargo build --release --target x86_64-pc-windows-gnu
```

**Dependencies:** `cmake`, `g++` (für wrapped2d/Box2D)

## Key Learnings

- **Environment trait ist nicht Send** — Box2D Worlds enthalten Raw Pointers. Lösung: ein Environment pro Thread erstellen.
- **wrapped2d Joint API** — `MetaJoint<U>` deref zu `UnknownJoint` enum. Pattern Matching statt `downcast_mut()`.
- **experiments.tar.gz Bundle-System obsolet** — Worker braucht kein Python/Bundle mehr.

## Status

- ✅ CartPole, LunarLander: voll funktional + benchmarked
- ✅ Cross-Compilation Linux + Windows
- 🔧 BipedalWalker: Terrain + LIDAR + Joints implementiert, Joint-Observation-Reading noch offen
- ⏳ Native Worker-Mode (Jobs direkt in Rust statt Python) — noch nicht integriert
