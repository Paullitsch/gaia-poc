# GAIA — Global Artificial Intelligence Architecture

> **Training neural networks without backpropagation — in pure Rust.**
> Gradient-free optimization + distributed compute = decentralized AI.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Phase](https://img.shields.io/badge/Phase-10-brightgreen)]()
[![Worker](https://img.shields.io/badge/Worker-v0.8.5-blue)]()
[![Rust](https://img.shields.io/badge/Pure-Rust%20🦀-orange)]()

## 🎯 What is GAIA?

GAIA proves that neural networks can be trained **without backpropagation** using evolutionary and gradient-free optimization methods — all implemented in **pure Rust**. No Python, no PyTorch, no autograd. Manual forward + backward pass, native environments, and 3-14× faster than Python.

### Key Results (v0.8.5 — Pure Rust)

#### LunarLander-v3 — SOLVED ✅ (5/6 gradient-free methods!)

| # | Method | Score | Evals | Time | Backprop? |
|---|--------|-------|-------|------|-----------|
| 1 | 🏆 CMA-ES | **+264.9** | 14.7K | 3.2s | ❌ No |
| 2 | Meta-Learning | **+260.0** | 21K | 4.2s | ❌ No |
| 3 | Curriculum | **+228.0** | 14.7K | 2.9s | ❌ No |
| 4 | Island Model | **+212.8** | 47.5K | 8.3s | ❌ No |
| 5 | Neuromod | **+209.6** | 13K | 2.2s | ❌ No |
| 6 | PPO (baseline) ⚡ | +47.9 | 100K | 37.1s | ✅ Yes |
| 7 | OpenAI-ES | -81.9 | 100K | 11.6s | ❌ No |

> **5 gradient-free methods solve LunarLander. PPO (backpropagation) doesn't.**

#### BipedalWalker-v3 — In Progress

| # | Method | Score | Evals | Time | Backprop? |
|---|--------|-------|-------|------|-----------|
| 1 | Neuromod | **+158.2** | 500K | 36min | ❌ No |
| 2 | Meta-Learning | +37.6 | 500K | 3.8min | ❌ No |
| 3 | OpenAI-ES | -1.7 | 500K | 4.8min | ❌ No |
| 4 | CMA-ES | -85.8 | 500K | 5.4min | ❌ No |

> Neuromod leads at +158. 1M eval runs queued. Previously solved with +566 in Phase 8.

### Rust Speedups 🦀

| Environment | Python | Rust (parallel) | Speedup |
|-------------|--------|-----------------|---------|
| CartPole | 152 evals/s | 2,073 evals/s | **13.6×** |
| LunarLander | ~150 evals/s | solved in 7.1s | **10.4×** |

## 🏗️ Architecture

```
┌──────────────────────────────────────┐
│          GAIA Server (Rust)          │
│  Job Queue · Results · Dashboard     │
│        Benchmarks · Releases         │
│              :7434                   │
├──────────────────────────────────────┤
│       P2P Gossip Protocol :7435      │
└────────┬─────────┬──────────┬────────┘
         │ HTTPS   │          │
    ┌────┴──┐  ┌───┴──┐  ┌───┴──┐
    │Worker │  │Worker│  │Worker│
    │RTX5070│  │ CPU  │  │Cloud │
    └───────┘  └──────┘  └──────┘
```

**Everything is Rust.** Server, worker, experiments, environments, optimizers — zero Python dependency.

- **Server:** Rust/Axum, job orchestration, real-time dashboard, release management
- **Worker:** Rust binary, native experiments, auto-update, 11 methods × 3 environments
- **Environments:** CartPole, LunarLander (Box2D), BipedalWalker (Box2D) — native Rust
- **Optimizers:** CMA-ES, OpenAI-ES, PPO (manual backprop) — native Rust
- **Protocol:** P2P gossip for decentralized job/model sharing

## 🚀 Quick Start

### Run the Server
```bash
docker compose up -d
# Dashboard at http://localhost:7434
```

### Connect a Worker
```bash
./gaia-worker --server https://your-server:7434 --token YOUR_TOKEN --name my-worker --auto-update
```

### Run a Benchmark
```bash
./gaia-worker --bench lunarlander    # CMA-ES on LunarLander
./gaia-worker --bench cartpole       # CMA-ES on CartPole
```

### Submit an Experiment
```bash
curl -X POST http://localhost:7434/api/jobs/submit \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"method": "cma_es", "environment": "LunarLander-v3", "max_evals": 100000, "params": {"sigma0": 0.5, "patience": 200}}'
```

## 📊 Research Phases

| Phase | Focus | Result |
|-------|-------|--------|
| 1-2 | Pure Evolution | ✅ CartPole, ❌ LunarLander |
| 3 | Forward-Forward | 🟡 50-70% of backprop |
| 4 | Meta-Plasticity | 🟡 Beats naive backprop |
| 5 | Neuromodulation | 🟡 +80.0 breakthrough |
| 6 | PPO Baseline | 🟡 PPO: +264.8 (reference) |
| 7 | **CMA-ES + Compute** | **✅ LunarLander SOLVED** |
| 8 | **BipedalWalker** | **✅ BipedalWalker SOLVED (+566)** |
| 9 | **Decentralization** | ✅ Island Model, P2P Gossip |
| 10 | **Rust + Meta-Learning** | 🔬 Pure Rust, scaling tests, learning rule evolution |

## 🧬 Methods (11 native Rust)

| Method | Type | Key Idea |
|--------|------|----------|
| `cma_es` | Evolutionary | CMA-ES with patience + restart |
| `openai_es` | Evolutionary | Antithetic sampling + weight decay |
| `curriculum` | Evolutionary | Reward shaping + difficulty ramp |
| `neuromod` | Neuroevolution | CMA-ES + neuromodulatory plasticity |
| `island_model` | Distributed | 4 CMA-ES populations + migration |
| `island_advanced` | Distributed | 6 heterogeneous islands |
| `neuromod_island` | Hybrid | Neuromod + Island Model |
| `meta_learning` | Meta | Evolve weights + learning rules |
| `meta_learning_pure` | Meta | Evolve ONLY learning rules (21 params) |
| `scaling_test` | Experiment | CMA-ES at 1K-500K params |
| `ppo_baseline` | Backprop ⚡ | PPO with manual backprop (control group) |

## 🎮 Environments (3 native Rust)

| Environment | Obs | Actions | Solved | Rust? |
|-------------|-----|---------|--------|-------|
| CartPole-v1 | 4D | Discrete(2) | ≥475 | ✅ |
| LunarLander-v3 | 8D | Discrete(4) | ≥200 | ✅ |
| BipedalWalker-v3 | 24D | Continuous(4) | ≥300 | ✅ |

## 🧠 Singularity Roadmap

| Stage | Goal | Status |
|-------|------|--------|
| 1 | Find scaling law (where does CMA-ES break?) | 🔬 Testing |
| 2 | Hierarchical optimization (ES evolves learning rules) | 🔬 Testing |
| 3 | Decentralized emergence (gossip + local rules) | ⏳ Planned |
| 4 | Open question: can local rules + evolution = intelligence? | ❓ |

## 📝 Whitepapers

- [GAIA v6](GAIA_v6_WhitePaper.md) — 60+ experiments, sample efficiency analysis
- [GAIA v5](GAIA_v5_WhitePaper.md) — Island Model, P2P Protocol
- [GAIA v4](GAIA_v4_WhitePaper.md) — Phase 7 breakthrough
- [Earlier](GAIA_v3_WhitePaper.md) — v2, v3

## 🧠 Core Thesis

> *Gradient-free optimization is not inferior to backpropagation — it's more compute-intensive but inherently parallelizable, decentralizable, and biologically plausible. With meta-learning (evolving learning rules instead of weights), it scales to arbitrary network sizes.*

## License

MIT — see [LICENSE](LICENSE)
