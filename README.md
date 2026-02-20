# GAIA — Global Artificial Intelligence Architecture

> **Training neural networks without backpropagation.**
> Gradient-free optimization + distributed compute = decentralized AI.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Phase](https://img.shields.io/badge/Phase-10-brightgreen)]()

## 🎯 What is GAIA?

GAIA proves that neural networks can be trained **without backpropagation** using evolutionary and gradient-free optimization methods. We built a distributed compute infrastructure that connects heterogeneous hardware (GPUs, CPUs, cloud) to run these experiments at scale.

### Key Results

#### LunarLander-v3 — SOLVED ✅ (9/10 methods)

| Method | Score | Evals | Backprop? |
|--------|-------|-------|-----------|
| 🏆 Curriculum + CMA-ES | **+341.9** | 8K | ❌ No |
| Neuromod CMA-ES | **+264.5** | 13K | ❌ No |
| Neuromod Island | **+256.3** | 48K | ❌ No |
| CMA-ES | **+235.3** | 12K | ❌ No |
| Island Model | **+235.0** | 46K | ❌ No |
| PPO (baseline) | +264.8 | — | ✅ Yes |

#### BipedalWalker-v3 — SOLVED ✅

| Method | Score | Evals | Backprop? |
|--------|-------|-------|-----------|
| 🏆 CMA-ES | **+566.6** | 40K | ❌ No |
| Curriculum CMA-ES | **+338.5** | — | ❌ No |

**Our best gradient-free methods match or exceed PPO.** No gradients, no computational graph, no backpropagation.

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

- **Server:** Rust/Axum, job orchestration, real-time dashboard, release management
- **Worker:** Rust binary, connects outbound (no open ports), auto-update, experiment hot-reload
- **Experiments:** Python (NumPy + optional PyTorch), multiprocessing, GPU-ready
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

### Submit an Experiment
```bash
curl -X POST http://localhost:7434/api/jobs/submit \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"method": "cma_es", "environment": "LunarLander-v3", "max_evals": 100000}'
```

## 📊 Research Phases

| Phase | Focus | Result |
|-------|-------|--------|
| 1-2 | Pure Evolution | ✅ CartPole, ❌ LunarLander (doesn't scale) |
| 3 | Forward-Forward | 🟡 50-70% of backprop |
| 4 | Meta-Plasticity | 🟡 Beats naive backprop |
| 5 | Neuromodulation | 🟡 +80.0 breakthrough |
| 6 | PPO Baseline | 🟡 PPO: +264.8 (reference) |
| 7 | **CMA-ES + Compute** | **✅ LunarLander SOLVED (+341.9)** |
| 8 | **BipedalWalker + Infra** | **✅ BipedalWalker SOLVED (+566.6)** |
| 9 | **Decentralization** | ✅ Island Model, P2P Gossip, 9/10 methods solved |
| 10 | **Atari + GPU** | 🔄 CNN policies, pixel-based envs, GPU acceleration |

## 🧬 Methods (all environment-agnostic)

All methods work on any Gymnasium environment — MLP for vector obs, CNN for pixel obs.

| Method | Type | Key Idea |
|--------|------|----------|
| `cma_es` | Evolutionary | Gold standard gradient-free optimization |
| `openai_es` | Evolutionary | Antithetic sampling, O(n) memory |
| `curriculum` | Evolutionary | Reward shaping + difficulty ramp |
| `neuromod` | Neuroevolution | CMA-ES + neuromodulatory plasticity |
| `neuromod_island` | Hybrid | Neuromod + Island Model |
| `island_model` | Distributed | 4 CMA-ES populations + migration |
| `island_advanced` | Distributed | 6 heterogeneous islands + adaptive migration |
| `hybrid_cma_ff` | Hybrid | CMA-ES + Forward-Forward local learning |
| `indirect_encoding` | Developmental | CPPN generates policy weights |
| `scaling` | Experiment | Network size scaling test |
| `ppo_baseline` | Backprop ⚠️ | PPO reference (control group) |

## 🎮 Supported Environments

| Environment | Obs | Type | Solved |
|-------------|-----|------|--------|
| LunarLander-v3 | 8D vector | Discrete | ≥200 |
| BipedalWalker-v3 | 24D vector | Continuous | ≥300 |
| BipedalWalkerHardcore-v3 | 24D vector | Continuous | ≥300 |
| ALE/Pong-v5 | 84×84×4 pixels | Discrete | ≥21 |
| ALE/Breakout-v5 | 84×84×4 pixels | Discrete | ≥30 |
| ALE/SpaceInvaders-v5 | 84×84×4 pixels | Discrete | ≥500 |

## 📝 Whitepapers

- [GAIA v6 WhitePaper](GAIA_v6_WhitePaper.md) — 60+ experiments analysis, sample efficiency
- [GAIA v5 WhitePaper](GAIA_v5_WhitePaper.md) — Island Model, P2P Protocol
- [GAIA v4 WhitePaper](GAIA_v4_WhitePaper.md) — Phase 7 breakthrough
- [Earlier versions](GAIA_v3_WhitePaper.md) — v2, v3

## 🔧 Infrastructure

- **Auto-Update:** Workers self-update binary + experiments on every heartbeat
- **Hot-Reload:** Experiments synced before each job (no restart needed)
- **Force-Update:** Server can flag workers for immediate update
- **GPU Detection:** Workers report GPU info, experiments can use CUDA
- **Early Stopping:** Plateau detection prevents wasted compute

## 🧠 Core Thesis

> *Gradient-free optimization is not fundamentally inferior to backpropagation — it's more compute-intensive, but inherently parallelizable, decentralizable, and biologically plausible. With sufficient distributed compute, the performance gap disappears.*

## License

MIT — see [LICENSE](LICENSE)
