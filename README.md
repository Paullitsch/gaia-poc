# GAIA — Global Artificial Intelligence Architecture

> **Training neural networks without backpropagation.**
> Gradient-free optimization + distributed compute = decentralized AI.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Phase](https://img.shields.io/badge/Phase-8-brightgreen)]()

## 🎯 What is GAIA?

GAIA proves that neural networks can be trained **without backpropagation** using evolutionary and gradient-free optimization methods. We built a distributed compute infrastructure that connects heterogeneous hardware (GPUs, CPUs, cloud) to run these experiments at scale.

### Key Result: LunarLander Solved ✅

| Method | Score | Backprop? |
|--------|-------|-----------|
| 🏆 Curriculum + CMA-ES | **+274.0** | ❌ No |
| CMA-ES | **+235.3** | ❌ No |
| OpenAI-ES | **+206.6** | ❌ No |
| PPO (baseline) | +264.8 | ✅ Yes |

**Our best gradient-free method exceeds the PPO baseline.** No gradients, no computational graph, no backpropagation.

## 🏗️ Architecture

```
┌──────────────────────────────────────┐
│          GAIA Server (Rust)          │
│  Job Queue · Results · Dashboard     │
│              :7434                   │
└────────┬─────────┬──────────┬────────┘
         │ HTTPS   │          │
    ┌────┴──┐  ┌───┴──┐  ┌───┴──┐
    │Worker │  │Worker│  │Worker│
    │RTX5070│  │ CPU  │  │Cloud │
    └───────┘  └──────┘  └──────┘
```

- **Server:** Rust/Axum, job orchestration, real-time dashboard, release management
- **Worker:** Rust binary, connects outbound (no open ports), auto-update, experiment sync
- **Experiments:** Python (NumPy + optional PyTorch), multiprocessing across all CPU cores

## 🚀 Quick Start

### Run the Server
```bash
docker compose up -d
# Dashboard at http://localhost:7434
```

### Connect a Worker
```bash
# Download latest worker binary
curl -L https://github.com/Paullitsch/gaia-poc/releases/latest/download/gaia-worker-linux -o gaia-worker
chmod +x gaia-worker

# Start with auto-update
./gaia-worker --server https://your-server:7434 --token YOUR_TOKEN --name my-worker --auto-update
```

### Submit an Experiment
```bash
curl -X POST http://localhost:7434/api/jobs/submit \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"method": "cma_es", "max_evals": 100000}'
```

## 📊 Research Phases

| Phase | Focus | Result |
|-------|-------|--------|
| 1 | Pure Evolution on CartPole | ✅ 500/500 |
| 2 | Evolution on LunarLander | ❌ +59.7 (doesn't scale) |
| 3 | Forward-Forward | 🟡 50-70% of backprop |
| 4 | Meta-Plasticity | 🟡 Beats naive backprop |
| 5 | Neuromodulation | 🟡 +80.0 breakthrough |
| 6 | Deep Neuromod + PPO baseline | 🟡 PPO: +264.8 |
| 7 | **CMA-ES + Compute** | **✅ +274.0 SOLVED** |
| 8 | **BipedalWalker + Infrastructure** | **✅ +338 SOLVED** |

## 🧬 Available Methods

### Phase 7 (LunarLander)
- `cma_es` — CMA-ES (gold standard gradient-free optimization)
- `openai_es` — OpenAI Evolution Strategies with antithetic sampling
- `curriculum` — CMA-ES with reward shaping + curriculum learning
- `hybrid_cma_ff` — CMA-ES + Forward-Forward local learning
- `indirect_encoding` — CPPN-based indirect encoding

### Phase 8 (BipedalWalker)
- `bipedal_cma` — CMA-ES with curriculum for continuous control
- `bipedal_es` — OpenAI-ES for BipedalWalker
- `bipedal_pbt` — Population-Based Training (multiple CMA-ES instances)
- `scaling` — Network scaling experiment (tiny → XL)

## 📝 Whitepapers

- [GAIA v4 WhitePaper](GAIA_v4_WhitePaper.md) — Phase 7 breakthrough + infrastructure
- [GAIA v3 WhitePaper](GAIA_v3_WhitePaper.md) — Neuromodulated meta-plasticity
- [GAIA v2 WhitePaper](GAIA_v2_WhitePaper.md) — Local learning rules

## 🔧 Auto-Update System

Workers update themselves automatically:
1. Server hosts release binaries at `/releases/`
2. Worker checks version on every heartbeat
3. Downloads new binary, verifies SHA-256, self-replaces, restarts
4. Experiment files synced automatically (`--auto-update` enables both)

## 📚 Documentation

- [Wiki](../../wiki) — Full documentation
- [QUICKSTART](QUICKSTART.md) — Setup guide
- [Server API](wiki/Server-API.md) — REST endpoints
- [SECURITY](SECURITY.md) — Security considerations

## 🧠 Core Thesis

> *Gradient-free optimization is not fundamentally inferior to backpropagation — it's more compute-intensive, but inherently parallelizable, decentralizable, and biologically plausible. The performance gap is closed through distributed compute infrastructure.*

## License

MIT — see [LICENSE](LICENSE)
