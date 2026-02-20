# GAIA — Global Artificial Intelligence Architecture

> Gradientenfreie Optimierung als Alternative zur Backpropagation

**Status:** Phase 10 — Meta-Learning + Rust-Migration + Skalierungstests 🧬🦀

---

## 🎯 Projektziel

Beweisen, dass neuronale Netze **ohne Backpropagation** trainiert werden können — und eine verteilte Infrastruktur bauen, die das auf beliebig vielen Maschinen parallelisiert.

## 🗺️ Singularity Roadmap

| Stufe | Ziel | Status |
|-------|------|--------|
| 1 | Skalierungsgesetz finden (wo brechen ES-Methoden?) | 🔬 In Arbeit |
| 2 | Hierarchische Optimierung (ES evolves Lernregeln) | 🔬 In Arbeit |
| 3 | Dezentrale Emergenz (Gossip + lokale Regeln) | ⏳ Geplant |
| 4 | Offene Frage — reicht das für Intelligenz? | ❓ |

## 📊 Benchmark-Ergebnisse

### LunarLander-v3 — 7/11 Methoden gelöst ✅

| Methode | Best Score | Evals | Backprop? |
|---------|-----------|-------|-----------|
| 🏆 Curriculum CMA-ES | **+790.1** | 100K | ❌ |
| Meta-Learning | **+245.2** | 100K | ❌ |
| Scaling (10K params) | **+227.2** | 100K | ❌ |
| Scaling (100K params) | **+225.0** | 100K | ❌ |
| Neuromod | **+217.6** | 100K | ❌ |
| Scaling (1K params) | **+215.1** | 100K | ❌ |
| CMA-ES | **+214.4** | 100K | ❌ |
| Scaling (33K params) | **+204.5** | 100K | ❌ |
| Neuromod Island | **+200.3** | 100K | ❌ |
| Island Model | +175.9 | 100K | ❌ |
| OpenAI-ES | +73.4 | 100K | ❌ |
| PPO (Baseline) ⚡ | +59.7 | 100K | ✅ |

### BipedalWalker-v3

| Methode | Best Score | Evals | Backprop? |
|---------|-----------|-------|-----------|
| CMA-ES (patience=500) | **+426.2** | 11K | ❌ |
| PPO (Baseline) ⚡ | +145.9 | 500K | ✅ |
| Island Model | +6.5 | 500K | ❌ |
| CMA-ES (standard) | -48.6 | 500K | ❌ |

> **Pending:** 8 weitere Jobs laufen (CMA-ES 500K, Scaling 1K-100K, Meta-Learning, Pure Meta-Learning)

### Scaling Tests (LunarLander)

| Netzgröße | Params | Score | Ergebnis |
|-----------|--------|-------|----------|
| 1K | 1.000 | +215.1 | ✅ Gelöst |
| 10K | 10.000 | +227.2 | ✅ Gelöst |
| 33K | 33.000 | +204.5 | ✅ Gelöst |
| 100K | 100.000 | +225.0 | ✅ Gelöst |

→ LunarLander zu einfach für Breakpoint-Suche. Scaling-Tests jetzt auf BipedalWalker.

### Rust Speedups 🦀

| Environment | Python | Rust | Speedup |
|-------------|--------|------|---------|
| CartPole | 152 evals/s | 2.073 evals/s | **13.6×** |
| LunarLander | ~150 evals/s | ~550 evals/s | **3.6×** |
| LunarLander (4 threads) | — | solved in 7.1s | **10.4×** |

## 📚 Wiki-Seiten

### Theorie & Forschung
- [[Hypothesen-Evolution]] — Von v1 bis v4
- [[Experimentelle Phasen]] — Phase 1-10 im Detail
- [[Methoden]] — Alle 14 Methoden erklärt
- [[Epistemische Architektur]] — Was wir wissen vs. vermuten
- [[Meta-Learning]] — Evolution von Lernregeln

### Infrastruktur
- [[Architektur]] — Server-Worker-System
- [[Server API]] — REST Endpoints
- [[Deployment]] — Docker, Binaries, Setup
- [[Auto-Update System]] — Self-Updating Worker
- [[Rust-Migration]] — Pure Rust Worker (v0.7.0)

### Phasen & Analyse
- [[Phase 8 Plan]] — BipedalWalker + Auto-Update
- [[Phase 9 Dezentralisierung]] — Island Model + P2P Gossip
- [[Phase 10 Atari]] — CNN + GPU (deprioritized)
- [[Scaling Hypothesen]] — Wo liegen die Grenzen?
- [[Benchmark-Ergebnisse]] — Systematische Vergleiche

---

**Repository:** https://github.com/Paullitsch/gaia-poc
**Dashboard:** https://gaia.kndl.at/
**Lizenz:** MIT
