# GAIA — Global Artificial Intelligence Architecture

> Gradientenfreie Optimierung als Alternative zur Backpropagation

**Status:** Phase 10 — Atari + GPU Acceleration 🎮

---

## 🎯 Projektziel

Beweisen, dass neuronale Netze **ohne Backpropagation** trainiert werden können — und eine verteilte Infrastruktur bauen, die das auf beliebig vielen Maschinen parallelisiert.

## 📊 Ergebnisse

### LunarLander-v3 — 9/10 Methoden gelöst ✅

| Methode | Best Score | Evals | Backprop? |
|---------|-----------|-------|-----------|
| 🏆 Curriculum CMA-ES | **+341.9** | 8K | ❌ |
| Neuromod CMA-ES | **+264.5** | 13K | ❌ |
| Neuromod Island | **+256.3** | 48K | ❌ |
| CMA-ES | **+235.3** | 12K | ❌ |
| Island Model | **+235.0** | 46K | ❌ |
| GPU CMA-ES | **+232.5** | 17K | ❌ |
| Scaling (XL) | **+215.5** | 12K | ❌ |
| Hybrid CMA+FF | **+209.5** | 9K | ❌ |
| OpenAI-ES | **+206.6** | 56K | ❌ |
| Island Advanced | **+201.7** | 70K | ❌ |
| Indirect Encoding | +9.1 | — | ❌ |
| PPO (Baseline) | +264.8 | — | ✅ |

### BipedalWalker-v3 — GELÖST ✅

| Methode | Best Score | Evals |
|---------|-----------|-------|
| 🏆 CMA-ES | **+566.6** | 40K |
| Curriculum CMA-ES | **+338.5** | — |
| CMA-ES (standard) | **+265.9** | 8K |

## 📚 Wiki-Seiten

### Theorie & Forschung
- [[Hypothesen-Evolution]] — Von v1 bis v4
- [[Experimentelle Phasen]] — Phase 1-10 im Detail
- [[Epistemische Architektur]] — Was wir wissen vs. vermuten
- [[Methoden]] — Alle 11 Methoden erklärt

### Infrastruktur
- [[Architektur]] — Server-Worker-System
- [[Server API]] — REST Endpoints
- [[Deployment]] — Docker, Binaries, Setup
- [[Auto-Update System]] — Self-Updating Worker

### Phasen
- [[Phase 8 Plan]] — BipedalWalker + Auto-Update
- [[Phase 9 Dezentralisierung]] — Island Model + P2P Gossip
- [[Phase 10 Atari]] — CNN + GPU Acceleration
- [[Scaling Hypothesen]] — Wo liegen die Grenzen?

---

**Repository:** https://github.com/Paullitsch/gaia-poc
**Dashboard:** https://gaia.kndl.at/
**Lizenz:** MIT
