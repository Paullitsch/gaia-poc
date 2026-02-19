# GAIA — Global Artificial Intelligence Architecture

> Gradientenfreie Optimierung als Alternative zur Backpropagation

**Status:** Phase 7 abgeschlossen — LunarLander GELÖST (+274.0) ✅

---

## 🎯 Projektziel

Beweisen, dass neuronale Netze **ohne Backpropagation** trainiert werden können — und eine verteilte Infrastruktur bauen, die das auf beliebig vielen Maschinen parallelisiert.

## 📊 Ergebnisse

| Methode | Best Score | Status |
|---------|-----------|--------|
| Curriculum + CMA-ES | **+274.0** | ✅ SOLVED |
| CMA-ES | **+235.3** | ✅ SOLVED |
| OpenAI-ES | **+206.6** | ✅ SOLVED |
| Indirect Encoding | -9.4 | ❌ |

## 📚 Wiki-Seiten

### Theorie & Forschung
- [[Hypothesen-Evolution]] — Von v1 bis v4
- [[Experimentelle Phasen]] — Alle 7 Phasen im Detail
- [[Epistemische Architektur]] — Was wir wissen vs. vermuten
- [[Methoden]] — CMA-ES, OpenAI-ES, Forward-Forward, Neuromodulation

### Infrastruktur
- [[Architektur]] — Server-Worker-System
- [[Server API]] — REST Endpoints
- [[Dashboard]] — Web UI Features
- [[Deployment]] — Docker, Binaries, Setup

### Roadmap
- [[Phase 8 Plan]] — GPU-Accelerated Experiments
- [[Scaling Hypothesen]] — Wo liegen die Grenzen?

---

**Repository:** https://github.com/Paullitsch/gaia-poc
**Dashboard:** https://gaia.kndl.at/
**Lizenz:** MIT
