# Experimentelle Phasen

## Übersicht

| Phase | Methode | Aufgabe | Best Score | Ergebnis |
|-------|---------|---------|-----------|----------|
| 1 | Reine Evolution | CartPole | 500/500 | ✅ Gelöst |
| 2 | Evolution + Hebbisch | LunarLander | +59.7 | ❌ Skaliert nicht |
| 3 | Forward-Forward | LunarLander | 50-70% v. Backprop | 🟡 Teilweise |
| 4 | Meta-Plastizität | LunarLander | -50.4 | 🟡 Schlägt naive BP |
| 5 | Neuromodulation | LunarLander | +80.0 | 🟡 Durchbruch |
| 6 | Deep Neuromod + PPO | LunarLander | +57.8 / +264.8 | 🟡 PPO gewinnt |
| 7 | CMA-ES + Compute | LunarLander | **+274.0** | ✅ **GELÖST** |
| 8 | BipedalWalker + Infra | BipedalWalker | **+566.6** | ✅ **GELÖST** |
| 9 | Dezentralisierung | Island Model + P2P | **+256.3** | ✅ Abgeschlossen |
| 10 | Scaling + Meta-Learning | Benchmarks + Rust | 🔬 | In Arbeit |

## Phase 1: CartPole (722 Parameter)

**Frage:** Kann Evolution neuronale Netze trainieren?
**Antwort:** Ja, aber 20x weniger effizient als Backprop.

Alle evolutionären Varianten (rein, Hebbisch, Reward-Hebbisch) lösten CartPole (500/500). REINFORCE brauchte nur 217 Episoden vs. 4.500 bei Evolution.

## Phase 2: LunarLander (6.948 Parameter)

**Frage:** Skaliert Evolution auf schwerere Probleme?
**Antwort:** Nein. Skalierungswand bei ~7K Parametern.

Bester Score: +59.7 (Reward-Hebbisch). Weit unter dem Lösungsschwellenwert von +200. Die Fitness-Landschaft wird zu komplex für gradientenfreie Suche im Gewichtsraum.

## Phase 3: Forward-Forward (10.000 Parameter)

**Frage:** Können lokale Lernregeln Backprop ersetzen?
**Antwort:** Sie kommen auf 50-70%.

Hintons Forward-Forward-Algorithmus, erweitert durch evolutionäre Hyperparameter-Optimierung.

## Phase 4: Meta-Plastizität (11.600 Parameter)

**Frage:** Was wenn Evolution Lernregeln statt Gewichte optimiert?
**Antwort:** Schlägt naive Backprop!

Meta-Plastizität (-50.4) übertraf REINFORCE (-158.4). Evolution als Meta-Lernalgorithmus ist der richtige Ansatz.

## Phase 5: Neuromodulation (20.000 Parameter)

**Frage:** Helfen biologisch inspirierte Modulationssignale?
**Antwort:** Dramatischer Durchbruch (+80.0).

Drei Signale (Dopamin, TD-Error, Novität) modulieren schichtenspezifisch die Plastizität. 3x compute-effizienter als Meta-Plastizität.

## Phase 6: Deep Neuromodulation (23K+ Parameter)

**Frage:** Können wir die Neuromodulation vertiefen?
**Antwort:** PPO bleibt überlegen. Die Credit-Assignment-Lücke bleibt.

5 Neuromodulationssignale + Eligibility Traces: +57.8. PPO Baseline: +264.8.

## Phase 7: CMA-ES + Compute (2.788 Parameter) ⭐

**Frage:** Was passiert mit genug Compute?
**Antwort:** GELÖST. +274.0 ohne Backpropagation.

Kleineres Netzwerk (2.788 statt 20K Parameter), aber massiv mehr Compute (100K Evaluierungen). CMA-ES lernt die Kovarianzstruktur und findet optimale Gewichte.

**Schlüsseleinblick:** Das Netzwerk war zu groß, nicht der Algorithmus zu schwach.

## Phase 8: BipedalWalker + Infrastruktur ✅

**Frage:** Skalieren gradientenfreie Methoden auf kontinuierliche Kontrolle?
**Antwort:** Ja! CMA-ES löst BipedalWalker mit +566.6.

BipedalWalker-v3: 24D Observation, 4 kontinuierliche Aktoren, Solved Threshold 300.

### Ergebnisse

| Methode | Best Score | Evals |
|---------|-----------|-------|
| CMA-ES (patience=500) | **+566.6** | 40K |
| Curriculum CMA-ES | **+338.5** | — |
| CMA-ES (standard) | **+265.9** | 8K |

### Infrastruktur-Meilensteine
- **Auto-Update System** (v0.4.0→v0.5.9): Worker aktualisiert sich selbst
- **Experiment-Sync**: Neue Experiments automatisch verteilt
- **Environment-agnostische Methoden**: Alle 11 Methoden laufen auf jedem Env
- **Dashboard**: Benchmarks-Tab, Leaderboard, Learning Curves

## Phase 9: Dezentralisierung ✅

**Frage:** Kann dezentrale Evolution mithalten?
**Antwort:** Ja — Neuromod Island (+256.3) übertrifft Einzelpopulationen.

### Kernresultate
- **Island Model** mit 4 CMA-ES Populationen + Migration
- **P2P Gossip Protocol** implementiert (Port 7435)
- **Neuromod Island** (+256.3) > Neuromod standalone (+264.5 solo, +217.6 Benchmark) > Islands (+175.9)
- Kombination von lokalen Lernregeln + Populationsdynamik ist stärker als beides allein

### Benchmark Architecture Refactor
- **Environment + Method getrennt**: Jobs haben `environment` Feld
- **Shared PolicyNetwork + evaluate()**: Alle Methoden importieren aus `cma_es.py`
- **PPO Baseline** als Backprop-Kontrollgruppe
- Dashboard zeigt 🧬 GRAD-FREE vs ⚡ BACKPROP Badges

## Phase 10: Scaling + Meta-Learning + Rust 🔬

Phase 10 hat drei Stränge:

### Strang 1: Atari (deprioritized)
CNN Policy + GPU Batch Evaluation für Pixel-Envs implementiert. **Erkenntnis:** `env.step()` ist der Bottleneck, nicht GPU. Atari war eine Ablenkung — der Weg führt über Meta-Learning, nicht größere Netze.

### Strang 2: Skalierungstests + Meta-Learning
- **Scaling Tests**: CMA-ES bei 1K→100K Parametern (LunarLander alle gelöst → zu einfach)
- **BipedalWalker Scaling**: Tests laufen (1K, 10K, 33K, 100K params)
- **Meta-Learning**: ES evolves Lernregeln statt Gewichte → biologischer Ansatz
- **Pure Meta-Learning**: Nur 21 Lernregel-Parameter, zufällige Gewichts-Init

### Strang 3: Rust-Migration 🦀
- **Worker v0.7.0**: Alle Python-Experimente durch native Rust ersetzt
- 7 Methoden + 3 Environments in Rust portiert
- Speedups: CartPole 13.6×, LunarLander 10.4× (parallel)
- Rayon für parallele Population-Evaluation
- Details: [[Rust-Migration]]

### KEY INSIGHT
> Der Weg zur Singularität ist nicht größere Netze mit ES, sondern ES evolves Lernregeln die Netze trainieren. — Das ist der biologische Weg.
