# Phase 9: Dezentralisierung

> Island Model + P2P Gossip Protocol

## Ziel

Dezentrale Optimierung: mehrere Populationen arbeiten unabhängig und tauschen periodisch ihre besten Lösungen aus — wie biologische Evolution auf isolierten Inseln.

## Ergebnisse

### LunarLander-v3 — 9/10 Methoden gelöst

| Methode | Score | Evals | Status |
|---------|-------|-------|--------|
| Curriculum CMA-ES | +341.9 | 8K | ✅ 🏆 |
| Neuromod CMA-ES | +264.5 | 13K | ✅ |
| Neuromod Island | +256.3 | 48K | ✅ |
| CMA-ES | +235.3 | 12K | ✅ |
| Island Model | +235.0 | 46K | ✅ |
| GPU CMA-ES | +232.5 | 17K | ✅ |
| Scaling (XL) | +215.5 | 12K | ✅ |
| Hybrid CMA+FF | +209.5 | 9K | ✅ |
| OpenAI-ES | +206.6 | 56K | ✅ |
| Island Advanced | +201.7 | 70K | ✅ |
| Indirect Encoding | +9.1 | — | ❌ |

### Key Findings

- **Neuromod Island (256.3) > Neuromod standalone (245.4) > Islands standalone (212)** — Kombination ist stärker als die Einzelteile
- **Island Model löst, aber braucht ~4x Evals** — erwartbar bei 4 Islands × Population
- **Curriculum ist sample-effizienteste Methode** — nur 8K Evals zum Lösen

## Island Model

4 CMA-ES Populationen mit unterschiedlichen Sigmas:
- Island 1: σ=0.3 (präzise Suche)
- Island 2: σ=0.5 (standard)
- Island 3: σ=0.8 (breite Suche)
- Island 4: σ=1.2 (chaotische Exploration)

Migration alle 10 Generationen: bestes Individuum → nächste Insel.

## P2P Gossip Protocol

`gaia-protocol` Crate — opt-in mit `--gossip` Flag:
- Peer Discovery via Seed-Nodes
- Job Broadcasting
- Model Sharing (beste Parameter zwischen Peers)
- Result Streaming

## Benchmark-Architektur

In Phase 9 wurde die gesamte Experiment-Architektur umgebaut:
- **Alle 11 Methoden sind environment-agnostisch** — laufen auf jedem Gymnasium-Env
- **Shared PolicyNetwork + evaluate()** aus `cma_es.py` (DRY)
- **Jobs haben `environment` Feld** — Server routet korrekt
- **Dashboard: Benchmarks-Tab** mit per-Environment Leaderboards

## Worker v0.5.9

- Experiment Hot-Reload vor jedem Job (kein Restart nötig)
- Binary Auto-Update mit SHA-256 Verifikation
- Force-Update: Server flaggt Worker für sofortiges Update
