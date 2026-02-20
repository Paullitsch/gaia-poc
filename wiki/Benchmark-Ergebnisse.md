# Benchmark-Ergebnisse

> Systematische Vergleiche: 12 abgeschlossene Jobs + 8 laufende

## Testaufbau

- **Budget:** LunarLander 100K Evals, BipedalWalker 500K Evals
- **Netzwerk:** Environment-abhängig (Standard hidden [64, 64])
- **Worker:** v0.6.0 (Python), v0.7.0 (Rust)
- **Baseline:** PPO mit gleicher Netzwerk-Architektur

## LunarLander-v3 (Solved: >200)

| # | Methode | Score | Status | Backprop? |
|---|---------|-------|--------|-----------|
| 1 | Curriculum CMA-ES | **+790.1** | ✅ Gelöst | ❌ |
| 2 | Meta-Learning | **+245.2** | ✅ Gelöst | ❌ |
| 3 | Scaling (10K) | **+227.2** | ✅ Gelöst | ❌ |
| 4 | Scaling (100K) | **+225.0** | ✅ Gelöst | ❌ |
| 5 | Neuromod | **+217.6** | ✅ Gelöst | ❌ |
| 6 | Scaling (1K) | **+215.1** | ✅ Gelöst | ❌ |
| 7 | CMA-ES | **+214.4** | ✅ Gelöst | ❌ |
| 8 | Scaling (33K) | **+204.5** | ✅ Gelöst | ❌ |
| 9 | Neuromod Island | **+200.3** | ✅ Gelöst | ❌ |
| 10 | Island Model | +175.9 | ❌ Knapp | ❌ |
| 11 | OpenAI-ES | +73.4 | ❌ | ❌ |
| — | PPO (Baseline) ⚡ | +59.7 | ❌ | ✅ |

**Highlight:** 7 gradientenfreie Methoden schlagen PPO Baseline!

## BipedalWalker-v3 (Solved: >300)

| # | Methode | Score | Status | Backprop? |
|---|---------|-------|--------|-----------|
| 1 | CMA-ES (patience=500) | **+426.2** | ✅ Gelöst | ❌ |
| — | PPO (Baseline) ⚡ | +145.9 | ❌ | ✅ |
| 2 | Island Model | +6.5 | ❌ | ❌ |
| 3 | CMA-ES (standard) | -48.6 | ❌ | ❌ |

**Beobachtung:** Standard CMA-ES mit 500K Budget scheitert — braucht patience/early-stopping. PPO ebenfalls unter Threshold.

### Laufende Jobs (BipedalWalker)

| Job | Methode | Params | Budget |
|-----|---------|--------|--------|
| `3c238e6f` | CMA-ES | Standard | 500K |
| `1ea0e653` | Scaling | 1K | 500K |
| `4287d037` | Scaling | 10K | 500K |
| `3f5ad281` | Scaling | 33K | 500K |
| `4ed2bd2b` | Scaling | 100K | 500K |
| `3a2ded7d` | Meta-Learning | Standard | 500K |
| `72a3751b` | Pure Meta-Learning (LL) | 21 | 200K |
| `0ac4f05c` | Pure Meta-Learning (BW) | 21 | 500K |

## Scaling-Analyse

### LunarLander — Kein Breakpoint gefunden

| Params | Score | CMA-ES Modus |
|--------|-------|--------------|
| 1.000 | +215.1 | Full |
| 10.000 | +227.2 | Diagonal |
| 33.000 | +204.5 | Diagonal |
| 100.000 | +225.0 | Diagonal |

→ LunarLander ist zu einfach. Alle Größen lösen es. BipedalWalker-Tests sollen den Breakpoint finden.

## Key Takeaways

1. **CMA-ES dominiert bei kleinen Netzen** — sample-effizienter als alles andere
2. **Curriculum Learning ist game-changing** — +790 vs +214 (standard CMA-ES)
3. **Meta-Learning funktioniert** — +245 zeigt dass Lernregel-Evolution viable ist
4. **PPO ist überraschend schwach** — bei gleichem Budget und gleicher Architektur
5. **BipedalWalker braucht mehr Budget** oder smartere Strategien (patience, curriculum)
6. **Scaling Breakpoint liegt nicht bei LunarLander** — zu einfach für alle Methoden
