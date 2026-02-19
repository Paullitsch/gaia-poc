# Phase 8: GPU-Accelerated Experiments

## Motivation

Phase 7 bewies: gradientenfreie Methoden lösen LunarLander mit genug Compute. Phase 8 testet die Grenzen:

- **Größere Netzwerke:** Wo bricht CMA-ES ein?
- **GPU-beschleunigte Evaluation:** Vectorized Environments auf CUDA
- **Komplexere Umgebungen:** BipedalWalker, Atari
- **Multi-Worker-Skalierung:** Linearer Speedup?

## GPU-Strategie

LunarLander ist CPU-bound (Box2D Physik). Für echte GPU-Nutzung brauchen wir:

### Option 1: Brax (JAX-basiert)
Komplett auf GPU simulierte Physik-Environments. 1000x Speedup möglich.
```
pip install brax jax[cuda]
```
**Pro:** Massiver Speedup
**Con:** Andere Environments als Gymnasium

### Option 2: EnvPool
GPU-beschleunigter Environment-Pool, kompatibel mit Gymnasium.
```
pip install envpool
```
**Pro:** Gymnasium-kompatibel
**Con:** Nicht alle Envs unterstützt

### Option 3: Vectorized Gymnasium
Multiple Envs parallel auf CPU, Neural Network Inference auf GPU (PyTorch).
```python
envs = gymnasium.vector.AsyncVectorEnv([make_env] * 64)
# Batch-Forward-Pass auf GPU
actions = policy.batch_act(obs_batch.to('cuda'))
```
**Pro:** Einfach zu implementieren
**Con:** Environment-Simulation bleibt auf CPU

## Experimentplan

### Experiment 8.1: Netzwerk-Skalierung auf LunarLander

CMA-ES mit verschiedenen Netzwerkgrößen:

| Netzwerk | Parameter | Population | Hypothese |
|----------|-----------|-----------|-----------|
| 8→32→16→4 | 756 | 17 | Leicht lösbar |
| 8→64→32→4 | 2.788 | 27 | Gelöst (Phase 7) |
| 8→128→64→4 | 9.604 | 35 | Grenzbereich |
| 8→256→128→4 | 36.228 | 42 | CMA-ES degradiert? |
| 8→512→256→4 | 140.804 | 50 | Zu groß für CMA-ES? |

**Erwartung:** CMA-ES performt gut bis ~10K Parameter, dann übernimmt OpenAI-ES.

### Experiment 8.2: BipedalWalker-v3

- 24 Observations, 4 continuous Actions
- Deutlich schwerer als LunarLander
- Netzwerk: 24→64→32→4 (2.276 Parameter)
- Budget: 500K Evaluierungen

### Experiment 8.3: Atari (Pong)

- Pixel-Input (210×160×3) → CNN → Policy
- Netzwerk: ~100K+ Parameter
- Braucht GPU für CNN-Inference
- Budget: 1M+ Evaluierungen

### Experiment 8.4: Multi-Worker-Skalierung

- 1, 2, 4, 8 Workers parallel
- Messen: Wallclock-Zeit bis Lösung
- Erwartung: ~linear bei unabhängiger Evaluation

### Experiment 8.5: Neuromod Revisited

Phase 5 Neuromodulation mit 100K+ Evaluierungen:
- War das Compute-Budget (10K Evals) der Bottleneck?
- Kann Neuromod CMA-ES schlagen bei gleichem Compute?

## Infrastruktur-Anforderungen

| Experiment | GPU | CPU Cores | RAM | Geschätzte Zeit |
|-----------|-----|-----------|-----|----------------|
| 8.1 (klein) | Optional | 8+ | 8 GB | 1-2h |
| 8.1 (groß) | Optional | 16+ | 16 GB | 4-8h |
| 8.2 | Optional | 16+ | 16 GB | 4-12h |
| 8.3 | **Nötig** | 16+ | 32 GB | 12-48h |
| 8.4 | Multi-GPU | 32+ | 32 GB | 2-4h |
| 8.5 | Optional | 8+ | 8 GB | 2-4h |

## Erfolgsmetriken

| Experiment | Erfolgskriterium |
|-----------|-----------------|
| 8.1 | Bestimme N_max wo CMA-ES noch funktioniert |
| 8.2 | BipedalWalker Score > 300 (gelöst) |
| 8.3 | Atari Pong Score > 0 (besser als Random) |
| 8.4 | >0.7x linearer Speedup |
| 8.5 | Neuromod vs CMA-ES bei gleichem Budget |
