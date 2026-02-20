# Phase 10: Atari + GPU Acceleration

> ⚠️ **DEPRIORITIZED** — `env.step()` (CPU-bound ALE Emulator) ist der Bottleneck, nicht GPU. Atari war eine Ablenkung. Fokus liegt jetzt auf Meta-Learning + Rust-Migration.

## Motivation (ursprünglich)

Bisherige Envs (LunarLander, BipedalWalker) sind CPU-bound: kleine Netze (2K-10K params), Bottleneck ist `gym.step()` (Box2D Physik). GPU bringt da nichts.

Atari ändert alles:
- **Pixel-Input** (210×160×3) → braucht CNNs → GPU lohnt sich
- **~85K+ params** → Matrix-Multiplikationen werden relevant
- **Industriestandard** — DQN, PPO, ES messen sich alle an Atari

## Environments

| Environment | Obs | Actions | Solved | Schwierigkeit |
|---|---|---|---|---|
| ALE/Pong-v5 | 84×84×4 | 6 | ≥21 | Einstieg |
| ALE/Breakout-v5 | 84×84×4 | 4 | ≥30 | Mittel |
| ALE/SpaceInvaders-v5 | 84×84×4 | 6 | ≥500 | Schwer |

## Architektur

### CNN Policy

Nature DQN-Style, aber kleiner:
```
Conv2d(4, 16, 8, stride=4) → ReLU   [84→20]
Conv2d(16, 32, 4, stride=2) → ReLU  [20→9]
Flatten → 2592
Linear(2592, 256) → ReLU
Linear(256, n_actions)
```
~85K params

### Frame Preprocessing
1. RGB → Grayscale
2. Resize 210×160 → 84×84
3. Stack 4 Frames → (4, 84, 84)
4. Normalize 0-255 → 0-1

### GPU Batch Evaluation
- Population von Parametervektoren → GPU
- Batch-Forward statt sequentiell
- Erwarteter Speedup: 10-50x für CNN forwards

## Erwartungen

| Methode | Prognose | Warum |
|---------|----------|-------|
| OpenAI-ES | ⭐ Beste Chance | O(n) Speicher, skaliert bei 85K params |
| CMA-ES (diagonal) | Gut | Diagonal-Modus ab >2K params |
| CMA-ES (full) | ❌ Zu groß | 85K² Kovarianzmatrix = unmöglich |
| PPO | Schnellster | Gradient-basiert, ideal für CNNs |
| Neuromod | Unbekannt | Spannend — plastische CNNs? |

## Status

- [x] CNN Policy implementiert (`cnn_policy.py`)
- [x] Frame-Preprocessing (`atari_eval.py`)
- [x] Atari-Envs in ENVIRONMENTS config
- [x] Dashboard Dropdown erweitert
- [ ] Methoden-Integration (obs_type="pixel" Weiche)
- [ ] Dependencies auf Worker (`ale-py`, `opencv-python`)
- [ ] Erster Pong-Benchmark
- [ ] GPU Batch-Evaluation
- [ ] Ergebnisvergleich GAIA vs PPO

## NPU-Vorbereitung

Von Anfang an abstrahiert:
- `ComputeBackend` Interface für CPU/CUDA/NPU
- Forward-Pass: `backend.forward(obs, weights) → actions`
- Später: OpenVINO (Intel NPU), CoreML (Apple), ONNX Runtime
