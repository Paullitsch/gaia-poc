# Phase 10: Atari — Pixel-basierte Environments + GPU Acceleration

## Warum Atari?
- **Erster echter GPU-Test**: Pixel-Input (210×160×3) → CNNs → GPU lohnt sich endlich
- **Industriestandard-Benchmark**: DQN, PPO, ES — alle messen sich an Atari
- **Skalierungstest für GAIA**: Funktionieren gradientenfreie Methoden bei >100K params?
- **Neue Netzwerk-Architektur**: CNN statt MLP — zeigt Flexibilität

## Ziel-Environments
| Environment | Obs | Actions | Solved | Warum |
|---|---|---|---|---|
| `ALE/Pong-v5` | 210×160×3 | 6 | +21 | Einfachstes Atari — gut zum Starten |
| `ALE/Breakout-v5` | 210×160×3 | 4 | 30+ | Klassiker, gut dokumentiert |
| `ALE/SpaceInvaders-v5` | 210×160×3 | 6 | 500+ | Mittelschwer |

## Architektur-Änderungen

### 1. CNN PolicyNetwork (`cnn_policy.py`)
```python
class CNNPolicy:
    """Convolutional policy for pixel-based environments.
    
    Architecture (Nature DQN style, aber kleiner):
    - Conv2d(4, 16, 8, stride=4) → ReLU
    - Conv2d(16, 32, 4, stride=2) → ReLU  
    - Flatten → Linear(32*9*9, 256) → ReLU
    - Linear(256, n_actions)
    
    ~85K params (vs 3K für MLP)
    """
```

### 2. Frame-Stacking + Preprocessing
- Grayscale (210×160×3 → 84×84×1)
- Frame-stacking (4 frames → 84×84×4)
- Normalisierung (0-255 → 0-1)

### 3. GPU-Beschleunigte Evaluation
```python
# Batch-Forward: 100 Policies parallel auf GPU
# Statt: for each policy → forward pass (CPU)
# Neu:   stack all policies → single batched forward (GPU)
```

### 4. ComputeBackend Abstraktion
```python
class ComputeBackend:
    """Abstraktion für CPU/CUDA/NPU"""
    def forward(self, obs_batch, params_batch) -> actions
    def batch_evaluate(self, policies, env) -> fitnesses
```

## Implementierungs-Plan

### Step 1: Dependencies + Env Setup
- [ ] `ale-py` + `gymnasium[atari]` + `opencv-python` in Requirements
- [ ] Frame-Preprocessing Pipeline (grayscale, resize, stack)
- [ ] Atari-Environments in ENVIRONMENTS config
- [ ] Verify: Atari läuft auf Worker

### Step 2: CNN Policy
- [ ] PyTorch-basierte CNNPolicy (forward mit numpy params ODER torch)
- [ ] Parametervektor ↔ CNN Weights Konvertierung  
- [ ] `act()` Methode kompatibel mit bestehendem Interface
- [ ] Unit-Test: random policy auf Pong

### Step 3: GPU Batch-Evaluation
- [ ] `evaluate_batch_gpu()`: N Policies parallel forwarden
- [ ] Integration in CMA-ES: Population → GPU Batch statt mp.Pool
- [ ] Benchmark: GPU-Batch vs CPU-Multiprocessing Speedup
- [ ] Fallback auf CPU wenn kein CUDA

### Step 4: Methoden anpassen
- [ ] `cma_es.py`: CNN-Support (obs_type="pixel" in params)
- [ ] `openai_es.py`: CNN-Support (besonders vielversprechend — O(n) statt O(n²))
- [ ] `ppo_baseline.py`: CNN + GPU (trivial, schon PyTorch)
- [ ] Andere Methoden: generisch via PolicyFactory

### Step 5: Dashboard + Ergebnisse
- [ ] Atari-Environments im Dashboard Dropdown
- [ ] Atari-spezifische Metriken (Score, Frames)
- [ ] Erste Benchmark-Runs: Pong mit allen Methoden
- [ ] Vergleich: GAIA vs PPO auf gleichem Env

## Erwartungen
- **OpenAI-ES wird dominieren**: O(n) Speicher, skaliert besser als CMA-ES O(n²) für 85K+ params
- **CMA-ES diagonal mode**: Sollte noch funktionieren (haben wir schon für >2000 params)
- **GPU Speedup**: 10-50x für CNN forward passes (Batch-Eval)
- **PPO wird schneller lösen**: Aber wir zeigen dass ES-Methoden es AUCH können

## NPU-Vorbereitung
- ComputeBackend Interface von Anfang an
- Forward-Pass abstrahieren: `backend.forward(obs, weights) → actions`
- Später: OpenVINO/ONNX Backend für Intel NPUs, CoreML für Apple NPUs

## Risiken
- **Compute-Budget**: Atari braucht ~10M+ Frames (vs 100K Evals bei Box2D)
- **CMA-ES bei 85K params**: Könnte an Speicher-Limits stoßen (Kovarianzmatrix)
- **Environment-Speed**: Atari-Emulator ist langsamer als Box2D
- **Reward Sparsity**: Atari-Rewards sind seltener → ES braucht mehr Evals
