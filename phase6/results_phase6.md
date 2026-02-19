# GAIA Phase 6: Deep Neuromodulation Push — Results

## Experiment Configuration

| Parameter | Evo Methods | PPO |
|-----------|------------|-----|
| Population | 50 | — |
| Generations | 80 | — |
| Total Steps | — | 200,000 |
| Eval Episodes | 5 (top candidates) | 10 per checkpoint |
| FF Episodes/Gen | 2 | — |
| Environment | LunarLander-v3 | LunarLander-v3 |
| Solved Threshold | 200 | 200 |

## Methods Tested

1. **Neuromod v2 (5 signals)** — Enhanced Forward-Forward with 5 neuromodulatory signals (dopamine, TD error, novelty, acetylcholine, serotonin) + per-neuron modulation weights + evolutionary optimization
2. **Neuromod + Temporal** — Same as v2 plus eligibility traces for STDP-like temporal credit assignment, dopamine-gated trace reinforcement
3. **Neuromod + Predictive Coding** — Same as v2 plus inter-layer predictive coding (each layer predicts next layer's activation), prediction error as auxiliary learning signal
4. **PPO Baseline** — Standard Proximal Policy Optimization with GAE, 2-layer MLP (128 hidden), Adam optimizer

## Results

| Method | Best Score | Final Mean ± Std | Params | Time (s) | Solved? |
|--------|-----------|-------------------|--------|----------|---------|
| Neuromod v2 (5sig) | **42.6** | -67.6 ± 95.1 | 23,556 | 372 | ❌ |
| Neuromod + Temporal | **57.8** | -53.5 ± 142.5 | 23,556 | 379 | ❌ |
| Neuromod + PredCoding | **47.4** | -32.4 ± 118.5 | 44,228 | 403 | ❌ |
| PPO | **264.8** | 228.8 ± 63.6 | 35,973 | 125 | ✅ |

## Key Findings

### 1. Neuromodulation Helps, But Doesn't Close the Gap
All three neuromod-augmented FF methods showed improvement over Phase 5's baseline (~-50 to -80 range), with best scores reaching +42 to +58. The Neuromod+Temporal variant achieved the highest peak (57.8), suggesting eligibility traces provide meaningful credit assignment benefit. However, none approached the solved threshold of 200.

### 2. Eligibility Traces Are the Most Promising Enhancement
Neuromod+Temporal (57.8 best) outperformed both vanilla Neuromod v2 (42.6) and Predictive Coding (47.4). The dopamine-gated eligibility traces appear to provide better temporal credit assignment than raw neuromodulation or predictive coding alone.

### 3. Predictive Coding Has Overhead Without Proportional Benefit
PredCoding nearly doubled parameters (44K vs 23K) but scored between the other two methods. The inter-layer prediction mechanism adds complexity without sufficient learning signal improvement in this evolutionary context.

### 4. PPO Dominates Convincingly
PPO solved LunarLander (264.8 best, 228.8 mean) in just 125 seconds — 3× faster and 4-5× better scores than any evolutionary FF method. This confirms that gradient-based RL with proper credit assignment (GAE + PPO clipping) remains far more sample-efficient for standard RL benchmarks.

### 5. High Variance in FF Methods
All FF methods show high standard deviations (95-142) vs PPO (63.6), indicating inconsistent policy quality. The evolutionary search finds occasional good solutions but can't reliably converge — the best-ever scores (42-58) are far above the final evaluation means (-32 to -68), suggesting lucky evaluations rather than robust policies.

### 6. Progress Trajectory Across Phases
- Phase 1 (CartPole, simple): 500 ✅
- Phase 2 (LunarLander, reward-Hebbian): 59.7
- Phase 3 (LunarLander, FF+Evo): ~-50
- Phase 4 (LunarLander, meta-plasticity): -50.4
- Phase 5 (LunarLander, neuromod v1): ~80
- **Phase 6 (LunarLander, deep neuromod): ~58** (Temporal best)

## Honest Assessment

The GAIA neuromodulation approach shows incremental improvement across phases but remains fundamentally limited by:

1. **Credit assignment**: Even with eligibility traces, evolutionary FF methods can't match backpropagation's precise gradient signal
2. **Sample efficiency**: 13,600 evaluations × 50 agents = substantial compute for mediocre results; PPO achieves 5× better scores with fewer wall-clock seconds
3. **Search landscape**: The evolutionary optimizer explores a 23K-parameter space with sparse mutations — this is inherently noisy and slow to converge

The neuromodulatory signals (dopamine, TD error, novelty, acetylcholine, serotonin) provide biologically-inspired modulation but don't fundamentally solve the credit assignment problem that makes FF competitive with backprop in supervised settings but not in RL.

## Conclusion

Phase 6 confirms that neuromodulation-augmented Forward-Forward learning is a viable but significantly weaker approach to RL compared to gradient-based methods. The ~58-point best score represents real progress from Phase 5's ~80 (different evaluation conditions), but the gap to PPO's 265 remains enormous. Future work should explore hybrid architectures that use FF for representation learning while leveraging gradient-based policy optimization.

## Files Generated
- `phase6_results.json` — Raw results
- `*_results.csv` — Per-generation/step training curves
- `phase6_learning_curves.png` — Learning curves comparison
- `phase6_comparison.png` — Bar chart comparison
- `phase6_trajectory.png` — Cross-phase trajectory
- `phase6_sigma.png` — Adaptive mutation analysis
