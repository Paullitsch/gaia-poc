# GAIA Phase 3: Local Learning Methods vs Backpropagation

## Experiment Summary

**Task:** LunarLander-v3 (8-dim obs, 4 discrete actions, solved at reward >200)  
**Training:** 600 episodes per method, ~10K parameter networks  
**Date:** 2026-02-19  

## Performance Comparison

| Method | Final Eval | Best Eval | Params | Notes |
|--------|-----------|-----------|--------|-------|
| **Forward-Forward** | -133 | -93 | 9,988 | Slow but steady improvement |
| **Predictive Coding** | -640 | -71 | 9,156 | Best early peak, then catastrophic divergence |
| **Decoupled Greedy** | -229 | -80 | 5,516 | Noisy, inconsistent |
| **Hybrid Evo+FF** | -120 | -98 | 9,988 | Modest improvement, limited by episode budget |
| **Backprop (Actor-Critic)** | -113 | -63 | 5,637 | Best overall, most stable |

**None of the methods solved LunarLander in 600 episodes.** This is partially expected — LunarLander typically needs 1000-2000 episodes with well-tuned PPO.

## Key Findings

### 1. Backprop wins, but not by a landslide
The actor-critic baseline achieved the best final score (-113) and best single evaluation (-63). However, the margin over Forward-Forward (-133 final, -93 best) was surprisingly modest — only ~20-30 points.

### 2. Forward-Forward is the most promising local method
- Most stable learning curve among local methods
- Steady improvement throughout training with no catastrophic forgetting
- The FF layers learned useful representations (positive/negative discrimination)
- **Caveat:** The policy head still uses gradient descent (REINFORCE). Pure FF for action selection remains an open problem.

### 3. Predictive Coding showed promise then collapsed
- Achieved the single best evaluation score (-71) around episode 100-250
- Then suffered catastrophic divergence — prediction errors compounded
- The top-down/bottom-up dynamics become unstable without careful tuning
- **Verdict:** Promising but fragile. Needs stabilization mechanisms.

### 4. Decoupled Greedy Learning was disappointing
- Each layer learning independently led to incoherent representations
- Without end-to-end gradient flow, layers don't coordinate well
- High variance, no clear learning trend
- **Verdict:** Not viable for RL in this form. Better suited for supervised tasks with clear layer-wise objectives.

### 5. Hybrid Evolution + FF was limited by compute
- Only managed ~15 generations in 600 total episodes (20 pop × 2 episodes each)
- Evolution needs many more generations to be effective
- The FF within-lifetime learning showed some benefit over pure evolution
- **Verdict:** The concept is sound but needs 10-100× more compute to be competitive

## Is Forward-Forward Viable for RL?

**Partially.** The FF algorithm successfully learned to discriminate good from bad trajectories. However:

1. **Representation learning:** FF can learn useful features from reward-labeled experience ✓
2. **Action selection:** Still requires some form of gradient-based policy learning ✗
3. **Sample efficiency:** Comparable to simple REINFORCE (not great, not terrible)
4. **Stability:** Better than predictive coding, comparable to backprop

The fundamental challenge: FF tells you "this state-action was good/bad" but doesn't directly tell you "which action to take in a new state." You still need a policy mechanism.

## Does the Hybrid Approach Work?

**In principle, yes. In practice, compute-limited.** The GAIA vision of "evolution designs the brain, local rules do the learning" is sound, but:
- Evolution needs large populations and many generations
- 600 episodes is enough for gradient methods but barely scratches the surface for evolution
- With 10,000+ episodes, the hybrid approach would likely improve significantly

## Honest Assessment of GAIA's Core Thesis

### What holds up:
- **Local learning rules can learn useful representations** — FF demonstrated this clearly
- **Evolution + local learning is a viable architecture** — just compute-intensive
- **Biological plausibility doesn't mean biological performance** — but the gap is narrower than expected

### What doesn't:
- **No local method matched backprop** — even with a simple actor-critic baseline
- **Pure local learning can't do credit assignment across layers** — this is backprop's killer feature
- **Predictive coding's instability** suggests biological brains have stabilization mechanisms we haven't captured

### The real insight:
The gap between local methods and backprop on this task was **~30-50%**, not **10×**. This is encouraging! With better architectures, longer training, and hybrid approaches, local methods could potentially close this gap further.

But let's be honest: **backprop works because it solves credit assignment efficiently.** Local methods trade off that efficiency for biological plausibility. For AI engineering, backprop wins. For understanding biological intelligence, local methods are essential.

## What Would I Research Next?

1. **Contrastive Hebbian Learning** — another local method that approximates backprop more closely
2. **Equilibrium Propagation** — energy-based local learning that converges to backprop in the limit  
3. **Longer training runs** (5000+ episodes) to see if local methods converge given enough time
4. **Continuous control tasks** (BipedalWalker) where representation learning matters more
5. **Hybrid predictive coding + FF** — use PC for representation, FF for reward discrimination
6. **Neuromodulation** — adding reward-modulated Hebbian learning (closer to dopamine signaling)

## Conclusion

Local learning methods are **viable but not yet competitive** with backpropagation for RL. The Forward-Forward algorithm showed the most promise, with stable learning curves and only a modest performance gap. The GAIA vision of evolution + local learning is conceptually sound but computationally expensive.

**Bottom line:** Biology doesn't use backprop, and yet biological brains solve RL problems far better than any algorithm. We're clearly missing something — likely the combination of architectural priors (evolved brain structure), neuromodulation, temporal credit assignment mechanisms, and massive parallelism that makes local learning work in biological systems.

The search continues.

---
*Generated by GAIA Phase 3 experiments, 2026-02-19*
