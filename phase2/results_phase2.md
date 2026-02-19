# GAIA Phase 2 Results: LunarLander-v3

## Setup
- **Environment:** LunarLander-v3 (8-dim continuous state, 4 discrete actions)
- **Network:** [8, 64, 64, 32, 4] = 6,948 parameters per agent
- **Population:** 100 agents, 50 generations, 2 eval episodes each
- **Total evaluations per method:** 10,000 episodes
- **Solved threshold:** reward > 200

## Results Summary

| Method | Best Fitness | Mean (final gen) | Solved? |
|--------|-------------|-----------------|---------|
| Pure Evolution | **-5.6** | -202 | ✗ No |
| Evo+Hebbian | **18.0** | -184 | ✗ No |
| Evo+Reward-Hebbian | **59.7** | -202 | ✗ No |
| Novelty+Evolution | -25.3 | -354 | ✗ No |
| REINFORCE (backprop) | **-117.0** | -177 | ✗ No |

## Key Findings

### 1. Nothing solved LunarLander
No method came close to the 200 threshold. LunarLander is a genuinely hard problem requiring precise multi-step coordination (thrust timing, angle control, landing). This is a massive step up from CartPole's simple balance task.

### 2. Evo+Reward-Hebbian was the best evolutionary method
Best fitness of **59.7** — significantly better than Pure Evolution (-5.6) and Novelty Search (-25.3). This confirms Phase 1's finding: **reward-gated Hebbian plasticity helps**, even at scale. The agent can adapt its weights during its lifetime based on reward signals, giving it a form of within-episode learning.

### 3. Hebbian learning still helps at larger scale
Evo+Hebbian (18.0) beat Pure Evolution (-5.6). Evo+Reward-Hebbian (59.7) was best overall. The pattern from Phase 1 holds: lifetime plasticity gives evolutionary agents an edge.

### 4. Novelty Search hurt performance
Novelty+Evolution (-25.3) performed worse than Pure Evolution (-5.6). The diversity pressure pulled the population away from exploiting good solutions. With only 50 generations and a hard problem, exploration cost outweighed benefits. Novelty search needs much longer runs to pay off.

### 5. REINFORCE performed worst on best-eval metric
REINFORCE's best eval was -117.0, worse than all evolutionary methods' best agents. However, this is a numpy-only implementation without modern tricks (Adam optimizer, entropy bonus, GAE, etc.). A proper PPO implementation would likely crush all evolutionary methods.

### 6. Evolutionary methods find rare good individuals
The gap between best and mean fitness is huge (~200+ points). Evolution found a few decent agents but couldn't lift the population average. This is characteristic of evolutionary methods on hard problems — they're good at finding lucky outliers, bad at systematic improvement.

## Compute Comparison
| Method | Wall time | Total episodes |
|--------|-----------|---------------|
| Pure Evolution | 38s | 10,000 |
| Evo+Hebbian | 67s | 10,000 |
| Evo+Reward-Hebbian | 66s | 10,000 |
| Novelty+Evolution | 43s | 10,000 |
| REINFORCE | 18s | 2,000 |

Hebbian methods are ~2x slower due to per-step weight updates.

## Honest Verdict on GAIA Scalability

**The GAIA hypothesis faces serious challenges at this scale.**

- CartPole (Phase 1) was solvable because it's almost trivially easy — random search can solve it. LunarLander requires coordinated sequential decision-making, and evolutionary methods struggle here.
- 50 generations × 100 agents = 5,000 genomes evaluated. This is nowhere near enough for a 6,948-parameter search space. You'd need orders of magnitude more evaluations.
- Hebbian plasticity is the most promising GAIA-style innovation — it adds within-lifetime learning without backprop, which is biologically plausible. But it's a weak learner compared to gradient descent.
- To truly solve LunarLander evolutionarily, you'd likely need: 1000+ generations, larger populations, more sophisticated mutation (CMA-ES), or indirect encodings (HyperNEAT).

**Bottom line:** Evolution + local learning rules is a beautiful idea that works on toy problems. Scaling it to real tasks requires either enormous compute or much cleverer algorithms. Modern RL (PPO, SAC) would solve LunarLander in minutes with a fraction of the compute. The gap widens as problems get harder.

## Plots
- `results/evo_fitness_curves.png` — Evolutionary method fitness curves
- `results/reinforce_curve.png` — REINFORCE learning curve
