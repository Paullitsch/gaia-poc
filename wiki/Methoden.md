# Methoden

> Alle Methoden sind environment-agnostisch — laufen auf jedem Gymnasium-Environment.
> Ab v0.7.0 auch nativ in Rust verfügbar (7 Methoden + 3 Environments).

## Gradientenfreie Methoden (GAIA)

### CMA-ES (`cma_es`)
**Covariance Matrix Adaptation Evolution Strategy** — Gold-Standard der gradientenfreien Optimierung.

- Population von Parametervektoren → evaluate → Selektion → Kovarianzmatrix anpassen
- O(n²) Speicher für volle Kovarianzmatrix, Diagonal-Modus ab >2000 params
- **Stärke:** Sehr sample-effizient bei kleinen-mittleren Netzen
- **Schwäche:** Skaliert nicht über ~50K params (Speicher)
- **Rust:** ✅ Portiert

### OpenAI-ES (`openai_es`)
**OpenAI Evolution Strategies** mit antithetischem Sampling.

- Perturbiere Parameter mit Gauß-Rauschen, nutze Reward als "Gradient-Schätzer"
- Antithetisch: teste +ε und -ε → varianzreduziert
- O(n) Speicher — skaliert beliebig
- **Stärke:** Massiv parallelisierbar, skaliert bei großen Netzen
- **Schwäche:** Weniger sample-effizient als CMA-ES
- **Rust:** ✅ Portiert

### Curriculum CMA-ES (`curriculum`)
CMA-ES mit **Reward Shaping + Curriculum Learning**.

- Difficulty ramp: 0.3 → 1.0 über Training
- Geformte Rewards: Survival-Bonus, Geschwindigkeits-Bonus, Aufrecht-Bonus
- Environment-spezifisches Shaping (LunarLander, BipedalWalker)
- **Stärke:** Sample-effizienteste Methode (8K Evals!)
- **Biologie:** Curriculum Learning spiegelt kindliche Entwicklung wider
- **Rust:** ✅ Portiert

### Neuromod CMA-ES (`neuromod`)
CMA-ES + **Neuromodulatory Plasticity**.

- Netzwerk hat zusätzliche Plastizitäts-Parameter
- Synapsen verändern sich während der Evaluation (Hebbian-artig)
- CMA-ES optimiert initiale Gewichte + Plastizitätsregeln
- **Stärke:** Biologisch plausibelste Methode
- **Key Finding:** +80 bei 2K Evals → +264.5 bei 200K Evals
- **Rust:** ✅ Portiert

### Island Model (`island_model`)
**4 CMA-ES Populationen** mit periodischer Migration.

- 4 Inseln mit verschiedenen Sigmas (0.3, 0.5, 0.8, 1.2)
- Migration alle 10 Gen: bestes Individuum → nächste Insel
- **Stärke:** Diversität durch parallele Suche, robust gegen lokale Optima
- **Finding:** Neuromod Island > Neuromod > Islands
- **Rust:** ✅ Portiert

### Island Advanced (`island_advanced`)
**6 heterogene Inseln** mit adaptiver Migration.

- Fully connected Topologie
- Adaptive Migrationsrate (mehr Migration bei niedriger Diversität)
- Migration-Tournament: nur akzeptieren wenn besser als 80% des Besten

### Neuromod Island (`neuromod_island`)
**Neuromodulation + Island Model** kombiniert.

- Plastische Netze auf mehreren Inseln
- Beweist: lokale Lernregeln + Populationsdynamik = stark
- **Best Result:** +256.3 auf LunarLander

### Hybrid CMA+FF (`hybrid_cma_ff`)
**CMA-ES + Forward-Forward** Local Learning.

- CMA-ES optimiert initiale Gewichte + FF-Hyperparameter
- "Goodness"-basiert: positive Erfahrungen → verstärken, negative → abschwächen

### Indirect Encoding (`indirect_encoding`)
**CPPN-basiert** — ein kleines Netz *generiert* die Policy-Gewichte.

- ~625 Genom-Parameter → ~3000 Policy-Parameter
- Inspiriert von Biologie: DNA encodiert Entwicklungsprogramm
- **Status:** Schwächste Methode (+9.1 auf LunarLander)

### Scaling Test (`scaling_test`)
**Network-Größen-Experiment** — testet CMA-ES bei verschiedenen Netzgrößen.

- Configs: 1K, 10K, 33K, 100K Parameter
- **LunarLander:** Alle lösen es → zu einfach für Breakpoint
- **BipedalWalker:** Tests laufen (erwarteter Breakpoint)
- **Rust:** ✅ Portiert

### Meta-Learning (`meta_learning`) 🆕
**CMA-ES evolves Lernregeln** statt nur Gewichte.

- Genom enthält Gewichte + Lernregel-Parameter (eta, decay, modulation gains)
- Netzwerk lernt *während* der Evaluation basierend auf evolvierten Regeln
- Verschmilzt Evolution mit lebenslangem Lernen
- **LunarLander:** +245.2 — zweitbeste Methode nach Curriculum!
- **Rust:** ✅ Portiert

### Pure Meta-Learning (`meta_learning_pure`) 🆕
**Evolve NUR Lernregeln** — der biologischste Ansatz.

- Nur 21 Parameter im Genom (Lernregel-Koeffizienten)
- Gewichte werden zufällig initialisiert → müssen durch Lernregeln konvergieren
- Spiegelt Biologie wider: Gene kodieren WIE gelernt wird, nicht WAS
- **Status:** Jobs laufen auf LunarLander + BipedalWalker
- **Bedeutung:** Wenn das funktioniert → Skalierung zu beliebig großen Netzen möglich (Genom bleibt klein)

## Backprop-Baseline (Kontrollgruppe)

### PPO Baseline (`ppo_baseline`) ⚠️
**Proximal Policy Optimization** — gradient-basiert.

- Standard-RL mit Backpropagation
- Gleiche Netzwerk-Architektur (fairer Vergleich)
- **⚠️ Nutzt Backpropagation** — klar gekennzeichnet
- **LunarLander:** +59.7 (schlechter als 7 gradientenfreie Methoden!)
- **BipedalWalker:** +145.9

## Methoden-Vergleich

### LunarLander-v3 (100K Evals)

```
Curriculum     ████████████████████████████████████████  790.1  🧬
Meta-Learning  ████████████                              245.2  🧬
Scaling 10K    ███████████                               227.2  🧬
Neuromod       ██████████                                217.6  🧬
CMA-ES         ██████████                                214.4  🧬
Neuromod Island██████████                                200.3  🧬
Island Model   ████████                                  175.9  🧬
OpenAI-ES      ███                                        73.4  🧬
PPO            ██                                         59.7  ⚡
```

🧬 = Gradientenfrei | ⚡ = Backpropagation
