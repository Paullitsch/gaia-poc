# Methoden

> Alle 11 Methoden sind environment-agnostisch — laufen auf jedem Gymnasium-Environment.

## Gradientenfreie Methoden (GAIA)

### CMA-ES (`cma_es`)
**Covariance Matrix Adaptation Evolution Strategy** — Gold-Standard der gradientenfreien Optimierung.

- Population von Parametervektoren → evaluate → Selektion → Kovarianzmatrix anpassen
- O(n²) Speicher für volle Kovarianzmatrix, Diagonal-Modus ab >2000 params
- **Stärke:** Sehr sample-effizient bei kleinen-mittleren Netzen
- **Schwäche:** Skaliert nicht über ~50K params (Speicher)

### OpenAI-ES (`openai_es`)
**OpenAI Evolution Strategies** mit antithetischem Sampling.

- Perturbiere Parameter mit Gauß-Rauschen, nutze Reward als "Gradient-Schätzer"
- Antithetisch: teste +ε und -ε → varianzreduziert
- O(n) Speicher — skaliert beliebig
- **Stärke:** Massiv parallelisierbar, skaliert bei großen Netzen
- **Schwäche:** Weniger sample-effizient als CMA-ES

### Curriculum CMA-ES (`curriculum`)
CMA-ES mit **Reward Shaping + Curriculum Learning**.

- Difficulty ramp: 0.3 → 1.0 über Training
- Geformte Rewards: Survival-Bonus, Geschwindigkeits-Bonus, Aufrecht-Bonus
- Environment-spezifisches Shaping (LunarLander, BipedalWalker)
- **Stärke:** Sample-effizienteste Methode (8K Evals!)
- **Biologie:** Curriculum Learning spiegelt kindliche Entwicklung wider

### Neuromod CMA-ES (`neuromod`)
CMA-ES + **Neuromodulatory Plasticity**.

- Netzwerk hat zusätzliche Plastizitäts-Parameter
- Synapsen verändern sich während der Evaluation (Hebbian-artig)
- CMA-ES optimiert initiale Gewichte + Plastizitätsregeln
- **Stärke:** Biologisch plausibelste Methode, skaliert mit Compute
- **Key Finding:** +80 bei 2K Evals → +264.5 bei 200K Evals

### Island Model (`island_model`)
**4 CMA-ES Populationen** mit periodischer Migration.

- 4 Inseln mit verschiedenen Sigmas (0.3, 0.5, 0.8, 1.2)
- Migration alle 10 Gen: bestes Individuum → nächste Insel
- **Stärke:** Diversität durch parallele Suche, robust gegen lokale Optima
- **Finding:** Neuromod Island > Neuromod > Islands (Kombination gewinnt)

### Island Advanced (`island_advanced`)
**6 heterogene Inseln** mit adaptiver Migration.

- Fully connected Topologie (jede Insel kann zu jeder senden)
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
- Forward-Forward lernt *innerhalb* jeder Evaluation
- "Goodness"-basiert: positive Erfahrungen → verstärken, negative → abschwächen

### Indirect Encoding (`indirect_encoding`)
**CPPN-basiert** — ein kleines Netz *generiert* die Policy-Gewichte.

- ~625 Genom-Parameter → ~3000 Policy-Parameter
- Massive Suchraum-Kompression
- Inspiriert von Biologie: DNA encodiert Entwicklungsprogramm, nicht einzelne Synapsen
- **Status:** Einzige Methode die LunarLander nicht löst (+9.1)

### Scaling (`scaling`)
**Network-Größen-Experiment** — testet CMA-ES bei verschiedenen Netzgrößen.

- Configs: tiny (32-16), small (64-32), medium (128-64), large (256-128), xl (512-256)
- Prüft: Ab wann degradiert CMA-ES?

## Backprop-Baseline (Kontrollgruppe)

### PPO Baseline (`ppo_baseline`) ⚠️
**Proximal Policy Optimization** — gradient-basiert.

- Standard-RL Algorithmus mit Backpropagation
- Gleiche Netzwerk-Architektur wie GAIA-Methoden (fairer Vergleich)
- **Zweck:** Kontrollgruppe — zeigt was Backprop auf gleichem Env/Netz erreicht
- **⚠️ Nutzt Backpropagation** — klar gekennzeichnet in allen Ergebnissen
