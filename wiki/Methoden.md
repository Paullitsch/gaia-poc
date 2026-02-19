# Methoden

## CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

**Der Gold-Standard gradientenfreier Optimierung.**

CMA-ES lernt die Kovarianzstruktur des Parameterraums. Statt alle Parameter unabhängig zu perturbieren (wie bei ES), entdeckt CMA-ES Korrelationen — welche Parameter zusammen geändert werden sollten.

### Algorithmus

```
1. Initialisiere Mittelwert μ, Kovarianzmatrix C = I, Schrittweite σ
2. Wiederhole:
   a. Ziehe λ Kandidaten: x_i ~ N(μ, σ²C)
   b. Evaluiere Fitness f(x_i) für alle i
   c. Sortiere nach Fitness
   d. Update μ = gewichteter Mittelwert der besten μ_eff
   e. Update C basierend auf erfolgreichen Schritten
   f. Update σ basierend auf Pfadlänge
```

### Warum CMA-ES funktioniert

- **Kovarianzlernen:** Entdeckt, dass z.B. Gewichte von Neuron A und B zusammen die Landing-Strategie steuern
- **Schrittweiten-Adaptation:** σ schrumpft automatisch bei Konvergenz
- **Rangselektion:** Robust gegen Outlier-Fitness-Werte

### Limitierungen

- **O(n²) Speicher** für die Kovarianzmatrix → problematisch bei >10K Parametern
- **Diagonal-Approximation** möglich, verliert aber Korrelationsinformation
- Population muss ~4+3ln(n) groß sein

### GAIA-Ergebnisse

| Parameter | Population | Evals bis gelöst | Best Score |
|-----------|-----------|------------------|-----------|
| 2.788 | 27 | ~12K | +235.3 |

---

## OpenAI Evolution Strategies

**Gradientenschätzung durch finite Differenzen.**

Nicht wirklich "Evolution" — eher ein Monte-Carlo-Gradient-Estimator. Perturbiert Parameter mit Gaussian Noise, evaluiert, nutzt belohnungsgewichteten Noise als Update.

### Algorithmus

```
1. Initialisiere θ (Parameter)
2. Wiederhole:
   a. Ziehe N Noise-Vektoren ε_i ~ N(0, I)
   b. Evaluiere f(θ + σε_i) und f(θ - σε_i)  (antithetisch)
   c. Update: θ += α/(2Nσ) * Σ(f+ - f-) * ε_i
```

### Vorteile

- **Trivial parallelisierbar:** Jede Perturbation ist unabhängig
- **Skaliert besser als CMA-ES** bei hohen Parameterzahlen (O(n) statt O(n²))
- **Kommunikationseffizient:** Nur Seeds + Rewards müssen geteilt werden

### GAIA-Ergebnisse

| Parameter | Population | Evals bis gelöst | Best Score |
|-----------|-----------|------------------|-----------|
| 2.788 | 50 (×2 mirror) | ~55K | +206.6 |

---

## Forward-Forward (Hinton 2022)

**Vollständig lokales Lernen — keine Rückwärtspässe.**

Jede Schicht optimiert eine lokale "Goodness"-Metrik. Positive Daten sollen hohe Goodness erzeugen, negative Daten niedrige.

### Goodness-Funktion

```
g(x) = ||ReLU(Wx + b)||²
```

### Lernziel pro Schicht

```
L = log(1 + exp(-(g(x+) - θ))) + log(1 + exp(g(x-) - θ))
```

### GAIA-Ergebnisse

- Phase 3: 50-70% von Backprop
- Phase 4: Meta-Plastizität (evolvierte Lernraten) schlägt naive Backprop
- Phase 5: Neuromodulation + FF → +80.0

---

## Neuromodulation

**Biologisch inspirierte Plastizitätssteuerung.**

Drei Signale modulieren schichtenspezifisch die Lernrate:

| Signal | Biologisches Analog | Funktion |
|--------|-------------------|----------|
| Dopamin-Analog | Dopamin (VTA/SNc) | Belohnungsmodulation |
| TD-Error | Dopamin Burst/Dip | Vorhersagefehler |
| Novitätssignal | Noradrenalin (LC) | Exploration bei Neuheit |

### Modulierte Lernrate

```
η_eff = η * (1 + α_d * dopamin + α_δ * td_error + α_n * novelty)
```

Die Modulationsgewichte (α) werden evolutionär optimiert.

### Emergente Muster

Evolution entdeckte nicht-triviale Strategien:
- **Sensorische Schichten:** Hohe Novitätsmodulation
- **Entscheidungsschichten:** Hohe Belohnungsmodulation

---

## Curriculum Learning

**Progressive Schwierigkeitserhöhung + Reward Shaping.**

Statt dem sparse LunarLander-Reward bekommt der Agent dichteres Feedback:
- Belohnung für Annäherung an Landeplatz
- Belohnung für Geschwindigkeitskontrolle
- Schwierigkeit steigt über Generationen

### GAIA-Ergebnisse

**Champion der Phase 7:** +274.0 — höchster Score aller GAIA-Experimente.
