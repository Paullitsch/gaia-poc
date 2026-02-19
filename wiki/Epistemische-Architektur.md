# Epistemische Architektur

GAIA verpflichtet sich zu radikaler epistemischer Ehrlichkeit. Jede Aussage wird einer Sicherheitsebene zugeordnet.

## Die vier Ebenen

```
┌──────────────────────────────────────────┐
│  Ebene 4: Spekulative Visionen  (<25%)   │
│  "Dezentrales GAIA-Netzwerk"             │
├──────────────────────────────────────────┤
│  Ebene 3: Hypothesen  (25-75%)           │
│  "Skaliert auf Atari"                    │
├──────────────────────────────────────────┤
│  Ebene 2: Empirisch gesichert  (>90%)    │
│  "CMA-ES löst LunarLander"              │
├──────────────────────────────────────────┤
│  Ebene 1: Axiomatisch  (~100%)           │
│  "No-Free-Lunch, Informationstheorie"    │
└──────────────────────────────────────────┘
```

## Aktuelle Einordnung (v4)

### Ebene 2: Empirisch gesichert ✅

| Aussage | Evidenz |
|---------|---------|
| Evolution skaliert nicht für Gewichte >7K Parameter | Phase 1-2, reproduzierbar |
| Forward-Forward erreicht 50-70% von Backprop | Phase 3 |
| Meta-Plastizität schlägt naive Backprop | Phase 4, reproduziert in Phase 5 |
| Neuromodulation verbessert lokales Lernen | Phase 5 (+80.0) |
| **CMA-ES löst LunarLander (+235.3)** | **Phase 7, neu** |
| **Curriculum + CMA-ES erreicht +274** | **Phase 7, neu** |
| **OpenAI-ES löst LunarLander (+206.6)** | **Phase 7, neu** |
| **Verteilte Infrastruktur funktioniert** | **Phase 7, neu** |
| Compute ist der entscheidende Faktor | Phase 7 (2K→100K = -43→+274) |

### Ebene 3: Theoretische Hypothesen 🔬

| Aussage | Einschätzung |
|---------|-------------|
| GAIA skaliert auf BipedalWalker | Plausibel, CMA-ES sollte funktionieren |
| CMA-ES degradiert ab ~10K Parameter | Theoretisch begründet (O(n²)) |
| Multi-Worker beschleunigt proportional | Architektur steht, ungetestet |
| Neuromod kann CMA-ES schlagen bei gleichem Compute | Offen |

### Ebene 4: Spekulativ 🌟

| Aussage | Einschätzung |
|---------|-------------|
| Dezentrales GAIA-Netzwerk mit 1000+ Knoten | Konzeptuell, keine Evidenz |
| GAIA löst Atari-Spiele | Möglich, braucht massive GPU |
| Demokratisiertes KI-Training | Langfristvision |

## Prinzipien

1. **Negative Ergebnisse publizieren:** Phase 1-2 (Evolution versagt) sind genauso wichtig wie Phase 7 (Durchbruch)
2. **Hypothesen revidieren:** v1→v2→v3→v4 zeigt den wissenschaftlichen Prozess
3. **Limitierungen benennen:** Nur 1 Benchmark, keine statistische Signifikanz, GPU nicht ausgelastet
4. **Kein Hype:** +274 auf LunarLander ist kein AGI. Es ist ein Proof-of-Concept.
