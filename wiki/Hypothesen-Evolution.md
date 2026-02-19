# Hypothesen-Evolution

Die GAIA-Hypothese hat sich über vier Versionen entwickelt. Jede Version wurde durch experimentelle Ergebnisse motiviert.

## v1: Evolution ersetzt Backpropagation (Phase 1-2)

> *Evolutionäre Algorithmen können Backpropagation als Trainingsmethode ersetzen.*

**Status: Widerlegt.**

Phase 1 zeigte, dass Evolution CartPole löst — aber 20x weniger effizient als Backprop. Phase 2 zeigte die Skalierungswand: bei >7.000 Parametern (LunarLander) scheitert reine Evolution komplett.

**Lektion:** Evolution ist ein schlechter Gewichts-Optimierer in hochdimensionalen Räumen.

## v2: Lokale Lernregeln statt globale Sync (Phase 3-4)

> *Nicht Evolution statt Backprop, sondern lokale Lernregeln + evolutionäre Meta-Optimierung.*

**Status: Teilweise bestätigt.**

Forward-Forward erreichte 50-70% von Backprop. Meta-Plastizität (Evolution optimiert Lernregeln statt Gewichte) schlug sogar naive Backprop (-50.4 vs -158.4).

**Lektion:** Evolution ist ein exzellenter Meta-Lernalgorithmus.

## v3: Neuromodulierte Meta-Plastizität (Phase 5-6)

> *Biologisch plausibles Lernen braucht drei Mechanismen: Evolution (Architektur), lokale Regeln (Lernen), Neuromodulation (Steuerung).*

**Status: Bestätigt.**

Drei Neuromodulationssignale (Dopamin, TD-Error, Novität) brachten den Score von -39.8 auf +80.0 — ein dramatischer Sprung. Aber LunarLander blieb ungelöst (+80 < +200).

**Lektion:** Biologische Inspiration hilft, aber löst das Problem nicht allein.

## v4: Compute + richtige Methode (Phase 7)

> *Gradientenfreie Optimierung ist nicht grundsätzlich unterlegen — sie braucht mehr Compute und die richtige Methode.*

**Status: BEWIESEN.**

CMA-ES mit 100K Evaluierungen: +235.3. Curriculum: +274.0. Drei von fünf Methoden lösen LunarLander.

| Evaluierungen | CMA-ES Score |
|--------------|-------------|
| 2.000 | -43 |
| 10.000 | +156 |
| 50.000 | +220 |
| 100.000 | +235 |

**Lektion:** Es war nie der Algorithmus, der fehlte — es war der Compute.

## Visualisierung

```
v1: "Evolution ersetzt Backprop"
     │
     │ Widerlegt durch Phase 1-2
     ▼
v2: "Lokale Regeln + Evo Meta-Optimierung"
     │
     │ Bestätigt, aber nicht ausreichend
     ▼
v3: "Neuromodulation als Schlüssel"
     │
     │ +80.0 — beeindruckend, aber nicht gelöst
     ▼
v4: "Compute + richtige Methode = Lösung"
     │
     │ +274.0 — GELÖST ✅
     ▼
v5: "Wo liegen die Grenzen?" (Phase 8)
```
