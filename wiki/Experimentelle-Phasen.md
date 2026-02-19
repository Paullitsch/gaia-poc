# Experimentelle Phasen

## Übersicht

| Phase | Methode | Aufgabe | Best Score | Ergebnis |
|-------|---------|---------|-----------|----------|
| 1 | Reine Evolution | CartPole | 500/500 | ✅ Gelöst |
| 2 | Evolution + Hebbisch | LunarLander | +59.7 | ❌ Skaliert nicht |
| 3 | Forward-Forward | LunarLander | 50-70% v. Backprop | 🟡 Teilweise |
| 4 | Meta-Plastizität | LunarLander | -50.4 | 🟡 Schlägt naive BP |
| 5 | Neuromodulation | LunarLander | +80.0 | 🟡 Durchbruch |
| 6 | Deep Neuromod + PPO | LunarLander | +57.8 / +264.8 | 🟡 PPO gewinnt |
| 7 | CMA-ES + Compute | LunarLander | **+274.0** | ✅ **GELÖST** |

## Phase 1: CartPole (722 Parameter)

**Frage:** Kann Evolution neuronale Netze trainieren?
**Antwort:** Ja, aber 20x weniger effizient als Backprop.

Alle evolutionären Varianten (rein, Hebbisch, Reward-Hebbisch) lösten CartPole (500/500). REINFORCE brauchte nur 217 Episoden vs. 4.500 bei Evolution.

## Phase 2: LunarLander (6.948 Parameter)

**Frage:** Skaliert Evolution auf schwerere Probleme?
**Antwort:** Nein. Skalierungswand bei ~7K Parametern.

Bester Score: +59.7 (Reward-Hebbisch). Weit unter dem Lösungsschwellenwert von +200. Die Fitness-Landschaft wird zu komplex für gradientenfreie Suche im Gewichtsraum.

## Phase 3: Forward-Forward (10.000 Parameter)

**Frage:** Können lokale Lernregeln Backprop ersetzen?
**Antwort:** Sie kommen auf 50-70%.

Hintons Forward-Forward-Algorithmus, erweitert durch evolutionäre Hyperparameter-Optimierung. Überraschend nahe an Backprop, aber die Lücke bleibt signifikant.

## Phase 4: Meta-Plastizität (11.600 Parameter)

**Frage:** Was wenn Evolution Lernregeln statt Gewichte optimiert?
**Antwort:** Schlägt naive Backprop!

Meta-Plastizität (-50.4) übertraf REINFORCE (-158.4). Evolution als Meta-Lernalgorithmus ist der richtige Ansatz.

## Phase 5: Neuromodulation (20.000 Parameter)

**Frage:** Helfen biologisch inspirierte Modulationssignale?
**Antwort:** Dramatischer Durchbruch (+80.0).

Drei Signale (Dopamin, TD-Error, Novität) modulieren schichtenspezifisch die Plastizität. 3x compute-effizienter als Meta-Plastizität. Erster positiver Score in GAIA-Geschichte.

## Phase 6: Deep Neuromodulation (23K+ Parameter)

**Frage:** Können wir die Neuromodulation vertiefen?
**Antwort:** Ja, aber PPO bleibt überlegen.

5 Neuromodulationssignale + Eligibility Traces: +57.8. PPO Baseline: +264.8. Die Credit-Assignment-Lücke zwischen lokalem FF-Lernen und globalem Backprop bleibt das fundamentale Hindernis.

## Phase 7: CMA-ES + Compute (2.788 Parameter) ⭐

**Frage:** Was passiert mit genug Compute?
**Antwort:** GELÖST. +274.0 ohne Backpropagation.

Kleineres Netzwerk (2.788 statt 20K Parameter), aber massiv mehr Compute (100K Evaluierungen statt 10K). CMA-ES lernt die Kovarianzstruktur und findet optimale Gewichte.

**Schlüsseleinblick:** Das Netzwerk war zu groß, nicht der Algorithmus zu schwach. CMA-ES skaliert O(n²) mit der Parameterzahl — ein kleineres Netz mit mehr Compute war der Weg.

### Lernkurve CMA-ES (Phase 7)

```
Score
+274 │                                          ●
+200 │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ SOLVED ─ ─ ─ ─/─ ─
+150 │                                    ●  /
+100 │                               ●  /
 +50 │                          ●  /
   0 │                     ●  /
 -50 │                ●  /
-100 │           ●  /
-150 │      ●  /
-200 │  ● /
     └──┬──┬──┬──┬──┬──┬──┬──┬──┬──
        10 20 30 40 50 60 70 80 90
                  Generation
```
