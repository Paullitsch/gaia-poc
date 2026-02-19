# Scaling-Hypothesen

## Die zentrale Frage

Phase 7 bewies: gradientenfreie Methoden lösen LunarLander. Aber **wo liegen die Grenzen?**

## Dimension 1: Parameterzahl

CMA-ES hat O(n²) Speicher- und Compute-Komplexität für die Kovarianzmatrix.

| Parameter | CMA-ES Kovarianzmatrix | Schätzung |
|-----------|----------------------|-----------|
| 2.788 | 7.8 M Einträge | ✅ Funktioniert |
| 10.000 | 100 M Einträge | 🟡 Grenzbereich |
| 50.000 | 2.5 G Einträge | ❌ Zu groß |
| 100.000 | 10 G Einträge | ❌ Unmöglich |

**Lösung für große Netzwerke:** Diagonal CMA-ES (sep-CMA-ES) oder OpenAI-ES (O(n)).

**Hypothese:** CMA-ES dominiert bis ~10K Parameter, danach OpenAI-ES.

## Dimension 2: Umgebungs-Komplexität

| Umgebung | Obs | Act | Schwierigkeit | Geschätzte Evals |
|----------|-----|-----|--------------|------------------|
| CartPole | 4 | 2 | Trivial | ~5K |
| LunarLander | 8 | 4 | Mittel | ~100K |
| BipedalWalker | 24 | 4 (cont.) | Schwer | ~500K |
| Atari (Pong) | 210×160×3 | 6 | Sehr schwer | ~5M |
| MuJoCo (Humanoid) | 376 | 17 | Extrem | ~50M |

**Hypothese:** Gradientenfreie Methoden skalieren bis BipedalWalker. Atari erfordert CNN → große Netzwerke → nur mit OpenAI-ES + massivem Compute.

## Dimension 3: Compute-Skalierung

Phase 7 zeigte sublineares Scaling:

```
Score
+280 ┤                          ────────────
+240 ┤                     ●──/
+200 ┤─ ─ ─ ─ SOLVED ─ ─/─ ─ ─ ─ ─ ─ ─
+160 ┤              ● /
+120 ┤           ●/
 +80 ┤        ●/
  +0 ┤     ●/
 -50 ┤  ●/
-100 ┤●
     └─┬──┬──┬──┬──┬──┬──┬──┬──┬─
      2K 10K 20K 40K 60K 80K 100K
              Evaluierungen
```

**Beobachtung:** Abnehmende Returns nach ~50K Evals. Mehr Compute hilft, aber der Marginalnutzen sinkt.

## Dimension 4: Multi-Worker-Parallelisierung

### Theoretisch
Population-Evaluation ist trivial parallel. N Workers → N× Speedup.

### Praktisch
- Kommunikations-Overhead (Ergebnisse streamen)
- Server wird Bottleneck bei >100 Workers
- CMA-ES `tell()` ist sequentiell (sammelt alle Fitness-Werte)

**Hypothese:** >0.7x linearer Speedup bis ~8 Workers, danach Overhead.

### Island-Modell (Future)
Für >8 Workers: unabhängige CMA-ES-Instanzen pro Worker, periodische Migration der besten Individuen. Voll dezentral, kein zentraler Bottleneck.

## Dimension 5: Methoden-Vergleich bei Skala

| Methode | Kleine Netze (<5K) | Mittlere (5-50K) | Große (>50K) |
|---------|-------------------|-------------------|-------------|
| CMA-ES | 🏆 Dominant | 🟡 Degradiert | ❌ Zu teuer |
| OpenAI-ES | 🟡 Okay | 🏆 Dominant | 🟡 Machbar |
| Neuromod | 🟡 Vielversprechend | ❓ Ungetestet | ❓ Ungetestet |
| GA | ❌ Schlecht | ❌ Schlecht | ❌ Schlecht |

## Offene Fragen

1. **Gibt es einen Crossover-Punkt** wo gradientenfreie Methoden effizienter als Backprop werden?
2. **Können lokale Lernregeln + Evolution** die Credit-Assignment-Lücke schließen?
3. **Skaliert das Island-Modell** auf 100+ heterogene Knoten?
4. **Welche Rolle spielt GPU** bei Environment-Simulation (Brax) vs. Network-Inference?
