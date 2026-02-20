# Meta-Learning

> Evolution von Lernregeln statt Gewichten — der biologische Weg zur Skalierung

## Die Idee

Klassische Neuroevolution: ES optimiert die **Gewichte** eines Netzes direkt.
Meta-Learning: ES optimiert **Lernregeln**, die das Netz während der Evaluation trainieren.

Das ist der biologische Weg: Gene kodieren nicht die Synapsengewichte, sondern die Regeln nach denen Synapsen lernen.

## Warum das wichtig ist

CMA-ES hat O(n²) Komplexität — skaliert nicht über ~50K Parameter. Aber wenn wir nur Lernregeln evolvieren:
- **Genom bleibt klein** (21 Parameter statt 10K+)
- **Netz kann beliebig groß sein** (Regeln sind netzgrößen-unabhängig)
- **Skalierung durch Architektur** statt durch Evolution

## Zwei Varianten

### Meta-Learning (`meta_learning`)
- Genom = Gewichte + Lernregel-Parameter
- CMA-ES optimiert beides gleichzeitig
- Lernregeln modifizieren Gewichte während Evaluation
- **LunarLander: +245.2** — zweitbeste Methode!

### Pure Meta-Learning (`meta_learning_pure`)
- Genom = **nur Lernregel-Parameter** (21 Stück)
- Gewichte werden zufällig initialisiert
- Netz muss durch Lernregeln allein konvergieren
- **Status:** Jobs laufen

### Lernregel-Parameter

Die 21 Parameter beschreiben eine Hebbian-artige Lernregel:

```
η (learning rate)          — Wie schnell lernen?
decay                      — Gewichtsverfall
pre_gain                   — Einfluss des Pre-Neurons
post_gain                  — Einfluss des Post-Neurons  
reward_gain                — Einfluss des Rewards
hebbian_gain               — Pre × Post Korrelation
modulation_gains[...]      — Schichtspezifische Modulation
```

## Einordnung im Singularity Roadmap

| Stufe | Beschreibung | Meta-Learning Rolle |
|-------|-------------|---------------------|
| 1 | Skalierungsgesetz | Standard CMA-ES bricht bei ~50K params |
| **2** | **Hierarchische Optimierung** | **ES evolves Lernregeln → Regeln trainieren Netz** |
| 3 | Dezentrale Emergenz | Jeder Agent hat eigene Lernregeln via Gossip |
| 4 | Offene Frage | Reichen lokale Regeln + Evolution? |

Meta-Learning ist **Stufe 2** des Roadmaps — die Brücke zwischen direkter Evolution und skalierbarer Intelligenz.

## Offene Fragen

1. **Funktioniert Pure Meta-Learning auf LunarLander?** Wenn ja → 21 Parameter reichen für beliebig große Netze
2. **Transferieren Lernregeln zwischen Environments?** Auf LunarLander evolve → auf BipedalWalker anwenden?
3. **Wie interagieren Lernregeln mit Island Model?** Verschiedene Inseln = verschiedene Lernstrategien?
4. **Gibt es universelle Lernregeln?** Oder sind sie immer task-spezifisch?

## Biologische Parallelen

- **Gene → Lernregeln:** DNA kodiert Synaptogenese-Regeln, nicht Synapsengewichte
- **Lifetime Learning:** Netz lernt in jeder Episode — wie ein Tier in seiner Lebenszeit
- **Evolution optimiert Lernfähigkeit:** Nicht was das Tier kann, sondern wie schnell es lernt
- **Kein Credit Assignment Problem:** Evolution braucht nur Fitness, nicht Gradienten
