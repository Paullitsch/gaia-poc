# Phase 8: BipedalWalker + Auto-Update Infrastruktur

## Status: Gestartet 🚀

Phase 8 begann am 19. Februar 2026. Zwei Hauptstränge:

1. **BipedalWalker-v3** — der nächste Schwierigkeitsgrad nach LunarLander
2. **Self-Updating Infrastructure** — Worker aktualisieren sich selbst

## BipedalWalker-v3: Die Herausforderung

| Aspekt | LunarLander (Phase 7) | BipedalWalker (Phase 8) |
|--------|----------------------|------------------------|
| Action Space | Diskret (4) | **Kontinuierlich (4D)** |
| Observation | 8D | **24D** (Lidar, Gelenke, Kontakt) |
| Solved Threshold | 200 | **300** |
| Netzwerk | 2.788 Params | **11.588 Params** (4x) |
| Architektur | 8→64→32→4 | **24→128→64→4** |
| Output | argmax (diskret) | **tanh (continuous [-1,1])** |
| Max Steps | 1.000 | **1.600** |

BipedalWalker erfordert koordinierte Steuerung von 4 Gelenkmotoren (Hüfte + Knie × 2 Beine) für aufrechtes Gehen über Terrain.

## Experimente

### 8.1: BipedalWalker CMA-ES + Curriculum
- CMA-ES mit shaped Rewards (Vorwärtsbewegung, Aufrechthaltung)
- Difficulty ramp von 0.3 → 1.0
- Budget: 500K Evaluierungen

### 8.2: BipedalWalker OpenAI-ES
- Antithetisches Sampling, Population 64
- Budget: 500K Evaluierungen

### 8.3: BipedalWalker CMA-ES (Kontrollgruppe)
- Reines CMA-ES ohne Reward Shaping
- Budget: 500K Evaluierungen

## Auto-Update System

### v0.4.0: Self-Updating Binary
- Server hostet Release-Binaries über `/releases/` Endpoints
- Worker prüft bei jedem Heartbeat auf neue Versionen
- SHA-256 Verifizierung, Self-Replace + Restart

### v0.4.1: Experiment Sync
- `experiments.tar.gz` im Release gebundelt
- Automatische Synchronisation beim Start und nach Updates
- Kein manuelles `git pull` mehr nötig

### v0.4.2: Fix Self-Update
- Temp-File + Rename statt direktem Überschreiben

### v0.4.3: Fix Working Directory
- Worker nutzt parent von experiments_dir als Working Directory

## Release-API

| Endpoint | Auth | Beschreibung |
|----------|------|-------------|
| `GET /releases/latest` | Nein | Neueste Version (JSON) |
| `GET /releases/{tag}` | Nein | Version nach Tag |
| `GET /releases/{tag}/{file}` | Nein | Binary download |
| `POST /api/releases/upload` | Ja | Binary hochladen |
| `GET /api/releases` | Ja | Alle Releases listen |

## Nächste Schritte

- [ ] BipedalWalker Ergebnisse analysieren
- [ ] Netzwerk-Skalierung auf LunarLander (Experiment 8.4)
- [ ] Multi-Worker-Skalierung testen
- [ ] Neuromodulation mit höherem Budget revisiten
