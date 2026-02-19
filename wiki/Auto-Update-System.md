# Auto-Update System

> Worker aktualisieren sich selbst — kein manuelles Deployment nötig.

## Architektur

```
┌─────────────┐          ┌──────────────┐
│ GAIA Server │◄────────►│ GAIA Worker  │
│             │          │              │
│ /releases/  │  Poll    │ --auto-update│
│ latest      │◄─────────│ Heartbeat    │
│ v0.4.3/     │          │ → check ver  │
│  linux      │  Download│ → download   │
│  windows    │─────────►│ → SHA-256    │
│  experiments│          │ → replace    │
│             │          │ → restart    │
└─────────────┘          └──────────────┘
```

## Flow

1. Worker sendet Heartbeat an Server
2. Server antwortet mit `latest_version` (z.B. "v0.4.3")
3. Worker vergleicht mit eigener Version (semver)
4. Wenn neuer: Download Binary von `/releases/{tag}/{filename}`
5. SHA-256 Verifizierung gegen Release-Metadata
6. Binary ersetzen (temp-file → rename Strategie)
7. Experiment-Sync: `experiments.tar.gz` extrahieren
8. Restart via `exec()` (Unix) oder Spawn (Windows)

## Worker-Flags

```bash
# Auto-Update aktivieren
./gaia-worker --server URL --token TOKEN --name NAME --auto-update

# Experiment-Sync separat (automatisch bei --auto-update)
./gaia-worker --server URL --token TOKEN --name NAME --sync-experiments
```

## Server-Seite: Release hochladen

```bash
# Binary hochladen
curl -X POST "https://gaia.kndl.at/api/releases/upload?tag=v0.5.0&filename=gaia-worker-linux" \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @target/release/gaia-worker

# Experiments hochladen
tar czf experiments.tar.gz -C worker experiments/ run_all.py
curl -X POST "https://gaia.kndl.at/api/releases/upload?tag=v0.5.0&filename=experiments.tar.gz" \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @experiments.tar.gz
```

## Sicherheit

- Downloads sind SHA-256 verifiziert
- Auth-Token wird bei Downloads mitgesendet
- Keine unsigned Binaries akzeptiert
- Backup des alten Binary (`.bak`)

## Version History

| Version | Änderung |
|---------|----------|
| v0.4.0 | Self-Updating Binary + Release API |
| v0.4.1 | Experiment Sync |
| v0.4.2 | Fix Self-Update auf Linux/WSL |
| v0.4.3 | Fix Working Directory |
