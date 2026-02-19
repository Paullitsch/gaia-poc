# Architektur

## Übersicht

GAIA verwendet eine **Server-Worker-Architektur** für verteilte Experiment-Ausführung.

```
                    Internet
                       │
          ┌────────────┼────────────┐
          │            │            │
    ┌─────┴─────┐ ┌───┴───┐ ┌─────┴─────┐
    │  Worker 1  │ │Worker 2│ │  Worker N  │
    │  RTX 5070  │ │  CPU   │ │   A100    │
    │  (outbound)│ │(Docker)│ │  (Cloud)  │
    └─────┬─────┘ └───┬───┘ └─────┬─────┘
          │            │            │
          └────────────┼────────────┘
                       │ HTTPS :7434
              ┌────────┴────────┐
              │   GAIA Server   │
              │  Job Queue      │
              │  Result Store   │
              │  Web Dashboard  │
              └─────────────────┘
```

**Kern-Design:** Worker verbinden sich **ausgehend** zum Server. Keine offenen Ports auf Worker-Seite nötig — funktioniert hinter NAT, Firewalls, etc.

## Server

**Technologie:** Rust, Axum, Tokio

**Verantwortlichkeiten:**
- Job-Queue (FIFO) verwalten
- Worker-Registry mit Heartbeats
- Ergebnisse speichern und streamen
- Web Dashboard bereitstellen
- State auf Disk persistieren (JSON)

**Endpoints:** Siehe [[Server API]]

**Deployment:**
```bash
# Docker
docker compose up -d

# Binary
./gaia-server --port 7434 --token mysecret --data-dir ./data
```

## Worker

**Technologie:** Rust (Orchestrierung) + Python (Experimente)

**Ablauf:**
1. Registriert sich beim Server (Name, GPU-Info)
2. Heartbeat alle 5 Sekunden
3. Pollt nach Jobs
4. Spawnt Python-Subprocess: `python3 run_all.py --method <method> --max-evals <n>`
5. Parsed stdout (Generationsdaten), streamt zum Server
6. Prüft alle 10 Generationen auf Cancellation
7. Meldet Completion mit Status + Error-Details

**GPU-Erkennung:** Automatisch via `nvidia-smi`. Wird beim Server als GPU-Info registriert.

**Parallelisierung:** Python-Experiments nutzen `multiprocessing.Pool` für Population-Evaluation auf allen CPU-Kernen.

## Dashboard

**Technologie:** Vanilla HTML/CSS/JS (kein Framework), eingebettet im Server-Binary

**Features:**
- **Overview:** Worker-Karten, Job-Queue, Best Scores, Activity Log
- **Charts:** Learning Curves, Method Comparison, Sigma Convergence, Score Distribution
- **Debug:** Live-Stream aller Generationsdaten, Raw API
- **Export:** CSV/JSON/PNG pro Job oder gesamt

**Polling:** 3 Sekunden Intervall, auto-login via localStorage

## Datenfluss

```
Worker:  python3 run_all.py --method cma_es --max-evals 100000
           │
           ├─ stdout: "Gen  1 | Best: -130.5 | Ever: -130.5 | Mean: -399.8 | σ: 0.498"
           │
Worker Rust: parse → HashMap{generation, best, best_ever, mean, sigma, evals}
           │
           ├─ POST /api/results/stream {job_id, worker_id, generation, data}
           │
Server:    results.push(row) → Dashboard polls → Charts update
```

## Security

- **Bearer Token** auf allen API Endpoints
- **Token via ENV** (`GAIA_TOKEN`) oder CLI Flag
- **Kein Default-Token** — muss gesetzt werden
- **HTTPS** über Reverse Proxy (Caddy) empfohlen
