# GAIA Worker Agent

Ein leichtgewichtiger Compute-Worker, der auf deinem lokalen PC mit GPU läuft.
Der VPS steuert, der Worker rechnet.

## Setup (auf deinem PC)

### 1. Voraussetzungen
- Python 3.10+
- NVIDIA GPU mit CUDA (RTX 5070)
- PyTorch mit CUDA-Support

### 2. Installation

```bash
# Repo klonen (oder vom VPS runterladen)
git clone <repo-url> gaia-poc
cd gaia-poc/worker

# Python Environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder: venv\Scripts\activate  # Windows

# Dependencies
pip install -r requirements.txt
```

### 3. Konfiguration

Kopiere `config.example.yaml` nach `config.yaml` und passe an:
- `vps_host`: IP/Hostname des VPS
- `vps_port`: SSH Port (default 22)
- `worker_name`: Name dieses Workers (z.B. "paul-rtx5070")

### 4. Starten

```bash
python worker.py
```

Der Worker:
1. Verbindet sich zum VPS
2. Wartet auf Jobs
3. Führt Berechnungen auf der GPU aus
4. Streamt Ergebnisse zurück zum VPS

## Architektur

```
VPS (Orchestrator)          Dein PC (Worker)
│                           │
│  POST /submit_job ──────► │ Nimmt Job entgegen
│                           │ Führt auf GPU aus
│ ◄──── POST /result_stream │ Streamt Ergebnisse
│                           │ (jede Generation)
│  Speichert CSVs/Plots     │
```

Kommunikation läuft über einen simplen HTTP-Tunnel (SSH Reverse Tunnel oder WireGuard).
