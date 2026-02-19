# Deployment

## Docker (empfohlen)

```bash
git clone https://github.com/Paullitsch/gaia-poc.git
cd gaia-poc
cp .env.example .env
# Edit .env: GAIA_TOKEN=your-secret-token
docker compose up -d
```

Startet:
- **gaia-server** (Port 7434)
- **gaia-worker-cpu** (optional, für Tests)

## Binaries

Download von [GitHub Releases](https://github.com/Paullitsch/gaia-poc/releases):

| Binary | Plattform |
|--------|-----------|
| `gaia-server-linux-x86_64` | Linux (VPS) |
| `gaia-worker-linux-x86_64` | Linux / WSL |
| `gaia-worker-windows-x86_64.exe` | Windows |

### Server starten
```bash
./gaia-server --port 7434 --token mysecret --data-dir ./data
```

### Worker starten
```bash
# 1. Repo klonen (für Python-Experiments)
git clone https://github.com/Paullitsch/gaia-poc.git
cd gaia-poc

# 2. Binary downloaden
curl -L -o gaia-worker https://github.com/Paullitsch/gaia-poc/releases/latest/download/gaia-worker-linux-x86_64
chmod +x gaia-worker

# 3. Python-Umgebung
cd worker
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..

# 4. Worker starten
./gaia-worker --server https://your-server:7434 --token mysecret --name my-worker --experiments-dir ./worker
```

## Update Worker

```bash
cd gaia-poc
git pull                    # Experiment-Code aktualisieren
curl -L -o gaia-worker \    # Neues Binary
  https://github.com/Paullitsch/gaia-poc/releases/latest/download/gaia-worker-linux-x86_64
chmod +x gaia-worker
# Worker neu starten
```

## HTTPS (Caddy)

```
gaia.example.com {
    reverse_proxy localhost:7434
}
```

## Umgebungsvariablen

| Variable | Default | Beschreibung |
|----------|---------|-------------|
| `GAIA_TOKEN` | (pflicht) | Auth Token |
| `GAIA_PORT` | 7434 | Server Port |
| `GAIA_DATA_DIR` | ./server-data | Persistenz-Verzeichnis |
| `GAIA_SERVER` | (pflicht) | Server URL (Worker) |
| `GAIA_WORKER_NAME` | (pflicht) | Worker Name |
