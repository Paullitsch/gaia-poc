# GAIA Quick Start Guide

## Architecture

GAIA uses a **server-worker** model:
- **`gaia-server`** runs on your VPS — manages jobs, collects results
- **`gaia-worker`** runs on GPU machines — connects outbound to server, executes experiments

Workers connect TO the server (no open ports needed on worker machines).

---

## 1. Download Binaries

From **[GitHub Releases](https://github.com/Paullitsch/gaia-poc/releases)**:

| Binary | Where to run |
|--------|-------------|
| `gaia-server-linux-x86_64` | VPS |
| `gaia-worker-linux-x86_64` | Linux GPU machine |
| `gaia-worker-windows-x86_64.exe` | Windows GPU machine |

---

## 2. Start the Server (VPS)

```bash
chmod +x gaia-server-linux-x86_64
./gaia-server-linux-x86_64 --token mysecret
# Listening on 0.0.0.0:7434
```

Options:
- `--port 7434` — HTTP port (default: 7434)
- `--token mysecret` — Auth token (required)
- `--data-dir ./server-data` — Persistence directory

---

## 3. Start a Worker (GPU Machine)

### Linux

```bash
# Clone repo (for Python experiments)
git clone https://github.com/Paullitsch/gaia-poc.git
cd gaia-poc/worker

# Python setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Start worker
chmod +x gaia-worker-linux-x86_64
./gaia-worker-linux-x86_64 \
  --server https://your-vps:7434 \
  --token mysecret \
  --name paul-rtx5070
```

### Windows

```powershell
git clone https://github.com/Paullitsch/gaia-poc.git
cd gaia-poc\worker

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124

.\gaia-worker-windows-x86_64.exe `
  --server https://your-vps:7434 `
  --token mysecret `
  --name paul-rtx5070
```

Worker options:
- `--poll-interval 5` — Seconds between job polls (default: 5)
- `--experiments-dir ../experiments` — Path to experiment scripts
- `--python python3` — Python binary
- `--job-timeout 3600` — Max job runtime in seconds

---

## 4. Submit Jobs

```bash
# Submit a job
curl -X POST https://your-vps:7434/api/jobs/submit \
  -H "Authorization: Bearer mysecret" \
  -H "Content-Type: application/json" \
  -d '{"method": "cma_es", "max_evals": 100000}'

# Check status
curl -H "Authorization: Bearer mysecret" https://your-vps:7434/api/status

# Get results
curl -H "Authorization: Bearer mysecret" https://your-vps:7434/api/results/<job_id>

# Download CSV
curl -H "Authorization: Bearer mysecret" https://your-vps:7434/api/results/<job_id>/csv
```

---

## 5. Build from Source (Optional)

```bash
git clone https://github.com/Paullitsch/gaia-poc.git
cd gaia-poc

# Build server
cd server-rust && cargo build --release
# Binary: ./target/release/gaia-server

# Build worker
cd ../worker-rust && cargo build --release
# Binary: ./target/release/gaia-worker
```

---

## Project Structure

```
gaia-poc/
├── server-rust/           # Central server (VPS)
│   └── src/
│       ├── main.rs        # CLI + startup
│       ├── routes.rs      # HTTP endpoints
│       ├── state.rs       # In-memory state
│       ├── models.rs      # Shared types
│       └── storage.rs     # JSON persistence
│
├── worker-rust/           # Worker agent (GPU machines)
│   └── src/
│       ├── main.rs        # CLI + main loop
│       ├── client.rs      # HTTP client (→ server)
│       ├── worker.rs      # Job execution
│       ├── config.rs      # Config types
│       └── gpu.rs         # GPU detection
│
├── worker/                # Python experiments
│   ├── run_all.py
│   └── experiments/
│
└── .github/workflows/
    └── release.yml        # CI: builds server + worker
```

---

## Troubleshooting

**Worker can't connect to server:**
- Check firewall allows port 7434 on VPS
- Verify `--token` matches on both sides
- Try `curl https://your-vps:7434/api/status -H "Authorization: Bearer mysecret"`

**GPU not detected:**
```bash
nvidia-smi  # Should show your GPU
python -c "import torch; print(torch.cuda.is_available())"
```

**Python experiments fail:**
```bash
sudo apt install swig  # For gymnasium[box2d]
pip install torch --index-url https://download.pytorch.org/whl/cu124
```
