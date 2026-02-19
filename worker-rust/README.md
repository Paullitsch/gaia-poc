# GAIA Worker (Rust)

Single-binary worker agent for distributed experiment execution. Replaces the Python worker with a fast, lightweight Rust binary.

## Quick Start

```bash
# Install Rust (if needed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build
cargo build --release
# or: ./build.sh

# Configure
cp config.example.yaml config.yaml
# Edit config.yaml with your settings

# Run
./target/release/gaia-worker
# or with custom config:
./target/release/gaia-worker -c /path/to/config.yaml
```

## What This Does

- **HTTP API** on configurable port (default 8090) for job submission and status
- **Spawns Python scripts** as subprocesses for actual experiment execution
- **Streams results** back to VPS in real-time as CSV rows are produced
- **Auto-detects GPU** via `nvidia-smi`
- **Auth token** support on all endpoints

The binary is ~5MB with no runtime dependencies. Python is only needed for the experiment scripts themselves.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/status` | GET | Worker status, GPU info, uptime |
| `/api/submit_job` | POST | Submit a job for execution |
| `/api/results/:job_id` | GET | Get results for a completed job |
| `/api/ping` | POST | Health check |

### Submit Job

```bash
curl -X POST http://localhost:8090/api/submit_job \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{"id": "job-1", "method": "bayesian", "max_evals": 50}'
```

## Environment

- `RUST_LOG=debug` for verbose logging
- `RUST_LOG=gaia_worker=debug` for worker-only debug logs
