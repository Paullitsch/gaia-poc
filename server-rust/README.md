# GAIA Server

Central orchestration server for GAIA distributed experiments.

## Usage

```bash
gaia-server --token mysecret [--port 7434] [--data-dir ./server-data]
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/workers/register` | Worker registers itself |
| GET | `/api/workers/heartbeat/:id` | Worker heartbeat |
| GET | `/api/jobs/next/:worker_id` | Worker fetches next job |
| POST | `/api/jobs/submit` | Submit a new job |
| POST | `/api/results/stream` | Stream result data |
| POST | `/api/results/complete` | Report job completion |
| GET | `/api/results/:job_id` | Get results for a job |
| GET | `/api/results/:job_id/csv` | Download CSV results |
| GET | `/api/status` | Dashboard overview |

All endpoints require `Authorization: Bearer <token>` header.

## Submit a Job

```bash
curl -X POST https://your-vps:7434/api/jobs/submit \
  -H "Authorization: Bearer mysecret" \
  -H "Content-Type: application/json" \
  -d '{"method": "cma_es", "max_evals": 100000}'
```

## Data Persistence

Jobs, workers, and results are persisted to JSON files in `--data-dir` for crash recovery.
