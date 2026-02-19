# Server API

Base URL: `https://gaia.kndl.at` (oder `http://localhost:7434`)

Alle Endpoints erfordern `Authorization: Bearer <token>`.

## Worker Management

### Register Worker
```
POST /api/workers/register
Body: { "name": "my-worker", "gpu": { "name": "RTX 5070", "memory_mb": 12227 } }
Response: { "worker_id": "uuid" }
```

### Heartbeat
```
GET /api/workers/heartbeat/:worker_id
Response: { "status": "ok" }
```

### Enable/Disable Worker
```
POST /api/workers/:worker_id/enable
Response: { "enabled": true/false }
```
Togglet den Worker. Deaktivierte Worker heartbeaten weiter, bekommen aber keine Jobs.

## Job Management

### Submit Job
```
POST /api/jobs/submit
Body: { "method": "cma_es", "params": { "max_evals": 100000 } }
Response: { "job_id": "uuid", "status": "queued" }
```

Verfügbare Methoden: `cma_es`, `openai_es`, `hybrid_cma_ff`, `curriculum`, `indirect_encoding`

### Get Next Job (Worker)
```
GET /api/jobs/next/:worker_id
Response (200): { "id": "...", "method": "cma_es", "params": {...}, "max_evals": 100 }
Response (204): No Content (keine Jobs in Queue)
```

### Cancel Job
```
POST /api/jobs/cancel/:job_id
Response: { "status": "cancelled" }
```
Cancelt queued oder running Jobs. Worker prüft alle 10 Generationen und killt den Subprocess.

## Results

### Stream Result (Worker)
```
POST /api/results/stream
Body: {
  "job_id": "...",
  "worker_id": "...",
  "generation": 42,
  "data": { "best": "-50.3", "best_ever": "-12.1", "mean": "-200.5", "sigma": "0.45" }
}
```

### Complete Job (Worker)
```
POST /api/results/complete
Body: { "job_id": "...", "worker_id": "...", "status": "completed", "error": null }
```
Status: `completed`, `failed`, `timed_out`

### Get Results
```
GET /api/results/:job_id
Response: { "job": {...}, "results": [...], "count": 86 }
```

### Get Results CSV
```
GET /api/results/:job_id/csv
Response: generation,best,best_ever,mean,sigma,evals,time\n1,-130.5,...
```

## Status

### Get Status
```
GET /api/status
Response: {
  "workers": [...],
  "jobs": [...],
  "queue_length": 3,
  "total_results": 456,
  "server_start_time": "2026-02-19T20:00:00Z"
}
```
