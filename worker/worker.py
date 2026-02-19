#!/usr/bin/env python3
"""
GAIA Worker Agent — Runs GPU compute jobs, streams results to VPS.

Usage:
    python worker.py                    # Start with config.yaml
    python worker.py --config my.yaml   # Custom config
    python worker.py --standalone       # Run without VPS connection (local mode)
"""

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import sys
import time
import threading
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import numpy as np
import yaml

# --- GPU Detection ---

def detect_device():
    """Detect best available compute device."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"🔥 GPU detected: {name} ({mem:.1f} GB)")
            return "cuda", {"name": name, "memory_gb": round(mem, 1)}
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("🍎 Apple MPS detected")
            return "mps", {"name": "Apple MPS"}
    except ImportError:
        pass
    print("💻 No GPU found, using CPU")
    return "cpu", {"name": "CPU"}


# --- Result Streaming ---

class ResultStreamer:
    """Streams results back to VPS orchestrator."""

    def __init__(self, vps_url=None, results_dir="./results"):
        self.vps_url = vps_url
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.buffer = []

    def send(self, job_id, data):
        """Send a result update. Saves locally + sends to VPS if connected."""
        entry = {
            "job_id": job_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **data
        }
        # Always save locally
        job_dir = self.results_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Append to JSONL
        with open(job_dir / "stream.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Save CSV if generation data
        if "generation" in data:
            csv_path = job_dir / f"{data.get('method', 'unknown')}.csv"
            write_header = not csv_path.exists()
            with open(csv_path, "a") as f:
                if write_header:
                    f.write(",".join(str(k) for k in data.keys()) + "\n")
                f.write(",".join(str(v) for v in data.values()) + "\n")

        # Send to VPS
        if self.vps_url:
            try:
                import requests
                requests.post(
                    f"{self.vps_url}/api/result",
                    json=entry,
                    timeout=5
                )
            except Exception as e:
                self.buffer.append(entry)
                if len(self.buffer) % 50 == 1:
                    print(f"⚠️  VPS unreachable, buffering ({len(self.buffer)} entries)")

    def flush_buffer(self):
        """Try to send buffered results."""
        if not self.vps_url or not self.buffer:
            return
        try:
            import requests
            requests.post(
                f"{self.vps_url}/api/results_bulk",
                json=self.buffer,
                timeout=30
            )
            print(f"✅ Flushed {len(self.buffer)} buffered results to VPS")
            self.buffer = []
        except Exception:
            pass


# --- Job Execution ---

def run_job(job, device, streamer):
    """Execute a compute job."""
    job_id = job["id"]
    method = job.get("method", "unknown")
    params = job.get("params", {})

    print(f"\n{'='*60}")
    print(f"🚀 Starting job: {job_id}")
    print(f"   Method: {method}")
    print(f"   Device: {device}")
    print(f"   Params: {json.dumps(params, indent=2)}")
    print(f"{'='*60}\n")

    streamer.send(job_id, {
        "type": "job_start",
        "method": method,
        "device": device,
        "params": params
    })

    start_time = time.time()

    try:
        # Dynamic import of experiment module (restricted to experiments/ directory)
        script_path = job.get("script")
        if script_path and os.path.exists(script_path):
            # Security: only allow loading scripts from the experiments/ directory
            abs_script = os.path.abspath(script_path)
            allowed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "experiments"))
            if not abs_script.startswith(allowed_dir):
                raise ValueError(f"Security: script must be in experiments/ directory, got: {script_path}")
            spec = importlib.util.spec_from_file_location("experiment", abs_script)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            # Call the run function with our streamer callback
            result = mod.run(
                method=method,
                params=params,
                device=device,
                callback=lambda data: streamer.send(job_id, {**data, "method": method})
            )
        else:
            # Built-in experiments
            from experiments import cma_es, openai_es, hybrid_ff, curriculum, indirect
            experiments = {
                "cma_es": cma_es,
                "openai_es": openai_es,
                "hybrid_cma_ff": hybrid_ff,
                "curriculum": curriculum,
                "indirect_encoding": indirect,
            }
            if method not in experiments:
                raise ValueError(f"Unknown method: {method}. Available: {list(experiments.keys())}")

            result = experiments[method].run(
                params=params,
                device=device,
                callback=lambda data: streamer.send(job_id, {**data, "method": method})
            )

        elapsed = time.time() - start_time

        streamer.send(job_id, {
            "type": "job_complete",
            "method": method,
            "result": result,
            "elapsed_seconds": round(elapsed, 1)
        })

        print(f"\n✅ Job {job_id} complete in {elapsed:.1f}s")
        print(f"   Result: {json.dumps(result, indent=2)}")
        return result

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        streamer.send(job_id, {
            "type": "job_error",
            "method": method,
            "error": error_msg,
            "elapsed_seconds": round(elapsed, 1)
        })
        print(f"\n❌ Job {job_id} failed after {elapsed:.1f}s: {error_msg}")
        import traceback
        traceback.print_exc()
        return {"error": error_msg}


# --- HTTP Server (receives jobs from VPS) ---

class WorkerHandler(BaseHTTPRequestHandler):
    """HTTP handler for receiving jobs from VPS."""

    server_ref = None

    def do_POST(self):
        if self.path == "/api/submit_job":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}

            # Auth check
            token = self.headers.get("X-Auth-Token", "")
            if self.server_ref and self.server_ref.auth_token and token != self.server_ref.auth_token:
                self.send_response(401)
                self.end_headers()
                self.wfile.write(b'{"error": "unauthorized"}')
                return

            self.send_response(200)
            self.send_headers_json()
            self.wfile.write(json.dumps({"status": "accepted", "job_id": body.get("id")}).encode())

            # Run job in thread
            threading.Thread(
                target=run_job,
                args=(body, self.server_ref.device, self.server_ref.streamer),
                daemon=True
            ).start()

        elif self.path == "/api/ping":
            self.send_response(200)
            self.send_headers_json()
            self.wfile.write(json.dumps({
                "status": "alive",
                "device": self.server_ref.device,
                "device_info": self.server_ref.device_info
            }).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/api/status":
            self.send_response(200)
            self.send_headers_json()
            self.wfile.write(json.dumps({
                "status": "ready",
                "worker": self.server_ref.worker_name,
                "device": self.server_ref.device,
                "device_info": self.server_ref.device_info,
                "uptime": round(time.time() - self.server_ref.start_time, 1)
            }).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def send_headers_json(self):
        self.send_header("Content-Type", "application/json")
        self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress default logging


class WorkerServer(HTTPServer):
    def __init__(self, addr, handler, config):
        self.worker_name = config.get("worker", {}).get("name", "unnamed")
        self.auth_token = config.get("vps", {}).get("token", "")
        self.device, self.device_info = detect_device()
        self.start_time = time.time()

        vps_host = config.get("vps", {}).get("host")
        vps_port = config.get("vps", {}).get("port", 7434)
        vps_url = f"http://{vps_host}:{vps_port}" if vps_host else None

        self.streamer = ResultStreamer(
            vps_url=vps_url,
            results_dir=config.get("results_dir", "./results")
        )

        handler.server_ref = self
        super().__init__(addr, handler)


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="GAIA Worker Agent")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--standalone", action="store_true", help="Run without VPS")
    parser.add_argument("--job", help="Run a single job JSON file and exit")
    parser.add_argument("--port", type=int, help="Override listen port")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        print(f"⚠️  Config not found at {config_path}, using defaults")
        config = {}

    if args.standalone:
        config.setdefault("vps", {})["host"] = None

    # Detect device
    device, device_info = detect_device()

    print(f"""
╔══════════════════════════════════════════╗
║         GAIA Worker Agent v1.0           ║
╠══════════════════════════════════════════╣
║  Worker:  {config.get('worker', {}).get('name', 'unnamed'):>30} ║
║  Device:  {device_info.get('name', 'unknown'):>30} ║
║  Mode:    {'Standalone' if args.standalone else 'Connected':>30} ║
╚══════════════════════════════════════════╝
""")

    # Single job mode
    if args.job:
        streamer = ResultStreamer(results_dir=config.get("results_dir", "./results"))
        with open(args.job) as f:
            job = json.load(f)
        run_job(job, device, streamer)
        return

    # Server mode
    port = args.port or config.get("server", {}).get("port", 7433)
    server = WorkerServer(("0.0.0.0", port), WorkerHandler, config)

    def shutdown(sig, frame):
        print("\n👋 Shutting down worker...")
        server.streamer.flush_buffer()
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"👂 Listening on port {port} for jobs...")
    print(f"   Status: http://localhost:{port}/api/status")
    print(f"   Submit: POST http://localhost:{port}/api/submit_job\n")

    server.serve_forever()


if __name__ == "__main__":
    main()
