#!/usr/bin/env python3
"""
GAIA Experiment Runner.

Runs gradient-free methods on any Gymnasium environment.

Usage:
    python run_all.py --method cma_es --environment LunarLander-v3 --max-evals 100000
    python run_all.py --method neuromod --environment BipedalWalker-v3 --max-evals 500000
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments import (cma_es, openai_es, hybrid_ff, curriculum, indirect,
                        scaling, neuromod, island_model, island_advanced, neuromod_island,
                        ppo_baseline)

# ─── Environment configs ───────────────────────────────────────────────
ENVIRONMENTS = {
    "LunarLander-v3": {
        "obs_dim": 8,
        "act_dim": 4,
        "act_type": "discrete",
        "solved": 200,
        "max_steps": 1000,
        "hidden": [64, 32],  # default network size
    },
    "BipedalWalker-v3": {
        "obs_dim": 24,
        "act_dim": 4,
        "act_type": "continuous",
        "solved": 300,
        "max_steps": 1600,
        "hidden": [128, 64],
    },
    "BipedalWalkerHardcore-v3": {
        "obs_dim": 24,
        "act_dim": 4,
        "act_type": "continuous",
        "solved": 300,
        "max_steps": 2000,
        "hidden": [256, 128],
    },
    # ─── Atari (Phase 10) ──────────────────────────────────────────
    "ALE/Pong-v5": {
        "obs_type": "pixel",
        "obs_dim": (4, 84, 84),  # 4 stacked grayscale frames
        "act_dim": 6,
        "act_type": "discrete",
        "solved": 21,
        "max_steps": 10000,
        "hidden": None,  # CNN, not MLP
        "n_frames": 4,
    },
    "ALE/Breakout-v5": {
        "obs_type": "pixel",
        "obs_dim": (4, 84, 84),
        "act_dim": 4,
        "act_type": "discrete",
        "solved": 30,
        "max_steps": 10000,
        "hidden": None,
        "n_frames": 4,
    },
    "ALE/SpaceInvaders-v5": {
        "obs_type": "pixel",
        "obs_dim": (4, 84, 84),
        "act_dim": 6,
        "act_type": "discrete",
        "solved": 500,
        "max_steps": 10000,
        "hidden": None,
        "n_frames": 4,
    },
}

# ─── Methods ───────────────────────────────────────────────────────────
METHODS = {
    "cma_es":            ("CMA-ES", cma_es),
    "openai_es":         ("OpenAI-ES", openai_es),
    "hybrid_cma_ff":     ("Hybrid CMA+FF", hybrid_ff),
    "curriculum":        ("Curriculum", curriculum),
    "indirect_encoding": ("Indirect Encoding", indirect),
    "scaling":           ("Scaling", scaling),
    "neuromod":          ("Neuromod CMA-ES", neuromod),
    "island_model":      ("Island Model", island_model),
    "island_advanced":   ("Island Advanced", island_advanced),
    "neuromod_island":   ("Neuromod Island", neuromod_island),
    "ppo_baseline":      ("PPO (Backprop)", ppo_baseline),
}


def run_experiment(name, label, module, params, results_dir):
    """Run a single experiment."""
    env_name = params.get("environment", "LunarLander-v3")
    env_cfg = ENVIRONMENTS.get(env_name, {})

    print(f"\n{'='*70}")
    print(f"🧬 {label} on {env_name}")
    print(f"   Budget: {params.get('max_evals', 100000):,} evals | Solved: ≥{env_cfg.get('solved', '?')}")
    print(f"{'='*70}\n")

    method_dir = results_dir / f"{env_name}_{name}"
    method_dir.mkdir(parents=True, exist_ok=True)

    csv_path = method_dir / "training.csv"
    csv_written_header = [False]

    def callback(data):
        if not csv_written_header[0]:
            with open(csv_path, "w") as f:
                f.write(",".join(data.keys()) + "\n")
            csv_written_header[0] = True
        with open(csv_path, "a") as f:
            f.write(",".join(str(v) for v in data.values()) + "\n")

    start = time.time()
    result = module.run(params=params, callback=callback)
    result["wall_time"] = round(time.time() - start, 1)
    result["environment"] = env_name

    import numpy as np
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(f"Not JSON serializable: {type(obj)}")
    with open(method_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=convert)

    solved = result.get("solved", False)
    print(f"\n{'='*70}")
    print(f"{'✅ SOLVED!' if solved else '❌ Not solved'} — {label} on {env_name}")
    print(f"   Best: {result.get('best_ever', 'N/A')} | Time: {result.get('wall_time', 'N/A')}s")
    print(f"{'='*70}\n")

    return result


def main():
    parser = argparse.ArgumentParser(description="GAIA Experiment Runner")
    parser.add_argument("--method", choices=list(METHODS.keys()), required=True, help="Algorithm to run")
    parser.add_argument("--environment", type=str, required=True, help="Gymnasium environment ID")
    parser.add_argument("--max-evals", type=int, default=100000, help="Max evaluations")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes per evaluation")
    parser.add_argument("--results-dir", default="./results", help="Results directory")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    env_cfg = ENVIRONMENTS.get(args.environment)
    if not env_cfg:
        print(f"⚠️  Unknown environment '{args.environment}' — running with defaults")
        env_cfg = {"obs_dim": 8, "act_dim": 4, "act_type": "discrete", "solved": 200, "max_steps": 1000, "hidden": [64, 32]}

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "max_evals": args.max_evals,
        "eval_episodes": args.eval_episodes,
        "environment": args.environment,
        # Pass env config so experiments can adapt
        "obs_dim": env_cfg["obs_dim"],
        "act_dim": env_cfg["act_dim"],
        "act_type": env_cfg["act_type"],
        "solved_threshold": env_cfg["solved"],
        "max_steps": env_cfg["max_steps"],
        "hidden": env_cfg.get("hidden"),
        # Pixel env support (Atari)
        "obs_type": env_cfg.get("obs_type", "vector"),
        "n_frames": env_cfg.get("n_frames", 4),
    }

    # Merge extra params from GAIA_JOB_PARAMS env var (set by worker)
    extra_json = os.environ.get("GAIA_JOB_PARAMS", "")
    if extra_json:
        try:
            extra = json.loads(extra_json)
            # Only merge known experiment params, don't override env config
            for key in ("pop_size", "lr", "noise_std", "sigma0", "n_workers",
                        "n_islands", "migration_interval", "config"):
                if key in extra:
                    params[key] = extra[key]
                    print(f"  [params] {key}={extra[key]}")
        except json.JSONDecodeError:
            pass

    label, module = METHODS[args.method]

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  GAIA — Gradient-Free RL                                ║
╠══════════════════════════════════════════════════════════╣
║  Method:      {label:<42s} ║
║  Environment: {args.environment:<42s} ║
║  Budget:      {args.max_evals:<42,} ║
║  Solved:      ≥{env_cfg['solved']:<41} ║
╚══════════════════════════════════════════════════════════╝
""")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("💻 CPU mode")
    except ImportError:
        print("💻 CPU mode (numpy)")

    result = run_experiment(args.method, label, module, params, results_dir)

    print(f"\n✅ Done. Best: {result.get('best_ever', 'N/A')} | Solved: {result.get('solved', False)}")


if __name__ == "__main__":
    main()
