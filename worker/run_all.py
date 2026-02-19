#!/usr/bin/env python3
"""
Run all GAIA Phase 7 experiments locally.

Usage:
    python run_all.py                    # Run all methods
    python run_all.py --method cma_es    # Run single method
    python run_all.py --max-evals 50000  # Custom budget
    python run_all.py --quick            # Quick test (10K evals)

Results are saved to ./results/phase7/
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments import cma_es, openai_es, hybrid_ff, curriculum, indirect
from experiments import bipedal_cma, bipedal_es, scaling


METHODS = {
    # Phase 7 — LunarLander
    "cma_es": ("CMA-ES", cma_es),
    "openai_es": ("OpenAI ES", openai_es),
    "hybrid_cma_ff": ("Hybrid CMA+FF", hybrid_ff),
    "curriculum": ("Curriculum CMA-ES", curriculum),
    "indirect_encoding": ("Indirect Encoding", indirect),
    # Phase 8 — BipedalWalker
    "bipedal_cma": ("BipedalWalker CMA-ES", bipedal_cma),
    "bipedal_es": ("BipedalWalker OpenAI-ES", bipedal_es),
    # Scaling experiments
    "scaling": ("Network Scaling", scaling),
}


def run_experiment(name, label, module, params, results_dir):
    """Run a single experiment and save results."""
    print(f"\n{'='*70}")
    print(f"🧬 Starting: {label}")
    print(f"   Budget: {params.get('max_evals', 100000)} evaluations")
    print(f"{'='*70}\n")

    method_dir = results_dir / name
    method_dir.mkdir(parents=True, exist_ok=True)

    # CSV callback
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

    # Save result (convert numpy types for JSON)
    import numpy as np
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    with open(method_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=convert)

    print(f"\n{'='*70}")
    print(f"{'✅ SOLVED!' if result.get('solved') else '❌ Not solved'} — {label}")
    print(f"   Best: {result.get('best_ever', 'N/A')}")
    print(f"   Final Mean: {result.get('final_mean', 'N/A')}")
    print(f"   Time: {result.get('wall_time', 'N/A')}s")
    print(f"{'='*70}\n")

    return result


def generate_plots(results_dir):
    """Generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("⚠️  matplotlib not installed, skipping plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Learning curves
    ax = axes[0]
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6"]

    for i, (name, (label, _)) in enumerate(METHODS.items()):
        csv_path = results_dir / name / "training.csv"
        if csv_path.exists():
            data = np.genfromtxt(csv_path, delimiter=",", names=True)
            if len(data) > 0:
                evals = data["total_evals"]
                best = data["best_ever"]
                ax.plot(evals, best, label=label, color=colors[i % len(colors)], linewidth=2)

    ax.axhline(y=200, color="gray", linestyle="--", alpha=0.7, label="Solved (200)")
    ax.set_xlabel("Total Evaluations", fontsize=12)
    ax.set_ylabel("Best Score Ever", fontsize=12)
    ax.set_title("GAIA Phase 7: Gradient-Free Methods on LunarLander", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Final comparison bars
    ax2 = axes[1]
    names = []
    best_scores = []
    final_means = []

    for name, (label, _) in METHODS.items():
        result_path = results_dir / name / "result.json"
        if result_path.exists():
            with open(result_path) as f:
                r = json.load(f)
            names.append(label)
            best_scores.append(r.get("best_ever", 0))
            final_means.append(r.get("final_mean", 0))

    if names:
        x = np.arange(len(names))
        width = 0.35
        ax2.bar(x - width/2, best_scores, width, label="Best Ever", color="#3498db")
        ax2.bar(x + width/2, final_means, width, label="Final Mean", color="#2ecc71")
        ax2.axhline(y=200, color="red", linestyle="--", alpha=0.7, label="Solved (200)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax2.set_ylabel("Score", fontsize=12)
        ax2.set_title("Final Performance Comparison", fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(results_dir / "phase7_results.png", dpi=300, bbox_inches="tight")
    print(f"📊 Plots saved to {results_dir / 'phase7_results.png'}")


def main():
    parser = argparse.ArgumentParser(description="GAIA Phase 7: Gradient-Free LunarLander")
    parser.add_argument("--method", choices=list(METHODS.keys()), help="Run single method")
    parser.add_argument("--max-evals", type=int, default=100000, help="Max evaluations per method")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes per evaluation")
    parser.add_argument("--quick", action="store_true", help="Quick test (10K evals)")
    parser.add_argument("--results-dir", default="./results/phase7", help="Results directory")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    if args.quick:
        args.max_evals = 10000

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "max_evals": args.max_evals,
        "eval_episodes": args.eval_episodes,
    }

    print(f"""
╔══════════════════════════════════════════════════════╗
║           GAIA Phase 7: The Final Push               ║
║    Gradient-Free RL: LunarLander + BipedalWalker     ║
╠══════════════════════════════════════════════════════╣
║  Budget: {args.max_evals:>8} evaluations per method            ║
║  Methods: {len(METHODS) if not args.method else 1:>2}                                          ║
║  Results: {str(results_dir):>40} ║
╚══════════════════════════════════════════════════════╝
""")

    # Detect GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("💻 No GPU detected, using CPU")
    except ImportError:
        print("💻 PyTorch not installed, using numpy (CPU)")

    all_results = {}
    start_total = time.time()

    methods_to_run = {args.method: METHODS[args.method]} if args.method else METHODS

    for name, (label, module) in methods_to_run.items():
        result = run_experiment(name, label, module, params, results_dir)
        all_results[name] = result

    # Summary
    total_time = time.time() - start_total
    print(f"\n{'='*70}")
    print(f"📋 PHASE 7 SUMMARY (total time: {total_time:.0f}s)")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'Best':>10} {'Final Mean':>12} {'Solved':>8} {'Time':>8}")
    print(f"{'-'*65}")

    for name, r in all_results.items():
        label = METHODS[name][0]
        solved = "✅" if r.get("solved") else "❌"
        print(f"{label:<25} {r.get('best_ever', 0):>10.1f} {r.get('final_mean', 0):>12.1f} {solved:>8} {r.get('wall_time', 0):>7.0f}s")

    # Save summary
    import numpy as np
    def _conv(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(f"Not JSON serializable: {type(obj)}")
    with open(results_dir / "summary.json", "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_time": round(total_time, 1),
            "params": params,
            "results": all_results,
        }, f, indent=2, default=_conv)

    if not args.no_plots:
        generate_plots(results_dir)

    print(f"\n✅ All results saved to {results_dir}")


if __name__ == "__main__":
    main()
