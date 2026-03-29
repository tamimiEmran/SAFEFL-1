#!/usr/bin/env python3
"""
SAFEFL Benchmark Sweep
======================
Runs all experiment combinations automatically. No interactive input required.

Usage:
    python run_benchmarks.py                    # run all on GPU 0
    python run_benchmarks.py --gpu 1            # run all on GPU 1
    python run_benchmarks.py --dry              # preview commands without running
    python run_benchmarks.py --filter krum      # only run experiments matching "krum"
    python run_benchmarks.py --filter "FEMNIST,scaling"  # match multiple terms (AND)
"""

import subprocess
import sys
import shlex
import argparse
import time
import os

# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

DATASETS = ["FEMNIST", "MNIST"]
BIAS_VALUES = [0, 0.25, 0.5, 0.75, 1]
MODELS = ["mobilenet_v3_small", "eff_net", "resnet18"]
ATTACK_TYPES = ["no", "scaling_attack", "label_flipping_attack"]
DEFENCES = ["fedavg", "krum", "shieldfl", "signguard", "factorGraphs"]
GROUP_SIZES = [5, 10, 20, 30]
NBYZ_LIST = [10, 20, 30]

# Per-dataset training configs
DATASET_ARGS = {
    "FEMNIST": {"nworkers": 300, "batch_size": 32, "niter": 2500, "lr": 0.1, "test_every": 10},
    "MNIST":   {"nworkers": 300, "batch_size": 32, "niter": 1500, "lr": 0.1, "test_every": 10},
}


# ============================================================
# EXPERIMENT GENERATION
# ============================================================

def generate_experiments(gpu):
    """
    Generate all experiment commands respecting these rules:
      - fedavg: always non-grouped, no group_size variation (grouping has no effect)
      - factorGraphs: always grouped (requires grouping to function)
      - krum, shieldfl, signguard: both grouped and non-grouped
    """
    experiments = []

    for dataset in DATASETS:
        cfg = DATASET_ARGS[dataset]
        base = [
            "--nworkers", str(cfg["nworkers"]),
            "--batch_size", str(cfg["batch_size"]),
            "--niter", str(cfg["niter"]),
            "--lr", str(cfg["lr"]),
            "--test_every", str(cfg["test_every"]),
            "--gpu", str(gpu),
        ]

        for bias in BIAS_VALUES:
            for model in MODELS:
                for attack in ATTACK_TYPES:
                    # nbyz values: only for actual attacks
                    nbyz_values = NBYZ_LIST if attack != "no" else [None]

                    for nbyz in nbyz_values:
                        for defence in DEFENCES:
                            nbyz_args = ["--nbyz", str(nbyz)] if nbyz is not None else []

                            if defence == "fedavg":
                                # FedAvg: grouping doesn't change results, run once non-grouped
                                cmd = base + [
                                    "--dataset", dataset,
                                    "--bias", str(bias),
                                    "--net", model,
                                    "--byz_type", attack,
                                    "--aggregation", defence,
                                    "--isGrouped", "False",
                                ] + nbyz_args
                                desc = f"{dataset} | bias={bias} | {model} | {attack}{f' nbyz={nbyz}' if nbyz else ''} | {defence}"
                                experiments.append((cmd, desc))

                            elif defence == "factorGraphs":
                                # FactorGraphs: always grouped, sweep group sizes
                                for gs in GROUP_SIZES:
                                    cmd = base + [
                                        "--dataset", dataset,
                                        "--bias", str(bias),
                                        "--net", model,
                                        "--byz_type", attack,
                                        "--aggregation", defence,
                                        "--isGrouped", "True",
                                        "--group_size", str(gs),
                                    ] + nbyz_args
                                    desc = f"{dataset} | bias={bias} | {model} | {attack}{f' nbyz={nbyz}' if nbyz else ''} | {defence} | gs={gs}"
                                    experiments.append((cmd, desc))

                            else:
                                # krum, shieldfl, signguard: grouped + non-grouped
                                # Non-grouped
                                cmd = base + [
                                    "--dataset", dataset,
                                    "--bias", str(bias),
                                    "--net", model,
                                    "--byz_type", attack,
                                    "--aggregation", defence,
                                    "--isGrouped", "False",
                                ] + nbyz_args
                                desc = f"{dataset} | bias={bias} | {model} | {attack}{f' nbyz={nbyz}' if nbyz else ''} | {defence} | non-grouped"
                                experiments.append((cmd, desc))

                                # Grouped with each group size
                                for gs in GROUP_SIZES:
                                    cmd = base + [
                                        "--dataset", dataset,
                                        "--bias", str(bias),
                                        "--net", model,
                                        "--byz_type", attack,
                                        "--aggregation", defence,
                                        "--isGrouped", "True",
                                        "--group_size", str(gs),
                                    ] + nbyz_args
                                    desc = f"{dataset} | bias={bias} | {model} | {attack}{f' nbyz={nbyz}' if nbyz else ''} | {defence} | gs={gs}"
                                    experiments.append((cmd, desc))

    return experiments


# ============================================================
# RUNNER
# ============================================================

def run_sweep(experiments, dry=False):
    total = len(experiments)
    failed = []
    print(f"\n{'DRY RUN — ' if dry else ''}Total experiments: {total}\n")

    for i, (cmd, desc) in enumerate(experiments, 1):
        full_cmd = [sys.executable, "main.py"] + cmd
        header = f"[{i}/{total}] {desc}"
        print(f"\n{'=' * 80}")
        print(header)
        print(f"{'=' * 80}")

        if dry:
            print(f"  {shlex.join(full_cmd)}")
            continue

        start = time.time()
        try:
            subprocess.run(full_cmd, check=True)
            elapsed = time.time() - start
            print(f"  DONE in {elapsed:.0f}s")
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start
            print(f"  FAILED (exit code {e.returncode}) after {elapsed:.0f}s")
            failed.append(desc)
        except FileNotFoundError:
            print("  ERROR: main.py not found. Run this script from the SAFEFL directory.")
            sys.exit(1)

    # Summary
    print(f"\n{'=' * 80}")
    print(f"SWEEP COMPLETE: {total - len(failed)}/{total} succeeded")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")
    print(f"{'=' * 80}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAFEFL Benchmark Sweep")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID (default: 0)")
    parser.add_argument("--dry", action="store_true", help="Preview commands without running")
    parser.add_argument("--filter", type=str, default=None,
                        help="Comma-separated terms to filter experiments (AND logic). "
                             "E.g. --filter 'FEMNIST,krum' runs only FEMNIST+krum experiments.")
    parser.add_argument("--baselines", action="store_true",
                        help="Run only baseline experiments: no attack, no grouping, no factorGraphs. "
                             "Tests that all models/datasets/defences converge properly.")
    args = parser.parse_args()

    # Ensure output directories exist
    for d in ["results3rdOct", "finalResults", "score_function_viz", "results/figures"]:
        os.makedirs(d, exist_ok=True)

    experiments = generate_experiments(args.gpu)

    # Baselines: no attack, no grouping, no factorGraphs
    if args.baselines:
        experiments = [(cmd, desc) for cmd, desc in experiments
                       if "| no |" in desc                    # no attack
                       and "factorGraphs" not in desc          # exclude factorGraphs
                       and "gs=" not in desc]                  # exclude grouped runs

    # Apply filter if provided
    if args.filter:
        terms = [t.strip().lower() for t in args.filter.split(",")]
        experiments = [(cmd, desc) for cmd, desc in experiments
                       if all(t in desc.lower() for t in terms)]

    if not experiments:
        print(f"No experiments match the given flags.")
        sys.exit(0)

    run_sweep(experiments, dry=args.dry)
