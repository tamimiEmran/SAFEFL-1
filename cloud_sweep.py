#!/usr/bin/env python3
"""
Cloud experiment sweep for SAFEFL.
Upload this file to your Vertex AI Workbench and run:

    python cloud_sweep.py --setup          # first time: clone repo, install deps
    python cloud_sweep.py --gpu 0          # run all experiments
    python cloud_sweep.py --gpu 0 --dry    # preview commands without running
"""

import subprocess
import sys
import argparse
import itertools
import os
import time


# ============================================================
# EXPERIMENT CONFIGURATION — edit these to change the sweep
# ============================================================

REPO_URL = "https://github.com/tamimiEmran/SAFEFL-1.git"
BRANCH = "March26"
REPO_DIR = "SAFEFL-1"

DATASETS = ["FEMNIST", "MNIST"]
BIASES = [0, 0.5]
MODELS = ["mobilenet_v3_small"]
ATTACKS = ["no", "scaling_attack", "label_flipping_attack"]
DEFENCES = ["factorGraphs", "krum", "shieldfl", "signguard"]
NBYZ = 20
GROUP_SIZE = 10

BASE_ARGS = {
    "nworkers": 200,
    "batch_size": 32,
    "niter": 2500,
    "lr": 0.1,
    "test_every": 10,
}


# ============================================================
# SETUP
# ============================================================

def setup():
    """Clone the repo and install all dependencies."""
    print("=== SETUP: Cloning repo ===")
    if os.path.exists(REPO_DIR):
        print(f"  {REPO_DIR}/ already exists, pulling latest...")
        run(f"cd {REPO_DIR} && git checkout {BRANCH} && git pull")
    else:
        run(f"git clone {REPO_URL}")
        run(f"cd {REPO_DIR} && git checkout {BRANCH}")

    print("\n=== SETUP: Installing dependencies ===")
    run("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    run("pip install jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
    run("pip install pgmax scikit-learn hdbscan scipy matplotlib pandas tqdm pyds")

    # Ensure output dirs exist
    run(f"mkdir -p {REPO_DIR}/score_function_viz")
    run(f"mkdir -p {REPO_DIR}/results/figures")

    print("\n=== SETUP: Verifying GPU ===")
    run('python -c "import torch; print(f\'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}\')"')
    run('python -c "import jax; print(f\'JAX devices: {jax.devices()}\')"')

    print("\n=== SETUP COMPLETE ===")


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def build_experiments(gpu):
    """Generate all (command_string, description) pairs."""
    experiments = []

    for ds, bias, model, atk, defence in itertools.product(
        DATASETS, BIASES, MODELS, ATTACKS, DEFENCES
    ):
        # Attack args
        atk_args = f"--nbyz {NBYZ}" if atk != "no" else ""

        # Grouped run
        desc = f"{ds} | bias={bias} | {atk} | {defence} | grouped"
        cmd = (
            f"python main.py"
            f" --nworkers {BASE_ARGS['nworkers']}"
            f" --batch_size {BASE_ARGS['batch_size']}"
            f" --niter {BASE_ARGS['niter']}"
            f" --lr {BASE_ARGS['lr']}"
            f" --test_every {BASE_ARGS['test_every']}"
            f" --gpu {gpu}"
            f" --dataset {ds}"
            f" --bias {bias}"
            f" --net {model}"
            f" --byz_type {atk}"
            f" --aggregation {defence}"
            f" --isGrouped True"
            f" --group_size {GROUP_SIZE}"
            f" {atk_args}"
        )
        experiments.append((cmd.strip(), desc))

        # Non-grouped run (skip for factorGraphs — always grouped)
        if defence != "factorGraphs":
            desc_ng = f"{ds} | bias={bias} | {atk} | {defence} | non-grouped"
            cmd_ng = (
                f"python main.py"
                f" --nworkers {BASE_ARGS['nworkers']}"
                f" --batch_size {BASE_ARGS['batch_size']}"
                f" --niter {BASE_ARGS['niter']}"
                f" --lr {BASE_ARGS['lr']}"
                f" --test_every {BASE_ARGS['test_every']}"
                f" --gpu {gpu}"
                f" --dataset {ds}"
                f" --bias {bias}"
                f" --net {model}"
                f" --byz_type {atk}"
                f" --aggregation {defence}"
                f" --isGrouped False"
                f" {atk_args}"
            )
            experiments.append((cmd_ng.strip(), desc_ng))

    return experiments


def run(cmd):
    """Run a shell command, printing it first."""
    print(f"  $ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def run_sweep(gpu, dry=False):
    """Run the full experiment sweep."""
    experiments = build_experiments(gpu)
    total = len(experiments)
    print(f"\n{'DRY RUN — ' if dry else ''}Total experiments: {total}\n")

    failed = []
    for i, (cmd, desc) in enumerate(experiments, 1):
        header = f"[{i}/{total}] {desc}"
        print(f"\n{'=' * len(header)}")
        print(header)
        print(f"{'=' * len(header)}")

        if dry:
            print(f"  $ cd {REPO_DIR} && {cmd}")
            continue

        start = time.time()
        try:
            subprocess.run(
                cmd,
                shell=True,
                check=True,
                cwd=REPO_DIR,
            )
            elapsed = time.time() - start
            print(f"  DONE in {elapsed:.0f}s")
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start
            print(f"  FAILED (exit code {e.returncode}) after {elapsed:.0f}s")
            failed.append(desc)

    # Summary
    print(f"\n{'=' * 40}")
    print(f"SWEEP COMPLETE: {total - len(failed)}/{total} succeeded")
    if failed:
        print(f"\nFailed experiments:")
        for f in failed:
            print(f"  - {f}")
    print(f"{'=' * 40}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAFEFL cloud experiment sweep")
    parser.add_argument("--setup", action="store_true", help="Clone repo and install dependencies")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (default: 0)")
    parser.add_argument("--dry", action="store_true", help="Print commands without running them")
    args = parser.parse_args()

    if args.setup:
        setup()
    else:
        run_sweep(args.gpu, dry=args.dry)
