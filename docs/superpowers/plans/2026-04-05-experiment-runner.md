# Experiment Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a phase-aware `experiment_runner.py` that orchestrates the 8-phase experiment plan, with a `phase_config.json` config file for inter-phase decisions and cascading defaults.

**Architecture:** Single file `experiment_runner.py` with four concerns: config management (load/save/init/validate `phase_config.json`), experiment generation (phase definitions as data, defaults resolution), execution (subprocess runner with timing/failure tracking), and decision prompts (interactive CLI for `--decide`). Phase definitions are dicts — adding/modifying phases is a data change.

**Tech Stack:** Python stdlib (`json`, `subprocess`, `sys`, `argparse`, `time`, `shlex`), no new dependencies.

---

## File Structure

| File | Responsibility |
|------|---------------|
| **Create:** `experiment_runner.py` | All orchestration: CLI, config management, phase generation, execution, decision prompts |

Single file. No need to split — the concerns are tightly coupled (phase generation reads config, decisions write config, execution calls generation) and the total size will be ~500 lines.

---

### Task 1: Config Management — Load, Init, Save, Validate

**Files:**
- Create: `experiment_runner.py`

This task creates the skeleton with config file handling. The config file is the backbone — everything else reads/writes it.

- [ ] **Step 1: Create `experiment_runner.py` with config management functions**

```python
#!/usr/bin/env python3
"""
SAFEFL Phase-Aware Experiment Runner
=====================================
Orchestrates the 8-phase experiment plan from experiment_plan.html.
Each phase generates only the experiments it needs, reading decisions
from earlier phases via phase_config.json.

Usage:
    python experiment_runner.py --phase 0 --gpu 0          # run phase 0
    python experiment_runner.py --phase 0 --gpu 0 --dry    # preview commands
    python experiment_runner.py --decide 0                  # record decisions
    python experiment_runner.py --status                    # show progress
    python experiment_runner.py --phase 4 --override bias=0.5,0.75
"""

import json
import os
import sys
import subprocess
import argparse
import time
import shlex

CONFIG_PATH = "phase_config.json"

# ============================================================
# PHASE METADATA — names, dependencies, descriptions
# ============================================================

PHASE_META = {
    0: {"name": "Baseline Landscape", "deps": [], "desc": "Dataset x Architecture selection"},
    1: {"name": "Grouping Failure", "deps": [0], "desc": "Do standard defenses collapse under grouping?"},
    2: {"name": "Inference Recovery", "deps": [1], "desc": "Does FactorGraphs recover accuracy?"},
    3: {"name": "Backdoor Defense", "deps": [1], "desc": "Can FactorGraphs suppress backdoors?"},
    4: {"name": "Non-IID Robustness", "deps": [2, 3], "desc": "Performance across data heterogeneity"},
    5: {"name": "Generalization", "deps": [4], "desc": "Results transfer to second dataset"},
    6: {"name": "Sensitivity Grid", "deps": [2], "desc": "Group size x Byzantine count sweep"},
    7: {"name": "Architecture Comparison", "deps": [2], "desc": "Defense transfer across architectures"},
}

ALL_ARCHS = ["mobilenet_v3_small", "resnet18", "eff_net"]
ALL_DATASETS = ["MNIST", "FEMNIST", "CIFAR10"]
PHASE1_BASELINES = ["krum", "shieldfl", "signguard"]


def default_config():
    """Return a fresh config with global defaults and all phases pending."""
    return {
        "global_defaults": {
            "nworkers": 200,
            "batch_size": 32,
            "niter": 2500,
            "lr": 0.1,
            "test_every": 10,
            "seed": 1,
            "nruns": 1,
        },
        "dataset_defaults": {
            "FEMNIST": {"niter": 2500},
            "MNIST": {"niter": 1500},
            "CIFAR10": {"niter": 3000},
        },
        "phases": {
            str(i): {"status": "pending", "decisions": {}}
            for i in range(8)
        },
    }


def load_config():
    """Load config from disk, or create default if missing."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    cfg = default_config()
    save_config(cfg)
    return cfg


def save_config(cfg):
    """Write config to disk."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def get_phase(cfg, phase_id):
    """Get a phase dict by integer id."""
    return cfg["phases"][str(phase_id)]


def resolve_defaults(cfg, dataset, phase_fixed=None):
    """
    Build the full parameter dict for one experiment.
    Resolution order: global_defaults < dataset_defaults < phase_fixed
    """
    params = dict(cfg["global_defaults"])
    ds_overrides = cfg.get("dataset_defaults", {}).get(dataset, {})
    params.update(ds_overrides)
    if phase_fixed:
        params.update(phase_fixed)
    return params


def check_dependencies(cfg, phase_id):
    """
    Check that all dependency phases are 'decided'.
    Returns (ok, list_of_missing_phase_ids).
    """
    deps = PHASE_META[phase_id]["deps"]
    missing = [d for d in deps if get_phase(cfg, d)["status"] != "decided"]
    return len(missing) == 0, missing
```

- [ ] **Step 2: Verify it runs**

Run: `python -c "import experiment_runner; c = experiment_runner.default_config(); print(len(c['phases']), 'phases'); print('OK')"`
Expected: `8 phases` then `OK`

- [ ] **Step 3: Commit**

```bash
git add experiment_runner.py
git commit -m "feat: add experiment_runner config management skeleton"
```

---

### Task 2: Experiment Generation — All 8 Phase Definitions

**Files:**
- Modify: `experiment_runner.py`

Each phase is a function that reads config decisions and returns a list of `(cmd_args, description)` tuples. A shared helper builds the base args from resolved defaults.

- [ ] **Step 1: Add the generation helpers and all 8 phase generators**

Append to `experiment_runner.py`:

```python
# ============================================================
# EXPERIMENT GENERATION
# ============================================================

def _base_args(params, gpu):
    """Build the common CLI args list from resolved params."""
    args = []
    for key in ["nworkers", "batch_size", "niter", "lr", "test_every", "seed", "nruns"]:
        if key in params:
            args.extend([f"--{key}", str(params[key])])
    args.extend(["--gpu", str(gpu)])
    return args


def _experiment(base, dataset, arch, attack, defence, grouped, group_size, nbyz, extra_desc=""):
    """Build one experiment (cmd_args, description) tuple."""
    cmd = list(base) + [
        "--dataset", dataset,
        "--net", arch,
        "--byz_type", attack,
        "--aggregation", defence,
    ]
    if grouped and group_size > 0:
        cmd.extend(["--isGrouped", "True", "--group_size", str(group_size)])
    else:
        cmd.extend(["--isGrouped", "False"])
    if attack != "no":
        cmd.extend(["--nbyz", str(nbyz)])
    desc = f"{dataset} | {arch} | {attack} | {defence}"
    if grouped and group_size > 0:
        desc += f" | gs={group_size}"
    else:
        desc += " | non-grouped"
    if attack != "no":
        desc += f" | nbyz={nbyz}"
    if extra_desc:
        desc += f" | {extra_desc}"
    return cmd, desc


def _anchor(cfg):
    """Read anchor dataset and arch from Phase 0 decisions."""
    d = get_phase(cfg, 0)["decisions"]
    return d["anchor_dataset"], d["anchor_arch"]


def generate_phase(phase_id, cfg, gpu, overrides=None):
    """
    Generate experiments for a given phase.
    Returns list of (cmd_args, description) tuples.
    """
    generators = {
        0: _gen_phase_0,
        1: _gen_phase_1,
        2: _gen_phase_2,
        3: _gen_phase_3,
        4: _gen_phase_4,
        5: _gen_phase_5,
        6: _gen_phase_6,
        7: _gen_phase_7,
    }
    return generators[phase_id](cfg, gpu, overrides or {})


def _apply_overrides(values, overrides, key):
    """If key is in overrides, replace values with the override list."""
    if key in overrides:
        return [type(values[0])(v) for v in overrides[key].split(",")]
    return values


def _gen_phase_0(cfg, gpu, overrides):
    """Phase 0: Baseline Landscape — 18 runs."""
    experiments = []
    datasets = _apply_overrides(ALL_DATASETS, overrides, "dataset")
    archs = _apply_overrides(ALL_ARCHS, overrides, "net")
    attacks = _apply_overrides(["no", "label_flipping_attack"], overrides, "byz_type")

    for dataset in datasets:
        fixed = {"bias": 0}
        params = resolve_defaults(cfg, dataset, fixed)
        base = _base_args(params, gpu)
        for arch in archs:
            for attack in attacks:
                exp = _experiment(base, dataset, arch, attack, "fedavg",
                                  grouped=False, group_size=0, nbyz=20)
                experiments.append(exp)
    return experiments


def _gen_phase_1(cfg, gpu, overrides):
    """Phase 1: Grouping Failure — 16 runs."""
    dataset, arch = _anchor(cfg)
    fixed = {"bias": 0}
    params = resolve_defaults(cfg, dataset, fixed)
    base = _base_args(params, gpu)

    attacks = _apply_overrides(["no", "label_flipping_attack"], overrides, "byz_type")
    defences = _apply_overrides(["fedavg", "krum", "shieldfl", "signguard"], overrides, "aggregation")
    group_size = int(overrides.get("group_size", "10"))
    nbyz = int(overrides.get("nbyz", "20"))

    experiments = []
    for attack in attacks:
        for defence in defences:
            # Non-grouped
            experiments.append(_experiment(
                base, dataset, arch, attack, defence,
                grouped=False, group_size=0, nbyz=nbyz))
            # Grouped
            experiments.append(_experiment(
                base, dataset, arch, attack, defence,
                grouped=True, group_size=group_size, nbyz=nbyz))
    return experiments


def _gen_phase_2(cfg, gpu, overrides):
    """Phase 2: Inference Recovery — 2 runs."""
    dataset, arch = _anchor(cfg)
    fixed = {"bias": 0}
    params = resolve_defaults(cfg, dataset, fixed)
    base = _base_args(params, gpu)

    attacks = _apply_overrides(["no", "label_flipping_attack"], overrides, "byz_type")
    group_size = int(overrides.get("group_size", "10"))
    nbyz = int(overrides.get("nbyz", "20"))

    experiments = []
    for attack in attacks:
        experiments.append(_experiment(
            base, dataset, arch, attack, "factorGraphs",
            grouped=True, group_size=group_size, nbyz=nbyz))
    return experiments


def _gen_phase_3(cfg, gpu, overrides):
    """Phase 3: Backdoor Defense — 4 runs."""
    dataset, arch = _anchor(cfg)
    fixed = {"bias": 0}
    params = resolve_defaults(cfg, dataset, fixed)
    base = _base_args(params, gpu)

    defences = _apply_overrides(["factorGraphs", "fedavg", "krum", "signguard"], overrides, "aggregation")
    group_size = int(overrides.get("group_size", "10"))
    nbyz = int(overrides.get("nbyz", "20"))

    experiments = []
    for defence in defences:
        grouped = defence != "fedavg"
        experiments.append(_experiment(
            base, dataset, arch, "scaling_attack", defence,
            grouped=grouped, group_size=group_size, nbyz=nbyz))
    return experiments


def _gen_phase_4(cfg, gpu, overrides):
    """Phase 4: Non-IID Robustness — 24 runs."""
    dataset, arch = _anchor(cfg)
    surviving = get_phase(cfg, 1)["decisions"].get("surviving_baselines", [])
    params_base = resolve_defaults(cfg, dataset)
    base = _base_args(params_base, gpu)

    biases = _apply_overrides([0.25, 0.5, 0.75, 1.0], overrides, "bias")
    attacks = _apply_overrides(["label_flipping_attack", "scaling_attack"], overrides, "byz_type")
    defences = ["factorGraphs", "fedavg"] + surviving
    defences = _apply_overrides(defences, overrides, "aggregation")
    group_size = int(overrides.get("group_size", "10"))
    nbyz = int(overrides.get("nbyz", "20"))

    experiments = []
    for bias in biases:
        for attack in attacks:
            for defence in defences:
                grouped = defence not in ("fedavg",)
                # Rebuild base with this bias
                p = resolve_defaults(cfg, dataset, {"bias": float(bias)})
                b = _base_args(p, gpu)
                experiments.append(_experiment(
                    b, dataset, arch, attack, defence,
                    grouped=grouped, group_size=group_size, nbyz=nbyz,
                    extra_desc=f"bias={bias}"))
    return experiments


def _gen_phase_5(cfg, gpu, overrides):
    """Phase 5: Generalization — 6 runs."""
    _, arch = _anchor(cfg)
    surviving = get_phase(cfg, 1)["decisions"].get("surviving_baselines", [])
    best_baseline = surviving[0] if surviving else "signguard"

    # Default second dataset is MNIST; override via config or CLI
    second_dataset = overrides.get("dataset", "MNIST")
    biases = _apply_overrides([0, 0.5], overrides, "bias")
    attacks = _apply_overrides(["label_flipping_attack", "scaling_attack"], overrides, "byz_type")
    defences = _apply_overrides(["factorGraphs", "fedavg", best_baseline], overrides, "aggregation")
    group_size = int(overrides.get("group_size", "10"))
    nbyz = int(overrides.get("nbyz", "20"))

    experiments = []
    for bias in biases:
        for attack in attacks:
            for defence in defences:
                grouped = defence not in ("fedavg",)
                p = resolve_defaults(cfg, second_dataset, {"bias": float(bias)})
                b = _base_args(p, gpu)
                experiments.append(_experiment(
                    b, second_dataset, arch, attack, defence,
                    grouped=grouped, group_size=group_size, nbyz=nbyz,
                    extra_desc=f"bias={bias}"))
    return experiments


def _gen_phase_6(cfg, gpu, overrides):
    """Phase 6: Sensitivity Grid — 12 runs."""
    dataset, arch = _anchor(cfg)
    fixed = {"bias": 0}
    params = resolve_defaults(cfg, dataset, fixed)
    base = _base_args(params, gpu)

    group_sizes = _apply_overrides([5, 10, 20, 30], overrides, "group_size")
    nbyz_values = _apply_overrides([10, 20, 30], overrides, "nbyz")
    attack = overrides.get("byz_type", "label_flipping_attack")

    experiments = []
    for gs in group_sizes:
        for nbyz in nbyz_values:
            experiments.append(_experiment(
                base, dataset, arch, attack, "factorGraphs",
                grouped=True, group_size=int(gs), nbyz=int(nbyz),
                extra_desc=f"k={gs} nbyz={nbyz}"))
    return experiments


def _gen_phase_7(cfg, gpu, overrides):
    """Phase 7: Architecture Comparison — 4 runs."""
    dataset, anchor_arch = _anchor(cfg)
    other_archs = [a for a in ALL_ARCHS if a != anchor_arch]
    other_archs = _apply_overrides(other_archs, overrides, "net")

    fixed = {"bias": 0}
    params = resolve_defaults(cfg, dataset, fixed)
    base = _base_args(params, gpu)

    defences = _apply_overrides(["factorGraphs", "fedavg"], overrides, "aggregation")
    group_size = int(overrides.get("group_size", "10"))
    nbyz = int(overrides.get("nbyz", "20"))
    attack = overrides.get("byz_type", "label_flipping_attack")

    experiments = []
    for arch in other_archs:
        for defence in defences:
            grouped = defence != "fedavg"
            experiments.append(_experiment(
                base, dataset, arch, attack, defence,
                grouped=grouped, group_size=group_size, nbyz=nbyz))
    return experiments
```

- [ ] **Step 2: Verify generation produces correct counts**

```bash
python -c "
from experiment_runner import *
cfg = default_config()
# Phase 0 needs no decisions
exps = generate_phase(0, cfg, gpu=0)
print(f'Phase 0: {len(exps)} experiments (expected 18)')
assert len(exps) == 18, f'Got {len(exps)}'

# Simulate Phase 0 decided for Phase 1
cfg['phases']['0']['status'] = 'decided'
cfg['phases']['0']['decisions'] = {'anchor_dataset': 'FEMNIST', 'anchor_arch': 'mobilenet_v3_small'}
exps = generate_phase(1, cfg, gpu=0)
print(f'Phase 1: {len(exps)} experiments (expected 16)')
assert len(exps) == 16, f'Got {len(exps)}'

# Phase 2
cfg['phases']['1']['status'] = 'decided'
cfg['phases']['1']['decisions'] = {'surviving_baselines': ['signguard'], 'collapsed_baselines': ['krum', 'shieldfl']}
exps = generate_phase(2, cfg, gpu=0)
print(f'Phase 2: {len(exps)} experiments (expected 2)')
assert len(exps) == 2, f'Got {len(exps)}'

# Phase 3
exps = generate_phase(3, cfg, gpu=0)
print(f'Phase 3: {len(exps)} experiments (expected 4)')
assert len(exps) == 4, f'Got {len(exps)}'

# Phase 6
cfg['phases']['2']['status'] = 'decided'
cfg['phases']['2']['decisions'] = {'factorgraphs_works': True}
exps = generate_phase(6, cfg, gpu=0)
print(f'Phase 6: {len(exps)} experiments (expected 12)')
assert len(exps) == 12, f'Got {len(exps)}'

# Phase 7
exps = generate_phase(7, cfg, gpu=0)
print(f'Phase 7: {len(exps)} experiments (expected 4)')
assert len(exps) == 4, f'Got {len(exps)}'

# Phase 4
cfg['phases']['3']['status'] = 'decided'
cfg['phases']['3']['decisions'] = {'factorgraphs_suppresses_backdoor': True}
exps = generate_phase(4, cfg, gpu=0)
print(f'Phase 4: {len(exps)} experiments (expected 24)')
assert len(exps) == 24, f'Got {len(exps)}'

# Phase 5
cfg['phases']['4']['status'] = 'decided'
cfg['phases']['4']['decisions'] = {'crossover_beta': 0.75, 'show_beta_1': True}
exps = generate_phase(5, cfg, gpu=0)
print(f'Phase 5: {len(exps)} experiments (expected 12)')
# 3 defences x 2 attacks x 2 betas = 12
assert len(exps) == 12, f'Got {len(exps)}'

print('All phase counts verified')
"
```

- [ ] **Step 3: Commit**

```bash
git add experiment_runner.py
git commit -m "feat: add all 8 phase generators with defaults resolution and overrides"
```

---

### Task 3: Execution Loop

**Files:**
- Modify: `experiment_runner.py`

Same pattern as `run_benchmarks.py`: sequential subprocess calls with timing and failure tracking.

- [ ] **Step 1: Add the execution function**

Append to `experiment_runner.py`:

```python
# ============================================================
# EXECUTION
# ============================================================

def run_experiments(experiments, dry=False):
    """Run a list of experiments sequentially. Returns list of failed descriptions."""
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

    print(f"\n{'=' * 80}")
    print(f"COMPLETE: {total - len(failed)}/{total} succeeded")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")
    print(f"{'=' * 80}\n")
    return failed
```

- [ ] **Step 2: Verify import**

Run: `python -c "from experiment_runner import run_experiments; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add experiment_runner.py
git commit -m "feat: add experiment execution loop with timing and failure tracking"
```

---

### Task 4: Interactive Decision Prompts

**Files:**
- Modify: `experiment_runner.py`

The `--decide N` command prompts the user for phase-specific decisions and writes them to config.

- [ ] **Step 1: Add decision prompt functions**

Append to `experiment_runner.py`:

```python
# ============================================================
# INTERACTIVE DECISION PROMPTS
# ============================================================

def _prompt_choice(prompt, options):
    """Prompt user to pick from a list. Returns the chosen option."""
    print(f"\n  {prompt}")
    for i, opt in enumerate(options, 1):
        print(f"    {i}. {opt}")
    while True:
        raw = input("  Choice (number): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(f"  Please enter a number between 1 and {len(options)}")


def _prompt_multi(prompt, options):
    """Prompt user to pick multiple from a list. Returns list of chosen options."""
    print(f"\n  {prompt}")
    for i, opt in enumerate(options, 1):
        print(f"    {i}. {opt}")
    while True:
        raw = input("  Choices (comma-separated numbers, or 'none'): ").strip()
        if raw.lower() == "none":
            return []
        try:
            indices = [int(x.strip()) for x in raw.split(",")]
            if all(1 <= idx <= len(options) for idx in indices):
                return [options[idx - 1] for idx in indices]
        except ValueError:
            pass
        print(f"  Please enter comma-separated numbers between 1 and {len(options)}")


def _prompt_yn(prompt):
    """Yes/no prompt. Returns bool."""
    while True:
        raw = input(f"\n  {prompt} (y/n): ").strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please enter y or n")


def _prompt_float(prompt, default=None):
    """Prompt for a float value."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"\n  {prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print("  Please enter a number")


def _prompt_text(prompt):
    """Prompt for optional free text."""
    return input(f"\n  {prompt} (press Enter to skip): ").strip()


def decide_phase(phase_id, cfg):
    """Run interactive decision prompts for a phase and save to config."""
    phase = get_phase(cfg, phase_id)
    if phase["status"] == "pending":
        print(f"\n  Warning: Phase {phase_id} hasn't been run yet (status: pending).")
        if not _prompt_yn("Record decisions anyway?"):
            return

    meta = PHASE_META[phase_id]
    print(f"\n{'=' * 60}")
    print(f"  Phase {phase_id}: {meta['name']}")
    print(f"  {meta['desc']}")
    print(f"{'=' * 60}")

    decisions = {}

    if phase_id == 0:
        decisions["anchor_dataset"] = _prompt_choice(
            "Which dataset had the best testbed properties?", ALL_DATASETS)
        decisions["anchor_arch"] = _prompt_choice(
            "Which architecture for that dataset?", ALL_ARCHS)

    elif phase_id == 1:
        decisions["surviving_baselines"] = _prompt_multi(
            "Which baselines SURVIVED grouping (accuracy didn't collapse)?",
            PHASE1_BASELINES)
        decisions["collapsed_baselines"] = [
            b for b in PHASE1_BASELINES if b not in decisions["surviving_baselines"]]
        print(f"\n  Collapsed: {decisions['collapsed_baselines']}")

    elif phase_id == 2:
        decisions["factorgraphs_works"] = _prompt_yn(
            "Did FactorGraphs recover plaintext-level accuracy under grouped SA?")
        if not decisions["factorgraphs_works"]:
            print("\n  WARNING: Phases 3+ may not be meaningful. Consider tuning BP params first.")

    elif phase_id == 3:
        decisions["factorgraphs_suppresses_backdoor"] = _prompt_yn(
            "Did FactorGraphs achieve high ACC + low ASR (bottom-right quadrant)?")

    elif phase_id == 4:
        decisions["crossover_beta"] = _prompt_float(
            "At which beta did FactorGraphs separate most from baselines?", default=0.75)
        decisions["show_beta_1"] = _prompt_yn("Include beta=1.0 (extreme non-IID) in the paper?")

    elif phase_id == 5:
        decisions["second_dataset"] = _prompt_choice(
            "Which second dataset was used?", [d for d in ALL_DATASETS])
        decisions["need_cifar"] = _prompt_yn("Need additional CIFAR10 runs?")

    elif phase_id == 6:
        decisions["sweet_spot_k"] = int(_prompt_float("What's the sweet-spot group size?", default=10))
        decisions["failure_boundary_notes"] = _prompt_text("Any failure boundaries to note?")

    elif phase_id == 7:
        decisions["arch_agnostic"] = _prompt_yn("Were results consistent across architectures?")

    decisions["notes"] = _prompt_text("Any additional notes for this phase?")

    phase["decisions"] = decisions
    phase["status"] = "decided"
    save_config(cfg)
    print(f"\n  Phase {phase_id} decisions saved. Status: decided.")
```

- [ ] **Step 2: Verify import**

Run: `python -c "from experiment_runner import decide_phase; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add experiment_runner.py
git commit -m "feat: add interactive decision prompts for all 8 phases"
```

---

### Task 5: Status Display

**Files:**
- Modify: `experiment_runner.py`

- [ ] **Step 1: Add the status display function**

Append to `experiment_runner.py`:

```python
# ============================================================
# STATUS DISPLAY
# ============================================================

def show_status(cfg):
    """Print the current state of all phases."""
    print(f"\n{'=' * 60}")
    print("  SAFEFL Experiment Runner — Phase Status")
    print(f"{'=' * 60}\n")

    for pid in range(8):
        meta = PHASE_META[pid]
        phase = get_phase(cfg, pid)
        status = phase["status"].upper()
        decisions = phase.get("decisions", {})

        # Status badge
        if status == "DECIDED":
            badge = f"\033[92m[{status}]\033[0m"
        elif status == "COMPLETED":
            badge = f"\033[93m[{status} - needs --decide {pid}]\033[0m"
        else:
            badge = f"\033[90m[{status}]\033[0m"

        print(f"  Phase {pid}: {meta['name']:<30s} {badge}")

        # Show key decisions if decided
        if status == "DECIDED":
            if pid == 0:
                print(f"    anchor: {decisions.get('anchor_dataset', '?')} + {decisions.get('anchor_arch', '?')}")
            elif pid == 1:
                surv = ", ".join(decisions.get("surviving_baselines", []))
                coll = ", ".join(decisions.get("collapsed_baselines", []))
                print(f"    surviving: {surv or 'none'} | collapsed: {coll or 'none'}")
            elif pid == 2:
                print(f"    works: {decisions.get('factorgraphs_works', '?')}")
            elif pid == 4:
                print(f"    crossover: beta={decisions.get('crossover_beta', '?')}")

        # Show blockers if pending
        if status == "PENDING":
            ok, missing = check_dependencies(cfg, pid)
            if missing:
                blocked = ", ".join(f"Phase {m}" for m in missing)
                print(f"    blocked by: {blocked}")

    print()
```

- [ ] **Step 2: Verify import**

Run: `python -c "from experiment_runner import show_status, default_config; show_status(default_config()); print('OK')"`
Expected: All 8 phases listed as PENDING, then `OK`

- [ ] **Step 3: Commit**

```bash
git add experiment_runner.py
git commit -m "feat: add --status display with dependency checking"
```

---

### Task 6: CLI Main — Wire Everything Together

**Files:**
- Modify: `experiment_runner.py`

- [ ] **Step 1: Add the main CLI entry point**

Append to `experiment_runner.py`:

```python
# ============================================================
# CLI ENTRY POINT
# ============================================================

def parse_overrides(override_str):
    """Parse 'key=val1,val2 key2=val3' into a dict."""
    overrides = {}
    if not override_str:
        return overrides
    for pair in override_str:
        if "=" not in pair:
            print(f"  Invalid override: {pair} (expected key=value)")
            sys.exit(1)
        key, val = pair.split("=", 1)
        overrides[key] = val
    return overrides


def main():
    parser = argparse.ArgumentParser(
        description="SAFEFL Phase-Aware Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python experiment_runner.py --phase 0 --gpu 0          Run Phase 0
  python experiment_runner.py --phase 0 --gpu 0 --dry    Preview Phase 0 commands
  python experiment_runner.py --decide 0                  Record Phase 0 decisions
  python experiment_runner.py --status                    Show progress
  python experiment_runner.py --phase 4 --override bias=0.5,0.75
""")
    parser.add_argument("--phase", type=int, choices=range(8), help="Phase to run (0-7)")
    parser.add_argument("--decide", type=int, choices=range(8), help="Record decisions for phase N")
    parser.add_argument("--status", action="store_true", help="Show phase status")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID (default: 0)")
    parser.add_argument("--dry", action="store_true", help="Preview commands without running")
    parser.add_argument("--filter", type=str, default=None,
                        help="Comma-separated terms to filter experiments (AND logic)")
    parser.add_argument("--override", nargs="*", default=None,
                        help="Override sweep dimensions: key=val1,val2 (e.g., bias=0.5,0.75)")

    args = parser.parse_args()

    # Must specify exactly one action
    actions = sum([args.phase is not None, args.decide is not None, args.status])
    if actions == 0:
        parser.print_help()
        sys.exit(0)
    if actions > 1:
        print("Error: specify exactly one of --phase, --decide, or --status")
        sys.exit(1)

    cfg = load_config()

    # --status
    if args.status:
        show_status(cfg)
        return

    # --decide
    if args.decide is not None:
        decide_phase(args.decide, cfg)
        return

    # --phase
    phase_id = args.phase

    # Check dependencies
    ok, missing = check_dependencies(cfg, phase_id)
    if not ok:
        blocked = ", ".join(f"Phase {m}" for m in missing)
        print(f"\n  Phase {phase_id} is blocked. Missing decisions from: {blocked}")
        print(f"  Run: python experiment_runner.py --decide <phase_id>\n")
        sys.exit(1)

    # Generate experiments
    overrides = parse_overrides(args.override)
    experiments = generate_phase(phase_id, cfg, args.gpu, overrides)

    # Apply filter
    if args.filter:
        terms = [t.strip().lower() for t in args.filter.split(",")]
        experiments = [(cmd, desc) for cmd, desc in experiments
                       if all(t in desc.lower() for t in terms)]

    if not experiments:
        print("  No experiments match the given flags.")
        sys.exit(0)

    meta = PHASE_META[phase_id]
    print(f"\n  Phase {phase_id}: {meta['name']}")
    print(f"  {meta['desc']}")

    # Run
    failed = run_experiments(experiments, dry=args.dry)

    # Update phase status (only if not dry run)
    if not args.dry:
        phase = get_phase(cfg, phase_id)
        phase["status"] = "completed"
        save_config(cfg)
        print(f"  Phase {phase_id} status: completed")
        print(f"  Next step: python experiment_runner.py --decide {phase_id}\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI help works**

Run: `python experiment_runner.py --help`

Expected: Help text with all arguments listed.

- [ ] **Step 3: Verify dry run of Phase 0**

Run: `python experiment_runner.py --phase 0 --gpu 0 --dry`

Expected: 18 experiments printed as commands, each invoking `python main.py` with the correct args. No actual execution.

- [ ] **Step 4: Verify status works**

Run: `python experiment_runner.py --status`

Expected: All 8 phases listed. Phase 0 should now show `COMPLETED` (from the dry run), rest `PENDING`.

- [ ] **Step 5: Verify dependency blocking**

Run: `python experiment_runner.py --phase 1 --dry`

Expected: Error message saying Phase 1 is blocked by Phase 0 decisions.

- [ ] **Step 6: Verify override works**

Run: `python experiment_runner.py --phase 0 --gpu 0 --dry --override dataset=MNIST net=resnet18`

Expected: Only experiments matching MNIST + resnet18 (2 experiments: no attack + label_flipping).

- [ ] **Step 7: Verify filter works**

Run: `python experiment_runner.py --phase 0 --gpu 0 --dry --filter FEMNIST`

Expected: Only FEMNIST experiments (6 out of 18).

- [ ] **Step 8: Commit**

```bash
git add experiment_runner.py
git commit -m "feat: add CLI entry point wiring phase/decide/status/override/filter"
```

---

### Task 7: End-to-End Smoke Test

**Files:**
- No new files

Verify the full workflow without actually running main.py.

- [ ] **Step 1: Clean any leftover config and test the full lifecycle**

```bash
python -c "
import os, json
# Clean up
if os.path.exists('phase_config.json'):
    os.remove('phase_config.json')

from experiment_runner import *

# 1. Load config (creates fresh)
cfg = load_config()
assert os.path.exists('phase_config.json')
assert len(cfg['phases']) == 8
assert all(get_phase(cfg, i)['status'] == 'pending' for i in range(8))

# 2. Generate Phase 0 (no deps)
exps = generate_phase(0, cfg, gpu=0)
assert len(exps) == 18

# 3. Check Phase 1 is blocked
ok, missing = check_dependencies(cfg, 1)
assert not ok and missing == [0]

# 4. Simulate Phase 0 decided
p0 = get_phase(cfg, 0)
p0['status'] = 'decided'
p0['decisions'] = {'anchor_dataset': 'FEMNIST', 'anchor_arch': 'mobilenet_v3_small', 'notes': ''}
save_config(cfg)

# 5. Phase 1 should now be unblocked
ok, _ = check_dependencies(cfg, 1)
assert ok
exps = generate_phase(1, cfg, gpu=0)
assert len(exps) == 16

# 6. Simulate Phase 1 decided
p1 = get_phase(cfg, 1)
p1['status'] = 'decided'
p1['decisions'] = {'surviving_baselines': ['signguard'], 'collapsed_baselines': ['krum', 'shieldfl'], 'notes': ''}

# 7. Phase 2, 3 unblocked
ok, _ = check_dependencies(cfg, 2)
assert ok
ok, _ = check_dependencies(cfg, 3)
assert ok

# 8. Phase 4 still blocked (needs 2 + 3 decided)
ok, missing = check_dependencies(cfg, 4)
assert not ok and set(missing) == {2, 3}

# 9. Simulate Phase 2 + 3 decided
get_phase(cfg, 2)['status'] = 'decided'
get_phase(cfg, 2)['decisions'] = {'factorgraphs_works': True}
get_phase(cfg, 3)['status'] = 'decided'
get_phase(cfg, 3)['decisions'] = {'factorgraphs_suppresses_backdoor': True}

# 10. Phase 4 unblocked, correct count
ok, _ = check_dependencies(cfg, 4)
assert ok
exps = generate_phase(4, cfg, gpu=0)
assert len(exps) == 24

# 11. Override narrows correctly
exps = generate_phase(6, cfg, gpu=0, overrides={'group_size': '10,20'})
assert len(exps) == 6  # 2 group_sizes x 3 nbyz

# 12. Verify a generated command looks right
cmd, desc = generate_phase(2, cfg, gpu=0)[0]
assert '--aggregation' in cmd
assert 'factorGraphs' in cmd
assert '--isGrouped' in cmd
assert 'True' in cmd

save_config(cfg)
print('All lifecycle tests passed')
os.remove('phase_config.json')
"
```

- [ ] **Step 2: Commit**

```bash
git add experiment_runner.py
git commit -m "test: verify full experiment_runner lifecycle"
```
