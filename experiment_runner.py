"""
experiment_runner.py -- Phase-aware experiment runner for SAFEFL.

Orchestrates an 8-phase experiment plan for the federated learning research
paper.  Generates main.py CLI commands, runs them sequentially, and manages
inter-phase decisions via a JSON config file (phase_config.json).
"""

import json
import os
import sys
import subprocess
import argparse
import time
import shlex

# ---------------------------------------------------------------------------
# Section 1: Config Management
# ---------------------------------------------------------------------------

CONFIG_PATH = "phase_config.json"

PHASE_META = {
    0: {"name": "Baseline Landscape",       "deps": [],     "desc": "Dataset x Architecture selection"},
    1: {"name": "Grouping Failure",          "deps": [0],    "desc": "Do standard defenses collapse under grouping?"},
    2: {"name": "Inference Recovery",        "deps": [1],    "desc": "Does FactorGraphs recover accuracy?"},
    3: {"name": "Backdoor Defense",          "deps": [1],    "desc": "Can FactorGraphs suppress backdoors?"},
    4: {"name": "Non-IID Robustness",        "deps": [2, 3], "desc": "Performance across data heterogeneity"},
    5: {"name": "Generalization",            "deps": [4],    "desc": "Results transfer to second dataset"},
    6: {"name": "Sensitivity Grid",          "deps": [2],    "desc": "Group size x Byzantine count sweep"},
    7: {"name": "Architecture Comparison",   "deps": [2],    "desc": "Defense transfer across architectures"},
}

ALL_ARCHS = ["mobilenet_v3_small", "resnet18", "eff_net"]
ALL_DATASETS = ["MNIST", "FEMNIST", "CIFAR10"]
PHASE1_BASELINES = ["krum", "shieldfl", "signguard"]


def default_config():
    """Return a fresh default configuration dictionary."""
    return {
        "global_defaults": {
            "nworkers": 500,
            "batch_size": 32,
            "niter": 2500,
            "lr": 0.1,
            "test_every": 10,
            "seed": 1,
            "nruns": 1,
        },
        "dataset_defaults": {
            "FEMNIST": {"niter": 2500},
            "MNIST":   {"niter": 1500},
            "CIFAR10": {"niter": 3000},
        },
        "phases": {str(i): {"status": "pending", "decisions": {}} for i in range(8)},
    }


def load_config():
    """Load config from disk, or create and return the default."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    cfg = default_config()
    save_config(cfg)
    return cfg


def save_config(cfg):
    """Persist config to disk."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def get_phase(cfg, phase_id):
    """Return the sub-dict for a given phase."""
    return cfg["phases"][str(phase_id)]


def resolve_defaults(cfg, dataset, phase_fixed=None):
    """Merge global_defaults < dataset_defaults < phase_fixed overrides."""
    params = dict(cfg["global_defaults"])
    ds_overrides = cfg.get("dataset_defaults", {}).get(dataset, {})
    params.update(ds_overrides)
    if phase_fixed:
        params.update(phase_fixed)
    return params


def check_dependencies(cfg, phase_id):
    """Check that every dependency phase has status 'decided'.

    Returns (ok: bool, missing: list[int]).
    """
    deps = PHASE_META[phase_id]["deps"]
    missing = [d for d in deps if get_phase(cfg, d).get("status") != "decided"]
    return (len(missing) == 0, missing)


# ---------------------------------------------------------------------------
# Section 2: Experiment Generation
# ---------------------------------------------------------------------------

def _base_args(params, gpu):
    """Build the common CLI arg list from resolved params."""
    args = []
    for key in ("nworkers", "batch_size", "niter", "lr", "test_every", "seed", "nruns", "bias"):
        if key in params:
            args.extend([f"--{key}", str(params[key])])
    args.extend(["--gpu", str(gpu)])
    return args


def _experiment(base, dataset, arch, attack, defence, grouped, group_size, nbyz, extra_desc=""):
    """Build a (cmd_args_list, description_string) tuple for one experiment."""
    cmd = list(base)
    cmd.extend(["--dataset", dataset])
    cmd.extend(["--net", arch])
    cmd.extend(["--byz_type", attack])
    cmd.extend(["--aggregation", defence])
    cmd.extend(["--isGrouped", "True" if grouped else "False"])
    if grouped:
        cmd.extend(["--group_size", str(group_size)])
    if attack != "no":
        cmd.extend(["--nbyz", str(nbyz)])

    desc_parts = [
        f"{dataset}/{arch}",
        f"atk={attack}",
        f"def={defence}",
        f"grp={'Y' if grouped else 'N'}",
    ]
    if grouped:
        desc_parts.append(f"gs={group_size}")
    if attack != "no":
        desc_parts.append(f"nbyz={nbyz}")
    if extra_desc:
        desc_parts.append(extra_desc)
    desc = " | ".join(desc_parts)
    return (cmd, desc)


def _anchor(cfg):
    """Read anchor_dataset and anchor_arch from Phase 0 decisions."""
    decs = get_phase(cfg, 0).get("decisions", {})
    return decs.get("anchor_dataset", "FEMNIST"), decs.get("anchor_arch", "resnet18")


def generate_phase(phase_id, cfg, gpu, overrides=None):
    """Dispatch to the per-phase generator.  Returns [(cmd, desc), ...]."""
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
    return generators[phase_id](cfg, gpu, overrides)


def _apply_overrides(values, overrides, key):
    """If *key* appears in *overrides*, replace *values* with the parsed override list.

    Casts override strings to match the type of the first element in *values*.
    """
    if overrides and key in overrides:
        if values and not isinstance(values[0], str):
            target_type = type(values[0])
            return [target_type(v) for v in overrides[key]]
        return overrides[key]
    return values


# -- Phase generators -------------------------------------------------------

def _gen_phase_0(cfg, gpu, overrides):
    datasets = _apply_overrides(list(ALL_DATASETS), overrides, "datasets")
    archs = _apply_overrides(list(ALL_ARCHS), overrides, "archs")
    attacks = _apply_overrides(["no", "label_flipping_attack"], overrides, "attacks")
    experiments = []
    for ds in datasets:
        fixed = {"bias": 0}
        params = resolve_defaults(cfg, ds, fixed)
        base = _base_args(params, gpu)
        for arch in archs:
            for atk in attacks:
                experiments.append(
                    _experiment(base, ds, arch, atk, "fedavg", False, 0, 20)
                )
    return experiments


def _gen_phase_1(cfg, gpu, overrides):
    ds, arch = _anchor(cfg)
    defences = _apply_overrides(["fedavg", "krum", "shieldfl", "signguard"], overrides, "defences")
    attacks = _apply_overrides(["no", "label_flipping_attack"], overrides, "attacks")
    fixed = {"bias": 0}
    params = resolve_defaults(cfg, ds, fixed)
    base = _base_args(params, gpu)
    experiments = []
    for defence in defences:
        for grouped in [False, True]:
            for atk in attacks:
                gs = 10 if grouped else 0
                experiments.append(
                    _experiment(base, ds, arch, atk, defence, grouped, gs, 20)
                )
    return experiments


def _gen_phase_2(cfg, gpu, overrides):
    ds, arch = _anchor(cfg)
    attacks = _apply_overrides(["no", "label_flipping_attack"], overrides, "attacks")
    fixed = {"bias": 0}
    params = resolve_defaults(cfg, ds, fixed)
    base = _base_args(params, gpu)
    experiments = []
    for atk in attacks:
        experiments.append(
            _experiment(base, ds, arch, atk, "factorGraphs", True, 10, 20)
        )
    return experiments


def _gen_phase_3(cfg, gpu, overrides):
    ds, arch = _anchor(cfg)
    defences = _apply_overrides(["factorGraphs", "fedavg", "krum", "signguard"], overrides, "defences")
    fixed = {"bias": 0}
    params = resolve_defaults(cfg, ds, fixed)
    base = _base_args(params, gpu)
    experiments = []
    for defence in defences:
        grouped = defence != "fedavg"
        gs = 10 if grouped else 0
        experiments.append(
            _experiment(base, ds, arch, "scaling_attack", defence, grouped, gs, 20)
        )
    return experiments


def _gen_phase_4(cfg, gpu, overrides):
    ds, arch = _anchor(cfg)
    biases = _apply_overrides([0.25, 0.5, 0.75, 1.0], overrides, "biases")
    attacks = _apply_overrides(["label_flipping_attack", "scaling_attack"], overrides, "attacks")
    surviving = get_phase(cfg, 1).get("decisions", {}).get("surviving_baselines", [])
    defences = ["factorGraphs", "fedavg"] + surviving
    defences = _apply_overrides(defences, overrides, "defences")
    experiments = []
    for bias_val in biases:
        fixed = {"bias": bias_val}
        params = resolve_defaults(cfg, ds, fixed)
        base = _base_args(params, gpu)
        for atk in attacks:
            for defence in defences:
                grouped = defence != "fedavg"
                gs = 10 if grouped else 0
                experiments.append(
                    _experiment(base, ds, arch, atk, defence, grouped, gs, 20,
                                extra_desc=f"bias={bias_val}")
                )
    return experiments


def _gen_phase_5(cfg, gpu, overrides):
    p5_decs = get_phase(cfg, 5).get("decisions", {})
    second_ds = p5_decs.get("second_dataset", "MNIST")
    _, arch = _anchor(cfg)
    biases = _apply_overrides([0, 0.5], overrides, "biases")
    attacks = _apply_overrides(["label_flipping_attack", "scaling_attack"], overrides, "attacks")
    surviving = get_phase(cfg, 1).get("decisions", {}).get("surviving_baselines", [])
    best_surviving = surviving[0] if surviving else "krum"
    defences = _apply_overrides(["factorGraphs", "fedavg", best_surviving], overrides, "defences")
    experiments = []
    for bias_val in biases:
        fixed = {"bias": bias_val}
        params = resolve_defaults(cfg, second_ds, fixed)
        base = _base_args(params, gpu)
        for atk in attacks:
            for defence in defences:
                grouped = defence != "fedavg"
                gs = 10 if grouped else 0
                experiments.append(
                    _experiment(base, second_ds, arch, atk, defence, grouped, gs, 20,
                                extra_desc=f"bias={bias_val}")
                )
    return experiments


def _gen_phase_6(cfg, gpu, overrides):
    ds, arch = _anchor(cfg)
    group_sizes = _apply_overrides([5, 10, 20, 30], overrides, "group_sizes")
    nbyz_list = _apply_overrides([10, 20, 30], overrides, "nbyz")
    fixed = {"bias": 0}
    params = resolve_defaults(cfg, ds, fixed)
    base = _base_args(params, gpu)
    experiments = []
    for gs in group_sizes:
        for nbyz in nbyz_list:
            experiments.append(
                _experiment(base, ds, arch, "label_flipping_attack", "factorGraphs",
                            True, gs, nbyz,
                            extra_desc=f"gs={gs} nbyz={nbyz}")
            )
    return experiments


def _gen_phase_7(cfg, gpu, overrides):
    ds, anchor_arch = _anchor(cfg)
    other_archs = [a for a in ALL_ARCHS if a != anchor_arch]
    other_archs = _apply_overrides(other_archs, overrides, "archs")
    defences = _apply_overrides(["factorGraphs", "fedavg"], overrides, "defences")
    fixed = {"bias": 0}
    params = resolve_defaults(cfg, ds, fixed)
    base = _base_args(params, gpu)
    experiments = []
    for arch in other_archs:
        for defence in defences:
            grouped = defence != "fedavg"
            gs = 10 if grouped else 0
            experiments.append(
                _experiment(base, ds, arch, "label_flipping_attack", defence, grouped, gs, 20)
            )
    return experiments


# ---------------------------------------------------------------------------
# Section 3: Execution Loop
# ---------------------------------------------------------------------------

def run_experiments(experiments, dry=False):
    """Run each experiment sequentially.  Returns list of failed descriptions."""
    total = len(experiments)
    failed = []
    print(f"\n{'=' * 60}")
    print(f"  Running {total} experiment(s)  {'[DRY RUN]' if dry else ''}")
    print(f"{'=' * 60}\n")

    for idx, (cmd, desc) in enumerate(experiments, 1):
        print(f"[{idx}/{total}] {desc}")
        full_cmd = [sys.executable, "main.py"] + cmd
        print(f"  CMD: {' '.join(shlex.quote(c) for c in full_cmd)}")

        if dry:
            print("  -> skipped (dry run)\n")
            continue

        t0 = time.time()
        try:
            result = subprocess.run(full_cmd, check=False)
            elapsed = time.time() - t0
            if result.returncode != 0:
                print(f"  -> FAILED (return code {result.returncode}, {elapsed:.1f}s)\n")
                failed.append(desc)
            else:
                print(f"  -> OK ({elapsed:.1f}s)\n")
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  -> ERROR: {exc} ({elapsed:.1f}s)\n")
            failed.append(desc)

    print(f"\n{'=' * 60}")
    print(f"  Summary: {total - len(failed)}/{total} succeeded, {len(failed)} failed")
    if failed:
        print("  Failed experiments:")
        for f in failed:
            print(f"    - {f}")
    print(f"{'=' * 60}\n")
    return failed


# ---------------------------------------------------------------------------
# Section 4: Interactive Decision Prompts
# ---------------------------------------------------------------------------

def _prompt_choice(prompt, options):
    """Pick one from a numbered list."""
    print(prompt)
    for i, opt in enumerate(options):
        print(f"  [{i}] {opt}")
    while True:
        raw = input("Choice (number): ").strip()
        try:
            idx = int(raw)
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print("Invalid choice, try again.")


def _prompt_multi(prompt, options):
    """Pick multiple (comma-separated numbers) or 'none'."""
    print(prompt)
    for i, opt in enumerate(options):
        print(f"  [{i}] {opt}")
    while True:
        raw = input("Choice (comma-separated numbers, or 'none'): ").strip()
        if raw.lower() == "none":
            return []
        try:
            indices = [int(x.strip()) for x in raw.split(",")]
            if all(0 <= i < len(options) for i in indices):
                return [options[i] for i in indices]
        except ValueError:
            pass
        print("Invalid input, try again.")


def _prompt_yn(prompt):
    """Yes/no prompt.  Returns bool."""
    while True:
        raw = input(f"{prompt} (y/n): ").strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Please enter y or n.")


def _prompt_float(prompt, default=None):
    """Float input with optional default."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("Invalid number, try again.")


def _prompt_text(prompt):
    """Optional free text."""
    return input(f"{prompt} (optional, press Enter to skip): ").strip()


def decide_phase(phase_id, cfg):
    """Run interactive prompts for a phase and persist decisions."""
    phase = get_phase(cfg, phase_id)
    meta = PHASE_META[phase_id]
    print(f"\n--- Decide Phase {phase_id}: {meta['name']} ---")
    print(f"    {meta['desc']}\n")

    decisions = phase.get("decisions", {})

    if phase_id == 0:
        decisions["anchor_dataset"] = _prompt_choice(
            "Select anchor dataset:", ALL_DATASETS
        )
        decisions["anchor_arch"] = _prompt_choice(
            "Select anchor architecture:", ALL_ARCHS
        )

    elif phase_id == 1:
        decisions["surviving_baselines"] = _prompt_multi(
            "Which baselines survived grouping?", PHASE1_BASELINES
        )
        decisions["collapsed"] = [
            b for b in PHASE1_BASELINES if b not in decisions["surviving_baselines"]
        ]
        print(f"  Collapsed: {decisions['collapsed']}")

    elif phase_id == 2:
        decisions["factorgraphs_works"] = _prompt_yn(
            "Does FactorGraphs recover accuracy under grouping?"
        )
        if not decisions["factorgraphs_works"]:
            print("  WARNING: Core hypothesis not supported. Consider revisiting Phase 1.")

    elif phase_id == 3:
        decisions["factorgraphs_suppresses_backdoor"] = _prompt_yn(
            "Does FactorGraphs suppress backdoors (scaling_attack)?"
        )

    elif phase_id == 4:
        decisions["crossover_beta"] = _prompt_float(
            "At what bias (beta) does crossover occur?", default=0.75
        )
        decisions["show_beta_1"] = _prompt_yn(
            "Include beta=1.0 (fully non-IID) in the paper?"
        )

    elif phase_id == 5:
        decisions["second_dataset"] = _prompt_choice(
            "Select second dataset:", ALL_DATASETS
        )
        decisions["need_cifar"] = _prompt_yn(
            "Do you need a CIFAR10 run as well?"
        )

    elif phase_id == 6:
        decisions["sweet_spot_k"] = int(_prompt_float(
            "Sweet-spot group size k?", default=10
        ))
        decisions["failure_boundary_notes"] = _prompt_text(
            "Notes on failure boundary"
        )

    elif phase_id == 7:
        decisions["arch_agnostic"] = _prompt_yn(
            "Is the defense architecture-agnostic?"
        )

    # Common: notes
    decisions["notes"] = _prompt_text("Any additional notes for this phase")

    phase["decisions"] = decisions
    phase["status"] = "decided"
    save_config(cfg)
    print(f"\nPhase {phase_id} marked as 'decided'.\n")


# ---------------------------------------------------------------------------
# Section 5: Status Display
# ---------------------------------------------------------------------------

_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_GRAY   = "\033[90m"
_RESET  = "\033[0m"

_STATUS_STYLE = {
    "decided":   (_GREEN,  "DECIDED"),
    "completed": (_YELLOW, "COMPLETED"),
    "pending":   (_GRAY,   "PENDING"),
}


def show_status(cfg):
    """Print a coloured overview of all 8 phases."""
    print(f"\n{'=' * 60}")
    print("  SAFEFL Experiment Phases")
    print(f"{'=' * 60}\n")

    for pid in range(8):
        meta = PHASE_META[pid]
        phase = get_phase(cfg, pid)
        status = phase.get("status", "pending")
        colour, label = _STATUS_STYLE.get(status, (_GRAY, status.upper()))

        print(f"  Phase {pid}: {meta['name']}")
        print(f"    Status : {colour}[{label}]{_RESET}")
        print(f"    Desc   : {meta['desc']}")

        if status == "decided":
            decs = phase.get("decisions", {})
            for k, v in decs.items():
                if k == "notes" and not v:
                    continue
                print(f"    >> {k}: {v}")
        elif status == "pending":
            deps = meta["deps"]
            if deps:
                missing = [
                    d for d in deps if get_phase(cfg, d).get("status") != "decided"
                ]
                if missing:
                    print(f"    Blocked by: Phase(s) {missing}")
        print()

    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Section 6: CLI Entry Point
# ---------------------------------------------------------------------------

def parse_overrides(override_str):
    """Parse a list of 'key=val1,val2' strings into a dict of lists."""
    result = {}
    if not override_str:
        return result
    for item in override_str:
        if "=" not in item:
            print(f"Warning: ignoring malformed override '{item}' (expected key=val1,val2)")
            continue
        key, vals = item.split("=", 1)
        result[key] = [v.strip() for v in vals.split(",")]
    return result


def main():
    parser = argparse.ArgumentParser(
        description="SAFEFL phase-aware experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--phase", type=int, choices=range(8), default=None,
                        help="Run experiments for this phase (0-7)")
    parser.add_argument("--decide", type=int, choices=range(8), default=None,
                        help="Record decisions for this phase (0-7)")
    parser.add_argument("--status", action="store_true",
                        help="Show status of all phases")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID (default: 0)")
    parser.add_argument("--dry", action="store_true",
                        help="Dry run -- print commands but do not execute")
    parser.add_argument("--filter", type=str, default=None,
                        help="Substring filter on experiment descriptions")
    parser.add_argument("--override", nargs="*", default=None,
                        help="Override phase values, e.g. --override attacks=no defences=fedavg,krum")
    parser.add_argument("--cloud_id", type=str, choices=["emran", "moaz"], default=None,
                        help="Split experiments across two clouds. "
                             "emran runs even-indexed experiments, moaz runs odd-indexed.")

    args = parser.parse_args()

    # Exactly one action required
    actions = sum([args.phase is not None, args.decide is not None, args.status])
    if actions == 0:
        parser.print_help()
        sys.exit(1)
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
    ok, missing = check_dependencies(cfg, phase_id)
    if not ok:
        print(f"Error: Phase {phase_id} is blocked.  Decide phase(s) {missing} first.")
        sys.exit(1)

    overrides = parse_overrides(args.override)
    experiments = generate_phase(phase_id, cfg, args.gpu, overrides)

    if args.filter:
        experiments = [(c, d) for c, d in experiments if args.filter in d]

    # Split across two cloud environments
    if args.cloud_id:
        total_before = len(experiments)
        if args.cloud_id == "emran":
            experiments = experiments[::2]
        else:
            experiments = experiments[1::2]
        print(f"  Cloud split: {args.cloud_id} gets {len(experiments)}/{total_before} experiments")

    if not experiments:
        print("No experiments match the current configuration / filter.")
        sys.exit(0)

    failed = run_experiments(experiments, dry=args.dry)

    if not args.dry:
        phase = get_phase(cfg, phase_id)
        if phase.get("status") != "decided":
            phase["status"] = "completed"
            save_config(cfg)
            print(f"Phase {phase_id} status updated to 'completed'.")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
