# Experiment Runner Design Spec

## Goal

Replace the brute-force `run_benchmarks.py` (generates the full 6,750-config cross-product) with a phase-aware `experiment_runner.py` that follows the 8-phase experiment plan. Each phase generates only the experiments it needs, reads decisions from earlier phases via a shared config file, and saves results into the unified `experiment_results/` directory.

## Architecture

One script (`experiment_runner.py`) + one config file (`phase_config.json`). The script does three things:

1. **Generate** — produce the list of `main.py` CLI commands for a given phase
2. **Execute** — run them sequentially, tracking success/failure
3. **Decide** — interactively prompt the user for decisions after a phase completes

Phase definitions are data (dicts), not code. Adding a new phase or modifying a sweep grid is a config-level change.

## CLI Interface

```bash
# Run a phase
python experiment_runner.py --phase 0 --gpu 0
python experiment_runner.py --phase 0 --gpu 0 --dry     # preview commands

# Record decisions after reviewing results
python experiment_runner.py --decide 0

# Check what's done and what's next
python experiment_runner.py --status

# Override sweep dimensions ad-hoc
python experiment_runner.py --phase 4 --override bias=0.5,0.75

# Filter within a phase (same as run_benchmarks)
python experiment_runner.py --phase 1 --filter krum
```

## Config File: `phase_config.json`

### Lifecycle

1. On first run, the script creates `phase_config.json` with global defaults and all 8 phases set to `pending`.
2. `--phase N` checks that dependencies are `decided`, then generates and runs experiments. Sets phase status to `completed`.
3. `--decide N` interactively prompts for the phase's decisions. Sets status to `decided`.
4. The user can also hand-edit the JSON directly — the script validates before each run.

### Schema

```json
{
  "global_defaults": {
    "nworkers": 200,
    "batch_size": 32,
    "niter": 2500,
    "lr": 0.1,
    "test_every": 10,
    "seed": 1,
    "nruns": 1
  },
  "dataset_defaults": {
    "FEMNIST": { "nworkers": 200, "niter": 2500 },
    "MNIST":   { "nworkers": 200, "niter": 1500 },
    "CIFAR10": { "nworkers": 200, "niter": 3000 }
  },
  "phases": {
    "0": {
      "status": "pending",
      "decisions": {}
    },
    "1": {
      "status": "pending",
      "decisions": {}
    }
  }
}
```

### Defaults resolution order

When building a command for a specific experiment, parameters are resolved in this order (later wins):

1. `global_defaults` — base values
2. `dataset_defaults[dataset]` — per-dataset overrides (e.g., MNIST needs fewer iterations)
3. Phase definition's `fixed` params — phase-level overrides (e.g., Phase 6 fixes attack to label_flipping)
4. `--override` CLI flag — ad-hoc narrowing for a single run

This means you can change `global_defaults.nworkers` to 300 and every phase picks it up, or set `dataset_defaults.CIFAR10.niter` to 5000 without touching any phase logic.

### Phase statuses

- `pending` — not yet run
- `completed` — experiments finished, awaiting user decisions
- `decided` — user has reviewed results and recorded decisions

### Decision fields per phase

Each phase's `decisions` object has specific required fields. The `--decide` command prompts for exactly these:

**Phase 0** — Baseline Landscape
```
decisions: {
  anchor_dataset: "FEMNIST",       # which dataset to use for all subsequent phases
  anchor_arch: "mobilenet_v3_small", # which architecture
  notes: "..."                      # optional free-text
}
```
Prompt: "Which dataset-architecture pair had the best testbed properties (meaningful accuracy, meaningful attack delta, practical convergence)?"

**Phase 1** — Grouping Failure
```
decisions: {
  surviving_baselines: ["signguard"],    # baselines that still work under grouping
  collapsed_baselines: ["krum", "shieldfl"],  # baselines to drop from future phases
  notes: "..."
}
```
Prompt: "Which baselines survived grouping (accuracy didn't collapse)?" — shows list of baselines, user picks which survived.

**Phase 2** — Inference Recovery
```
decisions: {
  factorgraphs_works: true,   # does FactorGraphs recover accuracy?
  notes: "..."
}
```
Prompt: "Did FactorGraphs recover plaintext-level accuracy under grouped SA?" — yes/no. If no, the script warns that Phases 3+ may not be meaningful.

**Phase 3** — Backdoor Defense
```
decisions: {
  factorgraphs_suppresses_backdoor: true,
  notes: "..."
}
```
Prompt: "Did FactorGraphs achieve high ACC + low ASR (bottom-right quadrant)?"

**Phase 4** — Non-IID Robustness
```
decisions: {
  crossover_beta: 0.75,         # beta where FactorGraphs' advantage is largest
  show_beta_1: true,            # whether to include extreme non-IID in paper
  notes: "..."
}
```
Prompt: "At which beta did FactorGraphs separate most from baselines? Include beta=1.0 in paper?"

**Phase 5** — Generalization
```
decisions: {
  second_dataset: "MNIST",       # or "CIFAR10"
  need_cifar: false,             # whether to add CIFAR10 runs
  notes: "..."
}
```
Prompt: "Did results transfer to the second dataset? Need CIFAR10 runs too?"

**Phase 6** — Sensitivity (k x nbyz)
```
decisions: {
  sweet_spot_k: 10,
  failure_boundary_notes: "k=30 with nbyz=30 is the edge case",
  notes: "..."
}
```
Prompt: "What's the sweet spot group size? Any failure boundaries?"

**Phase 7** — Architecture Comparison
```
decisions: {
  arch_agnostic: true,
  notes: "..."
}
```
Prompt: "Were results consistent across architectures?"

## Phase Definitions (Experiment Generation)

Each phase is defined as a dict specifying what to sweep and what's fixed. The generator reads decisions from earlier phases to fill in dynamic values.

### Phase 0 — Baseline Landscape (18 runs)
```
datasets:    [MNIST, FEMNIST, CIFAR10]
archs:       [mobilenet_v3_small, resnet18, eff_net]
attacks:     [no, label_flipping_attack]
defences:    [fedavg]
fixed:       { bias: 0, group_size: 10, nbyz: 20 }
grouped:     false
```
9 dataset-arch combos x 2 attack scenarios = 18 runs.

### Phase 1 — Grouping Failure (16 runs)
```
datasets:    [<anchor_dataset from Phase 0>]
archs:       [<anchor_arch from Phase 0>]
attacks:     [no, label_flipping_attack]
defences:    [fedavg, krum, shieldfl, signguard]
modes:       [grouped, non-grouped]
fixed:       { bias: 0, nbyz: 20, group_size: 10 }
```
4 defences x 2 modes x 2 attacks = 16 runs.

### Phase 2 — Inference Recovery (2 runs)
```
datasets:    [<anchor_dataset>]
archs:       [<anchor_arch>]
attacks:     [no, label_flipping_attack]
defences:    [factorGraphs]
fixed:       { bias: 0, nbyz: 20, group_size: 10 }
grouped:     true
```
1 defence x 2 attacks = 2 runs.

### Phase 3 — Backdoor Defense (4 runs)
```
datasets:    [<anchor_dataset>]
archs:       [<anchor_arch>]
attacks:     [scaling_attack]
defences:    [factorGraphs, fedavg, krum, signguard]
fixed:       { bias: 0, nbyz: 20, group_size: 10 }
grouped:     true
```
4 defences x 1 attack = 4 runs.

### Phase 4 — Non-IID Robustness (24 runs)
```
datasets:    [<anchor_dataset>]
archs:       [<anchor_arch>]
biases:      [0.25, 0.5, 0.75, 1.0]
attacks:     [label_flipping_attack, scaling_attack]
defences:    [factorGraphs, fedavg, <surviving_baselines from Phase 1>]
fixed:       { nbyz: 20, group_size: 10 }
grouped:     true (for factorGraphs and surviving baselines), false (for fedavg)
```
3 defences x 2 attacks x 4 betas = 24 runs.

### Phase 5 — Generalization (6 runs)
```
datasets:    [MNIST]   # or CIFAR10 if MNIST is trivial
archs:       [<anchor_arch>]
biases:      [0, 0.5]
attacks:     [label_flipping_attack, scaling_attack]
defences:    [factorGraphs, fedavg, <best surviving baseline>]
fixed:       { nbyz: 20, group_size: 10 }
grouped:     true (for factorGraphs + baseline), false (for fedavg)
```
3 defences x 2 attacks = 6 runs (but only at the 2 selected beta values, so actually up to 12 if both betas are swept — the plan says 6, we'll generate the minimal set).

### Phase 6 — Sensitivity Grid (12 runs)
```
datasets:    [<anchor_dataset>]
archs:       [<anchor_arch>]
attacks:     [label_flipping_attack]
defences:    [factorGraphs]
group_sizes: [5, 10, 20, 30]
nbyz_values: [10, 20, 30]
fixed:       { bias: 0 }
grouped:     true
```
4 group sizes x 3 nbyz = 12 runs.

### Phase 7 — Architecture Comparison (4 runs)
```
datasets:    [<anchor_dataset>]
archs:       [eff_net, resnet18]  # everything except the anchor arch
attacks:     [label_flipping_attack]
defences:    [factorGraphs, fedavg]
fixed:       { bias: 0, nbyz: 20, group_size: 10 }
grouped:     true (for factorGraphs), false (for fedavg)
```
2 defences x 2 archs = 4 runs.

## Dependency Enforcement

Before generating experiments for Phase N, the script checks:

| Phase | Requires |
|-------|----------|
| 0 | Nothing |
| 1 | Phase 0 decided |
| 2 | Phase 1 decided |
| 3 | Phase 1 decided |
| 4 | Phase 2 decided + Phase 3 decided |
| 5 | Phase 4 decided |
| 6 | Phase 2 decided |
| 7 | Phase 2 decided |

If a dependency isn't met, the script prints what's missing and exits.

## Execution

Same runner loop as `run_benchmarks.py`:
- Sequential subprocess calls to `python main.py ...`
- Timing per experiment
- Failed experiment tracking
- Summary at end

Results go into `experiment_results/` via the `experiment_logger.py` we already built. The runner doesn't need to know about result file locations — `main.py` handles that.

## `--status` Output

```
Phase 0: Baseline Landscape          [DECIDED]
  anchor: FEMNIST + mobilenet_v3_small
Phase 1: Grouping Failure            [DECIDED]
  surviving: signguard | collapsed: krum, shieldfl
Phase 2: Inference Recovery          [COMPLETED - needs --decide 2]
Phase 3: Backdoor Defense            [PENDING]
  blocked by: Phase 1
Phase 4: Non-IID Robustness          [PENDING]
  blocked by: Phase 2, Phase 3
Phase 5: Generalization              [PENDING]
  blocked by: Phase 4
Phase 6: Sensitivity Grid            [PENDING]
  blocked by: Phase 2
Phase 7: Architecture Comparison     [PENDING]
  blocked by: Phase 2
```

## `--override` Behavior

The `--override` flag lets you narrow any sweep dimension for a single invocation without editing config:

```bash
# Only run beta=0.5 and 0.75 in Phase 4 (instead of all 4)
python experiment_runner.py --phase 4 --override bias=0.5,0.75

# Only run group_size=10 in Phase 6
python experiment_runner.py --phase 6 --override group_size=10
```

Override keys must match parameter names used in the phase definitions. Values are comma-separated. The override applies only to that invocation — it doesn't modify the config file.

## What This Doesn't Do

- **Auto-analyze results** — you review results manually and record decisions via `--decide`
- **Parallelize** — experiments run sequentially (same as `run_benchmarks.py`). For parallelism, use `cloud_sweep.py`
- **Replace `main.py`** — this is purely an orchestration layer that builds CLI commands
