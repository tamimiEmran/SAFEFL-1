# Unified Experiment Results Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fragmented results saving system (`results3rdOct/`, `finalResults/`, `results/hierarchical/`, `score_function_viz/`) with a single experiment-namespaced directory structure that works for ALL aggregation methods, with factorGraphs-specific extensions (#1-7) layered on top.

**Architecture:** A new `experiment_logger.py` module handles all result I/O. Every experiment gets its own directory under `experiment_results/` with a human-readable prefix + 4-char hash. Universal files (config, accuracy, per-round metrics, data distribution) are saved for all methods. FactorGraphs-specific files (beliefs, detection metrics, latency, weights, groups, timing, observation scores) are saved only when `aggregation == factorGraphs`. A root `index.csv` maps every experiment for cross-experiment queries. The old `save_results_to_csv()` in `main.py` is replaced entirely.

**Tech Stack:** Python stdlib (`hashlib`, `json`, `time`, `os`), `pandas`, `numpy` (all already in the project)

---

## File Structure

| File | Responsibility |
|------|---------------|
| **Create:** `experiment_logger.py` | All result I/O: directory creation, hashing, CSV/JSON writing, index management |
| **Modify:** `main.py:73-166` | Replace `save_results_to_csv()` with logger calls |
| **Modify:** `main.py:684-949` | Initialize experiment dir before run loop, save per-round and data distribution, finalize at end |
| **Modify:** `aggregation_rules.py:1302-1498` | Call logger from `_run_inference_and_update()` instead of inline CSV writes |
| **Modify:** `bayesian/factor_graph.py:63-95` | Remove old CSV init from `_maybe_init_bayesian_and_csv()` |

### What gets deleted/replaced

| Old location | Replaced by |
|-------------|-------------|
| `results3rdOct/accuracy_*.csv` | `experiment_results/{exp}/accuracy.csv` |
| `results3rdOct/backdoor_*.csv` | `experiment_results/{exp}/backdoor.csv` |
| `results3rdOct/config_*.txt` | `experiment_results/{exp}/config.json` |
| `finalResults/allResults.json` | `experiment_results/{exp}/config.json` + `accuracy.csv` (queryable via `index.csv`) |
| `results/hierarchical/experiment_results.csv` | `experiment_results/{exp}/round_results.csv` |
| `score_function_viz/observation_scores.csv` | `experiment_results/{exp}/observation_scores.csv` |

---

## Output Directory Structure

```
experiment_results/
  index.csv                                         # one row per experiment, all key params
  mnist_fedavg_labelflip_f3_b0.5_a1c2/
    config.json                                     # full args dump
    accuracy.csv                                    # runs x iterations matrix
    backdoor.csv                                    # (only if scaling_attack)
    round_results.csv                               # per-round: round, accuracy, backdoor_success
    data_distribution.json                          # per-client class counts
  mnist_factorGraphs_labelflip_f3_b0.5_b3d4/
    config.json
    accuracy.csv
    backdoor.csv
    round_results.csv
    data_distribution.json
    client_beliefs.csv                              # #1: client_id, round, belief_score, is_malicious, excluded
    detection_metrics.csv                           # #2: round, tp, fp, fn, tn, precision, recall, f1
    detection_latency.csv                           # #3: client_id, round_first_excluded, is_malicious, was_detected
    client_weights.csv                              # #4: round, client_id, raw/group/effective weight, group_id
    group_assignments.csv                           # #5: round, group_id, client_ids, group_size
    inference_timing.csv                            # #6: round, inference_ms, total_round_ms
    observation_scores.csv                          # existing group-level scores (moved here)
```

---

### Task 1: Create `experiment_logger.py` — Directory, Index, and Universal Saves

**Files:**
- Create: `experiment_logger.py`

- [ ] **Step 1: Create `experiment_logger.py` with directory setup and universal save functions**

```python
import os
import hashlib
import json
import time
import pandas as pd
import numpy as np

RESULTS_ROOT = "experiment_results"


def _short_hash(config_dict):
    """First 4 chars of SHA256 of the JSON-serialized config."""
    raw = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:4]


def init_experiment_dir(args):
    """
    Create the experiment directory and register it in index.csv.
    Works for ALL aggregation methods.
    Returns the path to the experiment directory.
    
    Directory name format: {dataset}_{aggregation}_{attack}_{nbyz}_{bias}_{hash}
    """
    args_dict = vars(args) if hasattr(args, '__dict__') else dict(args)
    
    hash_config = {
        **args_dict,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }
    
    h = _short_hash(hash_config)
    dataset = args_dict["dataset"]
    aggregation = args_dict["aggregation"]
    attack = args_dict["byz_type"]
    nbyz = args_dict["nbyz"]
    bias = args_dict["bias"]
    
    dir_name = f"{dataset}_{aggregation}_{attack}_f{nbyz}_b{bias}_{h}"
    exp_dir = os.path.join(RESULTS_ROOT, dir_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Write full config as JSON (replaces results3rdOct/config_*.txt and finalResults metadata)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(hash_config, f, indent=2, default=str)
    
    # Update index.csv
    index_path = os.path.join(RESULTS_ROOT, "index.csv")
    index_row = {
        "experiment_id": dir_name,
        "dataset": dataset,
        "aggregation": aggregation,
        "attack_type": attack,
        "n_byzantine": nbyz,
        "bias": bias,
        "n_workers": args_dict["nworkers"],
        "group_size": args_dict.get("group_size", 0),
        "niter": args_dict["niter"],
        "nruns": args_dict["nruns"],
        "lr": args_dict["lr"],
        "seed": args_dict["seed"],
        "timestamp": hash_config["timestamp"],
    }
    row_df = pd.DataFrame([index_row])
    if os.path.exists(index_path):
        row_df.to_csv(index_path, mode="a", header=False, index=False)
    else:
        row_df.to_csv(index_path, index=False)
    
    return exp_dir


def save_accuracy(exp_dir, runs_test_accuracy, test_iterations):
    """
    Save accuracy CSV. Replaces results3rdOct/accuracy_*.csv.
    Shape: runs x iterations.
    """
    runs_test_accuracy = np.array(runs_test_accuracy)
    test_iterations = np.array(test_iterations)
    
    if runs_test_accuracy.ndim == 1:
        runs_test_accuracy = runs_test_accuracy.reshape(1, -1)
    
    num_cols = runs_test_accuracy.shape[1]
    if len(test_iterations) == num_cols:
        column_names = [f"Iter_{i}" for i in test_iterations]
    else:
        column_names = [f"Iter_{i}" for i in range(num_cols)]
    
    df = pd.DataFrame(runs_test_accuracy, columns=column_names)
    df.to_csv(os.path.join(exp_dir, "accuracy.csv"), index=False)


def save_backdoor(exp_dir, runs_backdoor_success, test_iterations):
    """
    Save backdoor success rate CSV. Only called for scaling_attack.
    Replaces results3rdOct/backdoor_*.csv.
    """
    if isinstance(runs_backdoor_success, list):
        runs_backdoor_success = np.array(runs_backdoor_success)
    test_iterations = np.array(test_iterations)
    
    if runs_backdoor_success.ndim == 1:
        runs_backdoor_success = runs_backdoor_success.reshape(1, -1)
    
    num_cols = runs_backdoor_success.shape[1]
    if len(test_iterations) == num_cols:
        column_names = [f"Iter_{i}" for i in test_iterations]
    else:
        column_names = [f"Iter_{i}" for i in range(num_cols)]
    
    df = pd.DataFrame(runs_backdoor_success, columns=column_names)
    df.to_csv(os.path.join(exp_dir, "backdoor.csv"), index=False)


def save_round_result(exp_dir, round_num, accuracy, backdoor_success, run_id):
    """
    Save per-round accuracy and backdoor success. Replaces results/hierarchical/experiment_results.csv.
    Appended every test interval.
    """
    record = {
        "run": run_id,
        "round": round_num,
        "accuracy": accuracy,
        "backdoor_success": backdoor_success,
    }
    df = pd.DataFrame([record])
    path = os.path.join(exp_dir, "round_results.csv")
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def save_data_distribution(exp_dir, each_worker_label, num_labels, f):
    """
    #7 Per-client data distribution summary. Called once at setup.
    Useful for all aggregation methods — explains non-IID effects.
    """
    distribution = {}
    for client_id, labels in enumerate(each_worker_label):
        if hasattr(labels, 'cpu'):
            labels_np = labels.cpu().numpy()
        else:
            labels_np = np.array(labels)
        counts = np.bincount(labels_np.astype(int), minlength=num_labels).tolist()
        distribution[str(client_id)] = {
            "is_malicious": client_id < f,
            "total_samples": int(len(labels_np)),
            "class_counts": counts,
            "dominant_class": int(np.argmax(counts)),
            "dominant_class_pct": round(float(max(counts) / len(labels_np)), 4) if len(labels_np) > 0 else 0.0,
        }
    
    with open(os.path.join(exp_dir, "data_distribution.json"), "w") as fp:
        json.dump(distribution, fp, indent=2)
```

- [ ] **Step 2: Verify file was created**

Run: `python -c "import experiment_logger; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add experiment_logger.py
git commit -m "feat: add experiment_logger with directory setup and universal result saves"
```

---

### Task 2: Add FactorGraphs-Specific Save Functions to `experiment_logger.py`

**Files:**
- Modify: `experiment_logger.py`

Append factorGraphs-specific functions. These are only called when `aggregation == factorGraphs`.

- [ ] **Step 1: Append factorGraphs-specific functions**

```python
# ---------------------------------------------------------------------------
# factorGraphs-specific saves (#1-6 + observation scores)
# ---------------------------------------------------------------------------

def save_client_beliefs(exp_dir, round_id, latent_variables, f):
    """
    #1 Per-client belief trajectory.
    Exclusion threshold: belief >= 0.95 means excluded (weight would be < 0.05).
    """
    records = []
    for client_id, belief in latent_variables.items():
        records.append({
            "client_id": client_id,
            "round": round_id,
            "belief_score": float(belief),
            "is_malicious": client_id < f,
            "excluded": float(belief) >= 0.95,
        })
    df = pd.DataFrame(records)
    path = os.path.join(exp_dir, "client_beliefs.csv")
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def save_detection_metrics(exp_dir, round_id, latent_variables, f):
    """
    #2 Detection metrics per round.
    Uses belief >= 0.95 as the exclusion threshold (same as weight < 0.05 -> zeroed out).
    """
    excluded_ids = {cid for cid, b in latent_variables.items() if b >= 0.95}
    malicious_ids = set(range(f))
    
    tp = len(excluded_ids & malicious_ids)
    fp = len(excluded_ids - malicious_ids)
    fn = len(malicious_ids - excluded_ids)
    tn = len(set(latent_variables.keys()) - malicious_ids - excluded_ids)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    record = {
        "round": round_id,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_excluded": len(excluded_ids),
        "num_malicious_included": fn,
    }
    df = pd.DataFrame([record])
    path = os.path.join(exp_dir, "detection_metrics.csv")
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def save_detection_latency(exp_dir, round_id, latent_variables, f, latency_tracker):
    """
    #3 Detection latency.
    Updates latency_tracker in-place: {client_id: round_first_excluded}.
    Only records the FIRST round a client crosses the threshold.
    Returns updated latency_tracker.
    """
    for client_id, belief in latent_variables.items():
        if belief >= 0.95 and client_id not in latency_tracker:
            latency_tracker[client_id] = round_id
    return latency_tracker


def finalize_detection_latency(exp_dir, latency_tracker, f, total_clients):
    """
    Called once at end of experiment. Writes the full latency table.
    """
    records = []
    for client_id in range(total_clients):
        records.append({
            "client_id": client_id,
            "round_first_excluded": latency_tracker.get(client_id, -1),
            "is_malicious": client_id < f,
            "was_detected": client_id in latency_tracker,
        })
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(exp_dir, "detection_latency.csv"), index=False)


def save_client_weights(exp_dir, round_id, groups, latent_variables):
    """
    #4 Per-client weights per round.
    Effective weight = 1 - belief. Group weight = min of member weights.
    If group weight < 0.95, all members get effective 0.
    """
    records = []
    for g_id, (_, indices) in groups.items():
        member_weights = {uid: float(1 - latent_variables.get(uid, 0.5)) for uid in indices}
        group_weight = min(member_weights.values())
        effective = 0.0 if group_weight < 0.95 else group_weight
        
        for uid in indices:
            records.append({
                "round": round_id,
                "client_id": uid,
                "raw_weight": member_weights[uid],
                "group_weight": group_weight,
                "effective_weight": effective,
                "group_id": g_id,
            })
    df = pd.DataFrame(records)
    path = os.path.join(exp_dir, "client_weights.csv")
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def save_group_assignments(exp_dir, round_id, groups):
    """
    #5 Group assignments per round.
    One row per group: round, group_id, comma-separated client IDs.
    """
    records = []
    for g_id, (_, indices) in groups.items():
        records.append({
            "round": round_id,
            "group_id": g_id,
            "client_ids": ",".join(str(i) for i in indices),
            "group_size": len(indices),
        })
    df = pd.DataFrame(records)
    path = os.path.join(exp_dir, "group_assignments.csv")
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def save_inference_timing(exp_dir, round_id, inference_ms, total_round_ms):
    """
    #6 Inference wall-clock time per round.
    """
    record = {
        "round": round_id,
        "inference_ms": inference_ms,
        "total_round_ms": total_round_ms,
    }
    df = pd.DataFrame([record])
    path = os.path.join(exp_dir, "inference_timing.csv")
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def save_observation_scores(exp_dir, round_id, observed_scores, groups, latent_variables, f, meta):
    """
    Existing observation_scores data, now saved per-experiment instead of global overwrite.
    """
    records = []
    normal_pairs = [(idx, s) for idx, s in latent_variables.items() if idx >= f]
    mal_pairs = [(idx, s) for idx, s in latent_variables.items() if idx < f]
    
    for gid, score in observed_scores.items():
        indices = groups[gid][1]
        numberOfMal = sum(1 for idx in indices if idx < f)
        record = {
            "round_id": round_id,
            "group_id": gid,
            "numberOfMal": numberOfMal,
            "score": float(score),
            "avgMalScore": np.mean([s for _, s in mal_pairs]) if mal_pairs else np.nan,
            "avgNormScore": np.mean([s for _, s in normal_pairs]) if normal_pairs else np.nan,
            "minMalScore": np.min([s for _, s in mal_pairs]) if mal_pairs else np.nan,
            "maxNormScore": np.max([s for _, s in normal_pairs]) if normal_pairs else np.nan,
            "idxOfMaxNormScore": max(normal_pairs, key=lambda x: x[1])[0] if normal_pairs else None,
            "idxOfMinMalScore": min(mal_pairs, key=lambda x: x[1])[0] if mal_pairs else None,
        }
        record.update({
            "dataset": meta["dataset"],
            "attack_type": meta["attack_type"],
            "n_byzantine": meta["n_byzantine"],
            "bias": meta["bias"],
            "n_workers": meta["n_workers"],
        })
        records.append(record)
    
    df = pd.DataFrame(records)
    path = os.path.join(exp_dir, "observation_scores.csv")
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
```

- [ ] **Step 2: Verify all functions importable**

Run: `python -c "from experiment_logger import init_experiment_dir, save_accuracy, save_backdoor, save_round_result, save_data_distribution, save_client_beliefs, save_detection_metrics, save_detection_latency, finalize_detection_latency, save_client_weights, save_group_assignments, save_inference_timing, save_observation_scores; print('All OK')"`
Expected: `All OK`

- [ ] **Step 3: Commit**

```bash
git add experiment_logger.py
git commit -m "feat: add factorGraphs-specific save functions to experiment_logger"
```

---

### Task 3: Remove Old CSV Init from `bayesian/factor_graph.py`

**Files:**
- Modify: `bayesian/factor_graph.py:63-95`

The old `_maybe_init_bayesian_and_csv()` creates a global `observation_scores.csv` that gets overwritten per experiment. Remove the CSV init logic and the pandas import; keep only the Bayesian state initialization.

- [ ] **Step 1: Edit `_maybe_init_bayesian_and_csv()` to remove CSV creation**

Remove the `import pandas as pd` on line 63 and replace the function (lines 64-95) with:

```python
def _maybe_init_bayesian_and_csv(bayesian_params, num_nodes):
    round_id = bayesian_params.get("current_round", 0)
    if round_id == 0:
        INITIAL_THRESHOLD = bayesian_params.get("initial_threshold", 0.5)
        bayesian_params["latent_variables"] = {id: INITIAL_THRESHOLD for id in range(num_nodes)}
        
        variables = vgroup.NDVarArray(num_states=2, shape=(num_nodes,))
        graph = fgraph.FactorGraph(
            variable_groups=[variables]
        )
        
        bayesian_params["graph"] = graph
        bayesian_params["variables"] = variables

    graph = bayesian_params["graph"]
    variables = bayesian_params["variables"]
    return round_id, graph, variables
```

This removes lines 79-91 (the `pd.DataFrame` header write, meta extraction, DIR construction).

- [ ] **Step 2: Verify import still works**

Run: `python -c "from bayesian.factor_graph import _maybe_init_bayesian_and_csv; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add bayesian/factor_graph.py
git commit -m "refactor: remove old global CSV init from _maybe_init_bayesian_and_csv"
```

---

### Task 4: Wire Logger into `_run_inference_and_update()` in `aggregation_rules.py`

**Files:**
- Modify: `aggregation_rules.py:1302-1498`

Replace the inline CSV writing block (lines 1453-1497) with calls to the logger. Add timing instrumentation around the BP inference section. Thread `latency_tracker` through `bayesian_params`.

- [ ] **Step 1: Add imports near the factorGraphs section (after line 1308)**

After `import random` on line 1308, add:

```python
import experiment_logger as explog
import time as _time
```

- [ ] **Step 2: Replace the body of `_run_inference_and_update()` (lines 1328-1498)**

Full replacement:

```python
def _run_inference_and_update(bayesian_params, graph, variables, groups, group_gradients,
                              round_id, f, param_list, net, lr):
    
    round_start = _time.perf_counter()
    
    if "factor_store" not in bayesian_params:
        bayesian_params["factor_store"] = {}
        bayesian_params['skippedFactorsCount'] = 0
    if "latency_tracker" not in bayesian_params:
        bayesian_params["latency_tracker"] = {}

    # observed group scores
    observed_scores = observation_function(group_gradients, bayesian_params)

    # build factors and run BP
    for group_id, indices in groups.values():
        group_key_for_factorGraphs = frozenset(indices)

        group_variables = [variables[idx] for idx in indices]
        factor_configs = np.array(list(product([0,1], repeat=len(indices))))

        likelihoods = []
        for config in factor_configs:
            obs = observed_scores[group_id]
            exp = 1 if any(config) else 0

            if obs == 1:
                p = bayesian_params.get("true_positive_rate", 0.8) if exp == 1 else (1 - bayesian_params.get("true_negative_rate", 0.8))
            else:
                p = (1 - bayesian_params.get("true_positive_rate", 0.8)) if exp == 1 else bayesian_params.get("true_negative_rate", 0.8)
            likelihoods.append(np.log(p + 1e-10))

        likelihoods = np.array(likelihoods)
        if group_key_for_factorGraphs not in bayesian_params["factor_store"]:
            factor = EnumFactor(
                factor_configs=factor_configs,
                variables=group_variables,
                log_potentials=likelihoods
            )
            graph.add_factors(factor)
            bayesian_params["factor_store"][group_key_for_factorGraphs] = factor
        else:
            factor = bayesian_params["factor_store"][group_key_for_factorGraphs]
            bayesian_params['skippedFactorsCount'] += 1
    
    # --- BP inference (timed) ---
    inference_start = _time.perf_counter()
    bp = BP(graph.bp_state, temperature=bayesian_params.get("factorGraphs_temperature", 0.1))
    bp_arrays = bp.init()
    bp_arrays = bp.run(bp_arrays, num_iters=bayesian_params.get("factorGraphs_num_iters", 100))
    beliefs = bp.get_beliefs(bp_arrays)
    marginals = get_marginals(beliefs)
    inference_ms = (_time.perf_counter() - inference_start) * 1000

    bayesian_params["current_round"] = round_id + 1
    bayesian_params["latent_variables"] = {i: marginals[variables][i, 1] for i in range(len(marginals[variables]))}

    # --- weighted aggregation ---
    global_update = torch.zeros_like(param_list[0])
    weights = []
    skippedGroups = 0
    for g_id, gradient in group_gradients.items():
        group_weights = []
        hasMalUser = False
        for u_id in groups[g_id][1]:
            latent_var = bayesian_params["latent_variables"].get(u_id)
            group_weights.append(float(1 - latent_var))
            if u_id < f:
                hasMalUser = True
            
        weight = min(group_weights)

        if weight < 0.95:
            weight = 0.0
            isIncludedToAverage = 1
            if not hasMalUser:
                skippedGroups += 1
        else:
            isIncludedToAverage = 1 / len(group_weights)
            if hasMalUser:
                print(f"Round {round_id}: Group {g_id} with malicious user accepted with weight {weight:.4f}")

        weights.append(weight)
        gradient *= weight
        gradient *= isIncludedToAverage
        global_update += gradient

    sum_weights = sum(weights)
    if sum_weights > 1e-5:
        global_update /= sum_weights

    if round_id > 0:
        idx = 0
        for j, (param) in enumerate(net.parameters()):
            param.add_(global_update[idx:(idx + torch.numel(param))].reshape(tuple(param.size())), alpha=-lr)
            idx += torch.numel(param)

    total_factors = len(bayesian_params["factor_store"])
    print(f"skipped factor percentage: {bayesian_params['skippedFactorsCount'] / total_factors}")
    print(f'Percentage of groups (out of {len(weights)}) that were skipped ({skippedGroups}): {skippedGroups / len(weights)}')

    # --- save factorGraphs-specific results ---
    exp_dir = bayesian_params.get("results_dir")
    if exp_dir:
        meta = bayesian_params["meta_data"]
        latent = bayesian_params["latent_variables"]
        
        explog.save_client_beliefs(exp_dir, round_id, latent, f)
        explog.save_detection_metrics(exp_dir, round_id, latent, f)
        bayesian_params["latency_tracker"] = explog.save_detection_latency(
            exp_dir, round_id, latent, f, bayesian_params["latency_tracker"]
        )
        explog.save_client_weights(exp_dir, round_id, groups, latent)
        explog.save_group_assignments(exp_dir, round_id, groups)
        
        total_round_ms = (_time.perf_counter() - round_start) * 1000
        explog.save_inference_timing(exp_dir, round_id, inference_ms, total_round_ms)
        
        explog.save_observation_scores(exp_dir, round_id, observed_scores, groups, latent, f, meta)

    return bayesian_params
```

- [ ] **Step 3: Verify syntax**

Run: `python -c "import aggregation_rules; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add aggregation_rules.py
git commit -m "feat: wire experiment_logger into _run_inference_and_update for factorGraphs metrics #1-6"
```

---

### Task 5: Replace `save_results_to_csv()` and Wire Universal Logging into `main.py`

**Files:**
- Modify: `main.py:1-58` (imports)
- Modify: `main.py:73-166` (delete `save_results_to_csv()`)
- Modify: `main.py:684-949` (run loop: init dir, save per-round, finalize)

This is the largest task. It replaces the old fragmented saving with the unified logger for ALL aggregation methods.

- [ ] **Step 1: Add import near top of `main.py` (after line 48)**

After the existing `import os` on line 48, add:

```python
import experiment_logger as explog
```

- [ ] **Step 2: Delete `save_results_to_csv()` function (lines 73-166)**

Delete the entire `save_results_to_csv` function body. It is fully replaced by `explog.save_accuracy()`, `explog.save_backdoor()`, and `config.json`.

Replace it with a minimal wrapper that calls the new logger:

```python
def save_results_to_csv(runs_test_accuracy, runs_backdoor_success, test_iterations, args, exp_dir):
    """Save final run results to the experiment directory."""
    explog.save_accuracy(exp_dir, runs_test_accuracy, test_iterations)
    if args.byz_type == "scaling_attack":
        explog.save_backdoor(exp_dir, runs_backdoor_success, test_iterations)
```

Note: the signature gains an `exp_dir` parameter.

- [ ] **Step 3: Initialize experiment directory before the run loop (after line 683, before `for run in range(1, args.nruns+1):`)**

Add before line 684:

```python
    # Initialize experiment results directory (all aggregation methods)
    exp_dir = explog.init_experiment_dir(args)
```

- [ ] **Step 4: For factorGraphs, store `results_dir` in `factorGraph_params` (after line 749)**

After the `factorGraph_params` dict is created (line 749), add:

```python
                factorGraph_params["results_dir"] = exp_dir
```

- [ ] **Step 5: Save data distribution after data poisoning (after line 764)**

After the data poisoning block, add for ALL aggregation methods on first run:

```python
        if run == 1:
            explog.save_data_distribution(exp_dir, each_worker_label, num_labels, args.nbyz)
```

- [ ] **Step 6: Replace the per-round hierarchical CSV write (lines 898-921)**

Replace the entire block from `record_round = {` through the `df.to_csv('results/hierarchical/experiment_results.csv', ...)` with:

```python
                    explog.save_round_result(exp_dir, e, test_accuracy,
                        test_success_rate if args.byz_type == "scaling_attack" else np.nan, run)
```

- [ ] **Step 7: Add detection latency finalization at end of run (after line 929, before `if args.mpspdz:`)**

```python
        if args.aggregation == "factorGraphs" and "latency_tracker" in factorGraph_params:
            explog.finalize_detection_latency(
                exp_dir, factorGraph_params["latency_tracker"], args.nbyz, args.nworkers
            )
```

- [ ] **Step 8: Update the `save_results_to_csv` call (line 949)**

Change:
```python
    save_results_to_csv(runs_test_accuracy, runs_backdoor_success, test_iterations, args)
```
To:
```python
    save_results_to_csv(runs_test_accuracy, runs_backdoor_success, test_iterations, args, exp_dir)
```

- [ ] **Step 9: Verify syntax**

Run: `python main.py --help`
Expected: No import errors, help text prints normally

- [ ] **Step 10: Commit**

```bash
git add main.py
git commit -m "feat: replace fragmented result saving with unified experiment_logger for all aggregation methods"
```

---

### Task 6: Clean Up Stale References

**Files:**
- Modify: `bayesian/factor_graph.py` — remove unused `import pandas as pd` if no other usage
- Modify: `aggregation_rules.py` — remove the old `import pandas as pd` near line 1539 and `from visualize_scoringfunction import round_full_scores` on line 1540 if unused

- [ ] **Step 1: Remove `import pandas as pd` from `bayesian/factor_graph.py` line 63**

Verify no other usage of `pd` in the file. The only usage was the CSV header write (removed in Task 3). Remove the import.

- [ ] **Step 2: Check and clean `aggregation_rules.py` line 1539-1540**

```python
import pandas as pd                                    # line 1539 - was used by old CSV write in _run_inference_and_update
from visualize_scoringfunction import round_full_scores # line 1540 - check if used anywhere
```

If these are no longer referenced after Task 4's replacement, remove them.

- [ ] **Step 3: Verify no references to old paths remain**

Run: `grep -rn "score_function_viz\|results3rdOct\|finalResults/allResults\|results/hierarchical/experiment_results" main.py aggregation_rules.py bayesian/factor_graph.py`
Expected: No matches

- [ ] **Step 4: Commit**

```bash
git add aggregation_rules.py bayesian/factor_graph.py
git commit -m "refactor: remove stale imports and old result path references"
```

---

### Task 7: End-to-End Smoke Test — All Aggregation Methods

**Files:**
- No new files

- [ ] **Step 1: Run a factorGraphs experiment**

```bash
python main.py --dataset mnist --nworkers 10 --nbyz 3 --byz_type label_flipping_attack --aggregation factorGraphs --niter 5 --test_every 1 --group_size 2 --nruns 1 --seed 42
```

- [ ] **Step 2: Run a fedavg experiment**

```bash
python main.py --dataset mnist --nworkers 10 --nbyz 3 --byz_type label_flipping_attack --aggregation fedavg --niter 5 --test_every 1 --nruns 1 --seed 42
```

- [ ] **Step 3: Run a krum experiment**

```bash
python main.py --dataset mnist --nworkers 10 --nbyz 3 --byz_type label_flipping_attack --aggregation krum --niter 5 --test_every 1 --nruns 1 --seed 42
```

- [ ] **Step 4: Verify directory structure**

```bash
python -c "
import os, pandas as pd, json, glob

root = 'experiment_results'
dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
print(f'Experiment directories: {dirs}')
assert len(dirs) == 3, f'Expected 3 directories, got {len(dirs)}'

# Check index.csv
idx = pd.read_csv(os.path.join(root, 'index.csv'))
print(f'index.csv: {len(idx)} experiments')
print(idx[['experiment_id', 'aggregation', 'attack_type']].to_string())
assert len(idx) == 3

# Check universal files exist for all experiments
for d in dirs:
    exp_path = os.path.join(root, d)
    assert os.path.exists(os.path.join(exp_path, 'config.json')), f'Missing config.json in {d}'
    assert os.path.exists(os.path.join(exp_path, 'accuracy.csv')), f'Missing accuracy.csv in {d}'
    assert os.path.exists(os.path.join(exp_path, 'round_results.csv')), f'Missing round_results.csv in {d}'
    assert os.path.exists(os.path.join(exp_path, 'data_distribution.json')), f'Missing data_distribution.json in {d}'
    print(f'{d}: universal files OK')

# Check factorGraphs-specific files
fg_dir = [d for d in dirs if 'factorGraphs' in d][0]
fg_path = os.path.join(root, fg_dir)
for f in ['client_beliefs.csv', 'detection_metrics.csv', 'detection_latency.csv',
          'client_weights.csv', 'group_assignments.csv', 'inference_timing.csv',
          'observation_scores.csv']:
    assert os.path.exists(os.path.join(fg_path, f)), f'Missing {f} in {fg_dir}'
print(f'{fg_dir}: factorGraphs-specific files OK')

# Check factorGraphs-specific files do NOT exist for fedavg/krum
for d in dirs:
    if 'factorGraphs' not in d:
        for f in ['client_beliefs.csv', 'detection_metrics.csv']:
            assert not os.path.exists(os.path.join(root, d, f)), f'Unexpected {f} in {d}'
        print(f'{d}: correctly has no factorGraphs files')

print('All checks passed')
"
```

- [ ] **Step 5: Verify CSV content for factorGraphs**

```bash
python -c "
import pandas as pd, os, glob

fg_dir = glob.glob('experiment_results/*factorGraphs*')[0]

beliefs = pd.read_csv(os.path.join(fg_dir, 'client_beliefs.csv'))
print(f'client_beliefs: {len(beliefs)} rows, cols: {list(beliefs.columns)}')

metrics = pd.read_csv(os.path.join(fg_dir, 'detection_metrics.csv'))
print(f'detection_metrics: {len(metrics)} rows')

latency = pd.read_csv(os.path.join(fg_dir, 'detection_latency.csv'))
print(f'detection_latency: {len(latency)} rows')

weights = pd.read_csv(os.path.join(fg_dir, 'client_weights.csv'))
print(f'client_weights: {len(weights)} rows')

groups = pd.read_csv(os.path.join(fg_dir, 'group_assignments.csv'))
print(f'group_assignments: {len(groups)} rows')

timing = pd.read_csv(os.path.join(fg_dir, 'inference_timing.csv'))
print(f'inference_timing: {len(timing)} rows')

rounds = pd.read_csv(os.path.join(fg_dir, 'round_results.csv'))
print(f'round_results: {len(rounds)} rows')

print('Content check passed')
"
```

- [ ] **Step 6: Verify a second factorGraphs run does NOT overwrite the first**

```bash
python main.py --dataset mnist --nworkers 10 --nbyz 5 --byz_type label_flipping_attack --aggregation factorGraphs --niter 3 --test_every 1 --group_size 2 --nruns 1 --seed 42
```

```bash
python -c "
import os, pandas as pd
idx = pd.read_csv('experiment_results/index.csv')
print(f'index.csv: {len(idx)} experiments')
assert len(idx) == 4, f'Expected 4, got {len(idx)}'
dirs = [d for d in os.listdir('experiment_results') if os.path.isdir(os.path.join('experiment_results', d))]
assert len(dirs) == 4, f'Expected 4 dirs, got {len(dirs)}'
print('No-overwrite check passed')
"
```

- [ ] **Step 7: Commit**

```bash
git commit --allow-empty -m "test: verify end-to-end unified results logging across fedavg, krum, factorGraphs"
```
