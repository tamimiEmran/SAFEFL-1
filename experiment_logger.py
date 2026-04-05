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

    Directory name format: {dataset}_{aggregation}_{attack}_f{nbyz}_b{bias}_{hash}
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
    Useful for all aggregation methods -- explains non-IID effects.
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
    excluded_ids = {cid for cid, b in latent_variables.items() if float(b) >= 0.95}
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
        if float(belief) >= 0.95 and client_id not in latency_tracker:
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
        member_weights = {uid: 1.0 - float(latent_variables.get(uid, 0.5)) for uid in indices}
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
    normal_pairs = [(idx, float(s)) for idx, s in latent_variables.items() if idx >= f]
    mal_pairs = [(idx, float(s)) for idx, s in latent_variables.items() if idx < f]

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
