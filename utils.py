import torch
import os
import pandas as pd
import numpy as np
import torch
# --- NEW helper: appends safely, writes header only once ----------
def _safe_append(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.isfile(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as fh:        # newline handled here
        df.to_csv(fh, header=write_header, index=False)


def MMD(x, y, device):
    """Empirical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.
       Taken and adapted from https://www.kaggle.com/code/onurtunali/maximum-mean-discrepancy/notebook by
       Onur Tunali

       We use the Gaussian kernel
    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        device: device where computation is performed
    """

    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    # RBF or Gaussian kernel
    bandwidth_range = [10, 15, 20, 50]
    for a in bandwidth_range:
        XX += torch.exp(-0.5 * dxx / a)
        YY += torch.exp(-0.5 * dyy / a)
        XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)


def save_data_to_csv(heirichal_params, f):
    """
    Save hierarchical data to CSV files for analysis and visualization.
    Captures which users are filtered out directly from the algorithm's logic.
    
    Args:
        heirichal_params: Dictionary containing hierarchical parameters
        f: Number of malicious users (the first f users are considered malicious)
    """

    # Create results directory
    os.makedirs("results/hierarchical", exist_ok=True)
    
    # Extract data from current round's record
    latest_record = heirichal_params["history"][-1]
    round_num = latest_record["round_num"]
    experiment_id = heirichal_params["experiment_id"]

    # Add ground truth malicious flag to heirichal_params if not already present
    if "GT malicious" not in heirichal_params:
        heirichal_params["GT malicious"] = [True if i < f else False for i in range(len(latest_record["user_scores"]))]
    
    # Save filtered users information if it exists in the latest record
    # This should be populated by the filter_malicious_users function in heirichalFL.py
    if "filtered_users" in latest_record:
        filtered_users = latest_record["filtered_users"]
    else:
        # Default to empty list if not available
        filtered_users = []
    
    # Create or update user membership CSV
    user_membership_data = []
    for user_id, group_id in enumerate(latest_record["user_membership"]):
        is_malicious = user_id < f
        is_filtered = user_id in filtered_users
        
        user_membership_data.append({
            "round": round_num,
            "user_id": user_id,
            "group_id": group_id,
            "is_actually_malicious": 1 if is_malicious else 0,
            "is_filtered_out": 1 if is_filtered else 0,
            "experiment_id": experiment_id,
            "attack_type": heirichal_params["attack type"]


        })
    
    user_membership_df = pd.DataFrame(user_membership_data)
    membership_file = f"results/hierarchical/user_membership.csv"

    _safe_append(user_membership_df, membership_file)
    

        
    # Create or update user scores CSV with malicious flag
    user_scores_data = []
    for user_id, score in enumerate(latest_record["user_scores"]):
        is_malicious = user_id < f
        is_filtered = user_id in filtered_users
        adjustment = latest_record["user_score_adjustment"][user_id] if user_id < len(latest_record["user_score_adjustment"]) else 0
        
        user_scores_data.append({
            "round": round_num,
            "user_id": user_id,
            "score": score,
            "adjustment": adjustment,
            "is_actually_malicious": 1 if is_malicious else 0,
            "is_filtered_out": 1 if is_filtered else 0,
            "experiment_id": experiment_id,
            "attack_type": heirichal_params["attack type"]
        })
    
    user_scores_df = pd.DataFrame(user_scores_data)
    scores_file = f"results/hierarchical/user_scores.csv"
    
    _safe_append(user_scores_df, scores_file)
    
    # Create or update group scores CSV with malicious user count
    group_scores_data = []
    
    # Count actual malicious and filtered users per group
    actual_malicious_count_per_group = {}
    filtered_count_per_group = {}
    total_users_per_group = {}
    
    for user_id, group_id in enumerate(latest_record["user_membership"]):
        total_users_per_group[group_id] = total_users_per_group.get(group_id, 0) + 1
        
        if user_id < f:  # Actual malicious
            actual_malicious_count_per_group[group_id] = actual_malicious_count_per_group.get(group_id, 0) + 1
            
        if user_id in filtered_users:  # Filtered out
            filtered_count_per_group[group_id] = filtered_count_per_group.get(group_id, 0) + 1
    
    for group_id, score in latest_record["group_scores"].items():
        actual_mal_count = actual_malicious_count_per_group.get(group_id, 0)
        filtered_count = filtered_count_per_group.get(group_id, 0)
        total_count = total_users_per_group.get(group_id, 0)
        
        group_scores_data.append({
            "round": round_num,
            "group_id": group_id,
            "score": score,
            "actual_malicious_count": actual_mal_count,
            "filtered_count": filtered_count,
            "total_users": total_count,
            "actual_malicious_ratio": actual_mal_count / total_count if total_count > 0 else 0,
            "filtered_ratio": filtered_count / total_count if total_count > 0 else 0,
            "experiment_id": experiment_id,
            "attack_type": heirichal_params["attack type"]
        })
    
    group_scores_df = pd.DataFrame(group_scores_data)
    group_scores_file = f"results/hierarchical/group_scores.csv"
    
    _safe_append(group_scores_df, group_scores_file)
    
    # Save global gradient norms
    if "global_gradient" in latest_record and latest_record["global_gradient"] is not None:
        global_gradient = latest_record["global_gradient"]
        
        # Calculate and save gradient norm
        gradient_norm = torch.norm(global_gradient, p=2).item()
        
        gradient_data = {
            "round": round_num,
            "gradient_norm": gradient_norm,
            "filtered_users_count": len(filtered_users),
            "experiment_id": experiment_id,
            "attack_type": heirichal_params["attack type"]
        }
        
        # Add first few components of the gradient vector (for visualization)
        max_components = min(10, global_gradient.numel())
        for i in range(max_components):
            gradient_data[f"component_{i}"] = global_gradient.flatten()[i].item()
            
        gradient_df = pd.DataFrame([gradient_data])
        gradient_file = f"results/hierarchical/global_gradients.csv"
        
        _safe_append(gradient_df, gradient_file)
    
    # Create or update summary statistics CSV with advanced metrics
    summary_data = {
        "round": round_num,
        "num_groups": heirichal_params["num groups"],
        "assumed_mal_prct": heirichal_params["assumed_mal_prct"],
        "filtered_users_count": len(filtered_users),
        "filtered_actually_malicious": sum(1 for uid in filtered_users if uid < f),
        "experiment_id": experiment_id,
        "attack_type": heirichal_params["attack type"]
    }
    
    if len(filtered_users) > 0:
        summary_data["precision"] = summary_data["filtered_actually_malicious"] / len(filtered_users)
    else:
        summary_data["precision"] = 0
        
    if f > 0:
        summary_data["recall"] = summary_data["filtered_actually_malicious"] / f
    else:
        summary_data["recall"] = 0
    
    # Add group statistics
    group_scores = latest_record["group_scores"]
    if group_scores:
        summary_data["max_group_score"] = max(group_scores.values())
        summary_data["min_group_score"] = min(group_scores.values())
        summary_data["avg_group_score"] = sum(group_scores.values()) / len(group_scores)
        summary_data["std_group_score"] = np.std(list(group_scores.values()))
        
    # Add user score statistics
    user_scores = latest_record["user_scores"]
    if user_scores:
        summary_data["max_user_score"] = max(user_scores)
        summary_data["min_user_score"] = min(user_scores)
        summary_data["avg_user_score"] = sum(user_scores) / len(user_scores)
        summary_data["std_user_score"] = np.std(user_scores)
        
        # Calculate metrics for malicious vs. non-malicious users
        if f > 0:
            malicious_scores = [user_scores[i] for i in range(min(f, len(user_scores)))]
            benign_scores = [user_scores[i] for i in range(f, len(user_scores))]
            
            if malicious_scores:
                summary_data["avg_malicious_score"] = sum(malicious_scores) / len(malicious_scores)
                summary_data["std_malicious_score"] = np.std(malicious_scores)
                
            if benign_scores:
                summary_data["avg_benign_score"] = sum(benign_scores) / len(benign_scores)
                summary_data["std_benign_score"] = np.std(benign_scores)
                
            # Calculate score gap between benign and malicious users
            if malicious_scores and benign_scores:
                summary_data["benign_malicious_score_gap"] = summary_data["avg_benign_score"] - summary_data["avg_malicious_score"]
    
    # Add filtering effectiveness metrics
    if f > 0 and len(filtered_users) > 0:
        summary_data["f1_score"] = 2 * (summary_data["precision"] * summary_data["recall"]) / (summary_data["precision"] + summary_data["recall"]) if (summary_data["precision"] + summary_data["recall"]) > 0 else 0

    summary_df = pd.DataFrame([summary_data])
    summary_file = f"results/hierarchical/summary_stats.csv"
    
    _safe_append(summary_df, summary_file)
    
