
import torch
import hdbscan
from collections import Counter
import torch
from sklearn.mixture import GaussianMixture
import torch
from typing import Optional

def softmax(x: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """
    Compute softmax of a tensor along a specified dimension.
    
    Args:
        x (torch.Tensor): Input tensor.
        dim (int, optional): Dimension along which to compute softmax.
            Defaults to the last dimension.
        
    Returns:
        torch.Tensor: Softmax of the input tensor.
    """
    if dim is None:
        dim = x.dim() - 1
    return torch.softmax(x, dim=dim)


def sigmoid(x, dim=None):
    """
    Compute sigmoid of a tensor along a specified dimension.
    Also handles the case when input is a single scalar value.
    
    Args:
        x (torch.Tensor): Input tensor or scalar.
        dim (int, optional): Dimension along which to compute sigmoid. If None, defaults to the last dimension.
        
    Returns:
        torch.Tensor: Sigmoid of the input tensor. Returns 1.0 if input is a scalar.
    """
    # Check if x is a scalar (0-dimensional tensor)
    if x.dim() == 0:
        return torch.ones_like(x)  # For scalar, sigmoid is always 1.0
    
    if dim is None:
        dim = -1
    return torch.sigmoid(x, dim=dim)

def signguard_anomaly_scores(gradients, device, seed,
                             L=0.1, R=3.0, selection_fraction=0.4, alpha=0):
    # [1] prepare flattened gradients (same as before)...
    param_list = [torch.cat([xx.reshape(-1) for xx in x]) for x in gradients]
    
    n = len(param_list)
    
    # [2] compute L2 norms and median
    l2_norm = torch.tensor([g.norm(p=2) for g in param_list], device=device)
    M = l2_norm.median()
    
    # [3] normalized norm‐deviation score
    norm_ratio = l2_norm / M
    score_norm = torch.clamp((norm_ratio - 1.0).abs() / R, 0.0, 1.0)
    
    # [4] pick a random subset of coordinates for sign‐features
    num_params = param_list[0].numel()
    num_sel = int(num_params * selection_fraction)
    idx = torch.randperm(num_params, device=device)[:num_sel]
    sign_grads = torch.stack([torch.sign(g[idx]) for g in param_list])
    
    # [5] compute sign fractions per client
    sign_pos  = (sign_grads ==  1).float().mean(dim=1)
    sign_zero = (sign_grads ==  0).float().mean(dim=1)
    sign_neg  = (sign_grads == -1).float().mean(dim=1)
    
    # [6] GMM clustering on CPU for probabilities
    feat = torch.stack([sign_pos, sign_zero, sign_neg], dim=1).cpu().numpy()
    gmm = GaussianMixture(n_components=2, random_state=seed).fit(feat)
    probs = gmm.predict_proba(feat)            # shape (n,2)
    labels = gmm.predict(feat)
    major = 0 if (labels==0).sum() > (labels==1).sum() else 1
    score_sign = 1.0 - torch.from_numpy(probs[:, major]).to(device)
    
    # [7] final anomaly score
    anomaly_scores = alpha * score_norm + (1 - alpha) * score_sign

    
    return anomaly_scores  # tensor of shape (n,)
import torch

def krum_anomaly_scores(gradients, f, device = torch.device('cpu')):
    """
    Compute per‐client anomaly scores based on the KRUM rule.

    gradients: list of per‐client gradient tensors (same shapes as model.parameters())
    f: number of assumed Byzantine (malicious) clients
    device: torch device

    Returns:
      anomaly_scores: torch.Tensor of shape (n_clients,)
        normalized KRUM scores in [0,1], where higher ⇒ more anomalous.
    """


    param_list = [torch.cat([g.reshape(-1) for g in client_grad]) for client_grad in gradients]
    n = len(param_list)
    stacked = torch.stack(param_list, dim=0).to(device)  # shape (n, D)

    # [2] pairwise squared distances matrix
    #    dist_matrix[i,j] = ||grad_i - grad_j||^2
    diffs = stacked.unsqueeze(1) - stacked.unsqueeze(0)      # (n, n, D)
    dist_sq = torch.sum(diffs * diffs, dim=2)               # (n, n)

    # [3] for each client i, compute Krum score = sum of the smallest (n - f - 2) distances to others
    nb = n - f - 2
    if nb < 1:
        raise ValueError(f"Need n - f - 2 >= 1; got n={n}, f={f}.")
    krum_scores = []
    for i in range(n):
        # exclude self‐distance by masking diag
        row = torch.cat([dist_sq[i, :i], dist_sq[i, i+1:]])  # length n-1
        smallest_dists, _ = torch.topk(row, k=nb, largest=False)
        krum_scores.append(smallest_dists.sum())
    krum_scores = torch.stack(krum_scores)  # (n,)

    # [4] normalize into [0,1] (higher = more anomalous)
    min_s, max_s = krum_scores.min(), krum_scores.max()
    if max_s > min_s:
        anomaly_scores = (krum_scores - min_s) / (max_s - min_s)
    else:
        anomaly_scores = torch.zeros_like(krum_scores)

    return anomaly_scores.to(device)

def create_groups_from_gradients(gradients, keys, grouping_strategy="average"):
    """
    Create grouped gradients by key.

    Args:
        gradients (list): List of gradients. Can be unflattened (List[List[Tensor]])
                          or flattened (List[Tensor]). This function handles both.
        keys (list[hashable]): List of group IDs, where keys[i] is the group ID 
                               for the client with gradient gradients[i].
        grouping_strategy (str): "average" or "aggregate".

    Returns:
        dict: Mapping group_id → aggregated gradient tensor.
    """
    if len(gradients) != len(keys):
        raise ValueError("`gradients` and `keys` must be the same length")

    # Create a dictionary to collect gradients by group ID
    grouped = {}
    # Renamed 'grad' to 'client_gradient' for clarity, as it can be a list or a tensor.
    for i, (key, client_gradient) in enumerate(zip(keys, gradients)):
        if key not in grouped:
            grouped[key] = []

        # --- THE FIX IS HERE ---
        # 1. Check if the gradient is a list of tensors (unflattened) or already a tensor.
        if isinstance(client_gradient, list):
            # 2. If it's a list, flatten it into a single tensor.
            grad_tensor = torch.cat([layer.reshape(-1) for layer in client_gradient])
        else:
            # If it's already a tensor, just use it.
            grad_tensor = client_gradient
        
        # Now `grad_tensor` is guaranteed to be a single tensor, so .detach() works.
        grouped[key].append(grad_tensor.detach().clone())
        # --- END OF FIX ---

    # Aggregate gradients within each group (this part is unchanged and now works correctly)
    result = {}
    for group_id, group_gradients in grouped.items():
        stacked = torch.stack(group_gradients, dim=0)
        if grouping_strategy == "average":
            result[group_id] = stacked.mean(dim=0)
        elif grouping_strategy == "aggregate":
            result[group_id] = stacked.sum(dim=0)
        else:
            raise ValueError("Unsupported strategy: " + grouping_strategy)

    return result


def generate_keys(n, f, group_size):
    """
    Returns one representative `keys` list per multiset of (red,blue)-patterns,
    treating groups as unlabeled and enforcing:
      - full groups of size=group_size,
      - at most one leftover of size L = n % group_size (if L>=2),
      - every group ≥2 members,
      - exactly f reds (first f indices) and n-f blues.

    Args:
        n (int): total number of gradients.
        f (int): number of red gradients (indices 0..f-1).
        group_size (int): desired size of each full group (>=2).

    Returns:
        List[List[int]]: each is a `keys` list mapping gradient→group-ID.
    """
    # 1) Compute how many full slots, and leftover size L
    full = n // group_size
    L    = n - full*group_size

    # 2) Validate minimum-size constraint
    if group_size < 2 or n < 2 or L == 1:
        # no valid grouping if leftover would be size 1
        return []

    # 3) Build pattern lists for each slot-size
    #    patterns_full: all (r,b) for a slot of size `group_size`
    #    patterns_left: all (r,b) for a slot of size `L` (if L>=2)
    def make_patterns(slot_size):
        pats = []
        for r in range(max(0, slot_size-(n-f)), min(slot_size, f)+1):
            b = slot_size - r
            # slot_size ≥2 guaranteed
            pats.append((r, b))
        return pats

    patterns_full = make_patterns(group_size)
    patterns_left = make_patterns(L) if L >= 2 else []

    results = []

    # 4) Enumerate which leftover pattern (if any) to use
    #    We treat "no leftover" as a special case when L==0
    leftover_choices = patterns_left or [None]

    for leftover in leftover_choices:
        # how many reds remain for full slots?
        if leftover is None:
            rem_reds = f
            rem_full = full
        else:
            rL, bL = leftover
            rem_reds = f - rL
            rem_full = full

        # quick prune
        if rem_reds < 0 or rem_reds > rem_full * group_size:
            continue

        # 5) Backtrack over counts c_j for patterns_full
        k = len(patterns_full)
        def bt_full(idx, slots_left, reds_left, count_vec):
            if idx == k:
                # base case: all full slots assigned?
                if slots_left == 0 and reds_left == 0:
                    # build one keys-list
                    red_q  = list(range(f))
                    blue_q = list(range(f, n))
                    keys   = [None] * n
                    gid    = 0

                    # first the leftover slot (if any)
                    if leftover is not None:
                        for _ in range(rL):
                            keys[red_q.pop(0)] = gid
                        for _ in range(bL):
                            keys[blue_q.pop(0)] = gid
                        gid += 1

                    # then each full-slot pattern in *canonical* order
                    for (r, b), c in zip(patterns_full, count_vec):
                        for _ in range(c):
                            for __ in range(r):
                                keys[red_q.pop(0)] = gid
                            for __ in range(b):
                                keys[blue_q.pop(0)] = gid
                            gid += 1

                    results.append(keys)
                return

            r_j, b_j = patterns_full[idx]
            # max how many slots of this pattern we can still fill
            max_c1 = min(slots_left, reds_left // r_j if r_j > 0 else slots_left)
            # also ensure we don't demand more blues than available:
            # blues_needed = sum_{previous} b + c_j * b_j + (slots_left - c_j)*0 <= n-f
            # but simpler to only prune reds; blues will automatically fit if reds fit
            for c_j in range(max_c1 + 1):
                bt_full(idx+1,
                        slots_left - c_j,
                        reds_left   - r_j*c_j,
                        count_vec   + [c_j])

        bt_full(0, rem_full, rem_reds, [])

    return results
import numpy as np


def _divide_and_conquer_bb_probs(gradients, device, f, niters, c, b, alpha, beta, generator):
    """(Internal Helper) Divide-and-conquer + Beta-Binomial maliciousness probabilities."""
    n = len(gradients)
    if n == 0: return {'prob_benign': []}
        
    param_list = [g.view(-1, 1) for g in gradients]
    survive_counts = np.zeros(n, dtype=int)
    D = param_list[0].shape[0]
    
    for _ in range(niters):
        d = np.random.randint(1, high=min(b, D) + 1)
        idx = torch.randperm(D, device=device, generator=generator)[:d]
        sel = [p[idx] for p in param_list]
        
        centered = (torch.cat(sel, dim=1) - torch.cat(sel, dim=1).mean(dim=1, keepdim=True)).T
        _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
        scores = (centered @ Vt[0]).cpu().numpy()**2
        
        k_good = max(0, n - int(f * c))
        good_idx = np.argsort(scores)[:k_good]
        
        survive_counts[good_idx] += 1
    
    denom = (alpha + beta + niters)
    theta_hat = (alpha + survive_counts) / denom
    p_benign = (theta_hat).tolist()
    
    return {'prob_benign': p_benign}

def _score_gradients_enhanced(gradients, device, cos_sim_matrix):
    """
    (Internal Helper) Scores gradients using a two-stage FLAME-inspired defense.
    
    FINAL FIX: Explicitly sets `min_samples` to be more sensitive to density,
               while keeping `min_cluster_size` as the final output filter.
    """
    n = len(gradients)
    if n < 3: 
        return {
            'direction_score': [1.0] * n, 
            'magnitude_score': [1.0] * n, 
            'combined_score': [1.0] * n
        }

    # --- Stage 1: Direction-based Scoring using Clustering ---
    
    cos_dist = 1.0 - cos_sim_matrix
    
    # Define the minimum cluster size based on the majority assumption
    min_cluster_size = (n // 2) + 1

    clusterer = hdbscan.HDBSCAN(
        metric='precomputed', 
        min_cluster_size=min_cluster_size,
        min_samples=2,
        allow_single_cluster=True,
        cluster_selection_method='leaf'
    ).fit(cos_dist.cpu().numpy())
    
    # Use the probabilistic scores for a more nuanced result.
    direction_scores = clusterer.probabilities_.tolist()

    # --- Stage 2: Magnitude-based Scoring ---
    
    euclid_norms = [torch.norm(g, p=2) for g in gradients]
    
    honest_norms = [norm for norm, score in zip(euclid_norms, direction_scores) if score > 0.0]
    
    if not honest_norms:
        median_norm = torch.median(torch.stack(euclid_norms)) if euclid_norms else torch.tensor(1.0)
    else:
        median_norm = torch.median(torch.stack(honest_norms))
        
    magnitude_scores = [min(1.0, (median_norm / norm).item()) if norm > 0 else 1.0 for norm in euclid_norms]

    combined_scores = [d_score * m_score for d_score, m_score in zip(direction_scores, magnitude_scores)]
    
    return {
        'direction_score': direction_scores, 
        'magnitude_score': magnitude_scores, 
        'combined_score': combined_scores
    }
def _score_gradients_simple(gradients, device, cos_sim_matrix):
    """(Internal Helper) Scores gradients by ensemble voting on cosine-similarity and distance."""
    n = len(gradients)
    if n < 3: return {'votes_sim': [1.0]*n, 'votes_dist': [0.0]*n, 'combined_score': [1.0]*n}

    votes_sim = cos_sim_matrix.sum(dim=1)

    dist = torch.cdist(torch.stack(gradients), torch.stack(gradients), p=2)
    maxd = dist.max()
    if maxd > 1e-12:
        dist = dist / maxd
    votes_dist = dist.sum(dim=1)

    assert len(votes_sim) == n and len(votes_dist) == n, "Length mismatch in votes arrays"

    # make both numpy to subtract them
    votes_sim = votes_sim.cpu().numpy()
    votes_dist = votes_dist.cpu().numpy()


    combined_score = votes_sim - votes_dist
    return {'votes_sim': votes_sim.tolist(), 'votes_dist': votes_dist.tolist(), 'combined_score': combined_score.tolist()}

import torch.nn.functional as F

def run_all_scoring_methods(
    flattened_gradients,  
    device=torch.device('cpu'),
    f=0,
    dnc_niters=100, dnc_c=1.5, dnc_b=100, dnc_alpha=1.0, dnc_beta=1.0,
    generator=None
):
    """
    Runs three different gradient scoring algorithms on a list of FLATTENED gradients.
    """
    
    n = len(flattened_gradients)
    if n == 0:
        return {}
        
    results = {}
    # The function now directly uses the flattened_gradients it receives.
    cos_sim_matrix = torch.zeros((n, n), dtype=torch.float64, device=device)
    for i in range(n):
        for j in range(i + 1, n):
            s = F.cosine_similarity(flattened_gradients[i].double(), flattened_gradients[j].double(), dim=0, eps=1e-9)
            cos_sim_matrix[i, j] = cos_sim_matrix[j, i] = s

    results['simple_ensemble'] = _score_gradients_simple(flattened_gradients, device, cos_sim_matrix.float())
    results['flame_enhanced'] = _score_gradients_enhanced(flattened_gradients, device, cos_sim_matrix)

    # 
    results['dnc_bb'] = _divide_and_conquer_bb_probs(
        flattened_gradients, device, f, dnc_niters, dnc_c, dnc_b, dnc_alpha, dnc_beta, generator
    )
    

    flatten_results_simple_ensemble = {
        f"simple_ensemble_{k}": torch.sigmoid(torch.tensor(v, dtype=torch.float32)).tolist()
        for k, v in results['simple_ensemble'].items()
    }

    flatten_results_flame_enhanced = {
        f"flame_enhanced_{k}": torch.sigmoid(torch.tensor(v, dtype=torch.float32)).tolist()
        for k, v in results['flame_enhanced'].items()
    }

    flatten_results_dnc_bb = {f"dnc_bb_{k}": v for k, v in results['dnc_bb'].items()}
    results = {**flatten_results_simple_ensemble, **flatten_results_flame_enhanced, **flatten_results_dnc_bb}
    
    return results



def generate_groups_and_their_scores(gradients, keys, grouping_strategy="average", f=0, device=torch.device('cpu'),
                                     dnc_niters=100, dnc_c=1.5, dnc_b=100, dnc_alpha=1.0, dnc_beta=1.0,
                                     generator=None):
    """
    Groups gradients and computes their scores.
    Returns a dictionary mapping group_id to a dictionary of its scores.
    """
    krum_signgaurd_scores = generate_groups_and_signguard_scores(gradients, keys= keys,  f = f, grouping_strategy= grouping_strategy)
    # output of krum_signguard_scores is a dictionary mapping group_id to a dictionary containing the SignGuard anomaly score
    grouped_gradients = create_groups_from_gradients(gradients, keys, grouping_strategy)
    
    # Sort keys to ensure consistent order
    sorted_group_ids = sorted(grouped_gradients.keys())
    flattened_grouped_gradients = [grouped_gradients[gid] for gid in sorted_group_ids]
    
    # Get all scores for the groups
    group_scores_all_methods = run_all_scoring_methods(
        flattened_grouped_gradients, device=device, f=f,
        dnc_niters=dnc_niters, dnc_c=dnc_c, dnc_b=dnc_b,
        dnc_alpha=dnc_alpha, dnc_beta=dnc_beta, generator=generator
    )

    
    # Reorganize the scores by group_id
    # The output will look like: {group_0: {'simple_..._score': 1.0, ...}, group_1: {...}}
    scores_by_group = {}
    for i, group_id in enumerate(sorted_group_ids):
        scores_by_group[group_id] = {
            method_name: scores_list[i]
            for method_name, scores_list in group_scores_all_methods.items()
        }
        # Add SignGuard scores to the group
        if group_id in krum_signgaurd_scores:
            scores_by_group[group_id].update(krum_signgaurd_scores[group_id])

    
        
    return scores_by_group

def generate_groups_and_signguard_scores(gradients, keys, f, device=torch.device('cpu'), 
                                        seed=42, L=0.1, R=2.0, 
                                        selection_fraction=0.4, alpha=0.1,
                                        grouping_strategy="average"):
    
    """
    Groups gradients and computes SignGuard anomaly scores for each group.
    
    Args:
        gradients: List of gradients (can be flattened or unflattened)
        keys: List of group IDs for each gradient
        device: Torch device to use
        seed: Random seed for SignGuard
        L, R, selection_fraction, alpha: SignGuard parameters
        grouping_strategy: "average" or "aggregate"
        
    Returns:
        Dictionary mapping group_id to a dictionary containing the SignGuard anomaly score
    """

    # Group the gradients
    grouped_gradients = create_groups_from_gradients(gradients, keys, grouping_strategy)
    
    # Sort keys to ensure consistent order
    sorted_group_ids = sorted(grouped_gradients.keys())
    flattened_grouped_gradients = [grouped_gradients[gid] for gid in sorted_group_ids]
    
    # SignGuard expects each gradient to be a list of tensors, but our gradients are already flattened.
    # We need to adapt by treating each flattened gradient as a single tensor in a list.
    adapted_gradients = [[grad] for grad in flattened_grouped_gradients]
    
    # Apply SignGuard to get anomaly scores
    anomaly_scores = signguard_anomaly_scores(
        adapted_gradients, device, seed,
        L=L, R=R, selection_fraction=selection_fraction, alpha=alpha
    )
    KRUM_ANOMALY_SCORES = krum_anomaly_scores(
        gradients= flattened_grouped_gradients,
        f=f
    )


    # Convert anomaly scores to a list
    anomaly_scores_list = anomaly_scores.tolist()
    KRUM_ANOMALY_SCORES = KRUM_ANOMALY_SCORES.tolist()
    
    # Map scores back to group IDs
    scores_by_group = {}
    for i, group_id in enumerate(sorted_group_ids):
        scores_by_group[group_id] = {'signguard_score': anomaly_scores_list[i], 
                                     'krum_score': KRUM_ANOMALY_SCORES[i]}
    
    return scores_by_group
def compute_signguard_and_krum_scores(gradients, f, device=torch.device('cpu'), 
                                     seed=42, L=0.1, R=2.0, 
                                     selection_fraction=0.4, alpha=0.1):
    
    """
    Computes SignGuard and Krum anomaly scores for each gradient.
    
    Args:
        gradients: List of gradients (can be flattened or unflattened)
        f: Number of assumed Byzantine clients for Krum
        device: Torch device to use
        seed: Random seed for SignGuard
        L, R, selection_fraction, alpha: SignGuard parameters
        
    Returns:
        Dictionary mapping gradient indices to a dictionary containing the SignGuard and Krum anomaly scores
    """
    
    # Ensure gradients are flattened
    flattened_gradients = []
    for grad in gradients:
        if isinstance(grad, list):
            # If it's a list, flatten it into a single tensor
            grad_tensor = torch.cat([layer.reshape(-1) for layer in grad])
        else:
            # If it's already a tensor, just use it
            grad_tensor = grad
        flattened_gradients.append(grad_tensor.detach().clone())
    
    # SignGuard expects each gradient to be a list of tensors, but our gradients are already flattened.
    # We need to adapt by treating each flattened gradient as a single tensor in a list.
    adapted_gradients = [[grad] for grad in flattened_gradients]
    
    # Apply SignGuard to get anomaly scores
    anomaly_scores = signguard_anomaly_scores(
        adapted_gradients, device, seed,
        L=L, R=R, selection_fraction=selection_fraction, alpha=alpha
    )
    
    # Apply Krum to get anomaly scores
    krum_scores = krum_anomaly_scores(
        gradients=flattened_gradients,
        f=f,
        device=device
    )

    # Convert anomaly scores to a list
    anomaly_scores_list = anomaly_scores.tolist()
    krum_scores_list = krum_scores.tolist()
    
    # Map scores to gradient indices
    scores_by_gradient = {
        'signguard_score': anomaly_scores_list,
        'krum_score': krum_scores_list
    }
    
    return scores_by_gradient
def generate_meta_data_records_from_keys(keys, f=0):
    """
    keys: List of keys where each key is a group ID. The length of keys is the number of gradients.
    f: Number of malicious gradients (first f indices are considered malicious).
    returns:
        List of dictionaries where each dictionary contains metadata for a gradient (client_id)
        - client_id: The index of the gradient in the original list of gradients.
        - is_user_malicious: True if the user is malicious, False otherwise. (based on first f indices being malicious)
        - group_id: The ID of the group the gradient belongs to.
        - group_id_size: The size of the group with the same group_id.
        - group_id_mal_size: The number of malicious users in the group with the same group_id.
    """
    n = len(keys)
    # Count group sizes
    group_sizes = Counter(keys)
    
    # Count malicious users in each group
    group_mal_sizes = {}
    for i, group_id in enumerate(keys):
        if i < f:  # First f indices are malicious
            group_mal_sizes[group_id] = group_mal_sizes.get(group_id, 0) + 1
    
    # Create metadata records
    records = []
    for i, group_id in enumerate(keys):
        record = {
            'client_id': i,
            'is_user_malicious': i < f,
            'group_id': group_id,
            'group_id_size': group_sizes[group_id],
            'group_id_mal_size': group_mal_sizes.get(group_id, 0),
            'n_total_groups': len(group_sizes),

        }
        records.append(record)
    
    return records


def explode_records(meta_data_records, scores_by_group, experiment_data):
    """
    Combines client metadata, group scores, and experiment data into final records.
    """
    exploded_records = []
    
    # `scoring_function_names` is your global list of score keys
    for meta_record in meta_data_records:
        group_id = meta_record['group_id']
        
        # Check if the group has scores (it always should)
        if group_id in scores_by_group:
            group_scores = scores_by_group[group_id]
            
            # Create one record for each scoring function
            for scoring_func_name, score_value in group_scores.items():
                # Start with a copy of the high-level experiment data
                final_record = experiment_data.copy()
                
                # Add the client-specific metadata
                final_record.update(meta_record)
                
                # Add the score data with corrected names
                final_record['scoring_func'] = scoring_func_name
                final_record['score'] = score_value
            
                
                exploded_records.append(final_record)
    
    return exploded_records

def round_full_scores(gradients, fixed_vis_data_for_this_round):
    # Extract high-level data once
    f_original = fixed_vis_data_for_this_round['n_total_mal']
    n = len(gradients)
    
    group_sizes = [2, 4, 6]
    f  = {
        2: 3,
        4: 2,
        6: 1
    }
    records = []
    
    for group_size in group_sizes:
        print(f"Generating keys for group size {group_size} with f={f}")
        list_of_keys = generate_keys(n=n, f=f_original, group_size=group_size)
        
        for keys in list_of_keys:
            for grouping_strategy in ['average', 'aggregate']:
            
                # 1. Create a dictionary of the high-level experiment data
                experiment_data = {
                    'round_id': fixed_vis_data_for_this_round.get('round_id'),
                    'byz_type': fixed_vis_data_for_this_round.get('byz_type'),
                    'n_total_clients': n,
                    'n_total_mal': f_original,
                    'group_size': group_size, # The target group size
                    'grouping_agg': grouping_strategy # Renamed key
                }

                # 2. Get scores organized by group_id
                assumed_mal_groups = f[group_size]


                
                scores_by_group = generate_groups_and_their_scores(
                    gradients, keys, grouping_strategy=grouping_strategy, f=assumed_mal_groups
                )
                
                # 3. Get client-level metadata
                meta_data_records = generate_meta_data_records_from_keys(keys, f=f_original)
                
                # 4. Explode records using all three data sources
                exploded_records = explode_records(meta_data_records, scores_by_group, experiment_data)

                #records.extend(exploded_records)
                records.extend(exploded_records)

    # Flatten the gradients for scoring
    user_flattened_gradients = []
    for userGrad in gradients:
        if isinstance(userGrad, list):
            # 2. If it's a list, flatten it into a single tensor.
            grad_tensor = torch.cat([layer.reshape(-1) for layer in userGrad])
        else:
            # If it's already a tensor, just use it.
            grad_tensor = userGrad

        user_flattened_gradients.append(grad_tensor.detach().clone())


    assert len(user_flattened_gradients) == len(gradients), "Length mismatch in user_flattened_gradients"
    user_scores_all_methods = run_all_scoring_methods(
        user_flattened_gradients, device=torch.device('cpu'),
                                     dnc_niters=100, dnc_c=1.5, dnc_b=100, dnc_alpha=1.0, dnc_beta=1.0,
                                     f = f[group_size]
    )
    user_scores_signguard_krum = compute_signguard_and_krum_scores(
        user_flattened_gradients, f=f_original)
    # Add SignGuard and Krum scores to the user scores
    user_scores_all_methods.update(user_scores_signguard_krum)
    # assert that there are no missing scoring functions
    scoring_function_names = [
        'simple_ensemble_votes_sim', 'simple_ensemble_votes_dist', 'simple_ensemble_combined_score',
        'flame_enhanced_direction_score', 'flame_enhanced_magnitude_score', 'flame_enhanced_combined_score',
        'dnc_bb_prob_benign', 'signguard_score', 'krum_score'
    ]
    # assert the length of the keys in user_scores_all_methods is equal to the number of scoring functions
    assert len(user_scores_all_methods) == len(scoring_function_names), \
        f"Expected {len(scoring_function_names)} scoring functions, but got {len(user_scores_all_methods)}"
    assert all(len(v) == len(user_flattened_gradients) for v in user_scores_all_methods.values()), \
        "All scoring methods must return scores for the same number of users"
    
    
    experiment_data = {
        'round_id': fixed_vis_data_for_this_round.get('round_id'),
        'byz_type': fixed_vis_data_for_this_round.get('byz_type'),
        'n_total_clients': n,
        'n_total_mal': f,
        'group_size': 1,  # Not applicable for individual scores
        'grouping_agg': "not applicable",  # Not applicable for individual scores
    }
    user_records = []
    for scoring_func_name, score_values in user_scores_all_methods.items():
        for i, score_value in enumerate(score_values):
            record = experiment_data.copy()
            record['group_id_size'] = 1  # Each user is its own group
            record['scoring_func'] = scoring_func_name
            record['client_id'] = i
            record['is_user_malicious'] = i < f_original  # First f indices are malicious
            record['group_id'] = i  # Each user is its own group
            record['group_id_mal_size'] = 1 if i < f_original else 0  # Malicious if within the first f indices
            record['score'] = score_value
            user_records.append(record)
    
    
    records.extend(user_records)

    return records
