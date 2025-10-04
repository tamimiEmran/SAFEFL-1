import numpy as np
import torch
from itertools import combinations
from functools import lru_cache
from tqdm import tqdm 
# =========================
# Exhaustive, deterministic scheduler (convex repeat penalty)
# =========================
from typing import List, Tuple

def _cyclic_schedule(v: int, group_size: int, rounds: int) -> List[List[Tuple[int, ...]]]:
    """
    Deterministic cyclic schedule:
      - Users are 0..v-1 placed column-wise into m = floor(v / group_size) base groups.
      - Round r shifts indices by +r mod v (or equivalently rotates rows), yielding new groups.
      - If v % group_size != 0, we keep a rotating overflow bench; each round one group
        absorbs a few overflow users so load is spread fairly.

    Returns:
      schedule[r] = list of groups (each a tuple of user ids), partitioning 0..v-1.
    """
    if group_size < 2:
        raise ValueError("group_size must be >= 2")

    m = v // group_size                  # number of full groups
    overflow = v - m * group_size        # users left after filling m groups
    users = list(range(v))

    # Base layout: column-wise fill (group j has indices j, j+m, j+2m, ...)
    base_groups = []
    for j in range(m):
        base_groups.append(tuple(j + k * m for k in range(group_size)))

    # Overflow list: the last 'overflow' user ids (deterministic choice)
    overflow_ids = list(range(m * group_size, v)) if overflow > 0 else []

    schedule: List[List[Tuple[int, ...]]] = []

    for r in range(rounds):
        # Shift every user id by +r mod v (cyclic “round-robin”)
        def sh(x: int) -> int:
            return (x + r) % v

        round_groups = [tuple(sh(x) for x in g) for g in base_groups]

        # Distribute overflow fairly: each round, attach the rotated overflow users
        # to a different group so the extra size rotates deterministically.
        if overflow > 0:
            # Rotate which base group receives extras this round
            target_gid = r % m if m > 0 else 0
            # Rotate which overflow users get attached this round
            rot = r % max(overflow, 1)
            extras = [sh(overflow_ids[(i + rot) % overflow]) for i in range(overflow)]
            # Attach all extras to a single group (simplest fair spread over rounds)
            # If you prefer spreading across several groups, you can scatter them.
            if m == 0:
                # No full groups exist (v < group_size). Single group is all users.
                round_groups = [tuple(sh(x) for x in range(v))]
            else:
                # Attach extras
                g_as_list = list(round_groups[target_gid])
                g_as_list.extend(extras)
                round_groups[target_gid] = tuple(sorted(g_as_list))

        # Canonicalize: sort members in each group, and groups by their smallest member
        round_groups = [tuple(sorted(g)) for g in round_groups]
        round_groups.sort(key=lambda t: (t[0], t))
        schedule.append(round_groups)

    return schedule

def _canonical_partitions(users, group_sizes):
    """
    Yield all partitions of 'users' (sorted tuple of ints) into groups whose sizes are 'group_sizes'
    (list of ints), without duplicates, in deterministic lexicographic order.

    Symmetry breaking:
      • Always anchor the smallest remaining user when forming the next group.
      • Choose companions as combinations of larger remaining users.
      • Sort group_sizes (descending) so equal-size groups are canonical.
    """
    users = tuple(sorted(users))
    group_sizes = tuple(sorted(group_sizes, reverse=True))

    @lru_cache(maxsize=None)
    def rec(remaining_users, remaining_sizes):
        remaining_users = list(remaining_users)
        if not remaining_sizes:
            yield ()
            return

        k = remaining_sizes[0]
        # anchor = smallest remaining user
        anchor = remaining_users[0]
        rest = remaining_users[1:]

        # pick the other k-1 members for this group
        for companions in combinations(rest, k - 1):
            group = (anchor,) + companions

            used = set(group)
            nxt_users = tuple(u for u in remaining_users if u not in used)

            for tail in rec(nxt_users, remaining_sizes[1:]):
                yield (group,) + tail

    return rec(tuple(users), tuple(group_sizes))


def _convex_increment_for_round(groups, pair_count):
    """
    Convex repeat penalty:
      For each pair (u,v) in this round's groups, add current pair_count[(u,v)] to the cost.
      (First time together: adds 0; second time: +1; third time: +2; ...)
    """
    inc = 0
    for g in groups:
        g = tuple(g)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                u, v = g[i], g[j]
                if u > v: u, v = v, u
                inc += pair_count.get((u, v), 0)
    return inc


def _apply_round(groups, pair_count):
    """Return updated pair_count after applying this round's groups."""
    new_pc = pair_count.copy()
    for g in groups:
        g = tuple(g)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                u, v = g[i], g[j]
                if u > v: u, v = v, u
                new_pc[(u, v)] = new_pc.get((u, v), 0) + 1
    return new_pc


def _sizes_sequence(v, K, X, m=None, rotate_sizes=True):
    """
    Replicates your size logic:
      - If m is None: m = v // K; distribute remainder as +1 to first t groups.
      - If rotate_sizes: rotate which groups get the +1 extras each round.
    Returns: list[list[int]] of sizes for each round.
    """
    if v < K:
        raise ValueError("v must be >= K")
    if m is None:
        m = v // K
        if m == 0:
            raise ValueError("With m=None, v must be at least K")
    else:
        if v < m * K:
            raise ValueError(f"Need v >= m*K = {m*K}, got v={v}")

    remainder = v - m * K
    q, t = divmod(remainder, m) if m > 0 else (0, 0)
    base = [K + q + (1 if i < t else 0) for i in range(m)]

    def sizes_for_round(r):
        if not rotate_sizes or m == 0:
            return list(base)
        shift = r % m
        return base[-shift:] + base[:-shift]

    sizes_per_round = [sizes_for_round(r) for r in range(X)]
    # Validate
    for s in sizes_per_round:
        if sum(s) != v:
            raise ValueError("Sizes must partition all users each round")
    return sizes_per_round


def schedule_exhaustive_min_pair_repeats(v, K, X, m=None, rotate_sizes=True, max_cost=float('inf')):
    """
    Deterministic exhaustive search with branch-and-bound under a convex repeat penalty.
    Returns: (schedule, best_cost, sizes_per_round)
      - schedule[r] is a list of groups (each group is a tuple of user ids).
    """
    sizes_per_round = _sizes_sequence(v, K, X, m=m, rotate_sizes=rotate_sizes)
    users = tuple(range(v))
    R = len(sizes_per_round)
    print("step 1 done")
    # Precompute all canonical partitions per round (deterministic order)
    all_round_parts = []
    for sizes in tqdm(sizes_per_round):
        parts = list(_canonical_partitions(users, sizes))
        parts.sort()  # canonical order of tuple-of-tuples
        all_round_parts.append(parts)
    print("step 2 done")
    # Lower bound: convex increment is always >= 0, so we only use running best to prune.
    best_sched = None
    best_cost = max_cost
    init_pc = {}

    def bt(r, pair_count, acc_cost, partial):
        nonlocal best_sched, best_cost
        if acc_cost >= best_cost:
            return
        if r == R:
            best_cost = acc_cost
            best_sched = [list(map(tuple, gs)) for gs in partial]
            return

        for groups in all_round_parts[r]:
            inc = _convex_increment_for_round(groups, pair_count)
            new_cost = acc_cost + inc
            if new_cost >= best_cost:
                continue
            new_pc = _apply_round(groups, pair_count)
            bt(r + 1, new_pc, new_cost, partial + [groups])

    bt(0, init_pc, 0, [])
    print("step 3 done")
    if best_sched is None:
        raise RuntimeError("No schedule found; check sizes_per_round validity.")
    return best_sched, best_cost, sizes_per_round

def initialize_grouping_params(num_users, min_users_per_group=2, gradient_threshold=0.5, aggregation_method='average'):
    """
    Initialize the grouping parameters dictionary with all necessary tracking information.
    
    Args:
        num_users: Total number of users in the system
        min_users_per_group: Minimum number of users per group (default: 2)
        gradient_threshold: Threshold for selecting gradients based on anomaly scores (default: 0.5)
        aggregation_method: Method for aggregating gradients ('average' or 'sum')
    
    Returns:
        Dictionary containing all grouping parameters
    """
    grouping_params = {
        # Core parameters
        "num_users": num_users,
        "min_users_per_group": min_users_per_group,
        "round": 1,
        "gradient_threshold": gradient_threshold,
        "aggregation_method": aggregation_method,
        
        # User tracking
        "user_membership": [0] * num_users,  # Current group assignment for each user
        "user_scores": [0.0] * num_users,    # Anomaly scores for each user
        "user_participation": [0] * num_users,  # Count of rounds each user participated
        "user_selected_rounds": [[] for _ in range(num_users)],  # Rounds where user was selected
        "prev_round_anomaly_scores": {},  # Dictionary of user_id: anomaly_score from previous round
        
        # Group tracking
        "group_history": [],  # History of group compositions per round
        "group_gradients_history": [],  # History of group gradients per round
        "filtered_users_history": [],  # History of filtered users per round
        
        # Performance tracking
        "anomaly_scores_history": [],  # History of anomaly scores per round
        "selected_users_per_round": [],  # Users selected based on threshold
        "group_sizes": []  # Sizes of groups after filtering
    }
    
    return grouping_params


def select_best_gradients(grouping_params):
    """
    Step 1: Select best gradients based on percentile before grouping.
    
    Args:
        grouping_params: Dictionary containing grouping parameters
    
    Returns:
        List of selected user indices and updated grouping_params
    """
    
    percentile = grouping_params["gradient_percentile"]
    num_users = grouping_params["num_users"]
    users_anomaly_scores = grouping_params.get("prev_round_anomaly_scores", {})
    
    # If no anomaly scores provided, include all users with score 0
    if not users_anomaly_scores:
        users_anomaly_scores = {i: 0.0 for i in range(num_users)}
    
    # Sort users by anomaly scores (lowest to highest)
    sorted_users = sorted([(user_id, score) for user_id, score in users_anomaly_scores.items() 
                          if user_id < num_users], key=lambda x: x[1])
    
    # Select top percentile of users with lowest anomaly scores
    num_to_select = max(1, int(len(sorted_users) * percentile / 100))
    selected_users = [user_id for user_id, _ in sorted_users[:num_to_select]]

    return selected_users


def shuffle_and_assign_groups(selected_users, grouping_params, seed=None):
    """
    Step 2 & 3: Shuffle selected users and assign them to groups based on minimum group size.
    All groups will have at least min_users_per_group users.
    
    Args:
        selected_users: List of user indices selected based on gradient threshold
        grouping_params: Dictionary containing grouping parameters
        seed: Random seed for reproducibility
    
    Returns:
        Updated grouping_params with new group assignments
    """
    if seed is not None:
        np.random.seed(seed)
    
    min_users_per_group = grouping_params["min_users_per_group"]
    num_selected = len(selected_users)
    
    # Calculate number of groups that ensures all groups have at least min_users_per_group
    # The last group may have up to (2 * min_users_per_group - 1) users
    if num_selected < min_users_per_group:
        # Not enough users for even one group of minimum size
        actual_num_groups = 1  # Create one group with all available users
    else:
        # Calculate maximum possible groups while ensuring minimum size
        actual_num_groups = num_selected // min_users_per_group
        remainder = num_selected % min_users_per_group
        
        # If remainder is non-zero and we have more than one group,
        # we need to reduce the number of groups to distribute the remainder
        if remainder > 0 and actual_num_groups > 1:
            # Reduce groups by 1 to ensure all groups meet minimum size
            actual_num_groups -= 1
    
    # Store the actual number of groups for this round
    grouping_params["num_groups_this_round"] = actual_num_groups
    
    # Shuffle the selected users
    shuffled_users = selected_users.copy()
    np.random.shuffle(shuffled_users)
    
    # Reset user membership for all users
    grouping_params["user_membership"] = [-1] * grouping_params["num_users"]
    
    # Assign shuffled users to groups
    if actual_num_groups == 1:
        # Special case: all users in one group
        for user_id in shuffled_users:
            grouping_params["user_membership"][user_id] = 0
            grouping_params["user_participation"][user_id] += 1
            grouping_params["user_selected_rounds"][user_id].append(grouping_params["round"])
    else:
        # Distribute users across groups
        # All groups except the last have exactly min_users_per_group
        # The last group gets all remaining users
        start_idx = 0
        for group_id in range(actual_num_groups):
            if group_id < actual_num_groups - 1:
                # All groups except the last have exactly min_users_per_group users
                group_size = min_users_per_group
            else:
                # Last group gets all remaining users
                group_size = num_selected - start_idx
            
            end_idx = start_idx + group_size
            
            # Assign users in this range to the current group
            for idx in range(start_idx, end_idx):
                if idx < num_selected:
                    user_id = shuffled_users[idx]
                    grouping_params["user_membership"][user_id] = group_id
                    # Update participation tracking
                    grouping_params["user_participation"][user_id] += 1
                    grouping_params["user_selected_rounds"][user_id].append(grouping_params["round"])
            
            start_idx = end_idx
    
    # Record group composition for this round
    groups_this_round = [[] for _ in range(actual_num_groups)]
    for user_id, group_id in enumerate(grouping_params["user_membership"]):
        if group_id != -1:  # User is assigned to a group
            groups_this_round[group_id].append(user_id)
    
    grouping_params["group_history"].append(groups_this_round)
    grouping_params["group_sizes"].append([len(g) for g in groups_this_round if len(g) > 0])
    
    # Log group distribution info
    print(f"Created {actual_num_groups} groups from {num_selected} selected users")
    print(f"Group sizes: {[len(g) for g in groups_this_round]}")
    
    return grouping_params


def aggregate_group_gradients(gradients, grouping_params, device='cpu'):
    """
    Step 4: Aggregate gradients within each group.
    
    Args:
        gradients: List of gradient tensors for each user
        grouping_params: Dictionary containing grouping parameters
        device: Computation device
    
    Returns:
        Dictionary mapping group IDs to aggregated gradients
    """
    aggregation_method = grouping_params["aggregation_method"]
    user_membership = grouping_params["user_membership"]
    
    # Initialize group gradients
    group_gradients = {}
    group_user_counts = {}
    
    # Get gradient shape from first gradient
    gradient_shape = gradients[0].shape if gradients else None
    
    # Aggregate gradients for each group
    for user_id, group_id in enumerate(user_membership):
        if group_id != -1 and user_id < len(gradients):  # User is assigned to a group
            if group_id not in group_gradients:
                group_gradients[group_id] = torch.zeros(gradient_shape).to(device)
                group_user_counts[group_id] = 0
            
            group_gradients[group_id] += gradients[user_id]
            group_user_counts[group_id] += 1
    
    # Apply aggregation method
    if aggregation_method == 'average':
        for group_id in group_gradients:
            if group_user_counts[group_id] > 0:
                group_gradients[group_id] /= group_user_counts[group_id]
    # If 'sum', gradients are already summed
    
    # Store gradient history
    grouping_params["group_gradients_history"].append(group_gradients)
    
    return group_gradients


def perform_grouping_round(gradients, grouping_params, anomaly_scores=None, device='cpu', seed=None):
    """
    Perform a complete grouping round with all steps.
    
    Args:
        gradients: List of gradient tensors for each user
        grouping_params: Dictionary containing grouping parameters
        anomaly_scores: Dictionary mapping user_id to anomaly scores for this round
        device: Computation device
        seed: Random seed for reproducibility
    
    Returns:
        group_gradients: Dictionary mapping group IDs to aggregated gradients
        grouping_params: Updated parameters dictionary
    """
    # Update anomaly scores if provided
    if anomaly_scores is not None:
        grouping_params["prev_round_anomaly_scores"] = anomaly_scores
    
    # Step 1: Select best gradients based on threshold
    selected_users = select_best_gradients(grouping_params)
    
    # Step 2 & 3: Shuffle users and assign to groups
    grouping_params = shuffle_and_assign_groups(selected_users, grouping_params, seed)
    
    # Step 4: Aggregate gradients within groups
    group_gradients = aggregate_group_gradients(gradients, grouping_params, device)
    
    # Increment round counter
    grouping_params["round"] += 1
    
    return group_gradients, grouping_params


def get_grouping_statistics(grouping_params):
    """
    Get statistics about the grouping process.
    
    Args:
        grouping_params: Dictionary containing grouping parameters
    
    Returns:
        Dictionary with various statistics
    """
    stats = {
        "total_rounds": grouping_params["round"] - 1,
        "average_users_selected": np.mean([len(users) for users in grouping_params["selected_users_per_round"]]),
        "user_participation_rates": [count / (grouping_params["round"] - 1) if grouping_params["round"] > 1 else 0 
                                     for count in grouping_params["user_participation"]],
        "average_group_sizes": [np.mean(sizes) if sizes else 0 for sizes in grouping_params["group_sizes"]],
        "most_active_users": sorted(enumerate(grouping_params["user_participation"]), 
                                   key=lambda x: x[1], reverse=True)[:10],
        "gradient_norm_trends": grouping_params["gradient_norms"]
    }
    return stats


import random
def _group_and_sum_gradients(param_list, bayesian_params):
    if bayesian_params.get("shuffling_strategy", "random") == 'mixed':
        if bayesian_params.get("current_round", 0) <= bayesian_params.get("mixing_rounds", 100):
            strategy = "greedy"
        else:
            strategy = "by_maliciousness"
    if bayesian_params.get("shuffling_strategy", "random") == 'mixed_optimal':
        if bayesian_params.get("current_round", 0) <= bayesian_params.get("mixing_rounds", 100):
            strategy = "cyclic"

        else:
            strategy = 'by_maliciousness'

    else:

        strategy = bayesian_params.get("shuffling_strategy", "random")

    

    if strategy == "random":
        # users included
        gradients_included = {id: grad for id, grad in enumerate(param_list)}

        # shuffle IDs
        shuffled_ids = list(gradients_included.keys())
        random.shuffle(shuffled_ids)
        gradients_included = {id: gradients_included[id] for id in shuffled_ids}

        # group IDs
        group_size = bayesian_params.get("group_size", 2)
        # ensure that all groups have the group size, if not divisible ensure the last group has more than the group size.
        groups = {}
        group_id = 0
        ids_list = list(gradients_included.keys())

        for i in range(0, len(ids_list), group_size):
            group_indices = ids_list[i:i + group_size]
            
            # If this is the last group and it's smaller than group_size, merge with previous group
            if len(group_indices) < group_size and group_id > 0:
                # Merge with the previous group
                groups[group_id - 1] = (group_id - 1, groups[group_id - 1][1] + group_indices)
            else:
                groups[group_id] = (group_id, group_indices)
                group_id += 1

        # Extract group gradients
        group_gradients = {}
        for gid, (_, indices) in groups.items():
            # compute global model update
            gradients_to_be_summed = [gradients_included[idx] for idx in indices]
            group_gradients[gid]  = torch.sum(torch.stack(gradients_to_be_summed), dim=0)

    elif strategy  == "by_maliciousness":
        # users included
        gradients_included = {id: grad for id, grad in enumerate(param_list)}
        # sort by maliciousness
        latent_variables = bayesian_params.get("latent_variables", {})
        sorted_ids = sorted(gradients_included.keys(), key=lambda x: latent_variables.get(x, 0.5))
        

        # group IDs
        group_size = bayesian_params.get("group_size", 2)
        # ensure that all groups have the group size, if not divisible ensure the last group has more than the group size.
        groups = {}
        group_id = 0

        for i in range(0, len(sorted_ids), group_size):
            group_indices = sorted_ids[i:i + group_size]
            
            # If this is the last group and it's smaller than group_size, merge with previous group
            if len(group_indices) < group_size and group_id > 0:
                # Merge with the previous group
                groups[group_id - 1] = (group_id - 1, groups[group_id - 1][1] + group_indices)
            else:
                groups[group_id] = (group_id, group_indices)
                group_id += 1

        # Extract group gradients
        group_gradients = {}
        for gid, (_, indices) in groups.items():
            # compute global model update
            gradients_to_be_summed = [gradients_included[idx] for idx in indices]
            group_gradients[gid]  = torch.sum(torch.stack(gradients_to_be_summed), dim=0)

    elif strategy == "greedy":
        round_id = bayesian_params.get("current_round", 0)
        if round_id == 0:
            #initialze schedule 
            global_best, global_best_stats, global_best_sizes = schedule_min_size_K(
                v=len(param_list), 
                K=bayesian_params.get("group_size", 2), 
                X=bayesian_params.get("mixing_rounds", 10) + 1, 
                attempts_per_round=bayesian_params.get("attempts_per_round", 5), 
                restarts=bayesian_params.get("restarts", 2),
                seed=bayesian_params.get("seed", 42),
                triple_penalty=bayesian_params.get("triple_penalty", 2),
                rotate_sizes=bayesian_params.get("rotate_sizes", True)
            )

            bayesian_params["schedule"] = global_best


        schedule = bayesian_params["schedule"]
        current_round_schedule = schedule[round_id]
        groups = {
            gid: (gid, list(group))
            for gid, group in enumerate(current_round_schedule)

        }

        # Extract group gradients
        group_gradients = {}
        for gid, (_, indices) in groups.items():
            # compute global model update
            gradients_to_be_summed = [param_list[idx] for idx in indices]
            group_gradients[gid]  = torch.sum(torch.stack(gradients_to_be_summed), dim=0)

    elif strategy == "prob_by_maliciousness":
        # users included
        gradients_included = {id: grad for id, grad in enumerate(param_list)}

        latent_variables = bayesian_params.get("latent_variables", {})
        temperature = bayesian_params.get("prob_sort_temp", 0.2)  # controls randomness

        # Create noisy scores for probabilistic sort
        noisy_scores = {
            uid: latent_variables.get(uid, 0.5) + random.uniform(-temperature, temperature)
            for uid in gradients_included.keys()
        }

        # Sort by noisy scores
        sorted_ids = sorted(noisy_scores.keys(), key=lambda x: noisy_scores[x])

        # group IDs
        group_size = bayesian_params.get("group_size", 2)
        groups = {}
        group_id = 0

        for i in range(0, len(sorted_ids), group_size):
            group_indices = sorted_ids[i:i + group_size]

            # If this is the last group and it's smaller than group_size, merge with previous group
            if len(group_indices) < group_size and group_id > 0:
                groups[group_id - 1] = (group_id - 1, groups[group_id - 1][1] + group_indices)
            else:
                groups[group_id] = (group_id, group_indices)
                group_id += 1

        # Extract group gradients
        group_gradients = {}
        for gid, (_, indices) in groups.items():
            gradients_to_be_summed = [gradients_included[idx] for idx in indices]
            group_gradients[gid] = torch.sum(torch.stack(gradients_to_be_summed), dim=0)


    elif strategy == 'optimal':
        # Build (once) a fully-deterministic, convex-penalty-minimizing schedule across the mixing window
        round_id = bayesian_params.get("current_round", 0)
        if round_id == 0 or "schedule" not in bayesian_params:
            v = len(param_list)
            K = bayesian_params.get("group_size", 2)
            X = bayesian_params.get("mixing_rounds", 10) + 1  # same as your greedy branch
            m = bayesian_params.get("num_groups", None)       # optional: let user override m
            rotate = bayesian_params.get("rotate_sizes", True)

            schedule, best_cost, sizes_per_round = schedule_exhaustive_min_pair_repeats(
                v=v, K=K, X=X, m=m, rotate_sizes=rotate
            )
            bayesian_params["schedule"] = schedule
            bayesian_params["schedule_cost"] = best_cost
            bayesian_params["sizes_per_round"] = sizes_per_round

        schedule = bayesian_params["schedule"]
        current_round_schedule = schedule[round_id]

        # Materialize groups dict in your expected format
        groups = {gid: (gid, list(group)) for gid, group in enumerate(current_round_schedule)}

        # Extract group gradients
        group_gradients = {}
        for gid, (_, indices) in groups.items():
            grads = [param_list[idx] for idx in indices]
            group_gradients[gid] = torch.sum(torch.stack(grads), dim=0)

    elif strategy == "cyclic":
        # Deterministic, user-balanced cyclic schedule (no search).
        v = len(param_list)
        K = bayesian_params.get("group_size", 2)
        # How many rounds do you want to precompute? Usually your mixing window + 1,
        # mirroring your greedy/optimal branches:
        X = bayesian_params.get("mixing_rounds", 10) + 1

        round_id = bayesian_params.get("current_round", 0)

        if round_id == 0 or "schedule" not in bayesian_params:
            schedule = _cyclic_schedule(v=v, group_size=K, rounds=X)
            bayesian_params["schedule"] = schedule
            # Optional: store the theoretical max rounds with no pair repeats
            bayesian_params["max_no_repeat_rounds"] = (v - 1) // (K - 1) if K > 1 else 0

        schedule = bayesian_params["schedule"]
        current_round_schedule = schedule[round_id]

        # Produce your expected groups dict
        groups = {
            gid: (gid, list(group))
            for gid, group in enumerate(current_round_schedule)
        }

        # Sum gradients per group
        group_gradients = {}
        for gid, (_, indices) in groups.items():
            gradients_to_be_summed = [param_list[idx] for idx in indices]
            group_gradients[gid] = torch.sum(torch.stack(gradients_to_be_summed), dim=0)


    else:
        raise ValueError(f"Unknown shuffling strategy: {bayesian_params.get('shuffling_strategy')}")




    return groups, group_gradients

from collections import defaultdict
from itertools import combinations
def _SG_group_and_sum_gradients(param_list, bayesian_params):

    sg_aggregators_len = bayesian_params.get('sg_aggregators_len', 5)
    n = len(param_list)
    seed = bayesian_params.get('seed', 47)
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    sg_agg_assignment = {}
    for agg_id in range(sg_aggregators_len):
        for idx in indices[agg_id::sg_aggregators_len]:
            sg_agg_assignment[idx] = agg_id
    agg_to_ids = defaultdict(list)

    for cid, aid in sg_agg_assignment.items():
        agg_to_ids[aid].append(cid)


    strategy = bayesian_params.get("shuffling_strategy", "random")



    

    if strategy == "random":
        # randomly pick 10% of each aid
        randomly_picked_pct_from_sg_agg = bayesian_params.get('randomly_picked_pct_from_sg_agg', 0.1)
        selected_ids = []
        for aid, ids in agg_to_ids.items():
            k = max(2, int(randomly_picked_pct_from_sg_agg * len(ids)))
            selected_ids.extend(rng.sample(ids, k))

        

        # Extract group gradients
        group_gradients = {}
        for aid, indices in agg_to_ids.items():
            # compute global model update
            gradients_to_be_summed = [param_list[idx] for idx in indices if idx in selected_ids]
            group_gradients[aid]  = torch.sum(torch.stack(gradients_to_be_summed), dim=0)
        groups = {aid: (aid, indices) for aid, indices in agg_to_ids.items()}
        return groups, group_gradients
    else:
        raise ValueError(f"Unknown shuffling strategy: {bayesian_params.get('shuffling_strategy')}")




def schedule_min_size_K(v, K, X, m=None, attempts_per_round=400, restarts=1,
                        seed=None, triple_penalty=0.0, rotate_sizes=True):
    """
    Schedule v users into groups across X rounds such that:
      • Each round partitions all users.
      • Every group has size >= K (no dummies).
      • If m is provided, each round has exactly m groups (requires v >= m*K).
      • If m is None, it uses m = v // K (max number of groups),
        and distributes the remainder as +1s to some groups.
      • Across rounds, it greedily minimizes repeated pairings.

    Parameters
    ----------
    v : int
        Number of users (0..v-1).
    K : int
        Minimum group size (per group, per round).
    X : int
        Number of rounds.
    m : int or None
        Groups per round. If None, uses floor(v/K).
    attempts_per_round : int
        Random candidate partitions tried per round (higher → better, slower).
    restarts : int
        Global restarts; best schedule kept.
    seed : int or None
        Random seed for reproducibility.
    triple_penalty : float
        Optional extra penalty for repeated triples (k>=3). Use 0.0 to disable.
    rotate_sizes : bool
        If True, cyclically rotates which groups get the +1 extras each round
        to balance large-group exposure.

    Returns
    -------
    schedule : list[list[tuple]]
        schedule[r] is a list of groups (tuples of user ids) for round r.
    stats : dict
        Summary (pair_count, max_pair_repeats, avg_pair_repeats, cost).
    sizes_per_round : list[list[int]]
        The exact group sizes used in each round (each >= K).
    """
    if v < K:
        raise ValueError("v must be >= K; otherwise you cannot form any group with size >= K.")

    rng = random.Random(seed)

    # Decide number of groups m and the base size pattern for one round
    if m is None:
        m = v // K  # as many groups as possible subject to min size K
        if m == 0:
            # Only possible group is the whole set, but that would be < K, already handled above
            raise ValueError("With m=None, v must be at least K to form any group.")
    else:
        if v < m * K:
            raise ValueError(f"Given m={m} and K={K}, need v >= m*K = {m*K} (got v={v}).")

    remainder = v - m * K  # total "extra seats" to distribute across m groups
    q, t = divmod(remainder, m) if m > 0 else (0, 0)

    # Base size vector: K + q everywhere, and +1 on the first t groups.
    base_sizes = [K + q + (1 if i < t else 0) for i in range(m)]

    def sizes_for_round(rnd_idx):
        if not rotate_sizes or m == 0:
            return list(base_sizes)
        # Rotate extras to balance who ends up in larger groups across rounds
        shift = rnd_idx % m
        return base_sizes[-shift:] + base_sizes[:-shift]

    # Cost function based on pair/triple reuse
    def partition_cost(groups, pair_count, triple_count):
        c = 0.0
        for g in groups:
            # Pairs
            for a, b in combinations(g, 2):
                key = (a, b) if a < b else (b, a)
                pc = pair_count[key]
                c += pc * pc
            # Triples (optional)
            if triple_penalty > 0 and len(g) >= 3:
                for a, b, c3 in combinations(g, 3):
                    key3 = tuple(sorted((a, b, c3)))
                    tc = triple_count[key3]
                    c += triple_penalty * (tc * tc)
        return c

    def update_counts(groups, pair_count, triple_count):
        for g in groups:
            for a, b in combinations(g, 2):
                key = (a, b) if a < b else (b, a)
                pair_count[key] += 1
            if triple_penalty > 0 and len(g) >= 3:
                for a, b, c3 in combinations(g, 3):
                    key3 = tuple(sorted((a, b, c3)))
                    triple_count[key3] += 1

    # Build schedule with possible restarts, keep best
    global_best = None
    global_best_stats = None
    global_best_sizes = None

    for _ in range(restarts):
        users = list(range(v))
        pair_count = defaultdict(int)
        triple_count = defaultdict(int)
        sched = []
        sizes_seq = []
        total_cost = 0.0

        for rnd in range(X):
            sizes = sizes_for_round(rnd)
            sizes_seq.append(list(sizes))

            best_groups = None
            best_score = float("inf")

            # Try many random permutations, slice into the target sizes, pick lowest cost
            for _try in range(attempts_per_round):
                rng.shuffle(users)
                groups = []
                idx = 0
                for sz in sizes:
                    groups.append(tuple(users[idx:idx+sz]))
                    idx += sz
                # (idx should be exactly v)
                score = partition_cost(groups, pair_count, triple_count)
                if score < best_score:
                    best_score = score
                    best_groups = groups

            sched.append(best_groups)
            total_cost += best_score
            update_counts(best_groups, pair_count, triple_count)

        # Summarize pair statistics
        counts = list(pair_count.values())
        max_rep = max(counts) if counts else 0
        avg_rep = (sum(counts) / len(counts)) if counts else 0.0
        stats = {
            "pair_count": dict(pair_count),
            "max_pair_repeats": max_rep,
            "avg_pair_repeats": avg_rep,
            "cost": total_cost,
        }

        if global_best is None or stats["cost"] < global_best_stats["cost"]:
            global_best = sched
            global_best_stats = stats
            global_best_sizes = sizes_seq

    return global_best, global_best_stats, global_best_sizes

# Example usage
if __name__ == "__main__":
    # Initialize parameters
    num_users = 100
    min_users_per_group = 3  # Minimum 3 users per group
    gradient_dim = 1000
    device = 'cpu'
    
    # Create dummy gradients for demonstration
    gradients = [torch.randn(gradient_dim) for _ in range(num_users)]
    
    # Initialize grouping parameters
    grouping_params = initialize_grouping_params(
        num_users=num_users,
        min_users_per_group=min_users_per_group,
        gradient_threshold=0.5,  # Select users with anomaly score < 0.5
        aggregation_method='average'
    )
    
    # Perform multiple rounds
    for round_num in range(5):
        print(f"\n--- Round {round_num + 1} ---")
        
        # Generate random anomaly scores for this round (simulate scores from larger framework)
        # Lower scores are better (less anomalous)
        anomaly_scores = {
            user_id: np.random.beta(2, 5)  # Beta distribution skewed towards lower values
            for user_id in range(num_users)
        }
        
        # Perform grouping round
        group_gradients, grouping_params = perform_grouping_round(
            gradients=gradients,
            grouping_params=grouping_params,
            anomaly_scores=anomaly_scores,
            device=device,
            seed=round_num  # Different seed for each round
        )
        
        print(f"Number of groups with gradients: {len(group_gradients)}")
        print(f"Selected users: {len(grouping_params['selected_users_per_round'][-1])}")
        print(f"Average anomaly score of selected users: {np.mean([anomaly_scores[u] for u in grouping_params['selected_users_per_round'][-1]]):.3f}")
        
        # Generate new gradients for next round (simulate training)
        gradients = [torch.randn(gradient_dim) for _ in range(num_users)]
    
    # Get final statistics
    stats = get_grouping_statistics(grouping_params)
    print(f"\n--- Final Statistics ---")
    print(f"Total rounds: {stats['total_rounds']}")
    print(f"Average users selected per round: {stats['average_users_selected']:.2f}")
    print(f"Top 5 most active users: {stats['most_active_users'][:5]}")
    
    # Test with different minimum group sizes
    print("\n\n--- Testing Different Minimum Group Sizes ---")
    test_cases = [
        (17, 2),  # 17 users, min 2 per group -> 8 groups of 2, 1 group of 3
        (17, 3),  # 17 users, min 3 per group -> 4 groups of 3, 1 group of 5
        (20, 3),  # 20 users, min 3 per group -> 6 groups of 3, 1 group of 2 (becomes 5 groups of 3, 1 of 5)
        (21, 3),  # 21 users, min 3 per group -> 7 groups of 3
        (10, 4),  # 10 users, min 4 per group -> 1 group of 4, 1 group of 6
        (15, 4),  # 15 users, min 4 per group -> 3 groups of 4, 1 group of 3 (becomes 2 groups of 4, 1 of 7)
        (16, 4),  # 16 users, min 4 per group -> 4 groups of 4
        (5, 3),   # 5 users, min 3 per group -> 1 group of 5
        (2, 3),   # 2 users, min 3 per group -> 1 group of 2 (less than min)
    ]
    
    for num_selected, min_size in test_cases:
        print(f"\nTest: {num_selected} users, min {min_size} per group")
        
        if num_selected < min_size:
            print(f"  → 1 group of {num_selected} (less than minimum)")
        else:
            num_groups = num_selected // min_size
            remainder = num_selected % min_size
            
            if remainder == 0:
                print(f"  → {num_groups} groups of {min_size}")
            else:
                # Need to reduce groups to ensure all meet minimum
                if num_groups > 1:
                    actual_groups = num_groups - 1
                    last_group_size = num_selected - (actual_groups * min_size)
                    print(f"  → {actual_groups} groups of {min_size}, 1 group of {last_group_size}")
                else:
                    print(f"  → 1 group of {num_selected}")