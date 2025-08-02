import numpy as np
import torch

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