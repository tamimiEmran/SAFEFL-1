# Hierarchical Federated Learning (HeirichalFL) Implementation

This document provides a comprehensive technical overview of the hierarchical federated learning implementation in SAFEFL, focusing on the `heirichalFL.py` module and the `heirichalFL` function in `aggregation_rules.py`.

## Overview

Hierarchical Federated Learning (HFL) is an advanced aggregation approach that organizes clients into groups, aggregates gradients within each group, scores groups based on their trustworthiness, and then applies robust aggregation across groups. This multi-layered approach provides enhanced robustness against Byzantine attacks.

## Architecture

The HFL implementation consists of several key phases:

1. **Group Formation & User Assignment** (`simulate_groups`, `shuffle_users`)
2. **Intra-Group Aggregation** (`aggregate_groups`, `compute_group_gradients`)
3. **Group Scoring & Trust Management** (`score_groups`, `update_user_scores`)
4. **Inter-Group Robust Aggregation** (`robust_groups_aggregation`)

## Key Files

- `heirichalFL.py`: Core implementation of all HFL algorithms and data structures
- `aggregation_rules.py:heirichalFL()`: Main orchestrator function (lines 944-1046)
- `main.py`: Parameter parsing and integration (lines 161-165, 547-667)

## Parameters

### Command Line Arguments (main.py:161-165)
```python
--n_groups          # Number of groups (default: 5)
--assumed_mal_prct   # Assumed percentage of malicious users (default: 0.1)
```

### Hierarchical Parameters Dictionary
The `heirichal_params` dictionary contains:
```python
{
    "user membership": List[int],     # Group assignment for each user
    "user score": List[float],        # Trust scores for each user  
    "round": int,                     # Current round number
    "num groups": int,                # Total number of groups
    "assumed_mal_prct": float,        # Assumed malicious percentage
    "GT malicious": List[bool],       # Ground truth malicious labels
    "history": List[Dict],            # Historical data for analysis
    "current group scores": Dict      # Current round group scores
}
```

## Core Functions in heirichalFL.py

### Group Management

#### `simulate_groups(heirichal_params, number_of_users, seed)` (lines 16-68)
**Purpose**: Initial group assignment ensuring equal distribution
- Creates balanced groups (base_per_group ± 1)
- Handles remainder users by distributing them across first groups
- Initializes user scores to 0.0 for all users
- Only executes on round 1

#### `shuffle_users(heirichal_params, number_of_users, seed)` (lines 71-146)
**Purpose**: Diagonal shuffling across groups between rounds
- Uses 2D representation with dummy users for equal group sizes
- Implements diagonal pattern traversal: `new_group = (old_group + diag_idx) % num_groups`
- Removes dummy users from final assignment
- Ensures all real users are properly reassigned

### Aggregation Pipeline

#### `aggregate_groups(user_gradient_vectors, computation_device, random_seed, hierarchical_parameters, skip_filtering=False)` (lines 352-424)
**Purpose**: Main aggregation function with optional malicious user filtering
- **Input**: User gradients, device, seed, hierarchical params, skip_filtering flag
- **Process**: 
  1. Organizes users by group
  2. Filters malicious users globally (if not skipping)
  3. Computes group gradients
- **Output**: Dictionary mapping group IDs to aggregated gradients, list of filtered users

#### `filter_malicious_users(groups_with_users, all_user_trust_scores, malicious_percentage, total_user_count)` (lines 167-218)
**Purpose**: Global filtering of assumed malicious users
- Calculates number of users to exclude: `max(1, int(malicious_percentage * total_user_count))`
- Sorts all users by trust scores globally
- Filters bottom-scoring users across all groups
- Removes groups with ≤1 user after filtering
- Returns valid groups and filtered user list

#### `compute_group_gradients(filtered_groups_with_ids, user_gradients, gradient_shape, computation_device)` (lines 220-255)
**Purpose**: Intra-group gradient aggregation
- Sums gradients within each group
- Optionally averages by group size (controlled by `isGroupGradientsToBeAveraged = True`)
- Returns dictionary mapping original group IDs to aggregated gradients

### Trust and Scoring System

#### `score_groups(group_to_gradient_mapping, hierarchical_parameters)` (lines 428-471)
**Purpose**: Multi-metric group trustworthiness scoring
- **Ensemble Voting Approach**:
  1. Calculates cosine similarity matrix between all group gradients
  2. Performs ensemble voting where each group rates others
  3. Calculates Euclidean distance matrix for norm-based scoring
  4. Combines similarity and distance metrics
- **Fallback**: Returns default scores (1.0) for <3 groups
- **Output**: Dictionary mapping group IDs to trust scores

##### Sub-functions:
- `_calculate_similarity_matrix_()`: Pairwise cosine similarities
- `_calculate_gradient_distance_matrix_()`: Pairwise Euclidean distances  
- `_perform_ensemble_voting_()`: Accumulates votes based on similarities
- `_convert_votes_to_scores_()`: Maps votes to group trust scores

#### `update_user_scores(heirichal_params, groups_scores)` (lines 475-513)
**Purpose**: Propagates group scores to individual users
- Ranks groups by scores and calculates relative adjustments
- Uses formula: `adjustment = (mid_point - rank) / number_of_groups`
- Assigns minimum score to filtered-out groups
- Updates user scores: `user_scores[user_id] += group_adjustments[group_id]`

### Robust Inter-Group Aggregation

#### `robust_groups_aggregation(group_gradients, net, lr, device, heirichal_params, number_of_users)` (lines 517-667)
**Purpose**: Final robust aggregation across groups with multiple strategies

**Global Configuration Flags** (lines 6-13):
```python
isGroupGradientsToBeAveraged = True    # Average gradients within groups
bypass_robust = False                  # Skip robust filtering, use normalized gradients
simple_average = False                 # Simple average of group gradients  
averageOverGroupLen = True             # Average over group count vs user count
skip_filtering = False                 # Skip malicious user filtering
safeFL_approach = False                # Use SafeFL robust mechanism
median_aggregation = False             # Use median aggregation
bestGroup = False                      # Use only the highest-scoring group
```

**Aggregation Strategies**:

1. **Single Group** (lines 541-557): Direct application of the only group's gradient
2. **Simple Average** (lines 560-571): Basic averaging with optional normalization
3. **Bypass Robust** (lines 574-589): L2-normalized gradient averaging
4. **SafeFL Approach** (lines 592-628): 
   - Calculates L2 norms and median norm
   - Filters/scales gradients based on norm comparison to median
   - Uses cubic scaling for gradients above median
5. **Median Aggregation** (lines 630-634): Element-wise median across group gradients
6. **Best Group** (lines 636-652): Uses gradient from highest-scoring group only

## Main Orchestrator Function

### `heirichalFL(gradients, net, lr, f, byz, device, heirichal_params, seed)` (aggregation_rules.py:944-1046)

**Purpose**: Main entry point that orchestrates the complete HFL pipeline

**Input Validation** (lines 955-967):
- Validates required keys in `heirichal_params`
- Ensures history structure integrity
- Initializes ground truth malicious labels if missing

**Execution Flow**:
1. **Preprocessing** (lines 974-996):
   - Increments round number
   - Creates current round record
   - Applies Byzantine attacks to gradients
   - Converts gradients to parameter lists

2. **Group Management** (lines 1000-1004):
   - Simulates/maintains group structure
   - Records user membership

3. **Scoring Phase** (lines 1006-1016):
   - Aggregates gradients for scoring (with `skip_filtering=True`)
   - Scores groups based on trustworthiness
   - Updates user scores based on group performance

4. **Aggregation Phase** (lines 1018-1032):
   - Aggregates gradients for model update (with filtering)
   - Shuffles users across groups for next round
   - Handles fallback if no gradients survive filtering

5. **Model Update** (lines 1034-1041):
   - Applies robust inter-group aggregation
   - Updates global model parameters
   - Records global gradient and filtered users

6. **Cleanup** (lines 1043-1045):
   - Saves data to CSV for analysis
   - Returns updated hierarchical parameters

## Key Design Decisions

### Gradient vs Model Parameter Aggregation
All operations work on gradients rather than model parameters for consistency with the broader SAFEFL framework and MPC compatibility.

### Two-Phase Aggregation
The algorithm performs aggregation twice:
1. **Scoring Phase**: With `skip_filtering=True` to get comprehensive group behavior
2. **Update Phase**: With filtering applied for robust model updates

### Dynamic User Shuffling
Users are shuffled diagonally across groups after each round to:
- Prevent persistent group-based attacks
- Distribute trust information across the network
- Maintain group balance over time

### Filtering Strategy
- **Early Rounds** (rounds < 20): Skip filtering to allow trust scores to develop
- **Later Rounds**: Apply global filtering based on accumulated trust scores
- **Fallback Mechanisms**: Multiple strategies when all groups are filtered

### Trust Score Evolution
User trust scores accumulate over time, creating a persistent reputation system that influences future group filtering and aggregation decisions.

## Integration Points

### Main Training Loop (main.py:664-667)
```python
elif args.aggregation == "heirichalFL":
    heirichal_params = aggregation_rules.heirichalFL(
        grad_list, net, args.lr, args.nbyz, byz, device, 
        heirichal_params, seed=args.seed
    )
```

### Parameter Initialization (main.py:547-576)
The hierarchical parameters are initialized once per experiment run with appropriate defaults and data structures for tracking user membership, scores, and historical data.

### Data Persistence
Results are saved via `utils.save_data_to_csv(heirichal_params, f)` for offline analysis and visualization in the Streamlit dashboard.

## Performance Characteristics

- **Computational Complexity**: O(n²) for group scoring due to pairwise similarity calculations
- **Memory Usage**: Maintains historical data for all rounds
- **Robustness**: Multi-layered defense against Byzantine attacks through group-based filtering and robust aggregation
- **Adaptability**: Dynamic trust scores adapt to changing client behavior over time