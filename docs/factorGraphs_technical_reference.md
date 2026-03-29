# FactorGraphs Aggregation Method -- Technical Reference

This document traces the complete code path of the `factorGraphs` aggregation method in SAFEFL, from the CLI entry point in `main.py` through every function call. All line numbers reference the current codebase.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Configuration and Initialization](#3-configuration-and-initialization)
4. [Per-Epoch Execution Flow](#4-per-epoch-execution-flow)
   - 4.1 [Entry Point: `factorGraphs()`](#41-entry-point-factorgraphs)
   - 4.2 [Stage 1: Flatten and Attack](#42-stage-1-flatten-and-attack)
   - 4.3 [Stage 2: Bayesian State Initialization](#43-stage-2-bayesian-state-initialization)
   - 4.4 [Stage 3: Grouping](#44-stage-3-grouping)
   - 4.5 [Stage 4: Inference and Model Update](#45-stage-4-inference-and-model-update)
5. [Observation Function (SignGuard)](#5-observation-function-signguard)
6. [Factor Construction and Likelihood Model](#6-factor-construction-and-likelihood-model)
7. [Belief Propagation](#7-belief-propagation)
8. [Weighted Aggregation and Model Update](#8-weighted-aggregation-and-model-update)
9. [CSV Logging](#9-csv-logging)
10. [Cross-Run Behavior](#10-cross-run-behavior)
11. [Data Flow Summary](#11-data-flow-summary)

---

## 1. Overview

The `factorGraphs` method is a Byzantine-robust aggregation rule for Federated Learning. It uses a probabilistic graphical model (a factor graph) to estimate the probability that each client is malicious, based on group-level observations accumulated over many training rounds.

The core idea:
- **The server cannot see individual client gradients.** It can only see the summed gradient of a group of clients.
- Each round, clients are randomly shuffled into groups. A binary anomaly detector (SignGuard) labels each group as "anomalous" or "normal".
- These binary observations become evidence in a factor graph. Over many rounds with different group compositions, belief propagation infers which specific clients are most likely malicious.
- The inferred fault probabilities are used to weight (or reject) group gradients before updating the global model.

### Key Files

| File | Role |
|------|------|
| `main.py` | CLI argument parsing, initialization, training loop |
| `aggregation_rules.py` | `factorGraphs()`, `_run_inference_and_update()`, `_apply_byzantine_attack_and_flatten()`, `groupParams()` |
| `bayesian/factor_graph.py` | `_maybe_init_bayesian_and_csv()` -- one-time Bayesian state setup |
| `bayesian/components.py` | `observation_function()`, `signguard()` -- anomaly detection on group gradients |
| `bayesian/grouping.py` | `_group_and_sum_gradients()`, `_SG_group_and_sum_gradients()` -- alternative grouping strategies (currently commented out in the main path) |

---

## 2. Architecture Diagram

```
                         main.py training loop
                                |
                                v
                    factorGraphs() [aggregation_rules.py:1500]
                                |
            +-------------------+-------------------+
            |                   |                   |
            v                   v                   v
     Stage 1: Flatten     Stage 2: Init       Stage 3: Group
     + Attack             Bayesian State      Clients
     [line 1310]          [factor_graph.py    [groupParams()
                           :63]                line 58]
                                |                   |
                                +--------+----------+
                                         |
                                         v
                          Stage 4: _run_inference_and_update() [line 1327]
                                         |
                    +--------------------+--------------------+
                    |                    |                    |
                    v                    v                    v
             Phase A:             Phase B:             Phase C:
             Observe              Build Factors        Belief Propagation
             [signguard()         [EnumFactor          [BP.run()
              components.py:9]     line 1346]           line 1382]
                                                            |
                                                            v
                                                      Phase D:
                                                      Weighted Aggregation
                                                      + Model Update
                                                      [line 1393]
                                                            |
                                                            v
                                                      Phase E:
                                                      CSV Logging
                                                      [line 1452]
```

---

## 3. Configuration and Initialization

### 3.1 CLI Arguments (`main.py:248-259`)

```
--factorGraphs_num_iters        BP message-passing iterations              default: 500
--factorGraphs_temperature      BP temperature (lower = sharper beliefs)   default: 0.1
--factorGraphs_initial_threshold  Starting P(faulty) for all users         default: 0.5
--factorGraphs_observation_method  Anomaly detector to use                 default: "binarySignguard"
--factorGraphs_likelihood_sigma   (configured but currently unused)        default: 2
--factorGraphs_true_negative_rate  P(detector clears | group is clean)     default: 0.6
--factorGraphs_true_positive_rate  P(detector flags | group has malicious) default: 0.6
--factorGraphs_shuffling_strategy  How to form groups each round           default: "random"
--factorGraphs_highProbThreshold   (for excludeHighProbUsers, currently off) default: 0.9
--factorGraphs_prob_sort_temp      Noise temperature for prob_sort strategy default: 0.1
```

### 3.2 State Dictionary (`main.py:725-753`)

On the first run (`run == 1`), a `factorGraph_params` dictionary is created. This dictionary is the **persistent state object** that is passed into `factorGraphs()` every epoch and returned with updated values. It accumulates evidence across all epochs and across multiple runs (intentional design -- see [Section 10](#10-cross-run-behavior)).

```python
factorGraph_params = {
    'factorGraphs_num_iters': ...,        # BP iterations
    'factorGraphs_temperature': ...,       # BP temperature
    'num_rounds': args.niter,              # total epochs
    'mixing_rounds': 100,                  # (used by _group_and_sum_gradients strategies)
    'initial_threshold': ...,              # initial P(faulty)
    'group_size': args.group_size,         # users per group
    'observation_method': ...,             # anomaly detector name
    'likelihood_sigma': ...,              # (unused)
    'true_negative_rate': ...,             # TNR for likelihood model
    'true_positive_rate': ...,             # TPR for likelihood model
    'shuffling_strategy': ...,             # grouping strategy
    'prob_sort_temp': ...,                 # noise for probabilistic sorting
    'excludeHighProbUsers': False,         # whether to pre-filter suspicious users
    'highProbThreshold': ...,              # threshold for pre-filtering
    'use_sg': False,                       # SignGuard-aggregator mode
    'meta_data': { ... }                   # dataset/attack metadata for logging
}
```

**Fields added dynamically during execution** (by `_maybe_init_bayesian_and_csv` and `_run_inference_and_update`):

| Field | Added When | Purpose |
|-------|-----------|---------|
| `latent_variables` | Round 0 init | `{user_id: P(faulty)}` for each user |
| `graph` | Round 0 init | The pgmax `FactorGraph` object (persists, accumulates factors) |
| `variables` | Round 0 init | pgmax `NDVarArray` -- the 20 binary random variables |
| `current_round` | Every round | Round counter, incremented after each inference |
| `factor_store` | First inference | `{frozenset(indices): EnumFactor}` -- tracks which group compositions already have factors |
| `skippedFactorsCount` | First inference | Counter for how many times a repeated group was skipped |

---

## 4. Per-Epoch Execution Flow

### 4.1 Entry Point: `factorGraphs()` (`aggregation_rules.py:1500`)

Called once per training epoch from `main.py:864-865`:

```python
factorGraph_params = aggregation_rules.factorGraphs(
    grad_list, net, args.lr, args.nbyz, byz, device, factorGraph_params
)
```

**Parameters:**
- `gradients` (`grad_list`): List of `nworkers` elements. Each element is a list of per-layer gradient tensors (one tensor per model layer). Workers `0` through `f-1` are malicious.
- `net`: The global model (a PyTorch `nn.Module`).
- `lr`: Learning rate.
- `f` (`args.nbyz`): Number of malicious clients. The first `f` clients are assumed malicious.
- `byz`: The attack function (e.g., `attacks.trim_attack`).
- `device`: CUDA/CPU device.
- `bayesian_params` (`factorGraph_params`): The persistent state dictionary.

The function executes 4 stages and returns the updated `bayesian_params`.

---

### 4.2 Stage 1: Flatten and Attack (`aggregation_rules.py:1310-1318`)

```python
def _apply_byzantine_attack_and_flatten(gradients, net, lr, f, byz, device):
    param_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    if byz == attacks.fltrust_attack:
        param_list = byz(param_list, net, lr, f, device)[:-1]
    else:
        param_list = byz(param_list, net, lr, f, device)
    return param_list
```

**What happens:**

1. **Flatten**: Each worker's per-layer gradients are concatenated into a single column vector of shape `(total_params, 1)`. If the model has layers of sizes [100, 50, 10], the result is a `(160, 1)` tensor per worker.

2. **Attack**: The attack function receives all flattened gradients. The first `f` workers are malicious -- the attack function modifies their gradients in-place or returns a modified list. For example, `trim_attack` crafts adversarial gradients designed to shift the median. The FLTrust attack is special-cased: its return value includes an extra server gradient that is stripped with `[:-1]`.

**Output**: `param_list` -- a list of `nworkers` tensors, each `(total_params, 1)`. Workers `0..f-1` now contain adversarial gradients.

---

### 4.3 Stage 2: Bayesian State Initialization (`bayesian/factor_graph.py:63-94`)

```python
num_nodes = len(param_list)
round_id, graph, variables = _maybe_init_bayesian_and_csv(bayesian_params, num_nodes)
```

**On round 0 (first epoch):**

```python
INITIAL_THRESHOLD = bayesian_params.get("initial_threshold", 0.5)
bayesian_params["latent_variables"] = {id: INITIAL_THRESHOLD for id in range(num_nodes)}
```

Every user starts with `P(faulty) = 0.5` (maximum uncertainty).

```python
variables = vgroup.NDVarArray(num_states=2, shape=(num_nodes,))
graph = fgraph.FactorGraph(variable_groups=[variables])
```

A pgmax factor graph is created with `num_nodes` binary variables. Each variable `F_i` has two states:
- State 0: user `i` is **not faulty**
- State 1: user `i` is **faulty**

The graph starts empty (no factors). Factors are added incrementally in Stage 4.

The function also initializes a CSV file at `score_function_viz/observation_scores.csv` for logging.

**On subsequent rounds:** Simply reads back `graph` and `variables` from `bayesian_params`. The graph retains all factors from previous rounds.

**Output**: `round_id` (0-indexed epoch counter), `graph` (the persistent factor graph), `variables` (the variable array).

---

### 4.4 Stage 3: Grouping (`aggregation_rules.py:58-105`)

```python
group_size = bayesian_params.get("group_size", 2)
groups, group_gradients = groupParams(param_list, group_size, isFactorGraph=True, round_id=round_id)
```

The `groupParams` function partitions users into groups and sums each group's gradients.

**Step-by-step (using 20 workers, group_size=2 as example):**

1. **Clone** all gradient tensors to avoid mutating the originals (`line 68`).

2. **Deterministic shuffle** (`line 74`): User IDs are shuffled using `random.seed(42 + round_id)`. This ensures:
   - Same round_id always produces the same grouping (reproducibility).
   - Different rounds produce different groupings (diversity of observations).

   Example for round 0: `[14, 3, 7, 0, 18, 11, 5, 2, 16, 9, 1, 12, 8, 19, 4, 17, 6, 15, 10, 13]`

3. **Chunk into groups** (`lines 83-92`): The shuffled list is sliced into consecutive chunks of `group_size`. If the last chunk is smaller than `group_size`, it is merged into the previous group (so no group is undersized).

   With 20 users and group_size=2, this produces 10 groups of exactly 2:
   ```
   Group 0: [14, 3]     Group 5: [1, 12]
   Group 1: [7, 0]      Group 6: [8, 19]
   Group 2: [18, 11]    Group 7: [4, 17]
   Group 3: [5, 2]      Group 8: [6, 15]
   Group 4: [16, 9]     Group 9: [10, 13]
   ```

4. **Sum gradients per group** (`lines 96-99`):
   ```python
   group_gradients[gid] = torch.sum(torch.stack(gradients_in_group), dim=0)
   ```
   Each group's gradient is the element-wise sum of its members' gradients. This is a `(total_params, 1)` tensor.

**Output:**
- `groups`: `{0: (0, [14, 3]), 1: (1, [7, 0]), ...}` -- maps group_id to (group_id, list_of_user_ids)
- `group_gradients`: `{0: tensor, 1: tensor, ...}` -- maps group_id to the summed gradient

**Key constraint**: The server only sees the summed gradient per group. It cannot decompose it back into individual contributions. This is the privacy model the framework operates under.

---

### 4.5 Stage 4: Inference and Model Update (`aggregation_rules.py:1327-1497`)

This is the core of the method. It has five phases described in the following sections.

---

## 5. Observation Function (SignGuard) (`bayesian/components.py:9-80`)

**Call site** (`aggregation_rules.py:1338`):
```python
observed_scores = observation_function(group_gradients, bayesian_params)
```

The `observation_function` (`components.py:171-193`) dispatches to `signguard()` when `observation_method == "binarySignguard"`:

```python
def observation_function(gradients, bayesian_params):
    method = bayesian_params.get("observation_method", "signguard")
    if method == "binarySignguard":
        return signguard(gradients)
```

### SignGuard Algorithm (`components.py:9-80`)

SignGuard is a binary anomaly detector. It takes the dictionary of group summed-gradients and classifies each group as anomalous (1) or normal (0).

**Input**: `groups_gradients` -- dict of `{group_id: summed_gradient_tensor}`.

**Step 1 -- L2 Norm Filtering** (`lines 37-49`):
```python
l2_norm = torch.stack([torch.norm(g.flatten(), p=2.0) for g in param_list])
M = torch.median(l2_norm)
for i in range(n):
    if L <= l2_norm[i] / M and l2_norm[i] / M <= R:
        S1.append(i)
```
Compute the L2 norm of each group's summed gradient. Compute the median norm `M`. A group passes if its norm ratio `norm / M` falls within `[L=0.1, R=3.0]`. Groups with abnormally large or small norms are filtered out. The passing set is `S1`.

**Step 2 -- Sign-Feature Clustering** (`lines 40-67`):
```python
sign_grads = [torch.sign(g[idx]) for g in param_list]
sign_pos  = torch.stack([grad.eq(1.0).float().mean()  for grad in sign_grads])
sign_zero = torch.stack([grad.eq(0.0).float().mean()  for grad in sign_grads])
sign_neg  = torch.stack([grad.eq(-1.0).float().mean() for grad in sign_grads])
sign_feat = torch.stack([sign_pos, sign_zero, sign_neg], dim=1)
```
For each group gradient, compute the fraction of coordinates that are positive, zero, and negative. This produces a 3-dimensional feature vector per group.

```python
cluster = KMeans(n_clusters=2, max_iter=10, random_state=seed)
labels = cluster.fit_predict(sign_feat)
```
Run KMeans (k=2) on the sign features. The assumption is that one cluster contains normal groups and the other contains anomalous groups. The **larger cluster** is assumed to be the normal one. Groups in the larger cluster form set `S2`.

**Step 3 -- Intersection** (`line 70`):
```python
S = [i for i in S1 if i in S2]
```
A group is classified as normal only if it passes **both** the norm test and the sign-clustering test.

**Output** (`lines 73-76`):
```python
isAnomalous = {
    index[i]: (0 if i in S else 1)
    for i in range(n)
}
```
Returns `{group_id: 0 or 1}` where `0 = normal`, `1 = anomalous`.

### Example

With 10 groups (5 containing a malicious user, 5 clean):
- Groups with a malicious member may have distorted norms or sign distributions, causing them to fail norm filtering or land in the minority KMeans cluster.
- The detector is imperfect -- some malicious groups may pass (false negatives) and some clean groups may fail (false positives). This is why the true_positive_rate and true_negative_rate parameters exist in the likelihood model.

---

## 6. Factor Construction and Likelihood Model (`aggregation_rules.py:1346-1379`)

For each group in the current round, the code builds a factor that encodes the likelihood of the observed score given every possible configuration of faulty/non-faulty users in that group.

### 6.1 Enumeration of Configurations

```python
for group_id, indices in groups.values():
    group_key_for_factorGraphs = frozenset(indices)
    group_variables = [variables[idx] for idx in indices]
    factor_configs = np.array(list(product([0,1], repeat=len(indices))))
```

For a group of 2 users (e.g., users [7, 0]), `factor_configs` contains all 4 possible configurations:

| Config | F_7 | F_0 | Meaning |
|--------|-----|-----|---------|
| [0, 0] | 0 | 0 | Both benign |
| [0, 1] | 0 | 1 | User 0 faulty, user 7 benign |
| [1, 0] | 1 | 0 | User 7 faulty, user 0 benign |
| [1, 1] | 1 | 1 | Both faulty |

For groups of size `k`, there are `2^k` configurations.

### 6.2 Likelihood Computation

```python
for config in factor_configs:
    obs = observed_scores[group_id]
    exp = 1 if any(config) else 0

    if obs == 1:
        p = TPR if exp == 1 else (1 - TNR)
    else:
        p = (1 - TPR) if exp == 1 else TNR
    likelihoods.append(np.log(p + 1e-10))
```

The expected observation `exp` is binary:
- `exp = 0` if all members are benign (the group *should* look normal)
- `exp = 1` if any member is faulty (the group *should* look anomalous)

The likelihood `p` is `P(observation | configuration)`, based on a confusion matrix model:

| | obs = 1 (flagged) | obs = 0 (cleared) |
|---|---|---|
| **exp = 1** (has malicious) | TPR (true positive) | 1 - TPR (false negative) |
| **exp = 0** (all benign) | 1 - TNR (false positive) | TNR (true negative) |

With defaults TPR = 0.6, TNR = 0.6:

| | obs = 1 | obs = 0 |
|---|---|---|
| **exp = 1** | 0.6 | 0.4 |
| **exp = 0** | 0.4 | 0.6 |

The likelihoods are stored as **log-potentials** (natural log), which is the format pgmax expects.

### 6.3 Factor Deduplication

```python
if group_key_for_factorGraphs not in bayesian_params["factor_store"]:
    factor = EnumFactor(factor_configs=..., variables=..., log_potentials=likelihoods)
    graph.add_factors(factor)
    bayesian_params["factor_store"][group_key_for_factorGraphs] = factor
else:
    bayesian_params['skippedFactorsCount'] += 1
```

The `factor_store` is keyed by `frozenset(indices)` -- the set of user IDs in the group, regardless of order. If the exact same group composition has been seen in a previous round, the new observation is **skipped** and the original factor's likelihoods are retained. This is an intentional design decision to avoid runtime errors with pgmax's frozen factor structure.

This means: once a group composition has been observed, only the **first** observation for that composition contributes to inference. The shuffling strategy is therefore critical -- it must produce diverse group compositions to maximize the information gained per round.

---

## 7. Belief Propagation (`aggregation_rules.py:1382-1391`)

```python
bp = BP(graph.bp_state, temperature=bayesian_params.get("factorGraphs_temperature", 0.1))
bp_arrays = bp.init()
bp_arrays = bp.run(bp_arrays, num_iters=bayesian_params.get("factorGraphs_num_iters", 100))
beliefs = bp.get_beliefs(bp_arrays)
marginals = get_marginals(beliefs)
```

### What Happens

Belief Propagation (BP) runs on the **entire accumulated factor graph** -- all factors from all previous rounds, plus any new factors added this round. BP is a message-passing algorithm that iteratively passes "messages" between variable nodes (the 20 user fault variables) and factor nodes (the group observation factors).

- **`bp.init()`**: Initializes all messages from scratch (no warm-starting from previous rounds).
- **`bp.run(num_iters=500)`**: Runs 500 iterations of max-sum message passing.
- **`temperature=0.1`**: A low temperature makes the beliefs sharper (more decisive). At temperature 0, BP becomes max-product (MAP inference). At temperature 1, it is standard sum-product.

### Output

```python
bayesian_params["latent_variables"] = {
    i: marginals[variables][i, 1] for i in range(num_nodes)
}
```

`marginals[variables]` is a `(num_nodes, 2)` array. Column 0 is `P(not faulty)`, column 1 is `P(faulty)`. The latent variables dict stores `{user_id: P(faulty)}` for each user.

### Convergence Over Rounds

- **Round 0**: Only ~10 factors (one per group). Limited evidence. Beliefs are uncertain.
- **Round 10**: ~100 factors accumulated (assuming unique compositions). Users who consistently appear in flagged groups see rising `P(faulty)`.
- **Round 50+**: Hundreds of factors. Strong evidence. Malicious users (0..f-1) should have `P(faulty)` near 1.0, honest users near 0.0.

The factor graph acts as a **memory of all past observations**. The shuffling ensures that different pairs of users are observed together over time, allowing BP to **triangulate** which specific users are malicious from group-level signals.

---

## 8. Weighted Aggregation and Model Update (`aggregation_rules.py:1393-1440`)

After inference, the group gradients are weighted by the inferred beliefs and aggregated into a global model update.

### 8.1 Weight Computation (`lines 1397-1423`)

For each group:

```python
for u_id in groups[g_id][1]:
    latent_var = bayesian_params["latent_variables"].get(u_id)
    group_weights.append(float(1 - latent_var))
```

Each user's weight is `1 - P(faulty)`. A user with `P(faulty) = 0.02` gets weight 0.98; a user with `P(faulty) = 0.9` gets weight 0.1.

```python
weight = min(group_weights)
```

The **minimum** weight in the group is used as the group's weight. This is a conservative choice: if any member of the group is suspicious, the entire group's gradient is downweighted.

### 8.2 Hard Threshold (`lines 1411-1421`)

```python
if weight < 0.95:
    weight = 0.0
    isIncludedToAverage = 1
else:
    isIncludedToAverage = 1 / len(group_weights)
```

- If `weight < 0.95` (any member has >5% chance of being faulty): the group is **completely rejected** (weight set to 0).
- If `weight >= 0.95` (all members have <5% chance of being faulty): the group is **accepted** and its summed gradient is divided by group size to produce an average.

The `0.95` threshold is hardcoded. The configurable `highProbThreshold` parameter serves a different purpose -- it controls `excludeHighProbUsers`, which pre-filters users before grouping (currently disabled with `excludeHighProbUsers: False`).

### 8.3 Accumulation (`lines 1426-1434`)

```python
weights.append(weight)
gradient *= weight
gradient *= isIncludedToAverage
global_update += gradient

...
sum_weights = sum(weights)
if sum_weights > 1e-5:
    global_update /= sum_weights
```

For accepted groups: `contribution = (summed_gradient * weight / group_size)`.
For rejected groups: `contribution = 0` (weight is 0).

The final `global_update` is divided by the sum of all weights (both accepted and rejected, though rejected weights are 0). This effectively produces a weighted average of the accepted groups' average gradients.

Note: each accepted group contributes its **average** gradient weighted equally, regardless of group size. This means the aggregation weights groups equally, not clients equally. In the typical case where all groups have the same size, this distinction does not matter.

### 8.4 Model Update (`lines 1436-1440`)

```python
if round_id > 0:
    idx = 0
    for j, (param) in enumerate(net.parameters()):
        param.add_(global_update[idx:(idx + torch.numel(param))].reshape(tuple(param.size())), alpha=-lr)
        idx += torch.numel(param)
```

Standard SGD: `param = param - lr * global_update`. The flat `global_update` vector is sliced back into per-layer chunks matching the model's parameter shapes.

**Round 0 is skipped** (`round_id > 0` check). This is a warm-up round: the observation is recorded and factors are added to the graph, but the model is not updated. This allows the Bayesian model to collect initial evidence before influencing training.

---

## 9. CSV Logging (`aggregation_rules.py:1452-1494`)

After the model update, observation scores and latent variable statistics are logged to `score_function_viz/observation_scores.csv` in append mode.

Each row records:

| Column | Description |
|--------|-------------|
| `round_id` | Current training epoch |
| `group_id` | Which group this row describes |
| `numberOfMal` | Number of malicious users in this group (ground truth, for analysis) |
| `score` | Binary observation score (0 or 1) from SignGuard |
| `avgMalScore` | Mean `P(faulty)` across all malicious users |
| `avgNormScore` | Mean `P(faulty)` across all honest users |
| `minMalScore` | Minimum `P(faulty)` among malicious users (the "hardest to detect") |
| `maxNormScore` | Maximum `P(faulty)` among honest users (the "most falsely accused") |
| `idxOfMaxNormScore` | User ID of the most falsely accused honest user |
| `idxOfMinMalScore` | User ID of the hardest-to-detect malicious user |
| `dataset`, `attack_type`, `n_byzantine`, `bias`, `n_workers` | Experiment metadata |

This CSV is used for post-hoc analysis and visualization of how well the Bayesian model separates malicious from honest users over time.

---

## 10. Cross-Run Behavior

The `factorGraph_params` dictionary is initialized only on `run == 1` (`main.py:724`). On subsequent runs (`run == 2, 3, ...`), the state from the previous run is **intentionally preserved**. This means:

- The factor graph retains all factors from previous runs.
- The latent variables carry over their beliefs from the previous run's last epoch.
- The `current_round` counter continues incrementing (it is not reset).
- The `factor_store` remembers all previously seen group compositions.

This is by design: multiple runs accumulate more evidence about user behavior, strengthening the Bayesian model's beliefs. The `groupParams` function receives `round_id` explicitly (which comes from `current_round` in `bayesian_params`), ensuring that the grouping shuffle seed is tied to the Bayesian round counter and not a separate static counter.

---

## 11. Data Flow Summary

Using a concrete example: **20 workers, 5 byzantine (IDs 0-4), group_size=2, round 10**.

```
grad_list: 20 lists of per-layer gradient tensors
    |
    v  [_apply_byzantine_attack_and_flatten]
param_list: 20 tensors of shape (total_params, 1)
    |       Workers 0-4 now contain adversarial gradients
    |
    v  [groupParams with seed 42+10]
groups:           {0: (0, [14, 3]), 1: (1, [7, 0]), ...}   -- 10 groups of 2
group_gradients:  {0: tensor, 1: tensor, ...}                -- summed gradients per group
    |
    v  [signguard]
observed_scores:  {0: 0, 1: 1, 2: 0, 3: 1, ...}            -- binary anomaly labels
    |
    v  [build EnumFactors with confusion-matrix likelihoods]
graph:            now has ~100+ factors (from rounds 0-10)
    |
    v  [BP.run(num_iters=500, temperature=0.1)]
marginals:        (20, 2) array
latent_variables: {0: 0.87, 1: 0.82, 2: 0.79, 3: 0.85, 4: 0.81,    -- malicious: high P(faulty)
                   5: 0.04, 6: 0.03, 7: 0.05, ..., 19: 0.02}        -- honest: low P(faulty)
    |
    v  [weighted aggregation]
For each group:
    weight = min(1 - P(faulty) for each member)
    Group (14,3):  weight = min(0.96, 0.97) = 0.96 >= 0.95 --> ACCEPTED
    Group (7,0):   weight = min(0.95, 0.13) = 0.13 <  0.95 --> REJECTED (user 0 is malicious)
    ...
    |
    v  [SGD update]
global_update = weighted_avg(accepted_group_averages)
net.parameters -= lr * global_update
```

The model is updated using only gradients from groups where **all members** have very low fault probability (<5%). Groups contaminated by even one suspicious user are completely excluded.
