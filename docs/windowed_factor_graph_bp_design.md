# Windowed Factor-Graph BP with Seeded Prior

**Status:** proposed redesign of the `factorGraphs` aggregation rule
**Affects:** `aggregation_rules.py:_run_inference_and_update`, `bayesian/factor_graph.py:_maybe_init_bayesian_and_csv`
**Motivation:** OOM under long training runs; noisy single-round observations on disjoint groupings; missing temporal carryover of posterior beliefs.

---

## 1. Background

### 1.1 The current aggregation rule

`factorGraphs` (`aggregation_rules.py:1462`) implements Byzantine-robust federated aggregation as Bayesian inference over a factor graph:

- **Variables.** One binary variable per worker. State `0` = benign, state `1` = malicious.
- **Group factors.** Each round, `groupParams` (`aggregation_rules.py:57`) randomly partitions the `n` workers into `n / group_size` disjoint groups. For each group, an `EnumFactor` is added that connects the group's worker variables and encodes the likelihood `P(observed_score | configuration of malicious workers in this group)` derived from the standard TPR/TNR likelihood model (`aggregation_rules.py:1360-1371`).
- **Inference.** Loopy BP (`pgmax.infer.BP`) is run for `factorGraphs_num_iters` iterations. Marginals over each variable yield a per-worker posterior `P(worker_i = malicious)`, stored in `bayesian_params["latent_variables"]`.
- **Aggregation.** Posterior probabilities are converted to scalar weights at `aggregation_rules.py:1397-1428` and used to weight each worker's gradient before averaging.

### 1.2 The two persistence choices made today

The current design persists state across rounds in two specific ways:

1. **Graph persistence.** `_maybe_init_bayesian_and_csv` (`bayesian/factor_graph.py:62`) creates the `FactorGraph` once at round 0 and reuses the same object across all subsequent rounds. New group factors are appended via `graph.add_factors(factor)` (`aggregation_rules.py:1378`); old factors are never removed.
2. **Factor cache.** A `factor_store` dict (`aggregation_rules.py:1344`) keys factors by `frozenset(indices)`. The intent appears to be reuse: if the same group of workers is sampled twice, fetch the cached factor instead of rebuilding it.

### 1.3 The latent_variables flow today

`latent_variables[i]` is updated each round from BP marginals (`aggregation_rules.py:1394`). It is then read once for weight computation (`aggregation_rules.py:1404`) and otherwise plays no role in subsequent inference. In particular, **no unary prior factors are added to the BP graph from `latent_variables`**, so each round's BP run starts from uniform messages on every variable.

---

## 2. Problems with the current design

### 2.1 Out-of-memory failure

Empirical: a CIFAR10 / ResNet18 / `factorGraphs` / `label_flipping_attack` / `nbyz=50` / grouped / `group_size=10` run failed at round 26 with `RESOURCE_EXHAUSTED: Failed to allocate request for 105.47 MiB on device ordinal 0`, traced to `bp.run` inside `pgmax/infer/bp.py:143`.

**Root cause:** the persisted graph monotonically accumulates k-ary factors. With `n = 500` workers and `group_size = 10`, each round contributes 50 new group factors. The `factor_store` cache effectively never hits, because two random partitions of 500 workers into groups of 10 produce non-overlapping `frozenset(indices)` keys with overwhelming probability (`C(500, 10) ≈ 2.45 × 10²⁰` possible 10-subsets). The runtime log confirms this: `skipped factor percentage: 0.0` is printed every round.

By round 26, the graph holds ~1300 factors, each with `2¹⁰ = 1024` log-potentials. BP message arrays scale with the factor count, and `bp.run(num_iters=100)` traces a `jax.lax.scan` whose carryout buffers grow with the message state. The 105 MB allocation that fails is one such scan-carry tensor.

### 2.2 Within-round inference is structurally degenerate

Because `groupParams` produces a **disjoint partition**, each worker variable v_i is connected to exactly one group factor per round. Variables in different groups share no factors, so the within-round factor graph decomposes into 50 completely independent connected components.

Consequences:
- BP message passing has no edges to traverse between components, so `num_iters=100` is doing redundant work per component.
- Within a single connected component (one group factor + 10 variables), and with uniform priors, the marginal posterior over each variable is symmetric across the group. BP can conclude "at least one of these workers is malicious" but cannot identify which one without asymmetric input.

### 2.3 Posterior beliefs are not carried into the next round's inference

`latent_variables` is computed by BP, used once for weighting, and ignored on the next round's BP. The system has no mechanism to say "worker v_i was suspected malicious in past rounds, so be biased that way." Instead, identification depends entirely on accumulated factors in the persistent graph, which is what causes the OOM.

The accumulated graph is the only mechanism the current code uses to carry temporal information, and that mechanism is unbounded in memory.

### 2.4 Observation noise is high in the realistic regime

The observation function emits a binary score per group. With a 10% byzantine ratio and `group_size = 10`, the expected count of malicious workers per group is ~1. A single malicious gradient contributes ~10% of the group aggregate, so the anomaly signal in the aggregate is faint. The effective true-positive rate of the observation function on 1-malicious groups is materially lower than the 0.8 default assumed in `aggregation_rules.py:1366`. A reasonable working estimate is `P(group flagged | exactly 1 malicious) ≈ 0.5–0.6`.

This means that one observation per worker per round (the current rate) carries a low log-likelihood ratio. With `P(flag | malicious in group) = 0.5` and `P(flag | no malicious) = 0.2`, a single observation produces an LLR update of only ~0.92 nats in the favorable case. That is too weak to drive a confident posterior in one shot.

---

## 3. Design goals

A redesign should achieve the following, ranked by priority:

1. **Bounded memory.** Per-round and total memory must not grow with training length.
2. **Bayesian-coherent temporal carryover.** Information from past rounds should inform current beliefs in a principled way that does not double-count evidence.
3. **Tractable noise reduction.** Multiple observations per worker should be combinable into sharper posteriors.
4. **Within-round identifiability.** BP should be able to disambiguate which worker in a group is responsible for an observed anomaly, given temporal context.
5. **Backwards-compatible interface.** The `latent_variables` output and weighting logic at `aggregation_rules.py:1397-1428` should continue to function unchanged.
6. **No new privacy assumptions.** The design must rely only on group-aggregated observations, consistent with secure aggregation.

---

## 4. Alternatives considered

### 4.1 Per-round rebuild, no priors

Discard the persistent graph and rebuild from scratch each round using only the current round's 50 group factors.

- Memory: bounded.
- Temporal carryover: none.
- Within-round identifiability: poor (symmetry within each disjoint component cannot be broken).
- Verdict: solves OOM but loses all historical information and fails goal 2 / 4.

### 4.2 Per-round rebuild + unary priors from last round's posterior

Rebuild graph each round; in addition to the 50 current-round group factors, attach a unary factor per variable holding `[log(1-p_i), log(p_i)]` where `p_i = latent_variables[i]` from the previous round.

- Memory: bounded.
- Temporal carryover: scalar summary per worker.
- Within-round identifiability: yes — priors break per-group symmetry.
- BP topology: forest of stars (each variable has degree 2: prior + group factor). Exact BP in 2 iterations.
- Verdict: clean and cheap. Sufficient for persistent-attacker models where suspicion accumulates monotonically.

### 4.3 Per-round rebuild + sliding window of K rounds of group factors, no priors

Rebuild graph each round; include all group factors from the last K rounds.

- Memory: bounded at `O(K · n / group_size)` factors.
- Temporal carryover: structural (last K rounds of co-occurrence preserved).
- Within-round identifiability: yes — variables are shared across windowed factors, so the graph becomes one connected component.
- BP topology: lightly loopy.
- Verdict: addresses noise via repeated independent observations, but loses information from rounds older than K.

### 4.4 Per-round rebuild + sliding window of K rounds + unary priors from round t-K (chosen)

Combine 4.2 and 4.3, but seed the unary priors from `latent_variables` *as it stood at round t-K* — the snapshot just outside the window. This gives a clean partition: priors carry "everything before the window," group factors carry "everything inside the window." No double-counting.

- Memory: `O(n + K · n / group_size)` factors. Bounded.
- Temporal carryover: scalar summary for older history, structural detail for recent K rounds.
- Within-round identifiability: strong.
- BP topology: lightly loopy on window overlaps.
- Verdict: chosen. Justification in §5.

### 4.5 Multiple disjoint partitions per round (rejected for now)

Run `P` independent partitions per round, giving each worker `P` group memberships per round.

- Pro: all observations from same training state, no drift.
- Con: requires `P` separate secure aggregations per round, which is the dominant cost in the production deployment.
- Verdict: rejected as primary design due to operational cost. Kept as a fallback if the windowed design proves insufficient.

---

## 5. Chosen design: windowed BP with seeded prior

### 5.1 Per-round procedure (round `t`)

1. **Pull** the cached `(observed_scores, groups)` history for rounds `max(0, t-K+1) … t-1` from `bayesian_params["history"]` (a `collections.deque(maxlen=K)`).
2. **Compute** this round's `observed_scores_t` and `groups_t` as today.
3. **Append** `(observed_scores_t, groups_t)` to `history`. If the deque was full, the popped entry is the round-`(t-K)` data — use this eviction event to seed the prior snapshot (see §5.2).
4. **Construct a fresh `FactorGraph`** with one variable per worker.
5. **Add unary priors.** For each worker `i`, attach a one-variable `EnumFactor` with log-potentials:
   ```
   [log(1 - p_i + ε), log(p_i + ε)]
   ```
   where `p_i = bayesian_params["prior_snapshot"][i]` and `ε = 1e-10` for numerical stability. The snapshot reflects `latent_variables` as of the most recent round whose factors are no longer in the window (round `t-K`, or the initial threshold if fewer than `K` rounds have elapsed).
6. **Add windowed group factors.** For each `(observed_scores, groups)` tuple in `history`, build the `EnumFactor` per group exactly as today (`aggregation_rules.py:1356-1378`) and add it to the graph. There is no factor cache.
7. **Run BP.** `bp.run(bp_arrays, num_iters=20)` (down from 100; justified in §5.4).
8. **Read marginals** into `latent_variables`.
9. **Apply weighted aggregation** (unchanged from `aggregation_rules.py:1397-1428`).

### 5.2 Prior snapshot maintenance

The prior snapshot must always reflect the posterior at round `t-K` (or initial threshold for early rounds). Procedure:

- On round 0: `prior_snapshot = {i: 0.5 for i in range(n)}`.
- After computing `latent_variables_t` at the end of each round, *before* trimming the history deque: if `len(history) == K + 1` (i.e., we're about to evict round `t-K`), copy the *previous* round's `latent_variables` snapshot — or, equivalently, maintain a separate `latent_variables_history` deque of length `K+1` and read the head when it becomes the snapshot.

Cleanest implementation: maintain `latent_variables_history` as a `deque(maxlen=K+1)` parallel to the factor history. The element at index 0 is always the snapshot to use as the prior next round.

### 5.3 State held across rounds

| Key | Type | Purpose |
|---|---|---|
| `latent_variables` | `dict[int, float]` | Current round posterior, used for weighting |
| `prior_snapshot` | `dict[int, float]` | Posterior from round `t-K`, used as unary prior at round `t` |
| `history` | `deque[(dict, dict)]` of length K | Cached `(observed_scores, groups)` for windowed factors |
| `latent_variables_history` | `deque[dict]` of length K+1 | Used to advance `prior_snapshot` as the window slides |

Removed from current design:
- `graph` — rebuilt each round.
- `factor_store` and `skippedFactorsCount` — caching by `frozenset(indices)` is dead code (never hits).

### 5.4 BP iterations

The windowed graph is loopy on overlaps: worker `v_i` appears in `K` group factors (from each round in the window), and shares edges with the other 9 workers from each of those groups. Cycles arise where two workers happen to be in the same group in two different rounds.

This is a sparse loopy graph dominated by short cycles. Empirically, loopy BP converges on such structures in 10–25 iterations. Setting `factorGraphs_num_iters = 20` is a safe default, with the option to log the per-iteration message delta and bail early if beliefs stabilize.

### 5.5 Bayesian consistency

The chosen design avoids double-counting by construction. Define:
- `E_old` = the set of all observations from rounds `1 … t-K`.
- `E_window` = the set of all observations from rounds `t-K+1 … t`.

The prior snapshot `p_i` is itself the posterior `P(v_i = malicious | E_old)` produced by BP at round `t-K`. The windowed factors encode the likelihood `P(E_window | configuration of variables)`. Under the assumption that observations in different rounds are conditionally independent given the variable states (a standard Bayesian filter assumption), the round-`t` posterior is:

```
P(v_i = malicious | E_old, E_window) ∝ P(v_i = malicious | E_old) · P(E_window | v_i = malicious, …)
                                     = p_i · likelihood from windowed factors
```

This is exactly what BP on the new graph computes: the unary prior contributes `p_i`, the k-ary factors contribute the windowed likelihood, and BP combines them multiplicatively (additively in log domain) into the posterior. No round contributes to both terms.

### 5.6 Noise reduction analysis

Suppose for a truly malicious worker:
- `P(group observed flagged | v_i in group) = q ≈ 0.5` (effective TPR on 1-malicious groups)
- `P(group observed flagged | v_i not in group) = r ≈ 0.2` (false positive rate baseline)

Per round, v_i is in exactly one group. The contribution to `v_i`'s log-odds from a round where its group is flagged is `log(q / r) ≈ +0.92`, and from an unflagged round is `log((1-q) / (1-r)) ≈ -0.47`.

For a truly malicious worker, `P(group flagged) ≈ q = 0.5`, so the expected per-round log-odds update is:
```
0.5 · (+0.92) + 0.5 · (-0.47) ≈ +0.225 nats per round
```

| K | Expected cumulative log-odds | Implied posterior probability |
|---|---|---|
| 1 | 0.225 | ~0.56 |
| 5 | 1.125 | ~0.75 |
| 10 | 2.25 | ~0.90 |
| 20 | 4.5 | ~0.99 |

Diminishing returns past K=10 in this noise regime. K=10 is selected as the default.

### 5.7 Drift consideration

Group factors in the window are constructed from log-potentials computed at the round they correspond to. The TPR/TNR likelihood model assumes a stationary observation distribution. In practice, gradient distributions drift across training, particularly in early epochs. A round-`(t-9)` factor may model a behavior pattern that no longer applies at round `t`.

Mitigations available without changing the core design:
- **Smaller K during warmup.** Use K=3 for the first 50 rounds, K=10 thereafter.
- **Time-decayed potentials.** Multiply each factor's `log_potentials` by `exp(-λ · age)` before adding to the graph. Avoids a hard cutoff and is easier to tune than `K`.

These are tunable; not part of the core proposal.

---

## 6. Implementation deltas

### 6.1 `bayesian/factor_graph.py`

Replace `_maybe_init_bayesian_and_csv` to initialize the windowed state instead of a persistent graph:

```python
from collections import deque

def _maybe_init_bayesian_and_csv(bayesian_params, num_nodes):
    if bayesian_params.get("current_round", 0) == 0:
        K = bayesian_params.get("window_size", 10)
        threshold = bayesian_params.get("initial_threshold", 0.5)

        bayesian_params["latent_variables"] = {i: threshold for i in range(num_nodes)}
        bayesian_params["prior_snapshot"]   = {i: threshold for i in range(num_nodes)}
        bayesian_params["history"]          = deque(maxlen=K)
        bayesian_params["latent_variables_history"] = deque(maxlen=K + 1)
        bayesian_params["variables"] = vgroup.NDVarArray(num_states=2, shape=(num_nodes,))

    return bayesian_params.get("current_round", 0), bayesian_params["variables"]
```

Note: returned signature drops the `graph` argument; it is now built inside `_run_inference_and_update`.

### 6.2 `aggregation_rules.py:_run_inference_and_update`

Restructure into:

1. Build `observed_scores_t = observation_function(group_gradients, bayesian_params)` and append `(observed_scores_t, groups)` to `bayesian_params["history"]`.
2. Build a fresh `FactorGraph(variable_groups=[variables])`.
3. Add 500 unary priors from `bayesian_params["prior_snapshot"]`.
4. Iterate `bayesian_params["history"]` and add k-ary group factors for every cached `(observed_scores, groups)` tuple.
5. `bp = BP(graph.bp_state, temperature=bayesian_params.get("factorGraphs_temperature", 0.1))`; `bp_arrays = bp.run(bp.init(), num_iters=bayesian_params.get("factorGraphs_num_iters", 20))`.
6. Update `latent_variables` from marginals.
7. Append the new `latent_variables` to `latent_variables_history`. If full, advance `prior_snapshot` to the head element.
8. Apply weighted aggregation (unchanged code from current `aggregation_rules.py:1397-1428`).
9. Persist run metadata via `experiment_logger` (preserve current logging at `aggregation_rules.py:1440-1457`).

### 6.3 Defaults

| Parameter | Old default | New default | Rationale |
|---|---|---|---|
| `window_size` | (n/a) | 10 | Noise/drift balance per §5.6 |
| `factorGraphs_num_iters` | 100 | 20 | Lightly loopy graph converges fast |
| `initial_threshold` | 0.5 | 0.5 | Unchanged |
| `factorGraphs_temperature` | 0.1 | 0.1 | Unchanged |
| `true_positive_rate` | 0.8 | 0.8 | Unchanged; revisit after empirical measurement |
| `true_negative_rate` | 0.8 | 0.8 | Unchanged |

### 6.4 Removals

- `bayesian_params["factor_store"]` and `bayesian_params["skippedFactorsCount"]`: no longer used. Dead code, can be deleted.
- `bayesian_params["graph"]`: graph is local to each round.

---

## 7. Memory and runtime footprint

### 7.1 Memory

| Item | Old (round 26) | New (any round) |
|---|---|---|
| Graph factors | ~1300 k-ary | 500 unary + 500 k-ary = 1000 |
| Log-potential storage | ~1300 × 1024 floats ≈ 5.3 MB | 500 × 2 + 500 × 1024 floats ≈ 2.0 MB |
| BP message buffers | scales with factors × 2 | ~10× current single-round size |
| JAX compilation cache | one entry per graph shape ≈ 26 entries | one entry (graph shape stable) |
| Persistent state | `factor_store` ~5 MB and growing | `history` deque ≈ 2 MB |

Net: memory is bounded and stable across training.

### 7.2 Runtime

BP work per round scales with `(factors × num_iters)`. Compared to the old design at round 26:

| Metric | Old (round 26) | New |
|---|---|---|
| Factors processed per BP call | ~1300 | ~1000 |
| BP iterations | 100 | 20 |
| Relative BP work | 1300 × 100 = 130,000 | 1000 × 20 = 20,000 |
| Approximate speedup | — | ~6.5× |

Compared to the old design at early rounds (round 5, ~250 factors):

| Metric | Old (round 5) | New |
|---|---|---|
| Factors processed | 250 | 1000 (window not yet full → fewer in practice) |
| BP iterations | 100 | 20 |
| Relative BP work | 25,000 | 20,000 (when full) |

So early rounds run roughly comparable to today; late rounds run materially faster.

JAX recompilation: today, each round changes the graph shape and triggers recompilation of `bp.run` (visible as ~1–3 second pauses every round). After the change, the graph shape stabilizes once the window fills (round K), so JAX compiles `bp.run` at most twice across the entire training run.

---

## 8. Validation plan

### 8.1 Functional checks

1. Run the previously-OOMing experiment (`CIFAR10/resnet18 | atk=label_flipping_attack | def=factorGraphs | grp=Y | gs=10 | nbyz=50`) for 100 rounds and confirm no OOM.
2. Confirm `latent_variables` values stabilize after round K and remain bounded in `[0, 1]`.
3. Confirm BP converges within 20 iterations by logging per-iteration max-message-delta.

### 8.2 Quality regression

Compare on existing benchmarks (Phase 0 ungrouped + grouped, all defenses, all attacks):

- Final test accuracy at round 3000 within ±1.5 percentage points of current `factorGraphs` results where current results exist.
- Best test accuracy across training: same tolerance.
- Per-worker malicious detection (precision/recall vs ground truth, computed via `experiment_logger.save_detection_metrics` artifacts): expected to improve due to noise averaging.

### 8.3 Hyperparameter sweep

After functional validation, sweep `K ∈ {3, 5, 10, 20}` on a single `(attack, dataset)` pair to confirm K=10 is near-optimal in this regime. Re-evaluate if the optimal K differs by more than 2 from the chosen default.

### 8.4 Stale-evidence sanity check

Disable the windowed factors (only unary priors remain) and confirm the system degrades to a memoryless filter as expected: per-round posteriors should match `prior_snapshot` (no new evidence). This confirms the prior path is wired correctly.

---

## 9. Risks and open questions

### 9.1 Drift in long training runs

The K-round window assumes observations from rounds `t-K+1 … t` are exchangeable for the purpose of estimating worker maliciousness. If gradient distributions drift quickly enough that round-`(t-9)` observations are misleading at round `t`, the windowed posterior will be biased toward stale evidence.

Mitigation path: implement time-decayed potentials (§5.7) as a follow-up if empirical drift effects are observed. Alternatively, reduce `K` and fall back to multiple-partitions-per-round (§4.5) as a long-term solution.

### 9.2 Observation function calibration

The TPR/TNR defaults of 0.8 / 0.8 in `aggregation_rules.py:1366-1368` are model-agnostic placeholders. They feed directly into the log-potentials and therefore into the LLR magnitudes computed by BP. If the empirical TPR on 1-malicious groups is materially below 0.8, the priors and posteriors are systematically miscalibrated.

Follow-up: measure empirical TPR/TNR per round from logged ground truth (which workers were byzantine, which groups were flagged) and adjust the likelihood model.

### 9.3 BP convergence on loopy graphs

Loopy BP can oscillate and produce non-converged beliefs, particularly at low temperature. The chosen `temperature = 0.1` is sharp. Should be monitored.

Mitigation: if oscillation appears, raise temperature to 0.3–0.5, enable damping in `pgmax.infer.BP`, or run more iterations.

### 9.4 Compatibility with `_SG_group_and_sum_gradients`

The existing code branches on `bayesian_params.get("use_sg", False)` (`aggregation_rules.py:1474-1481`) to use a different grouping function for SG experiments. The redesign should preserve this branch unchanged — the windowed BP logic operates on whatever `(groups, group_gradients)` are produced upstream and is grouping-agnostic.

---

## 10. Summary

The current `factorGraphs` aggregation rule fails OOM around round 26 because its persistent factor graph grows by 50 factors every round and never shrinks, while running 100 BP iterations over the accumulated graph. Beyond memory, the design exhibits two structural weaknesses: posterior beliefs are not fed back as priors into subsequent BP runs, and disjoint per-round groupings give each worker only one noisy observation per round.

The proposed redesign rebuilds the factor graph each round from a sliding K-round window of group factors, augmented with unary priors seeded from the posterior at round `t-K`. This bounds memory, preserves Bayesian consistency by partitioning the evidence into prior (rounds before window) and likelihood (rounds inside window), and provides K independent observations per worker for noise averaging. The BP graph remains lightly loopy, converges in ~20 iterations, and produces sharper per-worker posteriors than the current design while running roughly 6× faster per round at late training stages.

Default `K = 10` is justified by an LLR analysis of the noise regime: it lifts the posterior probability for a truly malicious worker from ~0.56 (K=1) to ~0.90, the threshold at which the existing weight rule (`weight ≥ 0.95` at `aggregation_rules.py:1411`) reliably activates.
