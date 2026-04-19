# Factor Graph × Secure Aggregation Explainer (light theme)

**Deliverable:** `factor_graph_secure_aggregation_explainer.html` — single-page, light-themed HTML explainer at the repo root, alongside the existing `factor_graph_unary_priors_explainer.html`.

## Purpose

Give a one-glance mental model of how the `factorGraphs` aggregation rule operates under secure aggregation, and how the per-round posterior feeds the next round's prior. Intended for someone who has never seen the code.

## Scope — what the diagram shows

Single panel, single SVG, horizontal strip of three consecutive rounds (`t-1` → `t` → `t+1`). Each round contains the same five-stage stack:

1. **Workers inside a secure-aggregation box.** 6 worker dots, colored by current-round group membership (group A vs group B).
2. **Group sum arrows** exiting the SA box and crossing a horizontal "privacy boundary" line.
3. **SignGuard flag badges** (flag=0 / flag=1) — the group-level observations.
4. **Mini factor graph.** Two k-ary group factors (orange squares) connected by edges to their 3 binary worker variables (teal circles). Edges reshuffle each round because group membership reshuffles.
5. **Posterior row.** Six small cells in a light→dark purple scale representing `P(worker_i = malicious)`.

Between consecutive rounds, an accent-blue arrow connects the posterior row of the earlier round to the variables row of the next, labeled "posterior → prior", showing the Bayesian recursion.

A horizontal dashed line spans all three rounds as the **privacy boundary**: worker gradients sit above it (hidden from the server); group sums and all downstream stages sit below (server-visible).

Toy narrative baked into the numbers: worker `w3` is the malicious one, and its posterior grows monotonically across the three rounds as it keeps landing in flagged groups.

## Out of scope

- Windowed vs. persistent factor-graph design distinction.
- BP iteration count, temperature, log-potential math, TPR/TNR.
- Number of BP iterations or convergence indicators.
- Secure-aggregation protocol details (the dashed box is a black box).

## Style

Light theme.

| Role | Color |
|---|---|
| Page background | `#f8fafc` |
| Panel | `#ffffff` with `#e2e8f0` border |
| Accent (headings, inter-round arrows) | `#2563eb` |
| Group A worker | `#2563eb` (blue) |
| Group B worker | `#059669` (emerald) |
| Group factor (square) | `#fb923c` fill, `#c2410c` stroke |
| Worker variable (circle) | `#99f6e4` fill, `#0d9488` stroke |
| Marginals scale | `#ede9fe` (lightest) → `#5b21b6` (darkest) |
| Privacy boundary line | `#cbd5e1` dashed |
| SA box | white-grey fill, dashed grey outline |
| Text | `#0f172a` / `#64748b` for labels |

Shape grammar matches `factor_graph_unary_priors_explainer.html`: circles = variables, squares = factors.

## Layout

`viewBox="0 0 1000 500"`. Three round sub-groups via `<g transform="translate(…)">`. The privacy boundary is drawn once at SVG level, spanning all rounds. Legend and explanatory note sit below the SVG.
