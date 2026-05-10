# Phase 4 Set A Economic Experiment Report

Date: 2026-05-10

Source artifacts:

- [`phase4_candidate_summary.csv`](/workspaces/fof8-scout/outputs/phase4/phase4_set_a_economic/phase4_candidate_summary.csv)
- [`phase4_position_group_summary.csv`](/workspaces/fof8-scout/outputs/phase4/phase4_set_a_economic/phase4_position_group_summary.csv)
- [`board_overlap_summary.csv`](/workspaces/fof8-scout/outputs/phase4/phase4_set_a_economic/board_overlap_summary.csv)
- [`board_rank_deltas.csv`](/workspaces/fof8-scout/outputs/phase4/phase4_set_a_economic/board_rank_deltas.csv)
- [`board_position_mix.csv`](/workspaces/fof8-scout/outputs/phase4/phase4_set_a_economic/board_position_mix.csv)

## Objective

Evaluate the first-pass Phase 4 Set A economic-target experiment matrix and identify the
provisional leading regressor target/loss configuration for the complete stitched draft board.

Set A candidates:

- `A1`: `Positive_Career_Merit_Cap_Share` + CatBoost Tweedie raw
- `A2`: `Positive_Career_Merit_Cap_Share` + CatBoost RMSE log1p
- `A3`: `Positive_Career_Merit_Cap_Share` + CatBoost MAE raw
- `A4`: `Positive_Career_Merit_Cap_Share` + CatBoost Expectile raw
- `A5`: `Positive_DPO` + CatBoost Tweedie raw

## Controlled Setup

Common settings across candidates:

- classifier target: `Economic_Success`
- classifier architecture: constant across the matrix
- feature ablation signature: `f637ec814dc4:no_interviewed,no_scout,no_delta,no_college`
- split strategy: `grouped_universe`
- evaluation stack: complete stitched model plus cross-outcome scorecard
- elite evaluation config:
  - source column: `Career_Merit_Cap_Share`
  - quantile: `0.95`
  - scope: `position_group`
  - fallback: `global`

Important comparability caveat:

- `A1` through `A4` share the same regressor target family and are directly comparable on
  `complete_draft_value_score`, `complete_mean_ndcg_at_64`, and other same-target complete metrics.
- `A5` uses `Positive_DPO` and should be treated primarily as a cross-target comparison candidate.
  Its complete metrics are less directly comparable to `A1` through `A4` on absolute scale.

## Aggregate Results

### Primary ranking among `A1` through `A4`

Ranked by `complete_draft_value_score`:

1. `A2`: `0.6978`
2. `A1`: `0.6941`
3. `A4`: `0.6908`
4. `A3`: `0.6897`

Key aggregate metrics:

| Candidate | Draft Value Score | Econ NDCG@64 | Talent NDCG@64 | Longevity NDCG@64 | Elite Prec@32 | Elite Recall@64 | Bust Rate@32 | Top64 Bias | Calib Slope |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A2 | 0.6978 | 0.8170 | 0.4602 | 0.2782 | 0.6681 | 0.7304 | 0.0542 | -0.0233 | 1.0640 |
| A1 | 0.6941 | 0.8129 | 0.4880 | 0.2781 | 0.6663 | 0.7295 | 0.0552 | -0.0132 | 1.0176 |
| A4 | 0.6908 | 0.8162 | 0.4561 | 0.2575 | 0.6646 | 0.7248 | 0.0559 | 0.0332 | 0.9048 |
| A3 | 0.6897 | 0.8075 | 0.4686 | 0.2884 | 0.6646 | 0.7253 | 0.0549 | -0.0307 | 1.0839 |
| A5 | 0.5965 | 0.8054 | 0.5214 | 0.2847 | 0.6771 | 0.7361 | 0.0521 | -1.0507 | 1.0147 |

### Aggregate interpretation

`A2` is the best same-target economic board candidate.

Why `A2` leads:

- best `complete_draft_value_score`
- best `complete_mean_ndcg_at_64`
- best economic ranking quality (`complete_econ_mean_ndcg_at_64`)
- best bust rate among `A1` through `A4`
- best elite precision and recall among `A1` through `A4`

Why `A2` is not an unquestioned winner:

- `A1` is better calibrated:
  - bias `-0.0132` vs `-0.0233`
  - slope `1.0176` vs `1.0640`
- `A1` is materially better on talent alignment:
  - talent NDCG@64 `0.4880` vs `0.4602`

Candidate-specific conclusions:

- `A1`: best balanced fallback; strongest calibration and best talent alignment within the economic family.
- `A2`: best primary same-target candidate; current provisional leader.
- `A3`: interesting for longevity but weaker overall and more negatively biased.
- `A4`: rejected on calibration and weak lower-end position behavior despite strong economic NDCG.
- `A5`: strongest cross-target talent and elite finder, but unsuitable as primary champion without a
  different evaluation framing due to severe magnitude miscalibration.

## Position-Group Findings

### `A2` versus `A1`

`A2` improves the most in:

- `DE`
- `DT`
- `S`
- `CB`
- `G`

`A1` remains better in:

- `FB`
- `TE`
- `K`
- `ILB`
- `RB`

`QB` is essentially flat between them.

Interpretation:

- `A2`'s aggregate edge is supported by several meaningful defensive/perimeter groups.
- The advantage is real but not broad dominance.
- The main cost is that `A2` gives back some quality in `TE`, `ILB`, and `RB`, plus more strongly in
  low-priority `FB`.

### Candidate shape summaries

Strongest groups by candidate:

- `A1`: `DE`, `DT`, `WR`, `G`, `TE`
- `A2`: `DE`, `DT`, `WR`, `G`, `S`
- `A3`: `DE`, `G`, `DT`, `WR`, `TE`
- `A4`: `DE`, `DT`, `WR`, `G`, `S`
- `A5`: `DE`, `FB`, `TE`, `DT`, `G`

Weakest groups by candidate:

- `A1`: `P`, `LS`, `K`, `RB`, `QB`
- `A2`: `P`, `LS`, `K`, `FB`, `RB`
- `A3`: `LS`, `P`, `K`, `RB`, `FB`
- `A4`: `P`, `LS`, `K`, `FB`, `QB`
- `A5`: `P`, `LS`, `K`, `RB`, `C`

Key rejection signals from the position-group summary:

- `A4` is structurally weak in `P`, `K`, `FB`, and `QB` and remains miscalibrated.
- `A5` is broadly unstable in specialist and several line groups, consistent with its aggregate
  scale-misalignment problem.

## Board-Diff Findings

### Board similarity

Most similar pair:

- `A1` vs `A2`
  - Top-32 overlap: `95.24%`
  - Top-64 overlap: `95.92%`

Least similar pair:

- `A3` vs `A4`
  - Top-32 overlap: `89.41%`
  - Top-64 overlap: `90.26%`

Interpretation:

- `A1` and `A2` are competing on margin and ordering, not on radically different player sets.
- `A3` and `A4` change board composition more, but neither is strong enough to justify promotion.

### Rank-delta behavior

The largest `A1` vs `A2` rank deltas are concentrated in lower-ranked tail players, often where
`A1` retained a small nonzero complete score and `A2` clipped the same player effectively to zero.

Interpretation:

- the biggest pairwise deltas are not concentrated at the top of the board
- this strengthens the view that the `A1` vs `A2` decision is a refinement choice, not a major
  strategic fork

## Recommendation

### Provisional winner

Adopt `A2` as the provisional Set A leader.

Reasoning:

- best same-target stitched score
- best economic ranking quality
- best same-target elite and bust results
- position-group results support a real, though modest, edge over `A1`

### Fallback candidate

Keep `A1` as the primary fallback / calibration-friendly reference.

Reasoning:

- best calibration profile in the economic family
- best talent alignment among `A1` through `A4`
- nearly identical top-of-board composition to `A2`

### Rejection decisions

Reject `A4` as a champion candidate because:

- positive top-64 bias
- under-dispersed calibration slope
- weak lower-end position behavior

Reject `A3` as a champion candidate because:

- lower same-target board quality than `A1`/`A2`
- stronger negative bias
- no enough compensating gain outside longevity

Do not promote `A5` to primary champion because:

- magnitude calibration is unusable (`complete_top64_bias = -1.0507`)
- same-target complete metrics are not directly comparable to the economic family
- despite this, retain `A5` as an important comparative reference because it is strongest on:
  - talent NDCG@64
  - elite precision/recall
  - bust rate

## Follow-Up Actions

1. Carry `A2` and `A1` forward into Set C ablation sensitivity.
2. Keep `A5` as a secondary comparison target during Phase 5 writeup.
3. Document explicitly in champion selection notes that `A5` changed the modeling narrative but did
   not clear the calibration bar for primary use.
4. If needed later, evaluate whether a DPO-based secondary board or ensemble signal is useful,
   separate from the primary expected-value board.
