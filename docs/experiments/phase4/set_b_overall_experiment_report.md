# Phase 4 Set B Overall-Target Experiment Report

Date: 2026-05-10

Source artifacts:

- [`phase4_candidate_summary.csv`](/workspaces/fof8-scout/outputs/phase4/phase4_set_b_overall/phase4_candidate_summary.csv)
- [`matrix_manifest.json`](/workspaces/fof8-scout/outputs/phase4/phase4_set_b_overall/matrix_manifest.json)

Reference economic-family comparators:

- [`set_a_economic_experiment_report.md`](/workspaces/fof8-scout/docs/experiments/phase4/set_a_economic_experiment_report.md)
- [`set_c_ablation_experiment_report.md`](/workspaces/fof8-scout/docs/experiments/phase4/set_c_ablation_experiment_report.md)

## Objective

Evaluate whether an overall-trained draft board can challenge the economic-target champion on
draft utility, and clarify whether talent-first targets are strong enough to replace the current
economic board without an explicit positional-value adjustment layer.

Set B candidates:

- `B1`: `Peak_Overall` + CatBoost RMSE
- `B2`: `Peak_Overall` + CatBoost MAE
- `B3`: `Top3_Mean_Current_Overall` + CatBoost RMSE

## Controlled Setup

Common settings across candidates:

- classifier target: `Economic_Success`
- classifier source policy: `train_once_per_matrix`
- split strategy: `grouped_universe`
- feature ablation signature: `f637ec814dc4:no_interviewed,no_scout,no_delta,no_college`
- evaluation stack: complete stitched model plus cross-outcome scorecard
- elite evaluation config:
  - source column: `Career_Merit_Cap_Share`
  - quantile: `0.95`
  - scope: `position_group`
  - fallback: `global`

Important interpretation rule:

- do not use `complete_draft_value_score` as the main cross-family winner metric for Set B
- it remains useful only for within-Set-B comparison because it is computed against the active
  regressor target
- the cross-family decision should be made from:
  - `complete_econ_mean_ndcg_at_64`
  - `complete_bust_rate_at_32`
  - `complete_elite_precision_at_32`
  - `complete_elite_recall_at_64`
  - manual board inspection

## Aggregate Results

### Within-Set-B comparison

Ranked by `complete_econ_mean_ndcg_at_64`:

1. `B3`: `0.6891`
2. `B1`: `0.6889`
3. `B2`: `0.6880`

Key aggregate metrics:

| Candidate | Target | Draft Value Score | Econ NDCG@64 | Talent NDCG@64 | Longevity NDCG@64 | Elite Prec@32 | Elite Recall@64 | Bust Rate@32 | Top64 Bias | Calib Slope |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 | `Peak_Overall` | 0.6364 | 0.6889 | 0.6450 | 0.2802 | 0.6830 | 0.7464 | 0.0465 | -3.1451 | 0.8559 |
| B2 | `Peak_Overall` | 0.6446 | 0.6880 | 0.6551 | 0.2716 | 0.6792 | 0.7471 | 0.0469 | -3.1389 | 0.8591 |
| B3 | `Top3_Mean_Current_Overall` | 0.6091 | 0.6891 | 0.6437 | 0.2831 | 0.6833 | 0.7458 | 0.0462 | -2.9015 | 0.8435 |

### Within-Set-B interpretation

`B3` is the best Set B candidate on the intended decision surface.

Why `B3` leads within Set B:

- best economic ranking quality
- best bust rate
- best elite precision
- best longevity NDCG

Why `B3` is not dominant within Set B:

- `B1` is essentially tied on economic NDCG
- `B2` is the best talent-aligning board
- all three candidates are clustered closely on the cross-outcome metrics that matter

Candidate-specific conclusions:

- `B1`: strongest balanced `Peak_Overall` candidate; nearly tied for best economic board inside Set B
- `B2`: best talent-alignment candidate in Set B, but weaker on the economic and bust-oriented
  decision surface
- `B3`: best overall Set B reference candidate; preferred talent-board baseline going forward

## Cross-Family Findings

### Set B versus economic champion

Set B does not beat the economic-target champion.

Reference leaders from earlier experiments:

| Candidate | Family | Econ NDCG@64 | Talent NDCG@64 | Elite Prec@32 | Elite Recall@64 | Bust Rate@32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| C1_A2 | economic champion | 0.8170 | 0.4602 | 0.6681 | 0.7304 | 0.0542 |
| C1_A1 | economic fallback | 0.8129 | 0.4880 | 0.6663 | 0.7295 | 0.0552 |
| B3 | best overall-target board | 0.6891 | 0.6437 | 0.6833 | 0.7458 | 0.0462 |

Interpretation:

- Set B boards are much stronger on talent alignment than the economic family
- Set B boards are modestly stronger on bust rate and elite discovery
- Set B boards are dramatically weaker on economic ranking quality

The gap is not marginal:

- `C1_A2` vs `B3` on economic NDCG@64:
  - `0.8170` vs `0.6891`
- `C1_A1` vs `B3` on economic NDCG@64:
  - `0.8129` vs `0.6891`

This is too large to support replacing the economic-target champion with a raw overall-target
board.

### What Set B actually taught us

Set B did not kill the talent-first idea. It narrowed it.

What failed:

- raw overall-trained boards are not strong enough to replace the economic board directly

What survived:

- talent targets appear to capture useful information that the economic board underweights
- especially:
  - talent alignment
  - elite finder behavior
  - bust avoidance

That makes talent a strong candidate for a second-stage adjusted board rather than a direct
primary target replacement.

## Recommendation

### Decision on raw overall targets

Do not promote Set B candidates to primary champion status.

Reasoning:

- none of the Set B boards come close to the economic champion on the core economic ranking metric
- the talent gains are real, but not enough to justify the economic loss in a direct replacement

### Keep one Set B reference candidate

Retain `B3` as the main talent-board reference.

Reasoning:

- best Set B board on the intended cross-outcome decision surface
- most useful starting point for any talent-plus-adjustment follow-up experiment

## Follow-Up Experiment Plan

Set B supports a new branch of experimentation rather than a champion change.

Next branch:

1. Start from `B3` as the talent-board baseline.
2. Add a train-only learned positional-value adjustment layer.
3. Freeze that adjustment on holdout.
4. Compare the adjusted talent board against `C1_A2` and `C1_A1`.

The hypothesis is:

- raw talent is cleaner than economic value as a football-quality signal
- economic value still contains important cross-position utility information
- a talent target combined with an explicit positional-value mapping may recover the useful
  cross-position economics without inheriting all of the economic target's distortions

This is a new experiment, not a reinterpretation of Set B.

## Next Actions

1. Keep `C1_A2` as the primary champion candidate entering Phase 5.
2. Keep `C1_A1` as the calibration-friendly fallback.
3. Keep `B3` as the talent-board baseline for a new positional-value-adjusted experiment branch.
4. Use the next experiment cycle to test:
   - talent prediction
   - explicit positional-value adjustment
   - holdout comparison against the economic champion
