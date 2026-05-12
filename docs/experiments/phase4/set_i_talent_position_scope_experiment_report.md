# Phase 4 Set I Talent Position Scope Experiment Report

Date: 2026-05-12

Source artifacts:

- [`candidate_summary.csv`](/workspaces/fof8-scout/outputs/matrices/talent_position_scope/candidate_summary.csv)
- [`matrix_manifest.json`](/workspaces/fof8-scout/outputs/matrices/talent_position_scope/matrix_manifest.json)
- [`regressor_position_slice_summary.csv`](/workspaces/fof8-scout/outputs/matrices/talent_position_scope/regressor_position_slice_summary.csv)

Reference context:

- [`set_h_talent_target_standardization_experiment_report.md`](/workspaces/fof8-scout/docs/experiments/phase4/set_h_talent_target_standardization_experiment_report.md)
- [`draft_utility_modeling_plan.md`](/workspaces/fof8-scout/docs/plans/draft_utility_modeling_plan.md)

## Objective

Evaluate whether the first-pass control-window talent regressor should remain a single shared model
across positions or move toward narrower position scopes.

Set I candidates:

- `I1`: all positions
- `I2`: `QB` only
- `I3`: `RB` only
- `I4`: core positions only, excluding `LS`, `FB`, `K`, and `P`

All candidates target `Control_Window_Mean_Current_Overall` in raw space.

## Controlled Setup

Common settings across candidates:

- classifier source policy: `none`
- regressor model: CatBoost RMSE
- regressor target: `Control_Window_Mean_Current_Overall`
- split strategy: grouped universe validation
- feature ablation signature: `f637ec814dc4:no_interviewed,no_scout,no_delta,no_college`
- evaluation scope: regressor-only validation metrics

Interpretation rule for Set I:

- use this set to judge whether narrower position scopes simplify the learning problem
- do not treat these results as direct shared-board quality comparisons
- reserve final board decisions for a common downstream evaluation layer

## Important Limitation

Set I does not compare candidates on the same evaluation cohort.

- `I1` is evaluated on all eligible validation players
- `I2` is evaluated on validation `QB`s only
- `I3` is evaluated on validation `RB`s only
- `I4` is evaluated on the core-position subset only

So higher metrics for `I2` or `I3` do not automatically mean that a dedicated `QB` or `RB` model
would improve the full draft board. They do mean those narrower tasks are materially easier and may
warrant specialized modeling.

`regressor_test_draft_value_score` is `0.0` for every row because Set I was run in regressor-only
mode with no stitched complete-model evaluation.

## Validation Cohort Sizes

Reproducing the grouped-universe split against the processed feature store yields these validation
cohort sizes for non-null `Control_Window_Mean_Current_Overall` targets:

| Candidate | Validation Rows | Non-Null Target Rows |
| --- | ---: | ---: |
| I1 all positions | 50,226 | 13,097 |
| I2 QB only | 3,647 | 702 |
| I3 RB only | 3,565 | 929 |
| I4 core positions | 44,477 | 12,250 |

This matters for interpretation:

- `QB` and `RB` are not tiny one-off slices; the validation cohorts are large enough for `@32` and
  `@64` metrics to be meaningful
- they are still much narrower and more homogeneous problems than the shared all-position run

## Aggregate Results

| Candidate | Scope | Top32 Capture | Top64 Capture | NDCG@32 | NDCG@64 | RMSE | MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| I1 | all positions | 0.9465 | 0.9534 | 0.6496 | 0.6560 | 7.0727 | 5.5004 |
| I2 | `QB` only | 1.0000 | 1.0000 | 0.8940 | 0.8940 | 5.5672 | 4.1978 |
| I3 | `RB` only | 1.0000 | 1.0000 | 0.8193 | 0.8193 | 6.8223 | 5.3159 |
| I4 | core positions, no `LS/FB/K/P` | 0.9479 | 0.9526 | 0.6728 | 0.6779 | 6.9883 | 5.4405 |

## Key Comparisons

### `I4` vs `I1`: exclude fringe positions

`I4 - I1`:

- `top32_target_capture_ratio`: `+0.0015`
- `top64_target_capture_ratio`: `-0.0008`
- `mean_ndcg_at_32`: `+0.0232`
- `mean_ndcg_at_64`: `+0.0218`
- `rmse`: `-0.0844`
- `mae`: `-0.0599`

Interpretation:

- removing `LS`, `FB`, `K`, and `P` does not change the problem much
- it does modestly improve ranking quality and point error
- there is no evidence here that keeping those fringe positions helps the main shared control-window
  target

This makes a core-position shared model look defensible as the default path.

### `I2` and `I3`: single-position scopes

Relative to `I1`, both single-position runs show much stronger validation metrics:

- `QB`:
  - `top32_target_capture_ratio`: `+0.0535`
  - `top64_target_capture_ratio`: `+0.0466`
  - `mean_ndcg_at_32`: `+0.2444`
  - `rmse`: `-1.5055`
- `RB`:
  - `top32_target_capture_ratio`: `+0.0535`
  - `top64_target_capture_ratio`: `+0.0466`
  - `mean_ndcg_at_32`: `+0.1697`
  - `rmse`: `-0.2504`

Interpretation:

- the single-position tasks are substantially easier than the all-position task
- `QB` in particular looks like a strong candidate for separate treatment
- `RB` also improves materially, though less dramatically than `QB`

The perfect `1.0000` capture ratios for both `QB` and `RB` should not be over-read. They show that
the model can recover the top of those within-position validation cohorts very well, not that a
dedicated `QB` or `RB` model has already proven superior on the shared draft-board objective.

## Same-Slice Position Comparison

Set I now has the direct comparison that the aggregate table was missing: each saved regressor was
scored on the same held-out position slices.

### `QB` validation slice

| Candidate | Training Scope | NDCG@32 | RMSE | MAE |
| --- | --- | ---: | ---: | ---: |
| I1 | all positions | 0.8954 | 5.7449 | 4.3471 |
| I2 | `QB` only | 0.8940 | 5.5672 | 4.1978 |
| I3 | `RB` only | 0.7861 | 9.6781 | 8.1552 |
| I4 | core positions | 0.8810 | 5.6838 | 4.3000 |

Interpretation:

- the dedicated `QB` model is only a marginal gain over the shared/core models on the actual `QB`
  slice
- the gain is visible in RMSE and MAE, but not in NDCG
- `I1` is still the best `QB`-slice ranker by a small margin

So the dedicated `QB` path remains plausible, but the case is weaker than the aggregate Set I table
initially suggested.

### `RB` validation slice

| Candidate | Training Scope | NDCG@32 | RMSE | MAE |
| --- | --- | ---: | ---: | ---: |
| I1 | all positions | 0.8254 | 6.8477 | 5.3270 |
| I2 | `QB` only | 0.7079 | 21.0394 | 18.8580 |
| I3 | `RB` only | 0.8193 | 6.8223 | 5.3159 |
| I4 | core positions | 0.8271 | 6.8339 | 5.2965 |

Interpretation:

- the dedicated `RB` model does not beat the shared/core models on the `RB` slice
- `I4` is the strongest `RB`-slice ranker
- `I3` is only narrowly competitive on point error

That is not enough evidence to justify a separate `RB` model path.

### Cross-position generalization

The single-position specialists generalize badly outside their home slice:

- `I2` on all-position validation: `NDCG@32 = 0.3267`, `RMSE = 21.6254`
- `I3` on all-position validation: `NDCG@32 = 0.2051`, `RMSE = 16.8375`

This matters because it shows the specialist runs are genuinely narrow models, not models that
incidentally happen to perform best on their own position while staying viable elsewhere.

## Decision

Set I supports two position-scope decisions.

### 1. Excluding fringe positions is reasonable

`I4` is slightly cleaner than `I1` on the main validation metrics and does not show any meaningful
loss in top-k capture. That is enough to justify treating `LS`, `FB`, `K`, and `P` as out-of-scope
for the primary shared control-window talent model.

### 2. `QB` is still the best specialized follow-up, but not a clear architectural winner yet

The aggregate `QB`-only run looked dominant, but the same-slice comparison narrows that result:

- `I2` slightly improves `QB`-slice RMSE/MAE
- `I1` slightly leads `QB`-slice NDCG
- `I4` stays close to both

So `QB` still deserves dedicated follow-up, but Set I does not yet justify splitting production
architecture around it.

`RB` is weaker still. Once judged on the same `RB` slice, the dedicated `RB` model does not
outperform the shared/core baselines. That moves `RB` out of the "likely specialist" bucket and
into "only revisit if later experiments create a stronger reason."

## Recommendation

1. Use the core-position scope (`I4`) as the default shared-model cohort for the next control-window
   regressor experiments.
2. Keep `QB` as the first specialized-position follow-up, but frame it as a tuning path rather than
   a pre-decided model split.
3. Do not prioritize a dedicated `RB` path from Set I alone.
4. Use the same-slice diagnostics as the required standard for any future specialist claim.
5. Leave `LS`, `FB`, `K`, and `P` out of the main B3/B4 control-window path unless there is a
   separate special-teams objective.
