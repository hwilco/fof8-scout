# Phase 5 Set K Talent Sample Weighting Experiment Report

Date: 2026-05-12

Source artifacts:

- [`candidate_summary.csv`](/workspaces/fof8-scout/outputs/matrices/talent_sample_weighting/candidate_summary.csv)
- [`matrix_manifest.json`](/workspaces/fof8-scout/outputs/matrices/talent_sample_weighting/matrix_manifest.json)
- [`regressor_position_slice_summary.csv`](/workspaces/fof8-scout/outputs/matrices/talent_sample_weighting/regressor_position_slice_summary.csv)

Reference context:

- [`set_i_talent_position_scope_experiment_report.md`](/workspaces/fof8-scout/docs/experiments/phase4/set_i_talent_position_scope_experiment_report.md)
- [`draft_utility_modeling_plan.md`](/workspaces/fof8-scout/docs/plans/draft_utility_modeling_plan.md)

## Objective

Evaluate whether sample weighting can improve the shared core-position control-window regressor
without changing the raw target definition.

Set K candidates:

- `K1`: unweighted baseline
- `K2`: top-25% global tail weighted
- `K3`: top-10% global tail weighted
- `K4`: top-10% plus top-5% heavier
- `K5`: within-position top-10% weighted

All candidates:

- use `Control_Window_Mean_Current_Overall` in raw space
- use CatBoost RMSE
- run on the core-position cohort excluding `LS`, `FB`, `K`, and `P`
- run in regressor-only mode

## Important Limitation

Set K is still a regressor-only matrix.

That means:

- no stitched complete-model board evaluation
- no direct downstream utility comparison
- decisions here should be treated as target-fit guidance, not final board decisions

The useful metrics are:

- `top32_target_capture_ratio`
- `top64_target_capture_ratio`
- `mean_ndcg_at_32`
- `mean_ndcg_at_64`
- RMSE / MAE as secondary diagnostics

## Aggregate Results

| Candidate | Weighting Scheme | Top32 Capture | Top64 Capture | NDCG@32 | NDCG@64 | RMSE | MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| K1 | unweighted | 0.9479 | 0.9526 | 0.6728 | 0.6779 | 6.9883 | 5.4405 |
| K2 | global top 25% | 0.9486 | 0.9522 | 0.6676 | 0.6724 | 7.0136 | 5.4730 |
| K3 | global top 10% | 0.9471 | 0.9512 | 0.6877 | 0.6928 | 7.0243 | 5.4707 |
| K4 | global top 10% + top 5% | 0.9481 | 0.9511 | 0.6572 | 0.6623 | 7.0545 | 5.4939 |
| K5 | position-group top 10% | 0.9467 | 0.9517 | 0.6702 | 0.6751 | 7.0223 | 5.4723 |

## Comparison vs Baseline

Relative to `K1`:

### `K2`: top-25% global weighting

- `top32_target_capture_ratio`: `+0.0007`
- `top64_target_capture_ratio`: `-0.0004`
- `mean_ndcg_at_32`: `-0.0052`
- `mean_ndcg_at_64`: `-0.0055`
- `rmse`: `+0.0252`
- `mae`: `+0.0324`

Interpretation:

- tiny gain in `top32` capture
- slight regression almost everywhere else

This is too small and too mixed to matter.

### `K3`: top-10% global weighting

- `top32_target_capture_ratio`: `-0.0009`
- `top64_target_capture_ratio`: `-0.0014`
- `mean_ndcg_at_32`: `+0.0149`
- `mean_ndcg_at_64`: `+0.0150`
- `rmse`: `+0.0360`
- `mae`: `+0.0302`

Interpretation:

- strongest aggregate NDCG improvement in Set K
- gives back both capture ratios
- slightly worse point error

This is the most interesting weighting variant, but it improves ranking breadth more than the
primary top-k capture objective.

### `K4`: top-10% plus top-5% heavier

- `top32_target_capture_ratio`: `+0.0002`
- `top64_target_capture_ratio`: `-0.0015`
- `mean_ndcg_at_32`: `-0.0156`
- `mean_ndcg_at_64`: `-0.0156`
- `rmse`: `+0.0662`
- `mae`: `+0.0533`

Interpretation:

- more aggressive tail weighting overshot
- materially worse NDCG
- worse point error
- no compensating capture gain large enough to justify it

This variant should be rejected.

### `K5`: within-position top-10% weighting

- `top32_target_capture_ratio`: `-0.0013`
- `top64_target_capture_ratio`: `-0.0009`
- `mean_ndcg_at_32`: `-0.0026`
- `mean_ndcg_at_64`: `-0.0028`
- `rmse`: `+0.0339`
- `mae`: `+0.0317`

Interpretation:

- no aggregate win
- position-conditional weighting did not rescue the tradeoff

This variant does not justify promotion.

## Position Diagnostics

The position-slice summary shows that weighting changes some local behavior, but not enough to
change the overall decision.

Notable slice effects:

- `K3` improves `WR` NDCG from `0.8162` to `0.8407`
- `K4` improves `RB` NDCG from `0.8271` to `0.8445`
- `K5` improves `CB` NDCG from `0.7692` to `0.7825`

But these gains come with offsetting losses:

- `K4` is materially worse on aggregate NDCG despite the `RB` slice improvement
- `K3` does not improve the aggregate top-k capture metrics that matter most here
- `K5` does not convert its local position benefits into an aggregate win

So Set K found local movement, not a new champion.

## Decision

Set K does not displace the unweighted baseline.

### Default decision

Keep `K1` as the default shared core-position control-window regressor.

Reasoning:

- best overall balance of top-k capture, NDCG, and point error
- no weighting variant produced a clear net win

### Most interesting follow-up

`K3` is the only weighting scheme that looks worth a second look.

Reasoning:

- best aggregate NDCG in Set K
- some meaningful position-slice improvements, especially `WR`
- but it still loses on the current primary selection metric family: top-k capture

That makes `K3` a research branch, not a replacement.

### Rejection decisions

Do not promote:

- `K2`: gain too small
- `K4`: aggressive weighting clearly hurts overall fit
- `K5`: position-aware weighting did not translate into aggregate improvement

## Recommendation

1. Keep the unweighted raw control-window target as the primary shared-model default.
2. If sample weighting is explored further, continue from the `K3` top-10% global weighting path.
3. Do not combine more aggressive weighting with additional architectural changes yet.
4. If `K3` is revisited, judge it in a common downstream evaluation layer rather than on regressor
   metrics alone, because its profile is "better NDCG, slightly worse capture."
