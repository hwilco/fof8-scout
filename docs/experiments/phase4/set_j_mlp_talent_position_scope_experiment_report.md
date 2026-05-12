# Phase 5 Set J MLP Talent Position Scope Experiment Report

Date: 2026-05-12

Source artifacts:

- [`matrix_manifest.json`](/workspaces/fof8-scout/outputs/matrices/mlp_talent_position_scope/matrix_manifest.json)
- [`regressor_position_slice_summary.csv`](/workspaces/fof8-scout/outputs/matrices/mlp_talent_position_scope/regressor_position_slice_summary.csv)
- [`J1.json`](/workspaces/fof8-scout/outputs/matrices/mlp_talent_position_scope/J1.json)
- [`J2.json`](/workspaces/fof8-scout/outputs/matrices/mlp_talent_position_scope/J2.json)
- [`J3.json`](/workspaces/fof8-scout/outputs/matrices/mlp_talent_position_scope/J3.json)
- [`J4.json`](/workspaces/fof8-scout/outputs/matrices/mlp_talent_position_scope/J4.json)

Reference context:

- [`set_i_talent_position_scope_experiment_report.md`](/workspaces/fof8-scout/docs/experiments/phase4/set_i_talent_position_scope_experiment_report.md)
- [`mlp_talent_regressor_experiment_report.md`](/workspaces/fof8-scout/docs/experiments/mlp_talent_regressor_experiment_report.md)

## Objective

Evaluate the same position-scope question from Set I, but with the dedicated control-window sklearn
MLP regressor instead of CatBoost.

Set J candidates:

- `J1`: all positions
- `J2`: `QB` only
- `J3`: `RB` only
- `J4`: core positions only, excluding `LS`, `FB`, `K`, and `P`

All candidates target `Control_Window_Mean_Current_Overall` in raw space.

## Controlled Setup

Common settings across candidates:

- classifier source policy: `none`
- regressor model: `sklearn_mlp_regressor_control_window`
- regressor target: `Control_Window_Mean_Current_Overall`
- split strategy: grouped universe validation
- feature ablation signature: `f637ec814dc4:no_interviewed,no_scout,no_delta,no_college`
- evaluation scope: regressor-only validation metrics plus same-slice position diagnostics

## Important Limitation

Set J has the same limitation as Set I at the aggregate level:

- `J1` is evaluated on all eligible validation players
- `J2` is evaluated on validation `QB`s only
- `J3` is evaluated on validation `RB`s only
- `J4` is evaluated on the core-position subset only

So the aggregate candidate metrics mostly tell us whether a narrower scope is easier for the MLP to
fit. The decisive evidence comes from the same-slice diagnostics in
`regressor_position_slice_summary.csv`.

## Aggregate Results

| Candidate | Scope | Top32 Capture | Top64 Capture | NDCG@32 | NDCG@64 | RMSE | MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| J1 | all positions | 0.9473 | 0.9510 | 0.6341 | 0.6383 | 7.1636 | 5.5669 |
| J2 | `QB` only | 1.0000 | 1.0000 | 0.8724 | 0.8724 | 5.8651 | 4.4403 |
| J3 | `RB` only | 1.0000 | 1.0000 | 0.8220 | 0.8220 | 6.8183 | 5.2816 |
| J4 | core positions, no `LS/FB/K/P` | 0.9454 | 0.9503 | 0.6640 | 0.6663 | 7.1089 | 5.5378 |

At the aggregate level, Set J initially looks similar to Set I:

- the single-position scopes are much easier than the all-position problem
- `J4` modestly improves NDCG over `J1`
- `J1` and `J4` are the only viable shared-board candidates

## Same-Slice Position Comparison

### `QB` validation slice

| Candidate | Training Scope | NDCG@32 | RMSE | MAE |
| --- | --- | ---: | ---: | ---: |
| J1 | all positions | 0.8596 | 6.0686 | 4.5365 |
| J2 | `QB` only | 0.8724 | 5.8651 | 4.4403 |
| J3 | `RB` only | 0.6658 | 36.5931 | 34.3903 |
| J4 | core positions | 0.8536 | 6.1164 | 4.6103 |

Interpretation:

- unlike CatBoost Set I, the MLP does show a clearer `QB`-slice gain from the dedicated `QB` model
- `J2` leads both `J1` and `J4` on `QB`-slice NDCG and point error
- the gain is still moderate rather than decisive

So `QB` remains the strongest specialist candidate in the MLP family, with somewhat stronger support
than it had in Set I.

### `RB` validation slice

| Candidate | Training Scope | NDCG@32 | RMSE | MAE |
| --- | --- | ---: | ---: | ---: |
| J1 | all positions | 0.8141 | 6.8045 | 5.2822 |
| J2 | `QB` only | 0.6818 | 15.5589 | 12.9939 |
| J3 | `RB` only | 0.8220 | 6.8183 | 5.2816 |
| J4 | core positions | 0.8247 | 6.8184 | 5.2909 |

Interpretation:

- `RB` remains weak evidence for a dedicated model path
- `J3` is competitive, but it does not clearly beat the shared/core alternatives
- `J4` is still the strongest `RB`-slice ranker

This is the same qualitative result as Set I.

### Cross-position generalization

The single-position MLP specialists generalize poorly outside their home slice:

- `J2` on all-position validation: `NDCG@32 = 0.3127`, `RMSE = 16.0439`
- `J3` on all-position validation: `NDCG@32 = 0.1143`, `RMSE = 21.3640`

That failure mode is even harsher for the MLP `RB` specialist than it was for CatBoost. These are
specialist models only, not candidates for a shared board.

## Comparison vs Set I CatBoost

The useful comparison is `J1/J4` vs `I1/I4` on the same shared slices.

### Shared all-position path

`J1` vs `I1` on all-position validation:

- `NDCG@32`: `0.6341` vs `0.6496` (`-0.0156`)
- `RMSE`: `7.1636` vs `7.0727` (`+0.0909`)
- `MAE`: `5.5669` vs `5.5004` (`+0.0664`)

MLP all-positions trails the CatBoost all-position baseline on the main shared metrics.

### Shared core-position path

`J4` vs `I4` on all-position validation:

- `NDCG@32`: `0.5455` vs `0.5468` (`-0.0013`)
- `RMSE`: `10.0426` vs `9.4176` (`+0.6250`)
- `MAE`: `6.7448` vs `6.3679` (`+0.3769`)

The MLP core-position run is much closer on ranking than on point error, but it still does not beat
the CatBoost core-position path.

### Position slices

The MLP does show some local strengths:

- `J1` beats `I1` on `WR` NDCG (`0.8304` vs `0.8171`)
- `J1` beats `I1` on `TE` NDCG (`0.8785` vs `0.8616`)
- `J2` has a stronger `QB`-slice specialist case than `I2`

But the broad picture stays the same:

- CatBoost remains better or equal on the main shared-board slices
- MLP is viable as a research path, not a replacement

## Decision

Set J supports three decisions.

### 1. `J1` and `J4` are the only viable shared MLP paths

The specialist runs collapse on all-position evaluation. Any serious MLP follow-up should start from
the shared all-position or shared core-position path, not from `QB`-only or `RB`-only models.

### 2. `QB` is still the only specialist worth carrying forward

Within the MLP family, `J2` gives the cleanest specialist win:

- best `QB`-slice NDCG
- best `QB`-slice RMSE
- best `QB`-slice MAE

That is enough to keep a `QB`-specialist MLP branch alive.

### 3. `RB` is not a justified split

`J3` does not establish a clear advantage over `J1` or `J4` on the `RB` slice, and its shared-board
generalization is poor. It should not be promoted as a separate architecture path.

## Recommendation

1. Keep `J1` and `J4` as the only shared MLP configurations worth comparing against CatBoost.
2. Keep `J2` as an optional `QB`-specialist follow-up inside the MLP branch.
3. Drop `J3` from priority follow-up work.
4. Do not replace the CatBoost control-window regressor with the MLP shared path from Set J.
5. If the MLP branch continues, the next useful step is a direct `I1/I4` vs `J1/J4` shared-board
   comparison through a common downstream layer, with `J2` reserved for `QB`-slice specialist
   testing.
