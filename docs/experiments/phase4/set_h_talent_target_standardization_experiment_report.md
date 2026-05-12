# Phase 4 Set H Talent Target Standardization Experiment Report

Date: 2026-05-12

Source artifacts:

- [`candidate_summary.csv`](/workspaces/fof8-scout/outputs/matrices/talent_target_standardization/candidate_summary.csv)
- [`matrix_manifest.json`](/workspaces/fof8-scout/outputs/matrices/talent_target_standardization/matrix_manifest.json)

Reference context:

- [`set_b_overall_experiment_report.md`](/workspaces/fof8-scout/docs/experiments/phase4/set_b_overall_experiment_report.md)
- [`draft_utility_modeling_plan.md`](/workspaces/fof8-scout/docs/plans/draft_utility_modeling_plan.md)

## Objective

Evaluate whether position-conditional target standardization is a useful replacement for raw-space
 talent targets in the first-pass regressor stage.

Set H candidates:

- `H1`: `Top3_Mean_Current_Overall`
- `H2`: `Top3_Mean_Current_Overall_Pos_Z`
- `H3`: `Control_Window_Mean_Current_Overall`
- `H4`: `Control_Window_Mean_Current_Overall_Pos_Z`

## Controlled Setup

Common settings across candidates:

- classifier source policy: `none`
- regressor model: CatBoost RMSE
- split strategy: grouped universe validation
- feature ablation signature: `f637ec814dc4:no_interviewed,no_scout,no_delta,no_college`
- evaluation scope: regressor-only validation metrics

Interpretation rule for Set H:

- compare raw vs positional-z targets within each target family
- treat `mean_ndcg_at_32`, `mean_ndcg_at_64`, and top-k capture ratios as the useful signals
- do not treat RMSE/MAE magnitudes as cross-candidate comparators when target definitions differ

## Important Limitation

Set H is not a shared-outcome bakeoff.

Each candidate was evaluated against its own target column. That means:

- `H1` vs `H2` answers "which model fits its own target definition better," not "which one produces a
  better draft board in a common evaluation space"
- `H3` vs `H4` has the same limitation
- `regressor_test_draft_value_score` is `0.0` for every row because these were regressor-only matrix
  runs with no complete stitched evaluation

So this report can support a modeling-direction decision, but it cannot by itself justify replacing
the raw targets in the main board pipeline.

## Aggregate Results

### Candidate table

| Candidate | Target | Top32 Capture | Top64 Capture | NDCG@32 | NDCG@64 | RMSE | MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H1 | `Top3_Mean_Current_Overall` | 0.9297 | 0.9260 | 0.6484 | 0.6553 | 6.9997 | 4.5662 |
| H2 | `Top3_Mean_Current_Overall_Pos_Z` | 0.8895 | 0.8738 | 0.8355 | 0.8450 | 0.6707 | 0.5689 |
| H3 | `Control_Window_Mean_Current_Overall` | 0.9465 | 0.9534 | 0.6496 | 0.6560 | 7.0727 | 5.5004 |
| H4 | `Control_Window_Mean_Current_Overall_Pos_Z` | 0.8445 | 0.8062 | 0.8282 | 0.8419 | 0.7953 | 0.6155 |

### Raw vs positional-z deltas

Top3 family, `H2 - H1`:

- `top32_target_capture_ratio`: `-0.0402`
- `top64_target_capture_ratio`: `-0.0521`
- `mean_ndcg_at_32`: `+0.1870`
- `mean_ndcg_at_64`: `+0.1897`

Control-window family, `H4 - H3`:

- `top32_target_capture_ratio`: `-0.1020`
- `top64_target_capture_ratio`: `-0.1473`
- `mean_ndcg_at_32`: `+0.1785`
- `mean_ndcg_at_64`: `+0.1859`

## Interpretation

The directional result is consistent across both target families:

- positional-z targets are easier for the model to rank against their own standardized labels
- positional-z targets materially reduce top-k target capture measured in that same target space

That combination is exactly what a position-standardized target should make you expect.

Why NDCG rises:

- the standardized targets compress position-level mean and variance differences
- the model has a simpler within-position ranking problem
- ranking quality against the standardized labels therefore looks much cleaner

Why target capture falls:

- the z-score transform removes absolute level information
- high-value cross-position outliers are less dominant after normalization
- a top-k list optimized to capture raw absolute talent is no longer the same object as a top-k list
  optimized to capture within-position relative talent

The control-window family shows the same pattern, but more strongly. The drop in capture ratio is
substantial enough that the standardized control-window target looks less suitable as the default
first-pass board target.

## Decision

Do not replace the raw-space targets with positional-z targets as the primary B3 targets.

Recommended defaults remain:

- `H1`-style raw `Top3_Mean_Current_Overall` when the goal is a top-end talent proxy
- `H3`-style raw `Control_Window_Mean_Current_Overall` when the goal is the draft-utility control
  window target selected in Phase B1

Position-standardized targets remain useful as research instruments:

- they are a clean way to test whether the model is mostly struggling with cross-position scale
  differences
- they may be useful for auxiliary heads, position-specific models, or downstream diagnostics
- they are not yet supported as better shared-board targets by Set H

## Recommendation

1. Keep raw-space targets as the main talent-regressor defaults.
2. Treat positional-z targets as experimental side paths, not replacements.
3. Use Set I position-scope experiments and any future QB-only / RB-only runs to answer the actual
   structural question: whether special handling should happen in the model topology rather than in
   the target definition.
4. If positional-z targets are revisited, rerun them through a common downstream evaluation layer so
   they can be judged on shared board-quality metrics rather than self-target fit alone.
