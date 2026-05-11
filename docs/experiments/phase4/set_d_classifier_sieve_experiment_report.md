# Phase 4 Set D Classifier-Sieve Experiment Report

Date: 2026-05-11

Source artifacts:

- [`candidate_summary.csv`](/workspaces/fof8-scout/outputs/matrices/classifier_sieve_definition/candidate_summary.csv)
- [`matrix_manifest.json`](/workspaces/fof8-scout/outputs/matrices/classifier_sieve_definition/matrix_manifest.json)

Reference comparators:

- [`set_a_economic_experiment_report.md`](/workspaces/fof8-scout/docs/experiments/phase4/set_a_economic_experiment_report.md)
- [`set_c_ablation_experiment_report.md`](/workspaces/fof8-scout/docs/experiments/phase4/set_c_ablation_experiment_report.md)

## Objective

Evaluate whether changing the classifier sieve definition improves complete stitched draft-board
quality, with decisions based on complete-model outcomes rather than classifier-only metrics.

Set D candidates:

- `D1`: `Economic_Success = Career_Merit_Cap_Share > 0`
- `D2`: `Career_Games_Played >= 16`
- `D3`: `Career_Games_Played >= 48`
- `D4`: top position-group economic percentile

## Controlled Setup

Common settings across candidates:

- classifier source policy: `train_per_candidate`
- regressor target: `Positive_Career_Merit_Cap_Share`
- regressor model: CatBoost RMSE
- regressor target space: `log`
- split strategy: `grouped_universe`
- feature ablation signature: `f637ec814dc4:no_interviewed,no_scout,no_delta,no_college`
- evaluation stack: complete stitched model plus cross-outcome scorecard
- elite evaluation config:
  - source column: `Career_Merit_Cap_Share`
  - quantile: `0.95`
  - scope: `position_group`
  - fallback: `global`

Interpretation rule for Set D:

- prioritize complete-model board outcomes, not classifier standalone fit
- use `complete_draft_value_score` and `complete_econ_mean_ndcg_at_64` as primary board-quality
  selectors
- use elite and bust metrics as secondary tradeoff signals

## Aggregate Results

### Overall ranking

Ranked by `complete_draft_value_score`:

1. `D1`: `0.6978`
2. `D4`: `0.6941`
3. `D2`: `0.6839`
4. `D3`: `0.6832`

Key aggregate metrics:

| Candidate | Classifier Target | Draft Value Score | Econ NDCG@64 | Talent NDCG@64 | Longevity NDCG@64 | Elite Prec@32 | Elite Recall@64 | Bust Rate@32 | Top64 Bias | Calib Slope |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D1 | `Economic_Success` | 0.6978 | 0.8170 | 0.4602 | 0.2782 | 0.6681 | 0.7304 | 0.0542 | -0.0233 | 1.0640 |
| D2 | `Career_Games_Played_GE_16` | 0.6839 | 0.8035 | 0.4645 | 0.2700 | 0.6656 | 0.7207 | 0.0566 | -0.0143 | 1.0351 |
| D3 | `Career_Games_Played_GE_48` | 0.6832 | 0.8012 | 0.4741 | 0.2722 | 0.6681 | 0.7269 | 0.0538 | -0.0230 | 1.0722 |
| D4 | `Top_Position_Group_Economic_Percentile` | 0.6941 | 0.8134 | 0.4655 | 0.3021 | 0.6740 | 0.7458 | 0.0545 | -0.0411 | 1.1100 |

### Aggregate interpretation

`D1` remains the best default classifier-sieve definition for overall board quality.

Why `D1` leads:

- best `complete_draft_value_score`
- best `complete_econ_mean_ndcg_at_64`
- no secondary metric gap large enough to offset that advantage

What `D1` gives up relative to alternatives:

- `D4` is better at elite discovery (`elite_precision@32`, `elite_recall@64`)
- `D3` is slightly better on bust rate and top-64 weighted MAE

Why these do not overturn `D1`:

- `D2` and `D3` both lose too much on overall/economic board quality
- `D4` has materially worse bias and calibration slope despite being close in overall score

## Candidate-Specific Findings

### `D1` (`Economic_Success`)

- best aggregate board-quality profile
- strongest economic ranking signal
- balanced calibration and bust/elite behavior relative to peers

### `D2` (`Career_Games_Played >= 16`)

- clearly below `D1` on primary board metrics
- slightly less negative bias than `D1`
- not competitive as a champion sieve definition

### `D3` (`Career_Games_Played >= 48`)

- similar to `D2`: underperforms on primary board metrics
- best bust rate and best top64 weighted MAE in Set D
- still insufficient overall quality to beat `D1`

### `D4` (top position-group economic percentile)

- strongest elite finder in Set D
- close second on overall score
- worst calibration profile in Set D:
  - most negative bias (`-0.0411`)
  - steepest slope (`1.1100`)

Interpretation:

- `D4` is a viable specialist option when elite-hit emphasis is explicitly prioritized
- `D4` is not the best default because of calibration drift

## Recommendation

### Default sieve decision

Keep `D1` (`Economic_Success`) as the default classifier sieve for the primary board.

Reasoning:

- best complete-model board quality
- best economic ranking quality
- no overriding downside on secondary metrics

### Secondary/specialized option

Retain `D4` as a conditional alternate for elite-focused scenarios.

Reasoning:

- strongest elite precision and recall
- near-best overall score
- but only acceptable when the calibration/bias tradeoff is intentional

### Rejection decisions

Do not promote `D2` or `D3` as default sieve definitions.

Reasoning:

- both are materially worse than `D1` on primary board-quality outcomes
- isolated improvements (bias for `D2`, bust/MAE for `D3`) are not enough to win the Set D objective

## Follow-Up Actions

1. Lock `D1` as the classifier-sieve default for the current champion path.
2. Optionally run a focused `D1` vs `D4` sensitivity review if elite-hit rate is prioritized in final
   deployment criteria.
3. Carry the Set D result into Phase 5 champion notes: sieve changes did not displace the
   `Economic_Success` baseline.
