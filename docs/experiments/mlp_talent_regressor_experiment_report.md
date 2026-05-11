# MLP Talent Regressor Experiment Report

## Summary

This experiment tested a first-pass sklearn MLP regressor targeting
`Top3_Mean_Current_Overall` in raw target space. The MLP was evaluated through
the same complete-model pipeline as the existing CatBoost talent regressor,
using the fixed economic classifier from the Set F CatBoost run.

The MLP is competitive with the same-target CatBoost baseline, but not clearly
better. It slightly improves aggregate draft value, talent NDCG, and economic
NDCG, while giving back elite precision, bust rate, and top-32 actual value.

## Same-Target Comparison

Comparison: `F2` sklearn MLP vs `F1` CatBoost RMSE, both targeting
`Top3_Mean_Current_Overall` with `target_space: raw`.

| Metric | CatBoost F1 | MLP F2 | Delta |
|---|---:|---:|---:|
| `complete_draft_value_score` | 0.60910 | 0.60996 | +0.00086 |
| `complete_talent_mean_ndcg_at_64` | 0.64369 | 0.64454 | +0.00085 |
| `complete_econ_mean_ndcg_at_64` | 0.68913 | 0.69939 | +0.01026 |
| `complete_longevity_mean_ndcg_at_64` | 0.28309 | 0.28495 | +0.00186 |
| `complete_elite_precision_at_32` | 0.68333 | 0.67986 | -0.00347 |
| `complete_elite_recall_at_64` | 0.74576 | 0.74677 | +0.00101 |
| `complete_bust_rate_at_32` | 0.04618 | 0.04792 | +0.00174 |
| `complete_top64_weighted_mae_normalized` | 0.13837 | 0.13833 | -0.00004 |
| `complete_top64_bias` | -2.90146 | -2.41089 | +0.49057 |
| `complete_top64_actual_value` | 3760.74 | 3761.07 | +0.33 |
| `complete_top32_actual_value` | 2099.23 | 2094.37 | -4.86 |

## Set E Context

`E1` is effectively the same CatBoost Top3/raw baseline as `F1`. `E2`, the
position-monotonic talent proxy variant, remains the strongest talent-position
candidate in this comparison group.

Relative to `E2`, the MLP:

- trails `complete_draft_value_score` by `-0.00176`;
- trails `complete_talent_mean_ndcg_at_64` by `-0.00188`;
- improves `complete_econ_mean_ndcg_at_64` by `+0.00858`;
- trails `complete_elite_precision_at_32` by `-0.00451`;
- has worse `complete_bust_rate_at_32` by `+0.00347`.

Relative to the economic Set A champion `A2`, the MLP is not competitive on
economic board utility, but it preserves the expected talent-target advantage:

- `complete_draft_value_score`: `-0.08783` behind `A2`;
- `complete_talent_mean_ndcg_at_64`: `+0.18435` ahead of `A2`;
- `complete_econ_mean_ndcg_at_64`: `-0.11761` behind `A2`;
- `complete_elite_precision_at_32`: `+0.01181` ahead of `A2`;
- `complete_bust_rate_at_32`: `-0.00625` better than `A2`.

## Position Diagnostics

The MLP's position-level results are uneven versus the CatBoost Top3/raw
baseline. The largest draft-score gains were:

| Position | Draft Score Delta |
|---|---:|
| TE | +0.01426 |
| OLB | +0.01067 |
| DT | +0.00787 |
| P | +0.00217 |

The largest drops were:

| Position | Draft Score Delta |
|---|---:|
| WR | -0.02503 |
| ILB | -0.01855 |
| CB | -0.01712 |
| RB | -0.01654 |
| S | -0.01174 |

Top-64 board mix moved only slightly. The largest share changes were:

- `C`: `+0.00625`
- `G`: `+0.00469`
- `WR`: `+0.00260`
- `CB`: `-0.00660`
- `OLB`: `-0.00347`

## Interpretation

The first-pass MLP is viable but not superior. Matching CatBoost this closely
with conservative defaults is a useful signal, especially with the improvement
in economic NDCG, but the aggregate gain is too small to justify replacing the
CatBoost talent regressor. The position-level regressions are the main concern:
the MLP appears to help some position groups while hurting several high-volume
groups.

## Recommendation

Do not replace the CatBoost Top3/raw talent regressor with this MLP yet.

Keep the MLP branch as a viable research path and run a targeted follow-up focused
on position handling. The most useful next comparisons are:

- current shared MLP vs `Position_Group`-only categorical handling;
- current shared MLP vs `no_position` with position-group retained;
- position-group-specific MLP heads or a two-stage position adjustment layer;
- a small MLP hyperparameter sweep only after position handling is clarified.

Primary result artifacts:

- `outputs/matrices/mlp_talent_regressor/F1.json`
- `outputs/matrices/mlp_talent_regressor/F2.json`
- `outputs/matrices/mlp_talent_regressor/candidate_summary.csv`
- `outputs/matrices/mlp_talent_regressor/position_group_summary.csv`
- `outputs/matrices/mlp_talent_regressor/comparison_vs_baselines.csv`
