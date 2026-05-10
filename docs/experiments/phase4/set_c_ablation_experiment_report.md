# Phase 4 Set C Ablation Sensitivity Report

Date: 2026-05-10

Source artifacts:

- [`phase4_candidate_summary.csv`](/workspaces/fof8-scout/outputs/phase4/phase4_set_c_ablation/phase4_candidate_summary.csv)
- [`matrix_manifest.json`](/workspaces/fof8-scout/outputs/phase4/phase4_set_c_ablation/matrix_manifest.json)

## Objective

Evaluate whether the provisional Set A leader (`A2`) and fallback (`A1`) remain credible under
feature-policy changes, and determine whether the Set A decision was robust or an artifact of the
default ablation profile.

Set C candidates:

- `C1_A2`: `A2` with current default ablation
- `C2_A2`: `A2` keeping college features
- `C3_A2`: `A2` keeping scout features
- `C4_A2`: `A2` keeping delta features
- `C5_A2`: `A2` removing `Position` and keeping `Position_Group`
- `C1_A1`: `A1` with current default ablation
- `C2_A1`: `A1` keeping college features
- `C3_A1`: `A1` keeping scout features
- `C4_A1`: `A1` keeping delta features
- `C5_A1`: `A1` removing `Position` and keeping `Position_Group`

## Controlled Setup

Common settings across candidates:

- classifier target: `Economic_Success`
- classifier source policy: `train_per_candidate`
- split strategy: `grouped_universe`
- evaluation stack: complete stitched model plus cross-outcome scorecard
- elite evaluation config:
  - source column: `Career_Merit_Cap_Share`
  - quantile: `0.95`
  - scope: `position_group`
  - fallback: `global`

Candidate families:

- `A2` family:
  - regressor target: `Positive_Career_Merit_Cap_Share`
  - model: CatBoost RMSE
  - target space: `log`
- `A1` family:
  - regressor target: `Positive_Career_Merit_Cap_Share`
  - model: CatBoost Tweedie
  - target space: `raw`

Feature-policy variants:

- `C1`: default signature `f637ec814dc4:no_interviewed,no_scout,no_delta,no_college`
- `C2`: keep college features, signature `737f1286d632:no_interviewed,no_scout,no_delta`
- `C3`: keep scout features, signature `e6880be05604:no_interviewed,no_delta,no_college`
- `C4`: keep delta features, signature `d9b5d8ec5165:no_interviewed,no_scout,no_college`
- `C5`: drop `Position` but keep `Position_Group`, signature `b4a672a2f4d8:no_position,no_interviewed,no_scout,no_delta,no_college`

Important comparability note:

- The main comparison is robustness within each Set A family and whether any `A1` ablation variant
  overtakes `A2` on the stitched draft-board objective.

## Aggregate Results

### Overall ranking

Ranked by `complete_draft_value_score`:

1. `C1_A2`: `0.6978`
2. `C5_A2`: `0.6967`
3. `C2_A2`: `0.6951`
4. `C4_A2`: `0.6945`
5. `C1_A1`: `0.6941`
6. `C3_A2`: `0.6941`
7. `C2_A1`: `0.6937`
8. `C3_A1`: `0.6937`
9. `C4_A1`: `0.6933`
10. `C5_A1`: `0.6928`

Key aggregate metrics:

| Candidate | Draft Value Score | Econ NDCG@64 | Talent NDCG@64 | Longevity NDCG@64 | Elite Prec@32 | Elite Recall@64 | Bust Rate@32 | Top64 Bias | Calib Slope |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1_A2 | 0.6978 | 0.8170 | 0.4602 | 0.2782 | 0.6681 | 0.7304 | 0.0542 | -0.0233 | 1.0640 |
| C5_A2 | 0.6967 | 0.8163 | 0.4607 | 0.2759 | 0.6677 | 0.7303 | 0.0542 | -0.0228 | 1.0579 |
| C2_A2 | 0.6951 | 0.8141 | 0.4698 | 0.2802 | 0.6722 | 0.7313 | 0.0569 | -0.0228 | 1.0575 |
| C4_A2 | 0.6945 | 0.8137 | 0.4648 | 0.2689 | 0.6663 | 0.7290 | 0.0531 | -0.0234 | 1.0609 |
| C1_A1 | 0.6941 | 0.8129 | 0.4880 | 0.2781 | 0.6663 | 0.7295 | 0.0552 | -0.0132 | 1.0176 |
| C3_A2 | 0.6941 | 0.8131 | 0.4645 | 0.2820 | 0.6691 | 0.7305 | 0.0542 | -0.0248 | 1.0686 |
| C2_A1 | 0.6937 | 0.8129 | 0.4900 | 0.2871 | 0.6694 | 0.7289 | 0.0535 | -0.0138 | 1.0187 |
| C3_A1 | 0.6937 | 0.8126 | 0.4825 | 0.2800 | 0.6667 | 0.7276 | 0.0542 | -0.0148 | 1.0247 |
| C4_A1 | 0.6933 | 0.8127 | 0.4928 | 0.2867 | 0.6656 | 0.7281 | 0.0542 | -0.0145 | 1.0270 |
| C5_A1 | 0.6928 | 0.8121 | 0.4845 | 0.2789 | 0.6684 | 0.7258 | 0.0552 | -0.0137 | 1.0174 |

### Aggregate interpretation

Set C does not overturn the Set A leader.

What the matrix says clearly:

- `A2` remains the strongest stitched-board family under every tested ablation profile.
- the original `A2` default setup (`C1_A2`) is still the best overall candidate
- no `A1` ablation variant overtakes even the weaker `A2` ablation variants at the top of the Set C
  ranking

Why this matters:

- the Set A result was not a fragile artifact of the default feature policy
- `A2`'s edge survives keeping college, scout, and delta information, and survives removing the
  raw `Position` feature while retaining `Position_Group`

## Family-Level Findings

### `A2` family

Ranked within the `A2` variants:

1. `C1_A2`: default ablation, `0.6978`
2. `C5_A2`: position-group only, `0.6967`
3. `C2_A2`: keep college, `0.6951`
4. `C4_A2`: keep delta, `0.6945`
5. `C3_A2`: keep scout, `0.6941`

Interpretation:

- the best `A2` configuration is still the original default policy
- dropping raw `Position` is the closest challenger, only about `0.0010` behind the leader
- keeping college helps talent alignment (`0.4698` vs `0.4602`) and elite precision (`0.6722` vs
  `0.6681`), but it gives back enough stitched-board quality to remain secondary
- keeping scout features is the weakest `A2` variant and also the most negatively biased

### `A1` family

Ranked within the `A1` variants:

1. `C1_A1`: default ablation, `0.6941`
2. `C2_A1`: keep college, `0.6937`
3. `C3_A1`: keep scout, `0.6937`
4. `C4_A1`: keep delta, `0.6933`
5. `C5_A1`: position-group only, `0.6928`

Interpretation:

- `A1` is also most stable under the default setup
- none of the ablation relaxations produce a meaningful stitched-score improvement
- `A1` continues to be the better-calibrated family:
  - bias range roughly `-0.0132` to `-0.0148`
  - slope range roughly `1.0174` to `1.0270`
- `A1` also remains materially better on talent alignment than any `A2` variant

## Cross-Family Learnings

### What changed from Set A

Set C refined the interpretation of the `A2` versus `A1` tradeoff rather than changing the winner.

Findings:

- `A2` remains the board-quality leader
- `A1` remains the calibration and talent-alignment fallback
- the gap between `C1_A2` and `C1_A1` persists under nearby feature-policy changes instead of
  collapsing

### What did not hold up

No feature-policy tweak created a cleaner champion than `C1_A2`.

Specifically:

- keeping college did not produce enough gain to replace the default
- keeping scout did not unlock a stronger `A2` board
- keeping delta did not improve either family
- removing `Position` did not break the model, but it still did not beat the default

## Recommendation

### Champion status after Set C

Promote `A2` from provisional leader to robust Set C-cleared leader.

Reasoning:

- best stitched draft-board score in the full Set C matrix
- best candidate is still the original default-ablation configuration
- no tested ablation exposed a hidden dependency that invalidates the Set A decision

### Fallback status

Keep `A1` as the calibration-friendly fallback and reference candidate.

Reasoning:

- strongest calibration profile in the matrix
- strongest talent alignment in the matrix family that remains close enough to the leader to matter
- still useful as the main contrast case for Phase 5 champion notes

## Follow-Up Actions

1. Treat Set C as validation that the Set A decision is robust enough to continue.
2. Move to Set B overall-target comparisons to test whether a talent/overall board can beat the
   economic-target champion on draft utility.
3. If Phase 5 needs a secondary variant check later, `C5_A2` is the only ablation close enough to
   the leader to merit attention.
4. Preserve the Set C lesson in champion notes: the winning economic board does not appear to be
   propped up by raw `Position`, suppressed college columns, or suppressed scout columns alone.
