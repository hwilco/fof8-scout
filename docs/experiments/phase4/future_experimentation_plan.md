# Future Experimentation Plan After Set B

Date: 2026-05-10

## Current State

The Phase 4 experiments now support three clear conclusions:

1. Set A identified `A2` as the best economic-target candidate.
2. Set C showed that `A2` remains the leader under the tested ablation variants.
3. Set B showed that raw talent-target boards improve talent alignment and some risk metrics, but
   do not preserve enough economic ranking quality to replace the economic champion directly.

That means the remaining interesting branch is not "raw overall target instead of economic target."
It is:

```text
talent target + explicit positional value adjustment
```

## New Hypothesis

A cleaner talent target may be the better base predictor of football ability, while an explicit
position-aware conversion layer may be a better way to represent cross-position draft utility than
using the economic target directly as the primary regression target.

In short:

- Stage 1: predict talent
- Stage 2: convert talent into draft-board utility with a learned positional-value mapping

## Proposed Experiment Branch

### Goal

Test whether a talent-centered board with a train-only learned positional adjustment can approach
or beat the current economic champion on draft utility without giving up the talent and elite-finder
advantages seen in Set B.

### Baseline candidates

- economic champion: `C1_A2`
- economic fallback: `C1_A1`
- talent-board baseline: `B3`

### Candidate family ideas

#### E1. Simple position-group multiplier

Train:

- regressor target: `Top3_Mean_Current_Overall`

Then fit on train only:

- one multiplier per `Position_Group`

Apply on holdout:

```text
adjusted_score = predicted_talent * position_group_multiplier
```

#### E2. Position-group monotonic mapping

Train:

- regressor target: `Top3_Mean_Current_Overall`

Then fit on train only:

- monotonic mapping from predicted talent to economic utility within each position group

Apply on holdout:

- position-specific transformed score

#### E3. Small second-stage utility model

Inputs:

- predicted talent
- `Position_Group`
- optional uncertainty or classifier probability

Target:

- `Positive_Career_Merit_Cap_Share` or another draft-utility proxy

Constraint:

- all training for the adjustment layer must use train only
- no holdout leakage in multiplier or mapping estimation

## Evaluation Rules

Do not select this branch by self-target stitched score alone.

Primary comparison metrics:

- `complete_econ_mean_ndcg_at_64`
- `complete_bust_rate_at_32`
- `complete_elite_precision_at_32`
- `complete_elite_recall_at_64`
- manual board inspection

Secondary diagnostics:

- `complete_talent_mean_ndcg_at_64`
- `complete_longevity_mean_ndcg_at_64`
- `complete_top64_bias`
- `complete_top64_calibration_slope`
- position-group scorecards

## Decision Standard

This branch is interesting only if it does at least one of the following:

1. materially narrows the economic NDCG gap versus `C1_A2`
2. preserves Set B's talent / elite / bust advantages while staying economically credible
3. produces a more defensible football-facing board in manual review

If it cannot do that, the repo should keep the economic target as the primary board and treat
talent models as diagnostics or secondary signals.
