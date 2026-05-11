# Draft Utility Target Exploration

## Summary

Phase B1 exploration is complete for the first-pass draft-utility target
design.

The current data supports a separate talent-first draft-utility flow without
replacing the economic path. The main outcomes from this pass are:

- `Top3_Mean_Current_Overall` is not meaningfully zero-inflated.
- Exhibition `Current_Overall` supports a rookie-control target based on years
  1-4.
- A first empirical veteran-market rating curve is stable enough to use as a
  downstream utility reference.
- The initial control-window modeling target should be
  `Control_Window_Mean_Current_Overall`, not the discounted mean variant.

## Data Audit Findings

Source notebook: [notebooks/draft_utility_data_audit.ipynb](/workspaces/fof8-scout/notebooks/draft_utility_data_audit.ipynb)

The processed outcome frame contains `626,464` rows. For
`Top3_Mean_Current_Overall`:

- zero count: `936`
- zero rate: `0.149%`
- `p90`: `48.0`
- `p95`: `58.0`
- `p99`: `74.67`

This confirms the target is mostly continuous, with a heavy low-end mass and a
thin elite tail rather than a meaningful excess-zero regime.

On the eight-universe exploration slice, exhibition rating coverage is strong
enough to support a rookie-control target:

| Career Year | Players With Rating |
|---|---:|
| 1 | 15,929 |
| 2 | 14,996 |
| 3 | 13,562 |
| 4 | 12,095 |

Recent classes are naturally right-censored, but the data is adequate for a
years 1-4 control window with explicit completeness flags.

## Empirical Utility Prototype

Source notebook: [notebooks/draft_utility_empirical_value_model.ipynb](/workspaces/fof8-scout/notebooks/draft_utility_empirical_value_model.ipynb)

The first-pass veteran-market model used:

- eight universes;
- first twenty simulation years per universe;
- veteran seasons with `Experience >= 5`;
- clipped annual cap share as the value target.

Observed fit on the prototype slice:

- rows: `34,280`
- cap-share MAE: `0.0105`
- log-cap-share R²: `0.6261`

The resulting position multipliers are directionally sane: `QB` is highest,
`K` and `P` are lowest, and most core positions cluster near the median band.

## Control Target Decision

The open B1 question was whether the first control-window modeling target
should be the plain mean or a discounted mean. On complete years 1-4 windows
(`27,551` players), the plain mean performed slightly better against the
market-derived utility reference:

| Candidate Score | Spearman To Utility | Top-16 Capture | Top-32 Capture | Top-64 Capture |
|---|---:|---:|---:|---:|
| `Control_Window_Mean_Current_Overall` | `0.7745` | `0.9411` | `0.9493` | `0.9375` |
| `Control_Window_Discounted_Mean_Current_Overall` | `0.7716` | `0.9400` | `0.9477` | `0.9355` |
| `Grade` | n/a | `0.7781` | `0.7680` | `0.7808` |

Decision:

- Use `Control_Window_Mean_Current_Overall` as the initial supervised target in
  Phase B2/B3.
- Also materialize `Control_Window_Discounted_Mean_Current_Overall` so draft
  board formulas can apply preference-weighting later without retraining the
  base talent model.

## Implications For Phase B2

Phase B2 should add target builders for:

- `Control_Y1_Current_Overall`
- `Control_Y2_Current_Overall`
- `Control_Y3_Current_Overall`
- `Control_Y4_Current_Overall`
- `Control_Window_Mean_Current_Overall`
- `Control_Window_Discounted_Mean_Current_Overall`

Each target should preserve missingness explicitly and carry a completeness or
right-censoring signal where needed. The market-derived utility target remains a
downstream evaluation and board-construction layer for now, not the first
training target.
