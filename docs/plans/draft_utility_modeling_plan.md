# Draft Utility Modeling Plan

## Summary
Build a separate draft-utility modeling flow for FOF8 draft-board support. This flow should predict interpretable talent and rookie-control outcomes, then produce configurable position-adjusted utility columns for draft-board use.

This does **not** replace the existing economic / zero-inflated modeling path. The economic targets (`Positive_Career_Merit_Cap_Share`, `Positive_DPO`, related classifier/regressor stitching, and existing matrices) remain intact as a separate capability. The new flow should live beside it.

## Important Note on Implementation
Follow architectural best practices, use polars over pandas where possible, ensure ruff and pyright checks pass (only necessary before commits). Prefer following proper encapsulation, reuse over repetition, and other best practices, using existing implementations and patterns where it makes sense. Create proper module-level docstrings, create method docstring for public methods, use well structured type-hinting, and make sure to update READMEs or documentation when necessary.

## Motivation
The economic target family is useful, but it is not necessarily the best direct objective for playing FOF8. The user-facing draft board should surface football utility signals:

- long-term talent upside;
- rookie-contract usefulness;
- elite probability;
- bust / low-outcome risk;
- position-adjusted value.

Recent distribution checks show `Top3_Mean_Current_Overall` is not meaningfully zero-inflated:

- zero rate is about `0.15%`;
- most players have a positive observed rating target;
- the actual challenge is a heavy low-end mass with a sparse elite tail.

So the new talent flow should use bounded continuous talent modeling and rank/tail-sensitive evaluation, not zero-inflated regression.

## Non-Goals
- Do not delete or rewrite existing economic target logic.
- Do not remove the current classifier/regressor complete-model path.
- Do not force all draft value into one opaque target.
- Do not make notebooks the source of truth for target generation.
- Do not hard-code final positional preferences inside model training.

## Proposed Capability Boundary
Keep three flows distinct:

1. **Economic Flow**
   - Existing zero-inflated/economic target path.
   - Targets include `Positive_Career_Merit_Cap_Share`, `Career_Merit_Cap_Share`, `Positive_DPO`.
   - Current classifier/regressor stitching remains valid here.

2. **Talent Flow**
   - Predict raw football/talent quantities.
   - Targets include `Top3_Mean_Current_Overall`, `Peak_Overall`, and rookie-control overall summaries.
   - Models should be selected with talent ranking and tail metrics.

3. **Draft Utility Flow**
   - Converts talent predictions into user-facing board columns.
   - Applies nonlinear value curves, replacement levels, elite probabilities, and positional multipliers.
   - Should be configurable without retraining the base talent models.

## Target Columns To Add
Add stable target builders in `fof8-core`, likely under:

```text
fof8-core/src/fof8_core/targets/draft_utility.py
```

First-pass target columns:

```text
Control_Y1_Current_Overall
Control_Y2_Current_Overall
Control_Y3_Current_Overall
Control_Y4_Current_Overall
Control_Window_Mean_Current_Overall
Control_Window_Discounted_Mean_Current_Overall
```

Existing target columns to keep using:

```text
Top3_Mean_Current_Overall
Peak_Overall
Career_Games_Played
```

Candidate derived target columns:

```text
Top3_Rating_Utility
Control_Window_Rating_Utility
Position_Adjusted_Top3_Utility
Position_Adjusted_Control_Utility
Elite_70_Flag
Elite_80_Flag
Replacement_Level_Flag
```

The derived utility columns can come after the control-window targets are validated.

## Target Construction Rules
- Use only post-draft/post-season rating files for targets, never for predictive features.
- Use `player_ratings_season_*.csv` exhibition `Current_Overall` for consistency with the existing `Top3_Mean_Current_Overall` builder.
- Do not require `Future_Overall`; the current loader drops `Future_*` columns.
- Preserve all rookies where possible. Missing observed ratings should be handled explicitly rather than silently becoming a modeling assumption.
- Right-censor recent draft classes when they do not have enough years to observe a full control window.

## Empirical Parameters
Estimate these from data where possible:

### Rating Value Curve
Estimate how rating maps to value using veteran market behavior:

```text
Annual_Cap_Share ~ f(Current_Overall, Position, Experience)
```

Use veteran seasons, not rookie-contract seasons, to reduce wage-scale distortion.

Outputs:

```text
Estimated_Rating_Value
Rating_Value_Curve_By_Position
```

### Replacement Level
Estimate by position using one or more:

- active-roster rating distribution;
- starter-ish players by `S_Games_Started`;
- veteran market inflection point;
- percentile among meaningful contributors.

Keep the chosen replacement method configurable.

### Position Multipliers
Start with empirical market-implied multipliers:

```text
Position_Multiplier =
  estimated value at fixed rating for position
  /
  median estimated value at fixed rating
```

Allow hand overrides because FOF8 gameplay preferences may differ from the market.

### Rookie-Control Discounts
These are partly user preference. Start with a documented default:

```text
Y1: 1.00
Y2: 0.95
Y3: 0.85
Y4: 0.75
```

Keep them configurable.

## Hand-Set Parameters
Some parameters are preferences, not discoverable truths:

- how much to value rookie-contract years versus full career;
- how aggressively to reward elite upside;
- how much to penalize bust risk;
- manual position overrides;
- whether late bloomers matter.

These should live in explicit config, not hidden in notebooks.

Suggested config location:

```text
pipelines/conf/draft_utility/default.yaml
```

or, if kept in core:

```text
fof8-core/src/fof8_core/targets/draft_value_config.py
```

## Modeling Plan
Train separate models for interpretable outputs instead of one opaque target.

First models:

```text
Top3_Mean_Current_Overall regressor
Control_Window_Mean_Current_Overall regressor
```

Follow-up models:

```text
Control_Y1_Current_Overall regressor
Control_Y2_Current_Overall regressor
Control_Y3_Current_Overall regressor
Control_Y4_Current_Overall regressor
Elite_70_Flag classifier
Elite_80_Flag classifier
Replacement_Level_Flag classifier
```

Model families:

- Start with CatBoost as the baseline.
- Keep sklearn MLP as an experimental architecture, but do not make it the default until it beats CatBoost on downstream board metrics.

## Evaluation Metrics
For talent regressors, use:

```text
RMSE
MAE
mean_ndcg_at_32 by draft class
mean_ndcg_at_64 by draft class
top32_actual_target_value
top64_actual_target_value
top32_target_capture_ratio
top64_target_capture_ratio
```

For elite classifiers, use:

```text
ROC AUC
PR AUC
precision_at_32
recall_at_64
lift_at_32
```

For draft-utility board evaluation, prefer:

```text
top32_utility_capture_ratio
top64_utility_capture_ratio
elite_recall_at_64
bust_rate_at_32
```

Avoid selecting talent models only by RMSE. The high-end tail is rare:

- `Top3_Mean_Current_Overall >= 70`: about `1.76%`;
- `>= 80`: about `0.47%`;
- `>= 90`: about `0.05%`.

## User-Facing Draft Board Outputs
The eventual draft-board export should expose raw predictions and derived utility columns separately.

Raw prediction columns:

```text
Pred_Top3_Mean_Current_Overall
Pred_Control_Window_Mean_Current_Overall
Pred_Elite_70_Probability
Pred_Elite_80_Probability
Pred_Replacement_Level_Probability
```

Derived board columns:

```text
Pred_Top3_Rating_Utility
Pred_Control_Window_Rating_Utility
Pred_Position_Adjusted_Top3_Utility
Pred_Position_Adjusted_Control_Utility
Pred_Upside_Adjusted_Control_Utility
```

This lets the user sort by different drafting lenses:

- best long-term talent;
- best rookie-contract contributor;
- best elite upside;
- safest contributor;
- best position-adjusted value.

## Repository Organization
Exploration:

```text
notebooks/draft_utility_data_audit.ipynb
notebooks/draft_utility_empirical_value_model.ipynb
notebooks/draft_utility_model_result_analysis.ipynb
```

Core target builders:

```text
fof8-core/src/fof8_core/targets/talent.py
fof8-core/src/fof8_core/targets/draft_utility.py
```

ML target configs:

```text
pipelines/conf/target/talent_top3.yaml
pipelines/conf/target/talent_control_window.yaml
pipelines/conf/target/draft_utility.yaml
```

Model configs / sweeps:

```text
pipelines/conf/experiment/catboost_top3_talent_regressor_sweep.yaml
pipelines/conf/experiment/catboost_control_window_regressor_sweep.yaml
pipelines/conf/experiment/mlp_top3_talent_regressor_sweep.yaml
```

Experiment groups:

```text
pipelines/conf/matrix/set_g_talent_targets.yaml
pipelines/conf/matrix/set_h_draft_utility.yaml
```

Documentation:

```text
docs/plans/draft_utility_modeling_plan.md
docs/experiments/draft_utility_target_exploration.md
docs/experiments/draft_utility_model_results.md
```

## Testing Plan
Add core tests for:

- control-year extraction;
- control-window mean and discounted mean;
- right-censoring behavior;
- missing rating behavior;
- no leakage into feature columns;
- position multiplier config resolution;
- utility formula reproducibility.

Add ML tests for:

- target config resolves expected active target column;
- new target columns are excluded from features;
- utility metrics compute expected top-k capture ratios;
- draft-board export includes raw prediction and derived utility columns.

## Implementation Phases
### Phase 1: Exploration
- Finish the two draft-utility notebooks.
- Validate target distributions.
- Estimate first empirical rating value curve.
- Decide whether initial control target is mean or discounted mean.

### Phase 2: Target Builders
- Add `draft_utility.py` target builder in `fof8-core`.
- Add control-window target columns to processed outcomes.
- Add tests.

### Phase 3: Baseline Talent Models
- Train CatBoost on `Top3_Mean_Current_Overall`.
- Train CatBoost on `Control_Window_Mean_Current_Overall`.
- Evaluate with rank/top-k/tail metrics.

### Phase 4: Board Utility Metrics
- Add top-k utility capture metrics.
- Evaluate draft-board quality using predicted talent columns.
- Compare against existing economic complete-model outputs without replacing them.

### Phase 5: Draft Board Export
- Export raw predicted talent columns.
- Export configurable utility-derived columns.
- Document how to sort/use them during drafting.

## Open Questions
- Should control window be years 1-4 or 1-5?
- Should missing/unobserved ratings become zero, null, or a separate survival target?
- Should `LS`, `K`, and `P` be modeled in the same flow or separated?
- Should position multipliers be empirical-only, hand-set-only, or empirical with overrides?
- Should elite probability be trained as separate classifiers or derived from regressor prediction distributions?
