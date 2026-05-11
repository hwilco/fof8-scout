# MLP Talent Regressor Experiment Plan

## Summary

Create a first-pass sklearn MLP regressor targeting `Top3_Mean_Current_Overall`.
The experiment compares the MLP directly against the existing CatBoost RMSE
talent baseline using the same target, split, classifier setup, and
complete-model evaluation path.

## Key Changes

- Implement a dedicated sklearn MLP regressor wrapper using
  `sklearn.neural_network.MLPRegressor`, rather than reusing the current
  Tweedie/Gamma sklearn wrapper.
- Register a new model key: `sklearn_mlp_regressor`.
- Add `pipelines/conf/model/sklearn_mlp_regressor.yaml` with conservative
  first-pass defaults: `[128, 64]` hidden layers, `relu`, `adam`, `alpha=0.0001`,
  `learning_rate_init=0.001`, `max_iter=500`, early stopping, `0.15`
  validation fraction, and `25` no-change patience.
- Add a compact matrix config comparing:
  - CatBoost RMSE baseline on `Top3_Mean_Current_Overall`, raw target space.
  - sklearn MLP on `Top3_Mean_Current_Overall`, raw target space.

## Feature And Preprocessing Policy

- Keep `mask_positional_features: true`; the position-aware nulling remains valid
  domain structure.
- Do not collapse masked rating nulls directly to zero before giving them to the
  MLP.
- For the MLP wrapper, use a dedicated preprocessing path:
  - one-hot encode categoricals with full one-hot encoding, not `drop_first=True`;
  - preserve training-time dummy column order and align validation/test columns
    to it;
  - add missingness indicators for numeric columns with nulls, especially masked
    `Mean_*` and related position skill columns;
  - impute numeric nulls after indicators are created with training-set medians;
  - scale final dense features with `StandardScaler`.
- Drop college features for this experiment using the existing `no_college: true`
  behavior:
  - exclude `College`;
  - exclude `College_*`.
- Keep `Position_Group` for the first pass because the MLP needs position
  archetype context.
- Keep `Position` for the first baseline unless a follow-up ablation
  intentionally tests `no_position: true`.

## Evaluation

- Use `Top3_Mean_Current_Overall` as the regressor target with
  `target_space: raw`.
- Keep the existing economic classifier target for the complete-model stitched
  evaluation so the only first-pass change is regressor architecture.
- Select using existing validation/holdout metrics:
  - regressor RMSE/MAE on raw Top3 talent;
  - talent ranking metrics from the cross-outcome scorecard;
  - complete-model draft-board metrics.
- Treat this as architecture feasibility, not final tuning. Add MLP
  hyperparameter search only if the baseline MLP is competitive.

## Tests

- Add model factory/registry tests proving `sklearn_mlp_regressor` resolves
  correctly and all model configs remain registered.
- Add MLP wrapper tests covering:
  - mixed numeric/categorical Polars input;
  - full one-hot category alignment across train and inference;
  - numeric missing indicators;
  - median imputation and scaling;
  - raw-scale predictions with no log conversion.
- Add matrix config tests confirming both candidates resolve and apply
  `Top3_Mean_Current_Overall` with `target_space: raw`.

## Assumptions

- Target is `Top3_Mean_Current_Overall`, not `Peak_Overall`.
- Backend is sklearn MLP, not PyTorch.
- College is dropped because prior experiments showed little value.
- Position-aware masking stays enabled, but MLP preprocessing must distinguish
  "not applicable" from a real low score.
