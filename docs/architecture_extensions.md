# Architecture And Extensions

## Package Responsibilities

- `fof8-core`: domain loading (`FOF8Loader`), feature generation (`fof8_core.features`), and target builders (`fof8_core.targets`).
- `fof8-ml`: dataset construction/cache/schema (`fof8_ml.data`), model wrappers/registry (`fof8_ml.models`), and training orchestration (`fof8_ml.orchestration`).
- `fof8-gen`: Windows GUI automation and snapshot export workflows.
- `pipelines/`: DVC/Hydra entrypoints that wire config into core/ml modules.

## Add A Target

1. Create a new builder in `fof8-core/src/fof8_core/targets/` with signature `fn(loader: FOF8Loader) -> pl.DataFrame`.
2. Register it in `fof8-core/src/fof8_core/targets/registry.py` (either via `@register_target("name")` or direct `TARGET_REGISTRY["name"] = fn` in builtin registration).
3. Select that target in pipeline config (`pipelines/conf/target/*.yaml`) so loaders resolve it with `get_target(...)`.
4. Add or update tests under `fof8-core/tests/` for output schema and edge cases.

## Add A Model

1. Add a model config in `pipelines/conf/model/*.yaml` with a stable `name`.
2. Register `(stage, name)` in `fof8-ml/src/fof8_ml/models/registry.py` using `register_model(stage, key, family, builder)`.
3. Ensure the wrapper constructor supports the config parameters.
4. Run classifier/regressor entrypoints with the model override and add/update tests in `fof8-ml/tests/`.

## Add A Feature Group

1. Add feature logic to `fof8-core/src/fof8_core/features/` (new module or `draft_class.py` helper).
2. Compose it inside `get_draft_class(...)` in `draft_class.py`.
3. If the new columns are position-specific, update `fof8-core/src/fof8_core/features/constants.py` and `position_masks.py`.
4. Validate downstream training compatibility:
   - rebuild `features.parquet` (`uv run python pipelines/process_features.py`)
   - retrain (`uv run python pipelines/train_classifier.py` / `train_regressor.py`)
   - confirm `feature_schema.json` captures the new columns.
