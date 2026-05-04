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

## Extend Feature Ablation And Sweep Categorical Setups

### Why this pattern exists

The Hydra Optuna sweeper + Optuna RDB storage stack cannot safely serialize list-valued
categorical choices (they can surface as OmegaConf `ListConfig`). To preserve feature
ablation sweeps with combinations, use scalar toggles and resolve them into concrete
include/exclude lists inside orchestration.

### Canonical config shape (shared across classifier/regressor)

- Base lists:
  - `include_features` (optional explicit include patterns)
  - `exclude_features` (optional explicit exclude patterns)
- Toggle-driven ablation:
  - `ablation.toggles.<name>: true|false`
  - `ablation.toggle_to_group.<name>: <group_key>`
  - `ablation.groups.<group_key>: [pattern, ...]`
  - `ablation.invalid_combinations: [[toggle_a, toggle_b], ...]`

Resolver behavior lives in:
- `fof8-ml/src/fof8_ml/orchestration/data_loader.py` (`resolve_feature_ablation_config`)
- wired in `fof8-ml/src/fof8_ml/orchestration/pipeline_runner.py`

Config source of truth now lives in:
- `pipelines/conf/ablation/default.yaml`

Both pipelines import that shared config via defaults:
- `pipelines/conf/classifier_pipeline.yaml` includes `- ablation: default`
- `pipelines/conf/regressor_pipeline.yaml` includes `- ablation: default`

This prevents drift in group definitions (`ablation.groups`) and toggle mappings
(`ablation.toggle_to_group`) between classifier and regressor runs.

The resolver:
- merges base include/exclude lists with enabled ablation groups
- validates missing mappings and invalid toggle combinations
- emits deterministic metadata for tracking (`ablation_signature`, enabled toggles)

### Struct-mode note for runtime-derived keys

Hydra/OmegaConf config is often struct-locked during runs. When adding runtime-derived keys
like `ablation_signature` or `ablation_enabled_toggles`, update config inside:

- `with open_dict(cfg): ...`

Current implementation does this in:
- `fof8-ml/src/fof8_ml/orchestration/pipeline_runner.py`

If you add additional derived metadata keys, follow the same pattern to avoid
`ConfigAttributeError: Key '<name>' is not in struct`.

### Adding new exclude or include groups

1. Add the pattern list under `ablation.groups`.
2. Add a toggle under `ablation.toggles`.
3. Map toggle to group under `ablation.toggle_to_group`.
4. Optionally add conflict rules in `ablation.invalid_combinations`.
5. If you want that dimension tuned, add the toggle as an Optuna categorical boolean.

### Fixed vs swept toggles (common pattern)

You can keep some ablations pinned while sweeping only one or two dimensions.

Example:
- fixed `true`: `no_interviewed`, `no_scout`, `no_delta`
- fixed `false`: `no_combine`
- swept: `no_position`

In shared base config (`ablation/default.yaml`), set pinned defaults:
- `ablation.toggles.no_interviewed: true`
- `ablation.toggles.no_scout: true`
- `ablation.toggles.no_delta: true`
- `ablation.toggles.no_combine: false`
- `ablation.toggles.no_position: false` (default before sweep override)

In sweep search space, only include:
- `ablation.toggles.no_position: {type: categorical, choices: [true, false]}`

Do not include pinned toggles in search space unless you want them tuned.

If classifier/regressor need different defaults, create tiny overlays such as:
- `pipelines/conf/ablation/classifier.yaml`
- `pipelines/conf/ablation/regressor.yaml`

and point each pipeline defaults entry to the appropriate ablation profile.

For include-only experiments, prefer:
- `include_features` as explicit patterns, or
- a string profile key (for example `feature_profile=baseline|light|wide`) that you resolve to
  include lists in code/config before training.

### Safe categorical sweep guidance

- Prefer scalar categorical values (`str`, `bool`, numeric) in Optuna search spaces.
- Avoid list/dict categorical values in `hydra.sweeper.search_space`.
- For structured alternatives, sweep a scalar key and resolve that key into rich config
  (feature lists, transforms, preprocessing bundles) in orchestration code.
