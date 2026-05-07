import fnmatch
import hashlib
import json
import os
from collections.abc import Iterable
from datetime import date, datetime
from typing import Any, Dict, List, Optional, cast

import polars as pl
from fof8_core.features.position_masks import apply_position_mask
from fof8_core.loader import FOF8Loader as FOF8Loader
from fof8_core.targets.draft_outcomes import DRAFT_OUTCOME_LEAKAGE_COLUMNS
from omegaconf import DictConfig

from fof8_ml.orchestration.pipeline_types import PreparedData, TimelineInfo

# Module-level cache for data persistence across trials in the same process
_GLOBAL_DATA_CACHE: Dict[str, Any] = {
    "X_train": None,
    "y_cls": None,
    "y_reg": None,
    "meta_train": None,
    "outcomes_train": None,
    "last_cfg_hash": None,
}


def _get_data_cache_cfg_hash(data_cfg: Dict[str, Any]) -> str:
    """Create a deterministic hash for cache keying across processes."""
    serialized = json.dumps(data_cfg, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _coerce_str_list(raw: object) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if not isinstance(raw, Iterable):
        return [str(raw)]
    return [str(v) for v in list(raw)]


def _resolve_outcome_scorecard_columns(cfg: DictConfig, available_columns: list[str]) -> list[str]:
    scorecard_cfg = cfg.target.get("outcome_scorecard")
    if scorecard_cfg is None:
        return []

    required_columns = _coerce_str_list(scorecard_cfg.get("columns"))
    optional_columns = _coerce_str_list(scorecard_cfg.get("optional_columns"))

    missing_required = sorted(col for col in required_columns if col not in available_columns)
    if missing_required:
        raise ValueError(
            "Processed features are missing required outcome_scorecard columns "
            f"{missing_required}. Re-run feature processing or update the target config."
        )

    resolved_columns = required_columns + [
        col for col in optional_columns if col in available_columns
    ]
    return list(dict.fromkeys(resolved_columns))


def _validate_active_target_columns(cfg: DictConfig, available_columns: list[str]) -> None:
    active_target_cols = [
        str(cfg.target.classifier_sieve.target_col),
        str(cfg.target.regressor_intensity.target_col),
    ]
    missing_active_targets = sorted(
        col for col in active_target_cols if col not in available_columns
    )
    if missing_active_targets:
        raise ValueError(
            "Processed features are missing configured active target columns "
            f"{missing_active_targets}. Re-run feature processing or update the target config."
        )


def resolve_feature_ablation_config(cfg: DictConfig) -> dict[str, Any]:
    """Resolve include/exclude feature lists from base lists + optional ablation toggles."""
    include_features = _coerce_str_list(cfg.get("include_features"))
    exclude_features = _coerce_str_list(cfg.get("exclude_features"))
    enabled_toggles: List[str] = []

    ablation_cfg = cfg.get("ablation")
    if ablation_cfg:
        toggles = cast(dict[str, Any], ablation_cfg.get("toggles", {}))
        groups = cast(dict[str, Any], ablation_cfg.get("groups", {}))
        toggle_to_group = cast(dict[str, Any], ablation_cfg.get("toggle_to_group", {}))
        invalid_combinations = cast(list[Any], ablation_cfg.get("invalid_combinations", []))

        enabled_toggles = [k for k, v in toggles.items() if bool(v)]
        for toggle_name in enabled_toggles:
            if toggle_name not in toggle_to_group:
                raise ValueError(
                    f"ablation.toggle_to_group is missing mapping for enabled toggle "
                    f"'{toggle_name}'"
                )
            group_name = str(toggle_to_group[toggle_name])
            if group_name not in groups:
                raise ValueError(
                    f"ablation.groups is missing group '{group_name}' "
                    f"mapped from toggle '{toggle_name}'"
                )
            exclude_features.extend(_coerce_str_list(groups[group_name]))

        enabled_set = set(enabled_toggles)
        for combo in invalid_combinations:
            combo_list = [str(v) for v in list(combo)]
            if combo_list and set(combo_list).issubset(enabled_set):
                raise ValueError(
                    f"Invalid ablation toggle combination enabled: {combo_list}. "
                    "Adjust ablation.toggles or ablation.invalid_combinations."
                )

    include_features = list(dict.fromkeys(include_features))
    exclude_features = list(dict.fromkeys(exclude_features))

    signature_payload = {
        "include": sorted(include_features),
        "exclude": sorted(exclude_features),
        "toggles": sorted(enabled_toggles),
    }
    signature_hash = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    signature = f"{signature_hash}:{','.join(enabled_toggles) if enabled_toggles else 'none'}"

    return {
        "include_features": include_features or None,
        "exclude_features": exclude_features or None,
        "enabled_toggles": enabled_toggles,
        "signature": signature,
    }


def _resolve_configured_leagues(cfg: DictConfig) -> list[str]:
    league_names = _coerce_str_list(cfg.data.get("league_names"))
    if league_names:
        return league_names

    legacy_name = cfg.data.get("league_name")
    if legacy_name:
        return [str(legacy_name)]

    league_glob = cfg.data.get("league_glob")
    if league_glob:
        return [str(league_glob)]

    return []


def _split_count(total: int, pct: float) -> int:
    count = int(total * pct)
    if pct > 0 and total > 1:
        count = max(1, count)
    return min(count, max(0, total - 1))


def _coerce_int(value: object, *, context: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    if value is None:
        raise ValueError(f"Missing integer value for {context}")
    if isinstance(value, (date, datetime)):
        raise TypeError(f"Expected integer-like value for {context}, got {type(value).__name__}")
    raise TypeError(f"Expected integer-like value for {context}, got {type(value).__name__}")


def _coerce_years(values: Iterable[object], *, context: str) -> list[int]:
    return [_coerce_int(value, context=context) for value in values]


def _global_year_range(per_universe: dict[str, dict[str, Any]], key: str) -> list[int]:
    ranges = [v[key] for v in per_universe.values() if v[key] is not None and v[key] != [0, 0]]
    starts = [r[0] for r in ranges]
    ends = [r[1] for r in ranges]
    if not starts or not ends:
        return [0, 0]
    return [min(starts), max(ends)]


def _eligible_by_universe(
    df: pl.DataFrame, right_censor_buffer: int
) -> tuple[pl.DataFrame, dict[str, dict[str, Any]]]:
    frames: list[pl.DataFrame] = []
    per_universe: dict[str, dict[str, Any]] = {}

    for universe in sorted(str(v) for v in df.get_column("Universe").unique().to_list()):
        universe_df = df.filter(pl.col("Universe") == universe)
        initial_year = _coerce_int(
            universe_df.get_column("Year").min(), context=f"{universe} initial year"
        )
        final_year = _coerce_int(
            universe_df.get_column("Year").max(), context=f"{universe} final year"
        )
        valid_start_year = initial_year
        valid_end_year = final_year - right_censor_buffer
        eligible = universe_df.filter(pl.col("Year") <= valid_end_year)
        if len(eligible) == 0:
            raise ValueError(
                f"No eligible rows remain for universe {universe!r} after applying "
                f"right_censor_buffer={right_censor_buffer}."
            )

        frames.append(eligible)
        per_universe[universe] = {
            "initial_year": initial_year,
            "final_sim_year": final_year,
            "valid_start_year": valid_start_year,
            "valid_end_year": valid_end_year,
            "train_year_range": None,
            "test_year_range": None,
        }

    return pl.concat(frames, how="vertical_relaxed"), per_universe


def _split_chronological(
    df: pl.DataFrame, per_universe: dict[str, dict[str, Any]], test_split_pct: float
) -> tuple[pl.DataFrame, pl.DataFrame]:
    train_frames: list[pl.DataFrame] = []
    test_frames: list[pl.DataFrame] = []

    for universe, info in per_universe.items():
        universe_df = df.filter(pl.col("Universe") == universe)
        years = sorted(
            _coerce_years(
                universe_df.get_column("Year").unique().to_list(),
                context=f"{universe} split years",
            )
        )
        test_years_count = _split_count(len(years), test_split_pct)
        test_years = set(years[-test_years_count:]) if test_years_count else set()
        train_years = [y for y in years if y not in test_years]

        if train_years:
            train_frames.append(universe_df.filter(pl.col("Year").is_in(train_years)))
            info["train_year_range"] = [min(train_years), max(train_years)]
        if test_years:
            test_frames.append(universe_df.filter(pl.col("Year").is_in(sorted(test_years))))
            info["test_year_range"] = [min(test_years), max(test_years)]
        else:
            info["test_year_range"] = [0, 0]

    train_df = pl.concat(train_frames, how="vertical_relaxed") if train_frames else df.head(0)
    test_df = pl.concat(test_frames, how="vertical_relaxed") if test_frames else df.head(0)
    return train_df, test_df


def _split_random(
    df: pl.DataFrame, cfg: DictConfig
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, dict[str, Any]]]:
    seed = int(cfg.split.get("seed", cfg.get("seed", 42)))
    unit = str(cfg.split.get("unit", "draft_class"))
    test_split_pct = float(cfg.split.test_split_pct)

    stratify_by = set(_coerce_str_list(cfg.split.get("stratify_by")))
    stratify_universe = "Universe" in stratify_by

    if unit == "row":
        indexed = df.with_row_index("__split_row_id")
        if stratify_universe:
            test_parts = []
            for idx, universe in enumerate(
                sorted(indexed.get_column("Universe").unique().to_list())
            ):
                universe_df = indexed.filter(pl.col("Universe") == universe)
                test_count = _split_count(len(universe_df), test_split_pct)
                if test_count:
                    test_parts.append(
                        universe_df.sample(n=test_count, shuffle=True, seed=seed + idx)
                    )
            test_df = (
                pl.concat(test_parts, how="vertical_relaxed") if test_parts else indexed.head(0)
            )
        else:
            test_count = _split_count(len(indexed), test_split_pct)
            test_df = (
                indexed.sample(n=test_count, shuffle=True, seed=seed)
                if test_count
                else indexed.head(0)
            )
        train_df = indexed.join(test_df.select("__split_row_id"), on="__split_row_id", how="anti")
        train_df = train_df.drop("__split_row_id")
        test_df = test_df.drop("__split_row_id")
    elif unit == "draft_class":
        groups = df.select(["Universe", "Year"]).unique().sort(["Universe", "Year"])
        if stratify_universe:
            test_parts = []
            for idx, universe in enumerate(
                sorted(groups.get_column("Universe").unique().to_list())
            ):
                universe_groups = groups.filter(pl.col("Universe") == universe)
                test_count = _split_count(len(universe_groups), test_split_pct)
                if test_count:
                    test_parts.append(
                        universe_groups.sample(n=test_count, shuffle=True, seed=seed + idx)
                    )
            test_groups = (
                pl.concat(test_parts, how="vertical_relaxed") if test_parts else groups.head(0)
            )
        else:
            test_count = _split_count(len(groups), test_split_pct)
            test_groups = (
                groups.sample(n=test_count, shuffle=True, seed=seed)
                if test_count
                else groups.head(0)
            )
        test_df = df.join(test_groups, on=["Universe", "Year"], how="inner")
        train_df = df.join(test_groups, on=["Universe", "Year"], how="anti")
    else:
        raise ValueError("cfg.split.unit must be either 'row' or 'draft_class' for random split")

    per_universe: dict[str, dict[str, Any]] = {}
    for universe in sorted(str(v) for v in df.get_column("Universe").unique().to_list()):
        universe_df = df.filter(pl.col("Universe") == universe)
        train_years = _coerce_years(
            train_df.filter(pl.col("Universe") == universe).get_column("Year").to_list(),
            context=f"{universe} train years",
        )
        test_years = _coerce_years(
            test_df.filter(pl.col("Universe") == universe).get_column("Year").to_list(),
            context=f"{universe} test years",
        )
        per_universe[universe] = {
            "initial_year": _coerce_int(
                universe_df.get_column("Year").min(), context=f"{universe} initial year"
            ),
            "final_sim_year": _coerce_int(
                universe_df.get_column("Year").max(), context=f"{universe} final year"
            ),
            "valid_start_year": _coerce_int(
                universe_df.get_column("Year").min(), context=f"{universe} valid start year"
            ),
            "valid_end_year": _coerce_int(
                universe_df.get_column("Year").max(), context=f"{universe} valid end year"
            ),
            "train_year_range": [min(train_years), max(train_years)] if train_years else [0, 0],
            "test_year_range": [min(test_years), max(test_years)] if test_years else [0, 0],
        }

    return train_df, test_df, per_universe


class DataLoader:
    """Loads and caches training/test data from Hydra config."""

    def __init__(self, exp_root: str, quiet: bool = False) -> None:
        self.exp_root = exp_root
        self.quiet = quiet

    def load(self, cfg: DictConfig) -> PreparedData:
        """Parse config, load data, apply filters, return PreparedData."""
        data_cfg = {
            "leagues": _resolve_configured_leagues(cfg),
            "features": cfg.data.features_path,
            "year_start_offset": cfg.data.get("year_start_offset", None),
            "year_count": cfg.data.get("year_count", None),
            "classifier_target_col": cfg.target.classifier_sieve.target_col,
            "threshold": cfg.target.classifier_sieve.merit_threshold,
            "regressor_target_col": cfg.target.regressor_intensity.target_col,
            "regressor_target_space": cfg.target.regressor_intensity.get("target_space", "log"),
            "outcome_scorecard_columns": _coerce_str_list(
                cfg.target.get("outcome_scorecard", {}).get("columns")
            ),
            "outcome_scorecard_optional_columns": _coerce_str_list(
                cfg.target.get("outcome_scorecard", {}).get("optional_columns")
            ),
            "positions": cfg.positions,
            "buffer": cfg.split.right_censor_buffer,
            "split_strategy": cfg.split.get("strategy", "chronological"),
            "split_unit": cfg.split.get("unit", None),
            "split_seed": cfg.split.get("seed", cfg.get("seed", None)),
            "stratify_by": _coerce_str_list(cfg.split.get("stratify_by")),
            "test_pct": cfg.split.test_split_pct,
            "mask": cfg.mask_positional_features,
        }
        cfg_hash = _get_data_cache_cfg_hash(data_cfg)

        if (
            _GLOBAL_DATA_CACHE.get("last_cfg_hash") == cfg_hash
            and _GLOBAL_DATA_CACHE.get("X_train") is not None
        ):
            if not self.quiet:
                print(">>> Reusing pre-loaded data from Global Cache...")

            timeline = TimelineInfo(
                initial_year=_GLOBAL_DATA_CACHE["initial_year"],
                final_sim_year=_GLOBAL_DATA_CACHE["final_sim_year"],
                valid_start_year=_GLOBAL_DATA_CACHE["valid_start_year"],
                valid_end_year=_GLOBAL_DATA_CACHE["valid_end_year"],
                train_year_range=_GLOBAL_DATA_CACHE["train_year_range"],
                test_year_range=_GLOBAL_DATA_CACHE["test_year_range"],
                universes=_GLOBAL_DATA_CACHE.get("universes"),
                per_universe=_GLOBAL_DATA_CACHE.get("per_universe"),
                split_strategy=_GLOBAL_DATA_CACHE.get("split_strategy"),
                split_unit=_GLOBAL_DATA_CACHE.get("split_unit"),
            )

            return PreparedData(
                X_train=_GLOBAL_DATA_CACHE["X_train"],
                X_test=_GLOBAL_DATA_CACHE["X_test"],
                y_cls=_GLOBAL_DATA_CACHE["y_cls"],
                y_reg=_GLOBAL_DATA_CACHE["y_reg"],
                meta_train=_GLOBAL_DATA_CACHE["meta_train"],
                meta_test=_GLOBAL_DATA_CACHE["meta_test"],
                timeline=timeline,
                metadata_columns=_GLOBAL_DATA_CACHE["metadata_columns"],
                target_columns=_GLOBAL_DATA_CACHE["target_columns"],
                outcomes_train=_GLOBAL_DATA_CACHE["outcomes_train"],
            )

        features_file = os.path.abspath(os.path.join(self.exp_root, cfg.data.features_path))
        if not os.path.exists(features_file):
            raise FileNotFoundError(f"Processed features not found at {features_file}.")

        df = pl.read_parquet(features_file)
        if "Universe" not in df.columns:
            fallback_universe = (
                _resolve_configured_leagues(cfg)[0]
                if _resolve_configured_leagues(cfg)
                else "default"
            )
            df = df.with_columns(pl.lit(fallback_universe).alias("Universe"))

        available_columns = list(df.columns)

        _validate_active_target_columns(cfg, available_columns)
        scorecard_cols = _resolve_outcome_scorecard_columns(cfg, available_columns)
        missing_target_cols = sorted(
            c for c in DRAFT_OUTCOME_LEAKAGE_COLUMNS if c not in df.columns
        )
        if missing_target_cols:
            raise ValueError(
                "Processed features are missing draft outcome target columns "
                f"{missing_target_cols}. Re-run feature processing."
            )

        if cfg.positions and cfg.positions != "all":
            pos_list = [cfg.positions] if isinstance(cfg.positions, str) else list(cfg.positions)
            df = df.filter(pl.col("Position_Group").is_in(pos_list))

        split_strategy = str(cfg.split.get("strategy", "chronological"))
        if split_strategy == "chronological":
            eligible_df, per_universe = _eligible_by_universe(
                df, int(cfg.split.right_censor_buffer)
            )
            train_df, test_df = _split_chronological(
                eligible_df, per_universe, float(cfg.split.test_split_pct)
            )
        elif split_strategy == "random":
            eligible_df, _ = _eligible_by_universe(df, int(cfg.split.right_censor_buffer))
            train_df, test_df, per_universe = _split_random(eligible_df, cfg)
        else:
            raise ValueError("cfg.split.strategy must be either 'chronological' or 'random'")

        metadata_cols = ["Universe", "Player_ID", "Year", "First_Name", "Last_Name"]
        target_cols = [
            cfg.target.classifier_sieve.target_col,
            cfg.target.regressor_intensity.target_col,
            *DRAFT_OUTCOME_LEAKAGE_COLUMNS,
            *scorecard_cols,
        ]
        target_cols = list(dict.fromkeys(target_cols))
        feature_cols = [c for c in df.columns if c not in metadata_cols and c not in target_cols]

        X_train = train_df.select(feature_cols)
        y_train_df = train_df.select(target_cols)
        meta_train = train_df.select(metadata_cols)
        outcomes_train = train_df.select(scorecard_cols) if scorecard_cols else None

        X_test = test_df.select(feature_cols)
        meta_test = test_df.select(metadata_cols)

        if cfg.mask_positional_features:
            X_train = apply_position_mask(X_train)
            X_test = apply_position_mask(X_test)

        y_cls = y_train_df.get_column(cfg.target.classifier_sieve.target_col).to_numpy()
        y_reg = y_train_df.get_column(cfg.target.regressor_intensity.target_col).to_numpy()

        timeline = TimelineInfo(
            initial_year=min(v["initial_year"] for v in per_universe.values()),
            final_sim_year=max(v["final_sim_year"] for v in per_universe.values()),
            valid_start_year=min(v["valid_start_year"] for v in per_universe.values()),
            valid_end_year=max(v["valid_end_year"] for v in per_universe.values()),
            train_year_range=_global_year_range(per_universe, "train_year_range"),
            test_year_range=_global_year_range(per_universe, "test_year_range"),
            universes=sorted(per_universe),
            per_universe=per_universe,
            split_strategy=split_strategy,
            split_unit=cfg.split.get("unit", None),
        )

        _GLOBAL_DATA_CACHE.update(
            {
                "X_train": X_train,
                "X_test": X_test,
                "y_cls": y_cls,
                "y_reg": y_reg,
                "meta_train": meta_train,
                "meta_test": meta_test,
                "initial_year": timeline.initial_year,
                "final_sim_year": timeline.final_sim_year,
                "valid_start_year": timeline.valid_start_year,
                "valid_end_year": timeline.valid_end_year,
                "train_year_range": timeline.train_year_range,
                "test_year_range": timeline.test_year_range,
                "metadata_columns": metadata_cols,
                "target_columns": target_cols,
                "outcomes_train": outcomes_train,
                "universes": timeline.universes,
                "per_universe": timeline.per_universe,
                "split_strategy": timeline.split_strategy,
                "split_unit": timeline.split_unit,
                "last_cfg_hash": cfg_hash,
            }
        )

        return PreparedData(
            X_train=X_train,
            X_test=X_test,
            y_cls=y_cls,
            y_reg=y_reg,
            meta_train=meta_train,
            meta_test=meta_test,
            timeline=timeline,
            metadata_columns=metadata_cols,
            target_columns=target_cols,
            outcomes_train=outcomes_train,
        )

    def apply_feature_ablation(
        self, data: PreparedData, include: Optional[List[str]], exclude: Optional[List[str]]
    ) -> PreparedData:
        """Include/exclude features. Separated because these are trial-specific in sweeps."""
        X_train = data.X_train
        X_test = data.X_test

        if include:
            all_cols = X_train.columns
            expanded_include = []
            for p in include:
                expanded_include.extend(
                    fnmatch.filter(all_cols, p) if "*" in p or "?" in p else [p]
                )
            include_cols = [c for c in list(dict.fromkeys(expanded_include)) if c in all_cols]
            if not self.quiet:
                print(
                    f"Applying Feature Ablation: Keeping {len(include_cols)} features "
                    f"matching {include}"
                )
            X_train = X_train.select(include_cols)
            X_test = X_test.select(include_cols)

        if exclude:
            all_cols = X_train.columns
            expanded_exclude = []
            for p in exclude:
                expanded_exclude.extend(
                    fnmatch.filter(all_cols, p) if "*" in p or "?" in p else [p]
                )
            cols_to_drop = [c for c in list(dict.fromkeys(expanded_exclude)) if c in all_cols]
            if cols_to_drop:
                if not self.quiet:
                    print(
                        f"Applying Feature Ablation: Excluding {len(cols_to_drop)} features "
                        f"matching {exclude}"
                    )
                X_train = X_train.drop(cols_to_drop)
                X_test = X_test.drop(cols_to_drop)

        # Return a new PreparedData instance with modified X_train and X_test
        return PreparedData(
            X_train=X_train,
            X_test=X_test,
            y_cls=data.y_cls,
            y_reg=data.y_reg,
            meta_train=data.meta_train,
            meta_test=data.meta_test,
            timeline=data.timeline,
            metadata_columns=data.metadata_columns,
            target_columns=data.target_columns,
            outcomes_train=data.outcomes_train,
        )

    def print_summary(self, data: PreparedData, cfg: DictConfig) -> None:
        """Prints a summary of the loaded data."""
        if self.quiet:
            return

        universes = data.timeline.universes or []
        print(f"Universes: {len(universes)} ({', '.join(universes) if universes else 'legacy'})")
        print(f"Simulation Range: {data.timeline.initial_year} to {data.timeline.final_sim_year}")
        print(
            f"Active Range: {data.timeline.valid_start_year} to {data.timeline.valid_end_year} "
            f"(Buffer: {cfg.split.right_censor_buffer} years)"
        )
        print(f"Split Strategy: {cfg.split.get('strategy', 'chronological')}")
        if cfg.split.get("strategy") == "random":
            print(f"Split Unit: {cfg.split.get('unit', 'draft_class')}")
        print(f"Training Set: {data.timeline.train_year_range} ({len(data.X_train)} rows)")
        print(f"Holdout Set: {data.timeline.test_year_range} ({len(data.X_test)} rows)")
        if cfg.mask_positional_features:
            print("Applying In-Memory Positional Feature Masking...")
