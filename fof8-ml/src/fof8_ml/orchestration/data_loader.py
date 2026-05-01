import fnmatch
import hashlib
import json
import os
from typing import Any, Dict, List, Optional

import polars as pl
from fof8_core.features.position_masks import apply_position_mask
from fof8_core.loader import FOF8Loader
from omegaconf import DictConfig

from fof8_ml.orchestration.pipeline_types import PreparedData, TimelineInfo

# Module-level cache for data persistence across trials in the same process
_GLOBAL_DATA_CACHE: Dict[str, Any] = {
    "X_train": None,
    "y_cls": None,
    "y_reg": None,
    "meta_train": None,
    "last_cfg_hash": None,
}


def _get_data_cache_cfg_hash(data_cfg: Dict[str, Any]) -> str:
    """Create a deterministic hash for cache keying across processes."""
    serialized = json.dumps(data_cfg, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class DataLoader:
    """Loads and caches training/test data from Hydra config."""

    def __init__(self, exp_root: str, quiet: bool = False):
        self.exp_root = exp_root
        self.quiet = quiet

    def load(self, cfg: DictConfig) -> PreparedData:
        """Parse config, load data, apply filters, return PreparedData."""
        absolute_raw_path = os.path.abspath(os.path.join(self.exp_root, cfg.data.raw_path))

        data_cfg = {
            "league": cfg.data.league_name,
            "features": cfg.data.features_path,
            "threshold": cfg.target.stage1_sieve.merit_threshold,
            "positions": cfg.positions,
            "buffer": cfg.split.right_censor_buffer,
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
            )

            return PreparedData(
                X_train=_GLOBAL_DATA_CACHE["X_train"],
                X_test=_GLOBAL_DATA_CACHE["X_test"],
                y_cls=_GLOBAL_DATA_CACHE["y_cls"],
                y_reg=_GLOBAL_DATA_CACHE["y_reg"],
                meta_train=_GLOBAL_DATA_CACHE["meta_train"],
                meta_test=_GLOBAL_DATA_CACHE["meta_test"],
                timeline=timeline,
            )

        loader = FOF8Loader(base_path=absolute_raw_path, league_name=cfg.data.league_name)

        # 1. Automate Timeline Discovery
        initial_year = loader.initial_sim_year
        final_sim_year = loader.final_sim_year
        valid_start_year = initial_year + 1

        # 2. Load Preprocessed Data
        features_file = os.path.abspath(os.path.join(self.exp_root, cfg.data.features_path))
        if not os.path.exists(features_file):
            raise FileNotFoundError(f"Processed features not found at {features_file}.")

        df = pl.read_parquet(features_file)

        # --- Runtime Filtering & Target Labeling ---
        df = df.with_columns(
            (pl.col("Career_Merit_Cap_Share") > cfg.target.stage1_sieve.merit_threshold)
            .alias("Cleared_Sieve")
            .cast(pl.Int8)
        )

        if cfg.positions and cfg.positions != "all":
            pos_list = [cfg.positions] if isinstance(cfg.positions, str) else list(cfg.positions)
            df = df.filter(pl.col("Position_Group").is_in(pos_list))

        valid_end_year = final_sim_year - cfg.split.right_censor_buffer
        df = df.filter(pl.col("Year") <= valid_end_year)

        total_valid_years = valid_end_year - valid_start_year + 1
        test_years_count = int(total_valid_years * cfg.split.test_split_pct)
        train_end_year = valid_end_year - test_years_count
        train_year_range = [valid_start_year, train_end_year]
        test_year_range = [train_end_year + 1, valid_end_year]

        train_df = df.filter(
            (pl.col("Year") >= train_year_range[0]) & (pl.col("Year") <= train_year_range[1])
        )
        test_df = df.filter(
            (pl.col("Year") >= test_year_range[0]) & (pl.col("Year") <= test_year_range[1])
        )

        metadata_cols = ["Player_ID", "Year", "First_Name", "Last_Name"]
        target_cols = [
            cfg.target.stage1_sieve.target_col,
            cfg.target.stage2_intensity.target_col,
        ] + list(cfg.target.leakage_prevention.drop_cols)
        feature_cols = [c for c in df.columns if c not in metadata_cols and c not in target_cols]

        X_train = train_df.select(feature_cols)
        y_train_df = train_df.select(target_cols)
        meta_train = train_df.select(metadata_cols)

        X_test = test_df.select(feature_cols)
        meta_test = test_df.select(metadata_cols)

        if cfg.mask_positional_features:
            X_train = apply_position_mask(X_train)
            X_test = apply_position_mask(X_test)

        y_cls = y_train_df.get_column(cfg.target.stage1_sieve.target_col).to_numpy()
        y_reg = y_train_df.get_column(cfg.target.stage2_intensity.target_col).to_numpy()

        timeline = TimelineInfo(
            initial_year=initial_year,
            final_sim_year=final_sim_year,
            valid_start_year=valid_start_year,
            valid_end_year=valid_end_year,
            train_year_range=train_year_range,
            test_year_range=test_year_range,
        )

        # Store in Global Cache for next trial
        _GLOBAL_DATA_CACHE.update(
            {
                "X_train": X_train,
                "X_test": X_test,
                "y_cls": y_cls,
                "y_reg": y_reg,
                "meta_train": meta_train,
                "meta_test": meta_test,
                "initial_year": initial_year,
                "final_sim_year": final_sim_year,
                "valid_start_year": valid_start_year,
                "valid_end_year": valid_end_year,
                "train_year_range": train_year_range,
                "test_year_range": test_year_range,
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
        )

    def print_summary(self, data: PreparedData, cfg: DictConfig) -> None:
        """Prints a summary of the loaded data."""
        if self.quiet:
            return

        print(f"Simulation Range: {data.timeline.initial_year} to {data.timeline.final_sim_year}")
        print(
            f"Active Range: {data.timeline.valid_start_year} to {data.timeline.valid_end_year} "
            f"(Buffer: {cfg.split.right_censor_buffer} years)"
        )
        print(
            f"Training Set: {data.timeline.train_year_range} "
            f"({data.timeline.train_year_range[1] - data.timeline.train_year_range[0] + 1} classes)"
        )
        print(
            f"Holdout Set: {data.timeline.test_year_range} "
            f"({data.timeline.test_year_range[1] - data.timeline.test_year_range[0] + 1} classes)"
        )
        if cfg.mask_positional_features:
            print("Applying In-Memory Positional Feature Masking...")
