import hashlib
import json
import os
import polars as pl
from pathlib import Path
from omegaconf import OmegaConf

from hydra.utils import to_absolute_path

import yaml


def get_data_hash(cfg, year_range=None):
    """
    Generates a unique hash based on the data and target configuration.
    """
    # 1. Identify Raw Data State (Prefer DVC if available)
    raw_dir = Path(to_absolute_path(cfg.data.raw_path))
    dvc_path = raw_dir.with_suffix(".dvc")

    data_identity = None
    if dvc_path.exists():
        try:
            with open(dvc_path, "r") as f:
                dvc_meta = yaml.safe_load(f)
                # Use the MD5 of the tracked directory
                data_identity = dvc_meta["outs"][0]["md5"]
        except Exception:
            pass

    if data_identity is None:
        # Fallback to mtime of the league subdirectory
        league_dir = raw_dir / cfg.data.league_name
        data_identity = os.path.getmtime(league_dir) if league_dir.exists() else 0

    relevant_keys = {
        "data": OmegaConf.to_container(cfg.data, resolve=True),
        "target": OmegaConf.to_container(cfg.target, resolve=True),
        "year_range": year_range,
        "data_identity": data_identity,
    }
    # Remove paths and cache settings from hash
    if "raw_path" in relevant_keys["data"]:
        del relevant_keys["data"]["raw_path"]
    if "cache_dir" in relevant_keys["data"]:
        del relevant_keys["data"]["cache_dir"]
    if "include_features" in relevant_keys["data"]:
        del relevant_keys["data"]["include_features"]
    if "exclude_features" in relevant_keys["data"]:
        del relevant_keys["data"]["exclude_features"]
    if "mask_positional_features" in relevant_keys["data"]:
        del relevant_keys["data"]["mask_positional_features"]

    cfg_json = json.dumps(relevant_keys, sort_keys=True)
    return hashlib.md5(cfg_json.encode()).hexdigest()


def load_cached_dataset(cache_dir, cfg, split_name="train", year_range=None):
    """
    Attempts to load X, y, and metadata from Parquet files if they exist and match the config hash.
    """
    h = get_data_hash(cfg, year_range)
    cache_path = Path(cache_dir) / h / split_name

    files = {
        "X": cache_path / "X.parquet",
        "y": cache_path / "y.parquet",
        "meta": cache_path / "metadata.parquet",
    }

    if all(f.exists() for f in files.values()):
        print(f"Loading cached {split_name} dataset (hash: {h[:8]})...")
        X = pl.read_parquet(files["X"])
        y = pl.read_parquet(files["y"])
        meta = pl.read_parquet(files["meta"])
        return X, y, meta

    return None, None, None


def save_cached_dataset(cache_dir, cfg, X, y, meta, split_name="train", year_range=None):
    """
    Saves the dataset to Parquet files in a versioned directory.
    """
    h = get_data_hash(cfg, year_range)
    cache_path = Path(cache_dir) / h / split_name
    cache_path.mkdir(parents=True, exist_ok=True)

    X.write_parquet(cache_path / "X.parquet")
    y.write_parquet(cache_path / "y.parquet")
    meta.write_parquet(cache_path / "metadata.parquet")

    print(f"Saved {split_name} dataset to cache (hash: {h[:8]}).")
    return cache_path
