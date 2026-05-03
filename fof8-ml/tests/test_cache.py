import tempfile

import polars as pl
from fof8_ml.data.cache import load_cached_dataset, save_cached_dataset
from omegaconf import OmegaConf


def test_cache_save_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock cfg using OmegaConf
        cfg = OmegaConf.create(
            {"data": {"raw_path": "dummy", "league_name": "dummy"}, "target": {}}
        )

        X = pl.DataFrame({"A": [1, 2]})
        y = pl.DataFrame({"B": [3, 4]})
        meta = pl.DataFrame({"C": [5, 6]})

        # Test Save
        save_cached_dataset(tmpdir, cfg, X, y, meta, "train")

        # Test Load
        loaded_X, loaded_y, loaded_meta = load_cached_dataset(tmpdir, cfg, "train")
        assert loaded_X is not None
        assert loaded_X.shape == (2, 1)
        assert loaded_y is not None
        assert loaded_meta is not None
