from types import SimpleNamespace

import polars as pl
import pytest
from fof8_core.targets.draft_outcomes import DRAFT_OUTCOME_LEAKAGE_COLUMNS
from fof8_ml.orchestration.data_loader import (
    _GLOBAL_DATA_CACHE,
    DataLoader,
    _get_data_cache_cfg_hash,
)
from omegaconf import OmegaConf


@pytest.fixture(autouse=True)
def reset_global_data_cache():
    _GLOBAL_DATA_CACHE.update(
        {
            "X_train": None,
            "X_test": None,
            "y_cls": None,
            "y_reg": None,
            "meta_train": None,
            "meta_test": None,
            "initial_year": None,
            "final_sim_year": None,
            "valid_start_year": None,
            "valid_end_year": None,
            "train_year_range": None,
            "test_year_range": None,
            "metadata_columns": None,
            "target_columns": None,
            "outcomes_train": None,
            "last_cfg_hash": None,
        }
    )


def _make_cfg(**overrides):
    cfg = OmegaConf.create(
        {
            "data": {
                "raw_path": "raw",
                "league_name": "league_a",
                "features_path": "features.parquet",
            },
            "target": {
                "classifier_sieve": {
                    "merit_threshold": 0.025,
                    "target_col": "Economic_Success",
                },
                "regressor_intensity": {
                    "target_col": "Positive_Career_Merit_Cap_Share",
                    "target_space": "raw",
                    "target_family": "economic",
                },
                "outcome_scorecard": {
                    "columns": [
                        "Positive_Career_Merit_Cap_Share",
                        "Career_Merit_Cap_Share",
                        "Positive_DPO",
                        "Peak_Overall",
                        "Career_Games_Played",
                    ],
                    "optional_columns": [
                        "Top3_Mean_Current_Overall",
                        "Award_Count",
                        "Hall_of_Fame_Flag",
                        "Hall_Of_Fame_Points",
                    ],
                },
            },
            "positions": "all",
            "split": {"right_censor_buffer": 3, "test_split_pct": 0.2},
            "mask_positional_features": False,
        }
    )
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=True)
    return cfg


def _data_cfg_for_hash(cfg):
    return {
        "league": cfg.data.league_name,
        "features": cfg.data.features_path,
        "classifier_target_col": cfg.target.classifier_sieve.target_col,
        "threshold": cfg.target.classifier_sieve.merit_threshold,
        "regressor_target_col": cfg.target.regressor_intensity.target_col,
        "regressor_target_space": cfg.target.regressor_intensity.target_space,
        "outcome_scorecard_columns": list(cfg.target.outcome_scorecard.columns),
        "outcome_scorecard_optional_columns": list(cfg.target.outcome_scorecard.optional_columns),
        "positions": cfg.positions,
        "buffer": cfg.split.right_censor_buffer,
        "test_pct": cfg.split.test_split_pct,
        "mask": cfg.mask_positional_features,
    }


def test_data_cache_hash_is_deterministic_for_same_config():
    cfg = _make_cfg()
    data_cfg = _data_cfg_for_hash(cfg)

    assert _get_data_cache_cfg_hash(data_cfg) == _get_data_cache_cfg_hash(data_cfg)


def test_data_cache_hash_changes_for_cache_relevant_fields():
    base_cfg = _make_cfg()
    base_hash = _get_data_cache_cfg_hash(_data_cfg_for_hash(base_cfg))

    changed_cfgs = [
        _make_cfg(**{"target.classifier_sieve.merit_threshold": 0.05}),
        _make_cfg(**{"target.classifier_sieve.target_col": "Cleared_Sieve"}),
        _make_cfg(**{"target.regressor_intensity.target_col": "Positive_DPO"}),
        _make_cfg(**{"target.regressor_intensity.target_space": "log"}),
        _make_cfg(**{"target.outcome_scorecard.columns": ["Positive_DPO", "Peak_Overall"]}),
        _make_cfg(**{"target.outcome_scorecard.optional_columns": ["Award_Count"]}),
        _make_cfg(positions=["QB", "WR"]),
        _make_cfg(**{"split.test_split_pct": 0.25}),
        _make_cfg(mask_positional_features=True),
    ]

    for cfg in changed_cfgs:
        assert _get_data_cache_cfg_hash(_data_cfg_for_hash(cfg)) != base_hash


def test_feature_ablation_does_not_poison_cached_base_data(monkeypatch):
    cfg = _make_cfg()
    loader = DataLoader(exp_root=".", quiet=True)

    source_df = pl.DataFrame(
        {
            "Player_ID": [1, 2],
            "Year": [2020, 2021],
            "First_Name": ["A", "B"],
            "Last_Name": ["One", "Two"],
            "Position_Group": ["QB", "WR"],
            "feat_keep": [10, 20],
            "feat_drop": [100, 200],
            "Career_Merit_Cap_Share": [0.8, 0.1],
            "DPO": [0.8, 0.1],
            "Positive_Career_Merit_Cap_Share": [0.8, 0.1],
            "Positive_DPO": [0.8, 0.1],
            "Economic_Success": [1, 1],
            "Cleared_Sieve": [1, 1],
            "Peak_Overall": [1.0, 1.0],
            "Career_Games_Played": [16, 16],
            "Top3_Mean_Current_Overall": [1.0, 1.0],
            "Award_Count": [0, 1],
        }
    )

    monkeypatch.setattr("fof8_ml.orchestration.data_loader.os.path.exists", lambda _: True)
    monkeypatch.setattr("fof8_ml.orchestration.data_loader.pl.read_parquet", lambda _: source_df)
    monkeypatch.setattr(
        "fof8_ml.orchestration.data_loader.FOF8Loader",
        lambda **_: SimpleNamespace(initial_sim_year=2019, final_sim_year=2025),
    )

    base_data = loader.load(cfg)
    base_cols = base_data.X_train.columns

    ablated = loader.apply_feature_ablation(base_data, include=None, exclude=["feat_drop"])
    assert "feat_drop" not in ablated.X_train.columns
    assert base_data.X_train.columns == base_cols

    loaded_again = loader.load(cfg)
    assert loaded_again.X_train.columns == base_cols
    assert "feat_drop" in loaded_again.X_train.columns


def test_loader_raises_clear_error_for_missing_active_target_columns(monkeypatch):
    cfg = _make_cfg()
    loader = DataLoader(exp_root=".", quiet=True)

    source_df = pl.DataFrame(
        {
            "Player_ID": [1],
            "Year": [2020],
            "First_Name": ["A"],
            "Last_Name": ["One"],
            "Position_Group": ["QB"],
            "feat_keep": [10],
            "Career_Merit_Cap_Share": [0.8],  # Missing all other economic target/context columns
        }
    )

    monkeypatch.setattr("fof8_ml.orchestration.data_loader.os.path.exists", lambda _: True)
    monkeypatch.setattr("fof8_ml.orchestration.data_loader.pl.read_parquet", lambda _: source_df)
    monkeypatch.setattr(
        "fof8_ml.orchestration.data_loader.FOF8Loader",
        lambda **_: SimpleNamespace(initial_sim_year=2019, final_sim_year=2025),
    )

    with pytest.raises(ValueError, match="missing configured active target columns"):
        loader.load(cfg)


def test_loader_dedupes_target_columns_when_targets_overlap_leakage(monkeypatch):
    cfg = _make_cfg()
    loader = DataLoader(exp_root=".", quiet=True)

    source_df = pl.DataFrame(
        {
            "Player_ID": [1, 2],
            "Year": [2020, 2021],
            "First_Name": ["A", "B"],
            "Last_Name": ["One", "Two"],
            "Position_Group": ["QB", "WR"],
            "feat_keep": [10, 20],
            "Career_Merit_Cap_Share": [0.8, 0.1],
            "DPO": [0.8, 0.1],
            "Positive_Career_Merit_Cap_Share": [0.8, 0.1],
            "Positive_DPO": [0.8, 0.1],
            "Economic_Success": [1, 1],
            "Cleared_Sieve": [1, 1],
            "Peak_Overall": [1.0, 1.0],
            "Career_Games_Played": [16, 16],
            "Top3_Mean_Current_Overall": [1.0, 1.0],
            "Award_Count": [0, 1],
        }
    )

    monkeypatch.setattr("fof8_ml.orchestration.data_loader.os.path.exists", lambda _: True)
    monkeypatch.setattr("fof8_ml.orchestration.data_loader.pl.read_parquet", lambda _: source_df)
    monkeypatch.setattr(
        "fof8_ml.orchestration.data_loader.FOF8Loader",
        lambda **_: SimpleNamespace(initial_sim_year=2019, final_sim_year=2025),
    )

    data = loader.load(cfg)
    assert len(data.target_columns) == len(set(data.target_columns))
    assert set(DRAFT_OUTCOME_LEAKAGE_COLUMNS).issubset(set(data.target_columns))


def test_loader_populates_outcomes_train_with_full_available_scorecard(monkeypatch):
    cfg = _make_cfg()
    loader = DataLoader(exp_root=".", quiet=True)

    source_df = pl.DataFrame(
        {
            "Player_ID": [1, 2],
            "Year": [2020, 2021],
            "First_Name": ["A", "B"],
            "Last_Name": ["One", "Two"],
            "Position_Group": ["QB", "WR"],
            "feat_keep": [10, 20],
            "Career_Merit_Cap_Share": [0.8, 0.1],
            "DPO": [0.8, 0.1],
            "Positive_Career_Merit_Cap_Share": [0.8, 0.1],
            "Positive_DPO": [0.8, 0.1],
            "Economic_Success": [1, 1],
            "Cleared_Sieve": [1, 1],
            "Peak_Overall": [70.0, 65.0],
            "Career_Games_Played": [16, 32],
            "Top3_Mean_Current_Overall": [72.0, 67.0],
            "Award_Count": [0, 1],
        }
    )

    monkeypatch.setattr("fof8_ml.orchestration.data_loader.os.path.exists", lambda _: True)
    monkeypatch.setattr("fof8_ml.orchestration.data_loader.pl.read_parquet", lambda _: source_df)
    monkeypatch.setattr(
        "fof8_ml.orchestration.data_loader.FOF8Loader",
        lambda **_: SimpleNamespace(initial_sim_year=2019, final_sim_year=2025),
    )

    data = loader.load(cfg)

    assert data.outcomes_train is not None
    assert set(data.outcomes_train.columns) == {
        "Positive_Career_Merit_Cap_Share",
        "Career_Merit_Cap_Share",
        "Positive_DPO",
        "Peak_Overall",
        "Career_Games_Played",
        "Top3_Mean_Current_Overall",
        "Award_Count",
    }
    assert "Top3_Mean_Current_Overall" not in data.X_train.columns
    assert "Award_Count" not in data.X_train.columns


def test_loader_fails_for_missing_required_outcome_scorecard_column(monkeypatch):
    cfg = _make_cfg()
    loader = DataLoader(exp_root=".", quiet=True)

    source_df = pl.DataFrame(
        {
            "Player_ID": [1],
            "Year": [2020],
            "First_Name": ["A"],
            "Last_Name": ["One"],
            "Position_Group": ["QB"],
            "feat_keep": [10],
            "Career_Merit_Cap_Share": [0.8],
            "DPO": [0.8],
            "Positive_Career_Merit_Cap_Share": [0.8],
            "Positive_DPO": [0.8],
            "Economic_Success": [1],
            "Cleared_Sieve": [1],
            "Career_Games_Played": [16],
        }
    )

    monkeypatch.setattr("fof8_ml.orchestration.data_loader.os.path.exists", lambda _: True)
    monkeypatch.setattr("fof8_ml.orchestration.data_loader.pl.read_parquet", lambda _: source_df)
    monkeypatch.setattr(
        "fof8_ml.orchestration.data_loader.FOF8Loader",
        lambda **_: SimpleNamespace(initial_sim_year=2019, final_sim_year=2025),
    )

    with pytest.raises(ValueError, match="missing required outcome_scorecard columns"):
        loader.load(cfg)
