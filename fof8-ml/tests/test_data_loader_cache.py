import importlib.util
from pathlib import Path
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
            "X_val": None,
            "X_test": None,
            "y_cls": None,
            "y_cls_val": None,
            "y_cls_test": None,
            "y_reg": None,
            "y_reg_val": None,
            "y_reg_test": None,
            "reg_weight": None,
            "reg_weight_val": None,
            "reg_weight_test": None,
            "meta_train": None,
            "meta_val": None,
            "meta_test": None,
            "initial_year": None,
            "final_sim_year": None,
            "valid_start_year": None,
            "valid_end_year": None,
            "train_year_range": None,
            "val_year_range": None,
            "test_year_range": None,
            "metadata_columns": None,
            "target_columns": None,
            "outcomes_train": None,
            "outcomes_val": None,
            "outcomes_test": None,
            "universes": None,
            "per_universe": None,
            "split_strategy": None,
            "split_unit": None,
            "last_cfg_hash": None,
        }
    )


def _load_process_features_module():
    module_path = Path(__file__).parents[2] / "pipelines" / "process_features.py"
    spec = importlib.util.spec_from_file_location("process_features", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
                    "elite": {
                        "enabled": True,
                        "source_column": "Career_Merit_Cap_Share",
                        "quantile": 0.95,
                        "scope": "position_group",
                        "scope_column": "Position_Group",
                        "fallback_scope": "global",
                        "min_group_size": 100,
                        "top_k_precision": 32,
                        "top_k_recall": 64,
                    },
                },
            },
            "positions": "all",
            "split": {"strategy": "chronological", "right_censor_buffer": 0, "test_split_pct": 0.2},
            "mask_positional_features": False,
        }
    )
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=True)
    return cfg


def _data_cfg_for_hash(cfg):
    return {
        "leagues": list(cfg.data.get("league_names") or [cfg.data.league_name]),
        "features": cfg.data.features_path,
        "year_start_offset": cfg.data.get("year_start_offset", None),
        "year_count": cfg.data.get("year_count", None),
        "classifier_target_col": cfg.target.classifier_sieve.target_col,
        "threshold": cfg.target.classifier_sieve.merit_threshold,
        "regressor_target_col": cfg.target.regressor_intensity.target_col,
        "regressor_target_space": cfg.target.regressor_intensity.target_space,
        "regressor_sample_weight": cfg.target.regressor_intensity.get("sample_weight", None),
        "outcome_scorecard_columns": list(cfg.target.outcome_scorecard.columns),
        "outcome_scorecard_optional_columns": list(cfg.target.outcome_scorecard.optional_columns),
        "outcome_scorecard_elite_enabled": cfg.target.outcome_scorecard.elite.enabled,
        "outcome_scorecard_elite_source_column": cfg.target.outcome_scorecard.elite.source_column,
        "outcome_scorecard_elite_quantile": cfg.target.outcome_scorecard.elite.quantile,
        "outcome_scorecard_elite_scope": cfg.target.outcome_scorecard.elite.scope,
        "outcome_scorecard_elite_scope_column": cfg.target.outcome_scorecard.elite.scope_column,
        "outcome_scorecard_elite_fallback_scope": (
            cfg.target.outcome_scorecard.elite.fallback_scope
        ),
        "outcome_scorecard_elite_min_group_size": (
            cfg.target.outcome_scorecard.elite.min_group_size
        ),
        "positions": cfg.positions,
        "buffer": cfg.split.right_censor_buffer,
        "split_strategy": cfg.split.get("strategy", "chronological"),
        "split_unit": cfg.split.get("unit", None),
        "split_seed": cfg.split.get("seed", cfg.get("seed", None)),
        "stratify_by": list(cfg.split.get("stratify_by", [])),
        "val_pct": cfg.split.get("val_split_pct", None),
        "test_pct": cfg.split.test_split_pct,
        "train_universe_pct": cfg.split.get("train_universe_pct", None),
        "val_universe_pct": cfg.split.get("val_universe_pct", None),
        "test_universe_pct": cfg.split.get("test_universe_pct", None),
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
        _make_cfg(
            **{
                "target.regressor_intensity.sample_weight": {
                    "enabled": True,
                    "target_col": "__regressor_sample_weight",
                    "source_col": "Positive_Career_Merit_Cap_Share",
                    "kind": "global_percentile_bucket",
                    "base_weight": 1.0,
                    "thresholds": [{"percentile": 0.9, "weight": 2.0}],
                }
            }
        ),
        _make_cfg(**{"target.outcome_scorecard.columns": ["Positive_DPO", "Peak_Overall"]}),
        _make_cfg(**{"target.outcome_scorecard.optional_columns": ["Award_Count"]}),
        _make_cfg(**{"target.outcome_scorecard.elite.quantile": 0.9}),
        _make_cfg(positions=["QB", "WR"]),
        _make_cfg(**{"split.test_split_pct": 0.25}),
        _make_cfg(**{"split.val_split_pct": 0.10}),
        _make_cfg(**{"split.strategy": "random"}),
        _make_cfg(
            **{
                "split.strategy": "grouped_universe",
                "split.train_universe_pct": 0.6,
                "split.val_universe_pct": 0.2,
                "split.test_universe_pct": 0.2,
            }
        ),
        _make_cfg(**{"split.seed": 99}),
        _make_cfg(**{"split.unit": "row"}),
        _make_cfg(**{"split.stratify_by": ["Universe"]}),
        _make_cfg(**{"data.league_names": ["league_a", "league_b"]}),
        _make_cfg(**{"data.year_start_offset": 1}),
        _make_cfg(**{"data.year_count": 30}),
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
            "Control_Y1_Current_Overall": [1.0, 1.0],
            "Control_Y2_Current_Overall": [1.0, 1.0],
            "Control_Y3_Current_Overall": [1.0, 1.0],
            "Control_Y4_Current_Overall": [1.0, 1.0],
            "Control_Window_Mean_Current_Overall": [1.0, 1.0],
            "Control_Window_Discounted_Mean_Current_Overall": [1.0, 1.0],
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


def test_loader_materializes_regressor_sample_weights(monkeypatch):
    cfg = _make_cfg(
        **{
            "target.regressor_intensity.target_col": "Control_Window_Mean_Current_Overall",
            "target.regressor_intensity.sample_weight": {
                "enabled": True,
                "target_col": "__regressor_sample_weight",
                "source_col": "Control_Window_Mean_Current_Overall",
                "kind": "global_percentile_bucket",
                "base_weight": 1.0,
                "thresholds": [{"percentile": 0.75, "weight": 2.0}],
            },
        }
    )
    loader = DataLoader(exp_root=".", quiet=True)

    source_df = pl.DataFrame(
        {
            "Universe": ["U1", "U1", "U1", "U1", "U1"],
            "Player_ID": [1, 2, 3, 4, 5],
            "Year": [2020, 2021, 2022, 2023, 2024],
            "First_Name": ["A", "B", "C", "D", "E"],
            "Last_Name": ["One", "Two", "Three", "Four", "Five"],
            "Position_Group": ["QB", "QB", "RB", "WR", "TE"],
            "feat_keep": [1, 2, 3, 4, 5],
            "Career_Merit_Cap_Share": [0.8, 0.7, 0.6, 0.5, 0.4],
            "DPO": [1, 1, 1, 1, 1],
            "Positive_Career_Merit_Cap_Share": [0.8, 0.7, 0.6, 0.5, 0.4],
            "Positive_DPO": [1, 1, 1, 1, 1],
            "Economic_Success": [1, 1, 1, 1, 1],
            "Cleared_Sieve": [1, 1, 1, 1, 1],
            "Peak_Overall": [1, 1, 1, 1, 1],
            "Career_Games_Played": [16, 16, 16, 16, 16],
            "Top3_Mean_Current_Overall": [1, 1, 1, 1, 1],
            "Control_Y1_Current_Overall": [1, 1, 1, 1, 1],
            "Control_Y2_Current_Overall": [1, 1, 1, 1, 1],
            "Control_Y3_Current_Overall": [1, 1, 1, 1, 1],
            "Control_Y4_Current_Overall": [1, 1, 1, 1, 1],
            "Control_Window_Mean_Current_Overall": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Control_Window_Discounted_Mean_Current_Overall": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Award_Count": [0, 0, 0, 0, 0],
        }
    )

    monkeypatch.setattr("fof8_ml.orchestration.data_loader.os.path.exists", lambda _: True)
    monkeypatch.setattr("fof8_ml.orchestration.data_loader.pl.read_parquet", lambda _: source_df)
    monkeypatch.setattr(
        "fof8_ml.orchestration.data_loader.FOF8Loader",
        lambda **_: SimpleNamespace(initial_sim_year=2019, final_sim_year=2025),
    )

    data = loader.load(cfg)
    assert data.reg_weight is not None
    combined = list(data.reg_weight) + list(data.reg_weight_val) + list(data.reg_weight_test)
    assert sorted(set(float(v) for v in combined)) == [1.0, 2.0]


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
            "Control_Y1_Current_Overall": [1.0, 1.0],
            "Control_Y2_Current_Overall": [1.0, 1.0],
            "Control_Y3_Current_Overall": [1.0, 1.0],
            "Control_Y4_Current_Overall": [1.0, 1.0],
            "Control_Window_Mean_Current_Overall": [1.0, 1.0],
            "Control_Window_Discounted_Mean_Current_Overall": [1.0, 1.0],
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
            "Control_Y1_Current_Overall": [68.0, 63.0],
            "Control_Y2_Current_Overall": [70.0, 64.0],
            "Control_Y3_Current_Overall": [72.0, 66.0],
            "Control_Y4_Current_Overall": [74.0, 68.0],
            "Control_Window_Mean_Current_Overall": [71.0, 65.25],
            "Control_Window_Discounted_Mean_Current_Overall": [70.6, 64.9],
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
    assert "Control_Window_Mean_Current_Overall" not in data.X_train.columns
    assert "Award_Count" not in data.X_train.columns


def test_loader_auto_includes_elite_source_column_in_scorecard(monkeypatch):
    cfg = _make_cfg(
        **{
            "target.outcome_scorecard.columns": [
                "Positive_Career_Merit_Cap_Share",
                "Positive_DPO",
                "Peak_Overall",
                "Career_Games_Played",
            ]
        }
    )
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
            "Top3_Mean_Current_Overall": [72.0, 67.0],
            "Career_Games_Played": [16, 32],
            "Control_Y1_Current_Overall": [68.0, 63.0],
            "Control_Y2_Current_Overall": [70.0, 64.0],
            "Control_Y3_Current_Overall": [72.0, 66.0],
            "Control_Y4_Current_Overall": [74.0, 68.0],
            "Control_Window_Mean_Current_Overall": [71.0, 65.25],
            "Control_Window_Discounted_Mean_Current_Overall": [70.6, 64.9],
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
    assert "Career_Merit_Cap_Share" in data.outcomes_train.columns


def test_loader_fails_for_missing_required_outcome_scorecard_column(monkeypatch):
    cfg = _make_cfg(**{"target.outcome_scorecard.columns": ["Peak_Overall", "Award_Count"]})
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
            "Peak_Overall": [70.0],
            "Top3_Mean_Current_Overall": [72.0],
            "Career_Games_Played": [16],
            "Control_Y1_Current_Overall": [68.0],
            "Control_Y2_Current_Overall": [70.0],
            "Control_Y3_Current_Overall": [72.0],
            "Control_Y4_Current_Overall": [74.0],
            "Control_Window_Mean_Current_Overall": [71.0],
            "Control_Window_Discounted_Mean_Current_Overall": [70.6],
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


def test_loader_builds_not_null_derived_targets(monkeypatch):
    cfg = _make_cfg(
        **{
            "target.classifier_sieve.target_col": "Control_Window_Target_Observed",
            "target.classifier_sieve.derived_targets": [
                {
                    "target_col": "Control_Window_Target_Observed",
                    "kind": "not_null",
                    "source_col": "Control_Window_Mean_Current_Overall",
                }
            ],
            "target.regressor_intensity.target_col": "Control_Window_Mean_Current_Overall",
        }
    )
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
            "Top3_Mean_Current_Overall": [72.0, 67.0],
            "Career_Games_Played": [16, 32],
            "Control_Y1_Current_Overall": [68.0, 63.0],
            "Control_Y2_Current_Overall": [70.0, 64.0],
            "Control_Y3_Current_Overall": [72.0, 66.0],
            "Control_Y4_Current_Overall": [74.0, 68.0],
            "Control_Window_Mean_Current_Overall": [71.0, None],
            "Control_Window_Discounted_Mean_Current_Overall": [70.6, None],
        }
    )

    monkeypatch.setattr("fof8_ml.orchestration.data_loader.os.path.exists", lambda _: True)
    monkeypatch.setattr("fof8_ml.orchestration.data_loader.pl.read_parquet", lambda _: source_df)
    monkeypatch.setattr(
        "fof8_ml.orchestration.data_loader.FOF8Loader",
        lambda **_: SimpleNamespace(initial_sim_year=2019, final_sim_year=2025),
    )

    data = loader.load(cfg)

    assert data.y_cls.tolist() == [1]
    assert data.y_reg.tolist() == [71.0]


def test_loader_builds_position_group_zscore_regressor_targets(monkeypatch):
    cfg = _make_cfg(
        **{
            "split.strategy": "random",
            "split.unit": "row",
            "split.test_split_pct": 0.0,
            "target.classifier_sieve.target_col": "Top3_Target_Observed",
            "target.classifier_sieve.derived_targets": [
                {
                    "target_col": "Top3_Target_Observed",
                    "kind": "not_null",
                    "source_col": "Top3_Mean_Current_Overall",
                }
            ],
            "target.regressor_intensity.target_col": "Top3_Mean_Current_Overall_Pos_Z",
            "target.regressor_intensity.derived_targets": [
                {
                    "target_col": "Top3_Mean_Current_Overall_Pos_Z",
                    "kind": "position_group_zscore",
                    "source_col": "Top3_Mean_Current_Overall",
                    "group_col": "Position_Group",
                }
            ],
        }
    )
    loader = DataLoader(exp_root=".", quiet=True)

    source_df = pl.DataFrame(
        {
            "Player_ID": [1, 2, 3, 4],
            "Year": [2020, 2020, 2021, 2021],
            "First_Name": ["A", "B", "C", "D"],
            "Last_Name": ["One", "Two", "Three", "Four"],
            "Position_Group": ["QB", "QB", "WR", "WR"],
            "feat_keep": [10, 20, 30, 40],
            "Career_Merit_Cap_Share": [0.8, 0.1, 0.4, 0.2],
            "DPO": [0.8, 0.1, 0.4, 0.2],
            "Positive_Career_Merit_Cap_Share": [0.8, 0.1, 0.4, 0.2],
            "Positive_DPO": [0.8, 0.1, 0.4, 0.2],
            "Economic_Success": [1, 1, 1, 1],
            "Cleared_Sieve": [1, 1, 1, 1],
            "Peak_Overall": [70.0, 65.0, 60.0, 58.0],
            "Top3_Mean_Current_Overall": [80.0, 60.0, 50.0, 30.0],
            "Career_Games_Played": [16, 32, 48, 64],
            "Control_Y1_Current_Overall": [68.0, 63.0, 55.0, 35.0],
            "Control_Y2_Current_Overall": [70.0, 64.0, 57.0, 37.0],
            "Control_Y3_Current_Overall": [72.0, 66.0, 59.0, 39.0],
            "Control_Y4_Current_Overall": [74.0, 68.0, 61.0, 41.0],
            "Control_Window_Mean_Current_Overall": [71.0, 65.25, 58.0, 38.0],
            "Control_Window_Discounted_Mean_Current_Overall": [70.6, 64.9, 57.5, 37.5],
        }
    )

    monkeypatch.setattr("fof8_ml.orchestration.data_loader.os.path.exists", lambda _: True)
    monkeypatch.setattr("fof8_ml.orchestration.data_loader.pl.read_parquet", lambda _: source_df)
    monkeypatch.setattr(
        "fof8_ml.orchestration.data_loader.FOF8Loader",
        lambda **_: SimpleNamespace(initial_sim_year=2019, final_sim_year=2025),
    )

    data = loader.load(cfg)

    assert len(data.y_reg) == 4
    assert sorted(round(v, 6) for v in data.y_reg.tolist()) == sorted(
        [round(v, 6) for v in [0.70710678, -0.70710678, 0.70710678, -0.70710678]]
    )


def _pooled_source_df(universes: list[str] | None = None) -> pl.DataFrame:
    rows = []
    player_id = 1
    universe_map = {
        "league_a": range(2020, 2025),
        "league_b": range(2030, 2035),
        "league_c": range(2040, 2045),
        "league_d": range(2050, 2055),
        "league_e": range(2060, 2065),
    }
    selected = universes or list(universe_map.keys())
    for universe in selected:
        years = universe_map[universe]
        for year in years:
            rows.append(
                {
                    "Universe": universe,
                    "Player_ID": player_id,
                    "Year": year,
                    "First_Name": "A",
                    "Last_Name": str(player_id),
                    "Position_Group": "QB",
                    "feat_keep": player_id,
                    "Career_Merit_Cap_Share": 0.8,
                    "DPO": 0.8,
                    "Positive_Career_Merit_Cap_Share": 0.8,
                    "Positive_DPO": 0.8,
                    "Economic_Success": 1,
                    "Cleared_Sieve": 1,
                    "Peak_Overall": 70.0,
                    "Career_Games_Played": 16,
                    "Top3_Mean_Current_Overall": 72.0,
                    "Control_Y1_Current_Overall": 68.0,
                    "Control_Y2_Current_Overall": 70.0,
                    "Control_Y3_Current_Overall": 72.0,
                    "Control_Y4_Current_Overall": 74.0,
                    "Control_Window_Mean_Current_Overall": 71.0,
                    "Control_Window_Discounted_Mean_Current_Overall": 70.6,
                    "Award_Count": 0,
                }
            )
            player_id += 1
    return pl.DataFrame(rows)


def test_chronological_split_is_computed_per_universe(monkeypatch):
    cfg = _make_cfg(
        **{
            "data.league_names": ["league_a", "league_b"],
            "split.test_split_pct": 0.4,
            "split.right_censor_buffer": 0,
        }
    )
    loader = DataLoader(exp_root=".", quiet=True)

    monkeypatch.setattr("fof8_ml.orchestration.data_loader.os.path.exists", lambda _: True)
    monkeypatch.setattr(
        "fof8_ml.orchestration.data_loader.pl.read_parquet",
        lambda _: _pooled_source_df(["league_a", "league_b"]),
    )

    data = loader.load(cfg)

    assert "Universe" not in data.X_train.columns
    assert set(data.meta_train.columns) == {
        "Universe",
        "Player_ID",
        "Year",
        "First_Name",
        "Last_Name",
    }
    assert set(data.meta_test.filter(pl.col("Universe") == "league_a")["Year"].to_list()) == {
        2023,
        2024,
    }
    assert set(data.meta_test.filter(pl.col("Universe") == "league_b")["Year"].to_list()) == {
        2033,
        2034,
    }
    assert len(data.X_train) == 6
    assert len(data.X_val) == 0
    assert len(data.X_test) == 4


def test_random_draft_class_split_is_deterministic_and_grouped(monkeypatch):
    cfg = _make_cfg(
        **{
            "data.league_names": ["league_a", "league_b"],
            "split.strategy": "random",
            "split.unit": "draft_class",
            "split.seed": 7,
            "split.test_split_pct": 0.3,
            "split.right_censor_buffer": 0,
        }
    )
    loader = DataLoader(exp_root=".", quiet=True)

    monkeypatch.setattr("fof8_ml.orchestration.data_loader.os.path.exists", lambda _: True)
    monkeypatch.setattr(
        "fof8_ml.orchestration.data_loader.pl.read_parquet",
        lambda _: _pooled_source_df(["league_a", "league_b"]),
    )

    first = loader.load(cfg)
    first_test_groups = set(
        zip(first.meta_test["Universe"].to_list(), first.meta_test["Year"].to_list())
    )

    _GLOBAL_DATA_CACHE["last_cfg_hash"] = None
    second = loader.load(cfg)
    second_test_groups = set(
        zip(second.meta_test["Universe"].to_list(), second.meta_test["Year"].to_list())
    )

    train_groups = set(
        zip(first.meta_train["Universe"].to_list(), first.meta_train["Year"].to_list())
    )
    assert first_test_groups == second_test_groups
    assert first_test_groups
    assert first_test_groups.isdisjoint(train_groups)


def test_grouped_universe_split_assigns_whole_universes_to_train_val_test(monkeypatch):
    cfg = _make_cfg(
        **{
            "data.league_names": ["league_a", "league_b", "league_c", "league_d", "league_e"],
            "split.strategy": "grouped_universe",
            "split.train_universe_pct": 0.6,
            "split.val_universe_pct": 0.2,
            "split.test_universe_pct": 0.2,
            "split.seed": 11,
            "split.right_censor_buffer": 0,
        }
    )
    loader = DataLoader(exp_root=".", quiet=True)

    monkeypatch.setattr("fof8_ml.orchestration.data_loader.os.path.exists", lambda _: True)
    monkeypatch.setattr(
        "fof8_ml.orchestration.data_loader.pl.read_parquet", lambda _: _pooled_source_df()
    )

    data = loader.load(cfg)

    train_universes = set(data.meta_train["Universe"].to_list())
    val_universes = set(data.meta_val["Universe"].to_list())
    test_universes = set(data.meta_test["Universe"].to_list())

    assert train_universes
    assert val_universes
    assert test_universes
    assert train_universes.isdisjoint(val_universes)
    assert train_universes.isdisjoint(test_universes)
    assert val_universes.isdisjoint(test_universes)
    assert train_universes | val_universes | test_universes == {
        "league_a",
        "league_b",
        "league_c",
        "league_d",
        "league_e",
    }
    assert len(data.X_val) == 5
    assert len(data.X_test) == 5


def test_resolve_year_range_uses_offset_and_count():
    module = _load_process_features_module()
    cfg = _make_cfg(**{"data.year_start_offset": 1, "data.year_count": 30})

    assert module.resolve_year_range(2020, 2100, cfg) == [2021, 2050]


def test_resolve_year_range_clamps_to_final_year():
    module = _load_process_features_module()
    cfg = _make_cfg(**{"data.year_start_offset": 1, "data.year_count": 30})

    assert module.resolve_year_range(2020, 2030, cfg) == [2021, 2030]


def test_normalize_pooled_frame_casts_enum_columns_to_string():
    module = _load_process_features_module()
    df = pl.DataFrame({"College": ["A", "B"], "Score": [1, 2]}).with_columns(
        pl.col("College").cast(pl.Enum(["A", "B"]))
    )

    normalized = module.normalize_pooled_frame(df)

    assert normalized.schema["College"] == pl.String
    assert normalized.schema["Score"] == pl.Int64


def test_write_universe_features_writes_normalized_parquet(tmp_path):
    module = _load_process_features_module()
    frame = pl.DataFrame({"Universe": ["league_a"], "College": ["A"], "Score": [1]}).with_columns(
        pl.col("College").cast(pl.Enum(["A"]))
    )
    normalized = module.normalize_pooled_frame(frame)

    out_file = module.write_universe_features(normalized, tmp_path, "league_a")

    assert out_file == tmp_path / "universes" / "league_a" / "features.parquet"
    loaded = pl.read_parquet(out_file)
    assert loaded.schema["College"] == pl.String
    assert loaded.to_dict(as_series=False) == {
        "Universe": ["league_a"],
        "College": ["A"],
        "Score": [1],
    }
