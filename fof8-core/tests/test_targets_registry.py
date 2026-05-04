"""Tests for target registry lookup, built-in registration, and error handling."""

import polars as pl
import pytest
from fof8_core.targets.registry import TARGET_REGISTRY, _register_builtin_targets, get_target


def test_registry_lookup_returns_target_frame(mock_loader, tmp_path):
    year_dir_2144 = tmp_path / "DRAFT003" / "2144"
    year_dir_2144.mkdir(parents=True)
    pl.DataFrame(
        {
            "Player_ID": [1, 2],
            "Career_Games_Played": [160, None],
            "Number_of_Seasons": [10, None],
            "Championship_Rings": [2, None],
            "Hall_of_Fame_Flag": [1, None],
        }
    ).write_csv(year_dir_2144 / "player_information_post_sim.csv")

    df = get_target("career_outcomes", mock_loader)

    assert isinstance(df, pl.DataFrame)
    assert df.shape == (2, 5)
    assert df.filter(pl.col("Player_ID") == 1)["Career_Games_Played"][0] == 160
    assert df.filter(pl.col("Player_ID") == 2)["Career_Games_Played"][0] == 0


def test_registry_unknown_target_error(mock_loader):
    with pytest.raises(ValueError, match="Unknown target 'does_not_exist'"):
        get_target("does_not_exist", mock_loader)


def test_registry_contains_builtin_targets():
    _register_builtin_targets()
    required = {
        "career_outcomes",
        "annual_financials",
        "peak_overall",
        "merit_cap_share",
        "economic_targets",
        "dpo_targets",
        "draft_outcome_targets",
        "career_value_metrics",
        "awards",
    }
    assert required.issubset(set(TARGET_REGISTRY.keys()))


def test_registry_passes_kwargs_to_parameterized_targets(monkeypatch, mock_loader):
    def fake_economic(loader, merit_threshold=0.0):
        assert loader is mock_loader
        return pl.DataFrame({"merit_threshold": [merit_threshold]})

    monkeypatch.setitem(TARGET_REGISTRY, "economic_targets", fake_economic)
    out = get_target("economic_targets", mock_loader, merit_threshold=0.2)
    assert out["merit_threshold"][0] == 0.2
