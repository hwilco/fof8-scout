"""Tests for rookie control-window draft utility target builders."""

from unittest.mock import MagicMock

import polars as pl
import pytest
from fof8_core.targets.draft_utility import (
    DRAFT_UTILITY_OUTPUT_COLUMNS,
    get_draft_utility_targets,
)


def test_get_draft_utility_targets_extracts_control_years_from_draft_year():
    loader = MagicMock()

    def scan_file(name: str):
        if name == "rookies.csv":
            return pl.LazyFrame(
                [
                    {"Player_ID": 1, "Year": 2024},
                    {"Player_ID": 2, "Year": 2025},
                ]
            )
        if name == "player_ratings_season_*.csv":
            return pl.LazyFrame(
                [
                    {"Player_ID": 1, "Year": 2024, "Scouting": "Exhibition", "Current_Overall": 40},
                    {"Player_ID": 1, "Year": 2025, "Scouting": "Exhibition", "Current_Overall": 50},
                    {"Player_ID": 1, "Year": 2026, "Scouting": "Exhibition", "Current_Overall": 60},
                    {"Player_ID": 1, "Year": 2027, "Scouting": "Exhibition", "Current_Overall": 70},
                    {"Player_ID": 1, "Year": 2028, "Scouting": "Exhibition", "Current_Overall": 80},
                    {"Player_ID": 2, "Year": 2025, "Scouting": "Exhibition", "Current_Overall": 20},
                    {"Player_ID": 2, "Year": 2026, "Scouting": "Training", "Current_Overall": 99},
                    {"Player_ID": 2, "Year": 2026, "Scouting": "Exhibition", "Current_Overall": 30},
                    {"Player_ID": 2, "Year": 2027, "Scouting": "Exhibition", "Current_Overall": 40},
                    {"Player_ID": 2, "Year": 2028, "Scouting": "Exhibition", "Current_Overall": 50},
                ]
            )
        raise AssertionError(name)

    loader.scan_file.side_effect = scan_file

    out = get_draft_utility_targets(loader).sort("Player_ID")

    assert out.columns == DRAFT_UTILITY_OUTPUT_COLUMNS
    assert out.filter(pl.col("Player_ID") == 1)["Control_Y1_Current_Overall"][0] == 40
    assert out.filter(pl.col("Player_ID") == 1)["Control_Y4_Current_Overall"][0] == 70
    assert out.filter(pl.col("Player_ID") == 2)["Control_Y2_Current_Overall"][0] == 30
    assert out.filter(pl.col("Player_ID") == 2)["Control_Y4_Current_Overall"][0] == 50


def test_get_draft_utility_targets_computes_mean_and_discounted_mean_for_complete_windows():
    loader = MagicMock()

    def scan_file(name: str):
        if name == "rookies.csv":
            return pl.LazyFrame([{"Player_ID": 1, "Year": 2024}])
        if name == "player_ratings_season_*.csv":
            return pl.LazyFrame(
                [
                    {"Player_ID": 1, "Year": 2024, "Scouting": "Exhibition", "Current_Overall": 40},
                    {"Player_ID": 1, "Year": 2025, "Scouting": "Exhibition", "Current_Overall": 50},
                    {"Player_ID": 1, "Year": 2026, "Scouting": "Exhibition", "Current_Overall": 60},
                    {"Player_ID": 1, "Year": 2027, "Scouting": "Exhibition", "Current_Overall": 70},
                ]
            )
        raise AssertionError(name)

    loader.scan_file.side_effect = scan_file
    out = get_draft_utility_targets(loader)

    row = out.row(0, named=True)
    assert row["Control_Window_Mean_Current_Overall"] == 55.0
    assert row["Control_Window_Discounted_Mean_Current_Overall"] == pytest.approx(
        (40 * 1.00 + 50 * 0.95 + 60 * 0.85 + 70 * 0.75) / (1.00 + 0.95 + 0.85 + 0.75)
    )


def test_get_draft_utility_targets_right_censors_incomplete_control_windows():
    loader = MagicMock()

    def scan_file(name: str):
        if name == "rookies.csv":
            return pl.LazyFrame([{"Player_ID": 1, "Year": 2024}])
        if name == "player_ratings_season_*.csv":
            return pl.LazyFrame(
                [
                    {"Player_ID": 1, "Year": 2024, "Scouting": "Exhibition", "Current_Overall": 40},
                    {"Player_ID": 1, "Year": 2025, "Scouting": "Exhibition", "Current_Overall": 50},
                    {"Player_ID": 1, "Year": 2026, "Scouting": "Exhibition", "Current_Overall": 60},
                ]
            )
        raise AssertionError(name)

    loader.scan_file.side_effect = scan_file
    out = get_draft_utility_targets(loader)

    row = out.row(0, named=True)
    assert row["Control_Y1_Current_Overall"] == 40
    assert row["Control_Y3_Current_Overall"] == 60
    assert row["Control_Y4_Current_Overall"] is None
    assert row["Control_Window_Mean_Current_Overall"] is None
    assert row["Control_Window_Discounted_Mean_Current_Overall"] is None


def test_get_draft_utility_targets_preserves_missing_internal_years():
    loader = MagicMock()

    def scan_file(name: str):
        if name == "rookies.csv":
            return pl.LazyFrame([{"Player_ID": 1, "Year": 2024}])
        if name == "player_ratings_season_*.csv":
            return pl.LazyFrame(
                [
                    {"Player_ID": 1, "Year": 2024, "Scouting": "Exhibition", "Current_Overall": 40},
                    {"Player_ID": 1, "Year": 2026, "Scouting": "Exhibition", "Current_Overall": 60},
                    {"Player_ID": 1, "Year": 2027, "Scouting": "Exhibition", "Current_Overall": 70},
                ]
            )
        raise AssertionError(name)

    loader.scan_file.side_effect = scan_file
    out = get_draft_utility_targets(loader)

    row = out.row(0, named=True)
    assert row["Control_Y1_Current_Overall"] == 40
    assert row["Control_Y2_Current_Overall"] is None
    assert row["Control_Y3_Current_Overall"] == 60
    assert row["Control_Window_Mean_Current_Overall"] is None


def test_get_draft_utility_targets_rejects_invalid_control_years_and_missing_discounts():
    loader = MagicMock()

    with pytest.raises(ValueError, match="control_years must contain at least one year"):
        get_draft_utility_targets(loader, control_years=[])

    with pytest.raises(ValueError, match="control_years must contain only positive integers"):
        get_draft_utility_targets(loader, control_years=[0, 1])

    with pytest.raises(ValueError, match="control_years must not contain duplicate years"):
        get_draft_utility_targets(loader, control_years=[1, 1, 2])

    with pytest.raises(ValueError, match="control_years must be sorted in ascending order"):
        get_draft_utility_targets(loader, control_years=[2, 1])

    with pytest.raises(ValueError, match="control_discounts is missing values"):
        get_draft_utility_targets(loader, control_years=[1, 2], control_discounts={1: 1.0})
