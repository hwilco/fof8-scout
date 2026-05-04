"""Tests for talent target builders."""

from unittest.mock import MagicMock

import polars as pl
import pytest
from fof8_core.targets import financial
from fof8_core.targets.talent import TALENT_OUTPUT_COLUMNS, get_peak_overall


def test_get_peak_overall_computes_top_k_mean():
    loader = MagicMock()
    loader.scan_file.return_value = pl.LazyFrame(
        [
            {"Player_ID": 1, "Year": 2024, "Scouting": "Exhibition", "Current_Overall": 60},
            {"Player_ID": 1, "Year": 2025, "Scouting": "Exhibition", "Current_Overall": 80},
            {"Player_ID": 1, "Year": 2026, "Scouting": "Exhibition", "Current_Overall": 70},
            {"Player_ID": 1, "Year": 2027, "Scouting": "Exhibition", "Current_Overall": 50},
            {"Player_ID": 2, "Year": 2024, "Scouting": "Training", "Current_Overall": 90},
            {"Player_ID": 2, "Year": 2024, "Scouting": "Exhibition", "Current_Overall": 40},
        ]
    )

    out = get_peak_overall(loader, k=3).sort("Player_ID")

    assert out.columns == TALENT_OUTPUT_COLUMNS
    assert out["Player_ID"].to_list() == [1, 2]
    assert out.filter(pl.col("Player_ID") == 1)["Peak_Overall"][0] == 70.0
    assert out.filter(pl.col("Player_ID") == 2)["Peak_Overall"][0] == 40.0


def test_get_peak_overall_timeframe_limits_to_first_years():
    loader = MagicMock()
    loader.scan_file.return_value = pl.LazyFrame(
        [
            {"Player_ID": 1, "Year": 2024, "Scouting": "Exhibition", "Current_Overall": 60},
            {"Player_ID": 1, "Year": 2025, "Scouting": "Exhibition", "Current_Overall": 80},
            {"Player_ID": 1, "Year": 2026, "Scouting": "Exhibition", "Current_Overall": 70},
            {"Player_ID": 1, "Year": 2027, "Scouting": "Exhibition", "Current_Overall": 90},
            {"Player_ID": 2, "Year": 2026, "Scouting": "Exhibition", "Current_Overall": 55},
            {"Player_ID": 2, "Year": 2027, "Scouting": "Exhibition", "Current_Overall": 65},
        ]
    )

    out = get_peak_overall(loader, k=2, timeframe=2).sort("Player_ID")
    assert out.filter(pl.col("Player_ID") == 1)["Peak_Overall"][0] == 70.0  # 80, 60
    assert out.filter(pl.col("Player_ID") == 2)["Peak_Overall"][0] == 60.0  # 65, 55


def test_get_peak_overall_rejects_non_positive_timeframe():
    loader = MagicMock()
    with pytest.raises(ValueError, match="timeframe must be >= 1"):
        get_peak_overall(loader, timeframe=0)


def test_get_peak_overall_timeframe_greater_than_career_length_uses_available_years():
    loader = MagicMock()
    loader.scan_file.return_value = pl.LazyFrame(
        [
            {"Player_ID": 1, "Year": 2024, "Scouting": "Exhibition", "Current_Overall": 60},
            {"Player_ID": 1, "Year": 2025, "Scouting": "Exhibition", "Current_Overall": 80},
        ]
    )

    out = get_peak_overall(loader, k=3, timeframe=5)
    assert out.filter(pl.col("Player_ID") == 1)["Peak_Overall"][0] == 70.0


def test_get_peak_overall_k_greater_than_available_observations():
    loader = MagicMock()
    loader.scan_file.return_value = pl.LazyFrame(
        [
            {"Player_ID": 1, "Year": 2024, "Scouting": "Exhibition", "Current_Overall": 50},
            {"Player_ID": 1, "Year": 2025, "Scouting": "Exhibition", "Current_Overall": 70},
        ]
    )

    out = get_peak_overall(loader, k=5)
    assert out.filter(pl.col("Player_ID") == 1)["Peak_Overall"][0] == 60.0


def test_get_peak_overall_rejects_non_positive_k():
    loader = MagicMock()
    loader.scan_file.return_value = pl.LazyFrame(
        [
            {"Player_ID": 1, "Year": 2024, "Scouting": "Exhibition", "Current_Overall": 50},
        ]
    )

    with pytest.raises(ValueError, match="k must be >= 1"):
        get_peak_overall(loader, k=0)


def test_get_peak_overall_handles_null_year_by_excluding_it_with_timeframe():
    loader = MagicMock()
    loader.scan_file.return_value = pl.LazyFrame(
        [
            {"Player_ID": 1, "Year": 2024, "Scouting": "Exhibition", "Current_Overall": 60},
            {"Player_ID": 1, "Year": None, "Scouting": "Exhibition", "Current_Overall": 90},
            {"Player_ID": 1, "Year": 2025, "Scouting": "Exhibition", "Current_Overall": 80},
        ]
    )

    out = get_peak_overall(loader, k=3, timeframe=2)
    assert out.filter(pl.col("Player_ID") == 1)["Peak_Overall"][0] == 70.0


def test_get_peak_overall_missing_year_column_raises_with_timeframe():
    loader = MagicMock()
    loader.scan_file.return_value = pl.LazyFrame(
        [
            {"Player_ID": 1, "Scouting": "Exhibition", "Current_Overall": 60},
            {"Player_ID": 1, "Scouting": "Exhibition", "Current_Overall": 80},
        ]
    )

    with pytest.raises(Exception):
        get_peak_overall(loader, k=2, timeframe=2)


def test_financial_module_no_longer_exports_peak_overall():
    assert not hasattr(financial, "get_peak_overall")
