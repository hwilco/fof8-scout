"""Tests for canonical economic target construction."""

from unittest.mock import MagicMock

import polars as pl
from fof8_core.targets.economic import (
    ECONOMIC_OUTPUT_COLUMNS,
    add_economic_derived_columns,
    get_economic_targets,
)


def test_get_economic_targets_expected_columns_and_semantics(monkeypatch):
    mock_loader = MagicMock()

    df_merit = pl.DataFrame(
        {
            "Player_ID": [1, 3, 5, 8],
            "Career_Merit_Cap_Share": [0.25, -0.10, None, 0.21],
        }
    )

    monkeypatch.setattr("fof8_core.targets.economic.get_merit_cap_share", lambda _: df_merit)

    out = get_economic_targets(mock_loader, merit_threshold=0.2).sort("Player_ID")

    assert out.columns == ECONOMIC_OUTPUT_COLUMNS

    # Economic targets are built from economic source IDs only.
    assert out["Player_ID"].to_list() == [1, 3, 5, 8]

    row1 = out.filter(pl.col("Player_ID") == 1)
    assert row1["Cleared_Sieve"][0] == 1  # 0.25 > 0.2
    assert row1["Economic_Success"][0] == 1
    assert row1["Positive_Career_Merit_Cap_Share"][0] == 0.25

    row3 = out.filter(pl.col("Player_ID") == 3)
    assert row3["Career_Merit_Cap_Share"][0] == -0.10
    assert row3["Positive_Career_Merit_Cap_Share"][0] == 0.0
    assert row3["Economic_Success"][0] == 0
    assert row3["Cleared_Sieve"][0] == 0

    row5 = out.filter(pl.col("Player_ID") == 5)
    assert row5["Career_Merit_Cap_Share"][0] == 0
    assert row5["Positive_Career_Merit_Cap_Share"][0] == 0
    assert row5["Economic_Success"][0] == 0

    assert "DPO" not in out.columns
    assert "Positive_DPO" not in out.columns
    assert "Peak_Overall" not in out.columns
    assert "Career_Games_Played" not in out.columns


def test_add_economic_derived_columns_handles_null_negative_and_threshold():
    df = pl.DataFrame(
        {
            "Player_ID": [1, 2, 3, 4],
            "Career_Merit_Cap_Share": [0.25, -0.1, 0.0, None],
        }
    )
    out = add_economic_derived_columns(df, merit_threshold=0.2).sort("Player_ID")

    assert out.filter(pl.col("Player_ID") == 1)["Cleared_Sieve"][0] == 1
    assert out.filter(pl.col("Player_ID") == 1)["Economic_Success"][0] == 1

    assert out.filter(pl.col("Player_ID") == 2)["Positive_Career_Merit_Cap_Share"][0] == 0.0
    assert out.filter(pl.col("Player_ID") == 2)["Cleared_Sieve"][0] == 0

    assert out.filter(pl.col("Player_ID") == 3)["Economic_Success"][0] == 0
    assert out.filter(pl.col("Player_ID") == 3)["Cleared_Sieve"][0] == 0

    assert out.filter(pl.col("Player_ID") == 4)["Career_Merit_Cap_Share"][0] == 0.0
