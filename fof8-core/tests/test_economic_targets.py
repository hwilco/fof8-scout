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

    df_peak = pl.DataFrame(
        {
            "Player_ID": [1, 2, 5],
            "Peak_Overall": [80.0, None, 50.0],
        }
    )
    df_merit = pl.DataFrame(
        {
            "Player_ID": [1, 3, 5],
            "Career_Merit_Cap_Share": [0.25, -0.10, None],
        }
    )
    df_outcomes = pl.DataFrame(
        {
            "Player_ID": [1, 4],
            "Career_Games_Played": [100, None],
        }
    )

    monkeypatch.setattr("fof8_core.targets.economic.get_peak_overall", lambda _: df_peak)
    monkeypatch.setattr("fof8_core.targets.economic.get_merit_cap_share", lambda _: df_merit)
    monkeypatch.setattr("fof8_core.targets.economic.get_career_outcomes", lambda _: df_outcomes)

    out = get_economic_targets(mock_loader, merit_threshold=0.2).sort("Player_ID")

    assert out.columns == ECONOMIC_OUTPUT_COLUMNS

    # Union-of-IDs is preserved across peak, merit, and outcomes inputs.
    assert out["Player_ID"].to_list() == [1, 2, 3, 4, 5]

    row1 = out.filter(pl.col("Player_ID") == 1)
    assert row1["DPO"][0] == 20.0  # 80 * 0.25
    assert row1["Cleared_Sieve"][0] == 1  # 0.25 > 0.2
    assert row1["Economic_Success"][0] == 1
    assert row1["Positive_DPO"][0] == 20.0
    assert row1["Positive_Career_Merit_Cap_Share"][0] == 0.25

    row3 = out.filter(pl.col("Player_ID") == 3)
    assert row3["Peak_Overall"][0] == 0
    assert row3["Career_Games_Played"][0] == 0
    assert row3["Career_Merit_Cap_Share"][0] == -0.10
    assert row3["DPO"][0] == 0.0
    assert row3["Positive_DPO"][0] == 0.0
    assert row3["Positive_Career_Merit_Cap_Share"][0] == 0.0
    assert row3["Economic_Success"][0] == 0
    assert row3["Cleared_Sieve"][0] == 0

    row4 = out.filter(pl.col("Player_ID") == 4)
    assert row4["Peak_Overall"][0] == 0
    assert row4["Career_Merit_Cap_Share"][0] == 0
    assert row4["Career_Games_Played"][0] == 0
    assert row4["DPO"][0] == 0
    assert row4["Positive_DPO"][0] == 0

    row5 = out.filter(pl.col("Player_ID") == 5)
    assert row5["Career_Merit_Cap_Share"][0] == 0
    assert row5["DPO"][0] == 0
    assert row5["Positive_Career_Merit_Cap_Share"][0] == 0
    assert row5["Economic_Success"][0] == 0


def test_add_economic_derived_columns_handles_null_negative_and_threshold():
    df = pl.DataFrame(
        {
            "Player_ID": [1, 2, 3, 4],
            "Peak_Overall": [80.0, 50.0, None, 40.0],
            "Career_Merit_Cap_Share": [0.25, -0.1, 0.0, None],
            "Career_Games_Played": [100, None, 0, 20],
        }
    )
    out = add_economic_derived_columns(df, merit_threshold=0.2).sort("Player_ID")

    assert out.filter(pl.col("Player_ID") == 1)["DPO"][0] == 20.0
    assert out.filter(pl.col("Player_ID") == 1)["Cleared_Sieve"][0] == 1
    assert out.filter(pl.col("Player_ID") == 1)["Economic_Success"][0] == 1

    assert out.filter(pl.col("Player_ID") == 2)["DPO"][0] == -5.0
    assert out.filter(pl.col("Player_ID") == 2)["Positive_DPO"][0] == 0.0
    assert out.filter(pl.col("Player_ID") == 2)["Positive_Career_Merit_Cap_Share"][0] == 0.0
    assert out.filter(pl.col("Player_ID") == 2)["Cleared_Sieve"][0] == 0

    assert out.filter(pl.col("Player_ID") == 3)["Peak_Overall"][0] == 0.0
    assert out.filter(pl.col("Player_ID") == 3)["Economic_Success"][0] == 0
    assert out.filter(pl.col("Player_ID") == 3)["Cleared_Sieve"][0] == 0

    assert out.filter(pl.col("Player_ID") == 4)["Career_Merit_Cap_Share"][0] == 0.0
    assert out.filter(pl.col("Player_ID") == 4)["DPO"][0] == 0.0
