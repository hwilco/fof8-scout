"""Tests for composite target builders."""

from unittest.mock import MagicMock

import polars as pl
from fof8_core.targets.composite import COMPOSITE_OUTPUT_COLUMNS, add_dpo_columns, get_dpo_targets


def test_add_dpo_columns_handles_nulls_and_clipping():
    df = pl.DataFrame(
        {
            "Player_ID": [1, 2, 3],
            "Peak_Overall": [80.0, None, 40.0],
            "Career_Merit_Cap_Share": [0.25, -0.1, None],
        }
    )
    out = add_dpo_columns(df).sort("Player_ID")

    assert out.filter(pl.col("Player_ID") == 1)["DPO"][0] == 20.0
    assert out.filter(pl.col("Player_ID") == 1)["Positive_DPO"][0] == 20.0
    assert out.filter(pl.col("Player_ID") == 2)["DPO"][0] == 0.0
    assert out.filter(pl.col("Player_ID") == 2)["Positive_DPO"][0] == 0.0
    assert out.filter(pl.col("Player_ID") == 3)["DPO"][0] == 0.0
    assert out.filter(pl.col("Player_ID") == 3)["Positive_DPO"][0] == 0.0


def test_get_dpo_targets_joins_union_ids(monkeypatch):
    mock_loader = MagicMock()
    df_peak = pl.DataFrame({"Player_ID": [1, 2], "Peak_Overall": [80.0, 40.0]})
    df_merit = pl.DataFrame({"Player_ID": [1, 3], "Career_Merit_Cap_Share": [0.25, -0.2]})

    monkeypatch.setattr(
        "fof8_core.targets.composite.get_peak_overall", lambda _loader, **_kwargs: df_peak
    )
    monkeypatch.setattr("fof8_core.targets.composite.get_merit_cap_share", lambda _: df_merit)

    out = get_dpo_targets(mock_loader).sort("Player_ID")
    assert out.columns == COMPOSITE_OUTPUT_COLUMNS
    assert out["Player_ID"].to_list() == [1, 2, 3]
    assert out.filter(pl.col("Player_ID") == 1)["DPO"][0] == 20.0
    assert out.filter(pl.col("Player_ID") == 3)["DPO"][0] == 0.0
