"""Tests for draft outcome target bundle builders."""

from unittest.mock import MagicMock

import polars as pl
from fof8_core.targets.draft_outcomes import DRAFT_OUTCOME_OUTPUT_COLUMNS, get_draft_outcome_targets


def test_get_draft_outcome_targets_builds_bundle_in_stable_order(monkeypatch):
    mock_loader = MagicMock()
    df_economic = pl.DataFrame(
        {
            "Player_ID": [1, 2],
            "Cleared_Sieve": [1, 0],
            "Economic_Success": [1, 0],
            "Career_Merit_Cap_Share": [0.25, -0.1],
            "Positive_Career_Merit_Cap_Share": [0.25, 0.0],
        }
    )
    df_dpo = pl.DataFrame({"Player_ID": [1, 3], "DPO": [20.0, -2.0], "Positive_DPO": [20.0, 0.0]})
    df_peak = pl.DataFrame({"Player_ID": [1, 3], "Peak_Overall": [80.0, 10.0]})
    df_top3 = pl.DataFrame({"Player_ID": [1, 2], "Top3_Mean_Current_Overall": [76.0, 40.0]})
    df_outcomes = pl.DataFrame({"Player_ID": [2, 3], "Career_Games_Played": [16, None]})
    df_draft_utility = pl.DataFrame(
        {
            "Player_ID": [1, 3],
            "Control_Y1_Current_Overall": [50.0, 12.0],
            "Control_Y2_Current_Overall": [60.0, None],
            "Control_Y3_Current_Overall": [70.0, None],
            "Control_Y4_Current_Overall": [80.0, None],
            "Control_Window_Mean_Current_Overall": [65.0, None],
            "Control_Window_Discounted_Mean_Current_Overall": [63.0, None],
        }
    )

    monkeypatch.setattr(
        "fof8_core.targets.draft_outcomes.get_economic_targets",
        lambda _loader, merit_threshold=0: df_economic,
    )
    monkeypatch.setattr("fof8_core.targets.draft_outcomes.get_dpo_targets", lambda _: df_dpo)
    monkeypatch.setattr(
        "fof8_core.targets.draft_outcomes.get_peak_overall", lambda _loader, **_kwargs: df_peak
    )
    monkeypatch.setattr(
        "fof8_core.targets.draft_outcomes.get_top3_mean_current_overall",
        lambda _loader, **_kwargs: df_top3,
    )
    monkeypatch.setattr(
        "fof8_core.targets.draft_outcomes.get_career_outcomes", lambda _: df_outcomes
    )
    monkeypatch.setattr(
        "fof8_core.targets.draft_outcomes.get_draft_utility_targets",
        lambda _: df_draft_utility,
    )

    out = get_draft_outcome_targets(mock_loader).sort("Player_ID")
    assert out.columns == DRAFT_OUTCOME_OUTPUT_COLUMNS
    assert out["Player_ID"].to_list() == [1, 2, 3]
    assert out.filter(pl.col("Player_ID") == 2)["DPO"][0] == 0.0
    assert out.filter(pl.col("Player_ID") == 3)["Career_Games_Played"][0] == 0
    assert out.filter(pl.col("Player_ID") == 1)["Control_Window_Mean_Current_Overall"][0] == 65.0
    assert out.filter(pl.col("Player_ID") == 3)["Control_Window_Mean_Current_Overall"][0] is None
