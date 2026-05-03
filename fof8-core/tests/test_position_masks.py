import polars as pl
from fof8_core.features.constants import MASKABLE_FEATURES, POSITION_FEATURE_MAP
from fof8_core.features.position_masks import apply_position_mask


def test_apply_position_mask_preserves_relevant_and_nulls_irrelevant_features():
    df = pl.DataFrame(
        {
            "Position_Group": ["QB", "RB"],
            "Mean_Short_Passes": [80, 55],
            "Delta_Short_Passes": [10, 8],
            "Mean_Run_Defense": [20, 30],
            "Delta_Run_Defense": [4, 6],
        }
    )

    masked = apply_position_mask(df)

    # QB keeps passing features, RB does not.
    assert masked.filter(pl.col("Position_Group") == "QB")["Mean_Short_Passes"][0] == 80
    assert masked.filter(pl.col("Position_Group") == "RB")["Mean_Short_Passes"][0] is None

    # RB keeps defensive features only if defined for RB in map (it is not).
    assert masked.filter(pl.col("Position_Group") == "QB")["Mean_Run_Defense"][0] is None
    assert masked.filter(pl.col("Position_Group") == "RB")["Mean_Run_Defense"][0] is None


def test_position_mask_constants_include_expected_features():
    assert "Mean_Short_Passes" in MASKABLE_FEATURES
    assert "QB" in POSITION_FEATURE_MAP
    assert "Mean_Short_Passes" in POSITION_FEATURE_MAP["QB"]
