import logging

import polars as pl

from fof8_core.loader import FOF8Loader

logger = logging.getLogger(__name__)

COMBINE_DRILLS = ["Dash", "Strength", "Agility", "Jump", "Position_Specific"]
PHYSICAL_TRAITS = ["Height", "Weight"]
LOWER_IS_BETTER_DRILLS = ["Dash", "Agility"]


def _add_age_features(lf_rookies: pl.LazyFrame, lf_info_pre: pl.LazyFrame) -> pl.LazyFrame:
    age_expr = pl.col("Year") - pl.col("Year_Born")
    return (
        lf_rookies.join(lf_info_pre, on=["Player_ID", "Year"], how="left")
        .with_columns(age_expr.alias("Age"), (age_expr**2).alias("Age_Squared"))
        .drop("Year_Born")
    )


def _add_position_relative_z_scores(lf_rookies: pl.LazyFrame) -> pl.LazyFrame:
    z_score_targets = COMBINE_DRILLS + PHYSICAL_TRAITS

    actual_rookies_cols = lf_rookies.collect_schema().names()
    lf_rookies = lf_rookies.with_columns(
        [
            pl.when(pl.col(c) == 0).then(None).otherwise(pl.col(c)).alias(c)
            for c in COMBINE_DRILLS
            if c in actual_rookies_cols
        ]
    )

    z_score_exprs = []
    actual_cols = lf_rookies.collect_schema().names()
    for col_name in [target for target in z_score_targets if target in actual_cols]:
        std_expr = pl.col(col_name).std().over("Position_Group")
        z_expr = (
            pl.when(std_expr > 0)
            .then((pl.col(col_name) - pl.col(col_name).mean().over("Position_Group")) / std_expr)
            .otherwise(None)
        )

        if col_name in LOWER_IS_BETTER_DRILLS:
            z_expr = -z_expr

        z_score_exprs.append(z_expr.alias(f"{col_name}_Z"))

    return lf_rookies.with_columns(z_score_exprs)


def _add_scouting_uncertainty_features(lf_personal: pl.LazyFrame) -> pl.LazyFrame:
    high_cols = [c for c in lf_personal.collect_schema().names() if c.startswith("High_")]
    engineering_exprs = []
    for high_col in high_cols:
        low_col = high_col.replace("High_", "Low_")
        engineering_exprs.append(
            (pl.col(high_col) - pl.col(low_col)).alias(high_col.replace("High_", "Delta_"))
        )
        engineering_exprs.append(
            ((pl.col(high_col) + pl.col(low_col)) / 2).alias(high_col.replace("High_", "Mean_"))
        )

    low_cols = [c.replace("High_", "Low_") for c in high_cols]
    return lf_personal.with_columns(engineering_exprs).drop(high_cols + low_cols)


def _add_staff_scouting_features(
    lf_rookies: pl.LazyFrame, loader: FOF8Loader, year: int, active_team_id: int | None
) -> pl.LazyFrame:
    if active_team_id is None:
        return lf_rookies

    try:
        lf_staff = loader.scan_file("staff.csv", year=year)
        df_staff = lf_staff.filter(
            (pl.col("Current_Team") == active_team_id)
            & (
                pl.col("Role").is_in(
                    [
                        "Head Coach",
                        "Offensive Coordinator",
                        "Defensive Coordinator",
                        "Assistant Coach",
                    ]
                )
            )
        ).collect()

        if len(df_staff) == 0:
            return lf_rookies

        hc_scouts = df_staff.filter(pl.col("Role") == "Head Coach")["Scouting_Ability"]
        scout_hc = hc_scouts[0] if len(hc_scouts) > 0 else 0

        oc_scouts = df_staff.filter(pl.col("Role") == "Offensive Coordinator")["Scouting_Ability"]
        scout_oc = oc_scouts[0] if len(oc_scouts) > 0 else 0

        dc_scouts = df_staff.filter(pl.col("Role") == "Defensive Coordinator")["Scouting_Ability"]
        scout_dc = dc_scouts[0] if len(dc_scouts) > 0 else 0

        asst_scouts = df_staff.filter(pl.col("Role") == "Assistant Coach")["Scouting_Ability"]
        scout_asst = asst_scouts.max() if len(asst_scouts) > 0 else 0

        return lf_rookies.with_columns(
            [
                pl.lit(scout_hc).alias("Scout_HC"),
                pl.lit(scout_oc).alias("Scout_OC"),
                pl.lit(scout_dc).alias("Scout_DC"),
                pl.lit(scout_asst).alias("Scout_ASST"),
            ]
        )
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Could not load staff for year {year}: {exc}")
        return lf_rookies


def get_draft_class(
    loader: FOF8Loader, year: int, active_team_id: int | None = None
) -> pl.DataFrame:
    with pl.StringCache():
        lf_rookies = loader.scan_file("rookies.csv", year=year)
        lf_personal = loader.scan_file("draft_personal.csv", year=year)
        lf_info_pre = loader.scan_file("player_information_pre_draft.csv", year=year).select(
            ["Player_ID", "Year", "Year_Born", "Position"]
        )

        lf_rookies = _add_age_features(lf_rookies, lf_info_pre)
        lf_rookies = _add_position_relative_z_scores(lf_rookies)
        lf_personal = _add_scouting_uncertainty_features(lf_personal)
        lf_rookies = _add_staff_scouting_features(lf_rookies, loader, year, active_team_id)

        return lf_rookies.join(lf_personal, on=["Player_ID", "Year"], how="left").collect()
