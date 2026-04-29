import logging

import polars as pl

from .loader import FOF8Loader

logger = logging.getLogger(__name__)


def get_draft_class(
    loader: FOF8Loader, year: int, active_team_id: int | None = None
) -> pl.DataFrame:
    """
    Loads and processes the draft class for a specific year with engineered features.

    Includes:
    - Relative Athleticism: Position-relative Z-Scores for combine metrics.
    - Scouting Uncertainty: Spread between Low and High scouted estimates.

    Args:
        loader: An instance of FOF8Loader.
        year: The draft year to process.

    Returns:
        A Polars DataFrame containing the processed features.
    """
    with pl.StringCache():
        lf_rookies = loader.scan_file("rookies.csv", year=year)
        lf_personal = loader.scan_file("draft_personal.csv", year=year)
        lf_info_pre = loader.scan_file("player_information_pre_draft.csv", year=year).select(
            ["Player_ID", "Year", "Year_Born", "Position"]
        )

        # Join, calculate Age, and immediately drop Year_Born
        age_expr = pl.col("Year") - pl.col("Year_Born")
        lf_rookies = (
            lf_rookies.join(lf_info_pre, on=["Player_ID", "Year"], how="left")
            .with_columns(age_expr.alias("Age"), (age_expr**2).alias("Age_Squared"))
            .drop("Year_Born")  # Crucial: Drop this so it doesn't leak to the model!
        )

        # 1. Feature Engineering: Relative Athleticism & Size (Z-Scores by Position Group)
        combine_drills = ["Dash", "Strength", "Agility", "Jump", "Position_Specific"]
        physical_traits = ["Height", "Weight"]
        z_score_targets = combine_drills + physical_traits
        lower_is_better = ["Dash", "Agility"]

        # Step A: Convert 0s to Nulls for combine drills ONLY (FOF8 doesn't generate 0 lb players)
        actual_rookies_cols = lf_rookies.collect_schema().names()
        lf_rookies = lf_rookies.with_columns(
            [
                pl.when(pl.col(c) == 0).then(None).otherwise(pl.col(c)).alias(c)
                for c in combine_drills
                if c in actual_rookies_cols
            ]
        )

        # Step B: Calculate Z-Scores safely for everything
        z_score_exprs = []
        actual_cols = lf_rookies.collect_schema().names()
        for c in [target for target in z_score_targets if target in actual_cols]:
            std_expr = pl.col(c).std().over("Position_Group")

            # Safely calculate Z-score, yielding Null if the drill was skipped
            # OR if the standard deviation is 0
            z_expr = (
                pl.when(std_expr > 0)
                .then((pl.col(c) - pl.col(c).mean().over("Position_Group")) / std_expr)
                .otherwise(None)
            )

            # Flip sign if lower is better (so positive Z always means 'better')
            if c in lower_is_better:
                z_expr = -z_expr

            z_score_exprs.append(z_expr.alias(f"{c}_Z"))

        # Step C: Append the Z-scores WITHOUT dropping the original absolute columns
        lf_rookies = lf_rookies.with_columns(z_score_exprs)

        # We find all 'High_' columns and calculate Delta and Mean
        # NOTE: 'Future_' columns are already dropped by the loader
        high_cols = [c for c in lf_personal.collect_schema().names() if c.startswith("High_")]
        engineering_exprs = []
        for c in high_cols:
            low_col = c.replace("High_", "Low_")
            engineering_exprs.append(
                (pl.col(c) - pl.col(low_col)).alias(c.replace("High_", "Delta_"))
            )
            engineering_exprs.append(
                ((pl.col(c) + pl.col(low_col)) / 2).alias(c.replace("High_", "Mean_"))
            )

        lf_personal = lf_personal.with_columns(engineering_exprs)

        # 3. Drop collinear High/Low columns, relying on Mean and Delta
        low_cols = [c.replace("High_", "Low_") for c in high_cols]
        lf_personal = lf_personal.drop(high_cols + low_cols)

        # 4. Feature Engineering: Coach Scouting Ability
        if active_team_id is not None:
            try:
                lf_staff = loader.scan_file("staff.csv", year=year)
                # Filter for Miami Dolphins and relevant coaching roles
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

                if len(df_staff) > 0:
                    # Extract specific roles safely
                    hc_scouts = df_staff.filter(pl.col("Role") == "Head Coach")["Scouting_Ability"]
                    scout_hc = hc_scouts[0] if len(hc_scouts) > 0 else 0

                    oc_scouts = df_staff.filter(pl.col("Role") == "Offensive Coordinator")[
                        "Scouting_Ability"
                    ]
                    scout_oc = oc_scouts[0] if len(oc_scouts) > 0 else 0

                    dc_scouts = df_staff.filter(pl.col("Role") == "Defensive Coordinator")[
                        "Scouting_Ability"
                    ]
                    scout_dc = dc_scouts[0] if len(dc_scouts) > 0 else 0

                    asst_scouts = df_staff.filter(pl.col("Role") == "Assistant Coach")[
                        "Scouting_Ability"
                    ]
                    scout_asst = asst_scouts.max() if len(asst_scouts) > 0 else 0

                    lf_rookies = lf_rookies.with_columns(
                        [
                            pl.lit(scout_hc).alias("Scout_HC"),
                            pl.lit(scout_oc).alias("Scout_OC"),
                            pl.lit(scout_dc).alias("Scout_DC"),
                            pl.lit(scout_asst).alias("Scout_ASST"),
                        ]
                    )
            except Exception as e:
                logger.warning(f"Could not load staff for year {year}: {e}")

        # Join features on Player_ID and Year
        return lf_rookies.join(lf_personal, on=["Player_ID", "Year"], how="left").collect()


# --- Migrated from fof8-ml ---

PASSING_FEATURES = [
    "Screen_Passes",
    "Short_Passes",
    "Medium_Passes",
    "Long_Passes",
    "Deep_Passes",
    "Third_Down",
    "Run_Frequency",
    "Accuracy",
    "Timing",
    "Sense_Rush",
    "Read_Defense",
    "Two-Minute_Offense",
]

RUSHING_FEATURES = [
    "Power_Inside",
    "Third-Down_Runs",
    "Hole_Recognition",
    "Elusiveness",
    "Speed_Outside",
    "Blitz_Pickup",
]

RECEIVING_FEATURES = [
    "Avoid_Drops",
    "Get_Downfield",
    "Route_Running",
    "Third-Down_Receiving",
    "Big_Play_Receiving",
    "Courage",
    "Adjust_to_Ball",
]

BLOCKING_FEATURES = ["Run_Blocking", "Pass_Blocking", "Blocking_Strength"]

DEFENSIVE_FEATURES = [
    "Run_Defense",
    "Pass_Rush_Technique",
    "Man-to-Man_Defense",
    "Zone_Defense",
    "Bump-and-Run_Defense",
    "Pass_Rush_Strength",
    "Play_Diagnosis",
    "Punishing_Hitter",
    "Intercepting",
]

KICKING_FEATURES = ["Kickoff_Distance", "Kickoff_Hang_Time", "Kicking_Accuracy", "Kicking_Power"]

PUNTING_FEATURES = [
    "Punting_Power",
    "Hang_Time",
    "Directional_Punting",
]

RETURNER_FEATURES = ["Punt_Returns", "Kick_Returns"]

LONG_SNAPPING_FEATURE = ["Long_Snapping"]

KICK_HOLDING_FEATURE = ["Kick_Holding"]

SPECIAL_TEAMS_FEATURE = ["Special_Teams"]


# Expand to include Mean_ and Delta_ prefixes
def expand_features(base_list: list[str]) -> list[str]:
    """
    Expands a list of base feature names into their Mean and Delta equivalents.

    Args:
        base_list: List of base feature names.

    Returns:
        List containing the expanded 'Mean_' and 'Delta_' prefixed feature names.
    """
    expanded = []
    for feat in base_list:
        expanded.append(f"Mean_{feat}")
        expanded.append(f"Delta_{feat}")
    return expanded


# The list of all features that are subject to being masked (nulled)
MASKABLE_FEATURES = expand_features(
    PASSING_FEATURES
    + RUSHING_FEATURES
    + RECEIVING_FEATURES
    + BLOCKING_FEATURES
    + DEFENSIVE_FEATURES
    + KICKING_FEATURES
    + PUNTING_FEATURES
    + RETURNER_FEATURES
    + LONG_SNAPPING_FEATURE
    + KICK_HOLDING_FEATURE
    + SPECIAL_TEAMS_FEATURE
)

# Explicit map of which features to KEEP for each position group.
# Anything in MASKABLE_FEATURES not in this list will be set to Null for that position.
POSITION_FEATURE_MAP = {
    "QB": expand_features(PASSING_FEATURES + KICK_HOLDING_FEATURE),
    "RB": expand_features(RUSHING_FEATURES + RECEIVING_FEATURES + SPECIAL_TEAMS_FEATURE),
    "FB": expand_features(
        RUSHING_FEATURES + RECEIVING_FEATURES + BLOCKING_FEATURES + SPECIAL_TEAMS_FEATURE
    ),
    "WR": expand_features(
        RUSHING_FEATURES
        + RECEIVING_FEATURES
        + BLOCKING_FEATURES
        + RETURNER_FEATURES
        + SPECIAL_TEAMS_FEATURE
    ),
    "TE": expand_features(
        RUSHING_FEATURES + RECEIVING_FEATURES + BLOCKING_FEATURES + SPECIAL_TEAMS_FEATURE
    ),
    "C": expand_features(BLOCKING_FEATURES + LONG_SNAPPING_FEATURE),
    "G": expand_features(BLOCKING_FEATURES + LONG_SNAPPING_FEATURE),
    "T": expand_features(BLOCKING_FEATURES + LONG_SNAPPING_FEATURE),
    "DE": expand_features(DEFENSIVE_FEATURES + SPECIAL_TEAMS_FEATURE),
    "DT": expand_features(DEFENSIVE_FEATURES + SPECIAL_TEAMS_FEATURE),
    "OLB": expand_features(DEFENSIVE_FEATURES + SPECIAL_TEAMS_FEATURE),
    "ILB": expand_features(DEFENSIVE_FEATURES + SPECIAL_TEAMS_FEATURE),
    "CB": expand_features(DEFENSIVE_FEATURES + RETURNER_FEATURES + SPECIAL_TEAMS_FEATURE),
    "S": expand_features(DEFENSIVE_FEATURES + RETURNER_FEATURES + SPECIAL_TEAMS_FEATURE),
    "K": expand_features(KICKING_FEATURES + KICK_HOLDING_FEATURE),
    "P": expand_features(PUNTING_FEATURES + KICK_HOLDING_FEATURE),
    "LS": expand_features(BLOCKING_FEATURES + LONG_SNAPPING_FEATURE + SPECIAL_TEAMS_FEATURE),
}

# Special handling for "All" or if we want to add more general ones
ALL_POSITIONS = list(POSITION_FEATURE_MAP.keys())


def apply_position_mask(df: pl.DataFrame) -> pl.DataFrame:
    """
    Nulls out features that are irrelevant for specific positions.

    This function ensures that features which are not meaningful or captured
    for certain player positions (e.g., passing features for a defensive tackle)
    are strictly set to Null. This prevents models from inappropriately learning
    from default or zero values in these columns.

    Migrated from fof8-ml for centralized feature store management.
    """
    # Get all maskable columns that actually exist in the dataframe
    existing_maskable = [col for col in MASKABLE_FEATURES if col in df.columns]

    # Pre-calculate which positions should have each feature nulled
    feature_null_positions = {}
    for col in existing_maskable:
        null_positions = []
        for pos, keeps in POSITION_FEATURE_MAP.items():
            if col not in keeps:
                null_positions.append(pos)
        if null_positions:
            feature_null_positions[col] = null_positions

    # Build all mask expressions
    mask_exprs = []
    for col, null_positions in feature_null_positions.items():
        mask_exprs.append(
            pl.when(pl.col("Position_Group").cast(pl.String).is_in(null_positions))
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        )

    # Apply the masks simultaneously
    if mask_exprs:
        df = df.with_columns(mask_exprs)

    return df
