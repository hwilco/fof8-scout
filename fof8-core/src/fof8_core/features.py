import polars as pl
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .loader import FOF8Loader


def get_draft_class(loader: FOF8Loader, year: int, active_team_id: int = None) -> pl.DataFrame:
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
                for c in combine_drills if c in actual_rookies_cols
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
        # NOTE: We explicitly skip 'Future_' columns as they are unpopulated in our datasets
        high_cols = [
            c
            for c in lf_personal.collect_schema().names()
            if c.startswith("High_") and "Future_" not in c
        ]
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
        # Also drop the dead 'Future_' columns explicitly
        dead_future_cols = [c for c in lf_personal.collect_schema().names() if "Future_" in c]
        lf_personal = lf_personal.drop(high_cols + low_cols + dead_future_cols)

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
                print(f"Warning: Could not load staff for year {year}: {e}")

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
def expand_features_ml_v1(base_list):
    expanded = []
    for feat in base_list:
        expanded.append(f"Mean_{feat}")
        expanded.append(f"Delta_{feat}")
    return expanded


# The list of all features that are subject to being masked (nulled)
MASKABLE_FEATURES_ML_V1 = expand_features_ml_v1(
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
POSITION_FEATURE_MAP_ML_V1 = {
    "QB": expand_features_ml_v1(PASSING_FEATURES + KICK_HOLDING_FEATURE),
    "RB": expand_features_ml_v1(RUSHING_FEATURES + RECEIVING_FEATURES + SPECIAL_TEAMS_FEATURE),
    "FB": expand_features_ml_v1(
        RUSHING_FEATURES + RECEIVING_FEATURES + BLOCKING_FEATURES + SPECIAL_TEAMS_FEATURE
    ),
    "WR": expand_features_ml_v1(
        RUSHING_FEATURES
        + RECEIVING_FEATURES
        + BLOCKING_FEATURES
        + RETURNER_FEATURES
        + SPECIAL_TEAMS_FEATURE
    ),
    "TE": expand_features_ml_v1(
        RUSHING_FEATURES + RECEIVING_FEATURES + BLOCKING_FEATURES + SPECIAL_TEAMS_FEATURE
    ),
    "C": expand_features_ml_v1(BLOCKING_FEATURES + LONG_SNAPPING_FEATURE),
    "G": expand_features_ml_v1(BLOCKING_FEATURES + LONG_SNAPPING_FEATURE),
    "T": expand_features_ml_v1(BLOCKING_FEATURES + LONG_SNAPPING_FEATURE),
    "DE": expand_features_ml_v1(DEFENSIVE_FEATURES + SPECIAL_TEAMS_FEATURE),
    "DT": expand_features_ml_v1(DEFENSIVE_FEATURES + SPECIAL_TEAMS_FEATURE),
    "OLB": expand_features_ml_v1(DEFENSIVE_FEATURES + SPECIAL_TEAMS_FEATURE),
    "ILB": expand_features_ml_v1(DEFENSIVE_FEATURES + SPECIAL_TEAMS_FEATURE),
    "CB": expand_features_ml_v1(DEFENSIVE_FEATURES + RETURNER_FEATURES + SPECIAL_TEAMS_FEATURE),
    "S": expand_features_ml_v1(DEFENSIVE_FEATURES + RETURNER_FEATURES + SPECIAL_TEAMS_FEATURE),
    "K": expand_features_ml_v1(KICKING_FEATURES + KICK_HOLDING_FEATURE),
    "P": expand_features_ml_v1(PUNTING_FEATURES + KICK_HOLDING_FEATURE),
    "LS": expand_features_ml_v1(BLOCKING_FEATURES + LONG_SNAPPING_FEATURE + SPECIAL_TEAMS_FEATURE),
}

# Special handling for "All" or if we want to add more general ones
ALL_POSITIONS_ML_V1 = list(POSITION_FEATURE_MAP_ML_V1.keys())


def preprocess_for_sklearn_ml_v1(X_pl: pl.DataFrame, scaler=None):
    """
    Prepares a Polars DataFrame for scikit-learn models by:
    1. Converting to pandas.
    2. One-hot encoding categorical features.
    3. Filling NaNs with 0.
    4. Scaling features using StandardScaler.
    """
    X_pd = X_pl.to_pandas()
    # Simple OHE for categorical columns
    cat_cols = X_pd.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X_pd = pd.get_dummies(X_pd, columns=cat_cols, drop_first=True)
    # Fill NaNs for GLMs
    X_pd = X_pd.fillna(0)

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_pd)
    else:
        X_scaled = scaler.transform(X_pd)

    X_final = pd.DataFrame(X_scaled, columns=X_pd.columns, index=X_pd.index)
    return X_final, scaler
