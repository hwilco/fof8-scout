import polars as pl

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
        lf_info_pre = loader.scan_file("player_information_pre_draft.csv", year=year).select(["Player_ID", "Year", "Year_Born", "Position"])
        
        # Join, calculate Age, and immediately drop Year_Born
        age_expr = pl.col("Year") - pl.col("Year_Born")
        lf_rookies = (
            lf_rookies.join(lf_info_pre, on=["Player_ID", "Year"], how="left")
            .with_columns(
                age_expr.alias("Age"),
                (age_expr ** 2).alias("Age_Squared")
            )
            .drop("Year_Born") # Crucial: Drop this so it doesn't leak to the model!
        )
        
        # 1. Feature Engineering: Relative Athleticism & Size (Z-Scores by Position Group)
        combine_drills = ["Dash", "Strength", "Agility", "Jump", "Position_Specific"]
        physical_traits = ["Height", "Weight"]
        z_score_targets = combine_drills + physical_traits
        lower_is_better = ["Dash", "Agility"]

        # Step A: Convert 0s to Nulls for combine drills ONLY (FOF8 doesn't generate 0 lb players)
        lf_rookies = lf_rookies.with_columns([
            pl.when(pl.col(c) == 0).then(None).otherwise(pl.col(c)).alias(c)
            for c in combine_drills
        ])
        
        # Step B: Calculate Z-Scores safely for everything
        z_score_exprs = []
        for c in z_score_targets:
            std_expr = pl.col(c).std().over("Position_Group")
            
            # Safely calculate Z-score, yielding Null if the drill was skipped
            # OR if the standard deviation is 0
            z_expr = pl.when(std_expr > 0).then(
                (pl.col(c) - pl.col(c).mean().over("Position_Group")) / std_expr
            ).otherwise(None)
            
            # Flip sign if lower is better (so positive Z always means 'better')
            if c in lower_is_better:
                z_expr = -z_expr
                
            z_score_exprs.append(z_expr.alias(f"{c}_Z"))

        # Step C: Append the Z-scores WITHOUT dropping the original absolute columns
        lf_rookies = lf_rookies.with_columns(z_score_exprs)  
      
        # We find all 'High_' columns and calculate Delta and Mean
        # NOTE: We explicitly skip 'Future_' columns as they are unpopulated in our datasets
        high_cols = [
            c for c in lf_personal.collect_schema().names() 
            if c.startswith("High_") and "Future_" not in c
        ]
        engineering_exprs = []
        for c in high_cols:
            low_col = c.replace("High_", "Low_")
            engineering_exprs.append((pl.col(c) - pl.col(low_col)).alias(c.replace("High_", "Delta_")))
            engineering_exprs.append(((pl.col(c) + pl.col(low_col)) / 2).alias(c.replace("High_", "Mean_")))

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
                df_staff = (
                    lf_staff.filter(
                        (pl.col("Current_Team") == active_team_id) & 
                        (pl.col("Role").is_in(["Head Coach", "Offensive Coordinator", "Defensive Coordinator", "Assistant Coach"]))
                    )
                    .collect()
                )

                if len(df_staff) > 0:
                    # Extract specific roles safely
                    hc_scouts = df_staff.filter(pl.col("Role") == "Head Coach")["Scouting_Ability"]
                    scout_hc = hc_scouts[0] if len(hc_scouts) > 0 else 0
                    
                    oc_scouts = df_staff.filter(pl.col("Role") == "Offensive Coordinator")["Scouting_Ability"]
                    scout_oc = oc_scouts[0] if len(oc_scouts) > 0 else 0
                    
                    dc_scouts = df_staff.filter(pl.col("Role") == "Defensive Coordinator")["Scouting_Ability"]
                    scout_dc = dc_scouts[0] if len(dc_scouts) > 0 else 0
                    
                    asst_scouts = df_staff.filter(pl.col("Role") == "Assistant Coach")["Scouting_Ability"]
                    scout_asst = asst_scouts.max() if len(asst_scouts) > 0 else 0

                    lf_rookies = lf_rookies.with_columns([
                        pl.lit(scout_hc).alias("Scout_HC"),
                        pl.lit(scout_oc).alias("Scout_OC"),
                        pl.lit(scout_dc).alias("Scout_DC"),
                        pl.lit(scout_asst).alias("Scout_ASST")
                    ])
            except Exception as e:
                print(f"Warning: Could not load staff for year {year}: {e}")

        # Join features on Player_ID and Year
        return lf_rookies.join(lf_personal, on=["Player_ID", "Year"], how="left").collect()
