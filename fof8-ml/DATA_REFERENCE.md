# DRAFT003 Data Reference

> A reference guide for the FOF8 DRAFT003 dataset, documenting data shape, file
> relationships, and key patterns discovered during exploratory analysis.
> Use this as a jumping-off point for building predictive models.

## Dataset Overview

- **Source**: Front Office Football 8 (FOF8 v8.4) simulation
- **League**: DRAFT003 (started with draft, Miami Dolphins)
- **Year Range**: 2020–2144 (125 annual snapshots)
- **Snapshot Location**: `data-generation/data/raw/DRAFT003/{year}/`
- **Encoding**: All CSVs are UTF-8 (converted from cp1252 at snapshot time)
- **Settings**: `combine_accuracy: 50`, `x_factor: true`, `personality_and_chemistry: false`

> [!IMPORTANT]
> **Timeline & Data Leakage Warning**
> The files in a single year's directory represent a split-timeline.
> * **Pre-Season Files:** `rookies.csv`, `draft_personal.csv`, and `player_information_pre_draft.csv` are captured at the *start* of the iteration (after the draft class is generated, but before the draft or season occurs). These are safe to use as predictive features.
> * **Post-Season Files:** All other files (including `player_information_post_sim.csv`) are exported at the *end* of the season. They contain the results of the simulated year (stats, injuries, development, rating changes). Using these as features for that same draft class will cause target leakage.

---

## Files Per Snapshot (27 files per year)

Each year directory contains the same set of files. They fall into three categories
based on how they behave over time.

### Cumulative Files (grow every year, contain all historical players)

| File | Rows (2050) | Rows (2144) | Description |
|---|---|---|---|
| `player_information_post_sim.csv` | 10,903 | 36,837 | Every player ever created. Grows by ~300/year. Retired players persist forever. Contains career totals and draft info. Note: Lacks unsigned undrafted rookies. |
| `awards.csv` | ~varies | ~varies | **Cumulative** history of all awards won. Every snapshot contains the full history up to that point. |
| `career_records.csv` | ~varies | ~varies | **Cumulative** all-time leaderboards by category. |
| `season_records.csv` | ~varies | ~varies | **Cumulative** single-season records by category. |

### Active-Only Files (roughly constant size, only current players)

| File | Rows (typical) | Description |
|---|---|---|
| `player_record.csv` | ~2,718 | Current roster players only. Season stats, contract, salary, personality, combine results, status. |
| `players_personal.csv` | ~1,917 | True skill ratings (Current and Future) for active players only. The "answer key." |
| `player_ratings_season_{Y}.csv` | ~varies | Scouted ratings at Pre-Camp and Exhibition stages. Contains Current_Overall and Future_Overall as seen by scouts. |
| `player_season_{Y}.csv` | ~varies | Per-week game statistics for the completed season. Populated with real stats in DRAFT003. |
| `universe_info.csv` | ~230 | **Economic Context** — Current salary cap, game stage, and draft order. Essential for contract normalization. |
| `staff.csv` / `staff_history.csv` | ~varies | Coaching staff info. |

### Draft-Specific Files (one draft class per year)

| File | Rows (typical) | Description |
|---|---|---|
| `rookies.csv` | ~801 | All rookie prospects for that year's draft. Combine scores + scouting grade. |
| `draft_personal.csv` | ~801 | Scouting report skill ranges (Low/High) for each rookie. |
| `player_information_pre_draft.csv` | ~801 | Baseline information including exact natural Position for rookies. |

### Other Files

`standings.csv`, `team_*.csv`, `game_information.csv`,
`current_schedule.csv`, `transactions_{Y}.csv`,
`active_free_agency.csv`

---

## Key File Schemas

### `rookies.csv` — Draft Prospect Data

**Timing:** Pre-Season (Safe for predictive features). Captured before the draft occurs.

The primary input feature source for modeling.

| Column | Type | Description | Notes |
|---|---|---|---|
| `Player_ID` | i64 | Unique identifier | Join key across all files |
| `Last_Name` | str | | |
| `First_Name` | str | | |
| `Position_Group` | str | Position category | QB, RB, FL, TE, T, G, C, DE, DT, LB, CB, S, K, P, LS |
| `College` | str | College name | |
| `Height` | i64 | Height in inches | |
| `Weight` | i64 | Weight in pounds | |
| `Dash` | i64 | 40-yard dash score | Lower = faster (inverted scale) |
| `Solecismic` | i64 | Solecismic test score | |
| `Strength` | i64 | Bench press / strength score | |
| `Agility` | i64 | Agility drill score | Higher = slower (inverted scale) |
| `Jump` | i64 | Broad jump score | |
| `Position_Specific` | i64 | Position-specific drill | 0 for some positions (e.g., linemen) |
| `Developed` | i64 | Development percentage | Estimated percentage of peak rating the player is currently at (e.g., 44 = 44% developed) |
| `Grade` | i64 | Overall scouting grade | Range 2–86; the game's noisy estimate of player quality |

**103,620 total rookies across 124 years** (2021–2144, ~836 per year on average).

### `player_information_post_sim.csv` — Career-Level Data (Cumulative)

**Timing:** Post-Season (Outcomes). Contains data generated *after* the season completes. Note that undrafted rookies who were never signed will NOT appear in this file.

The best single file for career outcomes. Found in every snapshot but the **final
snapshot (2144)** contains the most complete data for all players.

| Column | Position | Type | Description |
|---|---|---|---|
| `Player_ID` | 1 | i64 | Join key |
| `Last_Name` / `First_Name` | 2-3 | str | |
| `Position` | 6 | str | More specific than Position_Group (e.g., FL, SE, LE, RE) |
| `Championship_Rings` | 8 | i64 | |
| `Hall_of_Fame_Flag` | 13 | i64 | 1 = inducted |
| `Year_Inducted` | 14 | i64 | |
| `Draft_Round` | 21 | i64 | 1–7; 0 if undrafted |
| `Drafted_Position` | 22 | i64 | Pick number within round |
| `Drafted_By` | 23 | i64 | Team ID |
| `Draft_Year` | 24 | i64 | |
| `Career_Games_Played` | 31 | i64 | Total career games |
| `Number_of_Seasons` | 32 | i64 | Total seasons played |
| `Season_1_Year` … `SY_20` | 53-72 | i64 | Actual years played (0 = unused slot) |

### `player_record.csv` — Current Season Snapshot (Active Only)

**Timing:** Post-Season (Outcomes). Captured after the season completes.

One row per active roster player. Contains current-season stats and contract info.

Key columns (selected):

| Column | Type | Description |
|---|---|---|
| `Player_ID` | i64 | Join key |
| `Experience` | i64 | Years of experience |
| `Team` | i64 | Current team ID |
| `Status` | str | Always "Active Roster" in observed data |
| `How_Acquired` | str | "Rookie Draft", "Free Agency", etc. |
| `Contract_Length` | i64 | Years remaining |
| `Salary_Year_1` … `Salary_Year_5` | i64 | Annual salary (in tens of thousands) |
| `Bonus_Year_1` … `Bonus_Year_5` | i64 | Signing bonus (in tens of thousands) |
| `Hall_Of_Fame_Points` | i64 | Accumulated HOF points |
| `S_Games_Played` | i64 | Games played this season |
| `S_Games_Started` | i64 | Games started this season |
| `S_*` | i64 | ~50 season stat columns (passing, rushing, receiving, defense, etc.) |
| `P_*` | i64 | ~50 playoff stat columns (same structure) |

### `players_personal.csv` — Coach-Scouted Skill Ratings (Active Team Only)

**Timing:** Post-Season (Outcomes/Leakage Risk). Captured after the season completes.

> [!WARNING]
> **Data Leakage Risk:** Because this is exported *after* the year has run, information about how a rookie has begun to develop or decline in the NFL is already present. Ensure you do not use this in your feature set when predicting rookie performance, though it is perfect for target variable construction.
>
> Additionally, these are **not ground-truth ratings**. They represent the
> active team's coaching staff's *assessment* of abilities. Accuracy
> is determined by the evaluating coaches' `Scouting_Ability`.

| Column Pattern | Count | Description |
|---|---|---|
| `Current_{Skill}` | 58 cols | Team's scouted estimate of current ability (0–100 scale) |
| `Current_Overall` | 1 col | Composite scouted current rating |
| `Future_{Skill}` | 58 cols | Team's scouted estimate of future/potential ability |
| `Future_Overall` | 1 col | Composite scouted future rating |

### `draft_personal.csv` — Pre-Draft Scouting Ranges

**Timing:** Pre-Season (Safe for predictive features). Captured before the draft occurs.

Pre-draft scouting report with Low/High estimates for each of the ~58 skills.
Same skill names as `players_personal.csv` but prefixed with `Low_` and `High_`.

Also includes an `Interviewed` flag (0 or 1): players who were interviewed
have narrower scouting ranges, reflecting more information.

### `player_ratings_season_{Y}.csv` — Single-Scout Scouted Ratings

**Timing:** Post-Season. Contains snapshots from the completed year.

Contains two scouting stages per player per year: `Pre-Camp` and `Exhibition`. Each row represents **one scout's assessment** of a player at a specific point in the preseason.

### `staff.csv` — Coaching Staff & Scouting Abilities

**Timing:** Post-Season. Captured after the season completes.

One row per staff member (active and retired). This file is key for **weighting
or adjusting** scouted ratings.

### `universe_info.csv` — Economic Context

**Timing:** Post-Season. Captures the state of the league at year-end.

This file is a key-value store critical for normalizing financial data across different eras.

| Key (in `Information` column) | Description | Example Calculation |
|---|---|---|
| `Salary Cap (in tens of thousands)` | The total team salary cap for the year | `Cap = Value * 10,000` |
| `Minimum Salary (in thousands)` | The league minimum salary | `Min = Value * 10,000` |

> [!IMPORTANT]
> **Unit Discrepancy**: While `universe_info.csv` labels the minimum salary as "in thousands," cross-verification with player salaries in the game UI confirms that **all currency values use tens of thousands** (Value * 10,000) as the multiplier.

---

## Key Relationships & Join Patterns

### Player_ID is the universal join key

All player-level files use `Player_ID` as the primary key. This ID is assigned when
a player first appears in `rookies.csv` and persists across all snapshots forever.

### Cross-File Join Map

```text
=== PRE-SEASON (Safe Predictive Features) ===

rookies.csv (Year Y)          ← Draft prospect features
    │
    ├──→ draft_personal.csv (Year Y)     ← Pre-draft scouting skill ranges
    │
    └──→ player_information_pre_draft.csv (Year Y) ← Baseline info including exact Position


=== POST-SEASON (Outcomes & Leakage Risks) ===
    │
    ├──→ player_information_post_sim.csv (Year Y..2144)  ← Career totals, draft position
    │         NOTE: undrafted players that are not signed won't appear here
    │
    ├──→ player_record.csv (Year Y..Y+N)  ← Season stats, salary, status
    │         NOTE: only while player is on active roster
    │
    ├──→ players_personal.csv (Year Y..Y+N)  ← Coach-scouted ratings over time
    │
    └──→ awards.csv (Year Y..Y+N)  ← Awards won
```

---

## Statistical Properties

### Scouting Grade Distribution

- **Count**: 103,620 (all rookies, draft years 2021–2144)
- **Mean**: 38.79
- **Std**: 11.03
- **Min / Max**: 2 / 86
- **Quartiles**: Q25=31, Q50=38, Q75=46
- **Normality**: Rejected (D'Agostino-Pearson p ≈ 0). Normal but with heavier tails.
- **Proportion above 65**: ~1.18% (elite prospects are very rare)

### Draft Rate

- ~836 rookies enter the draft pool each year
- ~237 get drafted (Draft_Round > 0) → **~28.3% draft rate**
- ~599 go undrafted and never appear in career data

---

## Data Loading Patterns (Polars Best Practices)

Because of the dataset's size (~125 years of snapshots), loading performance and memory management are critical.

### 1. Schema Optimization & Downcasting
Using Polars' `schema_overrides` to downcast integer values saves significant memory. Categorical features should be loaded within a `pl.StringCache()` or explicitly defined via `pl.Enum` to ensure compatibility across lazy joins.

```python
import polars as pl
from pathlib import Path
from typing import Optional

# Use Enum for bounded string categories to prevent join errors
POSITIONS = pl.Enum(["QB", "RB", "FL", "TE", "T", "G", "C", "DE", "DT", "LB", "CB", "S", "K", "P", "LS"])

SCHEMAS = {
    "player_information_post_sim": {
        "Player_ID": pl.Int32,
        "Position": pl.Categorical,
        "Draft_Year": pl.Int16,
        "Draft_Round": pl.Int8,
        "Drafted_Position": pl.Int16,
        "Drafted_By": pl.Int8,
        "Championship_Rings": pl.Int8,
        "Hall_of_Fame_Flag": pl.Int8,
        "Career_Games_Played": pl.Int16,
        "Number_of_Seasons": pl.Int8,
    },
    "player_information_pre_draft": {
        "Player_ID": pl.Int32,
        "Position": pl.Categorical,
    },
    "rookies": {
        "Player_ID": pl.Int32,
        "Position_Group": POSITIONS,
        "College": pl.Categorical,
        "Height": pl.Int8,
        "Weight": pl.Int16,
        "Dash": pl.Int16,
        "Solecismic": pl.Int8,
        "Strength": pl.Int8,
        "Agility": pl.Int16,
        "Jump": pl.Int16,
        "Position_Specific": pl.Int8,
        "Developed": pl.Int8,
        "Grade": pl.Int8,
    }
}
```

### 2. A Robust DataLoader Architecture
Wrap your logic in a class to avoid hardcoding `DRAFT003` or paths, allowing easy pivoting to other simulations and standardizing glob patterns.

```python
class FOF8Loader:
    def __init__(self, base_path: str | Path, league_name: str = "DRAFT003"):
        self.base_path = Path(base_path)
        self.league_name = league_name
        self.league_dir = self.base_path / self.league_name

        if not self.league_dir.exists():
            raise FileNotFoundError(f"League directory not found: {self.league_dir}")

    def scan_file(self, filename: str, year: Optional[int] = None) -> pl.LazyFrame:
        """
        Scans a specific file. If year is provided, scans that year.
        If year is None, scans all years using a glob pattern.
        """
        if year is not None:
            path_pattern = str(self.league_dir / str(year) / filename)
        else:
            path_pattern = str(self.league_dir / "*" / filename)

        schema_override = SCHEMAS.get(Path(filename).stem, {})

        lf = pl.scan_csv(
            path_pattern,
            infer_schema_length=1000,
            schema_overrides=schema_override,
            include_file_paths="filepath"
        )

        # Make the regex dynamic based on the league_name
        regex_pattern = rf"{self.league_name}/(\d+)/"

        return lf.with_columns(
            pl.col("filepath")
              .str.extract(regex_pattern, 1)
              .cast(pl.Int16)
              .alias("Year")
        ).drop("filepath")

    def get_salary_cap(self, year: int) -> int:
        """Fetches the cap for a specific year, converted to total dollars."""
        lf = self.scan_file("universe_info.csv", year=year)

        cap_value = (
            lf.filter(pl.col("Information") == "Salary Cap (in tens of thousands)")
              .select("Value/Round/Position")
              .collect()
              .item()
        )
        return int(cap_value) * 10_000
```

### 3. Core Loading Patterns

**Loading Rookies & Career Outcomes:**
```python
loader = FOF8Loader(base_path="../../data-generation/data/raw", league_name="DRAFT003")

# StringCache is necessary for joining operations involving Categoricals
with pl.StringCache():
    lf_rookies = loader.scan_file("rookies.csv")

    # We only need the final snapshot (2144) for complete career totals
    lf_career = loader.scan_file("player_information_post_sim.csv", year=2144).select([
        "Player_ID", "Draft_Year", "Draft_Round", "Drafted_Position",
        "Career_Games_Played", "Number_of_Seasons",
        "Championship_Rings", "Hall_of_Fame_Flag"
    ])

    dataset = (
        lf_rookies.join(
            lf_career,
            on="Player_ID",
            how="left"
        )
        .with_columns(
            (pl.col("Draft_Round") > 0).alias("Was_Drafted"),
            pl.col("Number_of_Seasons").fill_null(0).cast(pl.Int8),
            pl.col("Career_Games_Played").fill_null(0).cast(pl.Int16),
            pl.col("Championship_Rings").fill_null(0).cast(pl.Int8),
            pl.col("Hall_of_Fame_Flag").fill_null(0).cast(pl.Int8)
        )
    ).collect()
```

**Loading a Player's True Rating Trajectory:**
Use lazy query execution and let Polars optimize the file reading.

```python
def get_rating_trajectory(loader: FOF8Loader, player_id: int) -> pl.DataFrame:
    """Get year-by-year true ratings for a single player efficiently."""
    return (
        loader.scan_file("players_personal.csv")
        .filter(pl.col("Player_ID") == player_id)
        .select(["Year", "Player_ID", "Current_Overall", "Future_Overall"])
        .sort("Year")
        .collect()
    )
```

**Normalizing Salaries against the Cap:**
```python
year_to_analyze = 2050
cap_value = loader.get_salary_cap(year_to_analyze)

df_records = (
    loader.scan_file("player_record.csv", year=year_to_analyze)
    .with_columns(
        (pl.col("Salary_Year_1") * 10000 / cap_value).alias("Cap_Pct_Year_1")
    )
).collect()
```
---

## Known Issues & Dead Features

### Generic "Future_N" Scouting Attributes
In the current `DRAFT003` (and related `DRAFT005`) datasets, the following features in `draft_personal.csv` are currently **unpopulated (contain only 0s)** across all position groups:
- `Low_Future_1` through `Low_Future_7`
- `High_Future_1` through `High_Future_7`

**Impact on Modeling:**
As of April 2026, the `fof8_core` feature engineering pipeline explicitly **filters these out** during the creation of the rookies DataFrame. They will not appear in the final feature matrix `X` as `Mean_Future_N` or `Delta_Future_N`.
