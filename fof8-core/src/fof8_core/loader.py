from pathlib import Path
from typing import Optional

import polars as pl

from .schemas import SCHEMAS


class FOF8Loader:
    """
    Standardized data loader for FOF8 CSV files across multiple simulation years.
    Handles directory navigation, schema enforcement, and year extraction.
    """

    def __init__(self, base_path: str | Path, league_name: str):
        """
        Initializes the loader.

        Args:
            base_path: The root directory containing simulation data (e.g., 'data/raw').
            league_name: The name of the league folder (e.g., 'DRAFT003').

        Raises:
            FileNotFoundError: If the league directory does not exist.
        """
        self.base_path = Path(base_path)
        self.league_name = league_name
        self.league_dir = self.base_path / self.league_name

        if not self.league_dir.exists():
            raise FileNotFoundError(f"League directory not found: {self.league_dir}")

    @property
    def final_sim_year(self) -> int:
        """Dynamically discovers the final simulated year."""
        years = [int(p.name) for p in self.league_dir.iterdir() if p.is_dir() and p.name.isdigit()]
        if not years:
            raise ValueError(f"No year directories found in {self.league_dir}")
        return max(years)

    @property
    def initial_sim_year(self) -> int:
        """Dynamically discovers the first simulated year."""
        years = [int(p.name) for p in self.league_dir.iterdir() if p.is_dir() and p.name.isdigit()]
        if not years:
            raise ValueError(f"No year directories found in {self.league_dir}")
        return min(years)

    def scan_file(self, filename: str, year: Optional[int] = None) -> pl.LazyFrame:
        """
        Scans a specific CSV file into a Polars LazyFrame.
        Supports wildcard filenames (e.g., 'player_ratings_season_*.csv').
        """
        # Note: wildcards in schemas.py keys (like 'player_ratings_season_*') 
        # won't map exactly to Path(filename).stem anymore if it resolves to a specific file.
        # But since we handle that fallback elsewhere, this is safe.
        schema_override = SCHEMAS.get(Path(filename).stem, {})

        if year is not None:
            # Handle potential wildcards for a single year
            year_dir = self.league_dir / str(year)
            paths = list(year_dir.glob(filename))
            if not paths:
                raise FileNotFoundError(f"File matching {filename} not found in {year_dir}")
            
            lfs = [self._scan_single_file(p, schema_override) for p in paths]
            return pl.concat(lfs)

        # Scan all years
        lfs = []
        year_dirs = sorted(
            [d for d in self.league_dir.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda x: int(x.name),
        )
        for d in year_dirs:
            # Using glob() natively evaluates the '*' character
            for path in d.glob(filename):
                if path.is_file():
                    lfs.append(self._scan_single_file(path, schema_override))

        if not lfs:
            raise FileNotFoundError(f"No files found for {filename} in {self.league_dir}")

        return pl.concat(lfs)

    def _scan_single_file(self, path: Path, schema_override: dict) -> pl.LazyFrame:
        """Helper to scan a single CSV file and apply cleaning logic."""
        lf = pl.scan_csv(
            path,
            infer_schema_length=10000,
            null_values=["", "null", "None", "N/A"],
            ignore_errors=True,
            # Force columns that are notorious for mixed types to String immediately
            dtypes={"Injury_Type": pl.String, "Season_Statistics_-_Injury_Type": pl.String}
        )

        # Clean up column names
        column_names = lf.collect_schema().names()
        column_map = {}
        prefixes_to_strip = ["Season_Statistics", "Playoff_Statistics"]

        for col in column_names:
            new_name = col
            if col == "Player_IDPlayer_ID":
                new_name = "Player_ID"
            elif "_-_" in col:
                prefix, suffix = col.split("_-_", 1)
                if prefix in prefixes_to_strip:
                    new_name = suffix

            if new_name != col:
                column_map[col] = new_name

        if column_map:
            lf = lf.rename(column_map)

        # Apply schema overrides as casts
        column_names_post_rename = lf.collect_schema().names()
        casts = [
            pl.col(col).cast(dtype)
            for col, dtype in schema_override.items()
            if col in column_names_post_rename
        ]
        if casts:
            lf = lf.with_columns(casts)

        # Inject Year from path
        year_val = int(path.parent.name)
        lf = lf.with_columns(pl.lit(year_val).cast(pl.Int16).alias("Year"))

        return lf

    def get_salary_cap(self, year: int) -> int:
        """
        Fetches the salary cap for a specific year, converted to total dollars.

        Args:
            year: The simulated year to fetch the cap for.

        Returns:
            The salary cap in total dollars (e.g., 150000000).
        """
        lf = self.scan_file("universe_info.csv", year=year)

        cap_value = (
            lf.filter(pl.col("Information") == "Salary Cap (in tens of thousands)")
            .select("Value/Round/Position")
            .collect()
            .item()
        )
        # item() returns the first value of the first column as a python type
        return int(cap_value) * 10_000

    def get_active_team_id(self) -> Optional[int]:
        """
        Dynamically discovers the active team ID from the league metadata.
        Uses metadata.yaml to find the team name and resolves it via team_information.csv.
        """
        metadata_path = self.league_dir / "metadata.yaml"
        if not metadata_path.exists():
            return None

        # Basic text parsing to avoid a PyYAML dependency in the core package
        team_name = None
        team_id = None
        try:
            for line in metadata_path.read_text().splitlines():
                if "team_id:" in line:
                    team_id = int(line.split(":", 1)[1].strip())
                    break
                if "team:" in line:
                    team_name = line.split(":", 1)[1].strip()
        except Exception:
            return None

        # If team_id was explicitly provided, use it
        if team_id is not None:
            return team_id

        if not team_name:
            return None

        # Hardcoded lookup first to resolve ambiguity (like New York)
        team_ids = {
            "arizona cardinals": 0, "atlanta falcons": 1, "baltimore ravens": 2, "buffalo bills": 3,
            "carolina panthers": 4, "chicago bears": 5, "cincinnati bengals": 6, "dallas cowboys": 7,
            "denver broncos": 8, "detroit lions": 9, "green bay packers": 10, "indianapolis colts": 11,
            "jacksonville jaguars": 12, "kansas city chiefs": 13, "miami dolphins": 14, "minnesota vikings": 15,
            "new england patriots": 16, "new orleans saints": 17, "new york giants": 18, "new york jets": 19,
            "las vegas raiders": 20, "oakland raiders": 20, "philadelphia eagles": 21, "pittsburgh steelers": 22,
            "los angeles rams": 23, "st. louis rams": 23, "seattle seahawks": 24, "san francisco 49ers": 25,
            "los angeles chargers": 26, "san diego chargers": 26, "tampa bay buccaneers": 27, "tennessee titans": 28,
            "washington commanders": 29, "washington redskins": 29, "washington football team": 29,
            "cleveland browns": 30, "houston texans": 31,
        }
        
        known_id = team_ids.get(team_name.lower())
        if known_id is not None:
            return known_id

        # Resolve name to ID using the first available year's team info as a fallback
        try:
            years = sorted(
                [
                    p.name
                    for p in self.league_dir.iterdir()
                    if p.is_dir() and p.name.isdigit()
                ]
            )
            if not years:
                return None

            df_teams = self.scan_file("team_information.csv", year=int(years[0])).collect()

            for row in df_teams.to_dicts():
                city = row.get("Home_City", "")
                if city and city in team_name:
                    return int(row.get("Team"))
        except Exception:
            pass

        return None
