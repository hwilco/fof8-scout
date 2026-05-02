"""Core CSV loader for FOF8 league data across one or more simulation years."""

from pathlib import Path

import polars as pl

from .loader_team_resolution import (
    read_metadata_team_fields,
    resolve_team_id_from_known_names,
    resolve_team_id_from_team_information,
)
from .schemas import SCHEMAS


class FOF8Loader:
    """
    Standardized data loader for FOF8 CSV files across multiple simulation years.
    Handles directory navigation, schema enforcement, and year extraction.
    """

    def __init__(self, base_path: str | Path, league_name: str) -> None:
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

    def scan_file(self, filename: str, year: int | None = None) -> pl.LazyFrame:
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
            schema_overrides={
                "Injury_Type": pl.String,
                "Season_Statistics_-_Injury_Type": pl.String,
            },
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

        # Drop unpopulated Future_ columns
        future_cols = [col for col in column_names_post_rename if "Future_" in col]
        if future_cols:
            lf = lf.drop(future_cols)

        casts = [
            pl.col(col).cast(dtype)
            for col, dtype in schema_override.items()
            if col in column_names_post_rename and col not in future_cols
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

    def get_active_team_id(self) -> int | None:
        """
        Resolve the active team ID for the league.

        Resolution order:
        1. Explicit ``team_id`` in ``metadata.yaml``
        2. Known-name lookup from a static mapping
        3. Fallback lookup in ``team_information.csv`` from earliest sim year

        Returns:
            Team ID if it can be resolved, otherwise ``None``.
        """
        metadata_path = self.league_dir / "metadata.yaml"
        try:
            team_id, team_name = read_metadata_team_fields(metadata_path)
        except Exception:
            return None

        # If team_id was explicitly provided, use it
        if team_id is not None:
            return team_id

        known_id = resolve_team_id_from_known_names(team_name)
        if known_id is not None:
            return known_id

        try:
            # Resolve name to ID using the first available year's team info as fallback.
            return resolve_team_id_from_team_information(
                league_dir=self.league_dir,
                scan_file=self.scan_file,
                team_name=team_name,
            )
        except Exception:
            return None

        return None
