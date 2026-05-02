"""Helpers and constants for resolving active-team identity from league metadata."""

from pathlib import Path
from typing import Callable

import polars as pl

# Hardcoded lookup first to resolve ambiguous city-name matches (e.g. New York).
TEAM_NAME_TO_ID = {
    "arizona cardinals": 0,
    "atlanta falcons": 1,
    "baltimore ravens": 2,
    "buffalo bills": 3,
    "carolina panthers": 4,
    "chicago bears": 5,
    "cincinnati bengals": 6,
    "dallas cowboys": 7,
    "denver broncos": 8,
    "detroit lions": 9,
    "green bay packers": 10,
    "indianapolis colts": 11,
    "jacksonville jaguars": 12,
    "kansas city chiefs": 13,
    "miami dolphins": 14,
    "minnesota vikings": 15,
    "new england patriots": 16,
    "new orleans saints": 17,
    "new york giants": 18,
    "new york jets": 19,
    "las vegas raiders": 20,
    "oakland raiders": 20,
    "philadelphia eagles": 21,
    "pittsburgh steelers": 22,
    "los angeles rams": 23,
    "st. louis rams": 23,
    "seattle seahawks": 24,
    "san francisco 49ers": 25,
    "los angeles chargers": 26,
    "san diego chargers": 26,
    "tampa bay buccaneers": 27,
    "tennessee titans": 28,
    "washington commanders": 29,
    "washington redskins": 29,
    "washington football team": 29,
    "cleveland browns": 30,
    "houston texans": 31,
}


def read_metadata_team_fields(metadata_path: Path) -> tuple[int | None, str | None]:
    """Read optional team_id/team fields from metadata.yaml without YAML dependency."""
    if not metadata_path.exists():
        return None, None

    team_name: str | None = None
    team_id: int | None = None

    for line in metadata_path.read_text().splitlines():
        if "team_id:" in line:
            team_id = int(line.split(":", 1)[1].strip())
            break
        if "team:" in line:
            team_name = line.split(":", 1)[1].strip()

    return team_id, team_name


def resolve_team_id_from_known_names(team_name: str | None) -> int | None:
    """Resolve a team ID via the static ``TEAM_NAME_TO_ID`` mapping."""
    if not team_name:
        return None
    return TEAM_NAME_TO_ID.get(team_name.lower())


def resolve_team_id_from_team_information(
    league_dir: Path,
    scan_file: Callable[..., pl.LazyFrame],
    team_name: str | None,
) -> int | None:
    """
    Resolve a team ID from ``team_information.csv`` using city-name matching.

    The first available year directory is used as the fallback source.
    """
    if not team_name:
        return None

    years = sorted([p.name for p in league_dir.iterdir() if p.is_dir() and p.name.isdigit()])
    if not years:
        return None

    df_teams = scan_file("team_information.csv", year=int(years[0])).collect()
    for row in df_teams.to_dicts():
        city = row.get("Home_City", "")
        if city and city in team_name:
            return int(row.get("Team"))

    return None
