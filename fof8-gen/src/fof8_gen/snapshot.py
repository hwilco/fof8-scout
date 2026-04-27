import os
import re
import shutil
from pathlib import Path


def copy_and_convert(src: Path, dst: Path):
    """
    Reads src as cp1252, writes to dst as utf-8, and preserves metadata.
    """
    content = src.read_text(encoding="cp1252")
    dst.write_text(content, encoding="utf-8")
    shutil.copystat(src, dst)


def detect_latest_year(league_dir):
    """
    Scans for player_season_YYYY.csv and returns the latest year found.
    """
    latest_year = 0
    pattern = re.compile(r"player_season_(\d{4})\.csv")

    for file in os.listdir(league_dir):
        match = pattern.match(file)
        if match:
            year = int(match.group(1))
            if year > latest_year:
                latest_year = year

    if latest_year == 0:
        raise ValueError(f"Could not find any player_season_YYYY.csv files in {league_dir}")

    return latest_year


def create_league_snapshot(
    fof8_dir, league_name, output_base_dir, file_filter=None, rename_map=None
):
    """
    Creates a snapshot of relevant CSV files for the current season.

    :param fof8_dir: Root directory of FOF8 data (contains 'leaguedata')
    :param league_name: Name of the league folder (e.g. 'DRAFT001')
    :param output_base_dir: Base directory for snapshots (e.g. 'data/raw/DRAFT001')
    :param file_filter: Optional list or dict of specific filenames to snapshot
                        (e.g. ["rookies.csv", "draft_personal.csv"] or
                        {"player_information.csv": "player_information_pre_draft.csv"}).
                        If None, snapshots all relevant files for the season.
    :param rename_map: Optional dict for renaming files during a full snapshot.
    """
    league_dir = os.path.join(fof8_dir, "leaguedata", league_name)
    if not os.path.exists(league_dir):
        raise FileNotFoundError(f"League directory not found: {league_dir}")

    year = detect_latest_year(league_dir)
    target_dir = os.path.join(output_base_dir, str(year))

    print(f"Detected Year: {year}")
    print(f"Creating snapshot at: {target_dir}")

    os.makedirs(target_dir, exist_ok=True)

    if file_filter is not None:
        # Targeted snapshot: only copy the specified files
        if isinstance(file_filter, list):
            file_map = {f: f for f in file_filter}
        else:
            file_map = file_filter

        for filename, dst_name in file_map.items():
            src = os.path.join(league_dir, filename)
            if os.path.exists(src):
                dst = os.path.join(target_dir, dst_name)
                copy_and_convert(Path(src), Path(dst))
                print(f"  Copied {filename} as {dst_name}")
            else:
                print(f"  Warning: {filename} not found in {league_dir}")
    else:
        # Full snapshot: copy all relevant CSV files for the current year
        files_to_copy = []
        for file in os.listdir(league_dir):
            if not file.endswith(".csv"):
                continue

            # Search for a year suffix like _2027.csv
            match = re.search(r"_(\d{4})\.csv$", file)
            if not match:
                # a) Include all undated files
                files_to_copy.append(file)
                continue

            file_year = int(match.group(1))

            # b) Include if it matches the current snapshot year
            if file_year == year:
                files_to_copy.append(file)
                continue

        for file in files_to_copy:
            src = os.path.join(league_dir, file)
            dst_name = rename_map.get(file, file) if rename_map else file
            dst = os.path.join(target_dir, dst_name)
            copy_and_convert(Path(src), Path(dst))
            if dst_name != file:
                print(f"  Copied {file} as {dst_name}")

    print(f"Success! Snapshot for year {year} created.")
    return target_dir
