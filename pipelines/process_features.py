import os
from collections.abc import Iterable
from pathlib import Path

import hydra
import polars as pl
from fof8_core.loader import FOF8Loader
from fof8_ml.data.economic_dataset import build_economic_dataset
from omegaconf import DictConfig


def _coerce_str_list(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if not isinstance(raw, Iterable):
        return [str(raw)]
    return [str(v) for v in list(raw)]


def resolve_year_range(initial_year: int, final_year: int, cfg: DictConfig) -> list[int]:
    """Resolve the per-universe raw year window to materialize.

    `year_start_offset=1` means start with the second simulation year.
    `year_count=30` means include exactly 30 years when available.
    """
    start_offset = int(cfg.data.get("year_start_offset", 1))
    if start_offset < 0:
        raise ValueError("data.year_start_offset must be >= 0")

    start_year = initial_year + start_offset
    year_count = cfg.data.get("year_count")
    if year_count is None:
        end_year = final_year
    else:
        year_count = int(year_count)
        if year_count < 1:
            raise ValueError("data.year_count must be >= 1 when provided")
        end_year = min(final_year, start_year + year_count - 1)

    if start_year > final_year:
        raise ValueError(
            f"Configured year_start_offset={start_offset} starts after final year {final_year}"
        )
    if start_year > end_year:
        raise ValueError(f"Resolved empty year range: {start_year} to {end_year}")

    return [start_year, end_year]


def resolve_league_names(raw_path: str, cfg: DictConfig) -> list[str]:
    """Resolve one or more raw universe folders from data config."""
    explicit_names = _coerce_str_list(cfg.data.get("league_names"))
    if explicit_names:
        return explicit_names

    legacy_name = cfg.data.get("league_name")
    if legacy_name:
        return [str(legacy_name)]

    league_glob = cfg.data.get("league_glob")
    if league_glob:
        raw_dir = Path(raw_path)
        matches = sorted(p.name for p in raw_dir.glob(str(league_glob)) if p.is_dir())
        if matches:
            return matches
        raise ValueError(f"No raw league folders matched league_glob={league_glob!r} in {raw_dir}")

    raise ValueError("Configure data.league_names, data.league_name, or data.league_glob.")


def normalize_pooled_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Normalize per-universe frames before pooled concatenation.

    Dataset builders cast string-like columns to Polars Enum with domains derived from
    each universe. Separate Enum domains cannot be concatenated, so pooled storage uses
    String for those columns. Training-time schema inference still treats them as
    categorical.
    """
    casts = [
        pl.col(col).cast(pl.String)
        for col, dtype in frame.schema.items()
        if dtype in (pl.Enum, pl.Categorical)
    ]
    return frame.with_columns(casts) if casts else frame


def universe_features_path(processed_path: str | Path, universe: str) -> Path:
    """Return the per-universe feature parquet path for a processed data root."""
    universe_name = Path(universe).name
    return Path(processed_path) / "universes" / universe_name / "features.parquet"


def write_universe_features(frame: pl.DataFrame, processed_path: str | Path, universe: str) -> Path:
    """Persist one normalized universe feature frame for debugging/reuse."""
    out_file = universe_features_path(processed_path, universe)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out_file)
    return out_file


@hydra.main(version_base=None, config_path="conf", config_name="transform")
def main(cfg: DictConfig) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_root = os.path.abspath(os.path.join(script_dir, ".."))
    absolute_raw_path = os.path.abspath(os.path.join(exp_root, cfg.data.raw_path))
    processed_path = os.path.abspath(os.path.join(exp_root, cfg.data.processed_dir))

    os.makedirs(processed_path, exist_ok=True)
    out_file = os.path.join(processed_path, "features.parquet")

    frames = []
    league_names = resolve_league_names(absolute_raw_path, cfg)
    for league_name in league_names:
        loader = FOF8Loader(base_path=absolute_raw_path, league_name=league_name)
        initial_year = loader.initial_sim_year
        final_sim_year = loader.final_sim_year

        year_range = resolve_year_range(initial_year, final_sim_year, cfg)

        print(
            f"Building Universal Data Store for {league_name} "
            f"years {year_range[0]} to {year_range[1]}..."
        )
        X, y, metadata = build_economic_dataset(
            absolute_raw_path,
            league_name,
            year_range,
            positions=None,
            active_team_id=cfg.data.active_team_id,
            merit_threshold=0,
            universe=league_name,
        )
        frame = normalize_pooled_frame(pl.concat([metadata, X, y], how="horizontal"))
        universe_out_file = write_universe_features(frame, processed_path, league_name)
        print(f"Universe feature store written to {universe_out_file} (Rows: {len(frame)})")
        frames.append(frame)

    if not frames:
        raise ValueError("No universe frames were built. Check data league configuration.")

    df = pl.concat(frames, how="vertical_relaxed")
    df.write_parquet(out_file)
    print(f"Universal Data Store written to {out_file} (Universes: {len(frames)}, Rows: {len(df)})")


if __name__ == "__main__":
    main()
