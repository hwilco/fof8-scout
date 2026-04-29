import os

import hydra
import polars as pl
from fof8_core.loader import FOF8Loader
from fof8_ml.data.dataset import build_economic_dataset
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="transform")
def main(cfg: DictConfig):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_root = os.path.abspath(os.path.join(script_dir, ".."))
    absolute_raw_path = os.path.abspath(os.path.join(exp_root, cfg.data.raw_path))
    processed_path = os.path.abspath(
        os.path.join(exp_root, cfg.data.get("processed_path", "fof8-ml/data/processed"))
    )

    os.makedirs(processed_path, exist_ok=True)
    out_file = os.path.join(processed_path, "features.parquet")

    loader = FOF8Loader(base_path=absolute_raw_path, league_name=cfg.data.league_name)
    initial_year = loader.initial_sim_year
    final_sim_year = loader.final_sim_year

    valid_start_year = initial_year + 1
    valid_end_year = final_sim_year

    # We build the entire universe of valid years and positions in the transform step
    year_range = [valid_start_year, valid_end_year]

    print(f"Building Universal Data Store for years {year_range[0]} to {year_range[1]}...")
    X, y, metadata = build_economic_dataset(
        absolute_raw_path,
        cfg.data.league_name,
        year_range,
        final_sim_year,
        positions="all",
        active_team_id=cfg.data.get("active_team_id"),
        merit_threshold=0,
    )

    df = pl.concat([metadata, X, y], how="horizontal")
    df.write_parquet(out_file)
    print(f"Universal Data Store written to {out_file} (Rows: {len(df)})")


if __name__ == "__main__":
    main()
