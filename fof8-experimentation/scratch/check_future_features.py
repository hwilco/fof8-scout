import polars as pl
from fof8_core.loader import FOF8Loader
from fof8_core.features import get_draft_class
from pathlib import Path


def main():
    raw_path = "../fof8-gen/data/raw"
    league_name = "DRAFT005"

    loader = FOF8Loader(base_path=raw_path, league_name=league_name)

    years = [2021, 2025, 2030]
    all_future_cols = set()
    non_zero_by_pos = {}

    for year in years:
        try:
            df = get_draft_class(loader, year)
            future_cols = [c for c in df.columns if "Future_" in c]

            for col in future_cols:
                all_future_cols.add(col)
                # Filter for non-zero and count by position
                non_zero_df = df.filter(pl.col(col) != 0)
                if len(non_zero_df) > 0:
                    counts = non_zero_df["Position_Group"].value_counts()
                    if col not in non_zero_by_pos:
                        non_zero_by_pos[col] = {}
                    for row in counts.to_dicts():
                        pos = row["Position_Group"]
                        count = row["count"]
                        non_zero_by_pos[col][pos] = non_zero_by_pos[col].get(pos, 0) + count

        except Exception as e:
            print(f"Error for year {year}: {e}")

    print("\n=== Summary Across Sample Years ===")
    if not all_future_cols:
        print("No Future columns found.")
    else:
        for col in sorted(list(all_future_cols)):
            pos_info = non_zero_by_pos.get(col, {})
            if not pos_info:
                print(f"{col}: 0 non-zero values found.")
            else:
                print(f"{col}: {sum(pos_info.values())} non-zero values. Breakdown: {pos_info}")


if __name__ == "__main__":
    main()
