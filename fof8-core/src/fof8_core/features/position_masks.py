import polars as pl

from .constants import MASKABLE_FEATURES, POSITION_FEATURE_MAP


def apply_position_mask(df: pl.DataFrame) -> pl.DataFrame:
    existing_maskable = [col for col in MASKABLE_FEATURES if col in df.columns]

    feature_null_positions = {}
    for col in existing_maskable:
        null_positions = []
        for pos, keeps in POSITION_FEATURE_MAP.items():
            if col not in keeps:
                null_positions.append(pos)
        if null_positions:
            feature_null_positions[col] = null_positions

    mask_exprs = []
    for col, null_positions in feature_null_positions.items():
        mask_exprs.append(
            pl.when(pl.col("Position_Group").cast(pl.String).is_in(null_positions))
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        )

    if mask_exprs:
        df = df.with_columns(mask_exprs)

    return df
