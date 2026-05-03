"""Categorical preprocessing helpers shared by dataset builders."""

import polars as pl


def bucket_rare_colleges(df: pl.DataFrame, *, min_count: int) -> pl.DataFrame:
    """Buckets infrequent college values into 'Other'."""
    if "College" not in df.columns:
        return df

    college_counts = df["College"].value_counts()
    top_colleges = college_counts.filter(pl.col("count") >= min_count)["College"].to_list()
    return df.with_columns(
        pl.when(pl.col("College").is_in(top_colleges))
        .then(pl.col("College"))
        .otherwise(pl.lit("Other"))
        .alias("College")
    )


def cast_categoricals_to_enum(df: pl.DataFrame) -> pl.DataFrame:
    """Casts String/Categorical columns to deterministic Enum domains."""
    cat_cols = [col for col, dtype in df.schema.items() if dtype in [pl.String, pl.Categorical]]

    for col in cat_cols:
        unique_categories = df.get_column(col).unique().sort().cast(pl.String)
        df = df.with_columns(pl.col(col).cast(pl.Enum(unique_categories)))

    return df
