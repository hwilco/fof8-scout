import polars as pl

df = pl.DataFrame({
    "Year": [2024, 2024],
    "Year_Born": [2000, 2001]
}).lazy()

q = df.with_columns(
    (pl.col("Year") - pl.col("Year_Born")).alias("Age"),
    ((pl.col("Year") - pl.col("Year_Born")) ** 2).alias("Age_Squared")
)

print(q.explain())
