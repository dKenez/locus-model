import polars as pl


def batch_iter(df: pl.LazyFrame, batch_size: int = 1000):
    """
    Iterate over a LazyFrame in batches
    """
    df_len: int = df.select(pl.len()).collect()["len"][0]
    for i in range(0, df_len, batch_size):
        yield df.slice(i, batch_size).collect()
