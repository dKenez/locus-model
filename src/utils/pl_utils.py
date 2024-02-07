import polars as pl


def batch_iter(df: pl.LazyFrame, batch_size: int = 1000):
    """Iterate over a LazyFrame in batches

    Args:
        df (pl.LazyFrame): LazyFrame to iterate over
        batch_size (int, optional): Batch size. Defaults to 1000.

    Returns:
        Generator: Generator that yields batches of the LazyFrame

    Yields:
        pl.LazyFrame: Batch of the LazyFrame
    """
    df_len: int = df.select(pl.len()).collect()["len"][0]
    for i in range(0, df_len, batch_size):
        yield df.slice(i, batch_size).collect()
