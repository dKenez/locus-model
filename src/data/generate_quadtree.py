from src.utils.paths import PROCESSED_DATA_DIR
import polars as pl






if __name__ == "__main__":
    my_data = PROCESSED_DATA_DIR / "LDoGI"
    data_files = list(my_data.glob("*.parquet"))
    print(len(data_files))