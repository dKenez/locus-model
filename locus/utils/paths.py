from pathlib import Path

# Project path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Top level folders
DATA_DIR = PROJECT_ROOT / "data"

# Data folders
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# SQL folder
SQL_DIR = PROJECT_ROOT / "locus/utils/sql"

# Weights folder
MODELS_DIR = PROJECT_ROOT / "models"

if __name__ == "__main__":
    from locus.utils.console import console

    # Check paths
    console.print(f"{PROJECT_ROOT=}")
    console.print(f"{DATA_DIR=}")
    console.print(f"{RAW_DATA_DIR=}")
    console.print(f"{PROCESSED_DATA_DIR=}")
    console.print(f"{SQL_DIR=}")
    console.print(f"{MODELS_DIR=}")
