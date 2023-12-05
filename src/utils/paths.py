from pathlib import Path

# Project path
project_root = Path(__file__).resolve().parents[2]

# Top level folders
data_dir = project_root / "data"

# Data folders
raw_data_dir = data_dir / "raw"
processed_data_dir = data_dir / "processed"

if __name__ == "__main__":
    from src.utils.console import console

    # Check paths
    console.print(f"{project_root=}")
    console.print(f"{data_dir=}")
    console.print(f"{raw_data_dir=}")
    console.print(f"{processed_data_dir=}")
