# imports
import json
import os
from pathlib import Path
from typing import Callable, Generator, Iterator

import msgpack
import polars as pl
from tqdm import tqdm

from src.utils.console import console
from src.utils.formatter import format_data_size
from src.utils.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.utils.interfaces import DescribeJsonStructure


def extract_from_shard(shard: Path) -> pl.DataFrame:
    """Extract the data from a shard.

    Args:
        shard (Path): path to the shard

    Returns:
        pl.DataFrame: data from the shard
    """

    # define lists to store the data
    ids = []
    lats = []
    longs = []
    ims = []

    # read the shard
    with open(shard, "rb") as infile:
        for record in msgpack.Unpacker(infile, raw=False):
            ids.append(record["id"])
            lats.append(record["latitude"])
            longs.append(record["longitude"])
            ims.append(record["image"])

    # Define dataframe
    data = {
        "id": ids,
        "latitude": lats,
        "longitude": longs,
        "image": ims,
    }
    schema = {
        "id": pl.Utf8,
        "latitude": pl.Float64,
        "longitude": pl.Float64,
        "image": pl.Binary,
    }

    # return the data
    return pl.DataFrame(data, schema)


def delete_processed_data(dir: str | Path = PROCESSED_DATA_DIR / "LDoGI", *, verbose: bool = False):
    """Delete processed data of the LDoGI dataset.

    Args:
        dir (str | Path, optional): path to the processed data directory. Defaults to processed_data_dir / "LDoGI".
        verbose (bool, optional): print verbose output. Defaults to False.
    """

    # convert to path objects
    dir = Path(dir)

    # check if dir exists
    if not dir.is_dir():
        raise FileNotFoundError(f"dir not found: {dir}")

    # get the files
    files = list(dir.glob("*.parquet"))

    # delete the files
    for file in files:
        os.remove(file)

    # print some stats
    if verbose:
        console.print(f"Deleted {len(files)} files")


def update_describe_data(describe_data: DescribeJsonStructure, file_name: str, count: int) -> DescribeJsonStructure:
    """Update the describe data.

    Args:
        describe_data (DescribeJsonStructure): describe data
        file_name (str): name of the file
        count (int): number of records in the file

    Returns:
        DescribeJsonStructure: updated describe data
    """

    # update the count
    describe_data["count"] += count

    # update the files
    describe_data["files"].append(
        {
            "name": file_name,
            "count": count,
            "min_index": describe_data["files"][-1]["max_index"] + 1 if len(describe_data["files"]) > 0 else 0,
            "max_index": describe_data["files"][-1]["max_index"] + count
            if len(describe_data["files"]) > 0
            else count - 1,
        }
    )

    # return the updated describe data
    return describe_data


def write_description(describe_data: DescribeJsonStructure, *, dst_dir: str | Path = PROCESSED_DATA_DIR / "LDoGI"):
    """Write the describe data to a JSON file.

    Args:
        describe_data (DescribeJsonStructure): describe data
        dst_dir (str | Path, optional): path to the processed data directory where the JSON file will be stored.
            Defaults to processed_data_dir / "LDoGI".
    """

    # convert to path objects
    dst_dir = Path(dst_dir)

    # check if dst dir exists, create if not
    if not dst_dir.is_dir():
        dst_dir.mkdir(parents=True)

    # define the describe file
    describe_file = dst_dir / "describe.json"

    # write the describe file
    with open(describe_file, "w") as f:
        json.dump(describe_data, f, indent=4)

def regenerate_description(dst_dir: str | Path = PROCESSED_DATA_DIR / "LDoGI"):
    """Regenerate the describe data JSON file.

    Args:
        dst_dir (str | Path, optional): path to the processed data directory where the JSON file will be stored.
            Defaults to processed_data_dir / "LDoGI".
    """

    # convert to path objects
    dst_dir = Path(dst_dir)

    # check if dst dir exists, create if not
    if not dst_dir.is_dir():
        dst_dir.mkdir(parents=True)

    # define the describe file
    describe_file = dst_dir / "describe.json"

    # delete the describe file if it exists
    if describe_file.is_file():
        os.remove(describe_file)

    # define the describe data
    describe_data = DescribeJsonStructure(count=0, files=[])

    # get the files
    files = list(dst_dir.glob("*.parquet"))

    # loop through the files
    for file in tqdm(files):
        # read the file
        data = pl.read_parquet(file)

        # update the describe data
        describe_data = update_describe_data(describe_data, file.stem, data.shape[0])

    # write the describe file
    with open(describe_file, "w") as f:
        json.dump(describe_data, f, indent=4)


def process_raw_data(
    src_dir: str | Path = RAW_DATA_DIR / "LDoGI/shards",
    dst_dir: str | Path = PROCESSED_DATA_DIR / "LDoGI",
    *,
    verbose: bool = True,
    delete_existing: bool = False,
    filter_files: Callable[[str | Path], bool] | None = None,
    describe: bool = True,
):
    """Process raw data of the LDoGI dataset into a more convenient parquet format.

    Args:
        src_dir (str | Path): path to the raw data directory containing the shards in msgpack format.
            Defaults to raw_data_dir / "LDoGI/shards".
        dst_dir (str | Path): path to the processed data directory where the parquet files will be stored.
            Defaults to processed_data_dir / "LDoGI".
        verbose (bool, optional): print verbose output. Defaults to True.
        delete_existing (bool, optional): delete existing files in dst_dir. Defaults to False.
        filter_files (Callable[[str | Path], bool] | None, optional): filter the files in src_dir. Defaults to None.

    Raises:
        FileNotFoundError: if src_dir does not exist
    """

    # convert to path objects
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    # check if src dir exists
    if not src_dir.is_dir():
        raise FileNotFoundError(f"src_dir not found: {src_dir}")

    # check if dst dir exists, create if not
    if not dst_dir.is_dir():
        dst_dir.mkdir(parents=True)

    # delete existing files if requested
    if delete_existing:
        delete_processed_data(dst_dir, verbose=verbose)

    # check if dst dir is empty
    dst_dir_files = len(list(dst_dir.glob("*.parquet"))) > 0
    if dst_dir_files > 0:
        raise FileExistsError(
            f"dst_dir is not empty, found {dst_dir_files} files. If you want to overwrite them,\
                  set delete_existing=True"
        )

    msgpack_files: Iterator[Path] | Generator[Path, None, None] = src_dir.glob("*.msg")
    msgpack_files = filter(filter_files, msgpack_files) if filter_files else msgpack_files

    # setup verbose accounting
    file_count = 0
    src_sum_size = 0
    dst_sum_size = 0

    # setup describe data
    describe_data = DescribeJsonStructure(count=0, files=[])

    for msgpack_file in tqdm(list(msgpack_files)):
        # get the name of the file
        file_name = msgpack_file.stem

        # read the msgpack file
        data = extract_from_shard(msgpack_file)

        # define the parquet file
        parquet_file = dst_dir / f"{file_name}.parquet"

        # save the data as a parquet file
        data.write_parquet(parquet_file)

        # some accounting
        file_count += 1
        src_sum_size += os.path.getsize(msgpack_file)
        dst_sum_size += os.path.getsize(parquet_file)

        # update the describe data
        describe_data = update_describe_data(describe_data, file_name, data.shape[0])

    if describe:
        write_description(describe_data, dst_dir=dst_dir)

    # print some stats
    if verbose:
        console.print(f"Processed {file_count} files")
        console.print(f"src size: {format_data_size(src_sum_size, precision=3)}")
        console.print(f"dst size: {format_data_size(dst_sum_size, precision=3)}")


if __name__ == "__main__":
    # test if process_raw_data works
    process_raw_data(
        dst_dir=PROCESSED_DATA_DIR / "LDoGI/test",
        delete_existing=True,
        filter_files=lambda x: Path(x).name in [f"shard_{i}.msg" for i in range(5)],
    )

    # test if process_raw_data throws an error
    try:
        process_raw_data("foo", "bar")
    except FileNotFoundError:
        console.print("[green]ValueError thrown as expected[/green]")
    else:
        console.print("[red]ValueError not thrown[/red]")
