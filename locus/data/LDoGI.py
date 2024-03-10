# imports
import json
import os
from pathlib import Path
from typing import Callable

import msgpack
import polars as pl
from tqdm import tqdm

from locus.utils.console import console
from locus.utils.formatter import format_data_size
from locus.utils.interfaces import DescribeJsonStructure
from locus.utils.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR


def extract_data_shard2shard(shard: Path, *, first_id) -> pl.DataFrame:
    """Extract the data from a shard.

    Args:
        shard (Path): path to the shard

    Returns:
        pl.DataFrame: data from the shard
    """

    # define lists to store the data
    lats = []
    lons = []
    ims = []

    # read the shard
    with open(shard, "rb") as infile:
        for record in msgpack.Unpacker(infile, raw=False):
            lats.append(record["latitude"])
            lons.append(record["longitude"])
            ims.append(record["image"])

    # define ids
    ids = list(range(first_id, first_id + len(lats)))

    # Define dataframe
    data = {
        "id": ids,
        "latitude": lats,
        "longitude": lons,
        "image": ims,
    }
    schema = {
        "id": pl.Int64,
        "latitude": pl.Float64,
        "longitude": pl.Float64,
        "image": pl.Binary,
    }

    # return the data
    return pl.DataFrame(data, schema)


def extract_data_shard2db(shard: Path) -> pl.DataFrame:
    """Extract the data from a shard.

    Args:
        shard (Path): path to the shard

    Returns:
        pl.DataFrame: data from the shard
    """

    # define lists to store the data
    lats = []
    lons = []
    ims = []

    # read the shard
    with open(shard, "rb") as infile:
        for record in msgpack.Unpacker(infile, raw=False):
            lats.append(record["latitude"])
            lons.append(record["longitude"])
            ims.append(record["image"])

    # Define dataframe
    data = {
        "latitude": lats,
        "longitude": lons,
        "image": ims,
    }
    schema = {
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
    files = list(dir.glob("*"))

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
    strategy: str = "db",  # "shard" | "db"
    *,
    dst_dir: str | Path = PROCESSED_DATA_DIR / "LDoGI",
    db_host: str = "localhost",
    db_port: str | int = 5432,
    db_name: str = "database",
    db_user: str = "username",
    db_password: str = "password",
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
    shard_dir = Path(dst_dir) / "shards"
    desc_dir = Path(dst_dir)

    # check if src dir exists
    if not src_dir.is_dir():
        raise FileNotFoundError(f"src_dir not found: {src_dir}")

    if strategy == "shard":
        # check if dst dir is defined
        if not dst_dir:
            raise ValueError("dst_dir must be defined if strategy is 'shard'")

        # check if dst dir exists, create if not
        if not shard_dir.is_dir():
            shard_dir.mkdir(parents=True)

        # delete existing files if requested
        if delete_existing:
            delete_processed_data(shard_dir, verbose=verbose)

        # check if dst dir is empty
        dst_dir_files = len(list(shard_dir.glob("*.parquet"))) > 0
        if dst_dir_files > 0:
            raise FileExistsError(
                f"dst_dir is not empty, found {dst_dir_files} files. If you want to overwrite them,\
                    set delete_existing=True"
            )

    if strategy == "db":
        # check if db parameters are defined
        if not all([db_host, db_port, db_name, db_user, db_password]):
            raise ValueError("db_host, db_port, db_name, db_user, db_password must be defined if strategy is 'db'")

    msgpack_files_glob = src_dir.glob("*.msg")
    msgpack_files_filter = filter(filter_files, msgpack_files_glob) if filter_files else msgpack_files_glob
    msgpack_files = [f for f in msgpack_files_filter]
    msgpack_files.sort(key=lambda x: int(x.stem.split("_")[1]))

    # setup verbose accounting
    file_count = 0
    src_sum_size = 0
    dst_sum_size = 0

    # setup describe data
    describe_data = DescribeJsonStructure(count=0, files=[])

    first_id = 0
    for msgpack_file in tqdm(list(msgpack_files)):
        # get the name of the file
        file_name = msgpack_file.stem

        if strategy == "shard":
            # read the msgpack file
            data = extract_data_shard2shard(msgpack_file, first_id=first_id)
            # update first_id
            first_id += data.shape[0]

            # define the parquet file
            parquet_file = shard_dir / f"{file_name}.parquet"

            # save the data as a parquet file
            data.write_parquet(parquet_file)

            # some accounting
            dst_sum_size += os.path.getsize(parquet_file)

            # update the describe data
            describe_data = update_describe_data(describe_data, file_name, data.shape[0])

        if strategy == "db":
            # read the msgpack file
            data = extract_data_shard2db(msgpack_file)

            # write the data to the database
            data.write_database(
                "dataset",
                f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",
                if_table_exists="append",
            )

        # some accounting
        file_count += 1
        src_sum_size += os.path.getsize(msgpack_file)
    if strategy == "shard" and describe:
        write_description(describe_data, dst_dir=desc_dir)

    # print some stats
    if verbose:
        console.print(f"Processed {file_count} files")
        console.print(f"src size: {format_data_size(src_sum_size, precision=3)}")
        if strategy == "shard":
            console.print(f"dst size: {format_data_size(dst_sum_size, precision=3)}")


if __name__ == "__main__":
    # test if process_raw_data works
    # process_raw_data(
    #     strategy="shard",
    #     dst_dir=PROCESSED_DATA_DIR / "LDoGI/test",
    #     delete_existing=True,
    #     filter_files=lambda x: Path(x).name in [f"shard_{i}.msg" for i in range(5)],
    # )

    process_raw_data(
        strategy="db",
        db_host="localhost",
        db_port=5432,
        db_name="ldogi",
        db_user="postgres",
        db_password="1425869",
        delete_existing=True,
        filter_files=lambda x: Path(x).name in [f"shard_{i}.msg" for i in range(142)],
    )

    # test if process_raw_data throws an error
    try:
        process_raw_data("foo", "bar")
    except FileNotFoundError:
        console.print("[green]ValueError thrown as expected[/green]")
    else:
        console.print("[red]ValueError not thrown[/red]")
