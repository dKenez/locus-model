import threading
import time
from pathlib import Path

import click
import networkx as nx
import polars as pl
from rich import box
from rich.align import Align
from rich.live import Live
from rich.table import Table

from locus.data.QuadTree import CellState, QuadTree
from locus.utils.paths import PROCESSED_DATA_DIR


def setup_table():
    table = Table(show_footer=True)
    table_centered = Align.center(table)
    table.add_column("Max Depth")
    table.add_column("Active Cells", "0%")
    table.add_column("Evaluating Cells", "100%")
    table.add_column("Stopped Cells", "0%")
    table.title = "Generating QuadTree..."
    table.row_styles = ["none", "dim"]

    table.border_style = "bright_yellow"

    table.caption = f"Elapsed time: [b magenta not dim]{0}[/]h [b magenta not dim]{0}[/]m [b magenta not dim]{0}[/]s"

    for box_style in [
        box.SQUARE,
        box.MINIMAL,
        box.SIMPLE,
        box.SIMPLE_HEAD,
    ]:
        table.box = box_style
    return table, table_centered


def update_time(start_time: float, table: Table):
    end_time = time.time()
    elapsed_time = end_time - start_time

    H = int(elapsed_time / 3600)
    M = int((elapsed_time % 3600) / 60)
    S = int(elapsed_time % 60)

    table.caption = f"Elapsed time: [b magenta not dim]{H}[/]h [b magenta not dim]{M}[/]m [b magenta not dim]{S}[/]s"


def update_time_thread(start_time, table, stop_flag):
    while not stop_flag.is_set():
        time.sleep(0.5)  # adjust sleep duration as needed
        update_time(start_time, table)


def update_table(tree: QuadTree, table: Table, prev_evaluating: int):
    curr_active = len(tree.get_nodes(CellState.ACTIVE))
    curr_evaluating = len(tree.get_nodes(CellState.EVALUATING))
    curr_stopped = len(tree.get_nodes(CellState.STOPPED))

    if curr_evaluating > prev_evaluating:
        s = "green"
    elif curr_evaluating < prev_evaluating:
        s = "red"
    else:
        s = "cyan"

    table.add_row(
        f"{nx.dag_longest_path_length(tree)}",
        f"{curr_active}",
        f"[{s}]{curr_evaluating}[/]",
        f"{curr_stopped}",
    )

    pct_active = 100 * curr_active / (curr_active + curr_evaluating + curr_stopped)
    pct_evaluating = 100 * curr_evaluating / (curr_active + curr_evaluating + curr_stopped)
    pct_stopped = 100 * curr_stopped / (curr_active + curr_evaluating + curr_stopped)

    table.columns[1].footer = f"[green]{pct_active:.2f}%[/]"
    table.columns[2].footer = f"[blue]{pct_evaluating:.2f}%[/]"
    table.columns[3].footer = f"[red]{pct_stopped:.2f}%[/]"

    return curr_evaluating


def partition_quadtree(tree: QuadTree, df: pl.LazyFrame, tau_min: int, tau_max: int):
    """Partition the dataframe using a quadtree algorithm

    Args:
        df (pl.LazyFrame): East boundary of the cell.
        tau_min (int): East boundary of the cell.
        tau_max (int): East boundary of the cell.
    """
    table, table_centered = setup_table()

    start_time = time.time()

    # Start the time update thread
    stop_flag = threading.Event()
    start_time_thread = threading.Thread(target=update_time_thread, args=(start_time, table, stop_flag))
    start_time_thread.daemon = True
    start_time_thread.start()

    prev_evaluating = 0
    with Live(table_centered, refresh_per_second=4):  # update 4 times a second to feel fluid
        while tree.is_evaluating():
            # update_time(start_time, table)
            prev_evaluating = update_table(tree, table, prev_evaluating)

            tree.expand()
            tree.evaluate_cells(df, tau_min, tau_max)

        # update_time(start_time, table)
        update_table(tree, table, prev_evaluating)

    # Set the stop flag to signal the time update thread to exit
    stop_flag.set()
    # Ensure that the time update thread finishes before the program exits
    start_time_thread.join()


@click.command()
@click.argument("shards", required=False, nargs=-1)
@click.option("--tau_min", default=50, help="Minimum number of data points in a cell.")
@click.option("--tau_max", default=2000, help="Maximum number of data points in a cell.")
@click.option("--output", default="quadtree.gml", help="Output file for saving the QuadTree.")
def main(shards: tuple[str], tau_min: int, tau_max: int, output: str):
    """Generate a quadtree from a dataset of image locations.

    Args:
        dataset (str): Name of the dataset to process.

    Raises:
        ValueError: If dataset is not supported.
    """

    # check if tau min is less than tau max and larger than 0
    if tau_min > tau_max:
        raise ValueError("tau_min must be less than tau_max")
    if tau_min < 0:
        raise ValueError("tau_min must be larger than 0")

    # check if quadtrees directory exists else create it
    quadtrees_dir = PROCESSED_DATA_DIR / "LDoGI" / "quadtrees"
    quadtrees_dir.mkdir(parents=True, exist_ok=True)

    shard_files: Path | list[Path]
    if len(shards) == 0:
        shard_files = PROCESSED_DATA_DIR / "LDoGI/shards/shard_*.parquet"
    else:
        if "all" in shards:
            shard_files = PROCESSED_DATA_DIR / "LDoGI/shards/shard_*.parquet"
        else:
            shard_files = [PROCESSED_DATA_DIR / f"LDoGI/shards/shard_{shard}.parquet" for shard in shards]

    df = pl.scan_parquet(shard_files)
    df = df.drop("id", "image")

    g = QuadTree()
    partition_quadtree(g, df, tau_min, tau_max)

    output_path = quadtrees_dir / output
    g.write_gml(output_path)


if __name__ == "__main__":
    main()
