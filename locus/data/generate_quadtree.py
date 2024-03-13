import json
import threading
import time

import click
import networkx as nx
import polars as pl
import psycopg2
from dotenv import dotenv_values
from psycopg2._psycopg import connection
from rich import box
from rich.align import Align
from rich.live import Live
from rich.table import Table

from locus.data.QuadTree import CellState, QuadTree
from locus.utils.paths import PROCESSED_DATA_DIR, SQL_DIR


def setup_table():
    table = Table(show_footer=True)
    table_centered = Align.center(table)
    table.add_column("Max Depth")
    table.add_column("Active Cells", "0%")
    table.add_column("Evaluating Cells", "100%")
    table.add_column("Stopped Cells", "0%")
    table.add_column("Excluded data points", "0%")
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


def update_table(tree: QuadTree, table: Table, prev_evaluating: int, total_data_points: int):
    curr_active = len(tree.get_nodes(CellState.ACTIVE))
    curr_evaluating = len(tree.get_nodes(CellState.EVALUATING))
    curr_stopped = len(tree.get_nodes(CellState.STOPPED))
    num_excluded = len(tree.excluded_ids)

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
        f"{num_excluded}",
    )

    pct_active = 100 * curr_active / (curr_active + curr_evaluating + curr_stopped)
    pct_evaluating = 100 * curr_evaluating / (curr_active + curr_evaluating + curr_stopped)
    pct_stopped = 100 * curr_stopped / (curr_active + curr_evaluating + curr_stopped)
    pct_stopped = 100 * curr_stopped / (curr_active + curr_evaluating + curr_stopped)
    pct_excluded = 100 * num_excluded / total_data_points

    table.columns[1].footer = f"[green]{pct_active:.2f}%[/]"
    table.columns[2].footer = f"[blue]{pct_evaluating:.2f}%[/]"
    table.columns[3].footer = f"[red]{pct_stopped:.2f}%[/]"
    table.columns[4].footer = f"[yellow]{pct_excluded:.2f}%[/]"

    return curr_evaluating


def partition_quadtree(tree: QuadTree, conn: connection, tau_min: int, tau_max: int, max_id: int):
    """Partition the dataframe using a quadtree algorithm

    Args:
        df (pl.LazyFrame): East boundary of the cell.
        tau_min (int): East boundary of the cell.
        tau_max (int): East boundary of the cell.
        max_id (int): Only generate quadtree on datapoints below this id.
    """
    table, table_centered = setup_table()

    start_time = time.time()

    # Start the time update thread
    stop_flag = threading.Event()
    start_time_thread = threading.Thread(target=update_time_thread, args=(start_time, table, stop_flag))
    start_time_thread.daemon = True
    start_time_thread.start()

    sql_string = tree.count_sql_string.format(-90, 90, -180, 180)
    total_data_points = pl.read_database(sql_string, conn)["count"][0]

    prev_evaluating = 0
    with Live(table_centered, refresh_per_second=4):  # update 4 times a second to feel fluid
        while tree.is_evaluating():
            # update_time(start_time, table)
            prev_evaluating = update_table(tree, table, prev_evaluating, total_data_points)

            tree.expand()
            tree.evaluate_cells(conn, tau_min, tau_max, max_id)

        # update_time(start_time, table)
        update_table(tree, table, prev_evaluating, total_data_points)

    # Set the stop flag to signal the time update thread to exit
    stop_flag.set()
    # Ensure that the time update thread finishes before the program exits
    start_time_thread.join()


@click.command()
@click.option("--tau-min", default=50, help="Minimum number of data points in a cell.")
@click.option("--tau-max", default=2000, help="Maximum number of data points in a cell.")
@click.option("--data-fraction", default=0.1, help="Fraction of the data to use.")
@click.option("--output", help="Output file for saving the QuadTree.")
def main(tau_min: int, tau_max: int, data_fraction: float, output: str):
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

    # check if manifest.json exists else create it
    manifest_file = quadtrees_dir / "manifest.json"
    if not manifest_file.exists():
        with open(manifest_file, "w") as f:
            f.write('{"quadtrees": []}')

    config = dotenv_values(".env")

    conn = psycopg2.connect(
        host=config["DB_HOST"] or "",
        port=config["DB_PORT"] or 0,
        dbname=config["DB_NAME"] or "",
        user=config["DB_USER"] or "",
        password=(config["DB_PASSWORD"] or ""),
    )

    cur = conn.cursor()

    with open(SQL_DIR / "select_max_id.sql") as f:
        cur.execute(f.read())

    # Retrieve query results
    max_id = int(cur.fetchall()[0][0] * data_fraction)

    g = QuadTree()
    partition_quadtree(g, conn, tau_min, tau_max, max_id)

    conn.close()
    cur.close()

    output_path = quadtrees_dir / output
    g.write_gml(output_path)

    # read the manifest.json file
    with open(manifest_file, "r") as f:
        manifest = json.load(f)

    # add the new QuadTree to the manifest
    manifest["quadtrees"].append(
        {
            "name": output,
            "params": {
                "tau_min": tau_min,
                "tau_max": tau_max,
                "max_id": max_id,
            },
            "excluded_ids": sorted(g.excluded_ids),
        }
    )

    # write the manifest back to the file
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=4)


if __name__ == "__main__":
    main()
