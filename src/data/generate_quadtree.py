import time
from enum import Enum
from typing import List

import click
import networkx as nx
import polars as pl
from rich import box
from rich.align import Align
from rich.live import Live
from rich.table import Table

from src.utils.paths import PROCESSED_DATA_DIR


class CellState(Enum):
    """
    CellState is an enumeration of the possible states of a cell in a QuadTree.

    Attributes:
        STOPPED: 0 - The cell has less than tau_min data points and will not be further subdivided.
        EVALUATING: 1 - The cell is being evaluated to determine if it should be subdivided.
        ACTIVE: 2 - The cell has between tau_min and tau_max data points and will not be further subdivided.
    """

    STOPPED = 0
    EVALUATING = 1
    ACTIVE = 2


def cell_bounds(cell: str):
    """Given a cell identifier, return the bounds of the cell in the form
    (west longitude, east longitude, south latitude, north latitude).



    Args:
        cell (str): Cell identifier string, corresponds to a path in the quadtree.

    Returns:
        tuple: (west longitude, east longitude, south latitude, north latitude)
    """
    # Initialize bounds to the entire world
    west_long = -180.0
    east_long = 180.0
    south_lat = -90.0
    north_lat = 90.0

    # For each character in the cell identifier, update the bounds by entering the corresponding quadrant
    for divide in cell:
        if int(divide) < 2:  # 0 or 1 is north
            south_lat = (south_lat + north_lat) / 2
        else:  # 2 or 3 is south
            north_lat = (south_lat + north_lat) / 2

        if int(divide) % 2 == 0:  # 0 or 2 is west
            east_long = (west_long + east_long) / 2
        else:  # 1 or 3 is east
            west_long = (west_long + east_long) / 2

    return west_long, east_long, south_lat, north_lat


def is_in_cell(cell: str, lat: float, long: float):
    """Check if a given latitude and longitude is within a given cell.

    Args:
        cell (str): Cell identifier string, corresponds to a path in the quadtree.
        lat (float): Latitude of the point to check.
        long (float): Longitude of the point to check.

    Returns:
        bool: True if the point is within the cell, False otherwise.
    """
    west_long, east_long, south_lat, north_lat = cell_bounds(cell)
    return west_long <= long <= east_long and south_lat <= lat <= north_lat


def calc_enclosing_cell(lon: float, lat: float, active_cells: List[str]):
    """
    Given a point (lon, lat) and a graph, return the cell that encloses the point.
    If the point is not enclosed by any cell, return None.

    Args:
        lon (float): Input longitude.
        lat (float): Input latitude.
        active_cells (List[str]): List of active cells in the graph.

    Returns:
        (str | None): The cell that encloses the point, or None if the point is not enclosed by any cell.
    """

    def get_next_cell(lon: float, lat: float, west_lon: float, east_lon: float, south_lat: float, north_lat: float):
        """Given a point (lon, lat) and the bounds of a cell, return the quadrant of the cell that the point is in,
        and the bounds of the new cell that the point is in.

        Args:
            lon (float): Input longitude.
            lat (float): Input latitude.
            west_lon (float): West bounary of the cell.
            east_lon (float): East boundary of the cell.
            south_lat (float): South boundary of the cell.
            north_lat (float): North boundary of the cell.

        Returns:
            tuple[Literal[3, 2, 1, 0], tuple[float, float, float, float]]: The quadrant of the cell that
                the point is in, and the bounds of the new cell that the point is in.
        """

        # Initialize return values to the input bounds
        ret_west_lon = west_lon
        ret_east_lon = east_lon
        ret_south_lat = south_lat
        ret_north_lat = north_lat

        quad = 0

        # Determine the horizontal half of the cell that the point is in
        half_lon = (west_lon + east_lon) / 2
        if lon > half_lon:
            quad += 1
            ret_west_lon = half_lon
        else:
            ret_east_lon = half_lon

        # Determine the vertical half of the cell that the point is in
        half_lat = (south_lat + north_lat) / 2
        if lat < half_lat:
            quad += 2
            ret_north_lat = half_lat
        else:
            ret_south_lat = half_lat

        return quad, (ret_west_lon, ret_east_lon, ret_south_lat, ret_north_lat)

    # Initialize bounds to the entire world
    west_lon = -180
    east_lon = 180
    south_lat = -90
    north_lat = 90

    cell = ""
    cell_pool = [c for c in active_cells]  # Make a copy of the active cells

    # While there are still cells to check...
    while True:
        quad, (west_lon, east_lon, south_lat, north_lat) = get_next_cell(
            lon, lat, west_lon, east_lon, south_lat, north_lat
        )

        # Build the cell identifier
        cell += str(quad)
        # Filter the cell pool to only include cells that are still possible
        cell_pool = [c for c in cell_pool if c.startswith(cell)]

        # If there is only one cell left in the pool, return it
        if len(cell_pool) == 1 and cell == cell_pool[0]:
            return cell

        # If there are no cells left in the pool, return None
        if len(cell_pool) == 0:
            f"Not found: {cell}"
            return None


class QuadTree(nx.DiGraph):
    """QuadTree is a subclass of networkx.DiGraph that represents a quadtree.

    ### A visual of the QuadTree can be seen below:

    ```
    ╔═════════╤════╤════╗
    ║         │ 10 ┊ 11 ║
    ║    0    ├╌╌╌╌┼╌╌╌╌╢
    ║         │ 12 ┊ 13 ║
    ╟─────────┼────┴────╢
    ║         │         ║
    ║    2    │    3    ║
    ║         │         ║
    ╚═════════╧═════════╝
    ```
    The root node is the empty string, and each child node is a concatenation of the parent node and a digit 0-3.

    0: North-West
    1: North-East
    2: South-West
    3: South-East
    """

    def __init__(self, incoming_graph_data=None, **attr):
        """Initialize a QuadTree. Has a single root node with state CellState.EVALUATING.

        Args:
            incoming_graph_data (_type_, optional): Data for initialising the base directed graph with custom parameters.
            Defaults to None.
        """
        super().__init__(incoming_graph_data, **attr)
        self.add_node("", state=CellState.EVALUATING)

    def expand(self):
        """Expand the quadtree by subdividing all cells in the evaluating state into 4 new cells."""

        # Iterate over nodes in the graph
        for node_id in list(self.nodes):
            node = self.nodes[node_id]

            # If the node is in the evaluating state...
            if node["state"] == CellState.EVALUATING:
                # ...set the node to the stopped state
                node["state"] = CellState.STOPPED

                # ...add 4 new nodes to the graph (evaluating state), and connect them to the parent node
                for i in range(4):
                    self.add_node(f"{node_id}{i}", state=CellState.EVALUATING)
                    self.add_edge(node_id, f"{node_id}{i}")

    def evaluate_cells(self, df: pl.LazyFrame, min_count: int, max_count: int):
        """Evaluate the cells in the graph to determine if they should be subdivided, stopped, or classed as active.

        - Cells with a number of data points less than min_count will be stopped.
        - Cells with a number of data points between min_count and max_count will be considered active.
        - Cells with a number of data points more than max_count will be subdivided further.

        Args:
            df (pl.LazyFrame): Polars LazyFrame containing the image locations with latitude and longitude columns.
            min_count (int): Threshold for minimum number of data points in a cell.
            max_count (int): Threshold for maximum number of data points in a cell.
        """
        # Iterate over nodes in the graph
        for node_id in list(self.nodes):
            node = self.nodes[node_id]

            # If the node is not in the evaluating state, skip it
            if not node["state"] == CellState.EVALUATING:
                continue

            # Get the bounds of the cell and filter the dataframe to only include points within the cell
            west_long, east_long, south_lat, north_lat = cell_bounds(node_id)
            filtered_df = df.filter(
                (pl.col("latitude") < north_lat)
                & (pl.col("latitude") > south_lat)
                & (pl.col("longitude") < east_long)
                & (pl.col("longitude") > west_long)
            )

            count_in_cell = filtered_df.count().collect()["latitude"][0]

            # If the number of data points in the cell is less than min_count, stop the cell
            # If the number of data points in the cell is between min_count and max_count, make the cell active
            # If the number of data points in the cell is more than max_count, keep the cell evaluating
            if count_in_cell > min_count:
                if count_in_cell < max_count:
                    node["state"] = CellState.ACTIVE
            else:
                node["state"] = CellState.STOPPED

    def is_evaluating(self):
        """Check if any cells in the graph are in the evaluating state.

        Returns:
            bool: True if any cells are in the evaluating state, False otherwise.
        """
        return CellState.EVALUATING in [node["state"] for node in self.nodes.values()]

    def get_nodes(self, state: CellState) -> List[str]:
        """Get all nodes in the graph with a given state.

        Args:
            state (CellState): The state to filter the nodes by.

        Returns:
            list: List of nodes with the given state.
        """
        return [node for node in self.nodes.values() if node["state"] == state]

    def partition_quadtree(self, df: pl.LazyFrame, tau_min: int, tau_max: int):
        """Partition the dataframe using a quadtree algorithm

        Args:
            df (pl.LazyFrame): East boundary of the cell.
            tau_min (int): East boundary of the cell.
            tau_max (int): East boundary of the cell.
        """
        table = Table(show_footer=True)
        table_centered = Align.center(table)
        table.add_column("Max Depth")
        table.add_column("Active Cells", "0%")
        table.add_column("Evaluating Cells", "100%")
        table.add_column("Stopped Cells", "0%")
        table.title = "Generating QuadTree..."
        table.row_styles = ["none", "dim"]

        table.border_style = "bright_yellow"

        for box_style in [
            box.SQUARE,
            box.MINIMAL,
            box.SIMPLE,
            box.SIMPLE_HEAD,
        ]:
            table.box = box_style
        start_time = time.time()
        prev_evaluating = 0
        with Live(table_centered, refresh_per_second=4):  # update 4 times a second to feel fluid
            while self.is_evaluating():
                end_time = time.time()
                elapsed_time = end_time - start_time
                table.caption = f"Elapsed time: [b magenta not dim]{elapsed_time:.3f}[/]s"

                curr_active = len(self.get_nodes(CellState.ACTIVE))
                curr_evaluating = len(self.get_nodes(CellState.EVALUATING))
                curr_stopped = len(self.get_nodes(CellState.STOPPED))

                if curr_evaluating > prev_evaluating:
                    s = "green"
                elif curr_evaluating < prev_evaluating:
                    s = "red"
                else:
                    s = "cyan"

                pct_active = 100 * curr_active / (curr_active + curr_evaluating + curr_stopped)
                pct_evaluating = 100 * curr_evaluating / (curr_active + curr_evaluating + curr_stopped)
                pct_stopped = 100 * curr_stopped / (curr_active + curr_evaluating + curr_stopped)

                table.add_row(
                    f"{nx.dag_longest_path_length(self)}",
                    f"{curr_active}",
                    f"[{s}]{curr_evaluating}[/]",
                    f"{curr_stopped}",
                )

                table.columns[1].footer = f"[green]{pct_active:.2f}%[/]"
                table.columns[2].footer = f"[red]{pct_evaluating:.2f}%[/]"
                table.columns[3].footer = f"[blue]{pct_stopped:.2f}%[/]"

                prev_evaluating = curr_evaluating
                self.expand()
                self.evaluate_cells(df, tau_min, tau_max)

            curr_active = len(self.get_nodes(CellState.ACTIVE))
            curr_evaluating = len(self.get_nodes(CellState.EVALUATING))
            curr_stopped = len(self.get_nodes(CellState.STOPPED))

            end_time = time.time()
            elapsed_time = end_time - start_time
            table.caption = f"Elapsed time: [b magenta not dim]{elapsed_time:.3f}[/]s"
            table.add_row(
                f"{nx.dag_longest_path_length(self)}",
                f"{curr_active}",
                f"[red]{curr_evaluating}[/]",
                f"{curr_stopped}",
            )

            pct_active = 100 * curr_active / (curr_active + curr_evaluating + curr_stopped)
            pct_evaluating = 100 * curr_evaluating / (curr_active + curr_evaluating + curr_stopped)
            pct_stopped = 100 * curr_stopped / (curr_active + curr_evaluating + curr_stopped)
            table.columns[1].footer = f"[green]{pct_active:.2f}%[/]"
            table.columns[2].footer = f"[red]{pct_evaluating:.2f}%[/]"
            table.columns[3].footer = f"[blue]{pct_stopped:.2f}%[/]"


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
    quadtrees_dir = PROCESSED_DATA_DIR / "quadtrees"
    quadtrees_dir.mkdir(parents=True, exist_ok=True)

    if len(shards) == 0:
        shard_files = PROCESSED_DATA_DIR / "LDoGI" / "shard_*.parquet"
    else:
        if "all" in shards:
            shard_files = PROCESSED_DATA_DIR / "LDoGI" / "shard_*.parquet"
        else:
            shard_files = [PROCESSED_DATA_DIR / "LDoGI" / f"shard_{shard}.parquet" for shard in shards]

    df = pl.scan_parquet(shard_files)
    df = df.drop("id", "image")

    g = QuadTree()
    g.partition_quadtree(df, tau_min, tau_max)

    output_path = quadtrees_dir / output
    nx.write_gml(g, output_path)


if __name__ == "__main__":
    main()
