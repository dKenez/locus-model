from pathlib import Path
from typing import Any

import networkx as nx
import polars as pl
from psycopg2._psycopg import connection

from locus.utils.cell_utils import CellState, cell_bounds
from locus.utils.paths import SQL_DIR


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

    excluded_ids: list[int]

    def __init__(self, incoming_graph_data=None, **attr):
        """Initialize a QuadTree. Has a single root node with state CellState.EVALUATING.

        Args:
            incoming_graph_data (_type_, optional): Data for initialising the base directed graph with
            custom parameters.

            Defaults to None.
        """
        super().__init__(incoming_graph_data, **attr)
        self.add_node("", state=CellState.EVALUATING)

        self.excluded_ids = []

        with open(SQL_DIR / "select_count_lat_lon.sql", "r") as f:
            self.count_sql_string = f.read()

        with open(SQL_DIR / "select_ids_lat_lon.sql", "r") as f:
            self.ids_sql_string = f.read()

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

    def evaluate_cells(self, conn: connection, min_count: int, max_count: int, max_id: int):
        """Evaluate the cells in the graph to determine if they should be subdivided, stopped, or classed as active.

        - Cells with a number of data points less than min_count will be stopped.
        - Cells with a number of data points between min_count and max_count will be considered active.
        - Cells with a number of data points more than max_count will be subdivided further.

        Args:
            df (pl.LazyFrame): Polars LazyFrame containing the image locations with latitude and longitude columns.
            min_count (int): Threshold for minimum number of data points in a cell.
            max_count (int): Threshold for maximum number of data points in a cell.
            max_id (int): Upper limit of datapoints to consider.
        """
        # Iterate over nodes in the graph
        for node_id in list(self.nodes):
            node = self.nodes[node_id]

            # If the node is not in the evaluating state, skip it
            if not node["state"] == CellState.EVALUATING:
                continue

            # Get the bounds of the cell and filter the dataframe to only include points within the cell
            south_lat, north_lat, west_lon, east_lon = cell_bounds(node_id)

            sql_string = self.count_sql_string.format(south_lat, north_lat, west_lon, east_lon, max_id)
            count_in_cell = pl.read_database(sql_string, conn)["count"][0]

            # If the number of data points in the cell is less than min_count, stop the cell
            # If the number of data points in the cell is between min_count and max_count, make the cell active
            # If the number of data points in the cell is more than max_count, keep the cell evaluating
            if count_in_cell > min_count:
                if count_in_cell < max_count:
                    node["state"] = CellState.ACTIVE
            else:
                node["state"] = CellState.STOPPED

                sql_string = self.ids_sql_string.format(south_lat, north_lat, west_lon, east_lon)
                ids = pl.read_database(sql_string, conn)["id"].to_list()
                self.excluded_ids.extend(ids)

    def is_evaluating(self):
        """Check if any cells in the graph are in the evaluating state.

        Returns:
            bool: True if any cells are in the evaluating state, False otherwise.
        """
        return CellState.EVALUATING in [node["state"] for node in self.nodes.values()]

    def get_nodes(self, state: CellState) -> list[dict[str, Any]]:
        """Get all nodes in the graph with a given state.

        Args:
            state (CellState): The state to filter the nodes by.

        Returns:
            list[str]: List of nodes with the given state.
        """
        return [node for node in self.nodes.values() if node["state"] == state]

    def write_gml(self, path: str | Path):
        """Write the graph to a GML file.

        Args:
            path (str): Path to write the GML file to.
        """
        # convert all states to their value
        for node in self.nodes:
            self.nodes[node]["state"] = self.nodes[node]["state"].value
        nx.write_gml(self, path)
