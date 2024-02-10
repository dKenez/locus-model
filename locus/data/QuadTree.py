from enum import Enum
from pathlib import Path

import networkx as nx
import polars as pl


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
    (south latitude, north latitude, west longitude, east longitude).



    Args:
        cell (str): Cell identifier string, corresponds to a path in the quadtree.

    Returns:
        tuple: (south latitude, north latitude, west longitude, east longitude)
    """
    # Initialize bounds to the entire world
    south_lat = -90.0
    north_lat = 90.0
    west_lon = -180.0
    east_lon = 180.0

    # For each character in the cell identifier, update the bounds by entering the corresponding quadrant
    for divide in cell:
        if int(divide) < 2:  # 0 or 1 is north
            south_lat = (south_lat + north_lat) / 2
        else:  # 2 or 3 is south
            north_lat = (south_lat + north_lat) / 2

        if int(divide) % 2 == 0:  # 0 or 2 is west
            east_lon = (west_lon + east_lon) / 2
        else:  # 1 or 3 is east
            west_lon = (west_lon + east_lon) / 2

    return south_lat, north_lat, west_lon, east_lon


def is_in_cell(cell: str, lat: float, lon: float):
    """Check if a given latitude and longitude is within a given cell.

    Args:
        cell (str): Cell identifier string, corresponds to a path in the quadtree.
        lat (float): Latitude of the point to check.
        lon (float): Longitude of the point to check.

    Returns:
        bool: True if the point is within the cell, False otherwise.
    """
    south_lat, north_lat, west_lon, east_lon = cell_bounds(cell)
    return south_lat <= lat < north_lat and west_lon <= lon < east_lon


def calc_enclosing_cell(lat: float, lon: float, active_cells: list[str]):
    """
    Given a point (lon, lat) and a graph, return the cell that encloses the point.
    If the point is not enclosed by any cell, return None.

    Args:
        lat (float): Input latitude.
        lon (float): Input longitude.
        active_cells (list[str]): List of active cells in the graph.

    Returns:
        (str | None): The cell that encloses the point, or None if the point is not enclosed by any cell.
    """

    def get_next_cell(lat: float, lon: float, south_lat: float, north_lat: float, west_lon: float, east_lon: float):
        """Given a point (lon, lat) and the bounds of a cell, return the quadrant of the cell that the point is in,
        and the bounds of the new cell that the point is in.

        Args:
            lat (float): Input latitude.
            lon (float): Input longitude.
            south_lat (float): South boundary of the cell.
            north_lat (float): North boundary of the cell.
            west_lon (float): West bounary of the cell.
            east_lon (float): East boundary of the cell.

        Returns:
            tuple[Literal[3, 2, 1, 0], tuple[float, float, float, float]]: The quadrant of the cell that
                the point is in, and the bounds of the new cell that the point is in (south latitude, north latitude,
                west longitude, east longitude).
        """

        # Initialize return values to the input bounds
        ret_south_lat = south_lat
        ret_north_lat = north_lat
        ret_west_lon = west_lon
        ret_east_lon = east_lon

        quad = 0

        # Determine the vertical half of the cell that the point is in
        half_lat = (south_lat + north_lat) / 2
        if lat < half_lat:
            quad += 2
            ret_north_lat = half_lat
        else:
            ret_south_lat = half_lat

        # Determine the horizontal half of the cell that the point is in
        half_lon = (west_lon + east_lon) / 2
        if lon > half_lon:
            quad += 1
            ret_west_lon = half_lon
        else:
            ret_east_lon = half_lon

        return quad, (ret_south_lat, ret_north_lat, ret_west_lon, ret_east_lon)

    # Initialize bounds to the entire world
    south_lat = -90
    north_lat = 90
    west_lon = -180
    east_lon = 180

    cell = ""
    cell_pool = [c for c in active_cells]  # Make a copy of the active cells

    # While there are still cells to check...
    while True:
        quad, (south_lat, north_lat, west_lon, east_lon) = get_next_cell(
            lat, lon, south_lat, north_lat, west_lon, east_lon
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
            south_lat, north_lat, west_lon, east_lon = cell_bounds(node_id)
            filtered_df = df.filter(
                (pl.col("latitude") > south_lat)
                & (pl.col("latitude") < north_lat)
                & (pl.col("longitude") > west_lon)
                & (pl.col("longitude") < east_lon)
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

                ids = filtered_df.collect()["id"].to_list()
                self.excluded_ids.extend(ids)

    def is_evaluating(self):
        """Check if any cells in the graph are in the evaluating state.

        Returns:
            bool: True if any cells are in the evaluating state, False otherwise.
        """
        return CellState.EVALUATING in [node["state"] for node in self.nodes.values()]

    def get_nodes(self, state: CellState) -> list[str]:
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
