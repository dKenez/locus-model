from enum import Enum

import torch
from geopy import distance


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


def cell_center(cell: str):
    """Given a cell identifier, return the center of the cell in the form
    (latitude, longitude).

    Args:
        cell (str): Cell identifier string, corresponds to a path in the quadtree.

    Returns:
        tuple: (latitude, longitude)
    """
    south_lat, north_lat, west_lon, east_lon = cell_bounds(cell)
    return (south_lat + north_lat) / 2, (west_lon + east_lon) / 2


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

    def get_next_cell(
        lat: torch.Tensor,
        lon: torch.Tensor,
        south_lat: torch.Tensor,
        north_lat: torch.Tensor,
        west_lon: torch.Tensor,
        east_lon: torch.Tensor,
    ):
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
        if lon >= half_lon:
            quad += 1
            ret_west_lon = half_lon
        else:
            ret_east_lon = half_lon

        return quad, (ret_south_lat, ret_north_lat, ret_west_lon, ret_east_lon)

    # Initialize bounds to the entire world
    point_lat = torch.tensor(lat, dtype=torch.float32)
    point_lon = torch.tensor(lon, dtype=torch.float32)

    west_lon = torch.tensor(-180, dtype=torch.float32)
    east_lon = torch.tensor(180, dtype=torch.float32)
    south_lat = torch.tensor(-90, dtype=torch.float32)
    north_lat = torch.tensor(90, dtype=torch.float32)

    cell = ""
    cell_pool = [c for c in active_cells]  # Make a copy of the active cells

    # While there are still cells to check...
    while True:
        quad, (south_lat, north_lat, west_lon, east_lon) = get_next_cell(
            point_lat, point_lon, south_lat, north_lat, west_lon, east_lon
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


def distance_to_cell_center(lat: float, lon: float, cell: str):
    """Calculate the distance from a point to the center of a cell.

    Args:
        lat (float): Input latitude.
        lon (float): Input longitude.
        cell (str): Cell identifier string, corresponds to a path in the quadtree.

    Returns:
        float: The distance from the point to the center of the cell.
    """
    cell_lat, cell_lon = cell_center(cell)
    return distance.great_circle((lat, lon), (cell_lat, cell_lon))


def distance_to_cell_bounds(lat: float, lon: float, cell: str):
    """
    Calculate the distance between a latitude-longitude coordinate
    and a cell bounded by south latitude, north latitude,
    west longitude, and east longitude.
    """

    south_lat, north_lat, west_lon, east_lon = cell_bounds(cell)
    if south_lat <= lat <= north_lat and west_lon <= lon <= east_lon:
        return 0  # The point is inside the square

    def clamp(n, low, high):
        return max(low, min(n, high))

    # Calculate the distance to each edge of the square
    distances = [
        distance.great_circle((lat, lon), (south_lat, clamp(lon, west_lon, east_lon))),  # Distance to south edge
        distance.great_circle((lat, lon), (north_lat, clamp(lon, west_lon, east_lon))),  # Distance to north edge
        distance.great_circle((lat, lon), (clamp(lat, south_lat, north_lat), west_lon)),  # Distance to west edge
        distance.great_circle((lat, lon), (clamp(lat, south_lat, north_lat), east_lon)),  # Distance to east edge
    ]

    # Return the minimum distance to any edge of the square
    return min(distances).km
