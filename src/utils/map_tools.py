import numpy as np
from mpl_toolkits.basemap import Basemap


def geo_rect_to_proj(lats: tuple[float, float], lons: tuple[float, float], m: Basemap, res: int):
    # Convert to linspace
    lats_linspace = np.linspace(lats[0], lats[1], res)
    lons_linspace = np.linspace(lons[0], lons[1], res)

    # Define the edge points' coordinates
    west_edge_lats = lats_linspace
    west_edge_lons = [lons_linspace[0]] * res
    north_edge_lats = [lats_linspace[-1]] * res
    north_edge_lons = lons_linspace
    east_edge_lats = lats_linspace[::-1]
    east_edge_lons = [lons_linspace[-1]] * res
    south_edge_lats = [lats_linspace[0]] * res
    south_edge_lons = lons_linspace[::-1]

    # Convert to map projection
    geo_rect_lats = np.concatenate(
        [
            west_edge_lats,
            north_edge_lats,
            east_edge_lats,
            south_edge_lats,
        ]
    )
    geo_rect_lons = np.concatenate(
        [
            west_edge_lons,
            north_edge_lons,
            east_edge_lons,
            south_edge_lons,
        ]
    )
    x, y = m(geo_rect_lons, geo_rect_lats)

    xy = zip(x, y)
    # get random color
    colors = ["green", "blue", "red", "yellow", "purple", "orange", "black"]
    facecolor = np.random.choice(colors)

    # poly = Polygon(list(xy), facecolor=facecolor, alpha=0.4)
    # plt.gca().add_patch(poly)
