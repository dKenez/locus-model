import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from mpl_toolkits.basemap import Basemap

from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT
from src.utils.pl_utils import batch_iter

df = pl.scan_parquet(PROCESSED_DATA_DIR / "LDoGI" / "shard_0.parquet")
df = df.drop("id", "image")
print(df.head().collect())
c = df.select(pl.len()).collect()

print(c["len"][0])


ARCTIC_ANTARCTIC_CIRCLE = 66.57

# polars filter out coordinates abovce the arctic circle and below the antarctic circle
# so we need to filter them out manually
df = df.filter((pl.col("latitude") < ARCTIC_ANTARCTIC_CIRCLE) & (pl.col("latitude") > -ARCTIC_ANTARCTIC_CIRCLE))

c = df.select(pl.len()).collect()
print(c["len"][0])


coords = df.first().collect().to_dict(as_series=False)

# lon_0 is central longitude of projection.
# resolution = 'c' means use crude resolution coastlines.
m = Basemap(projection="robin", lon_0=0, resolution="c")
m.drawcoastlines()
m.fillcontinents(color="coral", lake_color="aqua")
# draw parallels and meridians.
m.drawparallels(np.arange(-90.0, 120.0, 30.0))
m.drawmeridians(np.arange(0.0, 360.0, 60.0))
m.drawmapboundary(fill_color="aqua")

# iterating over df
all_rows = 0
print("iterating over df")
for coords in batch_iter(df):
    longs = coords["longitude"].to_numpy()
    lats = coords["latitude"].to_numpy()

    x, y = m(coords["longitude"], coords["latitude"])

    m.plot(x, y, "bo", markersize=1)  # plot a blue dot there
plt.title("Robinson Projection")
# plt.show()
print(PROJECT_ROOT / "reports" / "figures" / "test.png")
plt.savefig("test.png", dpi=600)
