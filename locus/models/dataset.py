import json
from io import BytesIO
from pathlib import Path
from typing import Iterable, cast

import networkx as nx
import numpy as np
import polars as pl
import psycopg2
import torch
from dotenv import dotenv_values
from networkx import DiGraph
from PIL import Image
from torch.utils.data import Dataset

from locus.models.transforms import LDoGItransforms
from locus.utils.cell_utils import CellState, calc_enclosing_cell, distance_to_cell_center
from locus.utils.normal_distribution import normal_distribution
from locus.utils.paths import PROCESSED_DATA_DIR, SQL_DIR


class LDoGIDataset(Dataset):
    __len = None

    def __init__(
        self,
        quadtree: str,
        from_id: int = 1,
        to_id: int | None = None,
        label_smoothing: bool = False,
        *,
        env: Path | str = ".env",
    ):
        config = dotenv_values(env)

        self.conn = psycopg2.connect(
            host=config["DB_HOST"],
            port=config["DB_PORT"],
            dbname=config["DB_NAME"],
            user=config["DB_USER"],
            password=config["DB_PASSWORD"],
        )
        self.cur = self.conn.cursor()

        # read PROCESSED_DATA_DIR / "LDoGI/quadtrees/maifest.json" into a dict

        manifest = json.load(open(PROCESSED_DATA_DIR / "LDoGI/quadtrees/manifest.json", "r"))

        # find the quadtree in the manifest
        quadtree_info = {}
        for qt in manifest["quadtrees"]:
            if qt["name"] == quadtree:
                quadtree_info = qt

        G = cast(DiGraph, nx.read_gml(PROCESSED_DATA_DIR / f"LDoGI/quadtrees/{quadtree_info['name']}"))
        active_cells = [node for node in list(G.nodes) if G.nodes[node]["state"] == CellState.ACTIVE.value]

        self.excluded_ids = [
            excl_id for excl_id in quadtree_info["excluded_ids"] if excl_id >= from_id and excl_id <= to_id
        ]

        self.excluded_ids = self.excluded_ids or [-1]

        self.active_cells = active_cells
        self.transforms = LDoGItransforms

        with open(SQL_DIR / "select_max_id.sql", "r") as f:
            sql_string = f.read()

        self.cur.execute(sql_string)

        # Retrieve query results
        max_id = self.cur.fetchall()[0][0]

        self.from_id = from_id
        self.to_id = to_id or max_id

        self.label_smoothing = label_smoothing

    def __del__(self):
        self.cur.close()
        self.conn.close()

    def __len__(self):
        if self.__len:
            return self.__len

        with open(SQL_DIR / "select_count_not_in.sql", "r") as f:
            sql_string = f.read()

        insert_string = "({})".format(", ".join((str(i) for i in self.excluded_ids)))
        sql_string = sql_string.format(self.from_id, self.to_id, insert_string)
        self.cur.execute(sql_string)

        # Retrieve query results
        records = self.cur.fetchall()
        self.__len = records[0][0]
        return self.__len

    def __getitem__(self, idx):
        def map_idx_to_id(idx):
            if idx >= len(self) or idx < -len(self):
                raise IndexError(f"Index out of bounds {idx}")

            if idx < 0:
                idx %= len(self)

            idx += self.from_id

            while idx in self.excluded_ids:
                idx_in_excluded_ids = self.excluded_ids.index(idx)
                idx = self.from_id + len(self) + idx_in_excluded_ids

            return idx

        # if idx is a list of indices
        if not isinstance(idx, Iterable):
            idx = [idx]

        idx = [map_idx_to_id(i) for i in idx]

        # read the sqls string from SQL_DIR / "select_batch.sql"
        with open(SQL_DIR / "select_batch.sql", "r") as f:
            sql_string = f.read()

        insert_string = "({})".format(", ".join((str(i) for i in idx)))
        sql_string = sql_string.format(insert_string)
        results_df = pl.read_database(sql_string, self.conn)
        # create dictionary of id to index in idx
        results_df = results_df.with_columns(pl.col("id").map_elements(lambda x: idx.index(x)).alias("order"))
        results_df = results_df.sort("order")

        ids = []
        ims = []
        label_names = []
        coords = []
        for row in results_df.iter_rows(named=True):
            ids.append(row["id"])

            im_io = BytesIO(row["image"])
            im_pil = Image.open(im_io)
            im = self.transforms(im_pil)
            ims.append(im)

            label_names.append(calc_enclosing_cell(row["latitude"], row["longitude"], self.active_cells))
            coords.append((np.array((row["latitude"], row["longitude"]))))

        ims_tensor = torch.stack(ims, dim=0)
        coords_array = np.stack(coords)

        labels = torch.zeros((len(label_names), len(self.active_cells)), dtype=torch.float32)
        for i, label in enumerate(label_names):
            labels[i][self.active_cells.index(label)] = 1

        if self.label_smoothing:
            smoothing_labels = torch.zeros((len(label_names), len(self.active_cells)), dtype=torch.float32)

            mean = 0  # Mean of the distribution
            std = 20  # [km] Standard deviation of the distribution

            for test_cell_idx in range(smoothing_labels.shape[1]):
                test_cell = self.active_cells[test_cell_idx]
                for batch_idx in range(smoothing_labels.shape[0]):
                    target_coords = coords_array[batch_idx]

                    # distance of cell from target cell
                    dist = distance_to_cell_center(target_coords[0], target_coords[1], test_cell)
                    smoothing_weight = normal_distribution(dist.km, mean, std)

                    smoothing_labels[batch_idx][test_cell_idx] = smoothing_weight

            labels += smoothing_labels

            labels = (labels.T / labels.sum(dim=1)).T

        return np.array(ids), ims_tensor, labels, np.array(label_names), coords_array


if __name__ == "__main__":
    dataset = LDoGIDataset("qt_min10_max1000_df10pct.gml", from_id=1, to_id=1_000_000, label_smoothing=True)
    fetch_idxs = [i for i in range(20)]
    # fetch_idxs.append(1_200_000)
    results = dataset[fetch_idxs]

    print(f"ids: {results[0]}")
    print(f"ids shape: {results[0].shape}")
    print(f"ids dtype: {results[0].dtype}")
    print()

    print(f"ims shape: {results[1].shape}")
    print(f"ims dtype: {results[1].dtype}")
    print()

    print(f"labels shape: {results[2].shape}")
    print(f"labels sum: {results[2].sum()}")
    print(f"labels dtype: {results[2].dtype}")
    print()

    print(f"labels name: {results[3]}")
    print(f"labels name shape: {results[3].shape}")
    print(f"labels name dtype: {results[3].dtype}")
    print()

    print(len(dataset))
