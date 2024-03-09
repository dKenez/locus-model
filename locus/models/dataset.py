import json
from io import BytesIO
from typing import Iterable

import networkx as nx
import polars as pl
import psycopg2
import torch
from dotenv import dotenv_values
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from locus.data.QuadTree import CellState, calc_enclosing_cell
from locus.utils.paths import PROCESSED_DATA_DIR, SQL_DIR


class LDoGIDataset(Dataset):
    __len = None

    def __init__(self, quadtree: str, from_id: int = 1, to_id: int | None = None, *, env: str = ".env"):
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

        G = nx.read_gml(PROCESSED_DATA_DIR / f"LDoGI/quadtrees/{quadtree_info['name']}")
        active_cells = [node for node in list(G.nodes) if G.nodes[node]["state"] == CellState.ACTIVE.value]

        self.excluded_ids = [
            excl_id for excl_id in quadtree_info["excluded_ids"] if excl_id >= from_id and excl_id <= to_id
        ]

        self.excluded_ids = self.excluded_ids or [-1]

        self.active_cells = active_cells
        self.transforms = v2.Compose(
            [
                v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
                # ...
                v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
                v2.Resize(224, antialias=True),
                v2.RandomCrop(size=(224)),  # Or Resize(antialias=True)
                # ...
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # v2.ToDtype(torch.uint8, scale=True),
            ]
        )

        with open(SQL_DIR / "select_max_id.sql", "r") as f:
            sql_string = f.read()

        self.cur.execute(sql_string)

        # Retrieve query results
        max_id = self.cur.fetchall()[0][0]

        self.from_id = from_id
        self.to_id = to_id or max_id

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

        ids = []
        ims = []
        labels = []
        for row in results_df.iter_rows(named=True):
            ids.append(row["id"])

            im_io = BytesIO(row["image"])
            im_pil = Image.open(im_io)
            im = self.transforms(im_pil)
            ims.append(im)

            labels.append(calc_enclosing_cell(row["latitude"], row["longitude"], self.active_cells))

        ims_tensor = torch.stack(ims, dim=0)

        return ids, ims_tensor, labels


if __name__ == "__main__":
    dataset = LDoGIDataset("quadtree.gml", from_id=1, to_id=1_000_000)
    fetch_idxs = [i for i in range(20)]
    # fetch_idxs.append(1_200_000)
    results = dataset[fetch_idxs]
    print(results)
    print(len(dataset))
    # print(results)
