from io import BytesIO

import polars as pl
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from locus.data.QuadTree import calc_enclosing_cell


class LDoGIDataset(Dataset):
    def __init__(self, annotations_file, active_cells):
        self.data = pl.scan_parquet(annotations_file)
        self.active_cells = active_cells
        self.transforms = v2.Compose(
            [
                v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
                v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
                # ...
                v2.RandomResizedCrop(size=(360, 360), antialias=True),  # Or Resize(antialias=True)
                # ...
                # v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
                # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return self.data.select(pl.len()).collect()["len"][0]

    def __getitem__(self, idx):

        # if idx is a list of indices
        if isinstance(idx, list):
            predicates = [pl.col("id") == idx_n for idx_n in idx]
            rows = self.data.filter(predicates).collect()

            lat = rows["latitude"]
            lon = rows["longitude"]

            labels = []
            for i in range(len(lat)):
                labels.append(calc_enclosing_cell(lat[i], lon[i], self.active_cells))

            im_io = BytesIO(rows["image"][0])
            im_pil = Image.open(im_io)
            im = self.transforms(im_pil)

            return im, labels

        row = self.data.filter((pl.col("id") == idx)).collect()

        lat = row["latitude"][0]
        lon = row["longitude"][0]

        label = calc_enclosing_cell(lat, lon, self.active_cells)

        im_io = BytesIO(row["image"][0])
        im_pil = Image.open(im_io)
        im = self.transforms(im_pil)

        return im, label
