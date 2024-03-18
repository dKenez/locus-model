from io import BytesIO
from typing import Union, cast

import networkx as nx  # Import the missing package
import torch
import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from networkx import DiGraph
from PIL import Image
from pydantic import BaseModel

from locus.models.model import LDoGIResnet
from locus.models.transforms import LDoGItransforms
from locus.utils.cell_utils import CellState
from locus.utils.paths import MODELS_DIR, PROCESSED_DATA_DIR

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:5173",
    "localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


QT = "qt_min10_max1000_df10pct.gml"
G = cast(DiGraph, nx.read_gml(PROCESSED_DATA_DIR / f"LDoGI/quadtrees/{QT}"))
active_cells = [node for node in list(G.nodes) if G.nodes[node]["state"] == CellState.ACTIVE.value]

num_classes = len(active_cells)

model = LDoGIResnet(classes=num_classes, layers=50)

model.load_state_dict(
    torch.load(MODELS_DIR / "runs/weary-subspace/weights/epoch_003.pth", map_location=torch.device("cpu"))
)


class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}


@app.post("/predict/")
async def predict(img: UploadFile):
    img_read = await img.read()
    img_bytes = BytesIO(img_read)
    pil_img = Image.open(img_bytes)
    t_img = LDoGItransforms(pil_img)
    t_img = t_img.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(t_img)

    cell_predictions = {active_cells[i]: float(prob) for i, prob in enumerate(output[0])}

    return cell_predictions


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
