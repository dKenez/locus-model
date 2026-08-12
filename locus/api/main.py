import os
from io import BytesIO
from pathlib import Path
from typing import cast

import networkx as nx  # Import the missing package
import torch
import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from networkx import DiGraph
from PIL import Image

from locus.models.model import LDoGIResnet
from locus.models.transforms import LDoGItransforms
from locus.utils.cell_utils import CellState
from locus.utils.paths import MODELS_DIR, PROCESSED_DATA_DIR

# Instantiate the FastAPI app
app = FastAPI()

# Add CORS middleware to allow requests from the specified origins
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

# Load the quadtree graph, and extract the active cells
QT = "qt_min10_max1000_df10pct.gml"
G = cast(DiGraph, nx.read_gml(PROCESSED_DATA_DIR / f"LDoGI/quadtrees/{QT}"))
active_cells = [node for node in list(G.nodes) if G.nodes[node]["state"] == CellState.ACTIVE.value]

num_classes = len(active_cells)

# Load the model and weights, use the CPU
model = LDoGIResnet(classes=num_classes, layers=50)

model.load_state_dict(
    torch.load(MODELS_DIR / "runs/weary-independence/weights/epoch_008.pth", map_location=torch.device("cpu"))
)


# Predict endpoint
@app.post("/predict/")
async def predict(img: UploadFile):
    # Read and transform the image
    img_read = await img.read()
    img_bytes = BytesIO(img_read)
    pil_img = Image.open(img_bytes)
    t_img = LDoGItransforms(pil_img)
    t_img = t_img.unsqueeze(0)

    # Make a prediction
    model.eval()
    with torch.no_grad():
        logits = model(t_img)
        # The model returns raw logits; convert to probabilities for the client
        output = torch.softmax(logits, dim=1)

    # Extract the cell predictions
    cell_predictions = [(active_cells[i], float(prob)) for i, prob in enumerate(output[0])]

    return cell_predictions


# Optionally serve the built locus-ui bundle from this same app. When LOCUS_UI_DIR points
# at a locus-ui `build/` directory, the UI and the API share an origin, so the browser
# calls /predict/ directly and no CORS negotiation is involved at all. Unset (the local
# dev case) the API serves only its own endpoints and the UI runs on the vite dev server.
# The mount must be registered last: it matches every path, so any route declared after it
# would be shadowed by the static handler.
_ui_dir = os.environ.get("LOCUS_UI_DIR")
if _ui_dir:
    ui_path = Path(_ui_dir)
    if not ui_path.is_dir():
        raise RuntimeError(f"LOCUS_UI_DIR is set to {_ui_dir!r}, which is not a directory")
    app.mount("/", StaticFiles(directory=ui_path, html=True), name="ui")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
