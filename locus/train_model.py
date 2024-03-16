import json
import time
from datetime import datetime
from math import inf
from typing import cast

import networkx as nx
import polars as pl
import psycopg2
import randomname
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import dotenv_values
from networkx import DiGraph

from locus.models.dataloader import LDoGIDataLoader
from locus.models.dataset import LDoGIDataset
from locus.models.model import LDoGIResnet
from locus.utils.cell_utils import CellState, distance_to_cell_bounds
from locus.utils.EpochProgress import EpochProgress
from locus.utils.Hyperparams import Hyperparams
from locus.utils.interfaces import EpochStats, TrainStats
from locus.utils.justify_table import justify_table
from locus.utils.paths import MODELS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, SQL_DIR
from locus.utils.RunLogger import RunLogger
from locus.utils.seeding import seeding

# Set the seed
seeding(42)

hyperparams = Hyperparams(PROJECT_ROOT / "train_conf.toml")

current_datetime = datetime.now()
date_string = current_datetime.strftime("%Y-%m-%d")
time_string = current_datetime.strftime("%H-%M-%S")

run_name = randomname.generate("adj/emotions", "n/linear_algebra")

monitoring_dir = MODELS_DIR / "monitoring"
run_dir = MODELS_DIR / "runs" / run_name
weights_dir = run_dir / "weights"

monitoring_dir.mkdir(parents=True, exist_ok=True)
weights_dir.mkdir(parents=True)

# remove all files from the monitoring directory
for file in monitoring_dir.glob("*"):
    if file.is_file():
        file.unlink()

logger = RunLogger(
    [
        run_dir / "run.log",
        # monitoring_dir / "run.log"
    ]
)
logger.info(f"Start run: {run_name}")
print(f"Start run: {run_name}")

config = dotenv_values()

conn = psycopg2.connect(
    host=config["DB_HOST"],
    port=config["DB_PORT"],
    dbname=config["DB_NAME"],
    user=config["DB_USER"],
    password=config["DB_PASSWORD"],
)
cur = conn.cursor()

with open(SQL_DIR / "select_max_id.sql", "r") as f:
    sql_string = f.read()
cur.execute(sql_string)
records = cur.fetchall()
# close the connection and cursor
cur.close()
conn.close()


# train test val - 70-20-10 split
max_id = int(records[0][0] * hyperparams.data_fraction)
from_id_train = 1
to_id_train = int(max_id * 0.7)

from_id_test = to_id_train + 1
to_id_test = from_id_test + int(max_id * 0.2)

from_id_val = to_id_test + 1
to_id_val = max_id

logger.info(f"Using quadtree: {hyperparams.quadtree}")
G = cast(DiGraph, nx.read_gml(PROCESSED_DATA_DIR / f"LDoGI/quadtrees/{hyperparams.quadtree}"))
active_cells = [node for node in list(G.nodes) if G.nodes[node]["state"] == CellState.ACTIVE.value]
num_classes = len(active_cells)

# Define datasets
train_data = LDoGIDataset(
    quadtree=hyperparams.quadtree,
    from_id=from_id_train,
    to_id=to_id_train,
    label_smoothing=hyperparams.label_smoothing,
    env=PROJECT_ROOT / ".env",
)
test_data = LDoGIDataset(
    quadtree=hyperparams.quadtree,
    from_id=from_id_test,
    to_id=to_id_test,
    label_smoothing=hyperparams.label_smoothing,
    env=PROJECT_ROOT / ".env",
)


# Define dataloaders
train_loader = LDoGIDataLoader(
    train_data,
    shuffle=True,
    batch_size=hyperparams.batch_size,
    num_workers=1,
    prefetch_factor=10,
)

test_loader = LDoGIDataLoader(
    test_data,
    shuffle=True,
    batch_size=hyperparams.batch_size,
    num_workers=1,
    prefetch_factor=10,
)

# Define the model
model = LDoGIResnet(num_classes, hyperparams.layers)


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()

optimizer: optim.Optimizer
match hyperparams.optim:
    case "sgd":
        optimizer = optim.SGD(model.parameters(), lr=hyperparams.lr, momentum=0.9)
    case "adam":
        optimizer = optim.Adam(model.parameters(), lr=hyperparams.lr)
    case other:
        raise ValueError(f"Optimizer {hyperparams.optim} not recognized")


# Move the model to the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
logger.info(f"Using device: {device}")


# Define the number of epochs
num_epochs = hyperparams.epochs

logger.info(f"Batch size: {hyperparams.batch_size}")
logger.info(f"Training data: {len(train_data)} datapoints in {len(train_loader)} batches")
logger.info(f"Test data    : {len(test_data)} datapoints in {len(test_loader)} batches")


logger.info(justify_table(["Epoch", "Train Loss", "Test Loss", "Test Acc", "MSE"], [11, 14, 13, 12, 12]))
# Train the model

run_start = datetime.now()
train_stat_dict: TrainStats = {
    "name": run_name,
    "quadtree": hyperparams.quadtree,
    "data_fraction": hyperparams.data_fraction,
    "label_smoothing": hyperparams.label_smoothing,
    "layers": hyperparams.layers,
    "batch_size": hyperparams.batch_size,
    "optim": hyperparams.optim,
    "epochs": hyperparams.epochs,
    "lr": hyperparams.lr,
    "grace_period": hyperparams.grace_period,
    "stopped_early": False,
    "last_epoch": 1,
    "best_epoch": 1,
    "best_epoch_train_loss": 0,
    "best_epoch_test_loss": 0,
    "best_epoch_test_acc": 0,
    "best_epoch_mean_squared_error": 0,
    "run_start": run_start.strftime("%Y-%m-%d %H:%M:%S"),
    "run_end": datetime(1, 1, 1).strftime("%Y-%m-%d %H:%M:%S"),
    "run_time": 0,
}


stats_schema = {
    "epoch": pl.UInt32,
    "epoch_start": pl.Datetime,
    "epoch_end": pl.Datetime,
    "train_loss": pl.Float64,
    "test_loss": pl.Float64,
    "test_acc": pl.Float64,
    "mean_squared_error": pl.Float64,
    "train_data_fetch_time": pl.Float64,
    "train_model_time": pl.Float64,
    "train_time": pl.Float64,
    "model_save_time": pl.Float64,
    "test_data_fetch_time": pl.Float64,
    "test_model_time": pl.Float64,
    "test_time": pl.Float64,
}

stats_df = pl.DataFrame({}, schema=stats_schema)
best_epoch = 1
best_epoch_train_loss = inf
best_epoch_test_loss = inf
best_epoch_test_acc = 0.0
best_epoch_mean_squared_error = inf

EARLY_STOP_TRIGGER = False

for epoch in range(1, num_epochs + 1):
    if EARLY_STOP_TRIGGER:
        break

    epoch_stat_dict: EpochStats = {
        "epoch": epoch,
        "epoch_start": datetime.now(),
        "epoch_end": datetime(1, 1, 1),
        "train_loss": 0,
        "test_loss": 0,
        "test_acc": 0,
        "mean_squared_error": 0,
        "train_data_fetch_time": 0,
        "train_model_time": 0,
        "train_time": 0,
        "model_save_time": 0,
        "test_data_fetch_time": 0,
        "test_model_time": 0,
        "test_time": 0,
    }
    epoch_start = time.time()
    train_model_end = epoch_start

    # Train the model on the training set
    model.train()

    for ids, inputs, labels, label_names, coordinates in EpochProgress(
        train_loader,
        desc="train",
        epoch=epoch,
        colour="blue",
        file_path=run_dir / "progress.log",
    ):
        data_fetch_end = time.time()
        epoch_stat_dict["train_data_fetch_time"] += data_fetch_end - train_model_end

        # Move the data to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update the training loss
        epoch_stat_dict["train_loss"] += loss.item() * inputs.size(0)

        train_model_end = time.time()
        epoch_stat_dict["train_model_time"] += data_fetch_end - epoch_start

    train_end = time.time()
    epoch_stat_dict["train_time"] = train_end - epoch_start

    # Save the model weights
    torch.save(model.state_dict(), weights_dir / f"epoch_{epoch:03}.pth")

    model_save_end = time.time()
    epoch_stat_dict["model_save_time"] = model_save_end - train_end
    train_model_end = model_save_end

    # Evaluate the model on the test set
    model.eval()

    with torch.no_grad():
        for ids, inputs, labels, label_names, coordinates in EpochProgress(
            test_loader,
            desc="test",
            epoch=epoch,
            colour="green",
            file_path=run_dir / "progress.log",
        ):
            test_data_fetch_end = time.time()
            epoch_stat_dict["test_data_fetch_time"] += test_data_fetch_end - train_model_end

            # Move the data to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Update the test loss and accuracy
            epoch_stat_dict["test_loss"] += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            _, ground_truth = torch.max(labels, 1)

            for coords, pred_cell_idx in zip(coordinates, preds):
                # Calculate the distance from the point to the center of the cell
                distance = distance_to_cell_bounds(coords[0], coords[1], test_data.active_cells[pred_cell_idx])
                epoch_stat_dict["mean_squared_error"] += distance**2

            epoch_stat_dict["test_acc"] += float(torch.sum(preds == ground_truth).to("cpu"))

            test_model_end = time.time()
            epoch_stat_dict["test_model_time"] += test_data_fetch_end - train_model_end

    test_end = time.time()
    epoch_stat_dict["test_time"] = test_end - train_model_end
    epoch_stat_dict["epoch_end"] = datetime.now()

    # Print the training and test loss and accuracy
    epoch_stat_dict["train_loss"] /= len(train_data)
    epoch_stat_dict["test_loss"] /= len(test_data)
    epoch_stat_dict["test_acc"] /= len(test_data)
    epoch_stat_dict["mean_squared_error"] /= len(test_data)

    logger.info(
        justify_table(
            [
                f"{epoch}/{num_epochs}",
                f"{epoch_stat_dict['train_loss']:.4f}",
                f"{epoch_stat_dict['test_loss']:.4f}",
                f"{epoch_stat_dict['test_acc']:.4f}",
                f"{epoch_stat_dict['mean_squared_error']:.4f}",
            ],
            [11, 14, 13, 12, 12],
        )
    )

    if epoch_stat_dict["test_loss"] < best_epoch_test_loss:
        best_epoch = epoch
        best_epoch_train_loss = epoch_stat_dict["train_loss"]
        best_epoch_test_loss = epoch_stat_dict["test_loss"]
        best_epoch_test_acc = epoch_stat_dict["test_acc"]
        best_epoch_mean_squared_error = epoch_stat_dict["mean_squared_error"]
    else:
        # check if we are in the early stopping phase
        if epoch - best_epoch >= hyperparams.grace_period:
            logger.info(f"Early stopping at epoch {epoch}")
            logger.info(f"Best epoch was {best_epoch}")
            logger.info(f"Best test loss {best_epoch_test_loss:.4f}")

            EARLY_STOP_TRIGGER = True
            train_stat_dict["stopped_early"] = True

    train_stat_dict["last_epoch"] = epoch
    train_stat_dict["best_epoch"] = best_epoch
    train_stat_dict["best_epoch_train_loss"] = best_epoch_train_loss
    train_stat_dict["best_epoch_test_loss"] = best_epoch_test_loss
    train_stat_dict["best_epoch_test_acc"] = best_epoch_test_acc
    train_stat_dict["best_epoch_mean_squared_error"] = best_epoch_mean_squared_error
    train_stat_dict["run_end"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    train_stat_dict["run_time"] = (datetime.now() - run_start).total_seconds()

    stats_df = stats_df.extend(pl.DataFrame(epoch_stat_dict, schema=stats_schema))
    stats_df.to_pandas().to_csv(run_dir / "stats.csv")

    with open(run_dir / "run.json", "w") as json_file:
        json.dump(train_stat_dict, json_file, indent=4)
