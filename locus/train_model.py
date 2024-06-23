import json
import time
from datetime import datetime
from math import inf
from pathlib import Path
from typing import cast

import click
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


def train_model(conf: str, cont: str):
    """
    Train a model using the specified configuration file.

    Args:
        conf (str): Path to the configuration file.
    """

    # Set the seed
    seeding(42)

    runs_dir = MODELS_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    runs_manifest = runs_dir / "manifest.csv"

    if not runs_manifest.exists():
        runs_manifest.write_text(
            "name,run_start,quadtree,data_fraction,label_smoothing,layers,batch_size,optim,epochs,lr,grace_period\n"
        )

    hyperparams = Hyperparams(Path(conf))
    run_name = randomname.generate("adj/emotions", "n/linear_algebra")
    base_run_name = cont
    if cont:
        if cont == "last":
            with open(runs_manifest, "r") as f:
                lines = f.readlines()
                last_line = lines[-1]
                base_run_name = last_line.split(",")[0]

        num = 1
        s = base_run_name.split("_")
        if len(s) > 1:
            num = int(s[-1]) + 1

        run_name = f"{s[0]}_{num}"

        print(f"Continuing training model {base_run_name} as {run_name}")

        hyperparams = Hyperparams(runs_dir / base_run_name / "run.json")
        seeding(43)

    else:
        if not Path(conf).exists():
            raise ValueError(f"Configuration file {conf} not found")
        print(f"Training model using configuration file: {conf}")

        with open(runs_manifest, "r") as f:
            for i in range(100):
                name_conflict = False
                for line in f.readlines():
                    if run_name == line.split(",")[0]:
                        name_conflict = True
                        break

                if name_conflict:
                    run_name = randomname.generate("adj/emotions", "n/linear_algebra")
                    continue
                else:
                    break
            else:
                raise ValueError("Couldn't find a unique run name after 100 tries")

    with open(runs_manifest, "a") as f:
        data_to_write = [
            run_name,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            hyperparams.quadtree,
            hyperparams.data_fraction,
            hyperparams.label_smoothing,
            hyperparams.layers,
            hyperparams.batch_size,
            hyperparams.optim,
            hyperparams.epochs,
            hyperparams.lr,
            hyperparams.grace_period,
        ]
        f.write(",".join([str(data) for data in data_to_write]) + "\n")

    current_run_dir = MODELS_DIR / "runs" / run_name
    weights_dir = current_run_dir / "weights"

    weights_dir.mkdir(parents=True, exist_ok=True)

    logger = RunLogger(
        [
            current_run_dir / "run.log",
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

    # from_id_val = to_id_test + 1
    # to_id_val = max_id

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

    # Move the model to the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the model
    model = LDoGIResnet(num_classes, hyperparams.layers)
    model = model.to(device)
    logger.info(f"Using device: {device}")

    # if we are continuing training, load the weights
    start_epoch = 1
    if cont:
        # get the last epoch
        # order the weights dir by the epoch number
        # load the weights from the last epoch
        base_weights_dir = runs_dir / base_run_name / "weights"
        b = sorted([_ for _ in base_weights_dir.glob("*.pth")])[-1]
        last_epoch = int(b.stem.split("_")[-1])
        start_epoch = last_epoch + 1

        model.load_state_dict(
            torch.load(base_weights_dir / f"epoch_{last_epoch:03}.pth", map_location=torch.device(device))
        )

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer: optim.Optimizer
    match hyperparams.optim:
        case "sgd":
            optimizer = optim.SGD(model.parameters(), lr=hyperparams.lr, momentum=0.9)
        case "adam":
            optimizer = optim.Adam(model.parameters(), lr=hyperparams.lr)
        case _:
            raise ValueError(f"Optimizer {hyperparams.optim} not recognized")

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

    for epoch in range(start_epoch, num_epochs + 1):
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
            file_path=current_run_dir / "progress.log",
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
        test_model_end = train_model_end

        # Evaluate the model on the test set
        model.eval()

        with torch.no_grad():
            for ids, inputs, labels, label_names, coordinates in EpochProgress(
                test_loader,
                desc="test",
                epoch=epoch,
                colour="green",
                file_path=current_run_dir / "progress.log",
            ):
                test_data_fetch_end = time.time()
                epoch_stat_dict["test_data_fetch_time"] += test_data_fetch_end - test_model_end

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
                epoch_stat_dict["test_model_time"] += test_data_fetch_end - test_model_end

        epoch_stat_dict["test_time"] = test_model_end - train_model_end
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
        stats_df.to_pandas().to_csv(current_run_dir / "stats.csv")

        with open(current_run_dir / "run.json", "w") as json_file:
            json.dump(train_stat_dict, json_file, indent=4)


@click.command()
@click.option("--conf", help="Training configuration file.", default=PROJECT_ROOT / "train_conf.toml")
@click.option(
    "--cont",
    help="Continue training of model. Supply the training name.",
    is_flag=False,
    flag_value="last",
    default="",
)
def main(conf: str, cont: str):
    """
    Train a model using the specified configuration file.
    """

    train_model(conf, cont)


if __name__ == "__main__":
    main()
