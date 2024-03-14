import shutil
from datetime import datetime
from typing import cast

import networkx as nx
import psycopg2
import randomname
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import dotenv_values
from networkx import DiGraph
from torchvision.models import ResNet50_Weights, resnet50

from locus.data.QuadTree import CellState
from locus.models.dataloader import LDoGIDataLoader
from locus.models.dataset import LDoGIDataset
from locus.utils.Hyperparams import Hyperparams
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

run_dir = MODELS_DIR / "runs" / run_name
monitoring_dir = MODELS_DIR / "monitoring"
weights_dir = run_dir / "weights"

run_dir.mkdir(parents=True)
monitoring_dir.mkdir(parents=True)
weights_dir.mkdir(parents=True)

shutil.copy(PROJECT_ROOT / "train_conf.toml", run_dir / "train_conf.toml")
log_path = run_dir / "run.log"

logger = RunLogger([run_dir / "run.log", monitoring_dir / "run.log"])

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
max_id = int(records[0][0] * hyperparams.data_fraction)

# close the connection and cursor
cur.close()
conn.close()

# train test val - 70-20-10 split
from_id_train = 1
to_id_train = int(max_id * 0.7)

from_id_test = to_id_train + 1
to_id_test = from_id_test + int(max_id * 0.2)

from_id_val = to_id_test + 1
to_id_val = max_id

QUADTREE = hyperparams.quadtree
G = cast(DiGraph, nx.read_gml(PROCESSED_DATA_DIR / f"LDoGI/quadtrees/{QUADTREE}"))
active_cells = [node for node in list(G.nodes) if G.nodes[node]["state"] == CellState.ACTIVE.value]
num_classes = len(active_cells)

# Define datasets
train_data = LDoGIDataset(quadtree=QUADTREE, from_id=from_id_train, to_id=to_id_train)
test_data = LDoGIDataset(quadtree=QUADTREE, from_id=from_id_test, to_id=to_id_test)


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
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)


# Replace the last layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Move the model to the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(device)


# Define the number of epochs
num_epochs = hyperparams.epochs

logger.info(f"training set length:{len(train_data)}")
logger.info(f"batches:{len(train_data)//32}")

# Train the model
for epoch in range(1, num_epochs + 1):
    # Train the model on the training set
    model.train()
    train_loss = 0.0

    for i, (ids, inputs, labels, label_names) in enumerate(train_loader):
        if (i + 1) % 100 == 0:
            logger.info(f"batch {i+1}/{len(train_data)//32}")
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
        train_loss += loss.item() * inputs.size(0)

    # Save the model weights
    torch.save(model.state_dict(), weights_dir / f"epoch_{epoch+1:03}.pth")

    # Evaluate the model on the test set
    model.eval()
    test_loss = torch.tensor(0.0)
    test_acc = torch.tensor(0.0)
    with torch.no_grad():
        for i, (ids, inputs, labels, label_names) in enumerate(test_loader):
            # Move the data to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Update the test loss and accuracy
            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            _, ground_truth = torch.max(labels, 1)

            test_acc += torch.sum(preds == ground_truth)

    # Print the training and test loss and accuracy
    train_loss /= len(train_data)
    test_loss /= len(test_data)
    test_acc = test_acc.double() / len(test_data)
    logger.info(
        f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.4f} Test Loss: {test_loss:.4f} Test Acc: {test_acc:.4f}"  # noqa: E501
    )
