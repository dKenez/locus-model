import logging
from datetime import datetime

import networkx as nx
import psycopg2
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import dotenv_values
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50

from locus.data.QuadTree import CellState
from locus.models.dataset import LDoGIDataset
from locus.utils.paths import PROCESSED_DATA_DIR, SQL_DIR, WEIGHTS_DIR

current_datetime = datetime.now()
date_string = current_datetime.strftime("%Y-%m-%d")
time_string = current_datetime.strftime("%H-%M-%S")
# Specify the file path where you want to save the weights
weights_path = WEIGHTS_DIR / f"test_weights_{date_string}_{time_string}.pth"
log_path = WEIGHTS_DIR / f"test_log_{date_string}_{time_string}.log"

# Set the logging configuration
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[])

# Create a file handler and set the file name
file_handler = logging.FileHandler(log_path)
file_handler.setLevel(logging.DEBUG)  # You can set the level for the file handler independently

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Set the level for the console handler independently

# Create a formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Create a logger and add the handlers
logger = logging.getLogger()
logger.addHandler(file_handler)
logger.addHandler(console_handler)

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
max_id = records[0][0] // 10000

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

QUADTREE = "qt_50_5000.gml"
G = nx.read_gml(PROCESSED_DATA_DIR / f"LDoGI/quadtrees/{QUADTREE}")
active_cells = [node for node in list(G.nodes) if G.nodes[node]["state"] == CellState.ACTIVE.value]
num_classes = len(active_cells)
# Load the data
train_data = LDoGIDataset(quadtree=QUADTREE, from_id=from_id_train, to_id=to_id_train)
test_data = LDoGIDataset(quadtree=QUADTREE, from_id=from_id_test, to_id=to_id_test)


def cf_factory(num_classes: int):
    def cf(*args, **kwargs):
        ids = [i[0] for i in args[0]]
        image_tensors = [i[1] for i in args[0]]
        labels_list = [i[2][0] for i in args[0]]
        labels = torch.zeros((len(labels_list), num_classes), dtype=torch.float32)
        for i, label in enumerate(labels_list):
            if label is None:
                logger.critical(f"None label at index {i}")
                logger.critical(f"ids: {ids[i]}")
            labels[i][active_cells.index(label)] = 1
        images = torch.cat(image_tensors, dim=0)
        return ids, images, labels

    return cf


# Define the dataloaders
train_loader = DataLoader(
    train_data,
    collate_fn=cf_factory(num_classes),
    shuffle=True,
    batch_size=32,
    num_workers=1,
    prefetch_factor=2,
)

test_loader = DataLoader(
    test_data,
    collate_fn=cf_factory(num_classes),
    shuffle=True,
    batch_size=32,
    num_workers=1,
    prefetch_factor=2,
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
num_epochs = 30

logger.info(f"training set length:{len(train_data)}")
logger.info(f"batches:{len(train_data)//32}")

# Train the model
for epoch in range(num_epochs):
    # Train the model on the training set
    model.train()
    train_loss = 0.0
    for i, (ids, inputs, labels) in enumerate(train_loader):
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

    # Evaluate the model on the test set
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for i, (ids, inputs, labels) in enumerate(test_loader):
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


# Alternatively, save only the model's state_dict
torch.save(model.state_dict(), weights_path)
