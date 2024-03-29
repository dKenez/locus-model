import datetime
from typing import Literal, TypedDict


class DescribeFileEntry(TypedDict):
    """Structure of a file entry in the describe JSON file.

    Params:
        name (str): name of the file
        count (int): number of records in the file
        min_index (int): minimum index of record in the file
        max_index (int): maximum index of record in the file
    """

    name: str
    count: int
    min_index: int
    max_index: int


class DescribeJsonStructure(TypedDict):
    """Structure of the describe JSON file.

    Params:
        count (int): number of files
        files (list[DescribeFileEntry]): list of file entries

    See Also:
        DescribeFileEntry
    """

    count: int
    files: list[DescribeFileEntry]


class QuadTreeItemParams(TypedDict):
    """Structure of the parameters of a QuadTree item.

    Params:
        tau_min (int): minimum tau value
        tau_max (int): maximum tau value
        shards (list[int]): list of shard IDs used to generate the QuadTree
    """

    tau_min: int
    tau_max: int
    shards: list[int]


class QuadTreeItem(TypedDict):
    """Structure of a QuadTree item.

    Params:
        name (str): name of the QuadTree item
        params (QuadTreeItemParams): parameters used to generate the QuadTree
        excluded_ids (list[int]): list of IDs with no corresponding cells in the QuadTree
    """

    name: str
    params: QuadTreeItemParams
    excluded_ids: list[int]


class QuadTreeManifest(TypedDict):
    """Structure of the QuadTree manifest JSON file.

    Params:
        quadtrees (list[QuadTreeItem]): list of QuadTree items
    """

    quadtrees: list[QuadTreeItem]


class TrainStats(TypedDict):
    """Structure of the epoch stats JSON file.

    Params:
        epoch (int): epoch number
        loss (float): loss value
        accuracy (float): accuracy value
        val_loss (float): validation loss value
        val_accuracy (float): validation accuracy value
    """

    name: str
    quadtree: str
    data_fraction: float
    label_smoothing: bool
    layers: int
    batch_size: int
    optim: Literal["sgd", "adam"]
    epochs: int
    lr: float
    grace_period: int

    stopped_early: bool
    last_epoch: int
    best_epoch: int
    best_epoch_train_loss: float
    best_epoch_test_loss: float
    best_epoch_test_acc: float
    best_epoch_mean_squared_error: float

    run_start: str
    run_end: str
    run_time: float


class EpochStats(TypedDict):
    """Structure of the epoch stats JSON file.

    Params:
        epoch (int): epoch number
        loss (float): loss value
        accuracy (float): accuracy value
        val_loss (float): validation loss value
        val_accuracy (float): validation accuracy value
    """

    epoch: int
    epoch_start: datetime.datetime
    epoch_end: datetime.datetime
    train_loss: float
    test_loss: float
    test_acc: float
    mean_squared_error: float
    train_data_fetch_time: float
    train_model_time: float
    train_time: float
    model_save_time: float
    test_data_fetch_time: float
    test_model_time: float
    test_time: float
