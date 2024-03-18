import json
import tomllib
from pathlib import Path
from typing import Literal


class Hyperparams:
    def __init__(self, path: Path):
        # if is toml file

        conf_dict = {}

        match path.suffix:
            case ".toml":
                conf_dict = tomllib.loads(path.read_text())

            case ".json":
                conf_dict = json.loads(path.read_text())
            case _:
                raise ValueError("Unsupported file format")

        self.quadtree = conf_dict["quadtree"]
        self.data_fraction = conf_dict["data_fraction"]
        self.label_smoothing = conf_dict["label_smoothing"]

        self.layers = conf_dict["layers"]

        self.batch_size = conf_dict["batch_size"]
        self.optim: Literal["sgd", "adam"] = conf_dict["optim"]
        self.epochs = conf_dict["epochs"]
        self.lr = conf_dict["lr"]

        self.grace_period = conf_dict["grace_period"]


if __name__ == "__main__":
    from locus.utils.paths import PROJECT_ROOT

    h = Hyperparams(PROJECT_ROOT / "train_conf.toml")
