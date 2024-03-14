import tomllib
from pathlib import Path


class Hyperparams:
    def __init__(self, path: Path):
        toml_dict = tomllib.loads(path.read_text())

        self.quadtree = toml_dict["quadtree"]
        self.data_fraction = toml_dict["data_fraction"]
        self.label_smoothing = toml_dict["label_smoothing"]

        self.layers = toml_dict["layers"]

        self.batch_size = toml_dict["batch_size"]
        self.optim = toml_dict["optim"]
        self.epochs = toml_dict["epochs"]
        self.lr = toml_dict["lr"]

        self.grace_period = toml_dict["grace_period"]


if __name__ == "__main__":
    from locus.utils.paths import PROJECT_ROOT

    h = Hyperparams(PROJECT_ROOT / "train_conf.toml")
