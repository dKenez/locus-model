import tomllib
from pathlib import Path


class Hyperparams:
    def __init__(self, path: Path):
        toml_dict = tomllib.loads(path.read_text())

        # To tune
        self.batch_size = toml_dict["batch_size"]
        self.epochs = toml_dict["epochs"]
        self.quadtree = toml_dict["quadtree"]
        self.data_fraction = toml_dict["data_fraction"]

    def model_name(self):
        return f"locus_model-B{self.batch_size}-E{self.epochs}-qt{self.quadtree[:-4]}.pth"


if __name__ == "__main__":
    from locus.utils.paths import PROJECT_ROOT

    h = Hyperparams(PROJECT_ROOT / "train_conf.toml")

    print(h.model_name())
