import runpy
from pathlib import Path

from Datasets.lib import dataset_utils
from Datasets.lib import pyg_utils

DATA_PATH = Path("Datasets/Data/RIKEN")


def prepare(force=False):
    namespace = runpy.run_path(str(DATA_PATH / "preprocess.py"))
    return namespace["prepare"](force=force)


def load_raw(sub_dataset_name=None):
    return dataset_utils.load_raw_frame(DATA_PATH, sub_dataset_name=sub_dataset_name)


def load_dataset(
    sub_dataset_name=None,
    shuffle=True,
    valid_ratio=0.1,
    test_ratio=0.1,
    random_seed=42,
    force_reload=False,
):
    return pyg_utils.load_graph_dataset(
        DATA_PATH,
        sub_dataset_name=sub_dataset_name,
        shuffle=shuffle,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
        force_reload=force_reload,
    )


def load_loader(
    sub_dataset_name=None,
    shuffle=True,
    valid_ratio=0.1,
    test_ratio=0.1,
    random_seed=42,
    batch_size=64,
    force_reload=False,
):
    return pyg_utils.load_graph_loader(
        DATA_PATH,
        sub_dataset_name=sub_dataset_name,
        shuffle=shuffle,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
        batch_size=batch_size,
        force_reload=force_reload,
    )


def get_graph_dimensions(sub_dataset_name=None):
    return pyg_utils.get_graph_dimensions()


def plot(path=None):
    return dataset_utils.plot_dataset(load_raw, DATA_PATH, path=path)


if __name__ == "__main__":
    print(prepare())
    train_dataset, valid_dataset, test_dataset = load_dataset()
    print(len(train_dataset), len(valid_dataset), len(test_dataset))
    print(plot())
