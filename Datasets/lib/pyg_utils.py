from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from Datasets.lib import dataset_utils
from Datasets.lib.smiles_to_pyg import (
    edge_feature_dim,
    node_feature_dim,
    smiles_to_pyg_data,
)
from settings import RANDOM_SEED


class RetentionTimeGraphDataset(Dataset):
    def __init__(self, csv_path, force_reload=False):
        self.csv_path = Path(csv_path)
        self.cache_path = self.csv_path.with_suffix(".pyg.pt")

        if force_reload or not self.cache_path.exists():
            frame = pd.read_csv(self.csv_path)
            self.graphs = []
            for sample_index, row in enumerate(
                tqdm(
                    frame.itertuples(index=False),
                    total=len(frame),
                    desc=f"SMILES->PyG {self.csv_path.name}",
                )
            ):
                self.graphs.append(
                    smiles_to_pyg_data(
                        row.smiles,
                        rt=float(row.rt_time),
                        sample_index=sample_index,
                    )
                )
            torch.save(self.graphs, self.cache_path)
        else:
            self.graphs = torch.load(
                self.cache_path, map_location="cpu", weights_only=False
            )

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]


def load_graph_collection(data_path, sub_dataset_name=None, force_reload=False):
    data_path = Path(data_path)

    if sub_dataset_name is not None:
        return RetentionTimeGraphDataset(
            data_path / f"{sub_dataset_name}.csv",
            force_reload=force_reload,
        )

    all_path = data_path / "all.csv"
    if all_path.exists():
        return RetentionTimeGraphDataset(all_path, force_reload=force_reload)

    dataset_list = [
        RetentionTimeGraphDataset(path, force_reload=force_reload)
        for path in dataset_utils.list_csv_paths(data_path)
    ]
    if len(dataset_list) == 1:
        return dataset_list[0]
    return ConcatDataset(dataset_list)


def shuffle_dataset(dataset, shuffle=True, random_seed=RANDOM_SEED):
    if not shuffle:
        return dataset
    generator = torch.Generator().manual_seed(random_seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    return Subset(dataset, indices)


def split_dataset(dataset, valid_ratio=0.1, test_ratio=0.1):
    total_size = len(dataset)
    train_end = int(total_size * (1 - valid_ratio - test_ratio))
    valid_end = int(total_size * (1 - test_ratio))

    train_dataset = Subset(dataset, range(0, train_end))
    valid_dataset = Subset(dataset, range(train_end, valid_end))
    test_dataset = Subset(dataset, range(valid_end, total_size))
    return train_dataset, valid_dataset, test_dataset


def split_dataset_by_folds(dataset, num_folds, fold_index, valid_ratio=0.1):
    if num_folds is None:
        raise ValueError("`num_folds` must not be None in fold mode.")
    if num_folds < 2:
        raise ValueError("`num_folds` must be at least 2.")
    if not 0 <= fold_index < num_folds:
        raise ValueError(
            f"`fold_index` must satisfy 0 <= fold_index < num_folds, got "
            f"fold_index={fold_index}, num_folds={num_folds}."
        )

    total_size = len(dataset)
    if total_size < num_folds:
        raise ValueError(
            f"Dataset size ({total_size}) must be >= `num_folds` ({num_folds})."
        )

    test_fold_size = total_size / num_folds
    if valid_ratio + test_fold_size / total_size >= 1:
        raise ValueError(
            "`valid_ratio` leaves no room for training samples in fold mode. "
            f"valid_ratio={valid_ratio}, num_folds={num_folds}."
        )

    fold_sizes = [total_size // num_folds] * num_folds
    for index in range(total_size % num_folds):
        fold_sizes[index] += 1

    fold_start = sum(fold_sizes[:fold_index])
    fold_end = fold_start + fold_sizes[fold_index]

    test_indices = list(range(fold_start, fold_end))
    remaining_indices = list(range(0, fold_start)) + list(range(fold_end, total_size))

    valid_size = int(total_size * valid_ratio)
    if valid_size <= 0:
        raise ValueError(
            "`valid_ratio` produced an empty validation set in fold mode. "
            f"total_size={total_size}, valid_ratio={valid_ratio}."
        )
    if valid_size >= len(remaining_indices):
        raise ValueError(
            "`valid_ratio` is too large for fold mode. "
            f"valid_size={valid_size}, available={len(remaining_indices)}."
        )

    valid_indices = remaining_indices[:valid_size]
    train_indices = remaining_indices[valid_size:]

    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    test_dataset = Subset(dataset, test_indices)
    return train_dataset, valid_dataset, test_dataset


def load_graph_dataset(
    data_path,
    sub_dataset_name=None,
    shuffle=True,
    valid_ratio=0.1,
    test_ratio=0.1,
    num_folds=None,
    fold_index=0,
    random_seed=RANDOM_SEED,
    force_reload=False,
):
    dataset = load_graph_collection(
        data_path,
        sub_dataset_name=sub_dataset_name,
        force_reload=force_reload,
    )
    dataset = shuffle_dataset(dataset, shuffle=shuffle, random_seed=random_seed)
    if num_folds is None:
        return split_dataset(dataset, valid_ratio=valid_ratio, test_ratio=test_ratio)
    return split_dataset_by_folds(
        dataset,
        num_folds=num_folds,
        fold_index=fold_index,
        valid_ratio=valid_ratio,
    )


def load_graph_loader(
    data_path,
    sub_dataset_name=None,
    shuffle=True,
    valid_ratio=0.1,
    test_ratio=0.1,
    num_folds=None,
    fold_index=0,
    random_seed=RANDOM_SEED,
    batch_size=64,
    force_reload=False,
):
    train_dataset, valid_dataset, test_dataset = load_graph_dataset(
        data_path,
        sub_dataset_name=sub_dataset_name,
        shuffle=shuffle,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        num_folds=num_folds,
        fold_index=fold_index,
        random_seed=random_seed,
        force_reload=force_reload,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader


def get_graph_dimensions():
    return node_feature_dim(), edge_feature_dim()
