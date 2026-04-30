import math
import os
from pathlib import Path

import matplotlib
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

SAMPLE_ROOT = Path("Datasets/Samples")
RANDOM_STATE = 42

matplotlib_cache = Path("/tmp/matplotlib")
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
matplotlib.use("Agg")

RDLogger.DisableLog("rdApp.*")


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def display_name(text):
    return str(text).replace("_", " ")


def list_csv_paths(data_path):
    data_path = Path(data_path)
    return sorted(path for path in data_path.glob("*.csv") if path.is_file())


def list_sub_dataset_names(data_path):
    return [path.stem for path in list_csv_paths(data_path) if path.stem != "all"]


def load_raw_frame(data_path, sub_dataset_name=None):
    data_path = Path(data_path)

    if sub_dataset_name is not None:
        return pd.read_csv(data_path / f"{sub_dataset_name}.csv")

    all_path = data_path / "all.csv"
    if all_path.exists():
        return pd.read_csv(all_path)

    frames = [pd.read_csv(path) for path in list_csv_paths(data_path)]
    return pd.concat(frames, ignore_index=True)


def sample_frame(frame, n, random_state=RANDOM_STATE):
    if len(frame) <= n:
        return frame.reset_index(drop=True)
    return frame.sample(n=n, random_state=random_state).reset_index(drop=True)


def draw_molecule(ax, smiles, caption, size=(260, 220)):
    image = Draw.MolToImage(Chem.MolFromSmiles(str(smiles)), size=size)
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(caption, fontsize=10, labelpad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def finalize_plot(fig, path):
    path = Path(path)
    ensure_parent(path)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def default_sample_path(data_path):
    data_path = Path(data_path)
    return SAMPLE_ROOT / f"{data_path.name}.png"


def plot_dataset(load_raw, data_path, path=None):
    data_path = Path(data_path)
    if path is None:
        path = default_sample_path(data_path)

    sub_dataset_names = list_sub_dataset_names(data_path)
    title = display_name(data_path.name)

    if len(sub_dataset_names) <= 1:
        frame = load_raw()
        sampled = sample_frame(frame, 8)

        fig = plt.figure(figsize=(11, 11.5))
        fig.suptitle(f"{title} overview", fontsize=16)
        outer = gridspec.GridSpec(2, 1, height_ratios=[2.9, 1.7], hspace=0.28)
        top = gridspec.GridSpecFromSubplotSpec(
            2, 4, subplot_spec=outer[0], wspace=0.08, hspace=0.18
        )

        for idx, (_, row) in enumerate(sampled.iterrows()):
            ax = fig.add_subplot(top[idx // 4, idx % 4])
            caption = f"{row['rt_time'] / 60.0:.2f} min"
            draw_molecule(ax, smiles=row["smiles"], caption=caption)

        ax_hist = fig.add_subplot(outer[1])
        ax_hist.hist(frame["rt_time"] / 60.0, bins=60, alpha=0.75)
        ax_hist.set_title("Retention time distribution")
        ax_hist.set_xlabel("RT (minutes)")
        ax_hist.set_ylabel("Count")

        return finalize_plot(fig, path)

    sampled_rows = []
    counts = []
    labels = []
    rt_values = []

    for offset, sub_dataset_name in enumerate(sub_dataset_names):
        subset = load_raw(sub_dataset_name)
        sampled_rows.append(
            sample_frame(subset, 1, random_state=RANDOM_STATE + offset).iloc[0]
        )
        counts.append(len(subset))
        labels.append(display_name(sub_dataset_name))
        rt_values.append((subset["rt_time"] / 60.0).tolist())

    num_cols = 5
    num_rows = math.ceil(len(sampled_rows) / num_cols)

    fig = plt.figure(figsize=(14, 12.5))
    fig.suptitle(f"{title} overview", fontsize=16)
    outer = gridspec.GridSpec(2, 1, height_ratios=[3.2, 1.8], hspace=0.42)
    top = gridspec.GridSpecFromSubplotSpec(
        num_rows,
        num_cols,
        subplot_spec=outer[0],
        wspace=0.12,
        hspace=0.48,
    )

    for idx, row in enumerate(sampled_rows):
        ax = fig.add_subplot(top[idx // num_cols, idx % num_cols])
        caption = f"{labels[idx]}\n{counts[idx]} molecules"
        draw_molecule(ax, smiles=row["smiles"], caption=caption)

    ax_box = fig.add_subplot(outer[1])
    ax_box.boxplot(rt_values, showfliers=False, labels=labels)
    ax_box.tick_params(axis="x", rotation=32)
    ax_box.set_ylabel("RT (minutes)")
    ax_box.set_title("Retention time range per subset")

    fig.subplots_adjust(top=0.95, bottom=0.08)
    return finalize_plot(fig, path)
