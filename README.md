# Graph-RT

Graph-RT is a graph neural network project for small-molecule retention time prediction. This repository contains dataset preprocessing code, PyTorch Geometric training pipelines, saved experiment outputs, and a report with analysis figures.

## Repository Layout

- `Datasets/`: dataset adapters and preprocessing entry points for `SMRT`, `RIKENMONA`, and `MiniDatasets`
- `Datasets/Data/`: raw source files and normalized CSV outputs
- `Networks/`: model definitions, including `DeepGCNRT`
- `Train/`: training scripts, evaluation utilities, and saved results
- `Report/`: figures and the final report
- `settings.py`: random seed and device configuration

## Requirements

The project depends on:

- `numpy`
- `pandas`
- `torch`
- `torchvision`
- `torch-geometric`
- `tqdm`
- `matplotlib`
- `openpyxl`
- `rdkit`

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

`rdkit` and `torch-geometric` often require environment-specific installation steps. If plain `pip install -r requirements.txt` fails, install those two packages using the instructions for your platform first, then rerun the command above.

## Datasets

The repository currently supports three dataset groups:

- `SMRT`: combined from `Datasets/Data/SMRT/source/SMRT_train_set.txt` and `SMRT_test_set.txt`
- `RIKENMONA`: built from `Datasets/Data/RIKEN/source/RIKEN_MONA_ALL.xlsx`
- `MiniDatasets`: multiple small LC/MS datasets stored under `Datasets/Data/MiniDatasets/source/*.xlsx`

All preprocessors normalize each dataset into a standard table with:

- `rt_time`: retention time
- `smiles`: canonicalized molecular SMILES

You can rebuild the normalized files by running any dataset module directly:

```bash
python -m Datasets.SMRT
python -m Datasets.RIKENMONA
python -m Datasets.MiniDatasets
```

## Training

The main training entry point is:

```bash
python Train/DeepGCNRT.py
```

That script currently imports `Datasets.RIKENMONA` by default:

```python
import Datasets.RIKENMONA as dataset
```

To switch datasets, edit the import at the top of `Train/DeepGCNRT.py`.

Important configuration values are also defined directly in `Train/DeepGCNRT.py`, including:

- `TRANSFER_LEARNING`
- `PRETRAIN_WEIGHT_PATH`
- `SUB_DATASET_NAME`
- `NUM_RUNS`
- `NUM_FOLDS`
- `BATCH_SIZE`
- `NUM_EPOCHS`
- `LEARNING_RATE`

Notes:

- `NUM_FOLDS` is only supported when using `Datasets.MiniDatasets`
- when `TRANSFER_LEARNING=True`, `PRETRAIN_WEIGHT_PATH` must point to an existing checkpoint
- device selection is automatic through `settings.py` and uses CUDA when available

## Outputs

Training outputs are written under `Train/Results/DeepGCNRT/`. A typical run directory contains:

- `best_model.pth`: saved checkpoint
- `log.csv`: epoch-level training, validation, and test metrics
- `test_predictions.csv`: target and prediction pairs on the test split

Dataset-level summaries are stored as `summary.csv` in the corresponding result directory.

## Current Repository State

This repository already includes generated artifacts such as:

- preprocessed CSV files
- trained model checkpoints
- experiment logs
- report figures and PDF output

So this is not a source-only snapshot; it is closer to a working research folder with code, data derivatives, and experiment results committed together.
