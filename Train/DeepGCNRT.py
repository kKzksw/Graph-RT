import os
from datetime import datetime
from uuid import uuid4

import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import Datasets.RIKENMONA as dataset
from Networks.DeepGCNRT import Network
from Train.lib.train_graph import run
from settings import DEVICE, RANDOM_SEED, set_random_seed

TRANSFER_LEARNING = True

SUB_DATASET_NAME = None
PRETRAIN_WEIGHT_PATH = (
    "Train/Results/DeepGCNRT/SMRT/20260324_103037_097_e9b75031/best_model.pth"
)

NUM_RUNS = 1
BATCH_SIZE = 64
NUM_EPOCHS = 200
EARLY_STOP = 30
LEARNING_RATE = 0.001
SCHEDULER_T0 = 200

VALID_RATIO = 0.1
TEST_RATIO = 0.1
SHUFFLE_DATASET = True
NUM_FOLDS = None

NUM_LAYERS = 16
HIDDEN_DIM = 200
READOUT_STEPS = 2
DROPOUT = 0.1
NORM = "none"
UPDATE_FUNC = "no_relu"

# RESULT_PATH = f"Train/Results/DeepGCNRT/RIKENMONA/{SUB_DATASET_NAME}"
RESULT_PATH = f"Train/Results/DeepGCNRT/RIKENMONA"

if __name__ == "__main__":
    os.makedirs(RESULT_PATH, exist_ok=True)
    summary_rows = []

    if NUM_FOLDS is None:
        fold_indices = [None]
    else:
        if NUM_FOLDS < 2:
            raise ValueError(
                "`NUM_FOLDS` must be at least 2 when fold mode is enabled."
            )
        fold_indices = list(range(NUM_FOLDS))

    for run_idx in range(NUM_RUNS):
        run_seed = RANDOM_SEED + run_idx
        set_random_seed(run_seed)

        for fold_index in fold_indices:
            run_id = str(uuid4())[:8]
            stage_name = "Transfer" if TRANSFER_LEARNING else "Pretrain"
            dataset_name = dataset.__name__.split(".")[-1]
            fold_text = (
                f" | fold={fold_index + 1}/{NUM_FOLDS}"
                if fold_index is not None
                else ""
            )

            print(
                f"=== DeepGCNRT {stage_name} Run {run_id} | "
                f"seed={run_seed} | dataset={dataset_name} | "
                f"sub_dataset={SUB_DATASET_NAME}{fold_text} ==="
            )

            loader_kwargs = {
                "sub_dataset_name": SUB_DATASET_NAME,
                "shuffle": SHUFFLE_DATASET,
                "valid_ratio": VALID_RATIO,
                "test_ratio": TEST_RATIO,
                "random_seed": run_seed,
                "batch_size": BATCH_SIZE,
            }
            if NUM_FOLDS is not None:
                if dataset_name != "MiniDatasets":
                    raise ValueError(
                        "`NUM_FOLDS` is only supported for `Datasets.MiniDatasets`. "
                        f"Current dataset is `{dataset_name}`."
                    )
                loader_kwargs["num_folds"] = NUM_FOLDS
                loader_kwargs["fold_index"] = fold_index

            train_loader, valid_loader, test_loader = dataset.load_loader(
                **loader_kwargs,
            )
            node_dim, edge_dim = dataset.get_graph_dimensions(
                sub_dataset_name=SUB_DATASET_NAME,
            )

            model = Network(
                node_dim=node_dim,
                edge_dim=edge_dim,
                hidden_dim=HIDDEN_DIM,
                num_layers=NUM_LAYERS,
                readout_steps=READOUT_STEPS,
                dropout=DROPOUT,
                norm=NORM,
                update_func=UPDATE_FUNC,
            ).to(DEVICE)

            if TRANSFER_LEARNING:
                if PRETRAIN_WEIGHT_PATH == "":
                    raise ValueError(
                        "`PRETRAIN_WEIGHT_PATH` must be set when `TRANSFER_LEARNING=True`."
                    )
                model.load(PRETRAIN_WEIGHT_PATH, map_location=DEVICE)

            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=SCHEDULER_T0)
            criterion = torch.nn.SmoothL1Loss()

            logs, summary = run(
                model,
                train_loader,
                valid_loader,
                test_loader,
                optimizer,
                criterion,
                NUM_EPOCHS,
                scheduler=scheduler,
                early_stop=EARLY_STOP,
            )

            fold_suffix = (
                f"_fold{fold_index + 1:02d}of{NUM_FOLDS}"
                if fold_index is not None
                else ""
            )
            result_dir = os.path.join(
                RESULT_PATH,
                f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}{fold_suffix}_{run_id}",
            )
            os.makedirs(result_dir, exist_ok=True)

            model.save(os.path.join(result_dir, "best_model.pth"))

            with open(os.path.join(result_dir, "log.csv"), "w") as f:
                f.write(
                    "epoch,lr,train_loss,valid_loss,test_loss,"
                    "train_mae,valid_mae,test_mae,"
                    "train_medae,valid_medae,test_medae,"
                    "train_mape,valid_mape,test_mape,"
                    "train_mse,valid_mse,test_mse,"
                    "train_rmse,valid_rmse,test_rmse,"
                    "train_r2,valid_r2,test_r2,best_valid_loss\n"
                )
                for row in logs:
                    f.write(
                        f"{row['epoch']},{row['lr']},{row['train_loss']},{row['valid_loss']},{row['test_loss']},"
                        f"{row['train_mae']},{row['valid_mae']},{row['test_mae']},"
                        f"{row['train_medae']},{row['valid_medae']},{row['test_medae']},"
                        f"{row['train_mape']},{row['valid_mape']},{row['test_mape']},"
                        f"{row['train_mse']},{row['valid_mse']},{row['test_mse']},"
                        f"{row['train_rmse']},{row['valid_rmse']},{row['test_rmse']},"
                        f"{row['train_r2']},{row['valid_r2']},{row['test_r2']},{row['best_valid_loss']}\n"
                    )

            pd.DataFrame(
                {
                    "target": summary["test_targets"].numpy(),
                    "prediction": summary["test_preds"].numpy(),
                }
            ).to_csv(os.path.join(result_dir, "test_predictions.csv"), index=False)

            test_metrics = summary["test_metrics"]
            valid_metrics = summary["valid_metrics"]
            summary_rows.append(
                {
                    "run_idx": run_idx,
                    "seed": run_seed,
                    "fold": "" if fold_index is None else fold_index + 1,
                    "num_folds": "" if fold_index is None else NUM_FOLDS,
                    "best_epoch": summary["best_epoch"],
                    "best_valid_loss": summary["best_valid_loss"],
                    "valid_mae": valid_metrics["mae"],
                    "valid_medae": valid_metrics["medae"],
                    "valid_mape": valid_metrics["mape"],
                    "valid_mse": valid_metrics["mse"],
                    "valid_rmse": valid_metrics["rmse"],
                    "valid_r2": valid_metrics["r2"],
                    "test_mae": test_metrics["mae"],
                    "test_medae": test_metrics["medae"],
                    "test_mape": test_metrics["mape"],
                    "test_mse": test_metrics["mse"],
                    "test_rmse": test_metrics["rmse"],
                    "test_r2": test_metrics["r2"],
                    "result_dir": result_dir,
                }
            )

            print(f"Best epoch: {summary['best_epoch']}")
            print(f"Validation metrics: {summary['valid_metrics']}")
            print(f"Test metrics: {summary['test_metrics']}")
            print(f"Artifacts saved to: {result_dir}")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(RESULT_PATH, "summary.csv"),
            index=False,
        )
