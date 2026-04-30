import copy

import torch
from tqdm.auto import tqdm

from Train.lib.evaluation import regression_metrics_torch
from settings import DEVICE


def _collect_predictions(model, loader, criterion):
    running_loss = 0.0
    total = 0
    preds = []
    targets = []

    for batch in tqdm(loader, desc="Evaluating...", leave=False):
        batch = batch.to(DEVICE)
        y = batch.y.view(-1).float()
        pred = model(batch).view(-1)
        loss = criterion(pred, y)
        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite loss detected during evaluation. "
                f"loss={loss.item()}, out_finite={torch.isfinite(pred).all().item()}"
            )

        batch_size = y.numel()
        running_loss += loss.item() * batch_size
        total += batch_size
        preds.append(pred.detach().cpu())
        targets.append(y.detach().cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    metrics = regression_metrics_torch(preds, targets)
    metrics["loss"] = running_loss / total
    return metrics, targets, preds


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    total = 0
    preds = []
    targets = []

    for batch in tqdm(loader, desc="Training...", leave=False):
        batch = batch.to(DEVICE)
        y = batch.y.view(-1).float()

        optimizer.zero_grad()
        pred = model(batch).view(-1)
        loss = criterion(pred, y)
        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite loss detected during training. "
                f"loss={loss.item()}, out_finite={torch.isfinite(pred).all().item()}"
            )
        loss.backward()
        optimizer.step()

        batch_size = y.numel()
        running_loss += loss.item() * batch_size
        total += batch_size
        preds.append(pred.detach().cpu())
        targets.append(y.detach().cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    metrics = regression_metrics_torch(preds, targets)
    metrics["loss"] = running_loss / total
    return metrics


def evaluate(model, loader, criterion):
    model.eval()
    with torch.no_grad():
        metrics, targets, preds = _collect_predictions(model, loader, criterion)
    return metrics, targets, preds


def run(
    model,
    train_loader,
    valid_loader,
    test_loader,
    optimizer,
    criterion,
    epochs,
    scheduler=None,
    early_stop=30,
    output=True,
):
    best_valid_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    wait = 0
    logs = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion)
        valid_metrics, _, _ = evaluate(model, valid_loader, criterion)
        test_metrics, test_targets, test_preds = evaluate(model, test_loader, criterion)

        if scheduler is not None:
            scheduler.step()

        if valid_metrics["loss"] < best_valid_loss:
            best_valid_loss = valid_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
        else:
            wait += 1

        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "valid_loss": valid_metrics["loss"],
            "test_loss": test_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "valid_mae": valid_metrics["mae"],
            "test_mae": test_metrics["mae"],
            "train_medae": train_metrics["medae"],
            "valid_medae": valid_metrics["medae"],
            "test_medae": test_metrics["medae"],
            "train_mape": train_metrics["mape"],
            "valid_mape": valid_metrics["mape"],
            "test_mape": test_metrics["mape"],
            "train_mse": train_metrics["mse"],
            "valid_mse": valid_metrics["mse"],
            "test_mse": test_metrics["mse"],
            "train_rmse": train_metrics["rmse"],
            "valid_rmse": valid_metrics["rmse"],
            "test_rmse": test_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "valid_r2": valid_metrics["r2"],
            "test_r2": test_metrics["r2"],
            "best_valid_loss": best_valid_loss,
        }
        logs.append(record)

        if output:
            print(
                f"Epoch {epoch}: "
                f"train_loss={record['train_loss']:.4f}, valid_loss={record['valid_loss']:.4f}, test_loss={record['test_loss']:.4f}, "
                f"train_mae={record['train_mae']:.4f}, valid_mae={record['valid_mae']:.4f}, test_mae={record['test_mae']:.4f}, "
                f"valid_r2={record['valid_r2']:.4f}, test_r2={record['test_r2']:.4f}"
            )

        if wait > early_stop:
            if output:
                print(f"Early stopping at epoch {epoch}, best epoch={best_epoch}.")
            break

    model.load_state_dict(best_state)
    final_valid_metrics, final_valid_targets, final_valid_preds = evaluate(model, valid_loader, criterion)
    final_test_metrics, final_test_targets, final_test_preds = evaluate(model, test_loader, criterion)
    summary = {
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid_loss,
        "valid_metrics": final_valid_metrics,
        "test_metrics": final_test_metrics,
        "valid_targets": final_valid_targets,
        "valid_preds": final_valid_preds,
        "test_targets": final_test_targets,
        "test_preds": final_test_preds,
    }
    return logs, summary
