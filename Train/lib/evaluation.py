import numpy as np
import torch


def _flatten_torch(pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return pred.reshape(-1).float(), y.reshape(-1).float()


def _flatten_np(pred: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    return pred, y


def accuracy_evaluator(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    Return number of correct predictions in a batch.
    `Train/lib/train.py` will divide by total sample count to get accuracy.
    """
    pred = torch.argmax(logits, dim=1)
    correct = (pred == y).sum()
    return float(correct.item())


def accuracy_evaluator_np(pred: np.ndarray, y: np.ndarray) -> float:
    pred = np.asarray(pred)
    y = np.asarray(y)
    if pred.ndim > 1:
        pred = pred.argmax(axis=1)
    return float((pred == y).mean())


def mae_evaluator(pred: torch.Tensor, y: torch.Tensor) -> float:
    pred, y = _flatten_torch(pred, y)
    return float(torch.abs(pred - y).mean().item())


def medae_evaluator(pred: torch.Tensor, y: torch.Tensor) -> float:
    pred, y = _flatten_torch(pred, y)
    return float(torch.quantile(torch.abs(pred - y), 0.5).item())


def mape_evaluator(pred: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    pred, y = _flatten_torch(pred, y)
    return float((torch.abs(pred - y) / y.abs().clamp_min(eps)).mean().item())


def mse_evaluator(pred: torch.Tensor, y: torch.Tensor) -> float:
    pred, y = _flatten_torch(pred, y)
    return float(torch.mean((pred - y) ** 2).item())


def r2_evaluator(pred: torch.Tensor, y: torch.Tensor) -> float:
    pred, y = _flatten_torch(pred, y)
    denominator = torch.sum((y - y.mean()) ** 2)
    if denominator <= 0:
        return 0.0
    numerator = torch.sum((y - pred) ** 2)
    return float((1.0 - numerator / denominator).item())


def regression_metrics_torch(pred: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> dict[str, float]:
    pred, y = _flatten_torch(pred, y)
    abs_error = torch.abs(pred - y)
    sq_error = (pred - y) ** 2
    denominator = torch.sum((y - y.mean()) ** 2)
    r2 = 0.0 if denominator <= 0 else float((1.0 - torch.sum(sq_error) / denominator).item())

    mse = float(sq_error.mean().item())
    return {
        "mae": float(abs_error.mean().item()),
        "medae": float(torch.quantile(abs_error, 0.5).item()),
        "mape": float((abs_error / y.abs().clamp_min(eps)).mean().item()),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": r2,
    }


def regression_metrics_np(pred: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> dict[str, float]:
    pred, y = _flatten_np(pred, y)
    abs_error = np.abs(pred - y)
    sq_error = (pred - y) ** 2
    denominator = np.sum((y - y.mean()) ** 2)
    r2 = 0.0 if denominator <= 0 else float(1.0 - np.sum(sq_error) / denominator)

    mse = float(np.mean(sq_error))
    return {
        "mae": float(np.mean(abs_error)),
        "medae": float(np.median(abs_error)),
        "mape": float(np.mean(abs_error / np.clip(np.abs(y), eps, None))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": r2,
    }


if __name__ == "__main__":
    y_true = torch.tensor([1, 2, 0, 3])
    y_pred_logits = torch.tensor(
        [
            [0.1, 0.8, 0.1, 0.0],  # 1
            [0.2, 0.1, 0.6, 0.1],  # 2
            [0.7, 0.1, 0.1, 0.1],  # 0
            [0.1, 0.3, 0.5, 0.1],  # 2 (wrong)
        ]
    )
    correct = accuracy_evaluator(y_pred_logits, y_true)
    acc = correct / y_true.size(0)
    print(f"Correct: {correct}, Accuracy: {acc:.4f}")

    y_reg = torch.tensor([10.0, 20.0, 30.0, 40.0])
    y_reg_pred = torch.tensor([12.0, 19.0, 28.0, 42.0])
    print(regression_metrics_torch(y_reg_pred, y_reg))
