import torch
from tqdm.auto import tqdm

from settings import *


def train_one_epoch(model, loader, optimizer, criterion, evaluator):
    model.train()
    running_loss = 0
    evaluation = 0
    total = 0

    for X, y in tqdm(loader, desc="Training...", leave=False):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Forward pass and backward pass
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite loss detected during training. "
                f"loss={loss.item()}, out_finite={torch.isfinite(out).all().item()}"
            )
        loss.backward()
        optimizer.step()

        # Loss and evaluation
        running_loss += loss.item() * y.size(0)
        evaluation += evaluator(out, y)
        total += y.size(0)

    # Train loss and train evaluation
    return running_loss / total, evaluation / total


def evaluate(model, loader, criterion, evaluator):
    model.eval()
    running_loss = 0
    evaluation = 0
    total = 0

    with torch.no_grad():
        for X, y in tqdm(loader, desc="Evaluating...", leave=False):
            X, y = X.to(DEVICE), y.to(DEVICE)

            # Forward pass
            out = model(X)
            loss = criterion(out, y)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite loss detected during evaluation. "
                    f"loss={loss.item()}, out_finite={torch.isfinite(out).all().item()}"
                )

            # Loss and evaluation
            running_loss += loss.item() * y.size(0)
            with torch.no_grad():
                evaluation += evaluator(out, y)
            total += y.size(0)

    # Test loss and test evaluation
    return running_loss / total, evaluation / total


def run(
    model,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    evaluator,
    epochs,
    output=True,
):
    log = []
    for epoch in (
        range(1, epochs + 1)
        if output
        else tqdm(range(1, epochs + 1), desc="Epoch: ", leave=False)
    ):
        # Training
        tr_loss, tr_eval = train_one_epoch(
            model, train_loader, optimizer, criterion, evaluator
        )

        # Evaluation
        te_loss, te_eval = evaluate(model, test_loader, criterion, evaluator)

        # Logging
        if output:
            print(
                f"Epoch {epoch}: train_loss={tr_loss:.4f}, train_acc={tr_eval:.4f}, "
                f"test_loss={te_loss:.4f}, test_acc={te_eval:.4f}"
            )
        log.append((epoch, tr_loss, tr_eval, te_loss, te_eval))

    return log
