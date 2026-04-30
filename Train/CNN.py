import os
from datetime import datetime
from uuid import uuid4

import torch
import torch.optim as optim

import Datasets.MNIST as mnist
from Networks.CNN import Network
from Train.lib.evaluation import accuracy_evaluator
from Train.lib.train import run
from settings import DEVICE

BATCH_SIZE = 64
NUM_EPOCHS = 10
NUM_RUNS = 10

INPUT_SHAPE = (1, 28, 28)
NUM_CLASSES = 10

CONV1 = (32, 3, 1)
CONV2 = (64, 3, 1)
HIDDEN_SIZE = 128

RESULT_PATH = "Train/Results/CNN"


if __name__ == "__main__":
    os.makedirs(RESULT_PATH, exist_ok=True)
    train_loader, test_loader = mnist.load_loader(batch_size=BATCH_SIZE)

    for _ in range(NUM_RUNS):
        run_id = str(uuid4())[:8]
        print(f"=== CNN Run {run_id} ===")

        model = Network(
            INPUT_SHAPE,
            NUM_CLASSES,
            conv1=CONV1,
            conv2=CONV2,
            hidden_size=HIDDEN_SIZE,
        ).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        log = run(
            model,
            train_loader,
            test_loader,
            optimizer,
            criterion,
            accuracy_evaluator,
            NUM_EPOCHS,
        )

        result_file = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}_{run_id}.csv"
        with open(os.path.join(RESULT_PATH, result_file), "w") as f:
            f.write("epoch,train_loss,train_eval,test_loss,test_eval\n")
            for epoch_idx, tr_loss, tr_eval, te_loss, te_eval in log:
                f.write(f"{epoch_idx},{tr_loss},{tr_eval},{te_loss},{te_eval}\n")
