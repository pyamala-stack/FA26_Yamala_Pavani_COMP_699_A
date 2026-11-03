"""Train one selected model and select the checkpoint by validation loss."""
import argparse
import json
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import (
    SPLIT_ROOT, OUTPUT_ROOT, BATCH_SIZE, EPOCHS,
    LEARNING_RATE, SEED, NUM_WORKERS
)
from src.data.dataset import ChestXrayDataset
from src.models import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet50", "densenet121"], default="resnet50")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--data-root", default=str(SPLIT_ROOT))
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = ChestXrayDataset(args.data_root, "train", train=True)
    val_data = ChestXrayDataset(args.data_root, "val", train=False)

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = build_model(args.model, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    output_folder = OUTPUT_ROOT / args.model
    output_folder.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")
    history = []

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * labels.size(0)

        model.eval()
        validation_loss_sum = 0.0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                validation_loss_sum += loss.item() * labels.size(0)

        record = {
            "epoch": epoch + 1,
            "training_loss": train_loss_sum / len(train_data),
            "validation_loss": validation_loss_sum / len(val_data)
        }
        history.append(record)
        print(record)

        if record["validation_loss"] < best_validation_loss:
            best_validation_loss = record["validation_loss"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": args.model,
                "seed": SEED,
                "class_names": ["Lung_Opacity", "Normal", "Viral Pneumonia"]
            }, output_folder / "best_model.pt")

    (output_folder / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"Training completed on: {device}")
    print(f"Best checkpoint: {output_folder / 'best_model.pt'}")


if __name__ == "__main__":
    main()
