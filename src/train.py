
"""CPU-efficient training for ResNet-50 and DenseNet-121."""

import argparse
import json
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import (
    SPLIT_ROOT,
    OUTPUT_ROOT,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    SEED,
    NUM_WORKERS,
    CLASS_NAMES,
    IMAGE_SIZE,
)
from src.data.dataset import ChestXrayDataset
from src.models import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def freeze_backbone(model, model_name):
    """Freeze pretrained feature layers and train the classifier."""

    for parameter in model.parameters():
        parameter.requires_grad = False

    if model_name == "resnet50":
        for parameter in model.fc.parameters():
            parameter.requires_grad = True

    elif model_name == "densenet121":
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered.")

            if training:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    if total == 0:
        raise RuntimeError("No images were processed.")

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["resnet50", "densenet121"],
        default="resnet50",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--data-root", default=str(SPLIT_ROOT))
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("Epochs, batch size and learning rate must be positive.")

    torch.set_num_threads(2)
    set_seed(args.seed)

    device = torch.device("cpu")

    train_data = ChestXrayDataset(
        args.data_root, "train", train=True
    )
    val_data = ChestXrayDataset(
        args.data_root, "val", train=False
    )

    generator = torch.Generator().manual_seed(args.seed)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(
        args.model,
        pretrained=not args.no_pretrained,
    )

    freeze_backbone(model, args.model)
    model.to(device)

    trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )

    output_folder = OUTPUT_ROOT / args.model
    output_folder.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_folder / "best_model.pt"
    history_path = output_folder / "training_history.json"
    config_path = output_folder / "experiment_config.json"

    config = {
        "model": args.model,
        "dataset": str(args.data_root),
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": "Adam",
        "loss_function": "CrossEntropyLoss",
        "pretrained_weights": not args.no_pretrained,
        "training_method": "Frozen pretrained backbone; train classifier only",
        "trainable_parameters": trainable_parameters,
        "seed": args.seed,
        "device": str(device),
        "training_images": len(train_data),
        "validation_images": len(val_data),
    }

    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    history = []
    best_validation_loss = float("inf")
    start_time = time.perf_counter()

    print("\n====================================")
    print("MODEL:", args.model)
    print("DEVICE:", device)
    print("TRAINING IMAGES:", len(train_data))
    print("VALIDATION IMAGES:", len(val_data))
    print("TRAINABLE PARAMETERS:", trainable_parameters)
    print("EPOCHS:", args.epochs)
    print("====================================\n")

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()

        train_results = run_epoch(
            model, train_loader, criterion, device, optimizer
        )

        val_results = run_epoch(
            model, val_loader, criterion, device
        )

        epoch_time = time.perf_counter() - epoch_start

        record = {
            "epoch": epoch + 1,
            "training_loss": train_results["loss"],
            "training_accuracy": train_results["accuracy"],
            "validation_loss": val_results["loss"],
            "validation_accuracy": val_results["accuracy"],
            "epoch_time_seconds": epoch_time,
        }

        history.append(record)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {record['training_loss']:.4f} | "
            f"Train Accuracy: {record['training_accuracy']:.4f} | "
            f"Val Loss: {record['validation_loss']:.4f} | "
            f"Val Accuracy: {record['validation_accuracy']:.4f} | "
            f"Time: {epoch_time:.2f}s",
            flush=True,
        )

        if record["validation_loss"] < best_validation_loss:
            best_validation_loss = record["validation_loss"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model,
                    "epoch": epoch + 1,
                    "best_validation_loss": best_validation_loss,
                    "seed": args.seed,
                    "class_names": CLASS_NAMES,
                    "experiment_config": config,
                },
                checkpoint_path,
            )

            print("Best checkpoint saved.", flush=True)

        history_path.write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

    total_time = time.perf_counter() - start_time

    summary = {
        "model": args.model,
        "completed_epochs": len(history),
        "best_validation_loss": best_validation_loss,
        "total_training_time_seconds": total_time,
        "checkpoint": str(checkpoint_path.resolve()),
    }

    summary_path = output_folder / "experiment_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nTRAINING COMPLETED", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
