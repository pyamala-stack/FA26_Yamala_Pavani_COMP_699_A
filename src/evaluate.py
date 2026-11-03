"""Evaluate a saved model on the held-out test split."""
import argparse
import csv
import json
import torch
from torch.utils.data import DataLoader

from src.config import SPLIT_ROOT, OUTPUT_ROOT, BATCH_SIZE, NUM_WORKERS, CLASS_NAMES
from src.data.dataset import ChestXrayDataset
from src.models import build_model
from src.metrics import calculate_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet50", "densenet121"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=str(SPLIT_ROOT))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = ChestXrayDataset(args.data_root, "test", train=False)
    test_loader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = build_model(args.model, pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true, y_pred, probabilities, prediction_rows = [], [], [], []
    with torch.no_grad():
        for images, labels, paths in test_loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            predictions = probs.argmax(axis=1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(predictions.tolist())
            probabilities.extend(probs.tolist())

            for path, actual, predicted, prob in zip(
                paths, labels.numpy(), predictions, probs
            ):
                prediction_rows.append([
                    path, CLASS_NAMES[int(actual)], CLASS_NAMES[int(predicted)],
                    *[float(value) for value in prob]
                ])

    results = calculate_metrics(
        y_true, y_pred, probabilities, len(CLASS_NAMES)
    )
    results["class_names"] = CLASS_NAMES
    results["test_image_count"] = len(test_data)

    output_folder = OUTPUT_ROOT / args.model
    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / "test_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    with (output_folder / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "filepath", "actual", "predicted",
            *[f"probability_{name}" for name in CLASS_NAMES]
        ])
        writer.writerows(prediction_rows)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
