
"""Compare ResNet-50 and DenseNet-121 test predictions."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.config import OUTPUT_ROOT, CLASS_NAMES


def read_predictions(path):
    with open(path, encoding="utf-8") as file:
        return {
            row["filepath"]: row
            for row in csv.DictReader(file)
        }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--resnet",
        default=str(OUTPUT_ROOT / "resnet50" / "test_predictions.csv")
    )
    parser.add_argument(
        "--densenet",
        default=str(OUTPUT_ROOT / "densenet121" / "test_predictions.csv")
    )
    parser.add_argument(
        "--output",
        default="outputs/model_comparison"
    )

    args = parser.parse_args()

    resnet = read_predictions(args.resnet)
    densenet = read_predictions(args.densenet)

    common = sorted(set(resnet) & set(densenet))

    if not common:
        raise RuntimeError("No matching test image paths found.")

    rows = []
    disagreements = []

    for path in common:
        r = resnet[path]
        d = densenet[path]

        if r["actual"] != d["actual"]:
            raise ValueError("Ground-truth labels do not match.")

        row = {
            "filepath": path,
            "actual": r["actual"],
            "resnet_prediction": r["predicted"],
            "densenet_prediction": d["predicted"],
            "agreement": r["predicted"] == d["predicted"]
        }

        rows.append(row)

        if not row["agreement"]:
            disagreements.append(row)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # Save individual prediction comparison.
    with (output / "prediction_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    # Save disagreements separately.
    with (output / "model_disagreements.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(disagreements[0].keys())
            if disagreements else list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(disagreements)

    summary = {
        "matched_test_images": len(common),
        "agreement_count": len(common) - len(disagreements),
        "agreement_rate": (
            (len(common) - len(disagreements)) / len(common)
        ),
        "disagreement_count": len(disagreements),
        "disagreement_rate": len(disagreements) / len(common),
        "note": (
            "Agreement is descriptive and does not establish "
            "which model is correct."
        )
    }

    (output / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    # Compare predicted class distributions.
    resnet_counts = [
        sum(row["resnet_prediction"] == name for row in rows)
        for name in CLASS_NAMES
    ]

    densenet_counts = [
        sum(row["densenet_prediction"] == name for row in rows)
        for name in CLASS_NAMES
    ]

    positions = list(range(len(CLASS_NAMES)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        [x - width / 2 for x in positions],
        resnet_counts,
        width,
        label="ResNet-50"
    )

    ax.bar(
        [x + width / 2 for x in positions],
        densenet_counts,
        width,
        label="DenseNet-121"
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(CLASS_NAMES, rotation=12)
    ax.set_ylabel("Number of predicted test images")
    ax.set_title("Predicted Class Distribution Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output / "prediction_distribution.png",
        dpi=160,
        bbox_inches="tight"
    )
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"\nComparison files saved in: {output.resolve()}")


if __name__ == "__main__":
    main()
