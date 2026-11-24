"""Class-wise error analysis for saved chest X-ray test predictions.

This is an additive analysis tool. It reads the CSV produced by src.evaluate
and writes summaries without changing model checkpoints or existing results.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from src.config import CLASS_NAMES, OUTPUT_ROOT


def analyze_prediction_rows(rows, class_names=CLASS_NAMES):
    """Return class-wise counts and misclassified rows from prediction records."""
    required = {"filepath", "actual", "predicted"}
    if not rows:
        raise ValueError("The prediction file contains no data rows.")

    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {sorted(missing)}")

    class_stats = {
        name: {"support": 0, "correct": 0, "incorrect": 0}
        for name in class_names
    }
    misclassified = []
    unknown_labels = set()

    for row in rows:
        actual = row["actual"]
        predicted = row["predicted"]

        if actual not in class_stats:
            unknown_labels.add(actual)
            continue
        if predicted not in class_stats:
            unknown_labels.add(predicted)
            continue

        stats = class_stats[actual]
        stats["support"] += 1
        if actual == predicted:
            stats["correct"] += 1
        else:
            stats["incorrect"] += 1
            misclassified.append(row)

    if unknown_labels:
        raise ValueError(
            "Prediction file contains labels not in configured classes: "
            + ", ".join(sorted(unknown_labels))
        )

    for name, stats in class_stats.items():
        support = stats["support"]
        stats["accuracy_recall"] = (
            stats["correct"] / support if support else None
        )
        stats["error_rate"] = (
            stats["incorrect"] / support if support else None
        )

    actual_counts = Counter(row["actual"] for row in rows)
    predicted_counts = Counter(row["predicted"] for row in rows)

    return {
        "total_images": len(rows),
        "correct_predictions": len(rows) - len(misclassified),
        "incorrect_predictions": len(misclassified),
        "overall_accuracy": (len(rows) - len(misclassified)) / len(rows),
        "class_statistics": class_stats,
        "actual_class_counts": dict(actual_counts),
        "predicted_class_counts": dict(predicted_counts),
        "misclassified_rows": misclassified,
    }


def read_prediction_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_outputs(result, output_dir, model_name):
    """Write a JSON summary and two CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        key: value for key, value in result.items()
        if key != "misclassified_rows"
    }
    summary["model"] = model_name

    summary_path = output_dir / f"{model_name}_error_analysis.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    class_path = output_dir / f"{model_name}_class_error_summary.csv"
    with class_path.open("w", newline="", encoding="utf-8") as file:
        fields = [
            "class_name", "support", "correct", "incorrect",
            "accuracy_recall", "error_rate"
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for class_name, stats in result["class_statistics"].items():
            writer.writerow({"class_name": class_name, **stats})

    mistakes_path = output_dir / f"{model_name}_misclassified_predictions.csv"
    with mistakes_path.open("w", newline="", encoding="utf-8") as file:
        rows = result["misclassified_rows"]
        fields = list(rows[0].keys()) if rows else [
            "filepath", "actual", "predicted"
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return summary_path, class_path, mistakes_path


def main():
    parser = argparse.ArgumentParser(
        description="Summarize class-wise errors from src.evaluate predictions."
    )
    parser.add_argument("--model", choices=["resnet50", "densenet121"], required=True)
    parser.add_argument("--predictions", default=None)
    parser.add_argument(
        "--output",
        default=str(OUTPUT_ROOT / "error_analysis")
    )
    args = parser.parse_args()

    prediction_path = Path(
        args.predictions or OUTPUT_ROOT / args.model / "test_predictions.csv"
    )
    if not prediction_path.is_file():
        raise FileNotFoundError(
            f"Prediction CSV not found: {prediction_path.resolve()}. "
            "Run src.evaluate first."
        )

    rows = read_prediction_csv(prediction_path)
    result = analyze_prediction_rows(rows)
    outputs = write_outputs(result, args.output, args.model)

    print(json.dumps({
        "model": args.model,
        "total_images": result["total_images"],
        "correct_predictions": result["correct_predictions"],
        "incorrect_predictions": result["incorrect_predictions"],
        "overall_accuracy": result["overall_accuracy"],
        "class_statistics": result["class_statistics"],
        "files": [str(path.resolve()) for path in outputs],
    }, indent=2))


if __name__ == "__main__":
    main()
