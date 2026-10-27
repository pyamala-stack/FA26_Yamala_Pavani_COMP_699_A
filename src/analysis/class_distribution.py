"""Summarize class counts from the image manifest and save a bar chart."""
from pathlib import Path
import argparse
import csv
from collections import Counter
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/week_1/image_manifest.csv")
    parser.add_argument("--output-dir", default="outputs/week_1")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} not found. Run src/data/inspect_dataset.py first."
        )

    counts = Counter()
    with manifest.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            counts[(row["split"], row["label"])] += 1

    splits = ["train", "val", "test"]
    labels = ["NORMAL", "PNEUMONIA"]
    lines = ["split,label,count"]
    for split in splits:
        for label in labels:
            lines.append(f"{split},{label},{counts[(split, label)]}")
    (output_dir / "class_distribution.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    x_labels = [f"{split}\\n{label}" for split in splits for label in labels]
    values = [counts[(split, label)] for split in splits for label in labels]
    plt.figure(figsize=(9, 5))
    plt.bar(x_labels, values)
    plt.title("Chest X-ray image counts by split and class")
    plt.xlabel("Dataset split and class")
    plt.ylabel("Number of images")
    plt.tight_layout()
    plot_path = output_dir / "class_distribution.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()

    print(f"Counts: {output_dir / 'class_distribution.csv'}")
    print(f"Chart: {plot_path}")
    for split in splits:
        print(f"{split}: NORMAL={counts[(split, 'NORMAL')]}, "
              f"PNEUMONIA={counts[(split, 'PNEUMONIA')]}")


if __name__ == "__main__":
    main()
