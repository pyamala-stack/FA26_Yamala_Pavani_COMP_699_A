"""Inventory the expected chest X-ray dataset folders and create a CSV manifest."""
from pathlib import Path
import argparse
import csv

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")
CLASSES = ("NORMAL", "PNEUMONIA")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/chest_xray")
    parser.add_argument("--output-dir", default="outputs/week_1")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    summary_lines = [f"Dataset root: {data_root.resolve()}", ""]
    total = 0

    for split in SPLITS:
        for label in CLASSES:
            folder = data_root / split / label
            if not folder.exists():
                summary_lines.append(f"MISSING: {folder.as_posix()}")
                continue
            paths = sorted(p for p in folder.rglob("*")
                           if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
            summary_lines.append(f"{split},{label},{len(paths)}")
            total += len(paths)
            for path in paths:
                manifest.append({
                    "split": split,
                    "label": label,
                    "filepath": path.as_posix(),
                    "extension": path.suffix.lower(),
                    "file_size_bytes": path.stat().st_size
                })

    summary_lines.insert(1, f"Total supported images found: {total}")
    with (output_dir / "image_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "split", "label", "filepath", "extension", "file_size_bytes"
        ])
        writer.writeheader()
        writer.writerows(manifest)
    (output_dir / "dataset_summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    print("\n".join(summary_lines))
    print(f"\nWrote: {output_dir / 'image_manifest.csv'}")
    print(f"Wrote: {output_dir / 'dataset_summary.txt'}")
    if total == 0:
        print("No images found. Check the dataset path and folder layout.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
