"""Try opening every image listed by the inventory script and report unreadable files."""
from pathlib import Path
import argparse
import csv
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/week_1/image_manifest.csv")
    parser.add_argument("--output-dir", default="outputs/week_1")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Run src/data/inspect_dataset.py first."
        )

    checked = 0
    failures = []
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            checked += 1
            image_path = Path(row["filepath"])
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                failures.append({
                    "filepath": str(image_path),
                    "split": row["split"],
                    "label": row["label"],
                    "error": f"{type(exc).__name__}: {exc}"
                })

    report = output_dir / "image_validation_report.txt"
    report.write_text(
        f"Images checked: {checked}\n"
        f"Unreadable/invalid images: {len(failures)}\n"
        f"Valid images (count minus failures): {checked - len(failures)}\n",
        encoding="utf-8"
    )
    with (output_dir / "invalid_images.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "split", "label", "error"])
        writer.writeheader()
        writer.writerows(failures)

    print(report.read_text(encoding="utf-8"))
    print(f"Invalid-file details: {output_dir / 'invalid_images.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
