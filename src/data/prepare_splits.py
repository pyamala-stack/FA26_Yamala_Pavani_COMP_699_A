"""Create reproducible image-level train/validation/test splits.

Expected source layout:
data/lung_xray/Lung X-Ray Image/
    Lung_Opacity/
    Normal/
    Viral Pneumonia/

This script copies source images and does not modify the originals.
"""
from pathlib import Path
import argparse
import random
import shutil

from src.config import CLASS_NAMES, SEED

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="data/lung_xray/Lung X-Ray Image",
        help="Folder containing the three class folders."
    )
    parser.add_argument(
        "--destination",
        default="data/lung_xray_split",
        help="Destination for generated train/val/test folders."
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    source = Path(args.source)
    destination = Path(args.destination)

    if not source.exists():
        raise FileNotFoundError(
            f"Source folder not found: {source.resolve()}. "
            "Check the extracted Kaggle folder location."
        )

    rng = random.Random(args.seed)
    split_counts = {}

    for class_name in CLASS_NAMES:
        class_folder = source / class_name
        if not class_folder.exists():
            raise FileNotFoundError(
                f"Missing class folder: {class_folder.resolve()}. "
                "The source folder must directly contain all three class folders."
            )

        images = sorted(
            p for p in class_folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(images) < 10:
            raise RuntimeError(
                f"Only {len(images)} images found in {class_folder}; "
                "at least 10 are required for this split procedure."
            )

        rng.shuffle(images)
        n = len(images)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        groups = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        for split, selected in groups.items():
            target = destination / split / class_name
            target.mkdir(parents=True, exist_ok=True)
            for index, image_path in enumerate(selected):
                # Prefix prevents filename collisions across nested folders.
                output_path = target / f"{index:06d}_{image_path.name}"
                shutil.copy2(image_path, output_path)
            split_counts[(split, class_name)] = len(selected)

    print("Dataset split completed.")
    print(f"Source: {source.resolve()}")
    print(f"Destination: {destination.resolve()}")
    for split in ("train", "val", "test"):
        for class_name in CLASS_NAMES:
            print(f"{split:5s} | {class_name:16s} | {split_counts[(split, class_name)]}")


if __name__ == "__main__":
    main()
