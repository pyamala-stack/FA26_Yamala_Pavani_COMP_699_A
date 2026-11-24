
"""Remove exact duplicates and create new reproducible splits
from an existing train/validation/test dataset.
"""

from pathlib import Path
import argparse
import hashlib
import random
import shutil

from src.config import CLASS_NAMES, SEED

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


def file_hash(path):
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="data/lung_xray_split"
    )
    parser.add_argument(
        "--destination",
        default="data/lung_xray_split_clean"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    source = Path(args.source)
    destination = Path(args.destination)

    if not source.is_dir():
        raise FileNotFoundError(source.resolve())

    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"Destination is not empty: {destination.resolve()}"
        )

    rng = random.Random(args.seed)

    # Collect all images from the existing three splits.
    unique = {name: {} for name in CLASS_NAMES}
    duplicate_count = 0

    for split in ("train", "val", "test"):
        for class_name in CLASS_NAMES:
            folder = source / split / class_name

            if not folder.is_dir():
                raise FileNotFoundError(
                    f"Missing folder: {folder.resolve()}"
                )

            for path in sorted(folder.rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                digest = file_hash(path)

                if digest in unique[class_name]:
                    duplicate_count += 1
                    continue

                unique[class_name][digest] = path

    # Check whether identical images have conflicting class labels.
    hashes_by_class = {}
    for class_name, images in unique.items():
        for digest in images:
            if digest in hashes_by_class:
                raise ValueError(
                    f"Identical image found in different classes: "
                    f"{hashes_by_class[digest]} and {class_name}. "
                    "Resolve the conflicting labels first."
                )
            hashes_by_class[digest] = class_name

    destination.mkdir(parents=True, exist_ok=True)

    print(f"Exact duplicate files removed: {duplicate_count}")

    for class_name in CLASS_NAMES:
        images = list(unique[class_name].values())

        if len(images) < 10:
            raise RuntimeError(
                f"Not enough unique images for {class_name}"
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

            for index, image in enumerate(selected):
                output = target / (
                    f"{index:06d}_{image.name}"
                )
                shutil.copy2(image, output)

            print(
                f"{split:5s} | {class_name:16s} | "
                f"{len(selected)}"
            )

    print("\nClean dataset created successfully.")
    print(f"Unique images retained: {len(hashes_by_class)}")
    print(f"Destination: {destination.resolve()}")


if __name__ == "__main__":
    main()
