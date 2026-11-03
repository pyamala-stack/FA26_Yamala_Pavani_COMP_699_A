"""PyTorch dataset for the three chest X-ray categories."""
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import CLASS_NAMES, IMAGE_SIZE

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class ChestXrayDataset(Dataset):
    def __init__(self, root, split, train=False):
        self.samples = []
        root = Path(root)

        for label, class_name in enumerate(CLASS_NAMES):
            class_folder = root / split / class_name
            if not class_folder.exists():
                raise FileNotFoundError(
                    f"Required class folder is missing: {class_folder.resolve()}"
                )

            image_paths = sorted(
                p for p in class_folder.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            self.samples.extend((path, label) for path in image_paths)

        if not self.samples:
            raise RuntimeError(f"No supported images found in {root / split}")

        operations = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
        if train:
            operations.append(transforms.RandomHorizontalFlip(p=0.5))
        operations.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.transform = transforms.Compose(operations)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        return self.transform(image), label, str(image_path)
