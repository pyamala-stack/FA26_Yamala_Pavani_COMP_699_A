"""Create Grad-CAM heatmaps for selected held-out test images.

Grad-CAM is an explanatory visualization, not proof of clinical reasoning.
Run after training both models. Selects the first N test images by default.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from src.config import SPLIT_ROOT, OUTPUT_ROOT, IMAGE_SIZE, CLASS_NAMES
from src.data.dataset import ChestXrayDataset
from src.models import build_model


def target_layer_for(model, model_name):
    if model_name == "resnet50":
        return model.layer4[-1]
    return model.features.denseblock4


def make_cam(model, target_layer, tensor, class_index):
    captured = {}

    def forward_hook(module, inputs, output):
        output.retain_grad()
        captured["activation"] = output

    handle = target_layer.register_forward_hook(forward_hook)
    model.zero_grad(set_to_none=True)
    logits = model(tensor)
    score = logits[:, class_index].sum()
    score.backward()
    activation = captured["activation"]
    gradients = activation.grad
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activation).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(
        cam, size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear", align_corners=False
    )
    cam = cam[0, 0].detach().cpu().numpy()
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    handle.remove()
    return cam, logits.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet50", "densenet121"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=str(SPLIT_ROOT))
    parser.add_argument("--count", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ChestXrayDataset(args.data_root, "test", train=False)
    model = build_model(args.model, pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    target_layer = target_layer_for(model, args.model)
    output_folder = OUTPUT_ROOT / args.model / "gradcam"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Deterministic sample: first N images from the sorted test dataset.
    for index in range(min(args.count, len(dataset))):
        tensor, true_label, image_path = dataset[index]
        input_tensor = tensor.unsqueeze(0).to(device)
        with torch.enable_grad():
            with torch.no_grad():
                initial_logits = model(input_tensor)
                predicted_class = int(initial_logits.argmax(dim=1).item())
            cam, logits = make_cam(
                model, target_layer, input_tensor, predicted_class
            )

        original = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        figure, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(original)
        axes[0].set_title(f"Original | Actual: {CLASS_NAMES[true_label]}")
        axes[0].axis("off")
        axes[1].imshow(original)
        axes[1].imshow(cam, cmap="jet", alpha=0.45)
        axes[1].set_title(
            f"Grad-CAM | Predicted: {CLASS_NAMES[predicted_class]} "
            f"({probabilities[predicted_class]:.2f})"
        )
        axes[1].axis("off")
        figure.tight_layout()
        figure.savefig(output_folder / f"gradcam_{index + 1:02d}.png", dpi=150)
        plt.close(figure)

    print(f"Saved Grad-CAM images to: {output_folder}")


if __name__ == "__main__":
    main()
