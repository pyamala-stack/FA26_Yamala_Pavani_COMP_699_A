
"""
Grad-CAM Explainability for Chest X-ray Classification.

Supports:
- ResNet-50
- DenseNet-121
- Original image and Grad-CAM overlay
- Prediction confidence
- Correct and incorrect prediction identification
- Optional model disagreement image selection
- CSV records of generated explanations

Grad-CAM is an explanatory visualization, not proof of
clinical reasoning or medical correctness.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image

from src.config import (
    SPLIT_ROOT,
    OUTPUT_ROOT,
    IMAGE_SIZE,
    CLASS_NAMES,
)

from src.data.dataset import ChestXrayDataset
from src.models import build_model


def get_target_layer(model, model_name):
    """Return the final convolutional feature layer."""

    if model_name == "resnet50":
        return model.layer4[-1]

    if model_name == "densenet121":
        return model.features.denseblock4

    raise ValueError(f"Unsupported model: {model_name}")


def generate_gradcam(model, target_layer, input_tensor, class_index):
    """Generate a Grad-CAM heatmap for one predicted class."""

    captured = {}

    def forward_hook(module, inputs, output):
        captured["activation"] = output
        output.retain_grad()

    handle = target_layer.register_forward_hook(forward_hook)

    try:
        model.zero_grad(set_to_none=True)

        logits = model(input_tensor)

        score = logits[:, class_index].sum()
        score.backward()

        activation = captured.get("activation")

        if activation is None or activation.grad is None:
            raise RuntimeError("Unable to capture Grad-CAM gradients.")

        gradients = activation.grad

        # Average gradients across spatial dimensions.
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Weighted combination of feature maps.
        cam = (weights * activation).sum(
            dim=1,
            keepdim=True
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0].detach().cpu().numpy()

        cam -= cam.min()

        maximum = cam.max()

        if maximum > 0:
            cam /= maximum

        return cam, logits.detach()

    finally:
        handle.remove()


def load_model(model_name, checkpoint_path, device):
    """Load the trained model checkpoint."""

    model = build_model(
        model_name,
        pretrained=False
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    if checkpoint.get("model_name") != model_name:
        raise ValueError(
            f"Checkpoint belongs to {checkpoint.get('model_name')}, "
            f"not {model_name}."
        )

    if checkpoint.get("class_names") != CLASS_NAMES:
        raise ValueError(
            "Checkpoint class order does not match current configuration."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def read_disagreement_images(csv_path):
    """Read image paths where two models made different predictions."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Disagreement CSV not found: {csv_path}"
        )

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        return {
            row["filepath"]
            for row in csv.DictReader(file)
        }


def save_gradcam_figure(
    image_path,
    cam,
    actual_label,
    predicted_label,
    confidence,
    output_path,
    model_name,
    prediction_correct
):
    """Save original X-ray and Grad-CAM overlay."""

    with Image.open(image_path) as image:
        original = image.convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5)
    )

    axes[0].imshow(original)

    axes[0].set_title(
        f"Original X-ray\nActual: {actual_label}"
    )

    axes[0].axis("off")

    axes[1].imshow(original)

    axes[1].imshow(
        cam,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1
    )

    axes[1].set_title(
        f"{model_name} Grad-CAM\n"
        f"Predicted: {predicted_label}\n"
        f"Confidence: {confidence:.2%}"
    )

    axes[1].axis("off")

    status = (
        "Correct prediction"
        if prediction_correct
        else "Incorrect prediction"
    )

    figure.suptitle(
        f"{model_name.upper()} | {status}",
        fontsize=12
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(figure)


def main():

    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM explanations for chest X-rays."
    )

    parser.add_argument(
        "--model",
        choices=["resnet50", "densenet121"],
        required=True
    )

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    parser.add_argument(
        "--data-root",
        default=str(SPLIT_ROOT)
    )

    parser.add_argument(
        "--count",
        type=int,
        default=6
    )

    parser.add_argument(
        "--disagreements",
        action="store_true",
        help="Select only images where ResNet-50 and DenseNet-121 disagree."
    )

    parser.add_argument(
        "--disagreement-csv",
        default="outputs/model_comparison/model_disagreements.csv"
    )

    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be at least 1.")

    torch.set_num_threads(2)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path.resolve()}"
        )

    print("\n====================================")
    print("GRAD-CAM EXPLAINABILITY")
    print("====================================")
    print("Model:", args.model)
    print("Device:", device)

    # Load held-out test dataset.
    dataset = ChestXrayDataset(
        args.data_root,
        "test",
        train=False
    )

    # Load trained model.
    model = load_model(
        args.model,
        checkpoint_path,
        device
    )

    target_layer = get_target_layer(
        model,
        args.model
    )

    # Prepare output directory.
    output_folder = (
        OUTPUT_ROOT / args.model / "gradcam"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    disagreement_paths = None

    if args.disagreements:
        disagreement_paths = read_disagreement_images(
            Path(args.disagreement_csv)
        )

    # Find eligible test images.
    selected_indices = []

    for index, (tensor, label, image_path) in enumerate(dataset):

        if disagreement_paths is not None:
            if str(Path(image_path)) not in disagreement_paths:
                continue

        selected_indices.append(index)

        if len(selected_indices) >= args.count:
            break

    if not selected_indices:
        raise RuntimeError(
            "No matching test images were found for Grad-CAM."
        )

    results = []

    print("Selected images:", len(selected_indices))
    print("Generating heatmaps...\n")

    # Generate one explanation at a time.
    for output_index, dataset_index in enumerate(
        selected_indices,
        start=1
    ):

        tensor, true_label, image_path = dataset[dataset_index]

        input_tensor = tensor.unsqueeze(0).to(device)

        # Obtain prediction and Grad-CAM.
        with torch.enable_grad():

            initial_logits = model(input_tensor)

            predicted_class = int(
                initial_logits.argmax(dim=1).item()
            )

            cam, logits = generate_gradcam(
                model,
                target_layer,
                input_tensor,
                predicted_class
            )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0].detach().cpu().numpy()

        confidence = float(
            probabilities[predicted_class]
        )

        actual_name = CLASS_NAMES[true_label]
        predicted_name = CLASS_NAMES[predicted_class]

        correct = predicted_class == true_label

        filename = (
            f"gradcam_{output_index:02d}_"
            f"{Path(image_path).stem}.png"
        )

        output_path = output_folder / filename

        save_gradcam_figure(
            image_path=image_path,
            cam=cam,
            actual_label=actual_name,
            predicted_label=predicted_name,
            confidence=confidence,
            output_path=output_path,
            model_name=args.model,
            prediction_correct=correct
        )

        results.append({
            "image_path": str(image_path),
            "actual_class": actual_name,
            "predicted_class": predicted_name,
            "confidence": confidence,
            "correct_prediction": correct,
            "gradcam_image": filename
        })

        print(
            f"{output_index}/{len(selected_indices)} "
            f"| Actual: {actual_name} "
            f"| Predicted: {predicted_name} "
            f"| Confidence: {confidence:.4f} "
            f"| Correct: {correct}"
        )

    # Save explanation records.
    csv_path = output_folder / "gradcam_results.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)

    summary = {
        "model": args.model,
        "selected_images": len(results),
        "correct_predictions": sum(
            row["correct_prediction"] for row in results
        ),
        "incorrect_predictions": sum(
            not row["correct_prediction"] for row in results
        ),
        "selection": (
            "model_disagreements"
            if args.disagreements
            else "first_sorted_test_images"
        ),
        "output_directory": str(output_folder.resolve()),
        "results_csv": str(csv_path.resolve())
    }

    summary_path = output_folder / "gradcam_summary.json"

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\n====================================")
    print("GRAD-CAM COMPLETED")
    print("====================================")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
