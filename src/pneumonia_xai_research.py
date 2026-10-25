import io
import os
import random
import json
import hashlib
from datetime import datetime

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

st.set_page_config(
    page_title="Explainable Pneumonia Detection",
    page_icon="🩻",
    layout="wide",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
CLASSES = ["NORMAL", "PNEUMONIA"]
APP_VERSION = "2.0"

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(model_name, pretrained=True):
    if model_name == "ResNet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == "DenseNet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model.to(DEVICE)


def get_last_conv_layer(model, model_name):
    if model_name == "ResNet50":
        return model.layer4[-1]
    return model.features.denseblock4.denselayer16.conv2


def get_data_loaders(dataset_path, batch_size, seed=42):
    required_splits = ["train", "val", "test"]
    for split in required_splits:
        folder = os.path.join(dataset_path, split)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Missing dataset folder: {folder}")

    datasets_by_split = {
        split: datasets.ImageFolder(
            os.path.join(dataset_path, split),
            transform=TRANSFORM,
        )
        for split in required_splits
    }

    for split, dataset in datasets_by_split.items():
        if dataset.classes != CLASSES:
            raise ValueError(
                f"The {split} folder must contain NORMAL and PNEUMONIA "
                f"subfolders in that order. Found: {dataset.classes}"
            )
        if len(dataset) == 0:
            raise ValueError(f"The {split} dataset contains no images.")

    generator = torch.Generator()
    generator.manual_seed(seed)

    loaders = {
        "train": DataLoader(
            datasets_by_split["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
        ),
        "val": DataLoader(
            datasets_by_split["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        ),
        "test": DataLoader(
            datasets_by_split["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        ),
    }

    sizes = {
        "Training": len(datasets_by_split["train"]),
        "Validation": len(datasets_by_split["val"]),
        "Test": len(datasets_by_split["test"]),
    }

    class_counts = {}
    for split, dataset in datasets_by_split.items():
        labels = [label for _, label in dataset.samples]
        counts = pd.Series(labels).value_counts().to_dict()
        class_counts[split] = {
            class_name: int(counts.get(index, 0))
            for index, class_name in enumerate(CLASSES)
        }

    return loaders, sizes, class_counts


def evaluate_model(model, loader, threshold=0.5):
    model.eval()
    actual = []
    probabilities = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            actual.extend(labels.numpy().tolist())
            probabilities.extend(probs.cpu().numpy().tolist())

    if not actual:
        raise ValueError("The evaluation dataset is empty.")

    predicted = [1 if prob >= threshold else 0 for prob in probabilities]
    metrics = {
        "Accuracy": accuracy_score(actual, predicted),
        "Precision": precision_score(actual, predicted, zero_division=0),
        "Recall": recall_score(actual, predicted, zero_division=0),
        "F1-score": f1_score(actual, predicted, zero_division=0),
        "ROC-AUC": (
            roc_auc_score(actual, probabilities)
            if len(set(actual)) == 2
            else np.nan
        ),
        "Average Precision": (
            average_precision_score(actual, probabilities)
            if len(set(actual)) == 2
            else np.nan
        ),
    }
    return metrics, actual, predicted, probabilities


def train_model(model, train_loader, val_loader, epochs, learning_rate,
                fine_tune, patience):
    if not fine_tune:
        for parameter in model.parameters():
            parameter.requires_grad = False
        if hasattr(model, "fc"):
            for parameter in model.fc.parameters():
                parameter.requires_grad = True
        else:
            for parameter in model.classifier.parameters():
                parameter.requires_grad = True
    else:
        for parameter in model.parameters():
            parameter.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("No trainable model parameters were found.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(trainable, lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    history = []
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    progress = st.progress(0)
    status = st.empty()
    history_placeholder = st.empty()

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_count = labels.size(0)
            train_loss_sum += loss.item() * batch_count
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_count += batch_count

        train_loss = train_loss_sum / max(train_count, 1)
        train_accuracy = train_correct / max(train_count, 1)

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_count = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                batch_count = labels.size(0)
                val_loss_sum += loss.item() * batch_count
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_count += batch_count

        val_loss = val_loss_sum / max(val_count, 1)
        val_accuracy = val_correct / max(val_count, 1)
        scheduler.step(val_loss)

        history.append({
            "Epoch": epoch + 1,
            "Training Loss": train_loss,
            "Training Accuracy": train_accuracy,
            "Validation Loss": val_loss,
            "Validation Accuracy": val_accuracy,
            "Learning Rate": optimizer.param_groups[0]["lr"],
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        progress.progress((epoch + 1) / epochs)
        status.write(
            f"Epoch {epoch + 1}/{epochs} completed — "
            f"training accuracy: {train_accuracy:.3f}, "
            f"validation accuracy: {val_accuracy:.3f}"
        )
        history_placeholder.dataframe(
            pd.DataFrame(history).round(4),
            use_container_width=True,
        )

        if patience > 0 and epochs_without_improvement >= patience:
            status.info(
                f"Early stopping: validation loss did not improve for "
                f"{patience} consecutive epoch(s)."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    progress.empty()
    return model, pd.DataFrame(history), best_val_loss


def generate_gradcam(model, image_tensor, model_name, target_class=None):
    activations = []
    gradients = []
    target_layer = get_last_conv_layer(model, model_name)

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.eval()
        model.zero_grad()
        image_tensor = image_tensor.to(DEVICE)
        output = model(image_tensor)
        predicted_class = output.argmax(dim=1).item()
        explanation_class = (
            predicted_class if target_class is None else int(target_class)
        )
        output[0, explanation_class].backward()

        activation = activations[0]
        gradient = gradients[0]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        probabilities = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
        return cam, predicted_class, probabilities, explanation_class
    finally:
        forward_handle.remove()
        backward_handle.remove()


def model_to_bytes(model, model_name, metadata=None):
    buffer = io.BytesIO()
    torch.save({
        "model_name": model_name,
        "state_dict": model.state_dict(),
        "classes": CLASSES,
        "image_size": IMAGE_SIZE,
        "metadata": metadata or {},
    }, buffer)
    buffer.seek(0)
    return buffer.getvalue()


def load_model_checkpoint(uploaded_file):
    checkpoint = torch.load(
        io.BytesIO(uploaded_file.getvalue()),
        map_location=DEVICE,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(
            "This file is not a supported checkpoint downloaded from this app."
        )
    model_name = checkpoint.get("model_name")
    if model_name not in ["ResNet50", "DenseNet121"]:
        raise ValueError("The checkpoint does not identify a supported model.")
    if checkpoint.get("classes") != CLASSES:
        raise ValueError("The checkpoint class labels do not match this app.")
    model = build_model(model_name, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, model_name, checkpoint.get("metadata", {})


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["Epoch"], history["Training Loss"], marker="o",
                 label="Training")
    axes[0].plot(history["Epoch"], history["Validation Loss"], marker="o",
                 label="Validation")
    axes[0].set_title("Loss by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["Epoch"], history["Training Accuracy"], marker="o",
                 label="Training")
    axes[1].plot(history["Epoch"], history["Validation Accuracy"], marker="o",
                 label="Validation")
    axes[1].set_title("Accuracy by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    return fig


def plot_confusion_matrix(actual, predicted):
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASSES,
    ).plot(ax=ax, colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_roc_and_pr(actual, probabilities):
    if len(set(actual)) < 2:
        return None, None
    fpr, tpr, _ = roc_curve(actual, probabilities)
    precision, recall, _ = precision_recall_curve(actual, probabilities)

    roc_fig, roc_ax = plt.subplots(figsize=(5, 4))
    roc_ax.plot(fpr, tpr, label="ROC curve")
    roc_ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    roc_ax.set_xlabel("False Positive Rate")
    roc_ax.set_ylabel("True Positive Rate")
    roc_ax.set_title("ROC Curve")
    roc_ax.legend()
    roc_fig.tight_layout()

    pr_fig, pr_ax = plt.subplots(figsize=(5, 4))
    pr_ax.plot(recall, precision)
    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_title("Precision-Recall Curve")
    pr_fig.tight_layout()
    return roc_fig, pr_fig


def image_hash(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def predict_image(model, image, threshold):
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    predicted_class = 1 if probabilities[1] >= threshold else 0
    return predicted_class, probabilities, tensor


set_seed()

if "trained_model" not in st.session_state:
    st.session_state["trained_model"] = None
if "trained_model_name" not in st.session_state:
    st.session_state["trained_model_name"] = None
if "test_metrics_by_model" not in st.session_state:
    st.session_state["test_metrics_by_model"] = {}
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []

st.title("Explainable AI for Pneumonia Detection")
st.write(
    "Research application for chest X-ray classification using ResNet50 and "
    "DenseNet121, model evaluation, batch predictions, and Grad-CAM explanations."
)
st.warning(
    "Research use only. This application is not a medical device and must not "
    "be used to diagnose, rule out, or make treatment decisions about pneumonia."
)

with st.sidebar:
    st.header("Experiment Settings")
    st.caption(f"Running on: {DEVICE}")

    model_name = st.selectbox("Select model", ["ResNet50", "DenseNet121"])
    dataset_path = st.text_input("Dataset folder", "data/chest_xray")
    batch_size = st.selectbox("Batch size", [4, 8, 16, 32], index=1)
    epochs = st.slider("Maximum training epochs", 1, 30, 5)
    learning_rate = st.selectbox(
        "Learning rate", [0.00001, 0.0001, 0.001], index=1
    )
    fine_tune = st.checkbox("Fine-tune all model layers", value=False)
    patience = st.slider(
        "Early stopping patience (0 disables it)", 0, 8, 3
    )
    seed = st.number_input("Random seed", min_value=0, max_value=999999,
                           value=42, step=1)
    prediction_threshold = st.slider(
        "Pneumonia probability threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="This changes the classification cutoff, not the model probabilities.",
    )

    st.divider()
    st.subheader("Load a Saved Model")
    checkpoint_file = st.file_uploader(
        "Upload a checkpoint downloaded from this app",
        type=["pth", "pt"],
        key="checkpoint_uploader",
    )
    if st.button("Load Saved Model", use_container_width=True):
        if checkpoint_file is None:
            st.error("Upload a checkpoint file first.")
        else:
            try:
                with st.spinner("Loading saved model..."):
                    loaded_model, loaded_name, metadata = load_model_checkpoint(
                        checkpoint_file
                    )
                st.session_state["trained_model"] = loaded_model
                st.session_state["trained_model_name"] = loaded_name
                st.session_state["loaded_model_metadata"] = metadata
                st.success(f"{loaded_name} loaded successfully.")
            except Exception as error:
                st.error(f"Could not load model: {error}")

tab_experiment, tab_prediction, tab_comparison, tab_reports, tab_about = st.tabs([
    "Train and Evaluate",
    "Prediction and Grad-CAM",
    "Compare Models",
    "Reports and History",
    "Project Information",
])

with tab_experiment:
    st.subheader("Dataset and Model Training")
    st.write(
        "Required structure: train, val, and test folders. Each split must "
        "contain NORMAL and PNEUMONIA subfolders."
    )
    with st.expander("View the required dataset structure"):
        st.code(
            "data/chest_xray/\n"
            "    train/\n"
            "        NORMAL/\n"
            "        PNEUMONIA/\n"
            "    val/\n"
            "        NORMAL/\n"
            "        PNEUMONIA/\n"
            "    test/\n"
            "        NORMAL/\n"
            "        PNEUMONIA/"
        )

    if st.button("Load Dataset and Train", type="primary"):
        try:
            set_seed(int(seed))
            with st.spinner("Loading and checking the dataset..."):
                loaders, sizes, class_counts = get_data_loaders(
                    dataset_path, batch_size, int(seed)
                )

            st.subheader("Dataset Summary")
            st.dataframe(
                pd.DataFrame([
                    {"Split": split, "Images": size}
                    for split, size in sizes.items()
                ]),
                use_container_width=True,
            )
            st.write("Class distribution")
            st.dataframe(
                pd.DataFrame(class_counts).T,
                use_container_width=True,
            )

            with st.spinner("Loading pretrained model..."):
                model = build_model(model_name, pretrained=True)

            with st.spinner("Training and validating model..."):
                model, history, best_val_loss = train_model(
                    model,
                    loaders["train"],
                    loaders["val"],
                    epochs,
                    learning_rate,
                    fine_tune,
                    patience,
                )

            val_metrics, _, _, _ = evaluate_model(
                model, loaders["val"], prediction_threshold
            )
            st.session_state["trained_model"] = model
            st.session_state["trained_model_name"] = model_name
            st.session_state["training_history"] = history
            st.session_state["validation_metrics"] = val_metrics
            st.session_state["dataset_path"] = dataset_path
            st.session_state["experiment_settings"] = {
                "model": model_name,
                "dataset_path": dataset_path,
                "batch_size": batch_size,
                "requested_epochs": epochs,
                "completed_epochs": int(len(history)),
                "learning_rate": learning_rate,
                "fine_tune_all_layers": fine_tune,
                "early_stopping_patience": patience,
                "random_seed": int(seed),
                "threshold": prediction_threshold,
                "device": str(DEVICE),
                "best_validation_loss": float(best_val_loss),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }

            st.success("Training and validation completed.")
            st.subheader("Validation Metrics")
            st.dataframe(pd.DataFrame([val_metrics]).round(4),
                         use_container_width=True)

            st.subheader("Training History")
            st.dataframe(history.round(4), use_container_width=True)
            history_fig = plot_history(history)
            st.pyplot(history_fig)
            plt.close(history_fig)

            checkpoint_bytes = model_to_bytes(
                model,
                model_name,
                st.session_state["experiment_settings"],
            )
            st.download_button(
                "Download Complete Model Checkpoint",
                data=checkpoint_bytes,
                file_name=f"{model_name.lower()}_checkpoint.pth",
                mime="application/octet-stream",
            )
            st.download_button(
                "Download Training History CSV",
                data=history.to_csv(index=False).encode("utf-8"),
                file_name=f"{model_name.lower()}_training_history.csv",
                mime="text/csv",
            )
        except Exception as error:
            st.error(f"Training failed: {error}")

    st.divider()
    st.subheader("Test Dataset Evaluation")
    if st.button("Evaluate on Test Dataset"):
        model = st.session_state.get("trained_model")
        trained_name = st.session_state.get("trained_model_name")
        trained_path = st.session_state.get("dataset_path")

        if model is None:
            st.error("Train or load a model before test evaluation.")
        elif trained_path and trained_path != dataset_path:
            st.error(
                "The current dataset path differs from the path used during "
                "training. Enter the original dataset path to evaluate it."
            )
        else:
            try:
                with st.spinner("Evaluating on the test dataset..."):
                    loaders, _, _ = get_data_loaders(
                        dataset_path, batch_size, int(seed)
                    )
                    metrics, actual, predicted, probabilities = evaluate_model(
                        model, loaders["test"], prediction_threshold
                    )

                st.session_state["test_metrics_by_model"][trained_name] = metrics
                st.session_state["latest_test_result"] = {
                    "model_name": trained_name,
                    "dataset_path": dataset_path,
                    "threshold": prediction_threshold,
                    "metrics": metrics,
                    "actual": actual,
                    "predicted": predicted,
                    "probabilities": probabilities,
                }

                st.subheader("Test Metrics")
                st.dataframe(pd.DataFrame([metrics]).round(4),
                             use_container_width=True)

                cm_fig = plot_confusion_matrix(actual, predicted)
                st.pyplot(cm_fig)
                plt.close(cm_fig)

                report = classification_report(
                    actual, predicted, labels=[0, 1],
                    target_names=CLASSES, output_dict=True, zero_division=0
                )
                st.subheader("Classification Report")
                st.dataframe(pd.DataFrame(report).T.round(4),
                             use_container_width=True)

                roc_fig, pr_fig = plot_roc_and_pr(actual, probabilities)
                if roc_fig is not None:
                    left, right = st.columns(2)
                    with left:
                        st.pyplot(roc_fig)
                    with right:
                        st.pyplot(pr_fig)
                    plt.close(roc_fig)
                    plt.close(pr_fig)
                else:
                    st.info(
                        "ROC and precision-recall curves need both classes "
                        "to be present in the test dataset."
                    )

                results = pd.DataFrame({
                    "Actual": [CLASSES[i] for i in actual],
                    "Predicted": [CLASSES[i] for i in predicted],
                    "Normal Probability": [1 - p for p in probabilities],
                    "Pneumonia Probability": probabilities,
                    "Threshold": [prediction_threshold] * len(actual),
                })
                st.download_button(
                    "Download Test Predictions CSV",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name=f"{trained_name.lower()}_test_predictions.csv",
                    mime="text/csv",
                )
                st.download_button(
                    "Download Classification Report CSV",
                    data=pd.DataFrame(report).T.to_csv().encode("utf-8"),
                    file_name=f"{trained_name.lower()}_classification_report.csv",
                    mime="text/csv",
                )
                st.caption(
                    "Test results are specific to this dataset split, trained "
                    "checkpoint, and threshold. Use the same test data and "
                    "settings when comparing models."
                )
            except Exception as error:
                st.error(f"Test evaluation failed: {error}")

with tab_prediction:
    st.subheader("Chest X-ray Prediction")
    model = st.session_state.get("trained_model")
    trained_name = st.session_state.get("trained_model_name")

    if model is None:
        st.info(
            "Train a model in the Train and Evaluate tab, or load a saved "
            "checkpoint from the sidebar."
        )
    else:
        st.write(f"Active model: **{trained_name}**")

    single_file = st.file_uploader(
        "Upload one chest X-ray image",
        type=["png", "jpg", "jpeg"],
        key="single_xray",
    )

    if single_file is not None:
        try:
            original_image = Image.open(single_file).convert("RGB")
            st.image(original_image, caption="Uploaded chest X-ray", width=350)
            if model is not None and st.button(
                "Generate Prediction and Grad-CAM", key="single_predict"
            ):
                with st.spinner("Generating prediction and explanation..."):
                    predicted_class, probabilities, tensor = predict_image(
                        model, original_image, prediction_threshold
                    )
                    cam, _, _, explanation_class = generate_gradcam(
                        model, tensor, trained_name
                    )

                left, right = st.columns(2)
                with left:
                    st.subheader("Prediction")
                    st.metric("Predicted class", CLASSES[predicted_class])
                    st.write(
                        f"Normal probability: {probabilities[0] * 100:.2f}%"
                    )
                    st.write(
                        f"Pneumonia probability: {probabilities[1] * 100:.2f}%"
                    )
                    st.write(
                        f"Decision threshold: {prediction_threshold:.2f}"
                    )
                    st.dataframe(pd.DataFrame({
                        "Class": CLASSES,
                        "Probability": probabilities,
                    }).round(4), use_container_width=True)

                with right:
                    st.subheader("Grad-CAM Explanation")
                    resized = original_image.resize((IMAGE_SIZE, IMAGE_SIZE))
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.imshow(resized)
                    ax.imshow(cam, cmap="jet", alpha=0.4)
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close(fig)
                    st.caption(
                        f"Heatmap generated for model class: "
                        f"{CLASSES[explanation_class]}."
                    )

                st.session_state["prediction_history"].append({
                    "Time": datetime.now().isoformat(timespec="seconds"),
                    "File": single_file.name,
                    "Model": trained_name,
                    "Predicted Class": CLASSES[predicted_class],
                    "Normal Probability": float(probabilities[0]),
                    "Pneumonia Probability": float(probabilities[1]),
                    "Threshold": float(prediction_threshold),
                    "Image SHA256": image_hash(original_image),
                })
                st.warning(
                    "This is a model output for research only. A heatmap does "
                    "not prove that the prediction is medically correct."
                )
        except (UnidentifiedImageError, OSError) as error:
            st.error(f"The uploaded file could not be read as an image: {error}")
        except Exception as error:
            st.error(f"Prediction failed: {error}")

    st.divider()
    st.subheader("Batch Prediction")
    batch_files = st.file_uploader(
        "Upload multiple chest X-ray images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="batch_xrays",
    )
    if st.button("Predict All Uploaded Images", key="batch_predict"):
        if model is None:
            st.error("Train or load a model before batch prediction.")
        elif not batch_files:
            st.error("Upload at least one image.")
        else:
            batch_rows = []
            seen_hashes = set()
            errors = []
            progress = st.progress(0)
            for index, file in enumerate(batch_files):
                try:
                    image = Image.open(file).convert("RGB")
                    digest = image_hash(image)
                    predicted_class, probabilities, _ = predict_image(
                        model, image, prediction_threshold
                    )
                    batch_rows.append({
                        "File": file.name,
                        "Predicted Class": CLASSES[predicted_class],
                        "Normal Probability": float(probabilities[0]),
                        "Pneumonia Probability": float(probabilities[1]),
                        "Threshold": prediction_threshold,
                        "Duplicate Image": digest in seen_hashes,
                    })
                    seen_hashes.add(digest)
                except Exception as error:
                    errors.append(f"{file.name}: {error}")
                progress.progress((index + 1) / len(batch_files))
            progress.empty()

            if batch_rows:
                batch_df = pd.DataFrame(batch_rows)
                st.dataframe(batch_df.round(4), use_container_width=True)
                st.download_button(
                    "Download Batch Predictions CSV",
                    data=batch_df.to_csv(index=False).encode("utf-8"),
                    file_name="batch_xray_predictions.csv",
                    mime="text/csv",
                )
            if errors:
                st.warning("Some files could not be processed:\n\n" + "\n".join(errors))
            st.warning(
                "Batch predictions are for research only and must not be used "
                "for medical decisions."
            )

with tab_comparison:
    st.subheader("Model Comparison")
    results = st.session_state.get("test_metrics_by_model", {})
    if results:
        comparison = pd.DataFrame(results).T
        st.dataframe(comparison.round(4), use_container_width=True)
        numeric_columns = comparison.select_dtypes(include=[np.number]).columns
        if len(numeric_columns):
            metric_to_plot = st.selectbox(
                "Metric to compare", list(numeric_columns), key="comparison_metric"
            )
            fig, ax = plt.subplots(figsize=(7, 4))
            comparison[metric_to_plot].plot(kind="bar", ax=ax)
            ax.set_title(f"Model Comparison — {metric_to_plot}")
            ax.set_ylabel(metric_to_plot)
            ax.set_ylim(0, 1.05)
            ax.tick_params(axis="x", rotation=0)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        st.download_button(
            "Download Model Comparison CSV",
            data=comparison.to_csv().encode("utf-8"),
            file_name="model_comparison.csv",
            mime="text/csv",
        )
        if len(results) < 2:
            st.info(
                "Evaluate both ResNet50 and DenseNet121 on the same test split "
                "to compare their results."
            )
    else:
        st.info(
            "No test results are available. Train or load a model and evaluate "
            "it on the test dataset first."
        )

    latest = st.session_state.get("latest_test_result")
    if latest:
        st.subheader("Latest Test Evaluation Details")
        st.json({
            "Model": latest["model_name"],
            "Dataset path": latest["dataset_path"],
            "Threshold": latest["threshold"],
            "Metrics": latest["metrics"],
        })

with tab_reports:
    st.subheader("Experiment Report")
    settings = st.session_state.get("experiment_settings")
    history = st.session_state.get("training_history")
    val_metrics = st.session_state.get("validation_metrics")
    latest_test = st.session_state.get("latest_test_result")

    if settings or val_metrics or latest_test:
        report = {
            "application_version": APP_VERSION,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "experiment_settings": settings or {},
            "validation_metrics": val_metrics or {},
            "latest_test_evaluation": (
                {
                    "model": latest_test["model_name"],
                    "dataset_path": latest_test["dataset_path"],
                    "threshold": latest_test["threshold"],
                    "metrics": latest_test["metrics"],
                } if latest_test else None
            ),
            "device": str(DEVICE),
            "disclaimer": (
                "Research use only. Not intended for clinical diagnosis."
            ),
        }
        st.json(report)
        st.download_button(
            "Download Experiment Report JSON",
            data=json.dumps(report, indent=2, default=str).encode("utf-8"),
            file_name="pneumonia_experiment_report.json",
            mime="application/json",
        )
        if history is not None:
            st.download_button(
                "Download Current Training History CSV",
                data=history.to_csv(index=False).encode("utf-8"),
                file_name="training_history.csv",
                mime="text/csv",
            )
    else:
        st.info(
            "Train a model or run an evaluation to create an experiment report."
        )

    st.divider()
    st.subheader("Prediction History")
    prediction_history = st.session_state.get("prediction_history", [])
    if prediction_history:
        history_df = pd.DataFrame(prediction_history)
        st.dataframe(history_df, use_container_width=True)
        st.download_button(
            "Download Prediction History CSV",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv",
        )
        if st.button("Clear Prediction History"):
            st.session_state["prediction_history"] = []
            st.rerun()
    else:
        st.info("No single-image prediction history has been recorded in this session.")

with tab_about:
    st.subheader("Research Objective")
    st.write(
        "This project compares ResNet50 and DenseNet121 for binary chest X-ray "
        "classification and explores model outputs using Grad-CAM."
    )

    st.subheader("Evaluation Metrics")
    st.write(
        "Accuracy measures overall correctness. Precision measures how many "
        "predicted pneumonia cases are correct. Recall measures how many actual "
        "pneumonia cases are detected. F1-score balances precision and recall. "
        "ROC-AUC measures how well the model ranks the two classes across "
        "thresholds. Average precision summarises the precision-recall curve."
    )

    st.subheader("New Features")
    st.markdown(
        "- Training and validation loss/accuracy charts\n"
        "- Early stopping and learning-rate reduction\n"
        "- Confusion matrix, classification report, ROC and precision-recall curves\n"
        "- Downloadable model checkpoints and uploadable saved checkpoints\n"
        "- Single-image Grad-CAM and batch predictions\n"
        "- Configurable probability threshold\n"
        "- CSV and JSON research reports\n"
        "- Session-based prediction history and duplicate-image flags for batch uploads"
    )

    st.subheader("Research Limitations")
    st.write(
        "Results depend on the dataset, image quality, preprocessing, model "
        "settings, and training procedure. Grad-CAM is a visual explanation, "
        "not proof that a model uses medically meaningful features. Model "
        "probabilities are not guaranteed to be calibrated. This application "
        "is not intended for clinical diagnosis or treatment decisions."
    )
