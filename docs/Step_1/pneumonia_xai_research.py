import os
import io
import random
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

st.set_page_config(
    page_title="Explainable Pneumonia Detection",
    layout="wide"
)

st.title("Explainable AI for Pneumonia Detection")
st.write(
    "Research application comparing ResNet50 and DenseNet121 "
    "for chest X-ray classification and Grad-CAM visualization."
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
CLASSES = ["NORMAL", "PNEUMONIA"]

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
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
    else:
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    return model.to(DEVICE)


def get_last_conv_layer(model, model_name):
    if model_name == "ResNet50":
        return model.layer4[-1]
    return model.features.denseblock4.denselayer16.conv2


def get_data_loaders(dataset_path, batch_size):
    splits = ["train", "val", "test"]
    for split in splits:
        folder = os.path.join(dataset_path, split)
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Missing folder: {folder}"
            )

    train_data = datasets.ImageFolder(
        os.path.join(dataset_path, "train"),
        transform=TRANSFORM
    )
    val_data = datasets.ImageFolder(
        os.path.join(dataset_path, "val"),
        transform=TRANSFORM
    )
    test_data = datasets.ImageFolder(
        os.path.join(dataset_path, "test"),
        transform=TRANSFORM
    )

    for dataset in [train_data, val_data, test_data]:
        if dataset.classes != CLASSES:
            raise ValueError(
                "Each dataset split must contain NORMAL and PNEUMONIA folders. "
                f"Found: {dataset.classes}"
            )
        if len(dataset) == 0:
            raise ValueError("A dataset split contains no images.")

    loaders = {
        "train": DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        ),
        "val": DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        ),
        "test": DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
    }

    return loaders, {
        "Training": len(train_data),
        "Validation": len(val_data),
        "Test": len(test_data)
    }


def train_model(model, loader, epochs, learning_rate, fine_tune):
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

    trainable = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(trainable, lr=learning_rate)
    history = []

    progress = st.progress(0)
    status = st.empty()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_images = 0

        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            total_images += labels.size(0)

        row = {
            "Epoch": epoch + 1,
            "Loss": total_loss / total_images,
            "Training Accuracy": total_correct / total_images
        }
        history.append(row)

        progress.progress((epoch + 1) / epochs)
        status.write(
            f"Completed epoch {epoch + 1} of {epochs}"
        )

    progress.empty()
    status.empty()

    return model, pd.DataFrame(history)


def evaluate_model(model, loader):
    model.eval()
    actual = []
    predicted = []
    probabilities = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            actual.extend(labels.numpy().tolist())
            predicted.extend(
                outputs.argmax(dim=1).cpu().numpy().tolist()
            )
            probabilities.extend(
                probs[:, 1].cpu().numpy().tolist()
            )

    if not actual:
        raise ValueError("The evaluation dataset is empty.")

    metrics = {
        "Accuracy": accuracy_score(actual, predicted),
        "Precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "Recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "F1-score": f1_score(
            actual, predicted, zero_division=0
        ),
        "ROC-AUC": (
            roc_auc_score(actual, probabilities)
            if len(set(actual)) == 2
            else np.nan
        )
    }

    return metrics, actual, predicted, probabilities


def generate_gradcam(model, image_tensor, model_name):
    activations = []
    gradients = []
    target_layer = get_last_conv_layer(model, model_name)

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )
    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    try:
        model.eval()
        model.zero_grad()

        image_tensor = image_tensor.to(DEVICE)
        output = model(image_tensor)
        predicted_class = output.argmax(dim=1).item()
        output[0, predicted_class].backward()

        activation = activations[0]
        gradient = gradients[0]
        weights = gradient.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = torch.relu(
            (weights * activation).sum(dim=1, keepdim=True)
        )

        cam = torch.nn.functional.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (
            cam.max() - cam.min() + 1e-8
        )

        probabilities = torch.softmax(
            output, dim=1
        )[0].detach().cpu().numpy()

        return cam, predicted_class, probabilities
    finally:
        forward_handle.remove()
        backward_handle.remove()


def model_to_bytes(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    return buffer.getvalue()


set_seed()

with st.sidebar:
    st.header("Experiment Settings")

    model_name = st.selectbox(
        "Select model",
        ["ResNet50", "DenseNet121"]
    )

    dataset_path = st.text_input(
        "Dataset folder",
        "data/chest_xray"
    )

    batch_size = st.selectbox(
        "Batch size",
        [4, 8, 16, 32],
        index=1
    )

    epochs = st.slider(
        "Training epochs",
        min_value=1,
        max_value=10,
        value=2
    )

    learning_rate = st.selectbox(
        "Learning rate",
        [0.00001, 0.0001, 0.001],
        index=1
    )

    fine_tune = st.checkbox(
        "Fine-tune all model layers",
        value=False
    )

tab_experiment, tab_prediction, tab_comparison, tab_about = st.tabs([
    "Train and Evaluate",
    "Prediction and Grad-CAM",
    "Compare Models",
    "Project Information"
])

with tab_experiment:
    st.subheader("Dataset and Model Training")
    st.write(
        "The dataset must contain train, val, and test folders. "
        "Each split must contain NORMAL and PNEUMONIA subfolders."
    )

    if st.button("Load Dataset and Train", type="primary"):
        try:
            with st.spinner("Loading dataset..."):
                loaders, sizes = get_data_loaders(
                    dataset_path, batch_size
                )

            st.write("Dataset sizes")
            st.dataframe(
                pd.DataFrame(
                    list(sizes.items()),
                    columns=["Split", "Number of Images"]
                ),
                use_container_width=True
            )

            with st.spinner("Loading pretrained model..."):
                model = build_model(model_name)

            with st.spinner("Training model..."):
                model, history = train_model(
                    model,
                    loaders["train"],
                    epochs,
                    learning_rate,
                    fine_tune
                )

            with st.spinner("Evaluating validation data..."):
                val_metrics, _, _, _ = evaluate_model(
                    model, loaders["val"]
                )

            st.subheader("Validation Metrics")
            st.dataframe(
                pd.DataFrame([val_metrics]).round(4),
                use_container_width=True
            )

            st.session_state["trained_model"] = model
            st.session_state["trained_model_name"] = model_name
            st.session_state["training_history"] = history
            st.session_state["validation_metrics"] = val_metrics
            st.session_state["dataset_path"] = dataset_path

            st.success("Training and validation completed.")

            st.subheader("Training History")
            st.dataframe(history, use_container_width=True)

            fig, ax = plt.subplots()
            ax.plot(
                history["Epoch"],
                history["Loss"],
                marker="o"
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Training Loss")
            ax.set_title("Training Loss by Epoch")
            st.pyplot(fig)
            plt.close(fig)

            st.download_button(
                "Download Trained Model Weights",
                data=model_to_bytes(model),
                file_name=f"{model_name.lower()}_weights.pth",
                mime="application/octet-stream"
            )

        except Exception as error:
            st.error(f"Training failed: {error}")

    if st.button("Evaluate on Test Dataset"):
        model = st.session_state.get("trained_model")
        trained_name = st.session_state.get("trained_model_name")
        trained_path = st.session_state.get("dataset_path")

        if model is None:
            st.error("Train a model before evaluating the test dataset.")
        elif trained_path != dataset_path:
            st.error("The dataset path differs from the training dataset.")
        else:
            try:
                with st.spinner("Evaluating test dataset..."):
                    loaders, _ = get_data_loaders(
                        trained_path, batch_size
                    )
                    metrics, actual, predicted, probabilities = (
                        evaluate_model(model, loaders["test"])
                    )

                st.session_state["test_metrics_by_model"] = (
                    st.session_state.get("test_metrics_by_model", {})
                )
                st.session_state["test_metrics_by_model"][trained_name] = metrics

                st.subheader("Test Metrics")
                st.dataframe(
                    pd.DataFrame([metrics]).round(4),
                    use_container_width=True
                )

                matrix = confusion_matrix(
                    actual, predicted, labels=[0, 1]
                )
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay(
                    confusion_matrix=matrix,
                    display_labels=CLASSES
                ).plot(ax=ax, colorbar=False)
                ax.set_title("Test Confusion Matrix")
                st.pyplot(fig)
                plt.close(fig)

                results = pd.DataFrame({
                    "Actual": [CLASSES[i] for i in actual],
                    "Predicted": [CLASSES[i] for i in predicted],
                    "Pneumonia Probability": probabilities
                })

                st.download_button(
                    "Download Test Predictions",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name=f"{trained_name.lower()}_test_predictions.csv",
                    mime="text/csv"
                )

                st.warning(
                    "Test metrics are for this particular trained model "
                    "and dataset. Use the same data splits and experimental "
                    "settings for a fair model comparison."
                )

            except Exception as error:
                st.error(f"Test evaluation failed: {error}")

with tab_prediction:
    st.subheader("Chest X-ray Prediction")

    uploaded_file = st.file_uploader(
        "Upload a chest X-ray image",
        type=["png", "jpg", "jpeg"]
    )

    model = st.session_state.get("trained_model")
    trained_name = st.session_state.get("trained_model_name")

    if model is None:
        st.info(
            "Train a model in the Train and Evaluate tab before making predictions."
        )
    else:
        st.write(f"Active trained model: {trained_name}")

    if uploaded_file is not None:
        try:
            original_image = Image.open(uploaded_file).convert("RGB")
            st.image(
                original_image,
                caption="Uploaded Chest X-ray",
                width=350
            )

            if model is not None and st.button(
                "Generate Prediction and Grad-CAM"
            ):
                _, image_tensor = (
                    original_image,
                    TRANSFORM(original_image).unsqueeze(0)
                )

                with st.spinner("Generating prediction and explanation..."):
                    cam, predicted_class, probabilities = generate_gradcam(
                        model, image_tensor, trained_name
                    )

                left, right = st.columns(2)

                with left:
                    st.subheader("Prediction")
                    st.write(
                        f"Predicted class: {CLASSES[predicted_class]}"
                    )
                    st.write(
                        f"Model confidence: "
                        f"{probabilities[predicted_class] * 100:.2f}%"
                    )
                    st.dataframe(
                        pd.DataFrame({
                            "Class": CLASSES,
                            "Probability": probabilities
                        }).round(4),
                        use_container_width=True
                    )

                with right:
                    st.subheader("Grad-CAM Explanation")
                    resized_image = original_image.resize(
                        (IMAGE_SIZE, IMAGE_SIZE)
                    )
                    fig, ax = plt.subplots()
                    ax.imshow(resized_image)
                    ax.imshow(cam, cmap="jet", alpha=0.4)
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close(fig)

                st.caption(
                    "Grad-CAM highlights image regions that influence "
                    "the model output. It does not establish medical correctness."
                )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

with tab_comparison:
    st.subheader("Model Comparison")

    results = st.session_state.get(
        "test_metrics_by_model", {}
    )

    if results:
        comparison = pd.DataFrame(results).T
        st.dataframe(
            comparison.round(4),
            use_container_width=True
        )

        if len(results) < 2:
            st.info(
                "Train and evaluate the other model using the same dataset "
                "splits and settings to compare both models."
            )
    else:
        st.info(
            "No test evaluation results are available yet. "
            "Train and evaluate a model first."
        )

with tab_about:
    st.subheader("Research Objective")
    st.write(
        "This project compares ResNet50 and DenseNet121 for binary "
        "chest X-ray classification and examines their predictions using Grad-CAM."
    )

    st.subheader("Evaluation Metrics")
    st.write(
        "Accuracy measures overall correctness. Precision measures the "
        "proportion of predicted pneumonia cases that are correct. Recall "
        "measures the proportion of pneumonia cases detected. F1-score balances "
        "precision and recall. ROC-AUC measures how well the model separates "
        "the two classes."
    )

    st.subheader("Required Dataset Structure")
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

    st.subheader("Research Limitations")
    st.write(
        "The results depend on the dataset, preprocessing, model settings, "
        "and training procedure. Grad-CAM is an explanatory visualization, "
        "not proof that a model uses medically meaningful features. This "
        "research application is not intended for clinical diagnosis."
    )
