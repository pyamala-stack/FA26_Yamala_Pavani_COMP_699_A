"""Model constructors for ResNet-50 and DenseNet-121."""
from torch import nn
from torchvision import models
from src.config import NUM_CLASSES


def build_model(model_name, pretrained=True):
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        return model

    if model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
        return model

    raise ValueError("model_name must be 'resnet50' or 'densenet121'")
