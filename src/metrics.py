"""Classification metrics for three-class chest X-ray experiments."""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, classification_report)


def calculate_metrics(y_true, y_pred, probabilities, num_classes, class_names=None):
    labels = list(range(num_classes))
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": classification_report(y_true, y_pred, labels=labels,
            target_names=class_names, output_dict=True, zero_division=0) if class_names else
            classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    }
    try:
        result["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, probabilities,
            multi_class="ovr", average="macro", labels=labels))
    except ValueError:
        result["roc_auc_ovr_macro"] = None
    return result
