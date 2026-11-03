"""Multi-class classification metrics."""
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)


def calculate_metrics(y_true, y_pred, probabilities, num_classes):
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(num_classes))
        ).tolist()
    }

    try:
        result["roc_auc_ovr_macro"] = float(
            roc_auc_score(
                y_true, probabilities, multi_class="ovr",
                average="macro", labels=list(range(num_classes))
            )
        )
    except ValueError:
        result["roc_auc_ovr_macro"] = None

    return result
