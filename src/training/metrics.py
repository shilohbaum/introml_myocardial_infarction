import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "auc": (
            roc_auc_score(y_true, y_prob)
            if len(np.unique(y_true)) == 2
            else np.nan
        ),
    }