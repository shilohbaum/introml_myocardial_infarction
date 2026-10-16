import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)


def compute_metrics(y_true, y_pred, y_prob):
    metrics = {
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
    }

    # ROC-AUC and AUPRC
    if len(np.unique(y_true)) == 2:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
        metrics["aupr"] = average_precision_score(y_true, y_prob)

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(
            y_true, y_prob
        )

        metrics["roc_curve"] = {
            "fpr": fpr,
            "tpr": tpr,
        }

        metrics["pr_curve"] = {
            "precision": precision,
            "recall": recall,
        }

    else:
        metrics["auc"] = np.nan
        metrics["aupr"] = np.nan
        metrics["roc_curve"] = None
        metrics["pr_curve"] = None

    return metrics