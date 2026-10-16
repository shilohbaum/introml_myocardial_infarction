import numpy as np

from sklearn.model_selection import StratifiedKFold

from src.models.model import create_model
from src.training.trainer import train_model, evaluate_model


def cross_validate(X, y, n_splits=5, n_spatial_features=540000):
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    results = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_train = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_val = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
        y_train = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
        y_val = y.iloc[val_idx] if hasattr(y, "iloc") else y[val_idx]

        model = create_model(n_spatial_features=n_spatial_features)

        model = train_model(model, X_train, y_train)
        metrics = evaluate_model(model, X_val, y_val)

        results.append(metrics)

        print(f"Fold {fold}: {metrics}")

    print("\nCross-validation results:")

    for metric in results[0]:
        values = [result[metric] for result in results]

        print(
            f"{metric}: "
            f"{np.mean(values):.4f} ± {np.std(values):.4f}"
        )

    return results