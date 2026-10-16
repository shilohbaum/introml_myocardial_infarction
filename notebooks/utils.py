
import numpy as np
import matplotlib.pyplot as plt

from src.training.trainer import evaluate_model
from IPython.display import display
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from src.models.model import (
    create_model,
    create_random_forest,
    create_catboost,
)
from src.training.trainer import train_model, evaluate_model
import pandas as pd

from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
)




def plot_model_results(model, X_test, y_test, model_name="Model"):
    test_metrics = evaluate_model(model, X_test, y_test)

    # Print scalar metrics
    print(f"\n{model_name} - Test Results:")
    for metric, value in test_metrics.items():
        if np.isscalar(value) and value is not None:
            print(f"{metric}: {value:.4f}")

    # Plot curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curve
    roc = test_metrics["roc_curve"]

    if roc is not None:
        axes[0].plot(
            roc["fpr"],
            roc["tpr"],
            label=f'AUC = {test_metrics["auc"]:.3f}',
            linewidth=2,
        )

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"{model_name} - ROC Curve",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Precision-Recall curve
    pr = test_metrics["pr_curve"]

    if pr is not None:
        axes[1].plot(
            pr["recall"],
            pr["precision"],
            label=f'AP = {test_metrics["aupr"]:.3f}',
            linewidth=2,
        )

    axes[1].axhline(
        np.mean(np.asarray(y_test) == 1),
        color="black",
        linestyle="--",
        alpha=0.5,
        label="Baseline",
    )
    axes[1].set(
        xlabel="Recall",
        ylabel="Precision",
        title=f"{model_name} - Precision-Recall Curve",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return test_metrics


def tune_model(
    create_fn,
    param_distributions,
    X_train,
    y_train,
    metadata_cols,
    n_iter=20,
    n_splits=5,
    n_pca_components=50,
    scoring="roc_auc",
    random_state=42,
):
    n_spatial_features = X_train.shape[1] - len(metadata_cols)

    model = create_fn(
        n_pca_components=n_pca_components,
        n_spatial_features=n_spatial_features,
    )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=1,
        verbose=2,
        random_state=random_state,
        refit=True,
        return_train_score=True,
    )

    search.fit(X_train, y_train)

    results = pd.DataFrame(search.cv_results_).sort_values(
        "rank_test_score"
    )

    print("\nBest CV score:", round(search.best_score_, 4))
    print("\nBest parameters:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")

    display(
        results[
            [
                "mean_test_score",
                "std_test_score",
                "mean_train_score",
                "params",
            ]
        ].head(10)
    )

    return search.best_estimator_, search, results


def run_experiments(
    X_train,
    y_train,
    metadata_cols,
    model_factories,
    X_test=None,
    y_test=None,
    mode="cv",
    n_splits=5,
    n_pca_components=50,
):
    if mode not in ("cv", "test"):
        raise ValueError("mode must be 'cv' or 'test'")

    if mode == "test" and (X_test is None or y_test is None):
        raise ValueError("Test mode requires X_test and y_test")

    n_spatial_features = X_train.shape[1] - len(metadata_cols)

    trained_models = {}
    results = {}

    scalar_metrics = [
        "accuracy", "precision", "recall",
        "f1", "auc", "aupr",
    ]

    for name, create_fn in model_factories.items():
        print(f"\n{'=' * 50}")
        print(f"{name} | {mode.upper()}")
        print(f"{'=' * 50}")

        def new_model():
            return create_fn(
                n_pca_components=n_pca_components,
                n_spatial_features=n_spatial_features,
            )

        if mode == "cv":
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=42,
            )

            fold_results = []

            for fold, (train_idx, val_idx) in enumerate(
                cv.split(X_train, y_train), start=1
            ):
                model = new_model()

                model = train_model(
                    model,
                    X_train[train_idx],
                    y_train[train_idx],
                )

                metrics = evaluate_model(
                    model,
                    X_train[val_idx],
                    y_train[val_idx],
                )

                fold_results.append(metrics)

                print(
                    f"Fold {fold}: "
                    f"AUC={metrics['auc']:.4f}, "
                    f"AUPR={metrics['aupr']:.4f}"
                )

            results[name] = fold_results

        else:
            model = new_model()

            model = train_model(
                model,
                X_train,
                y_train,
            )

            metrics = plot_model_results(
                model,
                X_test,
                y_test,
                model_name=name,
            )

            trained_models[name] = model
            results[name] = metrics

    # Create summary table
    summary = {}

    for name, metrics in results.items():
        if mode == "cv":
            summary[name] = {
                metric: np.nanmean(
                    [fold[metric] for fold in metrics]
                )
                for metric in scalar_metrics
            }
        else:
            summary[name] = {
                metric: metrics[metric]
                for metric in scalar_metrics
            }

    summary = pd.DataFrame(summary).T

    display(summary.round(4))

    return trained_models, results, summary