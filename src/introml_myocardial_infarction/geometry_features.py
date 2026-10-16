"""Cross-validated MI models from compact cardiac shape and motion summaries."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .dataloader import CardiacDataLoader


def labels_for(loader):
    """Return validated binary MI labels in loader row order."""
    if "MI" not in loader.demographics:
        raise ValueError("MI labels are required for training and evaluation")
    labels = loader.demographics["MI"].map({"healthy": 0, "pMI": 1})
    if labels.isna().any():
        raise ValueError("MI labels must be healthy or pMI with no missing values")
    return labels.to_numpy(dtype=np.int64)


def _summary(values, robust=False):
    """Finite, low-dimensional distribution summary for a point cloud scalar."""
    if robust:
        return np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
    return np.quantile(values, [0.1, 0.5, 0.9]).tolist() + [
        float(np.mean(values)), float(np.std(values)), float(np.max(values)),
    ]


def describe_cloud(cloud, robust_motion=False):
    """Summarize each surface's evolving shape and pointwise movement.

    Per-phase descriptors preserve relative surface placement and shape after
    removing the whole-heart centroid. Motion descriptors use corresponding
    point indices: displacement from frame zero and each adjacent-frame step.
    The model therefore avoids a learned 180k-dimensional trajectory PCA while
    retaining cycle timing, per-surface expansion, and displacement ranges.
    """
    if cloud.ndim != 3 or cloud.shape[-1] not in (3, 4) or cloud.shape[0] < 2:
        raise ValueError("Expected (frames, points, 3 or 4) cardiac point cloud")
    xyz = np.asarray(cloud[..., :3], dtype=np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError("Point cloud contains nonfinite coordinates")
    codes = np.asarray(cloud[0, :, 3]) if cloud.shape[-1] == 4 else np.zeros(cloud.shape[1])
    if cloud.shape[-1] == 4 and not np.all(cloud[..., 3] == codes[None, :]):
        raise ValueError("Surface codes must be stable for corresponding points across frames")
    surfaces = np.unique(codes)
    if len(surfaces) != 3:
        raise ValueError(f"Expected three point-cloud surfaces; found {len(surfaces)}")

    features = []
    frame_centroids = xyz.mean(axis=1, keepdims=True)
    centered = xyz - frame_centroids
    for surface in surfaces:
        mask = codes == surface
        if mask.sum() < 3:
            raise ValueError(f"Surface {surface} has fewer than three points")
        points = centered[:, mask, :]
        surface_centroids = points.mean(axis=1)
        relative_centroids = surface_centroids - surface_centroids[:1]
        for frame in range(xyz.shape[0]):
            block = points[frame]
            covariance = np.cov(block, rowvar=False)
            eigenvalues = np.linalg.eigvalsh(covariance)
            features.extend(surface_centroids[frame].tolist())
            features.extend(block.std(axis=0).tolist())
            features.extend(eigenvalues.tolist())
            features.extend(relative_centroids[frame].tolist())
        # Per-point motion summaries retain amplitude and the timing of each
        # phase transition, separately for LV endocardium, LV epicardium, RV.
        displacement = np.linalg.norm(points - points[:1], axis=-1)
        step_distance = np.linalg.norm(np.diff(points, axis=0), axis=-1)
        for values in (displacement, step_distance):
            for phase_values in values:
                features.extend(_summary(phase_values, robust=robust_motion))
    result = np.asarray(features, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Geometric summary features contain nonfinite values")
    return result


def feature_matrix(data, robust_motion=False):
    rows = []

    for index, (_, row) in enumerate(data.iterrows()):
        xyz = np.asarray(row["point_cloud"])
        surface_codes = np.asarray(row["surface_codes"])

        # Expected xyz: (frames, points, 3)
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(f"Invalid point cloud shape: {xyz.shape}")

        # Handle surface codes stored per point or per frame
        if surface_codes.ndim == 1:
            surface_codes = np.broadcast_to(
                surface_codes[None, :],
                xyz.shape[:2],
            )

        if surface_codes.shape != xyz.shape[:2]:
            raise ValueError(
                f"Surface codes shape {surface_codes.shape} "
                f"does not match {xyz.shape[:2]}"
            )

        # Combine XYZ coordinates and surface codes
        cloud = np.concatenate(
            [xyz, surface_codes[..., None]],
            axis=-1,
        )

        rows.append(
            describe_cloud(
                cloud,
                robust_motion=robust_motion,
            )
        )

        if (index + 1) % 100 == 0 or index + 1 == len(data):
            print(
                f"Extracted geometric summaries for "
                f"{index + 1}/{len(data)} patients",
                flush=True,
            )

    matrix = np.stack(rows)

    return pd.DataFrame(
        matrix,
        index=data.index,
        columns=[
            f"geometry_{i:04d}"
            for i in range(matrix.shape[1])
        ],
    )


def candidates():
    models = []
    for c in (0.001, 0.01, 0.1, 1.0):
        models.append((f"logistic_c{c:g}", make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
            LogisticRegression(C=c, max_iter=3000),
        )))
    for c in (0.1, 1.0, 10.0):
        models.append((f"rbf_svc_c{c:g}", make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
            CalibratedClassifierCV(
                SVC(C=c, kernel="rbf", random_state=42),
                method="sigmoid", cv=3, ensemble=False,
            ),
        )))
    return models


def choose_threshold(labels, probability):
    thresholds = np.unique(np.r_[0.0, probability, 1.0])
    scores = np.asarray([
        f1_score(labels, probability >= threshold, zero_division=0)
        for threshold in thresholds
    ])
    best = np.flatnonzero(scores == scores.max())
    index = best[np.argmin(np.abs(thresholds[best] - 0.5))]
    return float(thresholds[index])


def nested_evaluate(features, labels, n_splits, seed):
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(labels), np.nan)
    fold_rows = []
    for fold, (train, validation) in enumerate(outer.split(features, labels), 1):
        x_train, y_train = features.iloc[train], labels[train]
        inner = StratifiedKFold(n_splits=n_splits - 1, shuffle=True, random_state=seed + fold)
        options = []
        for name, estimator in candidates():
            probabilities = cross_val_predict(estimator, x_train, y_train, cv=inner, method="predict_proba")[:, 1]
            threshold = choose_threshold(y_train, probabilities)
            score = f1_score(y_train, probabilities >= threshold, zero_division=0)
            options.append((score, name, threshold, estimator))
        _, name, threshold, estimator = sorted(options, key=lambda item: (-item[0], item[1]))[0]
        estimator.fit(x_train, y_train)
        probability = estimator.predict_proba(features.iloc[validation])[:, 1]
        oof[validation] = probability
        fold_rows.append(dict(
            fold=fold, model=name, threshold=threshold,
            f1=float(f1_score(labels[validation], probability >= threshold, zero_division=0)),
            auroc=float(roc_auc_score(labels[validation], probability)),
        ))
        print(f"Completed geometric nested CV fold {fold}/{n_splits}: {name}", flush=True)
    return fold_rows, oof


def run(args):
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output is not empty; choose a new directory")
    loader = CardiacDataLoader(args.data_dir, samples_dir=args.samples_dir)
    labels = labels_for(loader)
    features = feature_matrix(loader)
    folds, nested_oof = nested_evaluate(features, labels, args.n_splits, args.seed)

    # Select the deployable estimator and threshold from training OOF scores.
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    selected = []
    for name, estimator in candidates():
        probability = cross_val_predict(estimator, features, labels, cv=cv, method="predict_proba")[:, 1]
        threshold = choose_threshold(labels, probability)
        score = f1_score(labels, probability >= threshold, zero_division=0)
        selected.append((score, name, threshold, estimator, probability))
    score, name, threshold, estimator, oof = sorted(selected, key=lambda item: (-item[0], item[1]))[0]
    estimator.fit(features, labels)

    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(folds).to_csv(output / "nested_cv_folds.csv", index=False)
    ids = loader.demographics.index.to_numpy()
    pd.DataFrame(dict(sample_id=ids, label=labels, probability=nested_oof)).to_csv(
        output / "nested_oof_predictions.csv", index=False,
    )
    pd.DataFrame(dict(sample_id=ids, label=labels, probability=oof,
                      prediction=(oof >= threshold).astype(int))).to_csv(output / "selected_oof_predictions.csv", index=False)
    summary = dict(
        representation="per-surface phase shape and pointwise motion summaries",
        n_samples=len(labels), n_features=features.shape[1], n_splits=args.n_splits,
        seed=args.seed, nested_cv_mean_f1=float(np.mean([row["f1"] for row in folds])),
        nested_cv_std_f1=float(np.std([row["f1"] for row in folds], ddof=1)),
        nested_cv_mean_auroc=float(np.mean([row["auroc"] for row in folds])),
        selected_model=name, selected_threshold=threshold,
        selected_oof_f1=float(score),
        note="Nested CV estimates model and threshold selection. Selected OOF F1 is optimistic.",
    )
    (output / "cv_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    joblib.dump(dict(model=estimator, threshold=threshold, feature_columns=list(features.columns),
                     train_ids=list(map(str, ids)), representation=summary["representation"]),
                output / "model.joblib", compress=3)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def predict(args):
    state = joblib.load(args.checkpoint)
    loader = CardiacDataLoader(args.data_dir, samples_dir=args.samples_dir)
    overlap = set(map(str, loader.demographics.index)) & set(state["train_ids"])
    if overlap:
        raise ValueError(f"Prediction patients overlap training IDs: {sorted(overlap)}")
    features = feature_matrix(loader).reindex(columns=state["feature_columns"])
    probability = state["model"].predict_proba(features)[:, 1]
    predictions = pd.DataFrame(dict(
        sample_id=loader.demographics.index,
        probability=probability,
        prediction=(probability >= state["threshold"]).astype(int),
    ))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    print(f"Wrote {len(predictions)} geometric predictions to {output}", flush=True)


def evaluate(args):
    """Evaluate a frozen checkpoint once on explicitly supplied labelled data."""
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output is not empty; choose a new directory")
    checkpoint = Path(args.checkpoint)
    state = joblib.load(checkpoint)
    loader = CardiacDataLoader(args.data_dir, samples_dir=args.samples_dir)
    overlap = set(map(str, loader.demographics.index)) & set(state["train_ids"])
    if overlap:
        raise ValueError(f"Evaluation patients overlap training IDs: {sorted(overlap)}")
    labels = labels_for(loader)
    features = feature_matrix(loader).reindex(columns=state["feature_columns"])
    probability = state["model"].predict_proba(features)[:, 1]
    threshold = float(state["threshold"])
    prediction = probability >= threshold
    result = pd.DataFrame(dict(
        sample_id=loader.demographics.index, label=labels,
        probability=probability, prediction=prediction.astype(int),
    ))
    scores = dict(
        n_samples=int(len(labels)), threshold=threshold,
        accuracy=float(accuracy_score(labels, prediction)),
        precision=float(precision_score(labels, prediction, zero_division=0)),
        recall=float(recall_score(labels, prediction, zero_division=0)),
        f1=float(f1_score(labels, prediction, zero_division=0)),
        auroc=float(roc_auc_score(labels, probability)),
    )
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "predictions.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(scores, indent=2) + "\n")
    provenance = dict(
        checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        data_dir=str(loader.data_dir.resolve()), samples_dir=str(loader.samples_dir.resolve()),
        n_training_patients=len(state["train_ids"]), n_evaluation_patients=len(loader),
        training_overlap=False,
        test_cloud_subject_mapping="assumed from CSV row positions; independent manifest unavailable",
        threshold_source="selected using training-only out-of-fold predictions",
    )
    (output / "evaluation_config.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(scores, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cv = commands.add_parser("cross-validate")
    cv.add_argument("--data-dir", default="data/training_data")
    cv.add_argument("--samples-dir")
    cv.add_argument("--n-splits", type=int, default=5)
    cv.add_argument("--seed", type=int, default=42)
    cv.add_argument("--output", required=True)
    cv.set_defaults(func=run)
    inference = commands.add_parser("predict")
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--data-dir", required=True)
    inference.add_argument("--samples-dir")
    inference.add_argument("--output", required=True)
    inference.set_defaults(func=predict)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--data-dir", required=True)
    evaluation.add_argument("--samples-dir")
    evaluation.add_argument("--output", required=True)
    evaluation.set_defaults(func=evaluate)
    args = parser.parse_args()
    if args.command == "cross-validate" and args.n_splits < 3:
        parser.error("Nested CV requires at least three folds")
    args.func(args)


if __name__ == "__main__":
    main()
