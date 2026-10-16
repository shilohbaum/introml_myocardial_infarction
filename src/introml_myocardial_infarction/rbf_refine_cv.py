"""Nested-CV refinement around the strongest geometric RBF SVM settings."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .dataloader import CardiacDataLoader
from .geometry_features import feature_matrix, labels_for, choose_threshold


def candidates(seed):
    for c in (10.0, 30.0, 100.0, 300.0):
        for gamma in (0.0015, 0.002, 0.0025, 0.004, 0.006):
            name = f"rbf_c{c:g}_g{gamma:g}"
            estimator = make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(),
                CalibratedClassifierCV(
                    SVC(C=c, gamma=gamma, kernel="rbf", random_state=seed),
                    method="sigmoid", cv=3, ensemble=False,
                ),
            )
            yield name, estimator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/training_data")
    parser.add_argument("--samples-dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty")
    loader = CardiacDataLoader(args.data_dir, samples_dir=args.samples_dir)
    labels = labels_for(loader)
    features = feature_matrix(loader)
    outer = StratifiedKFold(5, shuffle=True, random_state=42)
    outer_folds = list(outer.split(features, labels))
    rows, nested = [], np.full(len(labels), np.nan)
    for fold, (train, validation) in enumerate(outer_folds, 1):
        inner = StratifiedKFold(4, shuffle=True, random_state=42 + fold)
        choices = []
        for name, estimator in candidates(42 + fold):
            probability = cross_val_predict(estimator, features.iloc[train], labels[train], cv=inner,
                                            method="predict_proba", n_jobs=1)[:, 1]
            threshold = choose_threshold(labels[train], probability)
            score = f1_score(labels[train], probability >= threshold, zero_division=0)
            choices.append((score, name, threshold, estimator))
        _, name, threshold, estimator = sorted(choices, key=lambda item: (-item[0], item[1]))[0]
        estimator.fit(features.iloc[train], labels[train])
        probability = estimator.predict_proba(features.iloc[validation])[:, 1]
        nested[validation] = probability
        rows.append(dict(fold=fold, selected_model=name, threshold=float(threshold),
                         f1=float(f1_score(labels[validation], probability >= threshold, zero_division=0)),
                         f1_at_0_5=float(f1_score(labels[validation], probability >= 0.5, zero_division=0)),
                         auroc=float(roc_auc_score(labels[validation], probability))))
        print(f"Refined RBF fold {fold}/5: {name}, F1={rows[-1]['f1']:.4f}", flush=True)

    # Choose the final estimator and operating threshold from full-training OOF.
    selection_folds = StratifiedKFold(5, shuffle=True, random_state=42)
    final_choices = []
    for name, estimator in candidates(42):
        probability = cross_val_predict(estimator, features, labels, cv=selection_folds,
                                        method="predict_proba", n_jobs=1)[:, 1]
        threshold = choose_threshold(labels, probability)
        score = f1_score(labels, probability >= threshold, zero_division=0)
        final_choices.append((score, name, threshold, estimator, probability))
        print(f"Full OOF {name}: F1={score:.4f}", flush=True)
    score, name, threshold, estimator, selected_oof = sorted(
        final_choices, key=lambda item: (-item[0], item[1]),
    )[0]
    estimator.fit(features, labels)
    output.mkdir(parents=True, exist_ok=True)
    ids = loader.demographics.index.to_numpy()
    pd.DataFrame(rows).to_csv(output / "nested_cv_folds.csv", index=False)
    pd.DataFrame(dict(sample_id=ids, label=labels, probability=nested)).to_csv(
        output / "nested_oof_predictions.csv", index=False)
    pd.DataFrame(dict(sample_id=ids, label=labels, probability=selected_oof,
                      prediction=(selected_oof >= threshold).astype(int))).to_csv(
        output / "selected_oof_predictions.csv", index=False)
    summary = dict(representation="702 geometric shape/motion summaries",
                   candidate_count=20, n_samples=len(labels), n_features=features.shape[1],
                   n_splits=5, seed=42,
                   nested_mean_f1=float(np.mean([row["f1"] for row in rows])),
                   nested_sd_f1=float(np.std([row["f1"] for row in rows], ddof=1)),
                   nested_mean_f1_at_0_5=float(np.mean([row["f1_at_0_5"] for row in rows])),
                   nested_mean_auroc=float(np.mean([row["auroc"] for row in rows])),
                   selected_model=name, selected_threshold=float(threshold),
                   selected_oof_f1=float(score),
                   note="Nested mean F1 estimates refinement; selected full OOF F1 is optimistic.")
    (output / "cv_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    joblib.dump(dict(model=estimator, threshold=threshold, feature_columns=list(features.columns),
                     train_ids=list(map(str, ids)), selected_model=name),
                output / "model.joblib", compress=3)
    (output / "folds.json").write_text(json.dumps([
        dict(fold=i, train=list(map(str, ids[tr])), validation=list(map(str, ids[va])))
        for i, (tr, va) in enumerate(outer_folds, 1)
    ], indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
