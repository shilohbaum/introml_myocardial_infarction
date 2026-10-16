"""Nested stacking: RBF geometry scores and demographics into CatBoost."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .dataloader import CardiacDataLoader
from .geometry_features import feature_matrix, choose_threshold, labels_for


def rbf_model(seed, c, gamma):
    """RBF base learner, calibrated only on its current fit partition."""
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        CalibratedClassifierCV(
            SVC(C=c, gamma=gamma, kernel="rbf", random_state=seed),
            method="sigmoid", cv=3, ensemble=False,
        ),
    )


def rbf_candidates(seed):
    for c, gamma in ((10.0, 0.001), (10.0, 0.003), (10.0, 0.01),
                     (30.0, 0.002), (30.0, 0.004),
                     (100.0, 0.001), (100.0, 0.002), (100.0, 0.003)):
        yield c, gamma, rbf_model(seed, c, gamma)


def select_rbf(geometry, labels, folds, seed):
    """Choose the base kernel using only the supplied training partition."""
    choices = []
    for c, gamma, model in rbf_candidates(seed):
        probability = cross_val_predict(
            model, geometry, labels, cv=folds, method="predict_proba", n_jobs=1,
        )[:, 1]
        threshold = choose_threshold(labels, probability)
        score = f1_score(labels, probability >= threshold, zero_division=0)
        choices.append((score, c, gamma, threshold))
    return sorted(choices, key=lambda row: (-row[0], row[1], row[2]))[0]


def meta_candidates(seed, threads):
    for depth in (2, 3):
        for iterations in (100, 250):
            name = f"catboost_depth{depth}_iterations{iterations}"
            model = CatBoostClassifier(
                iterations=iterations, depth=depth, learning_rate=0.03,
                l2_leaf_reg=10.0, random_strength=1.5, rsm=1.0,
                loss_function="Logloss", eval_metric="Logloss",
                random_seed=seed, thread_count=threads,
                allow_writing_files=False, verbose=False,
            )
            yield name, model


def demographic_matrix(loader):
    frame = loader.demographics.drop(columns="MI").select_dtypes(
        include=["number", "bool"],
    ).astype(float)
    if not frame.index.equals(loader.demographics.index):
        raise RuntimeError("Demographic row order changed")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Expected complete finite numeric demographic fields")
    frame = frame.reset_index(drop=True)
    frame.columns = [f"demographic_{column}" for column in frame.columns]
    return frame


def stacked_features(geometry, demographics, labels, folds, seed, c, gamma):
    probability = cross_val_predict(
        rbf_model(seed, c, gamma), geometry, labels, cv=folds,
        method="predict_proba", n_jobs=1,
    )[:, 1]
    result = demographics.copy()
    result.insert(0, "rbf_probability", probability)
    return result, probability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/training_data")
    parser.add_argument("--samples-dir")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty")
    if args.threads < 1:
        raise ValueError("Thread count must be positive")
    loader = CardiacDataLoader(args.data_dir, samples_dir=args.samples_dir)
    labels = labels_for(loader)
    geometry = feature_matrix(loader).to_numpy(dtype=np.float32)
    demographics = demographic_matrix(loader)
    outer = StratifiedKFold(5, shuffle=True, random_state=42)
    assignments = list(outer.split(geometry, labels))
    nested = np.full(len(labels), np.nan)
    fold_rows = []

    with threadpool_limits(limits=args.threads):
        for fold, (train, validation) in enumerate(assignments, 1):
            y_train = labels[train]
            base_selection_folds = StratifiedKFold(4, shuffle=True, random_state=500 + fold)
            base_score, base_c, base_gamma, base_threshold = select_rbf(
                geometry[train], y_train, base_selection_folds, 100 + fold,
            )
            # Cross-fit the base model so every meta-training probability comes
            # from an RBF model that did not train on that patient's label.
            base_folds = StratifiedKFold(5, shuffle=True, random_state=1000 + fold)
            meta_train, _ = stacked_features(
                geometry[train], demographics.iloc[train].reset_index(drop=True),
                y_train, base_folds, 42 + fold, base_c, base_gamma,
            )
            fitted_base = rbf_model(42 + fold, base_c, base_gamma).fit(geometry[train], y_train)
            validation_probability = fitted_base.predict_proba(geometry[validation])[:, 1]
            meta_validation = demographics.iloc[validation].reset_index(drop=True).copy()
            meta_validation.insert(0, "rbf_probability", validation_probability)

            inner = StratifiedKFold(4, shuffle=True, random_state=42 + fold)
            choices = []
            for name, model in meta_candidates(42 + fold, args.threads):
                probability = cross_val_predict(
                    model, meta_train, y_train, cv=inner,
                    method="predict_proba", n_jobs=1,
                )[:, 1]
                threshold = choose_threshold(y_train, probability)
                score = f1_score(y_train, probability >= threshold, zero_division=0)
                choices.append((score, name, threshold, model))
            _, name, threshold, model = sorted(
                choices, key=lambda row: (-row[0], row[1]),
            )[0]
            model.fit(meta_train, y_train)
            probability = model.predict_proba(meta_validation)[:, 1]
            nested[validation] = probability
            fold_rows.append(dict(
                fold=fold, base_c=base_c, base_gamma=base_gamma,
                base_selection_f1=float(base_score), base_threshold=float(base_threshold),
                selected_model=name, threshold=float(threshold),
                inner_selection_f1=float(max(row[0] for row in choices)),
                f1=float(f1_score(labels[validation], probability >= threshold,
                                  zero_division=0)),
                f1_at_0_5=float(f1_score(labels[validation], probability >= 0.5,
                                         zero_division=0)),
                auroc=float(roc_auc_score(labels[validation], probability)),
            ))
            print(f"RBF+demographics stack fold {fold}/5: {name}, "
                  f"F1={fold_rows[-1]['f1']:.4f}", flush=True)

        # Full-cohort base OOF scores and meta-model selection for a saved model.
        full_base_selection = StratifiedKFold(5, shuffle=True, random_state=42)
        base_score, base_c, base_gamma, base_threshold = select_rbf(
            geometry, labels, full_base_selection, 42,
        )
        full_base_folds = StratifiedKFold(5, shuffle=True, random_state=42)
        full_meta, full_base_oof = stacked_features(
            geometry, demographics, labels, full_base_folds, 42, base_c, base_gamma,
        )
        select_folds = StratifiedKFold(5, shuffle=True, random_state=42)
        selected = []
        for name, model in meta_candidates(42, args.threads):
            probability = cross_val_predict(
                model, full_meta, labels, cv=select_folds,
                method="predict_proba", n_jobs=1,
            )[:, 1]
            threshold = choose_threshold(labels, probability)
            score = f1_score(labels, probability >= threshold, zero_division=0)
            selected.append((score, name, threshold, model, probability))
            print(f"Stack full OOF {name}: F1={score:.4f}", flush=True)
        score, name, threshold, model, selected_oof = sorted(
            selected, key=lambda row: (-row[0], row[1]),
        )[0]
        model.fit(full_meta, labels)
        final_base = rbf_model(42, base_c, base_gamma).fit(geometry, labels)

    output.mkdir(parents=True, exist_ok=True)
    ids = loader.demographics.index.to_numpy()
    pd.DataFrame(fold_rows).to_csv(output / "nested_cv_folds.csv", index=False)
    pd.DataFrame(dict(sample_id=ids, label=labels, probability=nested)).to_csv(
        output / "nested_oof_predictions.csv", index=False)
    meta_csv = full_meta.copy()
    meta_csv.insert(0, "sample_id", ids)
    meta_csv.insert(1, "label", labels)
    meta_csv.to_csv(output / "full_training_meta_features.csv", index=False)
    pd.DataFrame(dict(sample_id=ids, label=labels, rbf_probability=full_base_oof,
                      stacked_probability=selected_oof,
                      prediction=(selected_oof >= threshold).astype(int))).to_csv(
        output / "selected_oof_predictions.csv", index=False)
    summary = dict(
        representation="cross-fitted RBF geometric probability + seven numeric demographics",
        base_rbf_selected_within_outer_training=True,
        n_samples=len(labels), n_meta_features=full_meta.shape[1],
        demographic_features=list(demographics.columns), candidate_count=4,
        n_splits=5, seed=42,
        nested_mean_f1=float(np.mean([row["f1"] for row in fold_rows])),
        nested_sd_f1=float(np.std([row["f1"] for row in fold_rows], ddof=1)),
        nested_mean_f1_at_0_5=float(np.mean([row["f1_at_0_5"] for row in fold_rows])),
        nested_mean_auroc=float(np.mean([row["auroc"] for row in fold_rows])),
        final_base_rbf_c=float(base_c), final_base_rbf_gamma=float(base_gamma),
        final_base_rbf_selection_f1=float(base_score),
        selected_model=name, selected_threshold=float(threshold),
        selected_oof_f1=float(score),
        note="Outer folds evaluate stacking. Full-training OOF selection score is optimistic.",
    )
    (output / "cv_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    joblib.dump(dict(base_model=final_base, meta_model=model, threshold=threshold,
                     base_c=base_c, base_gamma=base_gamma,
                     demographic_columns=list(demographics.columns),
                     train_ids=list(map(str, ids)), selected_model=name),
                output / "model.joblib", compress=3)
    (output / "folds.json").write_text(json.dumps([
        dict(fold=i, train=list(map(str, ids[tr])), validation=list(map(str, ids[va])))
        for i, (tr, va) in enumerate(assignments, 1)
    ], indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
