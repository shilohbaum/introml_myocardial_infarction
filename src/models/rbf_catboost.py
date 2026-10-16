import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

# Put the reusable functions from your original file in
# src/models/original_stacking.py.
# Keep their implementations unchanged.

def choose_threshold(labels, probability):
    thresholds = np.unique(
        np.r_[0.0, probability, 1.0]
    )

    scores = np.asarray([
        f1_score(
            labels,
            probability >= threshold,
            zero_division=0,
        )
        for threshold in thresholds
    ])

    best = np.flatnonzero(scores == scores.max())

    index = best[
        np.argmin(np.abs(thresholds[best] - 0.5))
    ]

    return float(thresholds[index])

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



class RBFCatBoostClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        n_geometry_features,
        seed=42,
        threads=2,
    ):
        self.n_geometry_features = n_geometry_features
        self.seed = seed
        self.threads = threads

    def _split_features(self, X):
        geometry = np.asarray(
            X[:, :self.n_geometry_features],
            dtype=np.float32,
        )

        demographics = pd.DataFrame(
            X[:, self.n_geometry_features:],
            columns=self.demographic_columns_,
        )

        return geometry, demographics

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)

        if not np.array_equal(self.classes_, [0, 1]):
            raise ValueError("Expected binary labels 0 and 1")

        n_demographics = X.shape[1] - self.n_geometry_features

        if n_demographics < 1:
            raise ValueError("Expected geometry and metadata features")

        self.demographic_columns_ = [
            f"demographic_{i}"
            for i in range(n_demographics)
        ]

        geometry, demographics = self._split_features(X)

        # 1. Select RBF hyperparameters
        selection_cv = StratifiedKFold(
            5, shuffle=True, random_state=self.seed
        )

        _, c, gamma, _ = select_rbf(
            geometry, y, selection_cv, self.seed
        )

        self.best_c_ = c
        self.best_gamma_ = gamma

        # 2. Generate out-of-fold SVM probabilities
        base_cv = StratifiedKFold(
            5, shuffle=True, random_state=self.seed
        )

        meta_train, _ = stacked_features(
            geometry,
            demographics,
            y,
            base_cv,
            self.seed,
            c,
            gamma,
        )

        # 3. Select CatBoost using out-of-fold predictions
        meta_cv = StratifiedKFold(
            5, shuffle=True, random_state=self.seed
        )

        candidates = []

        for name, candidate in meta_candidates(
            self.seed, self.threads
        ):
            probabilities = cross_val_predict(
                candidate,
                meta_train,
                y,
                cv=meta_cv,
                method="predict_proba",
                n_jobs=1,
            )[:, 1]

            threshold = choose_threshold(y, probabilities)

            score = f1_score(
                y,
                probabilities >= threshold,
                zero_division=0,
            )

            candidates.append(
                (score, name, threshold, candidate)
            )

        _, name, threshold, candidate = sorted(
            candidates,
            key=lambda row: (-row[0], row[1]),
        )[0]

        self.selected_model_ = name
        self.threshold_ = threshold

        # 4. Fit final CatBoost on OOF stacking features
        self.meta_model_ = clone(candidate)
        self.meta_model_.fit(meta_train, y)

        # 5. Fit final RBF model on all training geometry
        self.base_model_ = rbf_model(
            self.seed, c, gamma
        )
        self.base_model_.fit(geometry, y)

        return self

    def predict_proba(self, X):
        check_is_fitted(
            self, ["base_model_", "meta_model_"]
        )

        geometry, demographics = self._split_features(
            np.asarray(X)
        )

        probabilities = self.base_model_.predict_proba(
            geometry
        )[:, 1]

        meta_features = demographics.copy()
        meta_features.insert(
            0, "rbf_probability", probabilities
        )

        return self.meta_model_.predict_proba(meta_features)

    def predict(self, X):
        probabilities = self.predict_proba(X)[:, 1]
        return (
            probabilities >= self.threshold_
        ).astype(int)