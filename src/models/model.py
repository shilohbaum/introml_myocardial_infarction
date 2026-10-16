from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier



def create_model(n_pca_components=50, n_spatial_features=540000):
    preprocessor = ColumnTransformer([
        (
            "spatial",
            Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(
                    n_components=n_pca_components,
                    svd_solver="randomized"
                )),
            ]),
            slice(0, n_spatial_features),
        ),
        (
            "metadata",
            StandardScaler(),
            slice(n_spatial_features, None),
        ),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=5000)),
    ])


def create_random_forest(
    n_pca_components=50,
    n_spatial_features=540000,
    n_estimators=200,
    max_depth=7,
):
    preprocessor = ColumnTransformer([
        (
            "spatial",
            Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(
                    n_components=n_pca_components,
                    svd_solver="randomized",
                    random_state=42,
                )),
            ]),
            slice(0, n_spatial_features),
        ),
        (
            "metadata",
            "passthrough",
            slice(n_spatial_features, None),
        ),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )),
    ])


from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def create_catboost(
    n_pca_components=50,
    n_spatial_features=540000,
    iterations=500,
    depth=5,
    learning_rate=0.03,
):
    preprocessor = ColumnTransformer([
        (
            "spatial",
            Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(
                    n_components=n_pca_components,
                    svd_solver="randomized",
                    random_state=42,
                )),
            ]),
            slice(0, n_spatial_features),
        ),
        (
            "metadata",
            "passthrough",
            slice(n_spatial_features, None),
        ),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=False,
            thread_count=-1,
        )),
    ])


from .rbf_catboost import RBFCatBoostClassifier


def create_rbf_catboost(
    n_geometry_features,
    seed=42,
    threads=2,
):
    return RBFCatBoostClassifier(
        n_geometry_features=n_geometry_features,
        seed=seed,
        threads=threads,
    )