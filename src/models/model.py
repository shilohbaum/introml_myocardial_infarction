from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


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