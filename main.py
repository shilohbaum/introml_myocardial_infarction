from sklearn.datasets import load_breast_cancer

from scripts.cross_validate import cross_validate

X, y = load_breast_cancer(return_X_y=True)

results = cross_validate(X, y, n_splits=5)