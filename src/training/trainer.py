from .metrics import compute_metrics


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_val, y_val):
    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)[:, 1]

    return compute_metrics(y_val, predictions, probabilities)