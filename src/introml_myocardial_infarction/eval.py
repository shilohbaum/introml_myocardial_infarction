
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score



def get_cv(pipeline, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    reports = []

    for i, (train_index, test_index) in enumerate(cv.split(X_train, y_train)):

        pipeline.fit(X_train[train_index], y_train[train_index])

        y_pred = pipeline.predict(X_train[test_index])
        y_proba = pipeline.predict_proba(X_train[test_index])

        report = classification_report(y_train[test_index], y_pred, output_dict=True)
        reports.append(report)

    return reports



def get_cv_f1(pipeline, X_train, y_train):
    reports = get_cv(pipeline, X_train, y_train)

    return sum([report['weighted avg']['f1-score'] for report in reports]) / len([report['weighted avg']['f1-score'] for report in reports])





