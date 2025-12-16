from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from .base import worm_level_aggregation, compute_metrics


def SVMModel(
    X_train,
    X_test,
    y_train,
    y_test,
    worm_ids,
    threshold=0.5,
    svm_params=None,
):
    svm_params = (
        {
            "C": svm_params.get("C", 1.0),
            "kernel": svm_params.get("kernel", "rbf"),
            "probability": True,
            "random_state": svm_params.get("random_state", 42),
        }
        if svm_params
        else {"C": 1.0, "kernel": "rbf", "probability": True, "random_state": 42}
    )
    clf = make_pipeline(StandardScaler(), SVC(**svm_params))
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]

    worm_preds, worm_truth = worm_level_aggregation(
        worm_ids, y_proba, y_test, threshold
    )
    acc, prec, rec, f1 = compute_metrics(worm_truth, worm_preds)

    return acc, prec, rec, f1, clf
