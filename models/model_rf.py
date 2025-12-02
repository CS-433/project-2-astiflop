import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from .base import worm_level_aggregation, compute_metrics


def RandomForestModel(
    X_train, X_test, y_train, y_test, worm_ids, threshold=0.5, rf_params=None
):
    rf_params = (
        {
            "n_estimators": rf_params.get("n_estimators", 100),
            "max_depth": rf_params.get("max_depth", None),
            "random_state": rf_params.get("random_state", 42),
        }
        if rf_params
        else {
            "n_estimators": 100,
            "max_depth": None,
            "random_state": 42,
        }
    )
    clf = RandomForestClassifier(**rf_params)
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]

    worm_preds, worm_truth = worm_level_aggregation(
        worm_ids, y_proba, y_test, threshold
    )
    acc, prec, rec, f1 = compute_metrics(worm_truth, worm_preds)
    return acc, prec, rec, f1
