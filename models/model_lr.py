import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from .base import worm_level_aggregation, compute_metrics


def LogisticRegModel(
    X_train, X_test, y_train, y_test, worm_ids, threshold=0.5, lr_params=None
):
    lr_params = lr_params or {"solver": "liblinear", "random_state": 42}

    clf = make_pipeline(StandardScaler(), LogisticRegression(**lr_params))
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]

    worm_preds, worm_truth = worm_level_aggregation(
        worm_ids, y_proba, y_test, threshold
    )
    acc, prec, rec, f1 = compute_metrics(worm_truth, worm_preds)

    return acc, prec, rec, f1
