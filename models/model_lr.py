import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from .base import worm_level_aggregation, compute_metrics


def LogisticRegModel(
    X_train, X_test, y_train, y_test, worm_ids, threshold=0.5, lr_params=None
):
    lr_params = (
        {
            "solver": lr_params.get("solver", "liblinear"),
            "random_state": lr_params.get("random_state", 42),
            "use_scaler": lr_params.get("use_scaler", True),
        }
        if lr_params
        else {"solver": "liblinear", "random_state": 42, "use_scaler": True}
    )

    steps = []
    if lr_params["use_scaler"]:
        steps.append(StandardScaler())
    
    # Remove use_scaler from params before passing to LogisticRegression
    lr_kwargs = {k: v for k, v in lr_params.items() if k != "use_scaler"}
    steps.append(LogisticRegression(**lr_kwargs))

    clf = make_pipeline(*steps)
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]

    worm_preds, worm_truth = worm_level_aggregation(
        worm_ids, y_proba, y_test, threshold
    )
    acc, prec, rec, f1 = compute_metrics(worm_truth, worm_preds)

    return acc, prec, rec, f1, clf
