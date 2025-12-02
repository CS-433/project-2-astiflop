import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold
from .base import worm_level_aggregation, compute_metrics


def XGBoostModel(X, y, worm_ids, n_splits=5, threshold=0.5, xgb_params=None):
    xgb_params = xgb_params or {
        "n_estimators": 100,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
    }
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores, precisions, recalls, f1s = [], [], [], []
    for train_idx, test_idx in cv.split(X, y, groups=worm_ids):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        worm_ids_test = worm_ids[test_idx]

        clf = XGBClassifier(**xgb_params)
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]

        worm_preds, worm_truth = worm_level_aggregation(
            worm_ids_test, y_proba, y_test, threshold
        )
        acc, prec, rec, f1 = compute_metrics(worm_truth, worm_preds)
        scores.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    return np.mean(scores), np.mean(precisions), np.mean(recalls), np.mean(f1s)
