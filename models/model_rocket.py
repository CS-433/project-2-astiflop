import numpy as np
from sklearn.model_selection import StratifiedKFold
from sktime.transformations.panel.rocket import (
    MiniRocketMultivariate,
    MultiRocketMultivariate,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import make_pipeline
from .base import worm_level_aggregation, compute_metrics

from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class PanelStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        n_instances, n_channels, n_timepoints = X.shape
        X_reshaped = X.reshape(n_instances, -1)
        self.scaler.fit(X_reshaped)
        return self

    def transform(self, X):
        n_instances, n_channels, n_timepoints = X.shape
        X_reshaped = X.reshape(n_instances, -1)
        X_scaled = self.scaler.transform(X_reshaped)
        return X_scaled.reshape(n_instances, n_channels, n_timepoints)


def RocketModel(
    X_train,
    X_test,
    y_train,
    y_test,
    worm_ids,
    threshold=0.5,
    rocket_params=None,
):
    rocket_params = (
        {
            "model_type": rocket_params.get("model_type", "Mini"),
            "num_kernels": rocket_params.get("num_kernels", 1000),
            "use_scaler": rocket_params.get("use_scaler", False),
            "use_logistic_regression": rocket_params.get(
                "use_logistic_regression", True
            ),
            "random_state": rocket_params.get("random_state", 42),
        }
        if rocket_params
        else {
            "model_type": "Mini",
            "num_kernels": 1000,
            "use_scaler": False,
            "use_logistic_regression": True,
            "random_state": 42,
        }
    )
    steps = []
    if rocket_params["use_scaler"]:
        steps.append(PanelStandardScaler())
    if rocket_params["model_type"] == "Mini":
        rocket_model = MiniRocketMultivariate(
            num_kernels=rocket_params["num_kernels"],
            random_state=rocket_params["random_state"],
        )
        print("Using MiniRocketMultivariate")
    elif rocket_params["model_type"] == "Multi":
        rocket_model = MultiRocketMultivariate(
            num_kernels=rocket_params["num_kernels"],
            random_state=rocket_params["random_state"],
        )
        print("Using MultiRocketMultivariate")
    steps.append(rocket_model)
    if rocket_params["use_logistic_regression"]:
        steps.append(
            LogisticRegression(
                solver="liblinear",
                random_state=rocket_params["random_state"],
            )
        )
    else:
        steps.append(
            RidgeClassifier(
                random_state=rocket_params["random_state"],
            )
        )
    pipeline = make_pipeline(*steps)

    pipeline.fit(X_train, y_train)
    if rocket_params["use_logistic_regression"]:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_pred = pipeline.predict(X_test)

    acc, prec, rec, f1 = compute_metrics(y_test, y_pred)

    return acc, prec, rec, f1
