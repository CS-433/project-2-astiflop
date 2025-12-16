from sktime.transformations.panel.rocket import (
    MiniRocketMultivariate,
    MultiRocketMultivariate,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import make_pipeline
from .base import BaseModel, worm_level_aggregation, compute_metrics

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


class RocketWrapper(BaseModel):
    def load_data(self, dataset):
        self.data, self.labels, self.worm_ids = dataset.get_data_for_rocket()

    def run_fold(self, train_worm_ids, test_worm_ids):
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        train_mask = np.isin(self.worm_ids, train_worm_ids)
        test_mask = np.isin(self.worm_ids, test_worm_ids)

        X_train = self.data[train_mask]
        y_train = self.labels[train_mask]
        X_test = self.data[test_mask]
        y_test = self.labels[test_mask]

        rocket_params = self.params.get("rocket_params", {})
        threshold = self.params.get("threshold", 0.5)
        
        # Default params logic
        if "model_type" not in rocket_params: rocket_params["model_type"] = "Mini"
        if "num_kernels" not in rocket_params: rocket_params["num_kernels"] = 1000
        if "use_scaler" not in rocket_params: rocket_params["use_scaler"] = False
        if "use_logistic_regression" not in rocket_params: rocket_params["use_logistic_regression"] = True
        if "random_state" not in rocket_params: rocket_params["random_state"] = 42

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

        return acc, prec, rec, f1, pipeline
