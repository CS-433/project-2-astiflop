import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from .base import BaseModel, worm_level_aggregation, compute_metrics

class LogisticRegWrapper(BaseModel):
    def load_data(self, dataset):
        self.data, self.labels, self.worm_ids = dataset.get_data_for_sklearn()

    def run_fold(self, train_worm_ids, test_worm_ids):
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        train_mask = np.isin(self.worm_ids, train_worm_ids)
        test_mask = np.isin(self.worm_ids, test_worm_ids)

        X_train = self.data[train_mask]
        y_train = self.labels[train_mask]
        X_test = self.data[test_mask]
        y_test = self.labels[test_mask]
        worm_ids_test = self.worm_ids[test_mask]

        lr_params = self.params.get("lr_params", {})
        threshold = self.params.get("threshold", 0.5)
        
        # Default params logic
        solver = lr_params.get("solver", "liblinear")
        random_state = lr_params.get("random_state", 42)
        use_scaler = lr_params.get("use_scaler", True)
        
        steps = []
        if use_scaler:
            steps.append(StandardScaler())

        # Filter out custom params
        lr_kwargs = {k: v for k, v in lr_params.items() if k not in ["use_scaler"]}
        # Ensure defaults are set if not present
        if "solver" not in lr_kwargs: lr_kwargs["solver"] = solver
        if "random_state" not in lr_kwargs: lr_kwargs["random_state"] = random_state

        steps.append(LogisticRegression(**lr_kwargs))

        clf = make_pipeline(*steps)
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]

        worm_preds, worm_truth = worm_level_aggregation(
            worm_ids_test, y_proba, y_test, threshold
        )
        acc, prec, rec, f1 = compute_metrics(worm_truth, worm_preds)

        return acc, prec, rec, f1, clf
