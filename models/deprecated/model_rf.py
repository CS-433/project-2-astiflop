import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .base import BaseModel, worm_level_aggregation, compute_metrics

class RandomForestWrapper(BaseModel):
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

        rf_params = self.params.get("rf_params", {})
        threshold = self.params.get("threshold", 0.5)
        
        # Default params logic
        if "n_estimators" not in rf_params: rf_params["n_estimators"] = 100
        if "max_depth" not in rf_params: rf_params["max_depth"] = None
        if "random_state" not in rf_params: rf_params["random_state"] = 42

        clf = RandomForestClassifier(**rf_params)
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]

        worm_preds, worm_truth = worm_level_aggregation(
            worm_ids_test, y_proba, y_test, threshold
        )
        acc, prec, rec, f1 = compute_metrics(worm_truth, worm_preds)
        return acc, prec, rec, f1, clf
