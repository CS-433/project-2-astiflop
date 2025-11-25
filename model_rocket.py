import os
import numpy as np
import pandas as pd
import json
import time
from sklearn.model_selection import StratifiedKFold
from sktime.transformations.panel.rocket import (
    MiniRocketMultivariate,
    MultiRocketMultivariate,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

CONTROL = "TERBINAFINE- (control)"
TREATED = "TERBINAFINE+"
PROCESSED_DIR = "preprocessed_data/"


def load_data(data_dir):
    """Load and pad time series data for ROCKET input."""
    X, y, worm_ids = [], [], []
    for treatment in [CONTROL, TREATED]:
        treatment_dir = os.path.join(data_dir, treatment)
        for file_name in os.listdir(treatment_dir):
            if file_name.endswith(".csv"):
                file_path = os.path.join(treatment_dir, file_name)
                df = pd.read_csv(file_path)
                time_series = df[["X", "Y", "Speed"]].values
                X.append(time_series)
                y.append(treatment)
                worm_ids.append(file_name.replace(".csv", ""))
    max_length = max(len(ts) for ts in X)
    X_padded = []
    for ts in X:
        padding_length = max_length - len(ts)
        if padding_length > 0:
            padding = np.zeros((padding_length, ts.shape[1]))
            ts_padded = np.vstack([ts, padding])
        else:
            ts_padded = ts
        X_padded.append(ts_padded)
    X_array = np.array(X_padded)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return X_array, y_encoded, np.array(worm_ids)


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


def run_rocket_cv(
    X,
    y,
    worm_ids,
    use_scaler=False,
    n_splits=5,
    threshold=0.5,
    num_kernels=1000,
    variant="Mini",
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores, precisions, recalls, f1s = [], [], [], []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold_idx+1}/{n_splits}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        worm_ids_test = worm_ids[test_idx]

        steps = []
        if use_scaler:
            steps.append(PanelStandardScaler())
        if variant == "Mini":
            rocket_model = MiniRocketMultivariate(
                num_kernels=num_kernels, random_state=42
            )
            print("Using MiniRocketMultivariate")
        elif variant == "Multi":
            rocket_model = MultiRocketMultivariate(
                num_kernels=num_kernels, random_state=42
            )
            print("Using MultiRocketMultivariate")
        steps.append(rocket_model)
        steps.append(
            LogisticRegression(
                solver="liblinear", class_weight="balanced", random_state=42
            )
        )
        pipeline = make_pipeline(*steps)

        pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # Worm-level aggregation (mean probability per worm)
        results_df = pd.DataFrame(
            {"Worm_ID": worm_ids_test, "Prob_Segment": y_proba, "True_Label": y_test}
        )
        worm_results = results_df.groupby("Worm_ID").agg(
            {"Prob_Segment": "mean", "True_Label": "first"}
        )
        worm_preds = (worm_results["Prob_Segment"] > threshold).astype(int)
        worm_truth = worm_results["True_Label"]

        acc = accuracy_score(worm_truth, worm_preds)
        prec = precision_score(
            worm_truth, worm_preds, average="weighted", zero_division=0
        )
        rec = recall_score(worm_truth, worm_preds, average="weighted", zero_division=0)
        f1 = f1_score(worm_truth, worm_preds, average="weighted", zero_division=0)

        scores.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        print(f"Worm-level accuracy: {acc:.4f}")

    print("-" * 30)
    print(f"Average accuracy: {np.mean(scores):.4f}")
    print(f"Average precision: {np.mean(precisions):.4f}")
    print(f"Average recall: {np.mean(recalls):.4f}")
    print(f"Average F1-score: {np.mean(f1s):.4f}")
    return (
        np.mean(scores),
        np.std(scores),
        np.mean(precisions),
        np.std(precisions),
        np.mean(recalls),
        np.std(recalls),
        np.mean(f1s),
        np.std(f1s),
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ROCKET CV for Worm Classification")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=PROCESSED_DIR,
        help="Directory with processed CSV files",
    )
    parser.add_argument(
        "--variant", type=str, default="Mini", help="ROCKET variant, can also be Multi"
    )
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Decision threshold"
    )
    parser.add_argument(
        "--use_scaler", action="store_true", help="Use PanelStandardScaler"
    )
    parser.add_argument(
        "--num_kernels", type=int, default=1000, help="Number of Rocket kernels"
    )
    args = parser.parse_args()

    X, y, worm_ids = load_data(args.data_dir)
    X_transposed = X.transpose(0, 2, 1)
    time_start = time.time()
    (
        avg_accuracy,
        std_accuracy,
        avg_precision,
        std_precision,
        avg_recall,
        std_recall,
        avg_f1,
        std_f1,
    ) = run_rocket_cv(
        X_transposed,
        y,
        worm_ids,
        use_scaler=args.use_scaler,
        n_splits=args.n_splits,
        threshold=args.threshold,
        num_kernels=args.num_kernels,
        variant=args.variant,
    )
    time_end = time.time()
    print(f"Total CV time: {time_end - time_start:.2f} seconds")
    # Save the result in a json
    results = {
        "n_splits": args.n_splits,
        "threshold": args.threshold,
        "use_scaler": args.use_scaler,
        "num_kernels": args.num_kernels,
        "average_accuracy": avg_accuracy,
        "average_precision": avg_precision,
        "average_recall": avg_recall,
        "average_f1": avg_f1,
        "std_accuracy": std_accuracy,
        "std_precision": std_precision,
        "std_recall": std_recall,
        "std_f1": std_f1,
    }
    with open("results_rocket.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
