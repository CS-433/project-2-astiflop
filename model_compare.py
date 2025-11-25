import os
import pandas as pd
import numpy as np
import json
import time
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

CONTROL = "TERBINAFINE- (control)"
TREATED = "TERBINAFINE+"
CLASSIFIER_DIR = "preprocessed_data_for_classifier/"


def load_data_for_classifier(classifier_dir):
    dfs = []
    for file_name in os.listdir(classifier_dir):
        if file_name.endswith(".csv"):
            df = pd.read_csv(os.path.join(classifier_dir, file_name))
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def run_cv_model(
    df, model_cls, model_params=None, n_splits=5, threshold=0.5, verbose=True
):
    feature_cols = [
        "Age_hours",
        "Mean_Speed",
        "Median_Speed",
        "Net_Displacement",
        "Tortuosity",
    ]
    X = df[feature_cols]
    y = df["Terbinafine"]
    groups = df["Worm_ID"]

    cv = StratifiedGroupKFold(n_splits=n_splits)
    model_params = model_params or {}

    # Use StandardScaler for SVM and LogisticRegression
    if model_cls in [LogisticRegression, SVC]:
        clf = make_pipeline(StandardScaler(), model_cls(**model_params))
    else:
        clf = model_cls(**model_params)

    scores, precisions, recalls, f1s = [], [], [], []
    feature_importances = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf.fit(X_train, y_train)
        # Use predict_proba if available, else decision_function, else predict
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_test)[:, 1]
            y_pred = (probs > threshold).astype(int)
        elif hasattr(clf, "decision_function"):
            probs = clf.decision_function(X_test)
            y_pred = (probs > 0).astype(int)
        else:
            y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        scores.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        # Feature importances if available
        # For pipeline, get feature_importances_ or coef_ from last step
        if hasattr(clf, "named_steps"):
            last_step = clf.named_steps[list(clf.named_steps.keys())[-1]]
            if hasattr(last_step, "feature_importances_"):
                feature_importances.append(last_step.feature_importances_)
            elif hasattr(last_step, "coef_"):
                feature_importances.append(np.abs(last_step.coef_[0]))
            else:
                feature_importances.append([np.nan] * len(feature_cols))
        else:
            if hasattr(clf, "feature_importances_"):
                feature_importances.append(clf.feature_importances_)
            elif hasattr(clf, "coef_"):
                feature_importances.append(np.abs(clf.coef_[0]))
            else:
                feature_importances.append([np.nan] * len(feature_cols))

        if verbose:
            print(f"Fold {fold_idx+1} finished. Accuracy: {acc:.4f}")

    mean_importances = np.nanmean(feature_importances, axis=0)
    sorted_idx = np.argsort(mean_importances)[::-1]

    print("-" * 30)
    print(f"Model: {model_cls.__name__}")
    print(f"Average accuracy: {np.mean(scores):.4f}")
    print(f"Average precision: {np.mean(precisions):.4f}")
    print(f"Average recall: {np.mean(recalls):.4f}")
    print(f"Average F1-score: {np.mean(f1s):.4f}")
    print("Feature importances (mean over folds):")
    for idx in sorted_idx:
        print(f"{feature_cols[idx]}: {mean_importances[idx]:.4f}")
    print("-" * 30)
    return {
        "scores": scores,
        "precisions": precisions,
        "recalls": recalls,
        "f1s": f1s,
        "mean_accuracy": float(np.mean(scores)),
        "std_accuracy": float(np.std(scores)),
        "mean_precision": float(np.mean(precisions)),
        "std_precision": float(np.std(precisions)),
        "mean_recall": float(np.mean(recalls)),
        "std_recall": float(np.std(recalls)),
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
        "feature_importances": mean_importances.tolist(),
        "params": model_params,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare ML models for Worm Classification"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=CLASSIFIER_DIR,
        help="Directory with classifier CSV files",
    )
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for proba models",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["all", "rf", "xgb", "lr", "svm"],
        default="all",
        help="Which model to run",
    )
    parser.add_argument(
        "--rf_estimators", type=int, default=1000, help="RandomForest n_estimators"
    )
    parser.add_argument(
        "--xgb_estimators", type=int, default=1000, help="XGBoost n_estimators"
    )
    parser.add_argument(
        "--svm_c", type=float, default=1.0, help="SVM regularization parameter"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="results_compare.json",
        help="Output JSON file for metrics",
    )
    args = parser.parse_args()

    df = load_data_for_classifier(args.data_dir)

    models = {
        "rf": (
            RandomForestClassifier,
            {"n_estimators": args.rf_estimators, "random_state": 42},
        ),
        "lr": (LogisticRegression, {"solver": "liblinear", "random_state": 42}),
        "svm": (
            SVC,
            {
                "C": args.svm_c,
                "kernel": "linear",
                "probability": True,
                "random_state": 42,
            },
        ),
        "xgb": (
            XGBClassifier,
            {
                "n_estimators": args.xgb_estimators,
                "use_label_encoder": False,
                "eval_metric": "logloss",
                "random_state": 42,
            },
        ),
    }

    results_dict = {}
    run_params = {
        "data_dir": args.data_dir,
        "n_splits": args.n_splits,
        "threshold": args.threshold,
        "rf_estimators": args.rf_estimators,
        "xgb_estimators": args.xgb_estimators,
        "svm_c": args.svm_c,
        "model": args.model,
    }
    time_start = time.time()
    if args.model == "all":
        for name, (cls, params) in models.items():
            print(f"\nRunning model: {name.upper()}")
            res = run_cv_model(
                df,
                cls,
                params,
                n_splits=args.n_splits,
                threshold=args.threshold,
                verbose=True,
            )
            results_dict[name] = res
    else:
        if args.model not in models:
            print(f"Model {args.model} not available.")
            return
        cls, params = models[args.model]
        print(f"\nRunning model: {args.model.upper()}")
        res = run_cv_model(
            df,
            cls,
            params,
            n_splits=args.n_splits,
            threshold=args.threshold,
            verbose=True,
        )
        results_dict[args.model] = res
    time_end = time.time()
    print(f"\nTotal time taken: {time_end - time_start:.2f} seconds")

    # Save results and parameters to JSON. One JSON per model, each with parameters, avg metrics and standard deviations
    output = {"run_params": run_params, "results": results_dict}
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {args.output_json}")


if __name__ == "__main__":
    main()
