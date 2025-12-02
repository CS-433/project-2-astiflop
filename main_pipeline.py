import pandas as pd
import numpy as np
import os
import argparse
import json
import matplotlib.pyplot as plt
import torch

from dataset import UnifiedCElegansDataset
from fold_utils import get_stratified_worm_splits
from models.model_lr import LogisticRegModel
from models.model_rocket import RocketModel
from models.model_rf import RandomForestModel
from models.model_xgboost import XGBoostModel
from models.model_svm import SVMModel
from models.model_tail_mil import TailMilModel


def train_models(models: dict):
    # Create a results dictionary to store metrics for each model
    models_results = {}
    for model_name, _ in models.items():
        models_results[model_name] = {}

    # Load the different datasets
    dataset = UnifiedCElegansDataset(
        pytorch_dir="preprocessed_data/",
        sklearn_dir="preprocessed_data_for_classifier/",
    )
    if "rocket" in models:
        X_rocket, y_rocket, worm_ids_rocket = dataset.get_data_for_rocket()
    if any(m in models for m in ["lr", "rf", "xgboost", "svm"]):
        X_sklearn, y_sklearn, worm_ids_sklearn = dataset.get_data_for_sklearn()
    
    # Prepare data for TailMil (PyTorch)
    if "tail_mil" in models:
        worm_ids_pytorch = dataset.get_worm_ids_for_pytorch()
        worm_id_to_idx = {wid: i for i, wid in enumerate(worm_ids_pytorch)}

    # Load dataset summary for stratified splits
    summary_csv_path = "data/lifespan_summary.csv"
    folds_data = get_stratified_worm_splits(summary_csv_path, n_splits=5)

    for fold_idx, fold in enumerate(folds_data):
        print(f"Starting fold {fold_idx + 1}/{len(folds_data)}")
        worm_train_indices = fold["train"]
        worm_test_indices = fold["val"]

        # Filter Rocket Data
        if "rocket" in models:
            train_mask_rocket = np.isin(worm_ids_rocket, worm_train_indices)
            test_mask_rocket = np.isin(worm_ids_rocket, worm_test_indices)

            X_train_rocket = X_rocket[train_mask_rocket]
            y_train_rocket = y_rocket[train_mask_rocket]

            X_test_rocket = X_rocket[test_mask_rocket]
            y_test_rocket = y_rocket[test_mask_rocket]
            worm_ids_test_rocket = worm_ids_rocket[test_mask_rocket]

        # Filter Sklearn Data
        if any(m in models for m in ["lr", "rf", "xgboost", "svm"]):
            train_mask_sklearn = np.isin(worm_ids_sklearn, worm_train_indices)
            test_mask_sklearn = np.isin(worm_ids_sklearn, worm_test_indices)

            X_train_sklearn = X_sklearn[train_mask_sklearn]
            y_train_sklearn = y_sklearn[train_mask_sklearn]

            X_test_sklearn = X_sklearn[test_mask_sklearn]
            y_test_sklearn = y_sklearn[test_mask_sklearn]
            worm_ids_test_sklearn = worm_ids_sklearn[test_mask_sklearn]
        
        # Filter TailMil Data (Indices)
        if "tail_mil" in models:
            # Map worm IDs to dataset indices
            train_indices_mil = [worm_id_to_idx[wid] for wid in worm_train_indices if wid in worm_id_to_idx]
            test_indices_mil = [worm_id_to_idx[wid] for wid in worm_test_indices if wid in worm_id_to_idx]

        # Train and evaluate each model
        for model_name, model_func in models.items():
            print(f"Training model: {model_name}")
            if model_name == "rocket":
                acc, prec, rec, f1 = model_func(
                    X_train_rocket,
                    X_test_rocket,
                    y_train_rocket,
                    y_test_rocket,
                    worm_ids_test_rocket,
                )
            elif model_name in ["lr", "rf", "xgboost", "svm"]:
                acc, prec, rec, f1 = model_func(
                    X_train_sklearn,
                    X_test_sklearn,
                    y_train_sklearn,
                    y_test_sklearn,
                    worm_ids_test_sklearn,
                )
            elif model_name == "tail_mil":
                acc, prec, rec, f1 = model_func(
                    dataset,
                    train_indices_mil,
                    test_indices_mil,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    epochs=20 
                )
            
            models_results[model_name][f"fold_{fold_idx}"] = {
                "acc": acc,
                "prec": prec,
                "rec": rec,
                "f1": f1,
            }
            print(f"Results for {model_name} fold {fold_idx}: acc={acc:.4f}, f1={f1:.4f}")

    return models_results


def calculate_average_results(results):
    avg_results = {}
    for model_name, folds in results.items():
        avg_results[model_name] = {}
        for metric in ["acc", "prec", "rec", "f1"]:
            values = [f[metric] for f in folds.values()]
            avg_results[model_name][metric] = np.mean(values)
            avg_results[model_name][f"{metric}_std"] = np.std(values)
    return avg_results


def save_results_to_json(results, filename="results.json"):
    # Convert numpy types to python types for json serialization
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    with open(filename, "w") as f:
        json.dump(results, f, cls=NpEncoder, indent=4)
    print(f"Results saved to {filename}")


def plot_results(avg_results):
    metrics = ["acc", "prec", "rec", "f1"]
    model_names = list(avg_results.keys())
    
    x = np.arange(len(metrics))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, model_name in enumerate(model_names):
        means = [avg_results[model_name][m] for m in metrics]
        stds = [avg_results[model_name][f"{m}_std"] for m in metrics]
        
        offset = width * i - width * (len(model_names) - 1) / 2
        rects = ax.bar(x + offset, means, width, yerr=stds, capsize=5, label=model_name)
        ax.bar_label(rects, padding=3, fmt="%.2f")

    ax.set_ylabel('Scores')
    ax.set_title('Average Model Performance by Metric (with Error Bars)')
    ax.set_xticks(x, metrics)
    ax.legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate models.")
    parser.add_argument("--plot", action="store_true", help="Plot average results")
    args = parser.parse_args()

    # Example usage
    models_to_run = {
        "lr": LogisticRegModel,
        "rf": RandomForestModel,
        # "tail_mil": TailMilModel,
        "rocket": RocketModel, 
        # Add other models here
    }
    results = train_models(models_to_run)
    
    # Calculate average results
    avg_results = calculate_average_results(results)
    print(f"Average Results: {avg_results}")
    
    # Save results to JSON
    save_results_to_json(avg_results, "avg_results.json")

    # Plot results if requested
    if args.plot:
        plot_results(avg_results)
