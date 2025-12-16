import numpy as np
import argparse
import joblib
import torch
import sys
import os

# Add the parent directory to sys.path to allow imports from utils and models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.train_utils.dataset import (
    UnifiedCElegansAugmentedDataset,
    UnifiedCElegansDataset,
)
from utils.train_utils.fold_utils import get_stratified_worm_splits
from models.model_lr import LogisticRegWrapper
from models.model_rocket import RocketWrapper
from models.model_rf import RandomForestWrapper
from models.model_xgboost import XGBoostWrapper
from models.model_svm import SVMWrapper
from models.model_tail_mil import TailMilWrapper, ScaledDataset
from utils.plot_utils.presents_results import (
    plot_results,
    save_results_to_json,
    calculate_average_results,
)
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
import warnings


def train_models(
    models_config: dict,
    pytorch_dir="preprocessed_data/",
    augmented_data=None,
    prod=False,
):
    # Create a results dictionary to store metrics for each model
    models_results = {}
    for model_name in models_config:
        models_results[model_name] = {}

    best_overall_f1 = -1
    best_overall_model_name = ""
    best_overall_model_instance = None

    # Load the different datasets
    dataset = (
        UnifiedCElegansDataset(
            pytorch_dir=pytorch_dir,
            sklearn_dir="preprocessed_data_for_classifier/",
        )
        if not augmented_data
        else UnifiedCElegansAugmentedDataset(
            pytorch_dir=pytorch_dir,
            sklearn_dir="preprocessed_data_for_classifier/",
            augmentations_per_sample=augmented_data,
        )
    )

    # Instantiate models and load data
    model_instances = {}
    for model_name, config in models_config.items():
        print(f"Initializing {model_name}...")
        model_cls = config["model_class"]
        params = config.get("params", {})
        model = model_cls(params)
        model.load_data(dataset)
        model_instances[model_name] = model

    # Load dataset summary for stratified splits
    summary_csv_path = "data/lifespan_summary.csv"
    folds_data = get_stratified_worm_splits(summary_csv_path, n_splits=5)

    for fold_idx, fold in enumerate(folds_data):
        print("-" * 40)
        print(f"Starting fold {fold_idx + 1}/{len(folds_data)}")
        worm_train_indices = fold["train"]
        worm_test_indices = fold["val"]

        # Train and evaluate each model
        for model_name, model in model_instances.items():
            print(f"Training model: {model_name}")
            acc, prec, rec, f1, trained_model = model.run_fold(
                worm_train_indices, worm_test_indices
            )

            if prod:
                if f1 > best_overall_f1:
                    best_overall_f1 = f1
                    best_overall_model_name = model_name
                    best_overall_model_instance = trained_model
                    print(f"-> New best model found: {model_name} (F1={f1:.4f})")
                    if isinstance(trained_model, torch.nn.Module):
                        torch.save(trained_model.state_dict(), "best_model.pth")
                    else:
                        joblib.dump(trained_model, "best_model.pkl")

            models_results[model_name][f"fold_{fold_idx}"] = {
                "acc": acc,
                "prec": prec,
                "rec": rec,
                "f1": f1,
            }
            print(
                f"Results for {model_name} fold {fold_idx+1}: acc={acc:.4f}, f1={f1:.4f}"
            )

    # Sanity Check for TailMil
    if (
        prod
        and best_overall_model_name.startswith("tail_mil")
        and best_overall_model_instance is not None
    ):
        print("\n=== Performing Sanity Check on Best TailMil Model ===")

        # Load non-augmented dataset
        sanity_dataset = UnifiedCElegansDataset(
            pytorch_dir=pytorch_dir,
            sklearn_dir="preprocessed_data_for_classifier/",
        )

        model = best_overall_model_instance
        device = next(model.parameters()).device

        if hasattr(model, "mean") and model.mean is not None:
            print("Applying scaling from training...")
            sanity_dataset = ScaledDataset(sanity_dataset, model.mean, model.std)

        sanity_loader = DataLoader(sanity_dataset, batch_size=32, shuffle=False)

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X, y in sanity_loader:
                X = X.to(device)
                preds, _, _ = model(X)
                preds = preds.squeeze().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        preds_binary = (all_preds > 0.5).astype(int)

        # Metrics
        acc = accuracy_score(all_labels, preds_binary)
        f1 = f1_score(all_labels, preds_binary)
        print(f"Sanity Check Results (Non-Augmented Data): Acc={acc:.4f}, F1={f1:.4f}")

        # Catastrophic Failure Check
        unique_preds = np.unique(preds_binary)
        if len(unique_preds) == 1:
            warnings.warn(
                f"Catastrophic Failure Detected: Model predicts only class {unique_preds[0]}!"
            )
            print(
                f"WARNING: Catastrophic Failure Detected: Model predicts only class {unique_preds[0]}!"
            )
        else:
            print("Sanity Check Passed: Model predicts both classes.")

    return models_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate models.")
    parser.add_argument(
        "--plot", 
        action="store_true", 
        help="Plot average results",
    )
    parser.add_argument(
        "--pytorch_dir",
        "-d",
        type=str,
        default="preprocessed_data/",
        help="Path to PyTorch preprocessed data directory",
    )
    parser.add_argument(
        "--augmented_data", 
        "-a", 
        nargs='?', 
        const=5, 
        type=int, 
        default=None, 
        help="Use augmented data for training. If specified without value, defaults to 5.",
    )
    parser.add_argument(
        "--output_json",
        "-o",
        type=str,
        default="avg_results",
        help="Output JSON file for average results",
    )
    parser.add_argument(
        "--prod", action="store_true", help="Run in production mode (save best model)"
    )
    args = parser.parse_args()

    # Example usage
    models_config = {
        # "logReg": {
        #     "model_class": LogisticRegWrapper,
        #     "params": {"lr_params": {"max_iter": 1000, "use_scaler": True}}
        # },
        # "rocket_500": {
        #     "model_class": RocketWrapper,
        #     "params": {"rocket_params": {"num_kernels": 500, "use_scaler": True}}
        # },
        "tail_mil_32b_16e_1e4": {
            "model_class": TailMilWrapper,
            "params": {
                "batch_size": 32,
                "embed_dim": 16,
                "lr": 1e-4,
                "patience": 30,
                "use_scaler": True,
                "device": "cuda:1",
            }
        },
        "tail_mil_64b_16e_1e4": {
            "model_class": TailMilWrapper,
            "params": {
                "batch_size": 64,
                "embed_dim": 16,
                "lr": 1e-4,
                "patience": 30,
                "use_scaler": True,
                "device": "cuda:1",
            }
        },
    }
    
    results = train_models(
        models_config,
        pytorch_dir=args.pytorch_dir,
        augmented_data=args.augmented_data,
        prod=args.prod,
    )

    # Calculate average results
    avg_results = calculate_average_results(results)
    print(f"Average Results: {avg_results}")

    # Save results to JSON
    save_results_to_json(avg_results, f"{args.output_json}.json")

    # Plot results if requested
    if args.plot:
        plot_results(avg_results)
