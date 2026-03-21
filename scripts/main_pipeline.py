import numpy as np
import argparse
import joblib
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.train_utils.dataset import LPBSDataset
from utils.plot_utils.presents_results import (
    plot_results,
    save_results_to_json,
    calculate_average_results,
)

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
import warnings

from models.model_lr import LogisticRegWrapper
from models.model_rocket import RocketWrapper
from models.model_rf import RandomForestWrapper
from models.model_xgboost import XGBoostWrapper
from models.model_svm import SVMWrapper
from models.model_tail_mil import TailMilClassificationWrapper


def train_models(
    models_config: dict,
    pytorch_dir="preprocessed_data/",
    augment_data=None,
    prod=False,
    scaler="standard",
):
    # Create a results dictionary to store metrics for each model
    models_results = {}
    for model_name in models_config:
        models_results[model_name] = {}

    best_overall_f1 = -1
    
    scaler_config_path = os.path.join(pytorch_dir, "scaler_config.json")

    # Load the dataset
    dataset = LPBSDataset(
        pytorch_dir, 
        scaler_type=scaler, 
        mode="train", 
        scaler_config_path=scaler_config_path
    )
    if augment_data:
        dataset.augment_data(n_augmentations_per_sample=augment_data)

    # Instantiate models and load data
    model_instances = {}

    for model_name, config in models_config.items():
        print(f"Initializing {model_name}...")
        model_cls = config["model_class"]
        params = config.get("params", {})
        model = model_cls(params)
        model_instances[model_name] = model

    gkf = GroupKFold(n_splits=5)

    for fold_idx, (worm_train_indices, worm_test_indices) in enumerate(gkf.split(dataset, groups=dataset.worm_ids)):
        print(f"\n=== Fold {fold_idx+1} ===")
        
        train_loader = DataLoader(
            Subset(dataset, indices=worm_train_indices), 
            batch_size=32, 
            shuffle=False
        )
        test_loader = DataLoader(
            Subset(dataset, indices=worm_test_indices), 
            batch_size=32, 
            shuffle=False
        )

        # Train and evaluate each model
        for model_name, model in model_instances.items():
            print(f"Training model: {model_name}")
            acc, prec, rec, f1, trained_model = model.train_on_fold(
                train_loader, test_loader
            )

            if prod:
                if f1 > best_overall_f1:
                    best_overall_f1 = f1
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
        "--augment_data", 
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
    parser.add_argument(
        "--scaler",
        "-s",
        type=str,
        default="none",
        help="Scaler type: 'none', 'minmax', 'standard'"
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
        "tail_mil_32b_64e_1e3": {
            "model_class": TailMilClassificationWrapper,
            "params": {
                "batch_size": 32,
                "embed_dim": 64,
                "lr": 1e-3,
                "patience": 50,
                "epochs": 200,
                "device": "cuda",
            }
        },
        "tail_mil_32b_32e_1e3": {
            "model_class": TailMilClassificationWrapper,
            "params": {
                "batch_size": 32,
                "embed_dim": 32,
                "lr": 1e-3,
                "patience": 50,
                "epochs": 200,
                "device": "cuda",
            }
        },
        "tail_mil_32b_32e_1e3": {
            "model_class": TailMilClassificationWrapper,
            "params": {
                "batch_size": 32,
                "embed_dim": 16,
                "lr": 1e-3,
                "patience": 50,
                "epochs": 200,
                "device": "cuda",
            }
        },
        "tail_mil_64b_32e_1e4": {
            "model_class": TailMilClassificationWrapper,
            "params": {
                "batch_size": 64,
                "embed_dim": 32,
                "lr": 1e-4,
                "patience": 50,
                "epochs": 200,
                "device": "cuda",
            }
        },
    }
    
    results = train_models(
        models_config,
        pytorch_dir=args.pytorch_dir,
        augment_data=args.augment_data,
        prod=args.prod,
        scaler=args.scaler
    )

    # Calculate average results
    avg_results = calculate_average_results(results)
    print(f"Average Results: {avg_results}")

    # Save results to JSON
    save_results_to_json(avg_results, f"{args.output_json}.json")

    # Plot results if requested
    if args.plot:
        plot_results(avg_results)
