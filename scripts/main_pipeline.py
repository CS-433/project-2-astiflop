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

from models.model_lr import LogisticRegWrapper
from models.model_rocket import RocketWrapper
from models.model_rf import RandomForestWrapper
from models.model_xgboost import XGBoostWrapper
from models.model_svm import SVMWrapper
from models.model_tail_mil import TailMilClassificationWrapper
from models.model_regression import RegressorWrapper


def train_models(
    models_config: dict,
    pytorch_dir="preprocessed_data/",
    augment_data=None,
    scaler="standard",
):
    # Create a results dictionary to store metrics for each model
    models_results = {}
    for model_name in models_config:
        models_results[model_name] = {}
    
    scaler_config_path = os.path.join(pytorch_dir, "scaler_config.json")

    # Load the dataset
    device = models_config[next(iter(models_config))]["params"]["device"]
    print(f"Loading dataset on device {device}.")
    dataset = LPBSDataset(
        pytorch_dir, 
        scaler_type=scaler, 
        mode="train", 
        scaler_config_path=scaler_config_path,
        device=device
    )
    if augment_data:
        dataset.augment_data(n_augmentations_per_sample=augment_data)

    # Instantiate models and load data
    model_instances = {}

    for model_name, config in models_config.items():
        print(f"Initializing {model_name} on device {config['params']['device']}...")
        model_cls = config["model_class"]
        params = config.get("params", {})
        model = model_cls(params)
        model_instances[model_name] = model

    gkf = GroupKFold(n_splits=5)

    for fold_idx, (worm_train_indices, worm_test_indices) in enumerate(gkf.split(dataset, groups=dataset.worm_ids)):
        print(f"\n=== Fold {fold_idx+1} ===")
        # Train and evaluate each model
        for model_name, model in model_instances.items():
            train_loader = DataLoader(
                Subset(dataset, indices=worm_train_indices), 
                batch_size=models_config[model_name]["params"]["batch_size"], 
                shuffle=True
            )
            test_loader = DataLoader(
                Subset(dataset, indices=worm_test_indices), 
                batch_size=models_config[model_name]["params"]["batch_size"], 
                shuffle=True
            )
            print(f"Training model: {model_name}")
            measures, _ = model.train_on_fold(train_loader, test_loader)

            models_results[model_name][f"fold_{fold_idx}"] = measures
            print(f"Results for {model_name} fold {fold_idx+1}: {measures}")

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
        # "tail_mil_32b_64e_1e3": {
        #     "model_class": TailMilClassificationWrapper,
        #     "params": {
        #         "batch_size": 32,
        #         "embed_dim": 64,
        #         "lr": 1e-3,
        #         "patience": 75,
        #         "epochs": 200,
        #         "device": "cuda",
        #     }
        # },
        # "tail_mil_32b_32e_1e3": {
        #     "model_class": TailMilClassificationWrapper,
        #     "params": {
        #         "batch_size": 32,
        #         "embed_dim": 32,
        #         "lr": 1e-3,
        #         "patience": 75,
        #         "epochs": 200,
        #         "device": "cuda",
        #     }
        # },
        # "tail_mil_32b_16e_1e3": {
        #     "model_class": TailMilClassificationWrapper,
        #     "params": {
        #         "batch_size": 32,
        #         "embed_dim": 16,
        #         "lr": 1e-3,
        #         "patience": 75,
        #         "epochs": 200,
        #         "device": "cuda",
        #     }
        # },
        # "tail_mil_8b_32e_1e4": {
        #     "model_class": TailMilClassificationWrapper,
        #     "params": {
        #         "batch_size": 8,
        #         "embed_dim": 32,
        #         "lr": 1e-4,
        #         "patience": 75,
        #         "epochs": 200,
        #         "device": "cuda",
        #     }
        # },
        # "Regressor": {
        #     "model_class": RegressorWrapper,
        #     "params": {
        #         "batch_size": 2,
        #         "loss": "huber",
        #         "embed_dim": 64,
        #         "lr": 1e-4,
        #         "patience": 10,
        #         "epochs": 100,
        #         "device": "cuda",
        #         "segment_len": 900,
        #     }
        # },
        "regr_128e_huber": {
            "model_class": RegressorWrapper,
            "measure_of_interest": "huber",
            "params": {
                "batch_size": 8,
                "loss": "huber",
                "embed_dim": 128,
                "lr": 5e-4,
                "patience": 25,
                "epochs": 500,
                "device": "cuda:1",
                "segment_len": 900,
            }
        },
        "regr_64e_huber": {
            "model_class": RegressorWrapper,
            "measure_of_interest": "huber",
            "params": {
                "batch_size": 8,
                "loss": "huber",
                "embed_dim": 64,
                "lr": 5e-4,
                "patience": 25,
                "epochs": 500,
                "device": "cuda:1",
                "segment_len": 900,
            }
        },
        # "regr_64e_mse": {
        #     "model_class": RegressorWrapper,
        #     "measure_of_interest": "mse",
        #     "params": {
        #         "batch_size": 8,
        #         "loss": "mse",
        #         "embed_dim": 64,
        #         "lr": 1e-4,
        #         "patience": 25,
        #         "epochs": 500,
        #         "device": "cuda:1",
        #         "segment_len": 900,
        #     }
        # },
    }
    
    results = train_models(
        models_config,
        pytorch_dir=args.pytorch_dir,
        augment_data=args.augment_data,
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
