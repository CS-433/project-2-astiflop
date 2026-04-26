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

# from models.model_lr import LogisticRegWrapper
# from models.model_rocket import RocketWrapper
# from models.model_rf import RandomForestWrapper
# from models.model_xgboost import XGBoostWrapper
# from models.model_svm import SVMWrapper
# from models.model_tail_mil import TailMilClassificationWrapper
from models.model_regression import RegressorTrainingWrapper


def train_models(
    models_config: dict,
    pytorch_dir="preprocessed_data/",
    augment_data=None,
    scaler="standard",
    n_splits=5,
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

    gkf = GroupKFold(n_splits=n_splits)
    for fold_idx, (worm_train_indices, worm_test_indices) in enumerate(gkf.split(dataset, groups=dataset.worm_ids)):
        print(f"\n=== Fold {fold_idx+1} ===")
        
        # Instantiate fresh models for each fold
        model_instances = {}
        for model_name, config in models_config.items():
            model_cls = config["model_class"]
            params = config.get("params", {})
            model = model_cls(params)
            model_instances[model_name] = model
        
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

    # models_config = {
    #     "regr_64e_3_1_5e4": {
    #         "model_class": RegressorTrainingWrapper,
    #         "params": {
    #             "name": "regr_64e_bs16_3_1",
    #             "embed_dim": 64,
    #             "feature_extractor_layers": 3,
    #             "bilstm_layers": 1,
    #             "batch_size": 16,
    #             "loss": "huber",                
    #             "lr": 5e-4,
    #             "patience": 25,
    #             "epochs": 500,
    #             "device": "cuda",
    #             "segment_len": 900,
    #         }
    #     },
    # }

    models_config = {
        "hmm_16_t": {
            "model_class": RegressorTrainingWrapper,
            "params": {
                "name": "hmm_regr_64e_bs16_fel3_ns16_t", 
                "model_type": "hmm",              
                "embed_dim": 64,
                "feature_extractor_layers": 3,
                "num_states": 16,                 
                "batch_size": 16,
                "loss": "huber",                
                "lr": 5e-4,
                "patience": 25,
                "epochs": 500,
                # "device": "cpu",
                "device": "cuda:1",
                "segment_len": 900,
            }
        },
        "hmm_64_t": {
            "model_class": RegressorTrainingWrapper,
            "params": {
                "name": "hmm_regr_64e_bs16_fel3_ns64_t", 
                "model_type": "hmm",              
                "embed_dim": 64,
                "feature_extractor_layers": 3,
                "num_states": 64,                 
                "batch_size": 16,
                "loss": "huber",                
                "lr": 5e-4,
                "patience": 25,
                "epochs": 500,
                # "device": "cpu",
                "device": "cuda:1",
                "segment_len": 900,
            }
        },
        "hmm_64_no": {
            "model_class": RegressorTrainingWrapper,
            "params": {
                "name": "hmm_regr_64e_bs16_fel3_ns64_no", 
                "model_type": "hmm",              
                "embed_dim": 64,
                "feature_extractor_layers": 3,
                "num_states": 64,  
                "use_time_encoding": False,              
                "batch_size": 16,
                "loss": "huber",                
                "lr": 5e-4,
                "patience": 25,
                "epochs": 500,
                # "device": "cpu",
                "device": "cuda:1",
                "segment_len": 900,
            }
        },
        "bilstm_t": {
            "model_class": RegressorTrainingWrapper,
            "params": {
                "name": "bilstm_regr_64e_bs16_fel3_t", 
                "model_type": "bilstm",              
                "embed_dim": 64,
                "feature_extractor_layers": 3,
                "num_states": 64,     
                "use_time_encoding": True,           
                "batch_size": 16,
                "loss": "huber",                
                "lr": 5e-4,
                "patience": 25,
                "epochs": 500,
                # "device": "cpu",
                "device": "cuda:1",
                "segment_len": 900,
            }
        },
        "bilstm_no": {
            "model_class": RegressorTrainingWrapper,
            "params": {
                "name": "bilstm_regr_64e_bs16_fel3_no", 
                "model_type": "bilstm",              
                "embed_dim": 64,
                "feature_extractor_layers": 3,
                "num_states": 64,     
                "use_time_encoding": False,           
                "batch_size": 16,
                "loss": "huber",                
                "lr": 5e-4,
                "patience": 25,
                "epochs": 500,
                # "device": "cpu",
                "device": "cuda:1",
                "segment_len": 900,
            }
        },
    }
    
    results = train_models(
        models_config,
        pytorch_dir=args.pytorch_dir,
        augment_data=args.augment_data,
        scaler=args.scaler,
        n_splits=2,
    )

    # Calculate average results
    avg_results = calculate_average_results(results)
    print(f"Average Results: {avg_results}")

    # Save results to JSON
    save_results_to_json(avg_results, f"{args.output_json}.json")

    # Plot results if requested
    if args.plot:
        plot_results(avg_results)
