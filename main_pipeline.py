import numpy as np
import argparse

from dataset import UnifiedCElegansAugmentedDataset, UnifiedCElegansDataset
from fold_utils import get_stratified_worm_splits
from models.model_lr import LogisticRegModel
from models.model_rocket import RocketModel
from models.model_rf import RandomForestModel
from models.model_xgboost import XGBoostModel
from models.model_svm import SVMModel
from models.model_tail_mil import TailMilModel
from presents_results import (
    plot_results,
    save_results_to_json,
    calculate_average_results,
)


def train_models(
    models: dict,
    model_params: dict = None,
    pytorch_dir="preprocessed_data/",
    use_augmented_data=False,
):
    # Create a results dictionary to store metrics for each model
    models_results = {}
    for model_name, _ in models.items():
        models_results[model_name] = {}

    # Load the different datasets
    dataset = UnifiedCElegansDataset(
        pytorch_dir=pytorch_dir,
        sklearn_dir="preprocessed_data_for_classifier/",
    ) if not use_augmented_data else UnifiedCElegansAugmentedDataset(
        pytorch_dir=pytorch_dir,
        sklearn_dir="preprocessed_data_for_classifier/",
    )
    if any(m.startswith("rocket") for m in models):
        X_rocket, y_rocket, worm_ids_rocket = dataset.get_data_for_rocket()
    if any(m in models for m in ["logReg", "rf", "xgboost", "svm"]):
        X_sklearn, y_sklearn, worm_ids_sklearn = dataset.get_data_for_sklearn()

    # Prepare data for TailMil (PyTorch)
    if any(m.startswith("tail_mil") for m in models):
        worm_ids_pytorch = dataset.get_worm_ids_for_pytorch()
        # Handle 1-to-many mapping (for augmented dataset)
        worm_id_to_indices = {}
        for idx, wid in enumerate(worm_ids_pytorch):
            if wid not in worm_id_to_indices:
                worm_id_to_indices[wid] = []
            worm_id_to_indices[wid].append(idx)

    # Load dataset summary for stratified splits
    summary_csv_path = "data/lifespan_summary.csv"
    folds_data = get_stratified_worm_splits(summary_csv_path, n_splits=5)

    for fold_idx, fold in enumerate(folds_data):
        print("-" * 40)
        print(f"Starting fold {fold_idx + 1}/{len(folds_data)}")
        worm_train_indices = fold["train"]
        worm_test_indices = fold["val"]

        # Filter Rocket Data
        if any(m.startswith("rocket") for m in models):
            train_mask_rocket = np.isin(worm_ids_rocket, worm_train_indices)
            test_mask_rocket = np.isin(worm_ids_rocket, worm_test_indices)

            X_train_rocket = X_rocket[train_mask_rocket]
            y_train_rocket = y_rocket[train_mask_rocket]

            X_test_rocket = X_rocket[test_mask_rocket]
            y_test_rocket = y_rocket[test_mask_rocket]
            worm_ids_test_rocket = worm_ids_rocket[test_mask_rocket]

        # Filter Sklearn Data
        if any(m in models for m in ["logReg", "rf", "xgboost", "svm"]):
            train_mask_sklearn = np.isin(worm_ids_sklearn, worm_train_indices)
            test_mask_sklearn = np.isin(worm_ids_sklearn, worm_test_indices)

            X_train_sklearn = X_sklearn[train_mask_sklearn]
            y_train_sklearn = y_sklearn[train_mask_sklearn]

            X_test_sklearn = X_sklearn[test_mask_sklearn]
            y_test_sklearn = y_sklearn[test_mask_sklearn]
            worm_ids_test_sklearn = worm_ids_sklearn[test_mask_sklearn]

        # Filter TailMil Data (Indices)
        if any(m.startswith("tail_mil") for m in models):
            # Map worm IDs to dataset indices
            train_indices_mil = []
            for wid in worm_train_indices:
                if wid in worm_id_to_indices:
                    train_indices_mil.extend(worm_id_to_indices[wid])

            test_indices_mil = []
            for wid in worm_test_indices:
                if wid in worm_id_to_indices:
                    test_indices_mil.extend(worm_id_to_indices[wid])

        # Train and evaluate each model
        for model_name, model_func in models.items():
            print(f"Training model: {model_name}")
            params = model_params.get(model_name, {}) if model_params else {}
            if model_name.startswith("rocket"):
                acc, prec, rec, f1 = model_func(
                    X_train_rocket,
                    X_test_rocket,
                    y_train_rocket,
                    y_test_rocket,
                    worm_ids_test_rocket,
                    **params,
                )
            elif model_name in ["logReg", "rf", "xgboost", "svm"]:
                acc, prec, rec, f1 = model_func(
                    X_train_sklearn,
                    X_test_sklearn,
                    y_train_sklearn,
                    y_test_sklearn,
                    worm_ids_test_sklearn,
                    **params,
                )
            elif model_name.startswith("tail_mil"):
                acc, prec, rec, f1 = model_func(
                    dataset, train_indices_mil, test_indices_mil, **params
                )
            else:
                raise ValueError(f"Unknown model name: {model_name}")

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
    parser.add_argument("--plot", action="store_true", help="Plot average results")
    parser.add_argument("--pytorch_dir", "-d", type=str, default="preprocessed_data/", help="Path to PyTorch preprocessed data directory")
    parser.add_argument("--augmented_data", "-a", action="store_true", help="Use augmented data for training")
    parser.add_argument("--output_json", "-o", type=str, default="avg_results", help="Output JSON file for average results")
    args = parser.parse_args()

    # Example usage
    models_to_run = {
        "logReg": LogisticRegModel,
        # "xgboost": XGBoostModel,
        "rocket_500": RocketModel,
        "tail_mil": TailMilModel,
        
        # "rocket_500": RocketModel,
        
        # "tail_mil_32e": TailMilModel,
        # "tail_mil_32b_32e": TailMilModel,
        # "tail_mil_32b": TailMilModel,
        # "tail_mil_32b_1-3lr": TailMilModel,
        # "tail_mil_64b": TailMilModel,
        
        # "rf": RandomForestModel,
        # "svm": SVMModel,
    }
    model_params = {
        # "rf": {"rf_params": {"n_estimators": 500, "max_depth": 5, "random_state": 42}},
        # "rocket_1000": {"rocket_params": {"num_kernels": 1000}},
        # "rocket_500": {"rocket_params": {"num_kernels": 500}},
        
        
        # "tail_mil_32e": {"batch_size": 8, "lr": 5e-4, "embed_dim": 32, "patience": 15}, # 0.69
        # "tail_mil_32b_32e": {"batch_size": 32, "lr": 5e-4, "embed_dim": 32, "patience": 15}, # 0.69
        # "tail_mil_32b": {"batch_size": 32, "lr": 5e-4, "embed_dim": 8, "patience": 15, "device": "cuda:2"}, # 0.72
        # "tail_mil_32b_1-3lr": {"batch_size": 32, "lr": 1e-3, "embed_dim": 8, "patience": 15, "device": "cuda:2"},
        # "tail_mil_64b": {"batch_size": 64, "lr": 5e-4, "embed_dim": 8, "patience": 15, "device": "cuda:2"}, 
        
        "logReg": {"lr_params": {"max_iter": 1000}},
        # "xgboost": {"xgb_params": {"n_estimators": 500, "max_depth": 5, "use_label_encoder": False, "eval_metric": "logloss"}},
        "rocket_500": {"rocket_params": {"num_kernels": 500}},
        "tail_mil": {"batch_size": 32, "lr": 1e-3, "embed_dim": 8, "patience": 15, "device": "cuda:1"},
    }
    results = train_models(
        models_to_run,
        model_params,
        pytorch_dir=args.pytorch_dir,
        use_augmented_data=args.augmented_data,
    )

    # Calculate average results
    avg_results = calculate_average_results(results)
    print(f"Average Results: {avg_results}")

    # Save results to JSON
    save_results_to_json(avg_results, f"{args.output_json}.json")

    # Plot results if requested
    if args.plot:
        plot_results(avg_results)
