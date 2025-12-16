import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import os
import argparse
from tqdm import tqdm

# Local imports
from utils.train_utils.dataset import CElegansCNNDataset
from models.model_cnn import get_cnn_model, train_epoch, validate
from utils.plot_utils.present_results import plot_cnn_comparison


def run_cnn_pipeline(data_dir, models_config, n_splits=5, img_size=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- TRANSFORMS ---
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # --- DATASET & FOLDS ---
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        return

    dataset_full = CElegansCNNDataset(data_dir, transform=None)

    if len(dataset_full) == 0:
        print("Error: No images found.")
        return

    indices, labels, groups = dataset_full.get_indices_labels_groups()
    sgkf = StratifiedGroupKFold(n_splits=n_splits)

    results_summary = {}

    for model_name, params in models_config.items():
        print(f"\n{'#'*40}")
        print(f"Running Model: {model_name}")
        print(f"Params: {params}")
        print(f"{'#'*40}")

        batch_size = params.get("batch_size", 32)
        lr = params.get("lr", 1e-4)
        epochs = params.get("epochs", 15)
        arch = params.get("architecture", "resnet18")

        all_folds_results = {"acc": [], "prec": [], "rec": [], "f1": []}

        for fold, (train_idx, val_idx) in enumerate(
            sgkf.split(indices, labels, groups=groups)
        ):
            print(f"\n>>> FOLD {fold+1}/{n_splits} - {model_name}")

            # Helper dynamic transform
            class TransformedSubset(Dataset):
                def __init__(self, subset, transform=None):
                    self.subset = subset
                    self.transform = transform

                def __getitem__(self, index):
                    img, label, worm_id = self.subset[index]
                    if self.transform:
                        img = self.transform(img)
                    return img, label, worm_id

                def __len__(self):
                    return len(self.subset)

            train_ds = TransformedSubset(
                torch.utils.data.Subset(dataset_full, train_idx),
                transform=train_transform,
            )
            val_ds = TransformedSubset(
                torch.utils.data.Subset(dataset_full, val_idx), transform=val_transform
            )

            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, num_workers=2
            )
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False, num_workers=2
            )

            model = get_cnn_model(model_name=arch, pretrained=True).to(device)
            criterion = nn.BCELoss()
            optimizer = optim.AdamW(model.parameters(), lr=lr)

            best_fold_f1 = 0.0
            best_fold_metrics = {"acc": 0, "prec": 0, "rec": 0, "f1": 0}

            for epoch in range(epochs):
                train_loss = train_epoch(
                    model, train_loader, criterion, optimizer, device
                )
                val_metrics = validate(model, val_loader, device)

                print(
                    f"Epoch {epoch+1}: Loss={train_loss:.4f} | Acc={val_metrics['acc']:.3f}, F1={val_metrics['f1']:.3f}"
                )

                if val_metrics["f1"] > best_fold_f1:
                    best_fold_f1 = val_metrics["f1"]
                    best_fold_metrics = val_metrics.copy()

            print(f"--> Best Result Fold {fold+1} ({model_name}): {best_fold_metrics}")

            for k in all_folds_results:
                all_folds_results[k].append(best_fold_metrics[k])

        # --- Average Results for this Model ---
        model_avg_results = {}
        for k in ["acc", "prec", "rec", "f1"]:
            model_avg_results[f"{k}_mean"] = np.mean(all_folds_results[k])
            model_avg_results[f"{k}_std"] = np.std(all_folds_results[k])

        results_summary[model_name] = model_avg_results

    # --- FINAL REPORT ---
    print(f"\n{'='*20} FINAL SUMMARY {'='*20}")
    print(f"{'Model':<20} | {'F1 Score (Mean ± Std)':<25}")
    print("-" * 50)

    for model_name, res in results_summary.items():
        f1_mean = res["f1_mean"]
        f1_std = res["f1_std"]
        print(f"{model_name:<20} | {f1_mean:.4f} ± {f1_std:.4f}")

    # --- PLOTTING ---
    plot_cnn_comparison(results_summary)

    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN Pipeline for C. Elegans")
    parser.add_argument(
        "--data_dir", type=str, default="/cnn_dataset", help="Path to dataset"
    )
    args = parser.parse_args()

    # --- MODELS CONFIGURATION ---
    # Define models and their specific parameters here, similar to main_pipeline
    models_config = {
        "resnet18_baseline": {
            "architecture": "resnet18",
            "batch_size": 32,
            "lr": 1e-4,
            "epochs": 15,
        },
        # "resnet50_baseline": {
        #     "architecture": "resnet50",
        #     "batch_size": 32,
        #     "lr": 1e-4,
        #     "epochs": 15,
        # },
        # "densenet_baseline": {
        #     "architecture": "densenet121",
        #     "batch_size": 16, # Smaller batch size for larger model
        #     "lr": 1e-4,
        #     "epochs": 15
        # }
    }

    run_cnn_pipeline(args.data_dir, models_config, n_splits=5)
