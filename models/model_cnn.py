import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import os
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt  # <--- AJOUT POUR LE PLOT
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

# ==========================================
# 1. Dataset Class
# ==========================================


class CElegansCNNDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        self.class_map = {"TERBINAFINE- (control)": 0, "TERBINAFINE+": 1}
        self._load_samples()

    def _load_samples(self):
        for treatment_name, label in self.class_map.items():
            treatment_path = os.path.join(self.root_dir, treatment_name)
            if not os.path.exists(treatment_path):
                print(f"Warning: Path not found: {treatment_path}")
                continue
            worm_dirs = [
                d
                for d in os.listdir(treatment_path)
                if os.path.isdir(os.path.join(treatment_path, d))
            ]
            for worm_id in worm_dirs:
                img_dir = os.path.join(treatment_path, worm_id, "photos_trajectories")
                if not os.path.exists(img_dir):
                    continue
                images = glob.glob(os.path.join(img_dir, "*.png"))
                for img_path in images:
                    self.samples.append((img_path, label, worm_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, worm_id = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transforms.ToPILImage()(image)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32), worm_id

    def get_indices_labels_groups(self):
        labels = [s[1] for s in self.samples]
        groups = [s[2] for s in self.samples]
        return np.arange(len(self.samples)), np.array(labels), np.array(groups)


# ==========================================
# 2. ResNet Model
# ==========================================


def get_resnet_model(pretrained=True):
    try:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    except:
        model = models.resnet18(pretrained=pretrained)

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 1),
        nn.Sigmoid(),
    )
    return model


# ==========================================
# 3. Training & Validation Logic
# ==========================================


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels, _ in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_worm_ids = []

    with torch.no_grad():
        for images, labels, worm_ids in tqdm(loader, desc="Val", leave=False):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.cpu().numpy())
            all_worm_ids.extend(worm_ids)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # --- Worm Level Metrics ---
    worm_preds_map = {}
    worm_labels_map = {}

    for i, wid in enumerate(all_worm_ids):
        if wid not in worm_preds_map:
            worm_preds_map[wid] = []
            worm_labels_map[wid] = all_labels[i]
        worm_preds_map[wid].append(all_preds[i])

    final_worm_labels = []
    final_worm_preds_prob = []

    for wid in worm_preds_map:
        avg_prob = np.mean(worm_preds_map[wid])
        final_worm_preds_prob.append(avg_prob)
        final_worm_labels.append(worm_labels_map[wid])

    final_worm_labels = np.array(final_worm_labels)
    final_worm_preds_prob = np.array(final_worm_preds_prob)
    final_worm_preds_bin = (final_worm_preds_prob > 0.5).astype(int)

    metrics = {
        "acc": accuracy_score(final_worm_labels, final_worm_preds_bin),
        "f1": f1_score(final_worm_labels, final_worm_preds_bin, zero_division=0),
        "prec": precision_score(
            final_worm_labels, final_worm_preds_bin, zero_division=0
        ),
        "rec": recall_score(final_worm_labels, final_worm_preds_bin, zero_division=0),
    }

    return metrics


# ==========================================
# 4. Main Execution
# ==========================================

if __name__ == "__main__":

    # --- CONFIG ---
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_LR = 1e-4
    DEFAULT_EPOCHS = 15
    IMG_SIZE = 128
    N_SPLITS = 5
    DATA_DIR = "worm/cnn_dataset"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- TRANSFORMS ---
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # --- DATASET & FOLDS ---
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found.")
        # Pour le test sans données, tu peux commenter les lignes ci-dessus
    else:
        dataset_full = CElegansCNNDataset(DATA_DIR, transform=None)

        if len(dataset_full) == 0:
            print("Error: No images found.")
            exit()

        indices, labels, groups = dataset_full.get_indices_labels_groups()
        sgkf = StratifiedGroupKFold(n_splits=N_SPLITS)

        all_folds_results = {"acc": [], "prec": [], "rec": [], "f1": []}

        for fold, (train_idx, val_idx) in enumerate(
            sgkf.split(indices, labels, groups=groups)
        ):
            print(f"\n{'='*20} FOLD {fold+1}/{N_SPLITS} {'='*20}")

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
                train_ds, batch_size=DEFAULT_BATCH_SIZE, shuffle=True, num_workers=2
            )
            val_loader = DataLoader(
                val_ds, batch_size=DEFAULT_BATCH_SIZE, shuffle=False, num_workers=2
            )

            model = get_resnet_model(pretrained=True).to(device)
            criterion = nn.BCELoss()
            optimizer = optim.AdamW(model.parameters(), lr=DEFAULT_LR)

            best_fold_f1 = 0.0
            best_fold_metrics = {}

            for epoch in range(DEFAULT_EPOCHS):
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
                    # torch.save(model.state_dict(), f"resnet_fold_{fold+1}_best.pth")

            print(f"--> Best Result Fold {fold+1}: {best_fold_metrics}")

            for k in all_folds_results:
                all_folds_results[k].append(best_fold_metrics[k])

        # --- RAPPORT FINAL & PLOT ---
        print(f"\n{'='*20} FINAL RESULTS (Average over {N_SPLITS} folds) {'='*20}")

        means = []
        stds = []
        metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
        keys = ["acc", "prec", "rec", "f1"]

        print(f"{'Metric':<10} | {'Mean':<10} | {'Std Dev':<10}")
        print("-" * 36)

        for k in keys:
            mean_val = np.mean(all_folds_results[k])
            std_val = np.std(all_folds_results[k])
            means.append(mean_val)
            stds.append(std_val)
            print(f"{k.upper():<10} | {mean_val:.4f}     | {std_val:.4f}")

        # ==========================================
        # 5. Plotting
        # ==========================================
        plt.figure(figsize=(10, 6))

        # Couleurs style "Science"
        colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3"]

        # Création des barres
        bars = plt.bar(
            metric_names,
            means,
            yerr=stds,
            capsize=10,
            color=colors,
            alpha=0.8,
            edgecolor="black",
            width=0.6,
        )

        plt.ylim(
            0, 1.1
        )  # L'axe Y va de 0 à 1.1 pour laisser de la place aux annotations
        plt.ylabel("Score")
        plt.title(
            f"Performance Metrics (Avg over {N_SPLITS} folds) - Best F1 Selection"
        )
        plt.grid(axis="y", linestyle="--", alpha=0.6)

        # Ajouter les valeurs exactes au-dessus des barres
        for bar, mean_val, std_val in zip(bars, means, stds):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + std_val + 0.02,
                f"{mean_val:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig("performance_metrics.png", dpi=300)
        print("\nPlot sauvegardé sous 'performance_metrics.png'")
        plt.show()
