import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold


def get_cnn_model(model_name="resnet18", pretrained=True):
    weights = "DEFAULT" if pretrained else None
    
    if model_name == "resnet18":
        model = models.resnet18(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    elif model_name == "resnet50":
        model = models.resnet50(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    elif model_name == "densenet121":
        model = models.densenet121(weights=weights)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")

    return model


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
