import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


def get_stratified_worm_splits(summary_csv_path, n_splits=5, random_state=42):
    """
    Arg:
        summary_csv_path (str): Path to the CSV file summarizing worm data.
        n_splits (int): Number of folds for Stratified K-Fold.
        random_state (int): Random seed for reproducibility.

    Returns:
        folds_data (list of dict): List containing train and val worm IDs for each fold.
    """
    df = pd.read_csv(summary_csv_path)

    # We get rid of / in Worm_ID to ensure consistency
    # Ex: "/20240924_piworm09_1" -> "20240924_piworm09_1"
    df["Worm_ID"] = df["Filename"].astype(str).apply(lambda x: x.strip("/"))

    X = df["Worm_ID"].values
    y = df["Terbinafine"].values  # La cible pour la stratification

    # 2. Création des Folds
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds_data = []

    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        train_ids = X[train_idx]
        val_ids = X[val_idx]

        # Calculate and print positive class ratios
        y_train = y[train_idx]
        y_val = y[val_idx]
        pos_ratio_train = np.mean(y_train)
        pos_ratio_val = np.mean(y_val)

        print(
            f"Fold {i+1}: Train={len(train_ids)} (Pos={pos_ratio_train:.0%}), Val={len(val_ids)} (Pos={pos_ratio_val:.0%})"
        )

        folds_data.append({"train": train_ids.tolist(), "val": val_ids.tolist()})

    return folds_data



