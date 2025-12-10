import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from glob import glob
from tqdm import tqdm

# Load the env variables
from dotenv import load_dotenv

load_dotenv()

import ast

FEATURES_ROCKET = os.getenv("features_cols_rock", ["X", "Y", "Speed"])
if isinstance(FEATURES_ROCKET, str):
    FEATURES_ROCKET = ast.literal_eval(FEATURES_ROCKET)

FEATURES_PYTORCH = os.getenv("features_cols_pytorch", ["X", "Y", "Speed"])
if isinstance(FEATURES_PYTORCH, str):
    FEATURES_PYTORCH = ast.literal_eval(FEATURES_PYTORCH)

FEATURES_SKLEARN = os.getenv(
    "features_cols_sklearn",
    [
        "Age_hours",
        "Mean_Speed",
        "Median_Speed",
        "Net_Displacement",
        "Tortuosity",
        "Worm_ID",
    ],
)
if isinstance(FEATURES_SKLEARN, str):
    FEATURES_SKLEARN = ast.literal_eval(FEATURES_SKLEARN)


class UnifiedCElegansDataset(Dataset):
    def __init__(
        self, pytorch_dir=None, sklearn_dir=None, max_segments=150, segment_len=900
    ):
        """
        Args:
            pytorch_dir (str): Path to CSVs for PyTorch and Rocket datasets
            sklearn_dir (str): Path to CSVs for Sklearn datasets
        """
        self.pytorch_dir = pytorch_dir
        self.sklearn_dir = sklearn_dir

        self.max_segments = max_segments
        self.segment_len = segment_len
        self.class_map = {"TERBINAFINE- (control)": 0, "TERBINAFINE+": 1}

        # Initialisation des listes de fichiers et labels
        self.pytorch_files = []
        self.pytorch_labels = []

        self.sklearn_files = []
        self.sklearn_labels = []

        # 1. Loading PyTorch paths (if folder provided)
        if self.pytorch_dir:
            self.pytorch_files, self.pytorch_labels = self._scan_folder(
                self.pytorch_dir
            )
            # Max lenghts for padding rocket data
            self.rocket_max_len = 0
            for f in self.pytorch_files:
                try:
                    nrows = pd.read_csv(f, usecols=[0]).shape[0]
                    if nrows > self.rocket_max_len:
                        self.rocket_max_len = nrows
                except:
                    pass  # Gestion erreurs lecture

        # 2. Loading Sklearn paths (if folder provided)
        if self.sklearn_dir:
            self.sklearn_files, self.sklearn_labels = self._scan_folder(
                self.sklearn_dir
            )

    def _scan_folder(self, root_path):
        """Helper function to scan a folder and retrieve file paths and labels."""
        files = []
        labels = []
        for group_name, label in self.class_map.items():
            path = os.path.join(root_path, group_name, "*.csv")
            found = glob(path)
            files.extend(found)
            labels.extend([label] * len(found))
        # On trie pour garantir que l'ordre est le même à chaque run
        # Astuce : zipper, trier, dézipper
        if files:
            zipped = sorted(zip(files, labels))
            files, labels = zip(*zipped)
            return list(files), list(labels)
        return [], []

    def __len__(self):

        return len(self.pytorch_files)

    def __getitem__(self, idx):
        if not self.pytorch_files:
            raise ValueError(
                "You are trying to access PyTorch data but 'pytorch_dir' was not provided!"
            )

        file_path = self.pytorch_files[idx]
        label = self.pytorch_labels[idx]

        df = pd.read_csv(file_path)
        feature_cols = FEATURES_PYTORCH

        data_tensor = torch.zeros(self.max_segments, len(feature_cols), self.segment_len)

        if not df.empty:
            if "Segment" in df.columns:
                segments = df.groupby("Segment")
                for i, (seg_id, seg_df) in enumerate(segments):
                    if i >= self.max_segments:
                        break
                    vals = seg_df[feature_cols].values
                    features = torch.tensor(vals.T, dtype=torch.float32)
                    curr_len = features.shape[1]
                    if curr_len > self.segment_len:
                        features = features[:, : self.segment_len]
                    data_tensor[i, :, : features.shape[1]] = features
            else:
                pass

        if torch.isnan(data_tensor).any():
            print(f"NaN detected in data tensor for file: {file_path}")
            # detailed debug info
            print(f"Data tensor shape: {data_tensor.shape}"
                  f"\nData tensor contents:\n{data_tensor}")
            print(f"Nan locations (column, row):")
            nan_indices = torch.isnan(data_tensor).nonzero(as_tuple=False)
            print(nan_indices)
            exit(0)
            
        return data_tensor, torch.tensor(label, dtype=torch.long)

    def get_data_for_rocket(self, feature_cols=FEATURES_ROCKET):
        """
        Load and pad time series data for ROCKET input.
        """
        X, y, worm_ids = [], [], []
        for file_path, label in zip(self.pytorch_files, self.pytorch_labels):
            df = pd.read_csv(file_path)
            ts = df[feature_cols].values
            X.append(ts)
            y.append(label)
            worm_ids.append(os.path.splitext(os.path.basename(file_path))[0])
        if not X:
            return None, None, None
        max_length = max(len(ts) for ts in X)
        X_padded = []
        for ts in X:
            padding_length = max_length - len(ts)
            if padding_length > 0:
                padding = np.zeros((padding_length, ts.shape[1]))
                ts_padded = np.vstack([ts, padding])
            else:
                ts_padded = ts
            X_padded.append(ts_padded)
        X_array = np.array(X_padded)
        X_transposed = X_array.transpose(0, 2, 1)

        return X_transposed, np.array(y), np.array(worm_ids)

    def get_data_for_sklearn(self, feature_cols=FEATURES_SKLEARN):
        """Load data for sklearn models.
        In each file we keep only the feature columns and return as numpy arrays.
        """
        data_list = []
        labels = []
        worm_ids = []
        for file_path, label in zip(self.sklearn_files, self.sklearn_labels):
            df = pd.read_csv(file_path)
            if not df.empty:
                # Aggregate features across segments (mean) to get a fixed-size vector per worm
                features = df[feature_cols].mean().values
                data_list.append(features)
            labels.append(label)
            worm_id = os.path.splitext(os.path.basename(file_path))[0]
            worm_id = worm_id.replace("_segments", "")
            worm_ids.append(worm_id)

        if not data_list:
            return None, None, None
        X_array = np.array(data_list)
        y_array = np.array(labels)
        worm_ids_array = np.array(worm_ids)

        return X_array, y_array, worm_ids_array

    def get_worm_ids_for_pytorch(self):
        """Returns a list of worm IDs corresponding to the PyTorch dataset indices."""
        worm_ids = []
        for file_path in self.pytorch_files:
            worm_ids.append(os.path.splitext(os.path.basename(file_path))[0])
        return np.array(worm_ids)
