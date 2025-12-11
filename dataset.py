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


class UnifiedCElegansAugmentedDataset(UnifiedCElegansDataset):
    """
    Extends UnifiedCElegansDataset. 
    This dataset augments each sample by creating modified versions of them.
    Each sample has the following augmentations:
        - Original  
        - 3 Trajectories rotated by a random angle
        - Trajectories with a random offset added to X and Y coordinates
        - Trajectories scaled by a random factor between 0.8 and 1.2 (arbitrary choice)
    The resulting dataset is 6 times larger than the original.
    All augmentations are computed and stored in memory at initialization.
    """
    def __init__(self, pytorch_dir=None, sklearn_dir=None, max_segments=150, segment_len=900):
        super().__init__(pytorch_dir, sklearn_dir, max_segments, segment_len)
        
        # Identify feature indices
        self.x_idx = -1
        self.y_idx = -1
        self.speed_idx = -1
        
        if "X" in FEATURES_PYTORCH:
            self.x_idx = FEATURES_PYTORCH.index("X")
        if "Y" in FEATURES_PYTORCH:
            self.y_idx = FEATURES_PYTORCH.index("Y")
        if "Speed" in FEATURES_PYTORCH:
            self.speed_idx = FEATURES_PYTORCH.index("Speed")
            
        self.augmented_data = []
        self.augmented_labels = []
        self.augmented_worm_ids = []
        
        print("Augmenting dataset in memory...")
        n_original = len(self.pytorch_files)
        
        for i in tqdm(range(n_original), desc="Augmenting Data"):
            # Get original data using parent's getitem which reads from file
            original_tensor, label = super().__getitem__(i)
            worm_id = os.path.splitext(os.path.basename(self.pytorch_files[i]))[0]
            
            # 1. Original
            self.augmented_data.append(original_tensor)
            self.augmented_labels.append(label)
            self.augmented_worm_ids.append(worm_id)
            
            # Apply augmentations if X and Y are present
            if self.x_idx != -1 and self.y_idx != -1:
                X = original_tensor[:, self.x_idx, :]
                Y = original_tensor[:, self.y_idx, :]
                
                # 2. Rotate by a random angle
                theta = np.radians(np.random.uniform(0, 360))
                c, s = np.cos(theta), np.sin(theta)
                tens_45 = original_tensor.clone()
                tens_45[:, self.x_idx, :] = X * c - Y * s
                tens_45[:, self.y_idx, :] = X * s + Y * c
                self.augmented_data.append(tens_45)
                self.augmented_labels.append(label)
                self.augmented_worm_ids.append(worm_id)
                
                # 3. Rotate by a random angle
                theta = np.radians(np.random.uniform(0, 360))
                c, s = np.cos(theta), np.sin(theta)
                tens_45 = original_tensor.clone()
                tens_45[:, self.x_idx, :] = X * c - Y * s
                tens_45[:, self.y_idx, :] = X * s + Y * c
                self.augmented_data.append(tens_45)
                self.augmented_labels.append(label)
                self.augmented_worm_ids.append(worm_id)

                # 4. Rotate by a random angle
                theta = np.radians(np.random.uniform(0, 360))
                c, s = np.cos(theta), np.sin(theta)
                tens_45 = original_tensor.clone()
                tens_45[:, self.x_idx, :] = X * c - Y * s
                tens_45[:, self.y_idx, :] = X * s + Y * c
                self.augmented_data.append(tens_45)
                self.augmented_labels.append(label)
                self.augmented_worm_ids.append(worm_id)
                
                # 5. Random offset
                dx = np.random.uniform(-50, 50)
                dy = np.random.uniform(-50, 50)
                tens_offset = original_tensor.clone()
                # Mask for padding (assuming 0 padding)
                mask = (tens_offset.abs().sum(dim=1) > 1e-6)
                tens_offset[:, self.x_idx, :][mask] += dx
                tens_offset[:, self.y_idx, :][mask] += dy
                self.augmented_data.append(tens_offset)
                self.augmented_labels.append(label)
                self.augmented_worm_ids.append(worm_id)

                # 6. Scale
                scale = np.random.uniform(0.8, 1.2)
                tens_scale = original_tensor.clone()
                tens_scale[:, self.x_idx, :] *= scale
                tens_scale[:, self.y_idx, :] *= scale
                if self.speed_idx != -1:
                    tens_scale[:, self.speed_idx, :] *= scale
                self.augmented_data.append(tens_scale)
                self.augmented_labels.append(label)
                self.augmented_worm_ids.append(worm_id)

    def get_data_for_rocket(self, feature_cols=None):
        """
        Returns the augmented data for ROCKET.
        Note: This uses the features defined in FEATURES_PYTORCH as that is what is stored in memory.
        The data is flattened from (Segments, Channels, Length) to (Channels, Segments*Length).
        """
        print("Loading augmented data for ROCKET from memory...")
        X = []
        y = []
        ids = []
        
        for tensor, label, worm_id in zip(self.augmented_data, self.augmented_labels, self.augmented_worm_ids):
            flat_ts = tensor.permute(1, 0, 2).reshape(tensor.shape[1], -1).numpy()
            X.append(flat_ts)
            y.append(label.item())
            ids.append(worm_id)
            
        return np.array(X), np.array(y), np.array(ids)

    def get_data_for_sklearn(self, feature_cols=None):
        """
        Returns the augmented data for Sklearn.
        Since we cannot easily compute the scalar features (Age, Tortuosity, etc.) for the augmented data,
        we return the flattened raw trajectories.
        
        Shape: (n_samples, n_channels * max_segments * segment_len)
        """
        print("Loading augmented data for Sklearn from memory (Flattened Trajectories)...")
        X = []
        y = []
        ids = []
        
        for tensor, label, worm_id in zip(self.augmented_data, self.augmented_labels, self.augmented_worm_ids):
            # tensor: (max_segments, n_channels, segment_len)
            # Flatten completely
            flat_features = tensor.numpy().flatten()
            X.append(flat_features)
            y.append(label.item())
            ids.append(worm_id)
            
        return np.array(X), np.array(y), np.array(ids)

    def get_worm_ids_for_pytorch(self):
        return np.array(self.augmented_worm_ids)

    def __len__(self):
        return len(self.augmented_data)

    def __getitem__(self, idx):
        return self.augmented_data[idx], self.augmented_labels[idx]
            
