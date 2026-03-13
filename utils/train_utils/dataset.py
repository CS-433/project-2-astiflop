import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import cv2
import json
import torchvision.transforms as transforms


# Load the env variables
from dotenv import load_dotenv

load_dotenv()

import ast

# ---------------------------------------- Env loading ----------------------------------------
FEATURES_PYTORCH = os.getenv("features_cols_pytorch", ["X", "Y", "Speed"])
if isinstance(FEATURES_PYTORCH, str):
    FEATURES_PYTORCH = ast.literal_eval(FEATURES_PYTORCH)
print(f"Using PyTorch features: {FEATURES_PYTORCH}")

FEATURES_ROCKET = os.getenv("features_cols_rock", ["X", "Y", "Speed"])
if isinstance(FEATURES_ROCKET, str):
    FEATURES_ROCKET = ast.literal_eval(FEATURES_ROCKET)

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




# ---------------------------------------- Deprecated -----------------------------------------
class UnifiedCElegansDataset(Dataset):
    def __init__(
        self, pytorch_dir=None, sklearn_dir=None, max_segments=150, segment_len=900
    ):
        """
        DEPRECATED
        Args:
            pytorch_dir (str): Path to CSVs for PyTorch and Rocket datasets
            sklearn_dir (str): Path to CSVs for Sklearn datasets
        """
        raise DeprecationWarning("This class is deprecated. Use LPBSDataset instead.")
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

        data_tensor = torch.zeros(
            self.max_segments, len(feature_cols), self.segment_len
        )

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
            print(
                f"Data tensor shape: {data_tensor.shape}"
                f"\nData tensor contents:\n{data_tensor}"
            )
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
    DEPRECATED
    Extends UnifiedCElegansDataset.
    This dataset augments each sample by creating modified versions of them.
    Each sample has the following augmentations:
        - Original
        - Random augmentations (Rotation, Offset, Scale) applied with p=0.5 each
    The resulting dataset size depends on augmentations_per_sample.
    All augmentations are computed and stored in memory at initialization.
    """

    def __init__(
        self, pytorch_dir=None, sklearn_dir=None, max_segments=150, segment_len=900, augmentations_per_sample=5
    ):
        super().__init__(pytorch_dir, sklearn_dir, max_segments, segment_len)
        self.augmentations_per_sample = augmentations_per_sample

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
                for _ in range(self.augmentations_per_sample):
                    augmented_tensor = original_tensor.clone()
                    
                    # Apply rotation with p=0.5
                    if np.random.rand() < 0.5:
                        augmented_tensor = self._apply_rotation(augmented_tensor)
                    
                    # Apply offset with p=0.5
                    if np.random.rand() < 0.5:
                        augmented_tensor = self._apply_offset(augmented_tensor)
                        
                    # Apply scaling with p=0.5
                    if np.random.rand() < 0.5:
                        augmented_tensor = self._apply_scaling(augmented_tensor)
                        
                    self.augmented_data.append(augmented_tensor)
                    self.augmented_labels.append(label)
                    self.augmented_worm_ids.append(worm_id)

    def _apply_rotation(self, tensor):
        theta = np.radians(np.random.uniform(0, 360))
        c, s = np.cos(theta), np.sin(theta)
        
        X = tensor[:, self.x_idx, :]
        Y = tensor[:, self.y_idx, :]
        
        new_tensor = tensor.clone()
        new_tensor[:, self.x_idx, :] = X * c - Y * s
        new_tensor[:, self.y_idx, :] = X * s + Y * c
        return new_tensor

    def _apply_offset(self, tensor):
        dx = np.random.uniform(-50, 50)
        dy = np.random.uniform(-50, 50)
        
        new_tensor = tensor.clone()
        # Mask for padding (assuming 0 padding)
        mask = new_tensor.abs().sum(dim=1) > 1e-6
        new_tensor[:, self.x_idx, :][mask] += dx
        new_tensor[:, self.y_idx, :][mask] += dy
        return new_tensor

    def _apply_scaling(self, tensor):
        scale = np.random.uniform(0.8, 1.2)
        new_tensor = tensor.clone()
        new_tensor[:, self.x_idx, :] *= scale
        new_tensor[:, self.y_idx, :] *= scale
        if self.speed_idx != -1:
            new_tensor[:, self.speed_idx, :] *= scale
        return new_tensor

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

        for tensor, label, worm_id in zip(
            self.augmented_data, self.augmented_labels, self.augmented_worm_ids
        ):
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
        print(
            "Loading augmented data for Sklearn from memory (Flattened Trajectories)..."
        )
        X = []
        y = []
        ids = []

        for tensor, label, worm_id in zip(
            self.augmented_data, self.augmented_labels, self.augmented_worm_ids
        ):
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
                images = glob(os.path.join(img_dir, "*.png"))
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





# ----------------------------------------- Datasets ------------------------------------------

class LPBSDataset(Dataset):
    def __init__(
        self, 
        data_dir, 
        scaler_type="none",
        mode="train",
        scaler_config_path="scaler_config.json",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Dataset object abstraction for Laboratory of the Physics of Biological Systems (LPBS) C. elegans data.
        This dataset loads all samples into memory at initialization, applies optional normalization, and provides access to the data for PyTorch models.

        Args:
            data_dir (str): Path to the directory containing the PyTorch CSV files organized in subdirectories by treatment.
            scaler_type (str): Type of normalization to apply ("none", "standard", "minmax"). Default is "none".
            mode (str): Mode of the dataset, either "train" or "test". This affects how normalization statistics are computed or loaded.
            scaler_config_path (str): Path to save/load normalization statistics when using "standard" or "minmax" scaling.
            device (str): Device to load tensors on ("cuda" or "cpu"). Default is "cuda" if available.
        """
        self.device = device

        # Data storage
        self.data = []
        self.treatments = []
        self.worm_ids = []
        self.lifespan_segments = []

        # 1. Loading PyTorch paths
        files, self.treatments = self._scan_folder(data_dir)
        # Load Data into memory immediately
        self._load_data(files)
        
        # Normalize if requested
        if scaler_type != "none":
            self._apply_normalization(mode, scaler_config_path, scaler_type)

    def _scan_folder(self, root_path):
        """Helper function to scan a folder and retrieve file paths and treatments."""
        class_map = {"TERBINAFINE- (control)": 0, "TERBINAFINE+": 1, "NoTerbinafine": 0, "Terbinafine": 1}
        
        files = []
        treatments = []
        subdirs = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
        
        for subdir in subdirs:
            # Simple heuristic or use class_map if it matches
            treatment = None
            if subdir in class_map:
                treatment = class_map[subdir]
            else:
                lower_name = subdir.lower()
                if "control" in lower_name or "no" in lower_name or "-" in lower_name:
                    treatment = 0
                    print(f"Warning: Subdir '{subdir}' not in class_map but matched as control (treatment 0)")
                else:
                    treatment = 1 
                    print(f"Warning: Subdir '{subdir}' not in class_map but matched as treated (treatment 1)")
            
            path = os.path.join(root_path, subdir, "*.csv")
            found = glob(path)
            files.extend(found)
            treatments.extend([treatment] * len(found))

        if files:
            zipped = sorted(zip(files, treatments))
            files, treatments = zip(*zipped)
            return list(files), list(treatments)
        return [], []

    def _load_data(self, files, max_segments=150, segment_len=900):
        """Loads all CSVs into tensors."""
        feature_cols = FEATURES_PYTORCH
        
        for idx, file_path in enumerate(tqdm(files, desc="Loading Data")):
            df = pd.read_csv(file_path)

            # Initialize tensor with zeros (default padding)    
            data_tensor = torch.zeros(max_segments, len(feature_cols), segment_len, device=self.device)
            
            segments = df.groupby("Segment")
            for i, (seg_id, seg_df) in enumerate(segments):
                if i >= max_segments:
                    raise ValueError(f"Number of segments in file {file_path} exceeds max_segments={max_segments}")
                
                vals = seg_df[feature_cols].values
                features = torch.tensor(vals.T, dtype=torch.float32)
                curr_len = features.shape[1]
                if curr_len > segment_len:
                    raise ValueError(f"Segment length in file {file_path} exceeds segment_len={segment_len}")
                data_tensor[i, :, : features.shape[1]] = features

            if torch.isnan(data_tensor).any():
                print(f"NaN detected in data tensor for file: {file_path}")
                exit(0)
                
            self.data.append(data_tensor)
            self.worm_ids.append(os.path.splitext(os.path.basename(file_path))[0])
            self.lifespan_segments.append(df["Segment"].max())

    def _apply_normalization(self, mode, scaler_config_path, scaler_type):
        """Calculates or loads stats and applies normalization."""
        # 1. Statistics         
        # Stack to (N, S, F, L)
        all_data = torch.stack(self.data) # This might be large
        
        if mode == "train":
            if scaler_type == "old":
                mask_old = (all_data.view(all_data.size(0), all_data.size(1), -1).abs().sum(dim=-1) > 1e-6)
                valid_x = all_data[mask_old] # (N_valid, F, L)
                num_features = all_data.shape[2]
                
                if valid_x.numel() > 0:
                    valid_x_flat = valid_x.transpose(1, 2).reshape(-1, num_features)
                    sum_x = valid_x_flat.sum(dim=0)
                    sum_sq_x = (valid_x_flat ** 2).sum(dim=0)
                    n_samples = valid_x_flat.shape[0]
                    mean_t = sum_x / n_samples
                    std_t = torch.sqrt(sum_sq_x / n_samples - mean_t ** 2)
                    means = mean_t.tolist()
                    stds = std_t.tolist()
                else:
                    means = [0.0] * num_features
                    stds = [1.0] * num_features
                mins = [0.0] * num_features
                maxs = [1.0] * num_features
            else:
                # Ignore padding            
                mask = (all_data.abs().sum(dim=2, keepdim=True) > 1e-6) # (N, S, 1, L)
                means = []
                stds = []
                mins = []
                maxs = []
                
                num_features = all_data.shape[2]
                
                for f in range(num_features):
                    feat_data = all_data[:, :, f, :] # (N, S, L)
                    feat_mask = mask[:, :, 0, :]   # (N, S, L)
                    
                    valid_vals = feat_data[feat_mask] # 1D tensor of valid values
                    
                    if valid_vals.numel() > 0:
                        means.append(valid_vals.mean().item())
                        stds.append(valid_vals.std().item())
                        mins.append(valid_vals.min().item())
                        maxs.append(valid_vals.max().item())
                    else:
                        means.append(0.0)
                        stds.append(1.0)
                        mins.append(0.0)
                        maxs.append(1.0)
            
            stats = {
                "mean": means,
                "std": stds,
                "min": mins,
                "max": maxs
            }
            
            # Save stats
            with open(scaler_config_path, "w") as f:
                json.dump(stats, f, indent=4)
            print(f"Scaler statistics saved to {scaler_config_path}")
            
        elif mode == "test":
            # Load stats
            if not os.path.exists(scaler_config_path):
                print(f"Warning: Scaler config {scaler_config_path} not found. Skipping normalization.")
                return
            
            with open(scaler_config_path, "r") as f:
                stats = json.load(f)
        
        # 2. Normalization
        mean = torch.tensor(stats["mean"]).view(1, 1, -1, 1)
        std = torch.tensor(stats["std"]).view(1, 1, -1, 1)
        min_v = torch.tensor(stats["min"]).view(1, 1, -1, 1)
        max_v = torch.tensor(stats["max"]).view(1, 1, -1, 1)
        
        new_data = []
        for i in range(len(self.data)):
            tensor = self.data[i].unsqueeze(0) # (1, S, F, L)
            
            if scaler_type == "old":
                x = tensor.squeeze(0)
                mask_old = (x.view(x.size(0), -1).abs().sum(dim=-1) > 1e-6).float().view(-1, 1, 1)
                x_scaled = (x - mean.squeeze(0)) / (std.squeeze(0) + 1e-8)
                tensor = (x_scaled * mask_old).unsqueeze(0)
            else:
                mask = (tensor.abs().sum(dim=2, keepdim=True) > 1e-6).float()
                
                if scaler_type == "standard":
                    tensor = (tensor - mean) / (std + 1e-8)
                elif scaler_type == "minmax":
                    tensor = (tensor - min_v) / (max_v - min_v + 1e-8)
                
                tensor = tensor * mask # Re-apply mask to clean padding
                
            new_data.append(tensor.squeeze(0))
            
        self.data = new_data
        print(f"Applied {scaler_type} normalization.")


    def augment_data(self, n_augmentations_per_sample=5):
        """
        When called, this function creates augmented versions of the data by applying random transformations (rotation, offset, scaling) to the original trajectories.
        The augmentations are done in place.

        Can only be called once.
        """
        if hasattr(self, "is_augmented") and self.is_augmented:
            raise ValueError("Data has already been augmented. Multiple augmentation is not supported.")
        self.is_augmented = True

        x_idx = FEATURES_PYTORCH.index("X")
        y_idx = FEATURES_PYTORCH.index("Y")
        speed_idx = FEATURES_PYTORCH.index("Speed") if "Speed" in FEATURES_PYTORCH else FEATURES_PYTORCH.index("ComputedSpeed_frames")

        # utils function
        def _apply_rotation(self, tensor):
            theta = np.radians(np.random.uniform(0, 360))
            c, s = np.cos(theta), np.sin(theta)
            
            X = tensor[:, x_idx, :]
            Y = tensor[:, y_idx, :]
            
            new_tensor = tensor.clone()
            new_tensor[:, x_idx, :] = X * c - Y * s
            new_tensor[:, y_idx, :] = X * s + Y * c
            return new_tensor

        def _apply_offset(self, tensor):
            dx = np.random.uniform(-100, 100)
            dy = np.random.uniform(-100, 100)
            
            new_tensor = tensor.clone()
            # Mask for padding (assuming 0 padding)
            mask = new_tensor.abs().sum(dim=1) > 1e-6
            new_tensor[:, x_idx, :][mask] += dx
            new_tensor[:, y_idx, :][mask] += dy
            return new_tensor

        def _apply_scaling(self, tensor):
            scale = np.random.uniform(0.75, 1.25)
            new_tensor = tensor.clone()
            new_tensor[:, x_idx, :] *= scale
            new_tensor[:, y_idx, :] *= scale
            if speed_idx != -1:
                new_tensor[:, speed_idx, :] *= scale
            return new_tensor
        

        # Apply augmentations to each data point
        augmented_data = []
        augmented_treatments = []
        augmented_worm_ids = []
        augmented_lifespan_segments = []
        for data_tensor, treatment, worm_id, lifespan_segment in zip(self.data, self.treatments, self.worm_ids, self.lifespan_segments):
            augmented_data.append(data_tensor)
            augmented_treatments.append(treatment)
            augmented_worm_ids.append(worm_id)
            augmented_lifespan_segments.append(lifespan_segment)

            for _ in range(n_augmentations_per_sample):
                augmented_tensor = data_tensor.clone()            
                # Apply rotation with p=0.5
                if np.random.rand() < 0.5:
                    augmented_tensor = _apply_rotation(self, augmented_tensor)
                
                # Apply offset with p=0.5
                if np.random.rand() < 0.5:
                    augmented_tensor = _apply_offset(self, augmented_tensor)
                    
                # Apply scaling with p=0.5
                if np.random.rand() < 0.5:
                    augmented_tensor = _apply_scaling(self, augmented_tensor)
                    
                augmented_data.append(augmented_tensor)
                augmented_treatments.append(treatment)
                augmented_worm_ids.append(worm_id)
                augmented_lifespan_segments.append(lifespan_segment)
        
        self.data = augmented_data
        self.treatments = augmented_treatments
        self.worm_ids = augmented_worm_ids
        self.lifespan_segments = augmented_lifespan_segments
        print(f"Data augmented with {n_augmentations_per_sample} augmentations per sample. Total samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_tensor = self.data[idx]
        treatment = self.treatments[idx]
        lifespan_segment = self.lifespan_segments[idx]


        if torch.isnan(data_tensor).any():
            print(f"NaN detected in data tensor for file: {self.pytorch_files[idx]}")
            # detailed debug info
            print(
                f"Data tensor shape: {data_tensor.shape}"
                f"\nData tensor contents:\n{data_tensor}"
            )
            print(f"Nan locations (column, row):")
            nan_indices = torch.isnan(data_tensor).nonzero(as_tuple=False)
            print(nan_indices)
            exit(0)

        return data_tensor, torch.tensor(treatment, dtype=torch.long), torch.tensor(lifespan_segment, dtype=torch.long)