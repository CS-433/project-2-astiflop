import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from glob import glob

class WormTrajectoryDataset(Dataset):
    def __init__(self, root_dir, max_segments=150, segment_len=900):
        """
        Args:
            root_dir (str): Path to 'preprocessed_data'
            max_segments (int): Fixed size for Time dimension (padding). 
                                Set this >= the max segments any worm has.
            segment_len (int): Frames per segment (matches your preprocessing).
        """
        self.root_dir = root_dir
        self.max_segments = max_segments
        self.segment_len = segment_len
        self.files = []
        self.labels = []

        # 0 = Control, 1 = Treated
        self.class_map = {"TERBINAFINE- (control)": 0, "TERBINAFINE+": 1}

        for group_name, label in self.class_map.items():
            path = os.path.join(root_dir, group_name, "*.csv")
            found_files = glob(path)
            self.files.extend(found_files)
            self.labels.extend([label] * len(found_files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]

        # Read CSV
        df = pd.read_csv(file_path)

        # Extract features (Order matters: X, Y, Speed)
        # Ensure we only take the columns we need
        feature_cols = ["X", "Y", "Speed"]
        
        # Container for the segments
        # Shape: (Max_Segments, 3, Segment_Len)
        data_tensor = torch.zeros(self.max_segments, 3, self.segment_len)
        
        # Group by Segment
        if not df.empty:
            segments = df.groupby("Segment")
            
            for i, (seg_id, seg_df) in enumerate(segments):
                if i >= self.max_segments:
                    break # Truncate if worm lived longer than max_segments
                
                # Extract values
                vals = seg_df[feature_cols].values # Shape (N_frames, 3)
                
                # Handle cases where a segment has < 900 frames (e.g. death in middle)
                # We pad the specific segment with zeros if needed
                curr_len = vals.shape[0]
                features = torch.tensor(vals.T, dtype=torch.float32) # Transpose to (3, N_frames)
                
                # Place into main tensor
                # If segment is full length (900), it fits perfectly.
                # If smaller, it fills the beginning, rest remains 0.
                if curr_len > self.segment_len:
                     features = features[:, :self.segment_len] # Crop
                
                data_tensor[i, :, :features.shape[1]] = features

        # Return: Data, Label, Actual number of segments (useful for masking if needed)
        return data_tensor, torch.tensor(label, dtype=torch.float32)