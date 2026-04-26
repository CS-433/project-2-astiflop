import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import random

from .wrappers import BenchmarkWrapper, VisualizationWrapper

class RandomDummyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        B = x.shape[0]
        # Output a random number between 0 and 150 for each item in the batch
        out = torch.FloatTensor(B).uniform_(0, 150).to(x.device)
        return out, None, None


class SegmentDummyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        # Calculate sequence length
        if mask is not None:
            lengths = mask.sum(dim=1)  # (B,)
        else:
            lengths = torch.tensor([x.shape[1]] * x.shape[0], device=x.device, dtype=torch.float32)
        
        # Output 60 - number of segments
        out = 60.0 - lengths
        return out, None, None


class DummyBenchmarkWrapper(BenchmarkWrapper):
    def load(self, path=None):
        device = self.params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        model_type = self.params.get("model_type", "random") # 'random' or 'segment'
        
        if model_type == "random":
            self.model = RandomDummyModel().to(device)
        elif model_type == "segment":
            self.model = SegmentDummyModel().to(device)
        else:
            raise ValueError(f"Unknown dummy model type: {model_type}")
        
        self.model.eval()

    def benchmark(self, test_loader):
        device = self.params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        all_trajectory_preds = []
        
        with torch.no_grad():
            for X, _, total_segment_len in test_loader:
                B, T_max, V, L = X.shape
                X = X.cpu()
                
                for i in range(B):
                    T_actual = int(total_segment_len[i].item())
                    full_trajectory = X[i]
                    
                    X_staircase = []
                    for t in range(1, T_actual + 1):
                        X_staircase.append(full_trajectory[:t])
                        
                    # Batch prediction for this single trajectory
                    X_padded = pad_sequence(X_staircase, batch_first=True).to(device)
                    indices = torch.arange(X_padded.size(1), device=device).expand(len(X_staircase), -1)
                    lengths_tensor = torch.tensor([len(x) for x in X_staircase], device=device).unsqueeze(1)
                    mask = (indices < lengths_tensor).float()
                    
                    trajectory_preds, _, _ = self.model(X_padded, mask=mask)
                    trajectory_preds = trajectory_preds.cpu().numpy()
                    
                    # No denormalization needed; output is already in desired format
                    all_trajectory_preds.append(trajectory_preds)
                    
        return {
            "predictions": all_trajectory_preds,
            "interpretability_score": 0.0
        }

class DummyVisualizationWrapper(VisualizationWrapper):
    def load(self, path=None):
        device = self.params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        model_type = self.params.get("model_type", "random") # 'random' or 'segment'
        
        if model_type == "random":
            self.model = RandomDummyModel().to(device)
        elif model_type == "segment":
            self.model = SegmentDummyModel().to(device)
        else:
            raise ValueError(f"Unknown dummy model type: {model_type}")
        
        self.model.eval()

    def get_trajectory_predictions(self, data_tensor, total_segments):
        device = self.params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        T_actual = int(total_segments)
        
        predictions = []
        variances = []
        
        with torch.no_grad():
            for t in range(1, T_actual + 1):
                x_t = data_tensor[:t].unsqueeze(0).to(device)
                mask = torch.ones(1, t).to(device)
                
                out, _, _ = self.model(x_t, mask=mask)
                pred_val = out.item()
                predictions.append(pred_val)
                variances.append(10.0 if pred_val > 45 else 2.0)
                
        return predictions, variances, {}
