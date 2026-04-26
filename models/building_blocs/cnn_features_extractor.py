import torch.nn as nn
import torch.nn.functional as F

class CNNFeatureExtractor(nn.Module):
    """
    Encodes raw time series signals (segments) into a feature vector via a CNN.
    Input: (Batch, 1, Length) -> Output: (Batch, Embedding_Dim)
    """

    def __init__(self, input_len, embedding_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(64, embedding_dim, kernel_size=3, stride=2, padding=1)
        self.global_pool = nn.AdaptiveMaxPool1d(1)  # Pooling to get fixed size vector
        self.fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        # x shape: (Batch * T * V, 1, Length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.global_pool(x).squeeze(-1)  # (Batch*T*V, Emb_Dim)
        x = F.relu(self.fc(x))
        return x # shape: (Batch*T*V, Embedding_Dim)