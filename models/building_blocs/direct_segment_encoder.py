import torch.nn as nn
import torch.nn.functional as F


class DirectSegmentEncoder(nn.Module):
    """
    Projects a raw 1D segment directly into an embedding (no CNN).

    Input:  (Batch, 1, Length)
    Output: (Batch, Embedding_Dim)
    """

    def __init__(self, input_len, embedding_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_len, embedding_dim * 2)
        self.fc2 = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, x):
        # x shape: (Batch * T * V, 1, Length)
        x = x.squeeze(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
