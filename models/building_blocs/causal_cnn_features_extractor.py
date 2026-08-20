import torch.nn as nn
import torch.nn.functional as F

from models.building_blocs.tcn_temporal import Chomp1d


class CausalCNNFeatureExtractor(nn.Module):
    """
    Encodes raw 1D segments with strictly causal convolutions.

    Each time step only depends on past (and current) samples within the
    segment. The final embedding is taken from the last causal state.

    Input:  (Batch, 1, Length)
    Output: (Batch, Embedding_Dim)
    """

    def __init__(self, input_len, embedding_dim=128, kernel_size=7):
        super().__init__()
        self.input_len = input_len
        self.embedding_dim = embedding_dim

        # Same channel progression as CNNFeatureExtractor, but causal.
        pad1 = kernel_size - 1
        self.conv1 = nn.Conv1d(1, 32, kernel_size=kernel_size, stride=1, padding=pad1)
        self.chomp1 = Chomp1d(pad1)

        pad2 = 5 - 1
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=pad2)
        self.chomp2 = Chomp1d(pad2)

        pad3 = 3 - 1
        self.conv3 = nn.Conv1d(64, embedding_dim, kernel_size=3, stride=1, padding=pad3)
        self.chomp3 = Chomp1d(pad3)

        self.fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        # x shape: (Batch * T * V, 1, Length)
        x = F.relu(self.chomp1(self.conv1(x)))
        x = F.relu(self.chomp2(self.conv2(x)))
        x = F.relu(self.chomp3(self.conv3(x)))
        # Last causal state summarises the full observed segment
        x = x[:, :, -1]
        x = F.relu(self.fc(x))
        return x
