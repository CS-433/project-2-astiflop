import torch.nn as nn
import torch.nn.functional as F

class GatedAttention(nn.Module):
    """
    Gated Attention (Ilse et al. 2018).
    """

    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(dim, hidden_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None):
        # x: (Batch, Num_Instances, Dim)

        # V-Attention: learn non-linearity
        # U-Attention: learn gating 
        A_V = self.attention_V(x)
        A_U = self.attention_U(x)

        # Element-wise multiplication (Conjunctive-like mechanism)
        A = self.attention_weights(A_V * A_U)  # (Batch, Num_Instances, 1)

        if mask is not None:
            mask = mask.unsqueeze(-1)  # (Batch, Num_Instances, 1)
            A = A.masked_fill(mask == 0, -1e9)

        # Softmax over instances to get probability distribution
        weights = F.softmax(A, dim=1)
        return weights # shape: (Batch, Num_Instances, 1)