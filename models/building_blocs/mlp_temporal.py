import torch
import torch.nn as nn


class MLPTemporal(nn.Module):
    """
    Per-segment MLP processing unit (embed_dim -> embed_dim).
    No cross-segment interaction.
    """

    def __init__(self, embed_dim, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, seg_emb, mask=None):
        out = self.mlp(seg_emb)
        aux_loss = torch.tensor(0.0, device=seg_emb.device)
        return out, aux_loss
