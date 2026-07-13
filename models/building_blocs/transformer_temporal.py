import torch
import torch.nn as nn


class CausalTransformerBlock(nn.Module):
    """Single causal transformer encoder block with pre-norm residuals."""

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        normed = self.norm1(x)
        attn_out, attn_weights = self.self_attn(
            normed,
            normed,
            normed,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, attn_weights


class TransformerTemporal(nn.Module):
    """
    Lightweight causal transformer for temporal modeling.
    Interpretability is preserved at the architecture level via the outer
    GatedAttention modules, while self-attention weights remain inspectable.
    """

    def __init__(self, embed_dim, num_layers=1, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.blocks = nn.ModuleList(
            [
                CausalTransformerBlock(embed_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    @staticmethod
    def _causal_mask(seq_len, device):
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(self, seg_emb, mask=None):
        B, T, _ = seg_emb.shape
        attn_mask = self._causal_mask(T, seg_emb.device)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = mask == 0

        x = seg_emb
        for block in self.blocks:
            x, _ = block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        aux_loss = torch.tensor(0.0, device=seg_emb.device)
        return x, aux_loss
