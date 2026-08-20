import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim, max_time=1000000.0):
        super().__init__()
        self.embed_dim = embed_dim
        # Frequencies scaled to handle time up to max_time
        inv_freq = 1.0 / (max_time ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, t):
        # t: (B, T, 1)
        sinusoid_inp = t * self.inv_freq  # (B, T, ceil(embed_dim / 2))
        emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        # Odd embed_dim: sin/cos concat is one longer than requested.
        if emb.size(-1) > self.embed_dim:
            emb = emb[..., : self.embed_dim]
        elif emb.size(-1) < self.embed_dim:
            emb = F.pad(emb, (0, self.embed_dim - emb.size(-1)))
        return emb
