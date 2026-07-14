import torch
import torch.nn as nn
import torch.nn.functional as F

from models.building_blocs.gated_attention import GatedAttention
from models.building_blocs.hf_patchtst_encoder import HFPatchTSTEncoder
from models.building_blocs.time_embedding import SinusoidalTimeEmbedding


class FoundationRegressor(nn.Module):
    """
    Remaining-lifetime regressor built on a pretrained PatchTST backbone.

    Pipeline: foundation encoder → gated attention → MLP head.
    No temporal stack (BiLSTM / TCN / …) is used after the foundation model.
    """

    def __init__(
        self,
        segment_len,
        embed_dim=128,
        dropout=0.3,
        use_time_encoding=True,
        output_type="point",
        pretrained_model_name=None,
        freeze_backbone=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_type = output_type

        self.hf_encoder = HFPatchTSTEncoder(
            segment_len=segment_len,
            embedding_dim=embed_dim,
            pretrained_model_name=pretrained_model_name,
            freeze_backbone=freeze_backbone,
        )

        self.variate_attention = GatedAttention(
            dim=embed_dim, hidden_dim=embed_dim // 4
        )

        self.use_time_encoding = use_time_encoding
        if self.use_time_encoding:
            self.time_projection = SinusoidalTimeEmbedding(
                embed_dim, max_time=1500000.0
            )

        self.segment_attention = GatedAttention(
            dim=embed_dim, hidden_dim=embed_dim // 4
        )

        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        if self.output_type == "point":
            self.output_layer = nn.Linear(embed_dim // 2, 1)
        elif self.output_type == "gaussian":
            self.output_layer = nn.Linear(embed_dim // 2, 2)
        elif self.output_type == "weibull":
            self.output_layer = nn.Linear(embed_dim // 2, 2)
        else:
            raise ValueError(f"Unknown output type: {self.output_type}")

    def forward(self, x, mask=None):
        B, T, V, L = x.shape

        if self.use_time_encoding:
            x_features = x[:, :, :-1, :]
            x_lifetime = x[:, :, -1, :]
            V_feat = V - 1
        else:
            x_features = x
            V_feat = V

        # One multivariate PatchTST forward per segment: (B*T, V_feat, L)
        x_reshaped = x_features.reshape(B * T, V_feat, L)
        features = self.hf_encoder(x_reshaped)  # (B*T, V_feat, embed_dim)

        v_weights = self.variate_attention(features, mask=None)
        seg_emb = torch.sum(features * v_weights, dim=1).view(B, T, self.embed_dim)

        if self.use_time_encoding:
            time_scalar = x_lifetime.mean(dim=-1).unsqueeze(-1)
            seg_emb = seg_emb + self.time_projection(time_scalar)

        s_weights = self.segment_attention(seg_emb, mask=mask)
        context_vector = torch.sum(seg_emb * s_weights, dim=1)

        reg_features = self.regressor(context_vector)
        output = self.output_layer(reg_features).squeeze(-1)

        if self.output_type == "weibull":
            output = F.softplus(output) + 1e-6

        aux_loss = torch.tensor(0.0, device=x.device)
        return output, s_weights, v_weights, aux_loss
