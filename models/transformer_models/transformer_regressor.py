"""
Transformer regressor for remaining-lifetime prediction.

Two encoder modes:
- ``direct``: linear projection of each raw segment into ``embed_dim``
- ``cnn``: CNN feature extractor(s) into ``embed_dim`` (typically 16)

Both feed a causal transformer stack over segments, then an MLP head that
outputs point / Gaussian / Weibull parameters (selected via ``output_type``).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.building_blocs.cnn_features_extractor import CNNFeatureExtractor
from models.building_blocs.direct_segment_encoder import DirectSegmentEncoder
from models.building_blocs.gated_attention import GatedAttention
from models.building_blocs.time_embedding import SinusoidalTimeEmbedding
from models.building_blocs.transformer_temporal import TransformerTemporal


class TransformerRegressor(nn.Module):
    """
    Causal transformer RUL regressor with direct or CNN segment encoding.

    Pipeline: encoder → variate gated attention → [time emb] →
    TransformerTemporal → segment gated attention → MLP → distribution head.
    """

    def __init__(
        self,
        segment_len,
        embed_dim=16,
        dropout=0.15,
        encoder_type="cnn",
        feature_extractor_layers=1,
        transformer_layers=1,
        transformer_heads=4,
        use_time_encoding=True,
        output_type="weibull",
    ):
        super().__init__()
        if embed_dim % transformer_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"transformer_heads ({transformer_heads})"
            )
        if encoder_type not in ("direct", "cnn"):
            raise ValueError(
                f"Unknown encoder_type '{encoder_type}'. Expected 'direct' or 'cnn'."
            )

        self.embed_dim = embed_dim
        self.encoder_type = encoder_type
        self.feature_extractor_layers = feature_extractor_layers
        self.output_type = output_type
        self.use_time_encoding = use_time_encoding

        if encoder_type == "direct":
            self.encoders = nn.ModuleList(
                [DirectSegmentEncoder(segment_len, embed_dim)]
            )
            self.encoder_aggregation = nn.Identity()
        else:
            self.encoders = nn.ModuleList(
                [
                    CNNFeatureExtractor(
                        input_len=segment_len, embedding_dim=embed_dim
                    )
                    for _ in range(feature_extractor_layers)
                ]
            )
            if feature_extractor_layers == 1:
                self.encoder_aggregation = nn.Identity()
            else:
                self.encoder_aggregation = nn.Linear(
                    embed_dim * feature_extractor_layers, embed_dim
                )

        self.variate_attention = GatedAttention(
            dim=embed_dim, hidden_dim=max(embed_dim // 4, 1)
        )

        if self.use_time_encoding:
            self.time_projection = SinusoidalTimeEmbedding(
                embed_dim, max_time=1500000.0
            )

        self.temporal_model = TransformerTemporal(
            embed_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            dropout=dropout,
        )

        self.segment_attention = GatedAttention(
            dim=embed_dim, hidden_dim=max(embed_dim // 4, 1)
        )

        hidden = max(embed_dim // 2, 1)
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        if self.output_type == "point":
            self.output_layer = nn.Linear(hidden, 1)
        elif self.output_type in ("gaussian", "weibull"):
            self.output_layer = nn.Linear(hidden, 2)
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

        x_reshaped = x_features.reshape(B * T * V_feat, 1, L)
        extracted = [enc(x_reshaped) for enc in self.encoders]
        features_cat = torch.cat(extracted, dim=-1)
        features_agg = self.encoder_aggregation(features_cat)
        features = features_agg.view(B * T, V_feat, self.embed_dim)

        v_weights = self.variate_attention(features, mask=None)
        seg_emb = torch.sum(features * v_weights, dim=1).view(B, T, self.embed_dim)

        if self.use_time_encoding:
            time_scalar = x_lifetime.mean(dim=-1).unsqueeze(-1)
            seg_emb = seg_emb + self.time_projection(time_scalar)

        temporal_out, aux_loss = self.temporal_model(seg_emb, mask=mask)

        s_weights = self.segment_attention(temporal_out, mask=mask)
        context_vector = torch.sum(temporal_out * s_weights, dim=1)

        reg_features = self.regressor(context_vector)
        output = self.output_layer(reg_features).squeeze(-1)

        if self.output_type == "weibull":
            output = F.softplus(output) + 1e-6

        return output, s_weights, v_weights, aux_loss
