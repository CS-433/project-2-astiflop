import torch
import torch.nn as nn
import torch.nn.functional as F

from models.building_blocs.cnn_features_extractor import CNNFeatureExtractor
from models.building_blocs.time_embedding import SinusoidalTimeEmbedding


class LinearScalarRegressor(nn.Module):
    """
    Direct scalar regressor: CNN feature extraction, mean-pooling over variates
    and segments, then a single linear output layer. No attention or temporal module.
    """

    def __init__(
        self,
        segment_len,
        embed_dim=64,
        feature_extractor_layers=1,
        use_time_encoding=True,
        output_type="point",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_time_encoding = use_time_encoding
        self.output_type = output_type

        self.feature_extractors = nn.ModuleList(
            [
                CNNFeatureExtractor(input_len=segment_len, embedding_dim=embed_dim)
                for _ in range(feature_extractor_layers)
            ]
        )
        self.cnn_aggregation = nn.Linear(
            embed_dim * feature_extractor_layers, embed_dim
        )

        if use_time_encoding:
            self.time_projection = SinusoidalTimeEmbedding(
                embed_dim, max_time=1500000.0
            )

        if output_type == "point":
            self.output_layer = nn.Linear(embed_dim, 1)
        else:
            raise ValueError(f"Unknown output type: {output_type}")

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
        extracted = [fe(x_reshaped) for fe in self.feature_extractors]
        features = self.cnn_aggregation(torch.cat(extracted, dim=-1))
        features = features.view(B, T, V_feat, self.embed_dim)

        seg_emb = features.mean(dim=2)

        if self.use_time_encoding:
            time_scalar = x_lifetime.mean(dim=-1).unsqueeze(-1)
            seg_emb = seg_emb + self.time_projection(time_scalar)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            pooled = (seg_emb * mask_expanded).sum(dim=1) / mask_expanded.sum(
                dim=1
            ).clamp(min=1.0)
        else:
            pooled = seg_emb.mean(dim=1)

        output = self.output_layer(pooled).squeeze(-1)

        aux_loss = torch.tensor(0.0, device=x.device)
        return output, None, None, aux_loss
