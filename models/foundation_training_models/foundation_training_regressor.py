"""
Foundation-training regressor: embedder + compute unit.

Pipeline: segment embedder → variate gated attention → [time emb] →
compute unit (TCN / BiLSTM / Transformer / MLP / RNN) → segment gated
attention → MLP → scalar or Weibull head.

The embedder is intended to be contrastively pretrained, then frozen
while the compute unit and head are trained for remaining-lifetime
prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.building_blocs.bilstm_temporal import BiLSTMTemporal
from models.building_blocs.causal_cnn_features_extractor import (
    CausalCNNFeatureExtractor,
)
from models.building_blocs.cnn_features_extractor import CNNFeatureExtractor
from models.building_blocs.gated_attention import GatedAttention
from models.building_blocs.mlp_temporal import MLPTemporal
from models.building_blocs.tcn_temporal import TCNTemporal
from models.building_blocs.time_embedding import SinusoidalTimeEmbedding
from models.building_blocs.transformer_temporal import TransformerTemporal
from models.building_blocs.vanilla_rnn_temporal import VanillaRNNTemporal


class FoundationTrainingRegressor(nn.Module):
    """
    Two-part foundation-training model.

    Args:
        embedder_type: ``"cnn"`` | ``"causal_cnn"``
        compute_type: ``"tcn"`` | ``"bilstm"`` | ``"transformer"`` | ``"mlp"`` | ``"rnn"``
        output_type: ``"point"`` | ``"gaussian"`` | ``"weibull"``
        freeze_embedder: if True, embedder params do not receive gradients
    """

    def __init__(
        self,
        segment_len,
        embed_dim=64,
        dropout=0.15,
        embedder_type="cnn",
        compute_type="tcn",
        temporal_params=None,
        use_time_encoding=True,
        output_type="point",
        freeze_embedder=False,
        causal_kernel_size=7,
    ):
        super().__init__()
        if embedder_type not in ("cnn", "causal_cnn"):
            raise ValueError(
                f"Unknown embedder_type '{embedder_type}'. "
                "Expected 'cnn' or 'causal_cnn'."
            )
        if compute_type not in ("tcn", "bilstm", "transformer", "mlp", "rnn"):
            raise ValueError(
                f"Unknown compute_type '{compute_type}'. "
                "Expected one of: tcn, bilstm, transformer, mlp, rnn."
            )

        self.embed_dim = embed_dim
        self.embedder_type = embedder_type
        self.compute_type = compute_type
        self.output_type = output_type
        self.use_time_encoding = use_time_encoding
        self._embedder_frozen = False

        temporal_params = temporal_params or {}

        # --- Embedder ---
        if embedder_type == "cnn":
            self.embedder = CNNFeatureExtractor(
                input_len=segment_len, embedding_dim=embed_dim
            )
        else:
            self.embedder = CausalCNNFeatureExtractor(
                input_len=segment_len,
                embedding_dim=embed_dim,
                kernel_size=causal_kernel_size,
            )

        self.variate_attention = GatedAttention(
            dim=embed_dim, hidden_dim=max(embed_dim // 4, 1)
        )

        if self.use_time_encoding:
            self.time_projection = SinusoidalTimeEmbedding(
                embed_dim, max_time=1500000.0
            )

        # --- Compute unit ---
        self.temporal_model, temporal_out_dim = self._build_compute_unit(
            compute_type, embed_dim, dropout, temporal_params
        )

        self.segment_attention = GatedAttention(
            dim=temporal_out_dim, hidden_dim=max(temporal_out_dim // 4, 1)
        )

        hidden = max(temporal_out_dim // 2, 1)
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(temporal_out_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        if self.output_type == "point":
            self.output_layer = nn.Linear(hidden, 1)
        elif self.output_type in ("gaussian", "weibull"):
            self.output_layer = nn.Linear(hidden, 2)
        else:
            raise ValueError(f"Unknown output type: {self.output_type}")

        # Optional projection head used only during contrastive pretraining
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, max(embed_dim // 2, 8)),
        )

        if freeze_embedder:
            self.freeze_embedder()

    @staticmethod
    def _build_compute_unit(compute_type, embed_dim, dropout, temporal_params):
        if compute_type == "tcn":
            model = TCNTemporal(
                embed_dim,
                kernel_size=temporal_params.get("kernel_size", 5),
                num_levels=temporal_params.get("num_levels", 5),
                dropout=dropout,
                dropout_1d=temporal_params.get("dropout_1d", False),
            )
            return model, embed_dim
        if compute_type == "bilstm":
            model = BiLSTMTemporal(
                embed_dim, temporal_params.get("bilstm_layers", 1)
            )
            return model, embed_dim
        if compute_type == "rnn":
            model = VanillaRNNTemporal(
                embed_dim, num_layers=temporal_params.get("num_layers", 1)
            )
            return model, embed_dim
        if compute_type == "transformer":
            model = TransformerTemporal(
                embed_dim,
                num_layers=temporal_params.get("num_layers", 1),
                num_heads=temporal_params.get("num_heads", 4),
                dropout=temporal_params.get("dropout", dropout),
            )
            return model, embed_dim
        # mlp
        model = MLPTemporal(
            embed_dim, dropout=temporal_params.get("dropout", 0.0)
        )
        return model, embed_dim

    def freeze_embedder(self):
        for param in self.embedder.parameters():
            param.requires_grad = False
        self.embedder.eval()
        self._embedder_frozen = True

    def unfreeze_embedder(self):
        for param in self.embedder.parameters():
            param.requires_grad = True
        self._embedder_frozen = False

    def embedder_parameters(self):
        return list(self.embedder.parameters()) + list(
            self.projection_head.parameters()
        )

    def compute_parameters(self):
        """Trainable params excluding the (typically frozen) embedder."""
        embedder_ids = {id(p) for p in self.embedder.parameters()}
        proj_ids = {id(p) for p in self.projection_head.parameters()}
        skip = embedder_ids | proj_ids
        return [p for p in self.parameters() if id(p) not in skip]

    def encode_channels(self, x_channels):
        """
        Encode raw channel segments.

        Args:
            x_channels: (N, 1, L)
        Returns:
            embeddings: (N, embed_dim)
        """
        if self._embedder_frozen:
            with torch.no_grad():
                return self.embedder(x_channels)
        return self.embedder(x_channels)

    def project(self, embeddings):
        """L2-normalised projection used by contrastive InfoNCE."""
        z = self.projection_head(embeddings)
        return F.normalize(z, dim=-1)

    def _embed_trajectory(self, x, mask=None):
        """Map ``(B, T, V, L)`` to segment embeddings ``(B, T, D)``."""
        B, T, V, L = x.shape

        if self.use_time_encoding:
            x_features = x[:, :, :-1, :]
            x_lifetime = x[:, :, -1, :]
            V_feat = V - 1
        else:
            x_features = x
            V_feat = V

        x_reshaped = x_features.reshape(B * T * V_feat, 1, L)
        features = self.encode_channels(x_reshaped)
        features = features.view(B * T, V_feat, self.embed_dim)

        v_weights = self.variate_attention(features, mask=None)
        seg_emb = torch.sum(features * v_weights, dim=1).view(B, T, self.embed_dim)

        if self.use_time_encoding:
            time_scalar = x_lifetime.mean(dim=-1).unsqueeze(-1)
            seg_emb = seg_emb + self.time_projection(time_scalar)

        return seg_emb, v_weights

    def forward(self, x, mask=None):
        B, T, _, _ = x.shape
        seg_emb, v_weights = self._embed_trajectory(x, mask=mask)

        temporal_out, aux_loss = self.temporal_model(seg_emb, mask=mask)

        s_weights = self.segment_attention(temporal_out, mask=mask)
        context_vector = torch.sum(temporal_out * s_weights, dim=1)

        reg_features = self.regressor(context_vector)
        output = self.output_layer(reg_features).squeeze(-1)

        if self.output_type == "weibull":
            output = F.softplus(output) + 1e-6

        return output, s_weights, v_weights, aux_loss

    def train(self, mode=True):
        super().train(mode)
        # Keep frozen embedder in eval so BatchNorm/Dropout (if any) stay fixed
        if self._embedder_frozen:
            self.embedder.eval()
        return self
