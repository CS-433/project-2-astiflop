import torch
import torch.nn as nn
import torch.nn.functional as F

from models.building_blocs.chronos2_encoder import Chronos2Encoder


class ChronosRULRegressor(nn.Module):
    """
    Paper-faithful RUL regressor: frozen Chronos-2 + lightweight MLP head.

    Follows El-Ghoussani et al., "Time-Series Foundation Model Embeddings for
    Remaining Useful Life Estimation" (arXiv:2606.11990):

    1. Build a multivariate context window of length ``context_len`` from the
       observed segment history (resampled along time).
    2. Extract frozen Chronos-2 context embeddings.
    3. Map the pooled embedding through a 2-layer ReLU MLP with dropout;
       a final ReLU enforces non-negative point predictions.

    Only the MLP head is trainable. No gated attention / temporal stack.
    """

    def __init__(
        self,
        segment_len,
        embed_dim=256,
        dropout=0.1,
        use_time_encoding=False,
        output_type="point",
        pretrained_model_name=None,
        freeze_backbone=True,
        context_len=80,
        pooling="reg",
        device=None,
    ):
        super().__init__()
        self.segment_len = segment_len
        self.embed_dim = embed_dim  # MLP hidden width m in the paper
        self.output_type = output_type
        self.use_time_encoding = use_time_encoding
        self.context_len = context_len

        # Lifetime is stripped when use_time_encoding=True (repo convention);
        # Chronos does not use sinusoidal time features (paper: sensor window only).

        self.encoder = Chronos2Encoder(
            pretrained_model_name=pretrained_model_name,
            freeze_backbone=freeze_backbone,
            context_len=context_len,
            pooling=pooling,
            device=device,
        )
        d_model = self.encoder.d_model

        # Paper: Linear(h → m) → ReLU → Dropout(p) → Linear(m → out) → ReLU
        self.head = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, self._output_dim()),
        )

    def _output_dim(self):
        if self.output_type == "point":
            return 1
        if self.output_type in ("gaussian", "weibull"):
            return 2
        raise ValueError(f"Unknown output type: {self.output_type}")

    def _features_from_input(self, x):
        """Strip Lifetime channel when present; Chronos sees sensor streams only."""
        if self.use_time_encoding:
            return x[:, :, :-1, :]
        return x

    def _build_context_windows(self, x, mask=None):
        """
        Map padded segment histories ``(B, T, V, L)`` to Chronos windows
        ``(B, V, context_len)`` by concatenating valid segments along time and
        resampling to the paper context length L.
        """
        B, T, V, L = x.shape
        windows = []
        for i in range(B):
            if mask is not None:
                t_i = int(mask[i].sum().item())
            else:
                t_i = T
            t_i = max(t_i, 1)

            # (t_i, V, L) → (V, t_i * L)
            history = x[i, :t_i].permute(1, 0, 2).reshape(V, -1).unsqueeze(0)
            if history.shape[-1] != self.context_len:
                history = F.interpolate(
                    history,
                    size=self.context_len,
                    mode="linear",
                    align_corners=False,
                )
            windows.append(history.squeeze(0))
        return torch.stack(windows, dim=0)

    def forward(self, x, mask=None):
        # x: (B, T, V, L)
        B = x.shape[0]
        x_features = self._features_from_input(x)
        context = self._build_context_windows(x_features, mask=mask)

        with torch.no_grad():
            embeddings = self.encoder(context)  # (B, d_model)

        # Detach so gradients never flow into the frozen encoder path
        embeddings = embeddings.detach()
        output = self.head(embeddings)

        if self.output_type == "point":
            output = F.relu(output).squeeze(-1)
        elif self.output_type == "weibull":
            output = F.softplus(output) + 1e-6
        else:
            output = output.squeeze(-1) if output.shape[-1] == 1 else output

        # Dummy attention weights for pipeline compatibility
        device = x.device
        t_len = x.shape[1]
        s_weights = torch.zeros(B, t_len, 1, device=device)
        v_weights = torch.zeros(B, 1, 1, device=device)
        if mask is not None:
            # Put unit mass on the last valid segment (paper uses the context end)
            lengths = mask.sum(dim=1).long().clamp(min=1) - 1
            s_weights[torch.arange(B, device=device), lengths, 0] = 1.0
        else:
            s_weights[:, -1, 0] = 1.0

        aux_loss = torch.tensor(0.0, device=device)
        return output, s_weights, v_weights, aux_loss

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.encoder.to(*args, **kwargs)
        return self
