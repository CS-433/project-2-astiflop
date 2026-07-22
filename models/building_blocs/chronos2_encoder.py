import torch
import torch.nn as nn


class Chronos2Encoder(nn.Module):
    """
    Frozen Chronos-2 encoder that maps a multivariate context window to a vector.

    Input:  (Batch, Channels, Length)
    Output: (Batch, d_model)

    Embeddings are taken from ``Chronos2Pipeline.embed`` and pooled to a single
    series-level vector (paper: context embedding fed to the RUL head).
    The backbone is kept as a plain attribute so its weights are not stored in
    ``state_dict`` (reloaded from the pretrained checkpoint on construction).
    """

    DEFAULT_PRETRAINED_MODEL = "amazon/chronos-2"
    POOLING_MODES = ("reg", "mean", "last")

    def __init__(
        self,
        pretrained_model_name=None,
        freeze_backbone=True,
        context_len=80,
        pooling="reg",
        device=None,
    ):
        super().__init__()
        try:
            from chronos import Chronos2Pipeline
        except ImportError as exc:
            raise ImportError(
                "Chronos-2 support requires the `chronos-forecasting` package. "
                "Install it with: pip install chronos-forecasting"
            ) from exc

        if pooling not in self.POOLING_MODES:
            raise ValueError(
                f"Unknown pooling '{pooling}'. Expected one of {self.POOLING_MODES}"
            )

        self.context_len = context_len
        self.pooling = pooling
        pretrained_model_name = pretrained_model_name or self.DEFAULT_PRETRAINED_MODEL

        load_kwargs = {}
        if device is not None:
            load_kwargs["device_map"] = device

        self._pipeline = Chronos2Pipeline.from_pretrained(
            pretrained_model_name, **load_kwargs
        )
        self.d_model = self._resolve_d_model(self._pipeline.model.config)

        if freeze_backbone:
            self._pipeline.model.eval()
            for param in self._pipeline.model.parameters():
                param.requires_grad = False

    @staticmethod
    def _resolve_d_model(config):
        if hasattr(config, "d_model") and config.d_model is not None:
            return int(config.d_model)
        chronos_cfg = getattr(config, "chronos_config", None) or {}
        if isinstance(chronos_cfg, dict) and "d_model" in chronos_cfg:
            return int(chronos_cfg["d_model"])
        raise AttributeError("Could not resolve Chronos-2 d_model from model config")

    def _pool_embedding(self, emb):
        """
        Pool Chronos-2 embeddings to a single vector of size ``d_model``.

        ``emb`` shape: (n_variates, num_patches + 2, d_model)
        The trailing two tokens are [REG] and a masked future patch.
        """
        # Mean over variates → (num_patches + 2, d_model)
        emb = emb.mean(dim=0)
        if self.pooling == "reg":
            # Second-to-last token is [REG]
            return emb[-2]
        if self.pooling == "last":
            # Last context patch (exclude [REG] and masked future)
            return emb[:-2][-1]
        # Mean over context patches only
        return emb[:-2].mean(dim=0)

    def forward(self, x):
        # x: (N, C, L) — already resized to the desired context length upstream
        n = x.shape[0]
        device = x.device

        embeddings, _ = self._pipeline.embed(
            x.detach().cpu(),
            batch_size=max(n, 1),
            context_length=self.context_len,
        )
        pooled = torch.stack(
            [self._pool_embedding(emb) for emb in embeddings], dim=0
        )
        return pooled.to(device=device, dtype=torch.float32)

    def train(self, mode=True):
        # Keep the frozen backbone in eval mode; only the surrounding module trains.
        super().train(mode)
        self._pipeline.model.eval()
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self._pipeline.model.to(*args, **kwargs)
        return self
