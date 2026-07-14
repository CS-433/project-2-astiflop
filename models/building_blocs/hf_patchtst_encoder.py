import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, PatchTSTModel


class HFPatchTSTEncoder(nn.Module):
    """
    Encodes multivariate time-series segments with a pretrained PatchTST backbone.

    Input:  (Batch, Channels, Length)
    Output: (Batch, Channels, embedding_dim)  — one embedding per input channel
    """

    DEFAULT_PRETRAINED_MODEL = "ibm-granite/granite-timeseries-patchtst"

    def __init__(
        self,
        segment_len,
        embedding_dim=128,
        pretrained_model_name=None,
        freeze_backbone=True,
    ):
        super().__init__()
        self.segment_len = segment_len
        self.embedding_dim = embedding_dim
        pretrained_model_name = pretrained_model_name or self.DEFAULT_PRETRAINED_MODEL

        config = AutoConfig.from_pretrained(pretrained_model_name)
        self.context_length = config.context_length
        self.num_input_channels = config.num_input_channels
        self.backbone_dim = config.d_model

        self.backbone = PatchTSTModel.from_pretrained(pretrained_model_name)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.projection = nn.Linear(self.backbone_dim, embedding_dim)

    def _resize_to_context(self, x):
        """Resample (N, C, L) segments to the pretrained model context length."""
        if x.shape[-1] == self.context_length:
            return x
        return F.interpolate(
            x,
            size=self.context_length,
            mode="linear",
            align_corners=False,
        )

    def _align_channels(self, x):
        """Pad or chunk-truncate channels to match the pretrained channel count."""
        n, c, l = x.shape
        target = self.num_input_channels
        if c == target:
            return x, c
        if c < target:
            pad = x.new_zeros(n, target - c, l)
            return torch.cat([x, pad], dim=1), c
        # Keep the first `target` channels; caller may encode leftovers separately.
        return x[:, :target], target

    def forward(self, x):
        # x: (N, C, L)
        n, c_orig, _ = x.shape
        x = self._resize_to_context(x)

        if c_orig <= self.num_input_channels:
            x_aligned, c_keep = self._align_channels(x)
            # PatchTST expects (batch, sequence_length, num_input_channels)
            hidden = self.backbone(past_values=x_aligned.transpose(1, 2)).last_hidden_state
            # hidden: (N, C_model, num_patches, d_model)
            pooled = hidden.mean(dim=2)[:, :c_keep]
            return self.projection(pooled)

        # More channels than the backbone supports: encode non-overlapping chunks.
        chunks = []
        for start in range(0, c_orig, self.num_input_channels):
            chunk = x[:, start : start + self.num_input_channels]
            chunk_aligned, c_keep = self._align_channels(chunk)
            hidden = self.backbone(
                past_values=chunk_aligned.transpose(1, 2)
            ).last_hidden_state
            pooled = hidden.mean(dim=2)[:, :c_keep]
            chunks.append(pooled)
        pooled = torch.cat(chunks, dim=1)[:, :c_orig]
        return self.projection(pooled)
