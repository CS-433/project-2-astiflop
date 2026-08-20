"""
MiniROCKET multivariate feature extractor for per-segment embeddings.

Fits ``sktime``'s ``MiniRocketMultivariate`` once on training segments, then
transforms ``(B, T, V, L)`` tensors into ``(B, T, F)`` feature sequences.
"""

import numpy as np
import torch
import torch.nn as nn


class RocketSegmentFeatureExtractor(nn.Module):
    """
    Frozen ROCKET frontend for segment sequences.

    Call :meth:`fit` on an array of shape ``(N, V, L)`` before training the
    readout. Transformed features have no gradient (kernels are fixed).
    """

    def __init__(self, num_kernels=1000, random_state=0):
        super().__init__()
        self.num_kernels = int(num_kernels)
        self.random_state = int(random_state)
        self._rocket = None
        self.output_dim = None
        self._is_fitted = False

    @property
    def is_fitted(self):
        return self._is_fitted

    def fit(self, segments):
        """
        Args:
            segments: ``np.ndarray`` of shape ``(N, V, L)``.
        """
        from sktime.transformations.panel.rocket import MiniRocketMultivariate

        if segments.ndim != 3:
            raise ValueError(
                f"ROCKET fit expects (N, V, L), got shape {segments.shape}"
            )

        rocket = MiniRocketMultivariate(
            num_kernels=self.num_kernels,
            random_state=self.random_state,
        )
        features = rocket.fit_transform(segments)
        features = np.asarray(features, dtype=np.float32)
        self._rocket = rocket
        self.output_dim = int(features.shape[1])
        self._is_fitted = True
        return self

    def get_rocket(self):
        return self._rocket

    def set_rocket(self, rocket):
        self._rocket = rocket
        self._is_fitted = rocket is not None
        if rocket is not None and self.output_dim in (None, 0):
            n_kernels = getattr(rocket, "num_kernels_", None) or self.num_kernels
            # Stable dims observed for MiniRocketMultivariate (see esn_regressor).
            known = {500: 420, 1000: 924, 2000: 1932}
            self.output_dim = known.get(int(n_kernels))

    def transform_numpy(self, segments):
        if not self._is_fitted or self._rocket is None:
            raise RuntimeError("RocketSegmentFeatureExtractor must be fit before transform")
        features = self._rocket.transform(segments)
        return np.asarray(features, dtype=np.float32)

    def forward(self, x_features):
        """
        Args:
            x_features: ``(B, T, V, L)`` feature channels only (no lifetime).

        Returns:
            ``(B, T, F)`` ROCKET features (detached).
        """
        if not self._is_fitted or self._rocket is None:
            raise RuntimeError(
                "ROCKET extractor is not fitted. Call fit() in the training wrapper "
                "before the optimization loop."
            )

        B, T, V, L = x_features.shape
        flat = x_features.detach().cpu().numpy().reshape(B * T, V, L)
        feats = self.transform_numpy(flat)
        if self.output_dim is None:
            self.output_dim = int(feats.shape[1])
        feats = feats.reshape(B, T, -1)
        return torch.from_numpy(feats).to(
            device=x_features.device, dtype=x_features.dtype
        )
