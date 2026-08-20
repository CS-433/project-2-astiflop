"""
Echo State Network (ESN) regressors for remaining-lifetime prediction.

Classical ESN training: only the linear readout is learned, via offline
ridge regression on reservoir states. Reservoir weights (and fixed
frontends) are never updated with gradient descent.

Architectures
-------------
``feature_extractor="raw"``
    Mean-pool each segment over time; feed variate vectors into the reservoir.
``feature_extractor="cnn"``
    Frozen random CNN encoder, then reservoir over the segment sequence.
``feature_extractor="rocket"``
    Fitted MiniROCKET features per segment, then reservoir.

Readout: ``y = W_out @ x + b`` fitted with ridge regression (``fit_readout``).
"""

import numpy as np
import torch
import torch.nn as nn

from models.building_blocs.cnn_features_extractor import CNNFeatureExtractor
from models.building_blocs.reservoir_temporal import ReservoirTemporal
from models.building_blocs.rocket_features_extractor import RocketSegmentFeatureExtractor
from models.building_blocs.time_embedding import SinusoidalTimeEmbedding


def _freeze_module(module):
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad_(False)


class ESNRegressor(nn.Module):
    """
    Reservoir-computing RUL regressor with a ridge linear readout.

    Forward signature matches other regressors::

        output, s_weights, v_weights, aux_loss = model(x, mask=mask)
    """

    def __init__(
        self,
        segment_len,
        feature_extractor="raw",
        units=500,
        embed_dim=64,
        feature_extractor_layers=1,
        use_time_encoding=True,
        dropout=0.0,
        output_type="point",
        leak_rate=0.3,
        spectral_radius=0.9,
        input_scaling=1.0,
        input_connectivity=0.1,
        rc_connectivity=0.1,
        reservoir_seed=0,
        rocket_num_kernels=1000,
        num_variates=3,
        ridge=1e-5,
    ):
        super().__init__()
        if output_type != "point":
            raise ValueError(
                "Classical ESNs only support point (scalar) readouts via ridge "
                f"regression; got output_type={output_type!r}."
            )

        self.segment_len = segment_len
        self.feature_extractor_type = feature_extractor
        self.embed_dim = embed_dim
        self.use_time_encoding = use_time_encoding
        self.output_type = output_type
        self.units = units
        self.num_variates = num_variates
        self.ridge = float(ridge)
        self.dropout_p = float(dropout)

        if feature_extractor not in ("raw", "cnn", "rocket"):
            raise ValueError(
                f"Unknown feature_extractor '{feature_extractor}'. "
                "Expected 'raw', 'cnn', or 'rocket'."
            )

        self.cnn_extractors = None
        self.cnn_aggregation = None
        self.rocket_extractor = None

        if feature_extractor == "cnn":
            self.cnn_extractors = nn.ModuleList(
                [
                    CNNFeatureExtractor(input_len=segment_len, embedding_dim=embed_dim)
                    for _ in range(feature_extractor_layers)
                ]
            )
            self.cnn_aggregation = nn.Linear(
                embed_dim * feature_extractor_layers, embed_dim
            )
            _freeze_module(self.cnn_extractors)
            _freeze_module(self.cnn_aggregation)
            reservoir_input_dim = embed_dim
        elif feature_extractor == "rocket":
            self.rocket_extractor = RocketSegmentFeatureExtractor(
                num_kernels=rocket_num_kernels,
                random_state=reservoir_seed,
            )
            dummy = np.zeros((8, num_variates, segment_len), dtype=np.float32)
            self.rocket_extractor.fit(dummy)
            reservoir_input_dim = self.rocket_extractor.output_dim
        else:
            # raw: mean-pooled variates go straight into the reservoir
            reservoir_input_dim = num_variates

        self.reservoir_input_dim = reservoir_input_dim

        if use_time_encoding:
            self.time_projection = SinusoidalTimeEmbedding(
                reservoir_input_dim, max_time=1500000.0
            )

        self.reservoir = ReservoirTemporal(
            input_dim=reservoir_input_dim,
            units=units,
            leak_rate=leak_rate,
            spectral_radius=spectral_radius,
            input_scaling=input_scaling,
            input_connectivity=input_connectivity,
            rc_connectivity=rc_connectivity,
            seed=reservoir_seed,
        )

        # Linear readout y = x @ W_out + b  (set by fit_readout)
        self.register_buffer("W_out", torch.zeros(units, 1))
        self.register_buffer("b_out", torch.zeros(1))
        self.readout_fitted = False

    def fit_rocket(self, segments):
        """Fit the MiniROCKET frontend. ``segments`` shape: ``(N, V, L)``."""
        if self.rocket_extractor is None:
            raise RuntimeError("fit_rocket() called but feature_extractor is not 'rocket'")
        self.rocket_extractor.fit(segments)
        feat_dim = self.rocket_extractor.output_dim
        if feat_dim != self.reservoir_input_dim:
            raise ValueError(
                f"ROCKET feature dim changed ({self.reservoir_input_dim} -> {feat_dim}). "
                "Rebuild the model with matching rocket_num_kernels."
            )

    def rocket_checkpoint_payload(self):
        if self.rocket_extractor is None:
            return None
        return self.rocket_extractor.get_rocket()

    def load_rocket_checkpoint(self, rocket):
        if self.rocket_extractor is None or rocket is None:
            return
        self.rocket_extractor.set_rocket(rocket)
        if self.rocket_extractor.output_dim in (None, 0):
            probe = np.zeros((1, self.num_variates, self.segment_len), dtype=np.float32)
            feats = np.asarray(rocket.transform(probe), dtype=np.float32)
            self.rocket_extractor.output_dim = int(feats.shape[1])

    def fit_readout(self, states, targets, ridge=None):
        """
        Offline ridge regression on reservoir states.

        Args:
            states: ``(N, units)`` numpy or torch array of reservoir final states.
            targets: ``(N,)`` or ``(N, 1)`` regression targets.
            ridge: L2 strength (defaults to ``self.ridge``).
        """
        from sklearn.linear_model import Ridge

        if ridge is None:
            ridge = self.ridge

        X = states.detach().cpu().numpy() if torch.is_tensor(states) else np.asarray(states)
        y = targets.detach().cpu().numpy() if torch.is_tensor(targets) else np.asarray(targets)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)

        reg = Ridge(alpha=float(ridge), fit_intercept=True)
        reg.fit(X, y)

        device = self.W_out.device
        self.W_out = torch.from_numpy(reg.coef_.reshape(self.units, 1).astype(np.float32)).to(
            device
        )
        self.b_out = torch.from_numpy(
            np.asarray(reg.intercept_, dtype=np.float32).reshape(1)
        ).to(device)
        self.ridge = float(ridge)
        self.readout_fitted = True
        return reg

    def _split_lifetime(self, x):
        if self.use_time_encoding:
            return x[:, :, :-1, :], x[:, :, -1, :]
        return x, None

    def _encode_segments(self, x_features, x_lifetime):
        B, T, V, L = x_features.shape

        if self.feature_extractor_type == "raw":
            seg_emb = x_features.mean(dim=-1)
            if seg_emb.size(-1) != self.reservoir_input_dim:
                raise ValueError(
                    f"raw extractor / reservoir input_dim mismatch: got "
                    f"{seg_emb.size(-1)} feature channels after stripping Lifetime, "
                    f"but reservoir was built for {self.reservoir_input_dim}. "
                    f"Set params['num_variates'] to the number of non-Lifetime "
                    f"channels (currently {seg_emb.size(-1)})."
                )

        elif self.feature_extractor_type == "cnn":
            flat = x_features.reshape(B * T * V, 1, L)
            extracted = [fe(flat) for fe in self.cnn_extractors]
            features = self.cnn_aggregation(torch.cat(extracted, dim=-1))
            features = features.view(B, T, V, self.embed_dim)
            seg_emb = features.mean(dim=2)

        else:  # rocket
            seg_emb = self.rocket_extractor(x_features)

        if self.use_time_encoding and x_lifetime is not None:
            time_scalar = x_lifetime.mean(dim=-1).unsqueeze(-1)
            seg_emb = seg_emb + self.time_projection(time_scalar)

        return seg_emb

    def encode_reservoir_state(self, x, mask=None):
        """Return final reservoir state ``(B, units)`` for ridge features."""
        x_features, x_lifetime = self._split_lifetime(x)
        seg_emb = self._encode_segments(x_features, x_lifetime)
        _, final_state = self.reservoir(seg_emb, mask=mask)
        return final_state

    def forward(self, x, mask=None):
        final_state = self.encode_reservoir_state(x, mask=mask)
        if self.dropout_p > 0.0 and self.training:
            final_state = torch.nn.functional.dropout(
                final_state, p=self.dropout_p, training=True
            )
        output = (final_state @ self.W_out).squeeze(-1) + self.b_out
        aux_loss = torch.tensor(0.0, device=x.device)
        return output, None, None, aux_loss
