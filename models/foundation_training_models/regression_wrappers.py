"""
Regression wrappers for the foundation-training family
(``model_type="foundation_training"``).

Training is done in two stages inside ``train_on_fold``:

1. Contrastive pretraining of the segment embedder (InfoNCE / SimCLR-style).
2. Lifespan (RUL) prediction with the embedder frozen and only the compute
   unit + head updated.

Config format
-------------
Used by ``scripts/training_pipeline.py``, ``scripts/benchmark_pipeline.py``, and
``scripts/visualization_pipeline.py``::

    {
      "<model_key>": {
        "wrappers": {
          "training": "FoundationTrainingTrainingWrapper",
          "benchmark": "FoundationTrainingBenchmarkWrapper",
          "visualization": "FoundationTrainingVisualizationWrapper"
        },
        "params": { ... }
      }
    }

Shared ``params`` (required to build/load the model via ``build_regressor``)
---------------------------------------------------------------------------
name                    str     Checkpoint filename prefix
model_type              str     Must be ``"foundation_training"``
embedder_type           str     ``"cnn"`` | ``"causal_cnn"``
compute_type            str     ``"tcn"`` | ``"bilstm"`` | ``"transformer"`` | ``"mlp"`` | ``"rnn"``
embed_dim               int     Embedding / hidden dimension
segment_len             int     Input segment length (must match the dataset)
use_time_encoding       bool    Strip Lifetime channel and add sin/cos time emb
dropout                 float   Dropout rate
loss                    str     ``"mse"`` (scalar) | ``"weibull"`` (Weibull params)
device                  str     e.g. ``"cuda:0"``, ``"cpu"``

``compute_type``-specific ``params`` (required when that type is selected)
---------------------------------------------------------------------------
tcn:          kernel_size (int), num_levels (int), dropout_1d (bool)
bilstm:       bilstm_layers (int)
rnn:          rnn_layers (int)
transformer:  transformer_layers (int), transformer_heads (int)
mlp:          (no extra keys beyond shared ``dropout``)
causal_cnn:   causal_kernel_size (int, optional, default 7)

Contrastive-stage ``params``
----------------------------
contrastive_epochs      int     Max epochs for embedder pretraining (default 50)
contrastive_lr          float   Adam LR for contrastive stage (default 1e-3)
contrastive_patience    int     Early-stopping patience (default 10)
contrastive_temperature float   InfoNCE temperature (default 0.07)
contrastive_samples     int     Segments sampled per worm per step (default 8)

RUL-stage ``params`` (same as other regressors)
-----------------------------------------------
lr              float   Adam learning rate for compute unit
epochs          int     Maximum RUL training epochs
patience        int     Early-stopping patience (tracks validation MSE)
batch_size      int     DataLoader batch size (read by ``training_pipeline``)
loss_shaping    str     ``"full"`` | ``"third_cut"`` | ``"half_cut"`` (optional)

Example::

    {
        "wrappers": {
            "training": "FoundationTrainingTrainingWrapper",
            "benchmark": "FoundationTrainingBenchmarkWrapper",
            "visualization": "FoundationTrainingVisualizationWrapper"
        },
        "params": {
            "name": "ft_cnn_tcn_64e_16bs_time_do-15_mse",
            "model_type": "foundation_training",
            "embedder_type": "cnn",
            "compute_type": "tcn",
            "kernel_size": 5,
            "num_levels": 5,
            "dropout_1d": false,
            "embed_dim": 64,
            "use_time_encoding": true,
            "dropout": 0.15,
            "loss": "mse",
            "contrastive_epochs": 50,
            "contrastive_lr": 0.001,
            "contrastive_patience": 10,
            "lr": 0.0005,
            "patience": 100,
            "epochs": 500,
            "device": "cuda:0",
            "batch_size": 16,
            "segment_len": 900
        }
    }
"""

import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models.cnn_attention_models.regression_wrappers import (
    RegressorBenchmarkWrapper,
    RegressorTrainingWrapper,
    RegressorVisualizationWrapper,
)
from utils.train_utils.model_factory import build_regressor


def info_nce_loss(z1, z2, temperature=0.07):
    """
    Symmetric NT-Xent (SimCLR) loss on two views of a batch.

    Args:
        z1, z2: (N, D) L2-normalised projections of two augmented views
        temperature: softmax temperature
    """
    n = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2N, D)
    sim = torch.matmul(z, z.t()) / temperature  # (2N, 2N)

    # Mask self-similarity
    mask = torch.eye(2 * n, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float("-inf"))

    # Positives: i ↔ i+N
    targets = torch.arange(n, device=z.device)
    targets = torch.cat([targets + n, targets], dim=0)

    return F.cross_entropy(sim, targets)


def _augment_segment(x, noise_std=0.05, scale_min=0.8, scale_max=1.2):
    """
    Lightweight time-series augmentation for contrastive views.

    Args:
        x: (N, 1, L)
    """
    noise = torch.randn_like(x) * noise_std
    scale = (
        torch.empty(x.size(0), 1, 1, device=x.device)
        .uniform_(scale_min, scale_max)
    )
    return x * scale + noise


class FoundationTrainingTrainingWrapper(RegressorTrainingWrapper):
    """
    Two-stage trainer: contrastive embedder pretraining, then frozen-embedder
    RUL prediction with the selected compute unit.
    """

    def _sample_channel_segments(self, batch_data, total_lengths, n_samples, device):
        """
        Sample raw channel segments from a batch of trajectories.

        Returns tensor of shape ``(N, 1, L)`` on ``device``.
        """
        use_time = self.params.get("use_time_encoding", True)
        segments = []

        for i in range(batch_data.shape[0]):
            T_actual = int(total_lengths[i].item())
            traj = batch_data[i]  # (T_max, V, L)
            if use_time:
                traj = traj[:, :-1, :]  # drop Lifetime channel
            V_feat = traj.shape[1]
            if T_actual < 1 or V_feat < 1:
                continue

            for _ in range(n_samples):
                t = random.randrange(T_actual)
                v = random.randrange(V_feat)
                segments.append(traj[t, v].unsqueeze(0))  # (1, L)

        if not segments:
            # Fallback: first channel of first valid segment
            traj = batch_data[0]
            if use_time:
                traj = traj[:, :-1, :]
            segments.append(traj[0, 0].unsqueeze(0))

        return torch.stack(segments, dim=0).to(device)  # (N, 1, L)

    def _contrastive_step(self, model, batch_data, total_lengths, device):
        n_samples = self.params.get("contrastive_samples", 8)
        temperature = self.params.get("contrastive_temperature", 0.07)

        segments = self._sample_channel_segments(
            batch_data, total_lengths, n_samples, device
        )
        view1 = _augment_segment(segments)
        view2 = _augment_segment(segments)

        emb1 = model.encode_channels(view1)
        emb2 = model.encode_channels(view2)
        z1 = model.project(emb1)
        z2 = model.project(emb2)
        return info_nce_loss(z1, z2, temperature=temperature)

    def _train_embedder_contrastive(self, model, training_loader, validation_loader):
        device = self.params.get("device")
        epochs = self.params.get("contrastive_epochs", 50)
        lr = self.params.get("contrastive_lr", 1e-3)
        patience = self.params.get("contrastive_patience", 10)

        model.unfreeze_embedder()
        optimizer = torch.optim.Adam(model.embedder_parameters(), lr=lr)

        best_val = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in tqdm(
            range(epochs),
            desc=f"Contrastive embedder ({model.embedder_type})",
        ):
            model.train()
            train_loss = 0.0
            for batch_data, _, total_lengths in training_loader:
                loss = self._contrastive_step(
                    model, batch_data, total_lengths, device
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train = train_loss / max(len(training_loader), 1)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_data, _, total_lengths in validation_loader:
                    loss = self._contrastive_step(
                        model, batch_data, total_lengths, device
                    )
                    val_loss += loss.item()
            avg_val = val_loss / max(len(validation_loader), 1)

            if avg_val < best_val:
                best_val = avg_val
                epochs_no_improve = 0
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.embedder.state_dict().items()
                }
                best_proj = {
                    k: v.detach().cpu().clone()
                    for k, v in model.projection_head.state_dict().items()
                }
            else:
                epochs_no_improve += 1

            if epoch % 10 == 0:
                tqdm.write(
                    f"[Contrastive] Epoch {epoch + 1}: "
                    f"train={avg_train:.4f}, val={avg_val:.4f}, "
                    f"patience={epochs_no_improve}/{patience}"
                )

            if epochs_no_improve >= patience:
                break

        if best_state is not None:
            model.embedder.load_state_dict(best_state)
            model.projection_head.load_state_dict(best_proj)

        model.freeze_embedder()
        print(
            f"Contrastive pretraining done (best val InfoNCE={best_val:.4f}). "
            "Embedder frozen."
        )
        return best_val

    def _train_compute_rul(self, model, training_loader, validation_loader):
        """Stage-2 RUL training with frozen embedder (mirrors parent loop)."""
        name = self.params.get("name")
        lr = self.params.get("lr")
        epochs = self.params.get("epochs")
        patience = self.params.get("patience")
        device = self.params.get("device")
        loss_type = self.params.get("loss")
        loss_shaping = self.params.get("loss_shaping", "full")
        max_segment_number = 150

        trainable = [p for p in model.compute_parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(
                "No trainable compute parameters after freezing the embedder."
            )
        optimizer = torch.optim.Adam(trainable, lr=lr)

        if loss_type == "mse":
            criterion = nn.MSELoss()
        elif loss_type == "mae":
            criterion = nn.L1Loss()
        elif loss_type == "huber":
            criterion = nn.SmoothL1Loss()
        elif loss_type == "nll":
            from models.cnn_attention_models.regression_wrappers import (
                gaussian_nll_loss,
            )

            criterion = gaussian_nll_loss
        elif loss_type == "weibull":
            from models.cnn_attention_models.regression_wrappers import (
                weibull_nll_loss,
            )

            criterion = weibull_nll_loss
        elif loss_type == "weibull_shifted":
            from models.cnn_attention_models.regression_wrappers import (
                weibull_nll_loss_shifted,
            )

            criterion = weibull_nll_loss_shifted
        elif loss_type == "weibull_beta":
            from models.cnn_attention_models.regression_wrappers import (
                weibull_nll_loss_beta_penalty,
            )

            criterion = weibull_nll_loss_beta_penalty
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        best_loss = float("inf")
        comparison_criterion = nn.MSELoss()
        best_comparison_loss = float("inf")
        epochs_no_improve = 0
        best_model_state = None

        for epoch in tqdm(
            range(epochs),
            desc=f"RUL compute ({model.compute_type})",
        ):
            model.train()
            train_loss = 0.0
            for batch_data, _, total_lengths in training_loader:
                loss = self._forward_pass(
                    model,
                    batch_data,
                    total_lengths,
                    criterion,
                    loss_shaping,
                    None,
                    device,
                    max_segment_number=max_segment_number,
                    is_training=True,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(training_loader)

            model.eval()
            val_loss = 0.0
            comparison_loss = 0.0
            with torch.no_grad():
                for X, _, total_segment_len in validation_loader:
                    loss, comparison = self._forward_pass(
                        model,
                        X,
                        total_segment_len,
                        criterion,
                        loss_shaping,
                        comparison_criterion,
                        device,
                        max_segment_number=max_segment_number,
                        is_training=False,
                    )
                    val_loss += loss.item()
                    comparison_loss += comparison.item()
            val_loss /= len(validation_loader)
            comparison_loss /= len(validation_loader)

            if val_loss < best_loss:
                best_loss = val_loss

            if comparison_loss < best_comparison_loss:
                best_comparison_loss = comparison_loss
                epochs_no_improve = 0
                best_model_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
            else:
                epochs_no_improve += 1

            if epoch % 10 == 0:
                tqdm.write(
                    f"[RUL] Epoch {epoch + 1}: Train={avg_train_loss:.4f}, "
                    f"Val={val_loss:.4f}, MSE={comparison_loss:.4f}. "
                    f"Patience={epochs_no_improve}/{patience}"
                    f"{' <- Best' if epochs_no_improve == 0 else ''}"
                )

            if epochs_no_improve >= patience:
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model.freeze_embedder()
            datetime_str = time.strftime("%H-%M")
            os.makedirs("ckpts", exist_ok=True)
            torch.save(model.state_dict(), f"ckpts/best_{name}_{datetime_str}.pth")
            print(
                f"Best foundation-training model saved "
                f"(MSE={best_comparison_loss:.4f}) at {datetime_str}"
            )

        return {"best_loss": best_loss, "mse_validation": best_comparison_loss}, model

    def train_on_fold(self, training_loader, validation_loader):
        device = self.params.get("device")
        model = build_regressor(self.params, device=device)

        contrastive_val = self._train_embedder_contrastive(
            model, training_loader, validation_loader
        )
        measures, model = self._train_compute_rul(
            model, training_loader, validation_loader
        )
        measures["contrastive_val_loss"] = contrastive_val
        return measures, model


class FoundationTrainingBenchmarkWrapper(RegressorBenchmarkWrapper):
    """Benchmark a trained foundation-training regressor on full trajectories."""


class FoundationTrainingVisualizationWrapper(RegressorVisualizationWrapper):
    """Step-by-step trajectory visualization for a foundation-training regressor."""
