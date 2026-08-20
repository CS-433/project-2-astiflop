"""
Regression wrappers for classical Echo State Network (ESN) architectures.

Training is **offline ridge regression** on reservoir states (no epochs,
no backprop). Only the linear readout ``W_out, b`` is learned.

Config format
-------------
::

    {
      "<model_key>": {
        "wrappers": {
          "training": "ESNTrainingWrapper",
          "benchmark": "ESNBenchmarkWrapper",
          "visualization": "ESNVisualizationWrapper"
        },
        "params": { ... }
      }
    }

Shared ``params``
-----------------
name                    str     Checkpoint filename prefix
model_type              str     Must be ``"esn"``
feature_extractor       str     ``"raw"`` | ``"cnn"`` | ``"rocket"``
units                   int     Reservoir size (fixed to 500 in search configs)
embed_dim               int     CNN embedding dim (cnn extractor only)
segment_len             int     Input segment length (e.g. 900)
feature_extractor_layers int    CNN depth (cnn extractor only)
use_time_encoding       bool    Strip Lifetime channel and add sin/cos time emb
dropout                 float   Optional dropout on states at fit time only (usually 0)
device                  str     e.g. ``"cuda:0"``
leak_rate               float   Reservoir leak rate (reservoirpy ``lr``)
spectral_radius         float   Reservoir spectral radius (reservoirpy ``sr``)
input_scaling           float   Reservoir input scaling
ridge                   float   L2 regularization for the linear readout
loss_shaping            str     ``residual_learning`` (default) | ``full`` | ``half_cut`` | ``third_cut``
input_connectivity      float   Density of Win (optional, default 0.1)
rc_connectivity         float   Density of W (optional, default 0.1)
reservoir_seed          int     Seed for reservoirpy weight init
num_variates            int     Feature channels excluding Lifetime (default 3; auto-inferred if omitted)
rocket_num_kernels      int     MiniROCKET kernels (rocket extractor only)
batch_size              int     DataLoader batch size (feature extraction only)
"""

import os
import random
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from models.wrappers import BenchmarkWrapper, TrainingWrapper, VisualizationWrapper
from utils.train_utils.model_factory import build_regressor


MAX_SEGMENT_NUMBER = 150
RESIDUAL_BASELINE = 80


def denormalization_scale(loss_shaping, max_segment_number=MAX_SEGMENT_NUMBER):
    """Map normalized model outputs back to segment-count RUL scale."""
    if loss_shaping == "half_cut":
        return max_segment_number / 2.0
    if loss_shaping == "full":
        return float(max_segment_number)
    if loss_shaping == "residual_learning":
        return float(max_segment_number - RESIDUAL_BASELINE)
    return max_segment_number / 3.0


def shaped_target(t, T_actual, loss_shaping, max_segment_number=MAX_SEGMENT_NUMBER):
    """Build a scalar RUL target for staircase index ``t`` (1-based length)."""
    if loss_shaping == "third_cut":
        y = min(T_actual - t, max_segment_number // 3)
        return 3.0 * float(y) / max_segment_number
    if loss_shaping == "half_cut":
        y = min(T_actual - t, max_segment_number // 2)
        return 2.0 * float(y) / max_segment_number
    if loss_shaping == "full":
        y = min(T_actual - t, max_segment_number)
        return float(y) / max_segment_number
    if loss_shaping == "residual_learning":
        return (RESIDUAL_BASELINE - t) / (max_segment_number - RESIDUAL_BASELINE)
    raise ValueError(f"Unknown loss_shaping: {loss_shaping}")


def _collect_rocket_segments(training_loader, use_time_encoding, max_segments=512):
    segments = []
    for batch_data, _, total_lengths in training_loader:
        B = batch_data.shape[0]
        for i in range(B):
            T_actual = int(total_lengths[i].item())
            traj = batch_data[i, :T_actual]
            feats = traj[:, :-1, :] if use_time_encoding else traj
            for t in range(T_actual):
                segments.append(feats[t].cpu().numpy())
                if len(segments) >= max_segments:
                    return np.stack(segments, axis=0).astype(np.float32)
    if not segments:
        raise RuntimeError("No segments collected to fit MiniROCKET")
    return np.stack(segments, axis=0).astype(np.float32)


def _save_esn_checkpoint(model, name):
    datetime_str = time.strftime("%H-%M")
    os.makedirs("ckpts", exist_ok=True)
    path = f"ckpts/best_{name}_{datetime_str}.pth"
    payload = {
        "state_dict": model.state_dict(),
        "ridge": getattr(model, "ridge", None),
        "readout_fitted": getattr(model, "readout_fitted", False),
    }
    rocket = getattr(model, "rocket_checkpoint_payload", lambda: None)()
    if rocket is not None:
        payload["rocket"] = rocket
        payload["feature_extractor"] = "rocket"
    torch.save(payload, path)
    return path, datetime_str


def load_esn_checkpoint(params, checkpoint_path=None, device=None):
    if device is None:
        device = params["device"]
    model = build_regressor(params, device=device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            if ckpt.get("rocket") is not None and hasattr(model, "load_rocket_checkpoint"):
                model.load_rocket_checkpoint(ckpt["rocket"])
            model.load_state_dict(ckpt["state_dict"])
            model.readout_fitted = bool(ckpt.get("readout_fitted", True))
            if ckpt.get("ridge") is not None:
                model.ridge = float(ckpt["ridge"])
        else:
            model.load_state_dict(ckpt)
            model.readout_fitted = True

    model.eval()
    return model


def _extract_states_and_targets(
    model,
    data_loader,
    device,
    loss_shaping,
    max_segment_number,
    is_training,
    num_samples_train=4,
    val_stride=10,
):
    """Run the frozen reservoir on staircase prefixes; return states and targets."""
    state_chunks = []
    target_chunks = []

    model.eval()
    with torch.no_grad():
        for batch_data, _, total_lengths in tqdm(
            data_loader, desc="ESN states", leave=False
        ):
            B = batch_data.shape[0]
            X_staircase = []
            Y_staircase = []

            for i in range(B):
                T_actual = int(total_lengths[i].item())
                full_trajectory = batch_data[i]

                if is_training:
                    if T_actual <= num_samples_train:
                        indices = list(range(1, T_actual + 1))
                    else:
                        indices = random.sample(
                            range(1, T_actual + 1), num_samples_train
                        )
                else:
                    indices = list(range(1, T_actual + 1, val_stride))
                    if indices[-1] != T_actual:
                        indices.append(T_actual)

                for t in indices:
                    X_staircase.append(full_trajectory[:t])
                    Y_staircase.append(
                        shaped_target(t, T_actual, loss_shaping, max_segment_number)
                    )

            if not X_staircase:
                continue

            X_padded = pad_sequence(X_staircase, batch_first=True).to(device)
            indices = torch.arange(X_padded.size(1), device=device).expand(
                len(X_staircase), -1
            )
            lengths_tensor = torch.tensor(
                [len(x) for x in X_staircase], device=device
            ).unsqueeze(1)
            mask = (indices < lengths_tensor).float()

            states = model.encode_reservoir_state(X_padded, mask=mask)
            state_chunks.append(states.cpu())
            target_chunks.append(torch.tensor(Y_staircase, dtype=torch.float32))

    if not state_chunks:
        raise RuntimeError("No reservoir states collected for ridge regression")

    return torch.cat(state_chunks, dim=0), torch.cat(target_chunks, dim=0)


class ESNTrainingWrapper(TrainingWrapper):
    """Fit an ESN readout with a single offline ridge regression."""

    def train_on_fold(self, training_loader, validation_loader):
        name = self.params.get("name")
        device = self.params.get("device")
        ridge = self.params.get("ridge", 1e-5)
        loss_shaping = self.params.get("loss_shaping", "residual_learning")
        max_segment_number = MAX_SEGMENT_NUMBER

        # Infer feature-channel count from data (Lifetime is a separate last channel).
        sample_x, _, _ = next(iter(training_loader))
        n_channels = int(sample_x.shape[2])
        if self.params.get("use_time_encoding", True):
            inferred = max(n_channels - 1, 1)
        else:
            inferred = n_channels
        if self.params.get("num_variates") != inferred:
            print(
                f"num_variates: config={self.params.get('num_variates')}, "
                f"inferred from data={inferred} (using inferred)"
            )
            self.params["num_variates"] = inferred

        model = build_regressor(self.params, device=device)

        if self.params.get("feature_extractor") == "rocket":
            print("Fitting MiniROCKET on training segments...")
            segments = _collect_rocket_segments(
                training_loader,
                use_time_encoding=self.params.get("use_time_encoding", True),
            )
            model.fit_rocket(segments)
            model = model.to(device)
            print(
                f"MiniROCKET fitted on {len(segments)} segments "
                f"(feature dim={model.rocket_extractor.output_dim})"
            )

        print("Collecting training reservoir states...")
        train_states, train_targets = _extract_states_and_targets(
            model,
            training_loader,
            device,
            loss_shaping,
            max_segment_number,
            is_training=True,
        )
        print(
            f"Fitting ridge readout on {len(train_targets)} samples "
            f"(ridge={ridge:g}, units={model.units})..."
        )
        model.fit_readout(train_states, train_targets, ridge=ridge)

        print("Evaluating on validation states...")
        val_states, val_targets = _extract_states_and_targets(
            model,
            validation_loader,
            device,
            loss_shaping,
            max_segment_number,
            is_training=False,
        )
        model.eval()
        with torch.no_grad():
            val_states = val_states.to(device)
            preds = (val_states @ model.W_out).squeeze(-1) + model.b_out.squeeze(-1)
            preds = preds.cpu()
            mse = float(torch.mean((preds - val_targets) ** 2).item())
            mae = float(torch.mean(torch.abs(preds - val_targets)).item())

        path, datetime_str = _save_esn_checkpoint(model, name)
        print(
            f"ESN ridge fit done — val MSE: {mse:.4f}, val MAE: {mae:.4f} "
            f"(saved {path} at {datetime_str})"
        )

        return {"best_loss": mse, "mse_validation": mse, "mae_validation": mae}, model


class ESNBenchmarkWrapper(BenchmarkWrapper):
    """Benchmark a ridge-fitted ESN on full worm trajectories."""

    def load(self, path):
        device = self.params.get("device")
        self.model = load_esn_checkpoint(self.params, path, device=device)

    def benchmark(self, test_loader):
        device = self.params.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        loss_shaping = self.params.get("loss_shaping", "residual_learning")
        scale = denormalization_scale(loss_shaping)

        all_trajectory_preds = []
        all_trajectory_vars = []

        with torch.no_grad():
            for X, _, total_segment_len in test_loader:
                B = X.shape[0]
                X = X.cpu()

                for i in range(B):
                    T_actual = int(total_segment_len[i].item())
                    full_trajectory = X[i]
                    X_staircase = [full_trajectory[:t] for t in range(1, T_actual + 1)]

                    X_padded = pad_sequence(X_staircase, batch_first=True).to(device)
                    indices = torch.arange(X_padded.size(1), device=device).expand(
                        len(X_staircase), -1
                    )
                    lengths_tensor = torch.tensor(
                        [len(x) for x in X_staircase], device=device
                    ).unsqueeze(1)
                    mask = (indices < lengths_tensor).float()

                    trajectory_preds, _, _, _ = self.model(X_padded, mask=mask)
                    trajectory_preds = trajectory_preds.cpu().numpy()
                    trajectory_vars = np.where(
                        trajectory_preds * scale > 45, 10.0, 2.0
                    )
                    trajectory_preds = trajectory_preds * scale

                    all_trajectory_preds.append(trajectory_preds)
                    all_trajectory_vars.append(trajectory_vars)

        return {
            "predictions": all_trajectory_preds,
            "variances": all_trajectory_vars,
            "interpretability_score": 6.5,
        }


class ESNVisualizationWrapper(VisualizationWrapper):
    """Step-by-step trajectory visualization for ridge-fitted ESNs."""

    def load(self, path):
        device = self.params.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = load_esn_checkpoint(self.params, path, device=device)

    def get_trajectory_predictions(self, data_tensor, total_segments):
        device = self.params.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        loss_shaping = self.params.get("loss_shaping", "residual_learning")
        scale = denormalization_scale(loss_shaping)
        T_actual = int(total_segments)

        predictions = []
        variances = []

        with torch.no_grad():
            for t in range(1, T_actual + 1):
                x_t = data_tensor[:t].unsqueeze(0).to(device)
                mask = torch.ones(1, t, device=device)
                output, _, _, _ = self.model(x_t, mask=mask)
                pred_val = output.item() * scale
                predictions.append(pred_val)
                variances.append(10.0 if pred_val > 45 else 2.0)

        return predictions, variances, {}
