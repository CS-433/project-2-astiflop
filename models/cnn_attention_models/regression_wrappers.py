import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from utils.train_utils.model_factory import (
    build_regressor,
    load_regressor_checkpoint,
)
from models.wrappers import BenchmarkWrapper, TrainingWrapper, VisualizationWrapper


def _weibull_positive_params(raw_params):
    alpha = torch.clamp(raw_params[..., 0], min=1e-7, max=1e5)
    beta = torch.clamp(raw_params[..., 1], min=1e-7, max=1e5)
    return alpha, beta


def _weibull_variance(alpha, beta):
    beta = torch.clamp(beta, min=1e-7)
    term_one = torch.exp(torch.lgamma(1.0 + 2.0 / beta))
    term_two = torch.exp(2.0 * torch.lgamma(1.0 + 1.0 / beta))
    variance = alpha.pow(2) * torch.clamp(term_one - term_two, min=0.0)
    return torch.nan_to_num(variance, nan=0.0, posinf=1e12, neginf=0.0)


def weibull_nll_loss(preds, y_true):
    alpha, beta = _weibull_positive_params(preds)
    y_true = torch.clamp(y_true, min=1e-7)

    log_likelihood = (
        torch.log(beta)
        - torch.log(alpha)
        + (beta - 1.0) * (torch.log(y_true) - torch.log(alpha))
        - torch.pow(y_true / alpha, beta)
    )
    return -log_likelihood.mean()


def weibull_nll_loss_shifted(preds, y_true, offset=5.0):
    """
    Deep Survival Weibull Loss with Target Shift.
    Adds a constant offset to y_true to prevent Zero-Bound hedging and variance explosion.
    """
    alpha, beta = _weibull_positive_params(preds)
    eps = 1e-7

    y_true_shifted = y_true + offset
    y_true_shifted = torch.clamp(y_true_shifted, min=eps)

    log_likelihood = (
        torch.log(beta)
        - torch.log(alpha)
        + (beta - 1.0) * (torch.log(y_true_shifted) - torch.log(alpha))
        - torch.pow(y_true_shifted / alpha, beta)
    )
    return -log_likelihood.mean()


def weibull_nll_loss_beta_penalty(preds, y_true, penalty_weight=2.0):
    """
    Deep Survival Weibull Loss with Beta-Forcing Regularizer.
    Penalizes low confidence (low beta) heavily when the worm is near death (y_true near 0).
    """
    alpha, beta = _weibull_positive_params(preds)
    eps = 1e-7

    y_true = torch.clamp(y_true, min=eps)

    log_likelihood = (
        torch.log(beta)
        - torch.log(alpha)
        + (beta - 1.0) * (torch.log(y_true) - torch.log(alpha))
        - torch.pow(y_true / alpha, beta)
    )

    # Beta penalty
    beta_penalty = penalty_weight * (1.0 / beta) * torch.exp(-y_true)

    return (-log_likelihood + beta_penalty).mean()


def gaussian_nll_loss(mu, s, target):
    return 0.5 * torch.mean(((target - mu) ** 2) / (torch.exp(s) + 1e-6) + s)


class RegressorBenchmarkWrapper(BenchmarkWrapper):
    def load(self, path):
        device = self.params.get("device")
        self.model = load_regressor_checkpoint(self.params, path, device=device)

    def benchmark(self, test_loader):
        device = self.params.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        max_segment_number = 150  # Set in the dataset

        loss_type = self.params.get("loss", "mse")

        all_trajectory_preds = []
        all_trajectory_vars = []

        with torch.no_grad():
            for X, _, total_segment_len in test_loader:
                B, T_max, V, L = X.shape
                X = X.cpu()

                for i in range(B):
                    T_actual = int(total_segment_len[i].item())
                    full_trajectory = X[i]

                    X_staircase = []
                    for t in range(1, T_actual + 1):
                        X_staircase.append(full_trajectory[:t])

                    # Batch prediction for this single trajectory
                    X_padded = pad_sequence(X_staircase, batch_first=True).to(device)
                    indices = torch.arange(X_padded.size(1), device=device).expand(
                        len(X_staircase), -1
                    )
                    lengths_tensor = torch.tensor(
                        [len(x) for x in X_staircase], device=device
                    ).unsqueeze(1)
                    mask = (indices < lengths_tensor).float()

                    trajectory_preds, _, _, _ = self.model(X_padded, mask=mask)

                    if self.model.output_type == "gaussian":
                        trajectory_vars_pred = np.exp(
                            trajectory_preds[..., 1].cpu().numpy()
                        )
                        trajectory_preds = trajectory_preds[..., 0].cpu().numpy()
                        trajectory_vars = trajectory_vars_pred * (
                            (max_segment_number / 3.0) ** 2
                        )
                    elif self.model.output_type == "weibull":
                        alpha, beta = _weibull_positive_params(trajectory_preds)

                        if loss_type == "weibull_shifted":
                            offset = self.params.get("weibull_offset", 5.0)
                            alpha = torch.clamp(alpha - offset, min=1e-5)
                        trajectory_preds = alpha.cpu().numpy()
                        trajectory_vars = _weibull_variance(
                            alpha, beta
                        ).cpu().numpy() * ((max_segment_number / 3.0) ** 2)
                    else:
                        trajectory_preds = trajectory_preds.cpu().numpy()
                        trajectory_vars = np.where(trajectory_preds > 45, 10.0, 2.0)

                    # Denormalize predictions to true RUL scale (number of segments)
                    trajectory_preds = trajectory_preds * (max_segment_number / 3.0)

                    all_trajectory_preds.append(trajectory_preds)
                    all_trajectory_vars.append(trajectory_vars)

        return {
            "predictions": all_trajectory_preds,
            "variances": all_trajectory_vars,
            "interpretability_score": 7.5,
        }


class RegressorTrainingWrapper(TrainingWrapper):
    def _forward_pass(
        self,
        model,
        batch_data,
        total_lengths,
        criterion,
        comparison_criterion,
        device,
        max_segment_number,
        is_training=True,
    ):
        B, T_max, V, L = batch_data.shape
        batch_data = batch_data

        # This acts as Ortho Beta for BiLSTM, or NLL Beta for the HMM
        aux_beta = self.params.get("aux_beta", self.params.get("ortho_beta", 0.01))

        X_staircase = []
        Y_staircase = []

        # Sampling parameters
        num_samples_train = 4
        val_stride = 10  # Striding in validation for reproducibility

        for i in range(B):
            T_actual = int(total_lengths[i].item())
            full_trajectory = batch_data[i]  # (T_max, V, L)

            if is_training:
                if T_actual <= num_samples_train:
                    indices = list(range(1, T_actual + 1))
                else:
                    indices = random.sample(range(1, T_actual + 1), num_samples_train)
            else:
                indices = list(range(1, T_actual + 1, val_stride))
                if indices[-1] != T_actual:
                    indices.append(T_actual)

            for t in indices:
                y = min(
                    T_actual - t, max_segment_number // 3
                )  # Reduce difficulty of the task
                y = (
                    3 * float(y) / max_segment_number
                )  # Normalized between 0 and 1 for easier gradients computations
                X_staircase.append(full_trajectory[:t])
                Y_staircase.append(y)

        X_padded = pad_sequence(X_staircase, batch_first=True)
        targets = torch.tensor(Y_staircase, device=device).float()

        # Attention mask
        indices = torch.arange(X_padded.size(1), device=device).expand(
            len(X_staircase), -1
        )
        lengths_tensor = torch.tensor(
            [len(x) for x in X_staircase], device=device
        ).unsqueeze(1)
        mask = (indices < lengths_tensor).float()

        # Forward pass
        preds, _, _, aux_loss = model(X_padded, mask=mask)
        loss_type = self.params.get("loss", "huber")

        if model.output_type == "gaussian":
            mu = preds[..., 0]
            s = preds[..., 1]
            loss = criterion(mu, s, targets) + aux_beta * aux_loss
            y_pred = mu

        elif model.output_type == "weibull":
            alpha, _ = _weibull_positive_params(preds)

            if loss_type == "weibull_shifted":
                offset = self.params.get("weibull_offset", 5.0)
                loss = (
                    weibull_nll_loss_shifted(preds, targets, offset=offset)
                    + aux_beta * aux_loss
                )
                y_pred = alpha - offset

            elif loss_type == "weibull_beta":
                penalty = self.params.get("weibull_penalty_weight", 2.0)
                loss = (
                    weibull_nll_loss_beta_penalty(
                        preds, targets, penalty_weight=penalty
                    )
                    + aux_beta * aux_loss
                )
                y_pred = alpha

            else:
                loss = criterion(preds, targets) + aux_beta * aux_loss
                y_pred = alpha

        else:
            loss = criterion(preds, targets) + aux_beta * aux_loss
            y_pred = preds

        if not is_training and comparison_criterion is not None:
            comparison_loss = comparison_criterion(y_pred, targets)
            return loss, comparison_loss

        return loss

    def train_on_fold(self, training_loader, validation_loader):
        name = self.params.get("name")

        lr = self.params.get("lr")
        epochs = self.params.get("epochs")
        patience = self.params.get("patience")
        device = self.params.get("device")

        loss_type = self.params.get("loss")
        max_segment_number = 150

        model = build_regressor(self.params, device=device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if loss_type == "mse":
            criterion = nn.MSELoss()
        elif loss_type == "mae":
            criterion = nn.L1Loss()
        elif loss_type == "huber":
            criterion = nn.SmoothL1Loss()
        elif loss_type == "nll":
            criterion = gaussian_nll_loss
        elif loss_type == "weibull":
            criterion = weibull_nll_loss
        elif loss_type == "weibull_shifted":
            criterion = weibull_nll_loss_shifted
        elif loss_type == "weibull_beta":
            criterion = weibull_nll_loss_beta_penalty
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        best_loss = float("inf")
        comparison_criterion = nn.MSELoss()
        best_comparison_loss = float("inf")

        epochs_no_improve = 0
        best_model_state = None

        for epoch in tqdm(range(epochs), desc=f"Training {model.__class__.__name__}"):
            model.train()
            train_loss = 0.0

            for batch_data, _, total_lengths in training_loader:
                loss = self._forward_pass(
                    model,
                    batch_data,
                    total_lengths,
                    criterion,
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

            # Validation
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
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1

            # Summary of epoch:
            if epoch % 10 == 0:  # Print every 10 epochs
                tqdm.write(
                    f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Comparison Loss: {comparison_loss:.4f}. Patience: {epochs_no_improve}/{patience} {'<- Best' if epochs_no_improve == 0 else ''}"
                )

            # Early stopping
            if epochs_no_improve >= patience:
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            datetime_str = time.strftime("%H-%M")
            os.makedirs("ckpts", exist_ok=True)
            torch.save(model.state_dict(), f"ckpts/best_{name}_{datetime_str}.pth")
            print(
                f"Best model saved with comparison loss: {best_comparison_loss:.4f} at time {datetime_str}"
            )

        return {"best_loss": best_loss, "mse_validation": best_comparison_loss}, model


class RegressorVisualizationWrapper(VisualizationWrapper):
    def load(self, path):
        device = self.params.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = load_regressor_checkpoint(self.params, path, device=device)

    def get_trajectory_predictions(self, data_tensor, total_segments):
        device = self.params.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        max_segment_number = 150
        T_actual = int(total_segments)

        loss_type = self.params.get("loss", "mse")

        predictions = []
        variances = []
        s_weights_all = []
        v_weights_all = []

        with torch.no_grad():
            for t in range(1, T_actual + 1):
                x_t = data_tensor[:t].unsqueeze(0).to(device)
                mask = torch.ones(1, t).to(device)

                out = self.model(x_t, mask=mask)
                if isinstance(out, tuple):
                    output = out[0]
                    s_weights = (
                        out[1]
                        if len(out) > 1 and out[1] is not None
                        else torch.zeros(1, t)
                    )
                    v_weights = (
                        out[2]
                        if len(out) > 2 and out[2] is not None
                        else torch.zeros(1, t, 3)
                    )
                else:
                    output = out
                    s_weights = torch.zeros(1, t)
                    v_weights = torch.zeros(1, t, 3)

                # Denormalize
                if self.model.output_type == "gaussian":
                    pred_val = output[0, 0].item() * (max_segment_number / 3.0)
                    predictions.append(pred_val)
                    var_val = np.exp(output[0, 1].item()) * (
                        (max_segment_number / 3.0) ** 2
                    )
                    variances.append(var_val)
                elif self.model.output_type == "weibull":
                    alpha, beta = _weibull_positive_params(output)

                    if loss_type == "weibull_shifted":
                        offset = self.params.get("weibull_offset", 5.0)
                        alpha = torch.clamp(alpha - offset, min=1e-5)

                    pred_val = alpha.item() * (max_segment_number / 3.0)
                    predictions.append(pred_val)

                    var_val = _weibull_variance(alpha, beta).item() * (
                        (max_segment_number / 3.0) ** 2
                    )
                    variances.append(var_val)
                else:
                    pred_val = output.item() * (max_segment_number / 3.0)
                    predictions.append(pred_val)
                    variances.append(10.0 if pred_val > 45 else 2.0)

                s_weights_all.append(s_weights.squeeze().cpu().numpy())
                v_weights_all.append(v_weights.squeeze().cpu().numpy())

        custom_data = {"s_weights": s_weights_all, "v_weights": v_weights_all}
        return predictions, variances, custom_data
