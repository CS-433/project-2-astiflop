import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from models.wrappers import BenchmarkWrapper, TrainingWrapper, VisualizationWrapper
from utils.train_utils.model_factory import (
    build_regressor,
    load_regressor_checkpoint,
)

MAX_SEGMENT_NUMBER = 150


def _denormalize_predictions(preds):
    return preds * (MAX_SEGMENT_NUMBER / 3.0)


def _point_variances(preds, denormalized=False):
    if denormalized:
        return np.where(preds > 45, 10.0, 2.0)
    return np.where(preds * (MAX_SEGMENT_NUMBER / 3.0) > 45, 10.0, 2.0)


class LinearScalarRegressorBenchmarkWrapper(BenchmarkWrapper):
    def load(self, path):
        device = self.params.get("device")
        self.model = load_regressor_checkpoint(self.params, path, device=device)

    def benchmark(self, test_loader):
        device = self.params.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

        all_trajectory_preds = []
        all_trajectory_vars = []

        with torch.no_grad():
            for X, _, total_segment_len in test_loader:
                B, _, _, _ = X.shape
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
                    trajectory_vars = _point_variances(trajectory_preds)
                    trajectory_preds = _denormalize_predictions(trajectory_preds)

                    all_trajectory_preds.append(trajectory_preds)
                    all_trajectory_vars.append(trajectory_vars)

        return {
            "predictions": all_trajectory_preds,
            "variances": all_trajectory_vars,
            "interpretability_score": 9.0,
        }


class LinearScalarRegressorTrainingWrapper(TrainingWrapper):
    def _forward_pass(
        self,
        model,
        batch_data,
        total_lengths,
        criterion,
        comparison_criterion,
        device,
        is_training=True,
    ):
        B, _, _, _ = batch_data.shape

        X_staircase = []
        Y_staircase = []

        num_samples_train = 4
        val_stride = 10

        for i in range(B):
            T_actual = int(total_lengths[i].item())
            full_trajectory = batch_data[i]

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
                y = min(T_actual - t, MAX_SEGMENT_NUMBER // 3)
                y = 3 * float(y) / MAX_SEGMENT_NUMBER
                X_staircase.append(full_trajectory[:t])
                Y_staircase.append(y)

        X_padded = pad_sequence(X_staircase, batch_first=True)
        targets = torch.tensor(Y_staircase, device=device).float()

        indices = torch.arange(X_padded.size(1), device=device).expand(
            len(X_staircase), -1
        )
        lengths_tensor = torch.tensor(
            [len(x) for x in X_staircase], device=device
        ).unsqueeze(1)
        mask = (indices < lengths_tensor).float()

        preds, _, _, _ = model(X_padded.to(device), mask=mask)
        loss = criterion(preds, targets)

        if not is_training and comparison_criterion is not None:
            comparison_loss = comparison_criterion(preds, targets)
            return loss, comparison_loss

        return loss

    def train_on_fold(self, training_loader, validation_loader):
        name = self.params.get("name")

        lr = self.params.get("lr")
        epochs = self.params.get("epochs")
        patience = self.params.get("patience")
        device = self.params.get("device")

        loss_type = self.params.get("loss")

        model = build_regressor(self.params, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if loss_type == "mse":
            criterion = nn.MSELoss()
        elif loss_type == "mae":
            criterion = nn.L1Loss()
        elif loss_type == "huber":
            criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(
                f"LinearScalarRegressor only supports mse, mae, or huber loss, got: {loss_type}"
            )

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
                        comparison_criterion,
                        device,
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

            if epoch % 10 == 0:
                tqdm.write(
                    f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Comparison Loss: {comparison_loss:.4f}. "
                    f"Patience: {epochs_no_improve}/{patience} "
                    f"{'<- Best' if epochs_no_improve == 0 else ''}"
                )

            if epochs_no_improve >= patience:
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            datetime_str = time.strftime("%H-%M")
            os.makedirs("ckpts", exist_ok=True)
            torch.save(model.state_dict(), f"ckpts/best_{name}_{datetime_str}.pth")
            print(
                f"Best model saved with comparison loss: {best_comparison_loss:.4f} "
                f"at time {datetime_str}"
            )

        return {"best_loss": best_loss, "mse_validation": best_comparison_loss}, model


class LinearScalarRegressorVisualizationWrapper(VisualizationWrapper):
    def load(self, path):
        device = self.params.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = load_regressor_checkpoint(self.params, path, device=device)

    def get_trajectory_predictions(self, data_tensor, total_segments):
        device = self.params.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        T_actual = int(total_segments)

        predictions = []
        variances = []

        with torch.no_grad():
            for t in range(1, T_actual + 1):
                x_t = data_tensor[:t].unsqueeze(0).to(device)
                mask = torch.ones(1, t).to(device)

                output, _, _, _ = self.model(x_t, mask=mask)
                pred_val = output.item() * (MAX_SEGMENT_NUMBER / 3.0)
                predictions.append(pred_val)
                variances.append(10.0 if pred_val > 45 else 2.0)

        return predictions, variances, {}


LinearScalarRegressorWrapper = LinearScalarRegressorTrainingWrapper
