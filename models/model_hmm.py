import numpy as np
from hmmlearn import hmm
import joblib
import os
import time
import torch

from .wrappers import TrainingWrapper

class HMMRegressionWrapper(TrainingWrapper):
    def __init__(self, params=None):
        super().__init__(params)

    def _prepare_regression_data(self, dataloader):
        X_list = []
        lengths = []
        Y_list = []

        for batch in dataloader:
            x, _, w_id = batch
            # w_id in the PyTorch dataloader behaves as lifespan_segment during return
            x_np = x.cpu().numpy()
            w_id_np = w_id.cpu().numpy()
            
            for i in range(x_np.shape[0]):
                valid_len = np.sum(~np.isnan(x_np[i, :, 0, 0]))
                if valid_len == 0:
                    valid_len = x_np.shape[1]
                
                x_seq = x_np[i, :valid_len, :, :]
                x_seq_flat = x_seq.reshape(valid_len, -1)
                x_seq_flat = np.nan_to_num(x_seq_flat)
                
                X_list.append(x_seq_flat)
                lengths.append(valid_len)
                
                full_lifespan = float(w_id_np[i])
                
                # Create a target for each valid segment mapped as RUL
                y_targets = []
                max_segment_number = 150
                for t in range(1, valid_len + 1):
                    y = min(full_lifespan - t, max_segment_number // 3)
                    y = max(y, 0)
                    y = 3 * float(y) / max_segment_number # Normalized between 0 and 1
                    y_targets.append(y)
                
                Y_list.extend(y_targets)

        X = np.concatenate(X_list, axis=0) if X_list else np.empty((0, 1))
        Y = np.array(Y_list)
        return X, lengths, Y

    def train_on_fold(self, training_loader, validation_loader):
        # 1. Prepare Data
        X_train, lengths_train, Y_train = self._prepare_regression_data(training_loader)
        X_val, lengths_val, Y_val = self._prepare_regression_data(validation_loader)
        
        n_components = self.params.get('n_components', 4)
        
        model_hmm = hmm.GaussianHMM(
            n_components=n_components, 
            covariance_type='diag', 
            n_iter=self.params.get('epochs', 10), 
            random_state=42
        )
        
        if X_train.shape[0] > 0 and X_train.shape[1] > 0:
            model_hmm.fit(X_train, lengths_train)
            
            # Use states probabilities as feature set for predicting RUL
            state_probs_train = model_hmm.predict_proba(X_train)
            
            try:
                from sklearn.linear_model import Ridge
                regressor = Ridge(alpha=1.0)
                regressor.fit(state_probs_train, Y_train)
            except Exception as e:
                print('Error fitting ridge regression on HMM states:', e)
                regressor = None
        else:
            regressor = None

        # 2. Evaluation on validation set
        val_mse = 0.0
        if regressor is not None and X_val.shape[0] > 0:
            state_probs_val = model_hmm.predict_proba(X_val)
            Y_pred = regressor.predict(state_probs_val)
            
            # Calculate MSE
            val_mse = np.mean((Y_val - Y_pred) ** 2)

        # Save best model checkpoint, similarly to regression wrappers.
        if regressor is not None:
            name = self.params.get("name", "hmm_regressor")
            datetime_str = time.strftime("%H-%M")
            os.makedirs("ckpts", exist_ok=True)
            ckpt_path = f"ckpts/best_{name}_{datetime_str}.joblib"
            joblib.dump({"hmm": model_hmm, "regressor": regressor}, ckpt_path)
            print(f"Best HMM model saved at {ckpt_path}")

        return {'best_loss': val_mse, 'comparison_loss': val_mse}, (model_hmm, regressor)

    def load(self, path):
        if path is None:
            raise ValueError("A checkpoint path is required to load HMM regression model.")

        payload = joblib.load(path)
        self.model = payload.get("hmm")
        self.regressor = payload.get("regressor")

        if self.model is None or self.regressor is None:
            raise ValueError("Invalid HMM checkpoint format. Expected keys: 'hmm' and 'regressor'.")

    def benchmark(self, test_loader):
        if not hasattr(self, "model") or self.model is None or not hasattr(self, "regressor") or self.regressor is None:
            raise ValueError("Model not loaded. Call load(path) before benchmark().")

        max_segment_number = 150  # Set in the dataset
        all_trajectory_preds = []
        all_trajectory_vars = []
        all_data_tensors = []

        dump_path = self.params.get("inference_dump_path", None)
        dump_sample_index = int(self.params.get("dump_sample_index", 0))

        for X, _, total_segment_len in test_loader:
            x_np = X.cpu().numpy()
            lengths_np = total_segment_len.cpu().numpy()

            for i in range(x_np.shape[0]):
                T_actual = int(lengths_np[i])
                full_trajectory = x_np[i, :T_actual, :, :]

                trajectory_preds = []
                for t in range(1, T_actual + 1):
                    x_prefix = full_trajectory[:t].reshape(t, -1)
                    x_prefix = np.nan_to_num(x_prefix)

                    state_probs = self.model.predict_proba(x_prefix)
                    # Causal prediction at step t: use the last segment state posterior.
                    pred_norm = float(self.regressor.predict(state_probs[-1:].reshape(1, -1))[0])

                    # Denormalize predictions to true RUL scale (number of segments)
                    pred_denorm = pred_norm * (max_segment_number / 3.0)
                    trajectory_preds.append(pred_denorm)

                trajectory_preds = np.array(trajectory_preds)

                # Variance estimation heuristic aligned with regression wrapper behavior
                trajectory_vars = [10.0 if pred > 45 else 2.0 for pred in trajectory_preds]

                all_trajectory_preds.append(trajectory_preds)
                all_trajectory_vars.append(trajectory_vars)
                all_data_tensors.append(X[i, :T_actual].detach().cpu())

        if dump_path is not None:
            if len(all_trajectory_preds) == 0:
                raise ValueError("No predictions available to export inference dump.")

            if dump_sample_index < 0 or dump_sample_index >= len(all_trajectory_preds):
                raise ValueError(
                    f"dump_sample_index={dump_sample_index} is out of range for {len(all_trajectory_preds)} samples."
                )

            preds = np.asarray(all_trajectory_preds[dump_sample_index], dtype=np.float32)
            vars_ = np.asarray(all_trajectory_vars[dump_sample_index], dtype=np.float32)
            data_tensor = all_data_tensors[dump_sample_index]
            T_actual = int(len(preds))

            # Compatible with regression plot script: true remaining RUL in segments.
            true_remaining = np.array([T_actual - t for t in range(1, T_actual + 1)], dtype=np.float32)
            true_objective = true_remaining.copy()

            # HMM has no attention blocks; provide synthetic weights with compatible shapes.
            if data_tensor.ndim != 3:
                raise ValueError(f"Unexpected data_tensor ndim={data_tensor.ndim}, expected 3 (T, V, L).")

            num_vars = int(data_tensor.shape[1])
            s_weights_cpu = [torch.full((t,), 1.0 / t, dtype=torch.float32) for t in range(1, T_actual + 1)]
            v_weights_cpu = [torch.full((t, num_vars), 1.0 / max(num_vars, 1), dtype=torch.float32) for t in range(1, T_actual + 1)]

            os.makedirs(os.path.dirname(dump_path) or ".", exist_ok=True)
            torch.save(
                {
                    "T_actual": T_actual,
                    "true_objective": torch.tensor(true_objective, dtype=torch.float32),
                    "true_remaining": torch.tensor(true_remaining, dtype=torch.float32),
                    "predictions": torch.tensor(preds, dtype=torch.float32),
                    "variances": torch.tensor(vars_, dtype=torch.float32),
                    "s_weights_cpu": s_weights_cpu,
                    "v_weights_cpu": v_weights_cpu,
                    "data_tensor": data_tensor,
                },
                dump_path,
            )
            print(f"Inference dump saved to {dump_path}")

        return {
            "predictions": all_trajectory_preds,
            "variances": all_trajectory_vars,
            "interpretability_score": 7.5,
        }
