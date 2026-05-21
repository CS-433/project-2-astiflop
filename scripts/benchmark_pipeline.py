import argparse
import glob
import os
import shutil
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json

import numpy as np
from scipy.stats import norm
from torch.utils.data import DataLoader

from models.cnn_attention_models.regression_wrappers import RegressorBenchmarkWrapper
from utils.train_utils.dataset import LPBSDataset


def compute_mae(preds, targets, vars):
    """
    Classic Mean Absolute Error (MAE) between predictions and targets.
    """
    return np.mean(np.abs(preds - targets))


def compute_tier_mae(preds, targets, vars, tier=1):
    """
    Compute MAE for a specific tier of the trajectory:
    - Tier 1: Early (first third)
    - Tier 2: Middle (second third)
    - Tier 3: Late (last third)
    """
    T_actual = len(preds)
    tier_size = T_actual // 3
    if tier_size == 0:
        return np.mean(np.abs(preds - targets))  # Fallback if sequence too short

    if tier == 1:
        start_idx, end_idx = 0, tier_size
    elif tier == 2:
        start_idx, end_idx = tier_size, 2 * tier_size
    else:  # tier == 3
        start_idx, end_idx = 2 * tier_size, T_actual

    return np.mean(np.abs(preds[start_idx:end_idx] - targets[start_idx:end_idx]))


def compute_stability(preds, targets, vars, last_k=20):
    """
    Compute the variance of the prediction errors in the last k segments to assess stability.
    - Lower -> More stable predictions in the final part of the trajectory.
    - Higher -> More volatile predictions near the end.
    """
    T_actual = len(preds)
    k = min(last_k, T_actual)
    if k > 1:
        diffs = preds[-k:] - targets[-k:]
        return np.var(diffs)
    return 0.0


def compute_nasa_loss(preds, targets, vars):
    """
    Custom NASA-inspired loss that penalizes early and late errors differently.
    - Early errors (pred < target) are penalized with an exponential decay.
    - Late errors (pred > target) are penalized with an exponential growth.
    """
    d = preds - targets  # Now predictions and targets are in actual segments
    return np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1))


def compute_earlyness(
    preds, targets, vars, epsilon_m=5.0, epsilon_v=25.0, window_size=5
):
    """
    Compute the earlyness factor, which measures how early in the trajectory the model's predictions become stable and accurate.
    - The model is considered to have "converged" at time t if:
        1. The MAE of predictions up to time t is less than epsilon_m.
        2. The variance of the prediction errors in a window of size `window_size` before time t is less than epsilon_v.
    - The earlyness factor is then defined as (T_actual - t_converged)

    Higher is better.
    """

    T_actual = len(preds)
    earlyness = 0
    for t in range(1, T_actual):
        current_mae = np.abs(preds[t - 1] - targets[t - 1])
        window = min(window_size, t)
        current_var = np.var(
            preds[max(0, t - window) : t] - targets[max(0, t - window) : t]
        )

        if current_mae < epsilon_m and current_var < epsilon_v:
            earlyness = T_actual - t
            break
    return earlyness


def compute_nll(preds, targets, variances):
    """
    Compute the Negative Log-Likelihood (NLL) of the predictions assuming a Gaussian distribution.
    """
    if variances is None or np.all(variances == 0):
        return np.nan
    # NLL for Gaussian: 0.5*log(2*pi*var) + (target-pred)^2 / (2*var)
    eps = 1e-6
    nll = 0.5 * np.log(2 * np.pi * (variances + eps)) + ((targets - preds) ** 2) / (
        2 * (variances + eps)
    )
    return np.mean(nll)


def compute_crps(preds, targets, variances):
    """
    Compute the Continuous Ranked Probability Score (CRPS) for Gaussian predictions.
    - CRPS is a proper scoring rule that evaluates the quality of probabilistic forecasts (taking account of the uncertainty in the predictions).

    Lower CRPS is better, with 0 being a perfect score.
    """

    if variances is None or np.all(variances == 0):
        return np.mean(np.abs(preds - targets))
    # CRPS for Normal Distribution
    sig = np.sqrt(variances + 1e-6)
    loc = (targets - preds) / sig
    crps = sig * (
        loc * (2 * norm.cdf(loc) - 1) + 2 * norm.pdf(loc) - 1 / np.sqrt(np.pi)
    )
    return np.mean(crps)


def compute_coverage(preds, targets, variances, confidence=0.95):
    """
    Compute the Prediction Interval Coverage Probability (PICP).
    - Measures the percentage of true targets that fall within the predicted confidence intervals.

    Higher is better, with 1.0 being perfect coverage.
    """
    if variances is None:
        return np.nan
    # PICP: Prediction Interval Coverage Probability
    z = norm.ppf(1 - (1 - confidence) / 2)
    upper = preds + z * np.sqrt(variances)
    lower = preds - z * np.sqrt(variances)
    covered = (targets >= lower) & (targets <= upper)
    return np.mean(covered)


def compute_sharpness(preds, targets, variances):
    """
    Compute the sharpness of the predictive distribution, which is related to the width of the prediction intervals.

    Lower is better, but only if coverage is also good.
    """

    if variances is None:
        return np.nan
    # MPIW: Mean Prediction Interval Width
    return np.mean(2 * 1.96 * np.sqrt(variances))


# List of metric functions to apply. Adding a new metric is as simple as adding its function here.
METRICS_FUNCTIONS = {
    "mae_mean": compute_mae,
    "mae_tier1": lambda p, t, v: compute_tier_mae(p, t, v, tier=1),
    "mae_tier2": lambda p, t, v: compute_tier_mae(p, t, v, tier=2),
    "mae_tier3": lambda p, t, v: compute_tier_mae(p, t, v, tier=3),
    "stability_last20": compute_stability,
    "nasa": compute_nasa_loss,
    "earlyness_factor": compute_earlyness,
    "nll": compute_nll,
    "crps": compute_crps,
    "coverage_95": compute_coverage,
    "sharpness": compute_sharpness,
}


def benchmark_models(
    models_config: dict,
    pytorch_dir: str,
    scaler_config_path: str,
    scaler="standard",
):
    models_results = {}

    # Assuming device is defined similarly across configs
    first_model_config = next(iter(models_config.values()))
    device = first_model_config["params"].get("device")
    print(f"Loading test dataset on device {device}.")

    # Normally mode="test" if dataset supports it, or use subset. We'll load the full set here or what's needed.
    # Adjust as per actual dataset use.
    dataset = LPBSDataset(
        pytorch_dir,
        scaler_type=scaler,
        mode="test",
        scaler_config_path=scaler_config_path,
        device=device,
    )

    for model_name, config in models_config.items():
        print(f"Benchmarking {model_name}...")
        model_cls = config["model_class"]
        params = config.get("params", {})
        ckpt_path = config.get("checkpoint_path")
        test_loader = DataLoader(
            dataset, batch_size=params.get("batch_size", 16), shuffle=False
        )

        model_wrapper = model_cls(params)
        model_wrapper.load(ckpt_path)
        raw_results = model_wrapper.benchmark(test_loader)
        predictions_list = raw_results.get("predictions")
        variances_list = raw_results.get("variances")  # Handle models without variance

        # Build targets
        targets_list = [
            np.array([len(p) - t for t in range(1, len(p) + 1)])
            for p in predictions_list
        ]

        measures = {name: 0.0 for name in METRICS_FUNCTIONS.keys()}
        num_samples = len(predictions_list)

        if num_samples > 0:
            for preds, targets, vars in zip(
                predictions_list, targets_list, variances_list
            ):
                for metric_name, metric_func in METRICS_FUNCTIONS.items():
                    measures[metric_name] += metric_func(preds, targets, vars)

            for metric_name in measures.keys():
                measures[metric_name] = float(measures[metric_name] / num_samples)

        if "interpretability_score" in raw_results:
            measures["Interpretability"] = float(raw_results["interpretability_score"])

        models_results[model_name] = measures

    return models_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark models.")
    parser.add_argument(
        "--pytorch_dir",
        "-d",
        type=str,
        required=True,
        help="Path to PyTorch preprocessed data directory",
    )
    parser.add_argument(
        "--scaler_config_path",
        "-c",
        type=str,
        required=True,
        help="Path to the scaler config JSON file",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="benchmark_results",
        help="Output directory for benchmark results",
    )
    parser.add_argument(
        "--scaler",
        "-s",
        type=str,
        default="standard",
        help="Scaler type: 'none', 'minmax', 'standard'",
    )
    args = parser.parse_args()

    def get_latest_ckpt(name_pattern):
        files = glob.glob(f"ckpts/best_{name_pattern}_[0-9][0-9]-[0-9][0-9].pth")
        if not files:
            return None
        return max(files, key=os.path.getctime)

    device = "cuda:1"
    models_config = {
        "bilstm": {
            "model_class": RegressorBenchmarkWrapper,
            "params": {
                "name": "bilstm_1l_64e_12bs_3fel_time",
                "model_type": "bilstm",
                "bilstm_layers": 1,
                "embed_dim": 64,
                "batch_size": 12,
                "feature_extractor_layers": 3,
                "use_time_encoding": True,
                "loss": "huber",
                "device": device,
                "segment_len": 900,
            },
        },
        "bilstm_gaussian": {
            "model_class": RegressorBenchmarkWrapper,
            "params": {
                "name": "gaussian_1l_64e_12bs_3fel_time",
                "model_type": "gaussian",
                "bilstm_layers": 1,
                "embed_dim": 64,
                "batch_size": 12,
                "feature_extractor_layers": 3,
                "use_time_encoding": True,
                "loss": "nll",
                "device": device,
                "segment_len": 900,
            },
        },
        "tcn_gaussian": {
            "model_class": RegressorBenchmarkWrapper,
            "params": {
                "name": "tcn_5ks_5lvl_64e_16bs_3fel_time_do-15_gaus",
                "model_type": "tcn",
                "kernel_size": 5,
                "num_levels": 5,
                "dropout": 0.15,
                "embed_dim": 64,
                "batch_size": 16,
                "feature_extractor_layers": 3,
                "dropout_1d": False,
                "dropout": 0.15,
                "use_time_encoding": True,
                "loss": "nll",
                "lr": 5e-4,
                "patience": 100,
                "epochs": 500,
                "device": device,
                "segment_len": 900,
            },
        },
        "bilstm_weibull": {
            "model_class": RegressorBenchmarkWrapper,
            "params": {
                "name": "gaussian_1l_64e_12bs_3fel_time_weibull",
                "model_type": "gaussian",
                "bilstm_layers": 1,
                "embed_dim": 64,
                "batch_size": 12,
                "feature_extractor_layers": 3,
                "dropout": 0.15,
                "use_time_encoding": True,
                "loss": "weibull",
                "lr": 5e-4,
                "patience": 100,
                "epochs": 500,
                "device": device,
                "segment_len": 900,
            },
        },
        "tcn_weibull": {
            "model_class": RegressorBenchmarkWrapper,
            "params": {
                "name": "tcn_5ks_5lvl_64e_16bs_3fel_time_do-15_weibull",
                "model_type": "tcn",
                "kernel_size": 5,
                "num_levels": 5,
                "dropout": 0.15,
                "dropout_1d": False,
                "embed_dim": 64,
                "batch_size": 16,
                "feature_extractor_layers": 3,
                "use_time_encoding": True,
                "loss": "weibull",
                "device": device,
                "segment_len": 900,
            },
        },
        "tcn_shifted": {
            "model_class": RegressorBenchmarkWrapper,
            "params": {
                "name": "tcn_5ks_5lvl_64e_16bs_3fel_time_do-15_shifted",
                "model_type": "tcn",
                "kernel_size": 5,
                "num_levels": 5,
                "dropout": 0.15,
                "dropout_1d": False,
                "embed_dim": 64,
                "batch_size": 16,
                "feature_extractor_layers": 3,
                "dropout": 0.15,
                "dropout_1d": False,
                "use_time_encoding": True,
                "loss": "weibull_shifted",
                "weibull_offset": 10.0,
                "lr": 5e-4,
                "patience": 100,
                "epochs": 500,
                "device": device,
                "segment_len": 900,
            },
        },
        "tcn_beta": {
            "model_class": RegressorBenchmarkWrapper,
            "params": {
                "name": "tcn_5ks_5lvl_64e_16bs_3fel_time_do-15_beta",
                "model_type": "tcn",
                "kernel_size": 5,
                "num_levels": 5,
                "dropout": 0.15,
                "dropout_1d": False,
                "embed_dim": 64,
                "batch_size": 16,
                "feature_extractor_layers": 5,
                "use_time_encoding": True,
                "loss": "weibull_beta",
                "weibull_penalty_weight": 2.0,
                "lr": 5e-4,
                "patience": 100,
                "epochs": 500,
                "device": device,
                "segment_len": 900,
            },
        },
    }

    for model_name, config in models_config.items():
        ckpt_path = get_latest_ckpt(config["params"]["name"])
        if ckpt_path:
            config["checkpoint_path"] = ckpt_path
            print(f"Found checkpoint for {model_name}: {ckpt_path}")
        else:
            raise FileNotFoundError(
                f"No checkpoint found for {model_name} with pattern {config['params']['name']}"
            )

    results = benchmark_models(
        models_config,
        pytorch_dir=args.pytorch_dir,
        scaler_config_path=args.scaler_config_path,
        scaler=args.scaler,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    output_json_path = os.path.join(args.output_dir, "results.json")
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved benchmark results to {output_json_path}")

    print(f"\nMoving used checkpoints to {args.output_dir}...")
    for model_name, config in models_config.items():
        ckpt_path = config.get("checkpoint_path")
        if ckpt_path and os.path.exists(ckpt_path):
            shutil.move(ckpt_path, args.output_dir)
            print(f"Moved {ckpt_path}")

    # --- Pretty Print ---
    print("\n" + "=" * 80)
    print(f"{'BENCHMARK RESULTS':^80}")
    print("=" * 80)

    # Get all metric names from the first model evaluated
    if results:
        first_model = list(results.keys())[0]
        metric_names = list(results[first_model].keys())

        # Header
        header = f"{'Model Name':<25} | " + " | ".join(
            [f"{m[:10]:>10}" for m in metric_names]
        )
        print(header)
        print("-" * len(header))

        # Rows
        for model_name, metrics in results.items():
            row = f"{model_name[:25]:<25} | "
            for m in metric_names:
                val = metrics.get(m, 0.0)
                if isinstance(val, (int, float)):
                    row += f"{val:10.4f} | "
                else:
                    row += f"{str(val)[:10]:>10} | "
            print(row)
    print("=" * 80 + "\n")
