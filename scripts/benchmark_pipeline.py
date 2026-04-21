import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.train_utils.dataset import LPBSDataset
from torch.utils.data import DataLoader

from models.model_regression import RegressorBenchmarkWrapper
from models.model_dummies import DummyBenchmarkWrapper

import json
import numpy as np

def compute_mae(preds, targets):
    return np.mean(np.abs(preds - targets))

def compute_tier_mae(preds, targets, tier=1):
    T_actual = len(preds)
    tier_size = T_actual // 3
    if tier_size == 0:
        return np.mean(np.abs(preds - targets)) # Fallback if sequence too short
        
    if tier == 1:
        start_idx, end_idx = 0, tier_size
    elif tier == 2:
        start_idx, end_idx = tier_size, 2 * tier_size
    else:  # tier == 3
        start_idx, end_idx = 2 * tier_size, T_actual

    return np.mean(np.abs(preds[start_idx:end_idx] - targets[start_idx:end_idx]))

def compute_stability(preds, targets, last_k=20):
    T_actual = len(preds)
    k = min(last_k, T_actual)
    if k > 1:
        diffs = preds[-k:] - targets[-k:]
        return np.var(diffs)
    return 0.0

def compute_nasa_loss(preds, targets):
    d = preds - targets  # Now predictions and targets are in actual segments
    return np.sum(np.where(d < 0, np.exp(-d/13.0) - 1, np.exp(d/10.0) - 1))

def compute_earlyness(preds, targets, epsilon_m=5.0, epsilon_v=25.0, window_size=5):
    T_actual = len(preds)
    earlyness = 0
    for t in range(1, T_actual):
        current_mae = np.abs(preds[t-1] - targets[t-1])
        window = min(window_size, t)
        current_var = np.var(preds[max(0, t-window):t] - targets[max(0, t-window):t])
        
        if current_mae < epsilon_m and current_var < epsilon_v:
            earlyness = T_actual - t
            break
    return earlyness

# List of metric functions to apply. Adding a new metric is as simple as adding its function here.
METRICS_FUNCTIONS = {
    "mae_mean": compute_mae,
    "mae_tier1": lambda p, t: compute_tier_mae(p, t, tier=1),
    "mae_tier2": lambda p, t: compute_tier_mae(p, t, tier=2),
    "mae_tier3": lambda p, t: compute_tier_mae(p, t, tier=3),
    "stability_last20": compute_stability,
    "nasa": compute_nasa_loss,
    "earlyness_factor": compute_earlyness,
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
    device = first_model_config["params"].get("device", "cpu")
    print(f"Loading test dataset on device {device}.")
    
    # Normally mode="test" if dataset supports it, or use subset. We'll load the full set here or what's needed.
    # Adjust as per actual dataset use.
    dataset = LPBSDataset(
        pytorch_dir, 
        scaler_type=scaler, 
        mode="test", 
        scaler_config_path=scaler_config_path,
        device=device
    )


    for model_name, config in models_config.items():
        print(f"Benchmarking {model_name}...")
        model_cls = config["model_class"]
        params = config.get("params", {})
        ckpt_path = config.get("checkpoint_path")
        test_loader = DataLoader(
            dataset, 
            batch_size=params.get("batch_size", 16),
            shuffle=False
        )
            
        model_wrapper = model_cls(params)
        model_wrapper.load(ckpt_path)
        raw_results = model_wrapper.benchmark(test_loader)
        
        predictions_list = raw_results.get("predictions", [])
        
        # Build targets systematically in the pipeline to ensure identical evaluation for all models
        targets_list = []
        for preds in predictions_list:
            T_actual = len(preds)
            trajectory_targets = [T_actual - t for t in range(1, T_actual + 1)]
            targets_list.append(np.array(trajectory_targets))
        
        # Compute metrics
        measures = {name: 0.0 for name in METRICS_FUNCTIONS.keys()}
        num_samples = len(predictions_list)
        
        if num_samples > 0:
            for preds, targets in zip(predictions_list, targets_list):
                for metric_name, metric_func in METRICS_FUNCTIONS.items():
                    metric_val = metric_func(preds, targets)
                    measures[metric_name] += metric_val
                    
            for metric_name in measures.keys():
                measures[metric_name] /= num_samples

        if "interpretability_score" in raw_results:
            measures["Interpretability"] = raw_results["interpretability_score"]
            
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
        "--output_json",
        "-o",
        type=str,
        default="benchmark_results.json",
        help="Output JSON file for benchmark results",
    )
    parser.add_argument(
        "--scaler",
        "-s",
        type=str,
        default="standard",
        help="Scaler type: 'none', 'minmax', 'standard'"
    )
    args = parser.parse_args()

    models_config = {
        "dummy_random": {
            "model_class": DummyBenchmarkWrapper,
            "checkpoint_path": None,
            "params": {
                "model_type": "random",
                "device": "cuda"
            }
        },
        "dummy_segment": {
            "model_class": DummyBenchmarkWrapper,
            "checkpoint_path": None,
            "params": {
                "model_type": "segment",
                "device": "cuda"
            }
        },
        "regr_64e_1_1_5e4": {
            "model_class": RegressorBenchmarkWrapper,
            "checkpoint_path": "ckpts/layers/best_regr_64e_bs16_1_1_13-42.pth",
            "params": {
                "name": "regr_64e_bs16_1_1",
                
                "embed_dim": 64,
                "feature_extractor_layers": 1,
                "bilstm_layers": 1,

                "batch_size": 16,
                "loss": "huber",                
                "lr": 5e-4,
                "patience": 25,
                "epochs": 500,
                "device": "cuda",
                "segment_len": 900,
            }
        },

        "regr_64e_3_1_5e4": {
            "model_class": RegressorBenchmarkWrapper,
            "checkpoint_path": "ckpts/layers/best_regr_64e_bs16_3_1_13-56.pth",
            "params": {
                "name": "regr_64e_bs16_3_1",
                
                "embed_dim": 64,
                "feature_extractor_layers": 3,
                "bilstm_layers": 1,

                "batch_size": 16,
                "loss": "huber",                
                "lr": 5e-4,
                "patience": 25,
                "epochs": 500,
                "device": "cuda",
                "segment_len": 900,
            }
        },

        "regr_64e_1_3_5e4": {
            "model_class": RegressorBenchmarkWrapper,
            "checkpoint_path": "ckpts/layers/best_regr_64e_bs16_1_3_14-06.pth",
            "params": {
                "name": "regr_64e_bs16_1_3",
                
                "embed_dim": 64,
                "feature_extractor_layers": 1,
                "bilstm_layers": 3,

                "batch_size": 16,
                "loss": "huber",                
                "lr": 5e-4,
                "patience": 25,
                "epochs": 500,
                "device": "cuda",
                "segment_len": 900,
            }
        },

        "regr_64e_3_3_5e4": {
            "model_class": RegressorBenchmarkWrapper,
            "checkpoint_path": "ckpts/layers/best_regr_64e_bs16_3_3_15-09.pth",
            "params": {
                "name": "regr_64e_bs16_3_3",
                
                "embed_dim": 64,
                "batch_size": 16,
                "feature_extractor_layers": 3,
                "bilstm_layers": 3,

                "loss": "huber",                
                "lr": 5e-4,
                "patience": 25,
                "epochs": 500,
                "device": "cuda",
                "segment_len": 900,
            }
        },

        "regr_128e_3_3_5e4": {
            "model_class": RegressorBenchmarkWrapper,
            "checkpoint_path": "ckpts/layers/best_regr_128e_bs16_3_3_15-29.pth",
            "params": {
                "name": "regr_128e_bs16_3_3",
                
                "embed_dim": 128,
                "batch_size": 16,
                "feature_extractor_layers": 3,
                "bilstm_layers": 3,

                "loss": "huber",                
                "lr": 5e-4,
                "patience": 25,
                "epochs": 500,
                "device": "cuda",
                "segment_len": 900,
            }
        },

        "regr_128e_3_3_1e3": {
            "model_class": RegressorBenchmarkWrapper,
            "checkpoint_path": "ckpts/layers/best_regr_128e_bs16_3_3_1e3_15-42.pth",
            "params": {
                "name": "regr_128e_bs16_3_3_1e3",
                
                "embed_dim": 128,
                "batch_size": 16,
                "feature_extractor_layers": 3,
                "bilstm_layers": 3,

                "loss": "huber",                
                "lr": 1e-3,
                "patience": 25,
                "epochs": 500,
                "device": "cuda",
                "segment_len": 900,
            }
        },
    }
    
    results = benchmark_models(
        models_config,
        pytorch_dir=args.pytorch_dir,
        scaler_config_path=args.scaler_config_path,
        scaler=args.scaler
    )

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nSaved benchmark results to {args.output_json}")
    
    # --- Pretty Print ---
    print("\n" + "="*80)
    print(f"{'BENCHMARK RESULTS':^80}")
    print("="*80)
    
    # Get all metric names from the first model evaluated
    if results:
        first_model = list(results.keys())[0]
        metric_names = list(results[first_model].keys())
        
        # Header
        header = f"{'Model Name':<25} | " + " | ".join([f"{m[:10]:>10}" for m in metric_names])
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
    print("="*80 + "\n")