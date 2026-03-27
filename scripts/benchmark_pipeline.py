import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.train_utils.dataset import LPBSDataset
from torch.utils.data import DataLoader
from models.model_regression import RegressorBenchmarkWrapper
import json

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

    test_loader = DataLoader(
        dataset, 
        batch_size=first_model_config["params"].get("batch_size", 8), 
        shuffle=False
    )

    for model_name, config in models_config.items():
        print(f"\n=== Benchmarking {model_name} ===")
        model_cls = config["model_class"]
        params = config.get("params", {})
        ckpt_path = config.get("checkpoint_path")
        
        if not ckpt_path or not os.path.exists(ckpt_path):
            print(f"Checkpoint not found for {model_name} at {ckpt_path}. Skipping.")
            continue
            
        model_wrapper = model_cls(params)
        model_wrapper.load(ckpt_path)
        
        measures = model_wrapper.benchmark(test_loader)
        models_results[model_name] = measures
        print(f"Results for {model_name}: {measures}")

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
        "regr_64e_huber": {
            "model_class": RegressorBenchmarkWrapper,
            "checkpoint_path": "ckpts/best_regressor_model.pth", 
            "params": {
                "batch_size": 8,
                "loss": "huber",
                "embed_dim": 64,
                "device": "cuda:0",
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