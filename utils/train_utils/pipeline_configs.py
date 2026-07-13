import glob
import os

from utils.train_utils.model_factory import load_regressor_checkpoint

def get_latest_ckpt(name_pattern, ckpt_dir="ckpts"):
    pattern = os.path.join(
        ckpt_dir, f"best_{name_pattern}_[0-9][0-9]-[0-9][0-9].pth"
    )
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)


def attach_latest_checkpoints(models_config, ckpt_dir="ckpts", required=True):
    """Resolve and attach checkpoint paths for benchmark/visualization pipelines."""
    for model_name, config in models_config.items():
        ckpt_path = get_latest_ckpt(config["params"]["name"], ckpt_dir=ckpt_dir)
        if ckpt_path:
            config["checkpoint_path"] = ckpt_path
            print(f"Found checkpoint for {model_name}: {ckpt_path}")
        elif required:
            raise FileNotFoundError(
                f"No checkpoint found for {model_name} "
                f"with pattern {config['params']['name']}"
            )
    return models_config


def load_wrappers_from_config(models_config):
    """Instantiate wrappers and load model weights via the model factory."""
    loaded = {}
    for model_name, config in models_config.items():
        print(f"Loading model {model_name}...")
        params = config["params"]
        device = params["device"]
        wrapper = config["wrapper_class"](params)
        wrapper.model = load_regressor_checkpoint(
            params, config["checkpoint_path"], device=device
        )
        loaded[model_name] = wrapper
    return loaded
