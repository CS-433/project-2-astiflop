import glob
import json
import os
from pathlib import Path


WRAPPER_REGISTRY = None

PIPELINE_ROLES = ("training", "benchmark", "visualization")


def _build_wrapper_registry():
    """Lazy import so importing this module does not pull every wrapper eagerly."""
    global WRAPPER_REGISTRY
    if WRAPPER_REGISTRY is not None:
        return WRAPPER_REGISTRY

    from models.cnn_attention_models.regression_wrappers import (
        RegressorBenchmarkWrapper,
        RegressorTrainingWrapper,
        RegressorVisualizationWrapper,
    )
    from models.esn_models.regression_wrappers import (
        ESNBenchmarkWrapper,
        ESNTrainingWrapper,
        ESNVisualizationWrapper,
    )
    from models.foundation_models.regression_wrappers import (
        ChronosRULRegressorBenchmarkWrapper,
        ChronosRULRegressorTrainingWrapper,
        ChronosRULRegressorVisualizationWrapper,
        FoundationRegressorBenchmarkWrapper,
        FoundationRegressorTrainingWrapper,
        FoundationRegressorVisualizationWrapper,
    )
    from models.foundation_training_models.regression_wrappers import (
        FoundationTrainingBenchmarkWrapper,
        FoundationTrainingTrainingWrapper,
        FoundationTrainingVisualizationWrapper,
    )
    from models.model_dummies import DummyBenchmarkWrapper, DummyVisualizationWrapper
    from models.simple_regression_models.regression_wrappers import (
        LinearScalarRegressorBenchmarkWrapper,
        LinearScalarRegressorTrainingWrapper,
        LinearScalarRegressorVisualizationWrapper,
        LinearScalarRegressorWrapper,
    )
    from models.transformer_models.regression_wrappers import (
        TransformerRegressorBenchmarkWrapper,
        TransformerRegressorTrainingWrapper,
        TransformerRegressorVisualizationWrapper,
    )

    WRAPPER_REGISTRY = {
        "RegressorTrainingWrapper": RegressorTrainingWrapper,
        "RegressorBenchmarkWrapper": RegressorBenchmarkWrapper,
        "RegressorVisualizationWrapper": RegressorVisualizationWrapper,
        "ESNTrainingWrapper": ESNTrainingWrapper,
        "ESNBenchmarkWrapper": ESNBenchmarkWrapper,
        "ESNVisualizationWrapper": ESNVisualizationWrapper,
        "FoundationRegressorTrainingWrapper": FoundationRegressorTrainingWrapper,
        "FoundationRegressorBenchmarkWrapper": FoundationRegressorBenchmarkWrapper,
        "FoundationRegressorVisualizationWrapper": FoundationRegressorVisualizationWrapper,
        "ChronosRULRegressorTrainingWrapper": ChronosRULRegressorTrainingWrapper,
        "ChronosRULRegressorBenchmarkWrapper": ChronosRULRegressorBenchmarkWrapper,
        "ChronosRULRegressorVisualizationWrapper": ChronosRULRegressorVisualizationWrapper,
        "FoundationTrainingTrainingWrapper": FoundationTrainingTrainingWrapper,
        "FoundationTrainingBenchmarkWrapper": FoundationTrainingBenchmarkWrapper,
        "FoundationTrainingVisualizationWrapper": FoundationTrainingVisualizationWrapper,
        "LinearScalarRegressorWrapper": LinearScalarRegressorWrapper,
        "LinearScalarRegressorTrainingWrapper": LinearScalarRegressorTrainingWrapper,
        "LinearScalarRegressorBenchmarkWrapper": LinearScalarRegressorBenchmarkWrapper,
        "LinearScalarRegressorVisualizationWrapper": LinearScalarRegressorVisualizationWrapper,
        "TransformerRegressorTrainingWrapper": TransformerRegressorTrainingWrapper,
        "TransformerRegressorBenchmarkWrapper": TransformerRegressorBenchmarkWrapper,
        "TransformerRegressorVisualizationWrapper": TransformerRegressorVisualizationWrapper,
        "DummyBenchmarkWrapper": DummyBenchmarkWrapper,
        "DummyVisualizationWrapper": DummyVisualizationWrapper,
    }
    return WRAPPER_REGISTRY


def resolve_config_path(config_arg: str) -> str:
    """
    Resolve a config argument to an existing JSON path.

    Accepts an absolute/relative path, or a bare name looked up under ``config/``.
    """
    candidates = [config_arg]
    if not config_arg.endswith(".json"):
        candidates.append(f"{config_arg}.json")
        candidates.append(os.path.join("config", f"{config_arg}.json"))
    else:
        candidates.append(os.path.join("config", os.path.basename(config_arg)))

    for path in candidates:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        f"Config file not found for '{config_arg}'. Tried: {candidates}"
    )


def get_output_dir_from_config(config_path: str) -> str:
    """Output folder named after the config file stem (e.g. config/foo.json -> foo/)."""
    return Path(config_path).stem


def _resolve_wrapper(wrapper_name, model_name, role, registry):
    if wrapper_name not in registry:
        known = ", ".join(sorted(registry))
        raise KeyError(
            f"Unknown wrappers.{role} '{wrapper_name}' for model '{model_name}'. "
            f"Known wrappers: {known}"
        )
    return registry[wrapper_name]


def load_models_config(config_path: str, role: str | None = None) -> dict:
    """
    Load a models config JSON and select the wrapper for ``role``.

    ``role`` must be one of: ``training``, ``benchmark``, ``visualization``.
    Required when model entries define ``wrappers``; optional for CNN-style configs.

    Expected format::

        {
          "<model_key>": {
            "wrappers": {
              "training": "RegressorTrainingWrapper",
              "benchmark": "RegressorBenchmarkWrapper",
              "visualization": "RegressorVisualizationWrapper"
            },
            "params": { ... }
          }
        }

    Models missing the requested ``role`` wrapper are skipped.
    CNN configs may omit ``wrappers`` entirely (flat or ``params`` hyperparameters).
    """
    if role is not None and role not in PIPELINE_ROLES:
        raise ValueError(
            f"Invalid pipeline role '{role}'. Expected one of {PIPELINE_ROLES}"
        )

    config_path = resolve_config_path(config_path)
    with open(config_path, "r") as f:
        raw = json.load(f)

    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Config '{config_path}' must be a non-empty JSON object")

    registry = _build_wrapper_registry()
    models_config = {}
    for model_name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"Config entry for '{model_name}' must be an object, got {type(entry)}"
            )

        wrappers = entry.get("wrappers")
        if wrappers is None:
            # CNN / param-only entries: keep as-is
            models_config[model_name] = dict(entry)
            continue

        if role is None:
            raise ValueError(
                f"Model '{model_name}' defines 'wrappers' but no pipeline role was "
                f"passed to load_models_config (expected one of {PIPELINE_ROLES})"
            )

        if not isinstance(wrappers, dict):
            raise ValueError(
                f"'wrappers' for model '{model_name}' must be an object, got {type(wrappers)}"
            )

        wrapper_name = wrappers.get(role)
        if wrapper_name is None:
            print(
                f"Skipping '{model_name}': no wrappers.{role} defined in config"
            )
            continue

        if not isinstance(wrapper_name, str):
            raise ValueError(
                f"wrappers.{role} for model '{model_name}' must be a string"
            )

        resolved = {k: v for k, v in entry.items() if k != "wrappers"}
        resolved["wrapper_class"] = _resolve_wrapper(
            wrapper_name, model_name, role, registry
        )
        models_config[model_name] = resolved

    if not models_config:
        raise ValueError(
            f"No usable models found in config '{config_path}'"
            + (f" for role '{role}'" if role else "")
        )

    return models_config


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
    to_remove = []
    for model_name, config in models_config.items():
        if "dummy" in model_name:
            continue
        ckpt_path = get_latest_ckpt(config["params"]["name"], ckpt_dir=ckpt_dir)
        if ckpt_path:
            config["checkpoint_path"] = ckpt_path
            print(f"Found checkpoint for {model_name}: {ckpt_path}")
        elif required:
            raise FileNotFoundError(
                f"No checkpoint found for {model_name} "
                f"with pattern {config['params']['name']}"
            )
        else:
            print(f"No checkpoint found for {model_name} "
                f"with pattern {config['params']['name']}. Ignored.")
            to_remove.append(model_name)
            
    for model_name in to_remove:
        del models_config[model_name]
    return models_config


def load_wrappers_from_config(models_config):
    """Instantiate wrappers and load model weights via each wrapper's ``load``."""
    loaded = {}
    for model_name, config in models_config.items():
        print(f"Loading model {model_name}...")
        params = config["params"]
        wrapper = config["wrapper_class"](params)
        if "dummy" not in model_name:
            wrapper.load(config["checkpoint_path"])
        else:
            wrapper.load()
        loaded[model_name] = wrapper
    return loaded
