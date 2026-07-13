import os

import torch

from models.simple_regression_models.linear_scalar_regressor import LinearScalarRegressor
from models.cnn_attention_models.cnn_attention_model import CNNAttentionRegressor


def resolve_output_type(loss_type):
    if loss_type == "nll":
        return "gaussian"
    if loss_type in ["weibull", "weibull_shifted", "weibull_beta"]:
        return "weibull"
    return "point"


def _temporal_params_from_config(model_type, params):
    if model_type == "hmm":
        return {"num_states": params["num_states"]}
    if model_type == "tcn":
        return {
            "kernel_size": params["kernel_size"],
            "num_levels": params["num_levels"],
            "dropout": params["dropout"],
            "dropout_1d": params["dropout_1d"],
        }
    if model_type == "bilstm":
        return {"bilstm_layers": params["bilstm_layers"]}
    if model_type == "rnn":
        return {"num_layers": params["rnn_layers"]}
    if model_type == "transformer":
        return {
            "num_layers": params["transformer_layers"],
            "num_heads": params["transformer_heads"],
            "dropout": params["dropout"],
        }
    if model_type == "mlp":
        return {"dropout": params["dropout"]}
    return {}


def build_regressor(params, device=None):
    """
    Factory for all regression models used in training, benchmarking, and visualization.
    """
    model_type = params["model_type"]
    embed_dim = params["embed_dim"]
    segment_len = params["segment_len"]
    feature_extractor_layers = params["feature_extractor_layers"]
    use_time_encoding = params["use_time_encoding"]
    dropout = params["dropout"]
    loss_type = params["loss"]
    output_type = resolve_output_type(loss_type)

    if model_type == "linear":
        model = LinearScalarRegressor(
            segment_len=segment_len,
            embed_dim=embed_dim,
            feature_extractor_layers=feature_extractor_layers,
            use_time_encoding=use_time_encoding,
            output_type=output_type,
        )
    else:
        temporal_params = _temporal_params_from_config(model_type, params)
        model = CNNAttentionRegressor(
            segment_len=segment_len,
            embed_dim=embed_dim,
            dropout=dropout,
            feature_extractor_layers=feature_extractor_layers,
            temporal_type=model_type,
            temporal_params=temporal_params,
            use_time_encoding=use_time_encoding,
            output_type=output_type,
        )

    if device is not None:
        model = model.to(device)
    return model


def load_regressor_checkpoint(params, checkpoint_path=None, device=None):
    """Build a regressor and optionally load weights from a checkpoint."""
    if device is None:
        device = params["device"]
    model = build_regressor(params, device=device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model
