"""
Transformer regressor wrappers (``model_type="transformer_regressor"``).

Config format
-------------
Used by ``scripts/training_pipeline.py``, ``scripts/benchmark_pipeline.py``, and
``scripts/visualization_pipeline.py``::

    models_config = {
        "<model_key>": {
            "wrappers": {
                "training": "TransformerRegressorTrainingWrapper",
                "benchmark": "TransformerRegressorBenchmarkWrapper",
                "visualization": "TransformerRegressorVisualizationWrapper"
            },
            "params": { ... },
        }
    }

Shared ``params`` (required to build/load via ``build_regressor``)
------------------------------------------------------------------
name                     str     Checkpoint filename prefix
model_type               str     Must be ``"transformer_regressor"``
encoder_type             str     ``"direct"`` | ``"cnn"``
embed_dim                int     Segment embedding size (use 16 for the CNN path)
segment_len              int     Input segment length (e.g. 900)
transformer_layers       int     Number of causal transformer blocks
transformer_heads        int     Attention heads (must divide ``embed_dim``)
feature_extractor_layers int     CNN extractors (``cnn`` only; ignored for ``direct``)
use_time_encoding        bool    Strip Lifetime channel and add sin/cos time emb
dropout                  float   Dropout rate
loss                     str     ``"nll"`` (Gaussian) | ``"weibull"`` | ``"weibull_*"`` | point losses
device                    str     e.g. ``"cuda:0"``

Training-only ``params``
------------------------
lr, epochs, patience, batch_size — same as ``RegressorTrainingWrapper``.

Example (CNN → 16-d → 2-layer transformer → Weibull)::

    {
        "wrappers": {
            "training": "TransformerRegressorTrainingWrapper",
            "benchmark": "TransformerRegressorBenchmarkWrapper",
            "visualization": "TransformerRegressorVisualizationWrapper"
        },
        "params": {
            "name": "xfmr_cnn16_2l_4h_do-15_weibull",
            "model_type": "transformer_regressor",
            "encoder_type": "cnn",
            "embed_dim": 16,
            "transformer_layers": 2,
            "transformer_heads": 4,
            "feature_extractor_layers": 1,
            "use_time_encoding": true,
            "dropout": 0.15,
            "loss": "weibull",
            "lr": 0.0005,
            "patience": 100,
            "epochs": 500,
            "device": "cuda:0",
            "batch_size": 16,
            "segment_len": 900
        }
    }
"""

from models.cnn_attention_models.regression_wrappers import (
    RegressorBenchmarkWrapper,
    RegressorTrainingWrapper,
    RegressorVisualizationWrapper,
)


class TransformerRegressorTrainingWrapper(RegressorTrainingWrapper):
    """Train a direct/CNN transformer regressor with staircase sampling and early stopping."""


class TransformerRegressorBenchmarkWrapper(RegressorBenchmarkWrapper):
    """Benchmark a trained transformer regressor on full worm trajectories."""


class TransformerRegressorVisualizationWrapper(RegressorVisualizationWrapper):
    """Step-by-step trajectory visualization for a transformer regressor."""
