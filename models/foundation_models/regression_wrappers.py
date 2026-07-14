"""
Regression wrappers for Hugging Face foundation-model regressors (``model_type="foundation"``).

Config format
-------------
Used by ``scripts/training_pipeline.py``, ``scripts/benchmark_pipeline.py``, and
``scripts/visualization_pipeline.py``::

    models_config = {
        "<model_key>": {
            "wrapper_class": FoundationRegressorTrainingWrapper,
            "params": { ... },
        }
    }

Shared ``params`` (all wrappers — required to build/load the model via ``build_regressor``)
------------------------------------------------------------------------------------------
name                    str     Checkpoint filename prefix
model_type              str     Must be ``"foundation"``
embed_dim               int     Projection dimension after the PatchTST backbone
segment_len             int     Input segment length (must match the dataset, e.g. 900)
use_time_encoding       bool    Append sin/cos time encoding to segment features
dropout                 float   Dropout rate
loss                    str     See loss options below
device                  str     e.g. ``"cuda:0"``, ``"cpu"``
pretrained_model_name   str     Hugging Face model id (default: ibm-granite/granite-timeseries-patchtst)
freeze_backbone         bool    Freeze pretrained PatchTST weights (default: True)

Architecture: PatchTST encoder → gated attention (variate + segment) → MLP head.
No temporal stack is used after the foundation backbone.

Training-only ``params`` (``FoundationRegressorTrainingWrapper``)
-------------------------------------------------------------------
lr              float   Adam learning rate
epochs          int     Maximum training epochs
patience        int     Early-stopping patience (tracks validation MSE)
batch_size      int     DataLoader batch size (read by ``training_pipeline``)

Example::

    {
        "wrapper_class": FoundationRegressorTrainingWrapper,
        "params": {
            "name": "patchtst_attn_mlp_128e_16bs_time_do-15_mse",
            "model_type": "foundation",
            "embed_dim": 128,
            "use_time_encoding": True,
            "dropout": 0.15,
            "freeze_backbone": True,
            "pretrained_model_name": "ibm-granite/granite-timeseries-patchtst",
            "loss": "mse",
            "lr": 5e-4,
            "patience": 100,
            "epochs": 500,
            "device": "cuda:0",
            "batch_size": 16,
            "segment_len": 900,
        },
    }
"""

from models.cnn_attention_models.regression_wrappers import (
    RegressorBenchmarkWrapper,
    RegressorTrainingWrapper,
    RegressorVisualizationWrapper,
)


class FoundationRegressorTrainingWrapper(RegressorTrainingWrapper):
    """Train a PatchTST foundation regressor with staircase sampling and early stopping."""


class FoundationRegressorBenchmarkWrapper(RegressorBenchmarkWrapper):
    """Benchmark a trained PatchTST foundation regressor on full worm trajectories."""


class FoundationRegressorVisualizationWrapper(RegressorVisualizationWrapper):
    """Step-by-step trajectory visualization for a PatchTST foundation regressor."""
