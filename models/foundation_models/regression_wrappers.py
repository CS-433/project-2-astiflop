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

Chronos-2 RUL embeddings (``model_type="chronos"``)
---------------------------------------------------
Paper method (El-Ghoussani et al., arXiv:2606.11990): frozen Chronos-2 context
embeddings + 2-layer ReLU MLP head (final ReLU for non-negative point RUL).

Extra / overridden ``params`` for Chronos wrappers
--------------------------------------------------
model_type              str     Must be ``"chronos"``
embed_dim               int     MLP hidden width ``m`` (paper)
context_len             int     Chronos context window length ``L`` (paper; try 5–80+)
pooling                 str     ``"reg"`` | ``"mean"`` | ``"last"`` (default ``"reg"``)
pretrained_model_name   str     Default: ``amazon/chronos-2``
use_time_encoding       bool    If True, drop the Lifetime channel before Chronos
freeze_backbone         bool    Must stay True for the paper setup

Training-only ``params`` (``FoundationRegressorTrainingWrapper`` /
``ChronosRULRegressorTrainingWrapper``)
-------------------------------------------------------------------
lr              float   Adam learning rate (paper uses ``1e-3`` for Chronos)
epochs          int     Maximum training epochs (paper: 50)
patience        int     Early-stopping patience (tracks validation MSE)
batch_size      int     DataLoader batch size (read by ``training_pipeline``)

Example (PatchTST)::

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

Example (Chronos-2 RUL embeddings)::

    {
        "wrappers": {
            "training": "ChronosRULRegressorTrainingWrapper",
            "benchmark": "ChronosRULRegressorBenchmarkWrapper",
            "visualization": "ChronosRULRegressorVisualizationWrapper"
        },
        "params": {
            "name": "chronos2_rul_m256_L80_mse",
            "model_type": "chronos",
            "embed_dim": 256,
            "context_len": 80,
            "pooling": "reg",
            "use_time_encoding": true,
            "dropout": 0.1,
            "freeze_backbone": true,
            "pretrained_model_name": "amazon/chronos-2",
            "loss": "mse",
            "loss_shaping": "full",
            "lr": 0.001,
            "patience": 20,
            "epochs": 50,
            "device": "cuda:0",
            "batch_size": 8,
            "segment_len": 900
        }
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


class ChronosRULRegressorTrainingWrapper(RegressorTrainingWrapper):
    """Train a Chronos-2 RUL embedding regressor (frozen backbone, MLP head only)."""


class ChronosRULRegressorBenchmarkWrapper(RegressorBenchmarkWrapper):
    """Benchmark a trained Chronos-2 RUL embedding regressor on full worm trajectories."""


class ChronosRULRegressorVisualizationWrapper(RegressorVisualizationWrapper):
    """Step-by-step trajectory visualization for a Chronos-2 RUL embedding regressor."""
