"""
Echo State Network (ESN) regressors built with ``reservoirpy`` reservoirs.

Classical setup: frozen random reservoir + offline ridge linear readout.
Optional fixed frontends: raw variates, frozen CNN, or MiniROCKET.
"""

from models.esn_models.esn_regressor import ESNRegressor

__all__ = ["ESNRegressor"]
