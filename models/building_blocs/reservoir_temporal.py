"""
Leaky-integrator reservoir dynamics initialized with ``reservoirpy``.

The random recurrent matrix ``W``, input matrix ``Win`` and bias are frozen
buffers (classic Echo State Network). Only a downstream readout should be
trained.
"""

import numpy as np
import torch
import torch.nn as nn


def _to_dense_float32(matrix):
    if hasattr(matrix, "todense"):
        matrix = matrix.todense()
    return np.asarray(matrix, dtype=np.float32)


def build_reservoirpy_weights(
    input_dim,
    units=500,
    leak_rate=0.3,
    spectral_radius=0.9,
    input_scaling=1.0,
    input_connectivity=0.1,
    rc_connectivity=0.1,
    seed=0,
):
    """Initialize a ``reservoirpy.nodes.Reservoir`` and return dense weight arrays."""
    from reservoirpy.nodes import Reservoir

    reservoir = Reservoir(
        units=units,
        lr=float(leak_rate),
        sr=float(spectral_radius),
        input_scaling=float(input_scaling),
        input_connectivity=float(input_connectivity),
        rc_connectivity=float(rc_connectivity),
        input_dim=int(input_dim),
        seed=int(seed),
    )
    reservoir.initialize(np.zeros((1, input_dim), dtype=np.float64))

    W = _to_dense_float32(reservoir.W)
    Win = _to_dense_float32(reservoir.Win)
    bias = np.asarray(reservoir.bias, dtype=np.float32).reshape(-1)
    if bias.size == 1:
        bias = np.full((units,), float(bias), dtype=np.float32)
    elif bias.size != units:
        bias = np.zeros((units,), dtype=np.float32)

    return W, Win, bias, float(reservoir.lr)


class ReservoirTemporal(nn.Module):
    """
    Batched reservoir over a sequence of feature vectors.

    Update rule (reservoirpy)::

        x[t+1] = (1 - lr) * x[t] + lr * tanh(Win @ u[t+1] + W @ x[t] + bias)

    Args:
        input_dim: Feature dimension per timestep.
        units: Number of reservoir neurons (500 in search configs).
        leak_rate: Neuron leak rate in ``(0, 1]``.
        spectral_radius: Spectral radius of ``W``.
        input_scaling: Gain applied to ``Win`` at initialization.
        input_connectivity: Density of ``Win``.
        rc_connectivity: Density of ``W``.
        seed: RNG seed for ``reservoirpy`` initialization.
    """

    def __init__(
        self,
        input_dim,
        units=500,
        leak_rate=0.3,
        spectral_radius=0.9,
        input_scaling=1.0,
        input_connectivity=0.1,
        rc_connectivity=0.1,
        seed=0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.units = int(units)
        self.leak_rate = float(leak_rate)

        W, Win, bias, lr = build_reservoirpy_weights(
            input_dim=self.input_dim,
            units=self.units,
            leak_rate=leak_rate,
            spectral_radius=spectral_radius,
            input_scaling=input_scaling,
            input_connectivity=input_connectivity,
            rc_connectivity=rc_connectivity,
            seed=seed,
        )
        self.leak_rate = lr

        self.register_buffer("W", torch.from_numpy(W))
        self.register_buffer("Win", torch.from_numpy(Win))
        self.register_buffer("bias", torch.from_numpy(bias))

        for buffer in (self.W, self.Win, self.bias):
            buffer.requires_grad_(False)

    def forward(self, u, mask=None):
        """
        Args:
            u: ``(B, T, F)`` input sequence.
            mask: optional ``(B, T)`` with 1 for valid timesteps.

        Returns:
            states: ``(B, T, units)`` reservoir activations.
            final_state: ``(B, units)`` state at the last valid timestep.
        """
        B, T, F = u.shape
        if F != self.input_dim:
            raise ValueError(
                f"ReservoirTemporal expected input_dim={self.input_dim}, got {F}"
            )

        state = u.new_zeros(B, self.units)
        states = []
        lr = self.leak_rate

        for t in range(T):
            u_t = u[:, t, :]
            preact = u_t @ self.Win.t() + state @ self.W.t() + self.bias
            candidate = torch.tanh(preact)
            new_state = (1.0 - lr) * state + lr * candidate

            if mask is not None:
                valid = mask[:, t].unsqueeze(-1)
                state = valid * new_state + (1.0 - valid) * state
            else:
                state = new_state
            states.append(state)

        states = torch.stack(states, dim=1)

        if mask is not None:
            lengths = mask.sum(dim=1).long().clamp(min=1)
            idx = (lengths - 1).view(B, 1, 1).expand(B, 1, self.units)
            final_state = states.gather(1, idx).squeeze(1)
        else:
            final_state = states[:, -1, :]

        return states, final_state
