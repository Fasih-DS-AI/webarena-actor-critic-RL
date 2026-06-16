"""Shared network building blocks."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def orthogonal_init(layer: nn.Linear, gain: float = 1.0) -> nn.Linear:
    """Orthogonal weight init (standard for on-policy RL stability)."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class MLPTorso(nn.Module):
    """A simple multi-layer perceptron feature extractor with Tanh activations.

    Tanh (rather than ReLU) is a common choice for PPO/A2C — it keeps
    activations bounded and tends to give more stable policy-gradient updates.
    """

    def __init__(self, input_dim: int, hidden_sizes: Sequence[int] = (128, 128)):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(orthogonal_init(nn.Linear(last, h), gain=2.0 ** 0.5))
            layers.append(nn.Tanh())
            last = h
        self.net = nn.Sequential(*layers)
        self.output_dim = last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
