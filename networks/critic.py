"""Implementations of critic networks for Q-learning."""

from typing import Tuple, Callable
import torch
import torch.nn as nn
from networks.mlp import mlp


class QfuncCriticNetwork(nn.Module):
    """
    Implements a Q-function as a Multi-Layer Perceptron (MLP).

    This Q-function estimates the scalar value Q(s, a) for given state-action pairs (s, a).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Tuple,
        activation: Callable,
    ) -> None:
        """
        Initialize the Q-function critic network.

        Args:
            obs_dim (int): Dimension of the observation (state) space.
            act_dim (int): Dimension of the action space.
            hidden_sizes (tuple): List of sizes for hidden layers in the MLP.
            activation (Callable): Activation function to apply between layers (e.g., nn.ReLU).
        """
        super().__init__()
        self.qfunc = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-value for a given state-action pair.

        Args:
            obs (torch.Tensor): The observation (state) tensor.
            act (torch.Tensor): The action tensor.

        Returns:
            torch.Tensor: The Q-value estimate.
        """

        q = self.qfunc(torch.cat([obs, act], dim=-1))  # Concatenate state & action
        return torch.squeeze(q, -1)  # output is a scalar per input sample
