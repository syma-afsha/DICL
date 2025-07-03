"""Implements an Actor-Critic network with an Actor (policy) and two Critics (Q-functions)."""

from typing import Tuple, Callable
import torch
import torch.nn as nn
import numpy as np


from networks.actor import SquashedGaussianPolicy
from networks.critic import QfuncCriticNetwork


class ActorCritic(nn.Module):
    """
    Implements an Actor-Critic network with:
    - An Actor (policy) that selects actions.
    - Two Critics (Q-functions) that estimate action-value functions.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Tuple,
        activation: Callable,
        act_limit: torch.Tensor,
        act_offset: torch.Tensor,
    ) -> None:
        """
        Initialize the actor-critic network.

        Args:
            obs_dim (int): Dimension of the observation (state) space.
            act_dim (int): Dimension of the action space.
            hidden_sizes (Tuple): List of sizes for hidden layers in the MLP.
            activation (Callable): Activation function to apply between layers (e.g., nn.ReLU).
            act_limit (torch.tensor): Maximum action limit (scaling factor).
            act_offset (torch.tensor): Offset to shift the action range.
        """
        super().__init__()

        # Initialize the policy (Actor) network.
        self.pi = SquashedGaussianPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            act_limit=act_limit,
            act_offset=act_offset,
        )

        # Initialize two Q-function Critic networks.
        self.qfunc1 = QfuncCriticNetwork(obs_dim, act_dim, hidden_sizes, activation)
        self.qfunc2 = QfuncCriticNetwork(obs_dim, act_dim, hidden_sizes, activation)

    def forward(
        self, obs: torch.Tensor, act: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that computes Q-values using both critic networks.

        Double Q-Learning prevents overestimation bias by using two Q-functions (Q1 & Q2), where one selects the action and the other evaluates it. This improves training stability and policy performance.

        Args:
            obs (torch.Tensor): The observation (state) input.
            act (torch.Tensor): The action input.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The Q-values from both critics (Q1, Q2).
        """
        q1 = self.qfunc1(obs, act)  # Compute Q-value from first critic
        q2 = self.qfunc2(obs, act)  # Compute Q-value from second critic
        return q1, q2  # Return both Q-values

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        """
        Samples an action from the policy network.

        Args:
            obs (torch.Tensor): The observation (state) input.
            deterministic (bool): If True, returns the mean action instead of sampling.

        Returns:
            np.ndarray: The selected action in NumPy format.
        """
        with torch.no_grad():
            action, _ = self.pi(obs, deterministic=deterministic)
            return action.cpu().numpy()  # Convert to NumPy array
