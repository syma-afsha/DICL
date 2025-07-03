"""
Policy Network using Squashed Gaussian Policy for continuous action spaces.
"""

from typing import Tuple, Callable, Optional
import torch
import torch.nn as nn
from torch.distributions import Normal
from networks.mlp import mlp


class SquashedGaussianPolicy(nn.Module):
    """
    Implements a Squashed Gaussian policy for continuous action spaces.

    The policy:
    1. Samples actions from a Gaussian distribution.
    2. Applies a Tanh function to squash the actions within valid limits.
    3. Adjusts log probabilities for stable training.

    Args:
        obs_dim (int): Dimension of the observation space.
        act_dim (int): Dimension of the action space.
        hidden_sizes (Tuple): Sizes of hidden layers for the MLP.
        activation (Callable): Activation function (e.g., ReLU).
        act_offset (torch.Tensor): Offset to shift the action range.
        act_limit (torch.Tensor): Maximum absolute value for actions.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Tuple,
        activation: Callable,
        act_offset: torch.Tensor,
        act_limit: torch.Tensor,
    ) -> None:
        super().__init__()

        print(f"DEBUG: hidden_sizes={hidden_sizes}, act_dim={act_dim}")

        if not isinstance(hidden_sizes[-1], int):
            raise ValueError(f"hidden_sizes[-1] must be an int, got {hidden_sizes[-1]}")

        if not isinstance(act_dim, int):
            raise ValueError(f"act_dim must be an int, got {act_dim}")

        # Feature extractor using MLP
        self.feature_extractor = mlp([obs_dim] + list(hidden_sizes), activation)

        # Store action offset and limit
        self.act_offset = act_offset
        self.act_limit = act_limit

        # Output layers for mean (mu) and log standard deviation (log_std)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False, with_logprob: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the policy.

        Args:
            obs (torch.Tensor): Observation input.
            deterministic (bool): If True, return the mean action (default: False).
            with_logprob (bool): If True, compute log probabilities (default: True).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Squashed actions and log probabilities.
        """

        # Extract features from observation
        extracted_features = self.feature_extractor(obs)

        # Compute mean and log standard deviation
        mu = self.mu_layer(extracted_features)
        log_std = self.log_std_layer(extracted_features)

        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, -20, 2)
        std_dev = torch.exp(log_std)

        # Create Gaussian distribution
        action_dist = Normal(mu, std_dev)
        action = mu if deterministic else action_dist.rsample()

        # Apply Tanh squashing and rescale to target range
        squashed_action = torch.tanh(action)
        action_pi = self.act_limit * squashed_action + self.act_offset

        log_prob = None  # Initialize log_prob as None, in case log probabilities are not needed

        if with_logprob:
            # Compute the log probability of the sampled action under the Gaussian distribution
            log_prob = action_dist.log_prob(action).sum(dim=-1)

            # The formula for the corrected log probability:
            # log p_squashed(a) = log p(a_raw) - sum( log(1 - tanh(a_raw)^2) )
            # To ensure numerical stability, this is rewritten using softplus:
            # log p_squashed(a) = log p(a_raw) - sum( 2 * ( log(2) - a_raw - softplus(-2 * a_raw) ) )

            softplus = torch.nn.Softplus()
            log_prob -= (
                2
                * (
                    torch.log(torch.tensor(2.0, device=action.device))
                    - action
                    - softplus(-2 * action)
                )
            ).sum(dim=-1)

        return action_pi, log_prob
