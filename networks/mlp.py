""" MLP for SAC (Soft Actor-Critic) algorithm."""

from typing import List, Callable
import torch.nn as nn


def mlp(
    sizes: List, activation: Callable, output_activation: Callable = nn.Identity
) -> Callable:
    """
    Creates a Multi-Layer Perceptron (MLP) model using PyTorch's Sequential API.

    Args:
        sizes (List[int]): Number of neurons in each layer. Non-numeric values are replaced with 256.
        activation (Callable): Activation function applied between hidden layers.
        output_activation (Callable): Activation function for the output layer (default: nn.Identity).

    Returns:
        nn.Sequential: A PyTorch Sequential model representing the MLP.
    """

    # Convert non-int values to 256
    sizes = [int(s) if isinstance(s, (int, float)) else 256 for s in sizes]

    print(f"MLP Sizes: {sizes}")  # Debugging print

    layers = []
    for i in range(len(sizes) - 1):
        is_output_layer = i == len(sizes) - 2
        current_activation = output_activation if is_output_layer else activation
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        layers.append(current_activation())

    return nn.Sequential(*layers)
