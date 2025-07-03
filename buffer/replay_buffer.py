"""Replay buffer for off-policy RL agents."""

from typing import Dict
import torch
import numpy as np


class ReplayBuffer:
    """
    A simple FIFO replay buffer for SAC agents.
    Stores and samples transitions (s, a, r, s', done) for off-policy RL training.
    """

    def __init__(self, obs_dim: int, act_dim: int, size: int) -> None:
        """
        Initialize the replay buffer.

        Args:
            obs_dim (int): Dimension of the observation space.
            act_dim (int): Dimension of the action space.
            size (int): Maximum number of experiences to store in the buffer.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        # Pre-allocate storage for transitions
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size,), dtype=np.float32)
        self.done_buf = np.zeros((size,), dtype=np.float32)

        self.ptr, self.size, self.max_size = (
            0,
            0,
            size,
        )  # Pointer and buffer size tracking

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: float,
    ) -> None:
        """
        Store a transition (s, a, r, s', done) in the buffer.

        Args:
            obs (np.ndarray): Current state (shape: (obs_dim,))
            act (np.ndarray): Action taken (shape: (act_dim,))
            rew (float): Reward received.
            next_obs (np.ndarray): Next state after action (shape: (obs_dim,))
            done (float): Whether the episode terminated (1.0 for done, 0.0 otherwise).
        """
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew  # Scalar reward
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done

        # Move pointer, wrap around if buffer is full
        """Circular buffer -> It allows the buffer to wrap around and overwrite the oldest data once the maximum size is reached."""
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int) -> Dict:
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            Dict: Batch of transitions in PyTorch tensor format.
        """
        idxs = np.random.randint(0, self.size, batch_size)  # Random indices
        batch = dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
            indices = idxs,
            weights=np.ones(batch_size))
        
        # Convert to PyTorch tensors and move to the correct device
        return {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for k, v in batch.items()
        }
    def get_all(self, batch_size) -> Dict:
        idxs = np.arange(self.size, dtype=int)
        batch = dict(obs=self.obs_buf[idxs],
                     next_obs=self.next_obs_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     indices = idxs,
                    weights=np.ones(batch_size))
        
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    def update_priorities(
        self, batch_indices: np.ndarray, batch_priorities: np.ndarray
    ) -> None:
        pass

    def get_beta(self) -> float:
        return 0.0
