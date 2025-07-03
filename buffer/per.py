"""Implementation of Priotized Experience Replay (PER) buffer."""

from typing import Dict
import torch
import numpy as np


class PrioritizedReplayBuffer:
    """Prioritized Replay Buffer for off-policy RL agents."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        size: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
    ) -> None:
        """
        Initialize the PER.

        Args:
            obs_dim (int): Dimension of the observation space.
            act_dim (int): Dimension of the action space.
            size (int): Maximum number of experiences to store in the buffer.
            alpha (float): Weight for prioritized sampling.
            beta_start (float): Initial weight for importance sampling.
            beta_frames (int): Number of frames to anneal beta to 1.0.

        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.alpha = alpha
        self.size = size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.priorities = np.zeros((size,), dtype=np.float32)
        self.pos = 0
        self.frame = 1
        self.size = size
        self.buffer = []

    def beta_by_frame(self, frame_idx: int) -> float:
        """

        When we prioritize certain experiences over others, it introduces bias in training. To counteract this, we use importance sampling weights to adjust the loss.

        β (beta) parameter controls the amount of bias correction.
        If β = 0, no correction is applied (pure prioritization).
        If β = 1, full correction is applied (equivalent to uniform sampling).

        Early Training (Low β):
        The agent relies more on priority-based sampling.
        Helps in faster learning by focusing on important experiences.

        Later Training (High β → 1.0):
        The agent gradually shifts towards uniform sampling.
        Helps in reducing bias and improving stability.

        Start with a small beta value and linearly increase it to 1.0 over beta frames steps.
        """
        return min(
            1.0,
            self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames,
        )

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store a transition (s, a, r, s', done) in the buffer.

        Args:
            obs (np.ndarray): Current state (shape: (obs_dim,))
            act (np.ndarray): Action taken (shape: (act_dim,))
            rew (float): Reward received.
            next_obs (np.ndarray): Next state after action (shape: (obs_dim,))
            done (bool): Whether the episode terminated (1.0 for done, 0.0 otherwise).
        """

        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.size:
            self.buffer.append((obs, act, rew, next_obs, done))

        else:
            self.buffer[self.pos] = (obs, act, rew, next_obs, done)

        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.size

    def sample_batch(self, batch_size: int) -> Dict:
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.
        """
        if len(self.buffer) == self.size:
            priorities = self.priorities
        else:
            priorities = self.priorities[: self.pos]

        probabilities = priorities**self.alpha
        probabilities /= probabilities.sum()
        # Sample transitions
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        # Retrieve samples from buffer
        samples = [self.buffer[idx] for idx in indices]
        beta = self.beta_by_frame(self.frame)
        self.frame += 1
        # Importance sampling weights
        # Formula :  ; N = buffer size, P(i) = priority of i-th sample
        N = len(self.buffer)
        weights = (N * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        obs, acts, rews, next_obs, dones = zip(*samples)

        batch = dict(
            obs=np.array(obs),
            act=np.array(acts),
            rew=np.array(rews),
            next_obs=np.array(next_obs),
            done=np.array(dones),
            indices=indices,
            weights=weights
          
        )

        # for key, value in batch.items():
        #     print(f"DEBUG: {key} - Type: {type(value)}, Shape: {np.array(value).shape}")

        return {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for k, v in batch.items()
        }

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update priorities of sampled transitions after calculating TD errors.

        Args:
            indices (np.ndarray): Indices of sampled transitions.
            priorities (np.ndarray): Updated priorities corresponding to the indices.
        """

        for idx, priority in zip(indices, priorities):
            # Update priorities with the absolute TD errors
            self.priorities[idx] = abs(priority)
    # def update_priorities(self, indices: np.ndarray, priorities: np.ndarray, decay: float = 0.9) -> None:
    #     for idx, new_td in zip(indices, priorities):
    #         # Get the old priority
    #         old_priority = self.priorities[idx]
    #         # Update the priority using a moving average (adaptive update)
    #         updated_priority = decay * old_priority + (1 - decay) * abs(new_td)
    #         self.priorities[idx] = updated_priority


    def __len__(self) -> int:
        return len(self.buffer)
    
    def get_beta(self) -> float:
        return self.beta_by_frame(self.frame)
