"""Implementation of the Soft Actor-Critic (SAC) agent."""

from typing import Dict, Tuple
import itertools
from copy import deepcopy
import torch
import torch.nn as nn
from torch.optim.adam import Adam
import numpy as np
from networks.actor_critic import ActorCritic


def count_vars(module: nn.Module) -> int:
    """Count the total number of parameters in a PyTorch module."""
    return sum(p.numel() for p in module.parameters())



class Agent:
    """
    Soft Actor-Critic (SAC) agent with Actor-Critic networks and replay buffer updates.

    This class handles:
    - Storing and training the actor (`π`) and critic (`Q1, Q2`).
    - Soft target updates using Polyak averaging.
    - Sampling actions for exploration and exploitation.
    """

    def __init__(self, config: Dict, device: str) -> None:
        """
        Initialize the SAC agent.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing hyperparameters.
            device (str): Device to use ('cpu' or 'cuda').
        """
        self.device = device
        self.gamma = config["agent"]["gamma"]
        self.lr = float(config["agent"]["learning_rate"])
        self.alpha = config["agent"]["sac"]["alpha"]
        # self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        # self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr)
        #  # Target entropy guides the balance between exploration and exploitation
        # # self.target_entropy = -3.0
        # self.alpha = self.log_alpha.exp()
        self.polyak = config["agent"]["polyak"]
        self.seed = config["general"]["seed"]

        # Set up dimensions
        self.obs_dim = config["environment"]["full_obs_dim"]
        self.act_dim = config["environment"]["action_dim"]
        self.hidden_sizes = config["agent"]["hidden_sizes"]
        self.boundary_min = np.array(config["agent"]["boundary_min"], dtype=np.float32)
        self.boundary_max = np.array(config["agent"]["boundary_max"], dtype=np.float32)
        # Action limits
        self.act_offset = np.array(
            (self.boundary_max + self.boundary_min) / 2.0, dtype=np.float32
        )
        self.act_limit = np.array(
            (self.boundary_max - self.boundary_min) / 2.0, dtype=np.float32
        )

        self.act_offset = torch.from_numpy(self.act_offset).to(self.device)
        self.act_limit = torch.from_numpy(self.act_limit).to(self.device)
        self.use_per = config["buffer"]["use_per"]

        # Initialize Actor-Critic
        self.ac = ActorCritic(
            self.obs_dim,
            self.act_dim,
            self.hidden_sizes,
            nn.ReLU,
            self.act_limit,
            self.act_offset,
        ).to(self.device)

        # Create target networks
        self.ac_targ = deepcopy(self.ac)
        for p in self.ac_targ.parameters():
            p.requires_grad = False  # Freeze target network

        # Optimizers
        self.q_params = itertools.chain(
            self.ac.qfunc1.parameters(), self.ac.qfunc2.parameters()
        )
        self.q_optimizer = Adam(self.q_params, lr=self.lr)
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.lr)

        # Count parameters
        self.var_counts = (
            count_vars(self.ac.pi),
            count_vars(self.ac.qfunc1),
            count_vars(self.ac.qfunc2),
        )
        print(
            f"Number of parameters: π: {self.var_counts[0]}, Q1: {self.var_counts[1]}, Q2: {self.var_counts[2]}"
        )
    # def compute_loss_alpha(self, logp_pi: torch.Tensor) -> torch.Tensor:
    #     """
    #     Compute the loss for adjusting the temperature parameter alpha.
    #     """
    #     # Get current alpha value
    #     self.alpha = self.log_alpha.exp()

    #     # The loss is designed to drive the policy's entropy toward the target entropy.
    #     # We detach logp_pi to avoid backpropagating through the policy network when updating alpha.
    #     alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()

    #     return alpha_loss
    def compute_loss_q(self, data: Dict) -> Tuple[torch.Tensor, np.ndarray, Dict]:
        """
        Compute Q-function loss for Soft Actor-Critic (SAC) using Double Q-learning.
        Supports both standard replay buffer and prioritized experience replay (PER).

        Args:
            data (Dict): A batch of transitions from the replay buffer, containing:
                - 'obs': Observations (states)
                - 'act': Actions taken
                - 'rew': Rewards received
                - 'obs2': Next state observations
                - 'done': Done flag (1 if terminal, 0 otherwise)
                - 'weights': Importance sampling weights (only used for PER)

        Returns:
            Tuple[torch.Tensor, Dict]:
                - Loss for Q-networks (torch.Tensor)
                - Logging info with Q-values (Dict)
        """
        # Extract batch data & move to device
        o, a, r, o2, d = (
            data["obs"].float().to(self.device),
            data["act"].to(self.device),
            data["rew"].to(self.device),
            data["next_obs"].float().to(self.device),
            data["done"].to(self.device),
        )

        # Compute current Q-values using both critics
        q1 = self.ac.qfunc1(o, a)
        q2 = self.ac.qfunc2(o, a)

        # Bellman backup for Q-functions (target value computation)
        with torch.no_grad():
            # Compute next action and its log probability using the current policy
            a2, logp_a2 = self.ac.pi(o2)

            # Compute target Q-values using the target critic networks
            q1_pi_targ = self.ac_targ.qfunc1(o2, a2)
            q2_pi_targ = self.ac_targ.qfunc2(o2, a2)

            # Use Double Q-learning to reduce overestimation bias
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

            # Compute the target Q-value using the Bellman equation

            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

     
        weights = data["weights"].to(self.device)  # PER importance sampling weights
        
        loss_q1 = (weights * ((q1 - backup) ** 2)).mean()
        loss_q2 = (weights * ((q2 - backup) ** 2)).mean()

        # Compute TD errors
        td_errors = torch.abs(q1 - backup) + torch.abs(q2 - backup)
        mean_td_error = (td_errors) / 2.0

        # Compute new priorities

        batch_priorities = (mean_td_error + 1e-6).detach().cpu().numpy()
            # Small epsilon for stability

        # else:  # Standard Replay Buffer
        #     loss_q1 = ((q1 - backup) ** 2).mean()
        #     loss_q2 = ((q2 - backup) ** 2).mean()
         
        #     td_errors = torch.abs(q1 - backup) + torch.abs(q2 - backup)
        #     mean_td_error = (td_errors) / 2.0

        #     # Compute new priorities

        #     batch_priorities = (mean_td_error + 1e-6).detach().cpu().numpy()

        # Total Q-loss
        loss_q = loss_q1 + loss_q2

        # Store useful Q-value info for logging
        q_info = {
            "Q1Vals": q1.cpu().detach().numpy(),
            "Q2Vals": q2.cpu().detach().numpy(),
        }

        return loss_q, batch_priorities, q_info

    def compute_loss_pi(self, data: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Compute policy loss for Soft Actor-Critic (SAC).

        Args:
            data (Dict): A batch of transitions from the replay buffer, containing:
                - 'obs': Observations (states)

        Returns:
            Tuple[torch.Tensor, Dict]:
                - Policy loss (torch.Tensor)
                - Logging info with log probabilities (Dict)
        """
        # Extract observations and move to the correct device
        obs = data["obs"].to(self.device)

        # Compute policy action and log probability

        pi, logp_pi = self.ac.pi(obs)

        # Compute Q-values for the new policy action
        q1_pi = self.ac.qfunc1(obs, pi)
        q2_pi = self.ac.qfunc2(obs, pi)

        # Use the minimum Q-value to reduce overestimation bias
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # loss_alpha = self.compute_loss_alpha(logp_pi)
        # self.alpha_optimizer.zero_grad()
        # loss_alpha.backward()
        # self.alpha_optimizer.step()

        # print("Alpha: ", self.alpha.item())

        # Store useful information for logging
        pi_info = {"LogPi": logp_pi.cpu().detach().numpy()}

        return loss_pi, pi_info

    def update(self, data: Dict) -> Tuple[float, float, np.ndarray]:
        """
        Perform one optimization step for Q-functions and policy in Soft Actor-Critic (SAC).

        Args:
            data (Dict[str, torch.Tensor]): A batch of transitions from the replay buffer.

        Returns:
            Tuple[float, float, np.ndarray]:
                - Q-loss value (float)
                - Policy loss value (float)
                - Batch priorities (np.ndarray)
        """
        # Update Q-networks (Critic update)
        self.q_optimizer.zero_grad()
        loss_q, batch_priorities, _ = self.compute_loss_q(data)

        loss_q.backward()
        self.q_optimizer.step()

        ret_loss_q = loss_q.item()

        # Freeze Q-networks to avoid unnecessary gradients during policy update
        for p in self.q_params:
            p.requires_grad = False

        # Update policy (Actor update)
        self.pi_optimizer.zero_grad()
        loss_pi, _ = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()
        

        ret_loss_pi = loss_pi.item()

        # Unfreeze Q-networks for next training step
        for p in self.q_params:
            p.requires_grad = True

        # Soft update target networks (Polyak averaging)
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                # Polyak averaging: θ_target = τθ_target + (1 - τ)θ
                p_targ.data.add_((1 - self.polyak) * p.data)

        return ret_loss_q, ret_loss_pi, batch_priorities

    def get_action(self, obs: np.ndarray, deterministic: bool) -> np.ndarray:
        """
        Get an action from the policy network.

        Args:
            obs (np.ndarray): The observation state.
            deterministic (bool): Whether to return deterministic actions.

        Returns:
            np.ndarray: The selected action.
        """
        obs_tensor = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        action = self.ac.act(obs_tensor, deterministic)
        
        return action.squeeze(0)  #  returns a NumPy array
    

    def save_model(self, model_path: str, mode: str) -> None:
        """ Save the model "
        Args:
            model_path (str): Path to save the model.
            mode (str): Mode to save the model ("all", "pi", "q").
        """
        if mode == "all":
            torch.save(self.ac.pi.state_dict(), model_path+"_pi")
            torch.save(self.ac.qfunc1.state_dict(), model_path+"_q1")
            torch.save(self.ac.qfunc2.state_dict(), model_path+"_q2")        
            torch.save(self.ac_targ.pi.state_dict(), model_path+"_targ_ppi")   
            torch.save(self.ac_targ.qfunc1.state_dict(), model_path+"_targ_q1")
            torch.save(self.ac_targ.qfunc2.state_dict(), model_path+"_targ_q2")
            torch.save(self.pi_optimizer.state_dict(), model_path+"_pi_optim")
            torch.save(self.q_optimizer.state_dict(), model_path+"_q_optim")
        elif mode == "pi":
            torch.save(self.ac.pi.state_dict(), model_path+"_pi")
        elif mode == "q":
            torch.save(self.ac.q1.state_dict(), model_path+"_q1")
            torch.save(self.ac.q2.state_dict(), model_path+"_q2")

