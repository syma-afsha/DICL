"""Implementation Heirarchical Experience Replay (HER) buffer."""

from typing import Dict, Tuple, Union, List
import random
import time
import gymnasium as gym
import numpy as np
from buffer.replay_buffer import ReplayBuffer
from buffer.per import PrioritizedReplayBuffer


Transition = Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]


class HER:
    """Agent tries to achieve a goal state but unfortunately fails. In sparse reward, if agent does not reach the goal state, it gets -1 reward. So that means no learning.
    The agent can still learn from the failed experience by using HER. HER pretends that the failed experience is a successful experience by changing the goal state to the state where the agent failed. This way, the agent can learn from the failed experience.
    """

    def __init__(
        self,
        config: Dict,
        env: gym.Env,
        buffer: Union["ReplayBuffer", "PrioritizedReplayBuffer"],
    ) -> None:
        """
        Initialize the HER buffer.

        Args:
            config (Dict): Configuration dictionary containing hyperparameters.
            env (gym.Env): Environment to interact with.
            buffer (Union[ReplayBuffer, PrioritizedReplayBuffer]): Replay buffer to store transitions.
        """
        self.buffer = buffer
        self.config = config
        self.env = env
        self.buffer = buffer

        # HER parameters
        # Defines how new goals are selected for replay
        self.goal_selection_strategy = config["buffer"]["her"][
            "goal_selection_strategy"
        ]

        #  Number of goals sampled per transition
        self.number_of_goal_transitions = config["buffer"]["her"][
            "number_of_goal_transitions"
        ]

        # HER strategy is enabled or not
        self.her_activity = config["buffer"]["her"]["active"]

        # Check if the state is changed
        self.state_check = config["buffer"]["her"]["state_check"]

    def get_achieved_goal(self, obs: np.ndarray) -> np.ndarray:
        """
        Get the achieved goal from the observation.

        Args:
            obs (np.ndarray): The observation from the environment.

        Returns:
            np.ndarray: The achieved goal from the observation.
        """
        if self.config["environment"]["name"] == "FetchReach-v4":
            return obs[:3]  # Gripper position
        else:

            state_dim = self.config["environment"]["observation_dim"]
            return obs[state_dim : state_dim + 3]

    def get_desired_goal(self, obs: np.ndarray) -> np.ndarray:
        """
        Get the desired goal from the observation.

        Args:
            obs (np.ndarray): The observation from the environment.

        Returns:
            np.ndarray: The desired goal from the observation.
        """

        state_dim = self.config["environment"]["observation_dim"]
        # For FetchReach-v4, the observation is [observation, desired_goal]
        if self.config["environment"]["name"] == "FetchReach-v4":
            return obs[state_dim:] 
        else:
            return obs[state_dim + 3 :] 

    def has_object_moved(
        self,
        o_start: np.ndarray,
        o_end: np.ndarray,
        dim: int = 2,
        threshold: float = 0.01,
    ) -> bool:
        """
        Checks if the object has moved significantly between two observations.

        Args:
            o_start (np.ndarray): Initial observation.
            o_end (np.ndarray): Final observation.
            dim (int): Number of dimensions to check movement (default: 2 for X, Y).
            threshold (float): Minimum movement threshold to consider it significant.

        Returns:
            bool: True if the object moved, False otherwise.

        """

        obj_start_pos = self.get_achieved_goal(o_start)
        obj_end_pos = self.get_achieved_goal(o_end)

        distance = np.linalg.norm(obj_start_pos[:dim] - obj_end_pos[:dim], axis=-1)

        return bool(np.array(distance > threshold, dtype=np.float32))

    def her_reward_and_done(self, obs: np.ndarray) -> Tuple[float, bool]:
        """
        Calculate the HER reward and done signal based on the current observation.

        Args:
            obs (np.ndarray): The current observation from the environment.

        Returns:
            Tuple[float, bool]: The HER reward and done signal.
        """
        # Get the current achieved and desired goals
        achieved_goal = self.get_achieved_goal(obs)
        desired_goal = self.get_desired_goal(obs)

        # Calculate the distance between the achieved and desired goals
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        threshold = 0.05
        # Check if the distance is less than or equal to the threshold
        if d <= threshold:
            reward = 0.0  # success
            done = True
        else:
            # Shaping: reward gradually increases as distance decreases.
            reward = -1.0
            done = False

        return reward, done

    def get_new_goals(
        self, episode: List[Transition], episode_t: int
    ) -> List[np.ndarray]:
        """
        Get new goals for relabeling experience in the HER strategy.
        """
        new_goals = []
        if self.goal_selection_strategy == "future":
            # Selects the future state as the new goal.
            for _ in range(self.number_of_goal_transitions):
                # Select a random future time step to sample the goal
                future_t = random.randint(episode_t, len(episode) - 1)
                # Get the next observation
                _, _, _, next_obs, _ = episode[future_t]
                new_goals.append(self.get_achieved_goal(next_obs))
            return new_goals

        elif self.goal_selection_strategy == "final":
            # Selects the final state as the new goal.
            for _ in range(self.number_of_goal_transitions):
                _, _, _, next_obs, _ = episode[-1]
                new_goals.append(self.get_achieved_goal(next_obs))
            return new_goals

        elif self.goal_selection_strategy == "near":
            # Selects a state near (5 steps ahead) the current state as the new goal.
            for _ in range(self.number_of_goal_transitions):
                near_t = random.randint(episode_t, min(len(episode) - 1, episode_t + 5))
                _, _, _, next_obs, _ = episode[near_t]
                new_goals.append(self.get_achieved_goal(next_obs))
            return new_goals

        elif self.goal_selection_strategy == "next":
            # Selects the next state as the new goal.
            for _ in range(self.number_of_goal_transitions):
                if episode_t + 1 < len(episode):
                    _, _, _, next_obs, _ = episode[episode_t + 1]
                    new_goals.append(self.get_achieved_goal(next_obs))
            return new_goals

        elif self.goal_selection_strategy == "final_valid":  # for slide task
            next_obs = None
            # Search for the first valid state where next_obs[5] < 0.35
            for _, _, _, o2, _ in episode:
                if o2[5] < 0.35:  # Object linear velocity in the z-direction
                    next_obs = o2
                    break
            if next_obs is None:
                next_obs = episode[-1][3]
            for _ in range(self.number_of_goal_transitions):
                new_goals.append(self.get_achieved_goal(next_obs))
            return new_goals

        elif self.goal_selection_strategy == "episode":
            for _ in range(self.number_of_goal_transitions):
                random_t = random.randint(0, len(episode) - 1)
                _, _, _, next_obs, _ = episode[random_t]
                new_goals.append(self.get_achieved_goal(next_obs))
            return new_goals

        else:
            raise ValueError(
                f"Invalid HER goal selection strategy: {self.goal_selection_strategy}"
            )

    def change_goal(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """
        Change the goal in the observation.

        Args:
            obs (np.ndarray): The observation from the environment.
            goal (np.ndarray): The new goal to set in the observation.

        Returns:
            np.ndarray: The observation with the new goal.
            
        """
        state_dim = self.config["environment"]["observation_dim"]
        # For FetchReach-v4, the observation is [observation, desired_goal]
        if self.config["environment"]["name"] == "FetchReach-v4":
            # Replace the desired goal portion, which starts at index state_dim
            obs[state_dim:] = goal
        else:
            # For other Fetch environments, the observation is [observation, achieved_goal, desired_goal]
            # and the desired goal starts after the achieved_goal (assumed to be 3-dimensional)
            obs[state_dim + 3 :] = goal
        return obs

    def relabel_experience(self, episode: List[Transition]) -> bool:
        """
        Relabel the experience in the episode using the HER strategy.
        This version iterates over each timestep and computes new goals (e.g., for a 'future' strategy)
        to relabel each transition individually.

        Args:
            episode (List[Transition]): The list of transitions in the episode.

        Returns:
            bool: True if relabeling was applied (i.e. the object has moved or state_check is disabled),
                False otherwise.
        """
        # Check if the object has moved significantly or if state checking is disabled.
        object_moved = self.has_object_moved(episode[0][0], episode[-1][0])
        if object_moved or not self.state_check:
            # print(
            #     "Relabeling experience with HER for {} transitions".format(len(episode))
            # )
            # if self.goal_selection_strategy in ['final','final_valid','future_once']:
            #     new_goals = self.get_new_goals(episode,0)
            #     for (o, a, r, o2, d) in episode:                  
            #         for new_goal in new_goals:
            #             o_new = self.change_goal(o, new_goal)
            #             o2_new = self.change_goal(o2, new_goal)
            #             r_new, d_new = self.her_reward_and_done(o2_new) 
            #             self.buffer.store(o_new, a, r_new, o2_new, d_new)

            if self.goal_selection_strategy in ["final_valid", "final"]:
                new_goals = self.get_new_goals(episode, 0)
                for (obs, act, _, next_obs, _), new_goal in zip(episode, new_goals):
                    new_obs = self.change_goal(obs.copy(), new_goal)
                    new_next_obs = self.change_goal(next_obs.copy(), new_goal)
                    new_reward, new_done = self.her_reward_and_done(new_next_obs)
                    self.buffer.store(new_obs, act, new_reward, new_next_obs, new_done)
            else:

                # # Iterate over each timestep in the episode.
                for t, _ in enumerate(episode):
                    # Compute new goals for the current transition t.
                    new_goals = self.get_new_goals(episode, t)
                    for new_goal in new_goals:
                        # Unpack the transition at time t.
                        obs, act, _, next_obs, _ = episode[t]
                        # Create copies to avoid modifying the original observations.
                        new_obs = self.change_goal(obs.copy(), new_goal)
                        new_next_obs = self.change_goal(next_obs.copy(), new_goal)
                        # Get the new reward and done signal based on the updated next_obs.
                        new_reward, new_done = self.her_reward_and_done(new_next_obs)
                        # Store the relabeled transition in the replay buffer.
                        self.buffer.store(
                            new_obs, act, new_reward, new_next_obs, new_done
                        )
                # ep_t = 0
                # for (o, a, r, o2, d) in episode:
                #     new_goals = self.get_new_goals(episode,ep_t)
                #     for new_goal in new_goals:
                #         o_new = self.change_goal(o, new_goal)
                #         o2_new = self.change_goal(o2, new_goal)
                #         r_new, d_new = self.her_reward_and_done(o2_new) 
                #         self.buffer.store(o_new, a, r_new, o2_new, d_new)
                #     ep_t += 1
        return object_moved
    



