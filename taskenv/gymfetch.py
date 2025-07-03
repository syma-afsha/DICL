from typing import Dict, Tuple
import numpy as np
import torch
import gymnasium as gym
from CL.new_alg import InitialState


class GymFetch:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initial_state = InitialState(config)
        self.env = gym.make(config["environment"]["name"], render_mode="rgb_array")
    

    def test_env(self):
        test_env=gym.make(self.config["environment"]["name"], render_mode="human")
        test_env_reset=self.reset_env()
        return test_env, test_env_reset
 
     
    def reset_env(self) -> Dict:
        """
        Resets the environment once per episode, randomizing the object's position 
        only here to avoid multiple objects accumulating in MuJoCo.
        """
        # Reset the environment and get the initial observation dictionary.
     
        obs_dict, _ = self.env.reset()

        self.current_success_rate = self.config["curriculum"]["success_rate"]
        self.current_time_step=self.config["curriculum"]["time_step"]
        # Randomize goals ONLY at the start of each episode.
        # new_achieved_goal, new_target_goal = self.initial_state.init_state_goal_range(
        #     t=self.current_time_step
        # )
        # new_achieved_goal, new_target_goal,range_factor_obj,range_factor_target,c = self.initial_state.init_state_goal_range(
        #     success_rate=self.current_success_rate,  t=self.current_time_step
        # )
        achieved_goals, target_goals, range_factor_obj, range_factor_target, c = self.initial_state.init_state_goal_range(
        success_rate=self.current_success_rate,
        t=self.current_time_step,
    )

        # Extract the first (and only) goal
        new_achieved_goal = achieved_goals[0]
        new_target_goal = target_goals[0] 

        # print("new_target_goal:", new_target_goal)
        # print("achieved_goals:", achieved_goals)
        # print("new_achieved_goal:", new_achieved_goal)

        self.config["curriculum"]["range_factor"] = range_factor_obj
        self.config["curriculum"]["c"] = c

        obs_dict["achieved_goal"] = new_achieved_goal
        obs_dict["desired_goal"] = new_target_goal
        obs_dict["observation"][3:6] = new_achieved_goal
        
        return obs_dict
    
    def step(self, action: np.ndarray) -> Tuple:
        """
        Steps the environment without re-randomizing the object's position.
        This prevents accumulating multiple objects in the MuJoCo scene.
        """
   
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
   
        # We do NOT re-randomize the object's position here.
        # 'obs_dict' will naturally update based on the environment's internal dynamics.
        
        return obs_dict, reward, terminated, truncated, info
    

    
