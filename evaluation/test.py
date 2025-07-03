from typing import Any, Tuple, Dict
import time
import numpy as np
import gymnasium as gym
import torch
from tqdm import tqdm
from CL.igc import InitialState
from reward.penalty_reward import PenaltyReward
penalty=PenaltyReward()

# Detect device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")




def convert_obs(obs_dict: Dict, env_name: str) -> np.ndarray:
    """
    Converts the observation dictionary into a flat NumPy array.
    Assumes that values are either NumPy arrays or torch Tensors.
    """
    def to_np(x):
        return x if isinstance(x, np.ndarray) else x.cpu().numpy()
    if env_name == "FetchReach-v4":
        return np.concatenate([to_np(obs_dict["observation"]), to_np(obs_dict["desired_goal"])])
    else:
        return np.concatenate([to_np(obs_dict["observation"]), 
                               to_np(obs_dict["achieved_goal"]),
                               to_np(obs_dict["desired_goal"])])

def test_agent(
    agent: Any,
    Config: Dict,
    test_env: gym.Env,
    slowdown: float = 0.0,  # Set slowdown to 0 for faster evaluation
) -> Tuple[float, float, float]:
    """
    Evaluate the agent on the test environment using a deterministic policy.
    """
    total_ep_reward = 0.0
    total_successes = 0
    total_ep_len = 0.0

    num_episodes = Config["evaluation"]["num_episodes"]
    success_rate_val = Config["curriculum"]["evaluation"]["success_rate"]
    env_name = test_env.spec.id
    

    print(f"✅ Evaluating {num_episodes} episodes on {env_name} | Initial success rate: {success_rate_val}")

    for _ in tqdm(range(num_episodes), desc="Evaluating", unit="episode"):
        obs_dict,_ = test_env.reset()
        o = convert_obs(obs_dict, env_name)

        ep_reward = 0.0
        done = False
        ep_len = 0
        episode_success = 0
        max_episode_steps = Config["evaluation"]["max_episode_steps"]
 
        while not done:
            

            # Create a tensor from the observation (using from_numpy avoids extra copy)
            o_tensor = torch.from_numpy(o).to(device)
            with torch.no_grad():
                action = agent.get_action(o_tensor, deterministic=True)

            # Convert action to NumPy array before stepping
            action = action.cpu().numpy() if isinstance(action, torch.Tensor) else action

            obs_dict, reward, terminated, truncated, info = test_env.step(action)
           

            penalty_for_object_dropped_from_table,terminated_epi = penalty.penalty(obs_dict)
            if penalty_for_object_dropped_from_table==-100:
                reward=penalty_for_object_dropped_from_table
            else:
                reward=reward
            
            reward = float(reward)   # type: ignore

            done = terminated or truncated or terminated_epi
            ep_reward += float(reward)
            ep_len += 1

            # Update observation using the helper
            o = convert_obs(obs_dict, env_name)

            if info.get("is_success", 0.0):
                episode_success = 1.0

            if slowdown > 0.0:
                time.sleep(slowdown)

        total_successes += episode_success
        total_ep_reward += ep_reward
        total_ep_len += ep_len

    avg_reward = total_ep_reward / num_episodes
    success_rate = total_successes / num_episodes
    mean_ep_len = total_ep_len / num_episodes

    print(f"\n **Evaluation Results over {num_episodes} episodes:**")
    print(f"✔ Cumulative Reward = {avg_reward:.2f}")
    print(f"✔ Success Rate = {success_rate * 100:.2f}%")
    print(f"✔ Mean Episode Length = {mean_ep_len:.2f}")

    return avg_reward, success_rate, mean_ep_len
