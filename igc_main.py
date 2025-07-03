"""Implementation of Soft Actor-Critic (SAC) algorithm with PER support.
Optimized by extracting helper functions, reducing redundant operations, and adding timing.
"""

from typing import Dict, Any, Tuple, List
import os
import torch
import yaml
import time
import numpy as np
import gymnasium as gym
import argparse
from tqdm import tqdm
from collections import deque

import math
import gymnasium_robotics

# Set up environment variables and backend optimizations
os.environ["MUJOCO_GL"] = "egl"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Imports for replay buffers, evaluation, logging, agent, HER, and environment.
from buffer.replay_buffer import ReplayBuffer
from buffer.per import PrioritizedReplayBuffer
from evaluation.test import test_agent
from logger.logger import Logger
from agents.sac import Agent
from buffer.her import HER
from taskenv.gymfetch import GymFetch
from reward.penalty_reward import PenaltyReward
from dual_buffer.predefined_dualBuffer import PredefinedDualBuffer
from dual_buffer.dual_buffer import DualBuffer
from taskenv.gymfetch import GymFetch
def init_cuda(gpu: int, cpumin: int, cpumax: int) -> None:

    # BEFORE IMPORTING PYTORCH
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.system(
        "taskset -p -c " + str(cpumin) + "-" + str(cpumax) + " %d" % os.getpid()
    )  # Limit CPU usage to cpumin-cpumax


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============================================================================
# Helper functions
# =============================================================================


def process_obs(
    obs_dict: Any, is_fetch_reach: bool, config: Dict[str, Any], act_dim: int
) -> np.ndarray:
    """
    Process an observation dictionary into a flat NumPy array.
    Converts torch Tensors to numpy arrays (only once per key) and stores
    state_dim, full_obs_dim, and act_dim in the config.
    """

    def to_np(x):
        return x if isinstance(x, np.ndarray) else x.cpu().numpy()

    if isinstance(obs_dict, dict):
        obs_np = to_np(obs_dict["observation"])
        state_dim = obs_np.shape[0]
        if is_fetch_reach:
            desired = to_np(obs_dict["desired_goal"])
            full_obs_dim = state_dim + desired.shape[0]
            o = np.concatenate([obs_np, desired])
        else:
            achieved = to_np(obs_dict["achieved_goal"])
            desired = to_np(obs_dict["desired_goal"])
            full_obs_dim = state_dim + achieved.shape[0] + desired.shape[0]
            o = np.concatenate([obs_np, achieved, desired])
        config["environment"]["observation_dim"] = state_dim
        config["environment"]["full_obs_dim"] = full_obs_dim
    elif isinstance(obs_dict, np.ndarray):
        state_dim = obs_dict.shape[0]
        full_obs_dim = state_dim
        o = obs_dict
        config["environment"]["observation_dim"] = state_dim
        config["environment"]["full_obs_dim"] = full_obs_dim
    else:
        raise ValueError("Unexpected observation format.")

    config["environment"]["action_dim"] = act_dim
    return o




def perform_updates(
    agent,
    buffer,
    batch_size: int,
    update_every: int,
    use_per: bool,
    device: torch.device,
    loss_q_dq: deque,
    loss_pi_dq: deque,
    success_rate:float
):

    """Perform several update steps for the agent."""
    for _ in range(update_every):
        

        if dual_buffer_active and dual_buffer.is_sampling_possible():
   
            primary_batch_size=main_dual_buffer.get_primary_batch_size()
            print(primary_batch_size)

            primary_batch=buffer.sample_batch(int(primary_batch_size))
            pos_batch=main_dual_buffer.sample_batch_positive()
     
            neg_batch=main_dual_buffer.sample_batch_negative()
            batch=dict(
                obs=torch.cat((primary_batch["obs"],pos_batch["obs"], neg_batch["obs"]),0),
                next_obs=torch.cat((primary_batch["next_obs"],pos_batch["next_obs"], neg_batch["next_obs"]),0),
                act=torch.cat((primary_batch["act"],pos_batch["act"],neg_batch["act"]),0),
                rew=torch.cat((primary_batch['rew'], pos_batch['rew'], neg_batch['rew']), 0),
                done=torch.cat((primary_batch['done'], pos_batch['done'], neg_batch['done']), 0),
                indices=torch.cat((primary_batch['indices'], pos_batch['indices'],neg_batch['indices']), 0),
                weights=torch.cat((primary_batch['weights'], pos_batch['weights'], neg_batch['weights']), 0))

        if use_per:

            batch = buffer.sample_batch(batch_size)
            batch["weights"] = torch.as_tensor(
                batch["weights"], dtype=torch.float32, device=device
            )
        else:
            batch = buffer.sample_batch(batch_size)
        loss_q, loss_pi, batch_priorities = agent.update(batch)
        if use_per:
            indices = batch["indices"].detach().cpu().numpy().astype(np.int32)
            buffer.update_priorities(indices, batch_priorities)
        if dual_buffer_active:
            main_dual_buffer.update_priorities(batch_priorities, success_rate)
        loss_q_dq.append(loss_q)
        loss_pi_dq.append(loss_pi)


# =============================================================================
# Main Code
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml", help="Path to config file")
parser.add_argument("--hwid", type=int, default=0, help="GPU id to use")
args = parser.parse_args()

config = load_config(args.config)

# Initialize CUDA and CPU threads.
init_cuda(
    config["hardware"]["gpu"][args.hwid],
    config["hardware"]["cpu_min"][args.hwid],
    config["hardware"]["cpu_max"][args.hwid],
)


exp_name = config["general"]["exp_name"]
use_per = config["buffer"]["use_per"]
her_active = config["buffer"]["her"]["active"]
max_ep_len = int(config["exploration"]["max_episode_length"])


print("Using PER:", use_per)
print("Using HER:", her_active)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache training parameters.
steps_per_epoch = config["trainer"]["eval_freq"]
total_steps = config["trainer"]["total_timesteps"]
start_steps = config["exploration"]["start_steps"]
batch_size = config["trainer"]["batch_size"]
update_after = config["trainer"]["update_after"]
update_every = config["trainer"]["update_interval"]
save_freq = config["trainer"]["save_freq"]

# PER parameters.
alpha = config["buffer"]["per"]["alpha"]
beta_start = config["buffer"]["per"]["beta_start"]

# Model saving parameters.
model_save_freq = config["logger"]["model"]["save"]["freq"]
model_save_mode = config["logger"]["model"]["save"]["mode"]
model_save_measure = config["logger"]["model"]["save"]["measure"]
model_save_best_t = config["logger"]["model"]["save"]["best_start_t"]

# Bonus reward parameter.
bonus_factor = config.get("bonus_factor", 0.1)

# Rollout and logging deques.
rollout_stats_window_size = int(config["logger"]["rollout"]["stats_window_size"])
ep_rew_dq = deque(maxlen=rollout_stats_window_size)
ep_len_dq = deque(maxlen=rollout_stats_window_size)
ep_success_dq = deque(maxlen=rollout_stats_window_size)
loss_q_dq = deque(maxlen=rollout_stats_window_size)
loss_pi_dq = deque(maxlen=rollout_stats_window_size)
virtual_experience_dq = deque(maxlen=rollout_stats_window_size)

# Initialize environments with TimeLimit wrappers.

env = gym.make(config["environment"]["name"], render_mode="rgb_array")
# env = TimeLimit(env, max_episode_steps=max_ep_len)
test_env = gym.make(config["environment"]["name"], render_mode="rgb_array")
# test_env = TimeLimit(test_env, max_episode_steps=max_ep_len)

# Cache check for observation type.
is_fetch_reach: bool = config["environment"]["name"] == "FetchReach-v4"

# Initialize task environment and get an initial observation.
taskenv = GymFetch(config)

obs_dict = taskenv.reset_env()

# Extract act_dim from environment.
if env.action_space is None:
    raise ValueError("Environment has no action space.")
if hasattr(env.action_space, "shape") and env.action_space.shape is not None:
    act_dim = env.action_space.shape[0]
elif isinstance(env.action_space, gym.spaces.Discrete):
    act_dim = env.action_space.n
else:
    raise ValueError("Unsupported action space type.")

# Process initial observation and store act_dim in config.
o = process_obs(obs_dict, is_fetch_reach, config, int(act_dim))
print(f"DEBUG: full_obs_dim={config['environment']['full_obs_dim']}, act_dim={act_dim}")

# Initialize agent and replay buffer.
agent = Agent(config, str(device))
if use_per:
    buffer = PrioritizedReplayBuffer(
        config["environment"]["full_obs_dim"],
        int(act_dim),
        config["buffer"]["replay_buffer"]["size"],
        alpha=alpha,
        beta_start=beta_start,
    )
    print("Using Prioritized Experience Replay (PER)")
else:
    buffer = ReplayBuffer(
        config["environment"]["full_obs_dim"],
        int(act_dim),
        config["buffer"]["replay_buffer"]["size"],
    )
    print("Using Standard Replay Buffer")

# Initialize HER if active.
Her = HER(config, env, buffer) if her_active else None
dual_buffer_active=config["buffer"]["dual_buffer"]["active"]
main_dual_buffer = PredefinedDualBuffer(config)
dual_buffer = main_dual_buffer  

# =============================================================================
# Main Training Loop with Timing
# =============================================================================

completion_steps = 0
ep_ret, ep_len = 0.0, 0
episode = []  # Store transitions for one episode.
logger = Logger(config)

# Initialize timing accumulators.
t_collect_total: float = float(0.0)
t_process_total: float = float(0.0)
t_train_total: float = float(0.0)
t_eval_total: float = float(0.0)

training_start = time.time()
penalty=PenaltyReward()

for t in tqdm(range(total_steps), desc="Training", leave=True):
    config["curriculum"]["time_step"]=t
    
    
    t_collect_start = time.time()
    
    # Action selection.
    if t > start_steps:
        action = agent.get_action(o, deterministic=False)
    else:
        action = env.action_space.sample()
    t_collect_total += time.time() - t_collect_start
    

    # Step environment.
    t_step_start = time.time()
    next_obs_dict, r, terminated, truncated, info = taskenv.step(np.array(action))
 

    r=float(r)
    

    o2 = process_obs(next_obs_dict, is_fetch_reach, config, int(act_dim))
    t_step = time.time() - t_step_start


    d = terminated or truncated
    ep_ret += float(r)
    ep_len += 1
    d = False if ep_len == max_ep_len else d

    episode.append((o, action, r, o2, d))
    o = o2
   
    # End-of-episode processing.
    if d or (ep_len == max_ep_len):
        t_process_start = time.time()
        # episode, bonus = apply_bonus(episode, bonus_factor, completion_steps, max_ep_len)
        ep_success = info.get("is_success", 0.0)
        ep_success_dq.append(ep_success)
    
        
        for o_i, a_i, r_i, o2_i, d_i in episode:
            buffer.store(o_i, a_i, r_i, o2_i, d_i)
        if her_active:
            print("Using HER")
            if Her is not None:
                virtual_experience = Her.relabel_experience(episode)
                virtual_experience_dq.append(virtual_experience)
                    
        obs_dict = taskenv.reset_env()
        o = process_obs(obs_dict, is_fetch_reach, config, int(act_dim))
      

        ep_rew_dq.append(ep_ret)
        ep_len_dq.append(ep_len)
        ep_ret, ep_len = 0.0, 0
        # completion_steps = 0
        episode = []
        t_process_total += time.time() - t_process_start
        
    

    # Perform updates after warmup.
    if t >= update_after and t % update_every == 0:
        mean_success = float(np.mean(list(ep_success_dq))) if ep_success_dq else 0.0
        t_train_start = time.time()
        perform_updates(
            agent,
            buffer,
            batch_size,
            update_every,
            use_per,
            device,
            loss_q_dq,
            loss_pi_dq,
            mean_success
        )
        t_train_total += time.time() - t_train_start
        
   
    # Periodic evaluation and model saving.
    if (t + 1) % steps_per_epoch == 0:
        t_eval_start = time.time()
        epoch = (t + 1) // steps_per_epoch

        print(f"\n=== Epoch {epoch} ===")
        with torch.no_grad():
            eval_cum_reward, eval_success, eval_ep_len = test_agent(agent,config,test_env)
        config["curriculum"]["evaluation"]["success_rate"] = eval_success

        logger.tb_writer_and_scalar("eval/eval_cumulative_reward", eval_cum_reward, t)
        logger.tb_writer_and_scalar("eval/eval_success_rate", eval_success, t)
        logger.tb_writer_and_scalar("eval/eval_mean_ep_len", eval_ep_len, t)
        if dual_buffer_active:
            lambda_pos=current_lambda_pos # type: ignore
            lambda_neg=current_lambda_neg # type: ignore
            pos_sampling_ratio=config["buffer"]["dual_buffer"]["xi"]["pos_ratio"]
            neg_sampling_ratio=config["buffer"]["dual_buffer"]["xi"]["neg_ratio"]
            primary_sampling_ratio=1-(pos_sampling_ratio+neg_sampling_ratio)
            neg_ratio=config["buffer"]["dual_buffer"]["xi"]["neg_ratio"]
            logger.tb_writer_and_scalar("cl/lambda_pos", lambda_pos, t)
            logger.tb_writer_and_scalar("cl/lambda_neg", lambda_neg, t)
            logger.tb_writer_and_scalar("cl/pos_ratio", pos_sampling_ratio, t)
            logger.tb_writer_and_scalar("cl/neg_ratio", neg_ratio, t)
            logger.tb_writer_and_scalar("cl/primary_ratio",primary_sampling_ratio,t)
            dual_buffer.save_metrics_csv("buffer_metrics.csv")
            dual_buffer.save_event_log_csv("buffer_event_log.csv")
        best_model_changed = False
        model_best_eval_measure = -float("inf")
        eval_measure = eval_cum_reward if model_save_measure == "reward" else eval_success
        if t > model_save_best_t and eval_measure > model_best_eval_measure:
            model_best_eval_measure = eval_measure
            best_model_changed = True

        if best_model_changed:
            model_path = logger.backup_model_save_path("best_model")
            agent.save_model(model_path, model_save_mode)
        if epoch % model_save_freq == 0:
            model_path = logger.backup_model_save_path(f"epoch_{epoch}")
            agent.save_model(model_path, model_save_mode)

        mean_success = float(np.mean(list(ep_success_dq))) if ep_success_dq else 0.0
        config["curriculum"]["success_rate"] = mean_success
        print(f"Mean success rate: {config['curriculum']['success_rate']:.2f}")
        range_factor = config["curriculum"]["range_factor"]
        c = config["curriculum"]["c"]
        print(f"Range factor: {range_factor:.2f}, c: {c:.2f}")
        
        mean_ep_len_val = float(np.mean(list(ep_len_dq))) if ep_len_dq else 0.0
        mean_ep_reward = float(np.mean(list(ep_rew_dq))) if ep_rew_dq else 0.0
        loss_q_mean = float(np.mean(list(loss_q_dq))) if loss_q_dq else 0.0
        loss_pi_mean = float(np.mean(list(loss_pi_dq))) if loss_pi_dq else 0.0
        her_virt = float(np.mean(list(virtual_experience_dq))) if virtual_experience_dq else 0.0
        per_beta = float(np.mean(buffer.get_beta())) if use_per else 0.0

        logger.tb_writer_and_scalar("rollout/mean_episode_reward", mean_ep_reward, t)
        logger.tb_writer_and_scalar("rollout/mean_episode_len", mean_ep_len_val, t)
        logger.tb_writer_and_scalar("rollout/mean_success_rate", mean_success, t)
        logger.tb_writer_and_scalar("train_loss/critic_loss", loss_q_mean, t)
        logger.tb_writer_and_scalar("train_loss/actor_loss", loss_pi_mean, t)
        logger.tb_writer_and_scalar("cl/range", range_factor, t)
        logger.tb_writer_and_scalar("cl/c", c, t)
        if her_active:
            logger.tb_writer_and_scalar("her/virtual_experience_her", her_virt, t)
        if use_per:
            logger.tb_writer_and_scalar("per/beta_per", per_beta, t)
        logger.tb_writer_and_scalar("time/collect", t_collect_total, t)
        logger.tb_writer_and_scalar("time/process_ep", t_process_total, t)
        logger.tb_writer_and_scalar("time/train", t_train_total, t)
        logger.tb_writer_and_scalar("time/eval", time.time() - t_eval_start, t)

        logger.tb_to_csv(logger.tb_logdir)
        t_eval_total += time.time() - t_eval_start

logger.tb_close()
total_time = time.time() - training_start
hour = total_time // 3600
minute = (total_time % 3600) // 60
second = total_time % 60
print(f"Training complete in {int(hour)}h {int(minute)}min {int(second)}s")
