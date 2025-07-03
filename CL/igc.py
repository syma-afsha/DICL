from typing import List, Tuple, Dict
import numpy as np
import torch
import math


class InitialState:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = config["environment"]["name"]

        # Curriculum strategy: "fullyadaptive", "self-paced", or "predefined"
        self.strategy = config["curriculum"]["type"]

        # Range limits
        self.min_range_obj = 0.02
        self.max_range_obj = 0.1
        self.min_range_target = 0.02
        self.max_range_target = 0.3

        self.range_factor_obj = self.min_range_obj
        self.range_factor_target = self.min_range_target
        self.max_achieved_range_obj = self.min_range_obj
        self.max_achieved_range_target = self.min_range_target

        # Predefined schedule settings
        self.profile = config.get("increment", {}).get("profile", "log")
        self.total_timesteps = config.get("trainer", {}).get("total_timesteps", 1e6)
        self.saturation_t = config.get("increment", {}).get("saturation_t", 0.6)
        self.max_range_predefined = 0.15

        # Curriculum progression indicator
        self.c = 0.0

        # Set ranges based on environment
        self.set_environment_ranges()

        self.obj_range_center = (self.obj_low_range + self.obj_high_range) / 2.0
        self.goal_range_center = (self.target_low_range + self.target_high_range) / 2.0

        self.achieved_goal = None
        self.target_goal = None

    def set_environment_ranges(self):
        if self.env == "FetchPush-v4":
            low, high = [-0.15, -0.15, 0.0], [0.15, 0.15, 0.0]
        elif self.env == "FetchSlide-v4":
            low, high = [-0.1, -0.1, 0.0], [0.1, 0.1, 0.0]
        elif self.env == "FetchPickAndPlace-v4":
            low, high = [-0.15, -0.15, 0.0], [0.15, 0.15, 0.45]
        elif self.env == "FetchReach-v4":
            low, high = [-0.15, -0.15, 0.0], [0.15, 0.15, 0.3]
        else:
            low, high = [-0.15, -0.15, 0.0], [0.15, 0.15, 0.0]

        self.obj_low_range = np.array(low)
        self.obj_high_range = np.array(high)
        self.target_low_range = np.array(low)
        self.target_high_range = np.array(high)

    def update_range_factor(self, success_rate: float = None, t: int = None) -> Tuple[float, float, float]:
        if self.strategy == "fullyadaptive":
            return self._adaptive_update(success_rate)
        elif self.strategy == "self-paced":
            return self._self_paced_update(success_rate)
        elif self.strategy == "predefined":
            return self._predefined_update(t)
        else:
            raise ValueError(f"Unknown curriculum strategy: {self.strategy}")

    def _adaptive_update(self, success_rate: float) -> Tuple[float, float, float]:
        min_s, max_s = 0.2, 0.6

        if success_rate <= min_s:
            self.range_factor_obj = self.min_range_obj
            self.range_factor_target = self.min_range_target
            self.c = 0.0
        elif success_rate >= max_s:
            self.range_factor_obj = self.max_range_obj
            self.range_factor_target = self.max_range_target
            self.c = 1.0
        else:
            normalized = (success_rate - min_s) / (max_s - min_s)
            log_factor = np.log1p(normalized) / np.log1p(1.0)
            self.range_factor_obj = self.min_range_obj + (self.max_range_obj - self.min_range_obj) * log_factor
            self.range_factor_target = self.min_range_target + (self.max_range_target - self.min_range_target) * log_factor
            self.c = (self.range_factor_obj - self.min_range_obj) / (self.max_range_obj - self.min_range_obj)

        return self.range_factor_obj, self.range_factor_target, self.c

    def _self_paced_update(self, success_rate: float) -> Tuple[float, float, float]:
        min_s, max_s = 0.2, 0.6

        if success_rate >= max_s:
            self.range_factor_obj = min(self.range_factor_obj + 0.01, self.max_range_obj)
            self.range_factor_target = min(self.range_factor_target + 0.01, self.max_range_target)
            self.max_achieved_range_obj = self.range_factor_obj
            self.max_achieved_range_target = self.range_factor_target
        else:
            normalized = (success_rate - min_s) / (max_s - min_s)
            log_factor = np.log1p(max(0, normalized)) / np.log1p(1.0)
            self.range_factor_obj = max(self.max_achieved_range_obj,
                                        self.min_range_obj + (self.max_range_obj - self.min_range_obj) * log_factor)
            self.range_factor_target = max(self.max_achieved_range_target,
                                           self.min_range_target + (self.max_range_target - self.min_range_target) * log_factor)
            self.max_achieved_range_obj = self.range_factor_obj
            self.max_achieved_range_target = self.range_factor_target

        self.c = (self.range_factor_obj - self.min_range_obj) / (self.max_range_obj - self.min_range_obj)
        return self.range_factor_obj, self.range_factor_target, self.c

    def _predefined_update(self, t: int) -> Tuple[float, float, float]:
        normalized_t = min(1.0, t / (self.total_timesteps * self.saturation_t))

        if self.profile == "linear":
            progress = normalized_t
        elif self.profile == "sqrt":
            progress = math.sqrt(normalized_t)
        elif self.profile == "quad":
            progress = normalized_t ** 2
        elif self.profile == "log":
            progress = math.log1p(normalized_t) / math.log1p(1.0)
        else:
            raise ValueError(f"Unknown pacing profile: {self.profile}")

        self.range_factor_obj = self.range_factor_target = progress * self.max_range_predefined
        self.c = progress
        return self.range_factor_obj, self.range_factor_target, self.c

    def init_state_goal_range(
    self,
    success_rate: float = None,
    t: int = None,
    num_objects: int = 1, # number of objects 
    multi_goal: bool = False #add multiple goal if you have
) -> Tuple[List[np.ndarray], List[np.ndarray], float, float, float]:
        """
        Returns:
            - List of non-overlapping achieved goals (object positions)
            - List of corresponding target goals (one per object if multi_goal=True)
            - range factor for object
            - range factor for target
            - normalized curriculum progress c
        """
        r_obj, r_target, c = self.update_range_factor(success_rate, t)

        obj_low = self.obj_range_center - r_obj
        obj_high = self.obj_range_center + r_obj
        goal_low = self.goal_range_center - r_target
        goal_high = self.goal_range_center + r_target

        if self.env == "FetchPickAndPlace-v4":
            goal_low[2] = max(goal_low[2], 0.0)
            goal_high[2] = min(goal_high[2], 0.45)

        # Generate non-overlapping achieved goals
        offsets = self.generate_nonoverlapping_offsets(num_items=num_objects, range_radius=r_obj)
        achieved_goals = [self.obj_range_center + offset for offset in offsets]
        self.achieved_goal = achieved_goals

        # Generate target goals
        if multi_goal:
            target_goals = [np.random.uniform(low=goal_low, high=goal_high) for _ in range(num_objects)]
        else:
            # Same goal for all objects
            shared_goal = np.random.uniform(low=goal_low, high=goal_high)
            target_goals = [shared_goal] * num_objects

        self.target_goal = target_goals

        return achieved_goals, target_goals, r_obj, r_target, c


    def generate_nonoverlapping_offsets(self, num_items: int, range_radius: float, min_dist: float = 0.005) -> List[np.ndarray]:
        """
        Generate non-overlapping 3D offset vectors within a cube of side 2*range_radius.
        """
        offsets = []
        max_attempts = 1000
        while len(offsets) < num_items and max_attempts > 0:
            candidate = np.random.uniform(-range_radius, range_radius, size=3)
            if all(np.linalg.norm(candidate - o) >= min_dist for o in offsets):
                offsets.append(candidate)
            max_attempts -= 1
        return offsets
