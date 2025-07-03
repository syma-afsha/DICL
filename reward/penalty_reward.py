from typing import Tuple
import numpy as np
import torch

import mujoco


class PenaltyReward:
    def __init__(self):
        self.penalty_for_drop_from_table = 0.0
        self.penalty_for_drop_from_gripper = 0.0
        self.best_ep_len=50

    def object_dropped_from_the_table(self,obs)->bool:
        """Check if the object was dropped from the table."""
        

        object_pos = obs["observation"][3:6]  # Extract object position (x, y, z)
        object_height = object_pos[2] # Z-coordinate (height from the table)
        
        # print(env.unwrapped.model.body)
        table_center_z=0.4
        object_floor=object_height
        threshold=table_center_z-object_floor
        diff=0.4-0.2

        if object_height<0.39:

            print(f"Object height: {object_height}")
            print(f"DEBUG: Object dropped from the table: {True}")
            return True
        else:
         
            
            return False
     

    def object_dropped_from_gripper(self, obs: np.ndarray) -> bool:
        """
        Check if the object was dropped from the gripper.
        Logic:
          - If the gripper is closed (gripper_state < 0.02), set a flag that the object was grasped.
          - Later, if the gripper opens (gripper_state >= 0.02) and the distance between gripper and object 
            exceeds a threshold, then the object is considered dropped.
        """
        # Extract positions and gripper state from observation.
        object_pos = obs["observation"][3:6]  # Expected [x, y, z] for object.
        gripper_pos = obs["observation"][0:3]  # Expected [x, y, z] for gripper.
        object_z=object_pos[2]
        self.has_picked=False
       
        
        # Compute distance between gripper and object.
        distance = np.linalg.norm(gripper_pos - object_pos)
        # For debugging:
        
        was_grasped=object_z>0.42 and distance<0.05
        if was_grasped:
            self.has_picked=True

        if self.has_picked and object_z<0.39 and distance>0.1:
            print("Object dropped after pickup")
            self.has_picked=False

    # def calculate_penalty(self,t):
    #            # Define the maximum penalty value
        

    #     # Define the threshold after which penalties start increasing (e.g., after 10,000 steps)
    #     penalty_start_step = 1000

   
            
    #     if t < penalty_start_step:
    #         return 0.0  # No penalty in the early stages of training
    #     else:
    #         # Gradually increase the penalty as training progresses
    #         penalty = -0.1
    #         return penalty  # Cap the penalty at max_penalty

    def penalty(self,obs)->Tuple[float,bool]:
        """
        Penalize the agent if the object is dropped from the table."
        """
       
    

        if self.object_dropped_from_the_table(obs):
            print(obs)
            self.penalty_for_drop_from_table = -100.0

            print(f"DEBUG: Penalty for dropping object: {self.penalty_for_drop_from_table}")
            # self.penalty_for_drop_from_table=-0.1
            terminated_episode=True
        # elif self.object_dropped_from_gripper(obs):
        #     print(f"Penalty for Object Dropped from Gripper:{self.object_dropped_from_gripper}")
        #     self.penalty_for_drop_from_gripper=-70
        #     terminated_episode=True
     
        else:
            self.penalty_for_drop_from_table=0.0
            terminated_episode=False
          
            
        return self.penalty_for_drop_from_table,  terminated_episode

    # def calculate_bonus(self, ep_len,success)->float:
    #     self.ep_len = ep_len  
       

    #     if success==1.0 and self.ep_len <= self.best_ep_len:
    #         print(f"DEBUG: Episode length: {self.ep_len}")
    #         print(f"DEBUG: Best episode length: {self.best_ep_len}")
    #         "if episode is successful and the episode length is less than the best episode length then best episode length is updated nd stored in best_ep_len"
    #         print(f"Succesful: {success}")
    #         self.bonus=100.0
    #         self.best_ep_len=self.ep_len
    #         print(f"DEBUG: After Best episode length: {self.best_ep_len}")
    #         print(f"DEBUG: After Bonust: {self.bonus}")
    #     else:
    #         self.bonus=0.0

    #     return self.bonus