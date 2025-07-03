from typing import Dict,Tuple,Union
import numpy as np
import torch
import hashlib
from buffer.replay_buffer import ReplayBuffer



class DualBuffer:
    """ Generating Dual Bufferes one for storing positive samples and one for storing negative samples"""

    def __init__(self, config: Dict) -> None:
        self.config = config
        
      
          # Create separate buffers for positive and negative samples.
        obs_dim = config["environment"]["full_obs_dim"]
        act_dim = config["environment"]["action_dim"]
        self.buffer_size = config["buffer"]["replay_buffer"]["size"]
        self.replay_buffer= ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=self.buffer_size)
        self.pos_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=self.buffer_size)
        self.neg_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=self.buffer_size)
        
        self.batch_size = config["trainer"]["batch_size"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
        self.xi_mode = config['buffer']['dual_buffer']['xi']['mode']
        self.pos_buffer_samples = None  # Initialize pos_buffer_samples
        
        self.neg_buffer_samples = None  # Initialize neg_buffer_samples
   
        assert self.xi_mode in ['fix', 'fullyAdaptive', 'hybridAdaptive']  

     
        self.pos_ratio = config['buffer']['dual_buffer']['xi']['pos_ratio']
        self.neg_ratio = config['buffer']['dual_buffer']['xi']['neg_ratio']
        

        self.pos_batch_size = int(self.batch_size * self.pos_ratio)
        self.neg_batch_size = int(self.batch_size * self.neg_ratio)
      
        # Success rate tracking (moving average)
        # self.success_rate_wma = 0.0
        # self.beta = 0.8  # Weight for moving average
        # self.sigmoid_scale = 10  # Controls steepness of the sigmoid transition
        # self.sigmoid_shift = 0.5  # Shift to center sigmoid transition
        # self.log_scale = 0.5  # Log scale for negative sampling
        self.base_success_rate = 0.2

        
    def get_primary_batch_size(self) -> int:
        """Return the primary buffer batch size."""
        return self.batch_size- self.pos_batch_size - self.neg_batch_size

    def get_secondary_batch_sizes(self) -> Tuple[int, int]:
        """Return (pos_batch_size, neg_batch_size)."""
        return self.pos_batch_size, self.neg_batch_size
    
    def sample_batch_positive(self) -> Dict:
      
        if self.pos_buffer.size > 0:
            self.pos_buffer_samples = self.pos_buffer.sample_batch(self.pos_batch_size)
            return {
                "obs":      self.pos_buffer_samples["obs"],
                "next_obs": self.pos_buffer_samples["next_obs"],
                "act":      self.pos_buffer_samples["act"],
                "rew":      self.pos_buffer_samples["rew"],
                "done":     self.pos_buffer_samples["done"],
                "indices":  self.pos_buffer_samples.get("indices", torch.empty((self.pos_batch_size,), device=self.device)),
                "weights":  self.pos_buffer_samples.get("weights", torch.ones((self.pos_batch_size,), device=self.device)),
            }

    def sample_batch_negative(self) ->Dict:
    
        if self.neg_buffer.size > 0:
            self.neg_buffer_samples = self.neg_buffer.sample_batch(self.neg_batch_size)
            return {
                "obs":      self.neg_buffer_samples["obs"],
                "next_obs": self.neg_buffer_samples["next_obs"],
                "act":      self.neg_buffer_samples["act"],
                "rew":      self.neg_buffer_samples["rew"],
                "done":     self.neg_buffer_samples["done"],
                "indices":  self.neg_buffer_samples.get("indices", torch.empty((self.neg_batch_size,), device=self.device)),
                "weights":  self.neg_buffer_samples.get("weights", torch.ones((self.neg_batch_size,), device=self.device)),
            }

    def is_sampling_possible(self) -> bool:
         return True if self.replay_buffer.size > 0 else False
    
    def update_priorities(
        self, priorities: np.ndarray, success_rate: float) -> None:
        
       
        """ Update the priorities of the given transitions. """
        
        self.primary_batch_size = self.get_primary_batch_size()
        if self.xi_mode == 'fix':
            self.pos_batch_size=self.pos_batch_size
            self.neg_batch_size=self.neg_batch_size

            return



        ##### Hybrid Adaptive #####
        #Primary Fixed Negative High when Success is High
        # elif self.xi_mode == 'hybridAdaptive':
        #     # Update moving average of success rate
        #     self.success_rate_wma=success_rate
        #     # Linearly interpolate xi (positive sampling ratio)
        #     adaptive_xi = self.xi_max - (self.xi_max - self.xi_min) * self.success_rate_wma
        #     reserved_primary_ratio = 0.3 
        #     adaptive_budget = self.batch_size * (reserved_primary_ratio)

        #     # Convert xi into batch sizes
        #     self.pos_batch_size = int(adaptive_budget * adaptive_xi)
        #     self.neg_batch_size = int(adaptive_budget * (1 - adaptive_xi))
           
        #     self.primary_batch_size = self.batch_size - self.neg_batch_size - self.pos_batch_size


        #     # Optional safety clamps
        #     self.pos_batch_size = max(1, self.pos_batch_size)
        #     self.neg_batch_size = max(1, self.neg_batch_size)
        

        #     # Store for tracking if needed
        #     self.current_xi = adaptive_xi


        #     print(f"[DualBuffer] Linear adaptive xi: {adaptive_xi:.3f}, Success WMA: {self.success_rate_wma:.3f}")
        #     print(f"[DualBuffer] pos_batch_size: {self.pos_batch_size}, neg_batch_size: {self.neg_batch_size}. primary_batch_size: {self.primary_batch_size}")

        #### Fully Adaptive #####

        
        # elif self.xi_mode == 'adaptive':
        #     # Update moving average of success rate
        #     self.success_rate_wma = success_rate
        #     # Higher adaptive_xi ⇒ more negative samples

        #     # Lower adaptive_xi ⇒ more positive samples


        #     # Linearly interpolate xi (positive sampling ratio)
        #     adaptive_xi = self.xi_max - (self.xi_max - self.xi_min) * self.success_rate_wma

        #     # Optionally, interpolate primary_ratio as well (you can customize this function)
        #     adaptive_primary_ratio = 1.0 - self.success_rate_wma  # For example: high success = less primary

        #     # Total samples from positive and negative
        #     adaptive_budget = self.batch_size * (1.0 - adaptive_primary_ratio)

        #     # Adaptive positive and negative sizes based on xi
        #     self.neg_batch_size = int(adaptive_budget * adaptive_xi)
        #     self.pos_batch_size = int(adaptive_budget * (1.0 - adaptive_xi))

        #     # Adaptive primary size
        #     self.primary_batch_size = self.batch_size - self.neg_batch_size - self.pos_batch_size

        #     # Safety clamps to ensure all > 0
        #     self.pos_batch_size = max(1, self.pos_batch_size)
        #     self.neg_batch_size = max(1, self.neg_batch_size)
        #     self.primary_batch_size = max(1, self.primary_batch_size)

        #     # Store for tracking
        #     self.current_xi = adaptive_xi
        #     self.current_primary_ratio = adaptive_primary_ratio

        #     print(f"[DualBuffer] Adaptive xi: {adaptive_xi:.3f}, Primary Ratio: {adaptive_primary_ratio:.3f}, Success WMA: {self.success_rate_wma:.3f}")
        #     print(f"[DualBuffer] pos: {self.pos_batch_size}, neg: {self.neg_batch_size}, primary: {self.primary_batch_size}")

#         #Dual_Buffer
#         # 
#         # 
#         # +HER_Fetch_Slide_adaptive_xi_fixed_her_high_neg_low_pos_change"
#         # elif self.xi_mode == 'adaptive':
#         #     # Update moving average of success rate
#         #     self.success_rate_wma=success_rate
#         #     # Linearly interpolate xi (positive sampling ratio)
#         #     adaptive_xi = self.xi_max - (self.xi_max - self.xi_min) * self.success_rate_wma
#         #     reserved_primary_ratio = 0.3 
#         #     adaptive_budget = self.batch_size * (reserved_primary_ratio)

#         #     # Convert xi into batch sizes
#         #     self.neg_batch_size = int(adaptive_budget * adaptive_xi)
#         #     self.pos_batch_size = int(adaptive_budget * (1 - adaptive_xi))
           
#         #     self.primary_batch_size = self.batch_size - self.neg_batch_size - self.pos_batch_size


#         #     # Optional safety clamps
#         #     self.pos_batch_size = max(1, self.pos_batch_size)
#         #     self.neg_batch_size = max(1, self.neg_batch_size)
        

#         #     # Store for tracking if needed
#         #     self.current_xi = adaptive_xi


#         #     print(f"[DualBuffer] Linear adaptive xi: {adaptive_xi:.3f}, Success WMA: {self.success_rate_wma:.3f}")
#         #     print(f"[DualBuffer] pos_batch_size: {self.pos_batch_size}, neg_batch_size: {self.neg_batch_size}. primary_batch_size: {self.primary_batch_size}")

#         #"Dual_Buffer
#         # 
#         # 
#         # +HER_Fetch_Slide_adaptive_xi_fixed_pos_low_neg_high_her_change"
#         # elif self.xi_mode == 'adaptive':
#         #     # Update moving average of success rate
#         #     self.success_rate_wma=success_rate
#         #     # Linearly interpolate xi (positive sampling ratio)
#         #     adaptive_xi = self.xi_max - (self.xi_max - self.xi_min) * self.success_rate_wma
#         #     reserved_primary_ratio = 0.3 
#         #     adaptive_budget = self.batch_size * (reserved_primary_ratio)

#         #     # Convert xi into batch sizes
#         #     self.pos_batch_size = int(adaptive_budget * adaptive_xi)
#         #     self.neg_batch_size = int(adaptive_budget * (1 - adaptive_xi))
           
#         #     self.primary_batch_size = self.batch_size - self.neg_batch_size - self.pos_batch_size


#         #     # Optional safety clamps
#         #     self.pos_batch_size = max(1, self.pos_batch_size)
#         #     self.neg_batch_size = max(1, self.neg_batch_size)
        

#         #     # Store for tracking if needed
#         #     self.current_xi = adaptive_xi


#         #     print(f"[DualBuffer] Linear adaptive xi: {adaptive_xi:.3f}, Success WMA: {self.success_rate_wma:.3f}")
#         #     print(f"[DualBuffer] pos_batch_size: {self.pos_batch_size}, neg_batch_size: {self.neg_batch_size}. primary_batch_size: {self.primary_batch_size}")

#         #"Dual_Buffer+HER_Fetch_Slide_adaptive_xi_fixed_pos_high_neg_low_her_change" 
#         elif self.xi_mode == 'adaptive':
#             # Update moving average of success rate
#             self.success_rate_wma=success_rate
#             # Linearly interpolate xi (positive sampling ratio)
#             adaptive_xi = self.xi_max - (self.xi_max - self.xi_min) * self.success_rate_wma
#             reserved_primary_ratio = 0.3 
#             adaptive_budget = self.batch_size * (reserved_primary_ratio)

#             # Convert xi into batch sizes
#             self.neg_batch_size = int(adaptive_budget * adaptive_xi)
#             self.pos_batch_size = int(adaptive_budget * (1 - adaptive_xi))
           
#             self.primary_batch_size = self.batch_size - self.neg_batch_size - self.pos_batch_size


#             # Optional safety clamps
#             self.pos_batch_size = max(1, self.pos_batch_size)
#             self.neg_batch_size = max(1, self.neg_batch_size)
        
#             # Store for tracking if needed
#             self.current_xi = adaptive_xi


#             print(f"[DualBuffer] Linear adaptive xi: {adaptive_xi:.3f}, Success WMA: {self.success_rate_wma:.3f}")
#             print(f"[DualBuffer] pos_batch_size: {self.pos_batch_size}, neg_batch_size: {self.neg_batch_size}. primary_batch_size: {self.primary_batch_size}")




#         # elif self.xi_mode == 'adaptive':
#         #      # Update moving average of success rate
#         #     self.success_rate_wma = success_rate

#         #     # Linearly interpolate xi (positive sample ratio)
#         #     adaptive_xi = self.xi_max - (self.xi_max - self.xi_min) * self.success_rate_wma

#         #     # Define how much of the batch should be adaptive
#         #     adaptive_budget = self.batch_size

#         #     # Pos size based on xi
#         #     self.pos_batch_size = int(adaptive_budget * adaptive_xi)

#         #     # Remaining split between neg and primary equally
#         #     remaining = adaptive_budget - self.pos_batch_size
#         #     half_remaining = remaining // 2

#         #     self.neg_batch_size = half_remaining
#         #     self.primary_batch_size = adaptive_budget - self.pos_batch_size - self.neg_batch_size

#         #     # Safety clamps (optional)
#         #     self.pos_batch_size = max(1, self.pos_batch_size)
#         #     self.neg_batch_size = max(1, self.neg_batch_size)
#         #     self.primary_batch_size = max(1, self.primary_batch_size)

#         #     # Logging for debug
#         #     print(f"[DualBuffer] success_wma: {self.success_rate_wma:.3f}, xi: {adaptive_xi:.3f}")
#         #     print(f"[DualBuffer] pos: {self.pos_batch_size}, neg: {self.neg_batch_size}, primary: {self.primary_batch_size}")


#         # elif self.xi_mode == 'prioritized':
#         #     batch_priorities = np.abs(priorities)
         

#         #     # Split priorities into primary, pos, neg segments
#         #     prio_primary = np.mean(batch_priorities[:self.primary_batch_size]) if batch_priorities[:self.primary_batch_size].size > 0 else 0.0
#         #     prio_pos = np.mean(batch_priorities[self.primary_batch_size : self.primary_batch_size + self.pos_batch_size]) if batch_priorities[self.primary_batch_size : self.primary_batch_size + self.pos_batch_size].size > 0 else 0.0
#         #     prio_neg = np.mean(batch_priorities[self.primary_batch_size + self.pos_batch_size :]) if batch_priorities[self.primary_batch_size + self.pos_batch_size :].size > 0 else 0.0

#         #     # Compute weighted priorities
#         #     weighted_primary = prio_primary ** self.xi_prioritized_alpha
#         #     weighted_pos = prio_pos ** self.xi_prioritized_alpha
#         #     weighted_neg = prio_neg ** self.xi_prioritized_alpha

#         #     total_priority = weighted_primary + weighted_pos + weighted_neg
            
#         #     # Compute probabilities (clipped)
#         #     prob_primary = np.clip(weighted_primary / total_priority, self.xi_min, self.xi_max)

#         #     prob_pos = np.clip(weighted_pos / total_priority, self.xi_min, self.xi_max)
#         #     prob_neg = np.clip(weighted_neg / total_priority, self.xi_min, self.xi_max)
            

#         #     # Update sampling ratios
#         #     self.xi_primary_ratio = prob_primary  # Primary buffer ratio
#         #     self.xi_pos= prob_pos                # Positive buffer ratio
#         #     self.xi_neg = prob_neg                # Negative buffer ratio

#         #     # Calculate batch sizes
            
      

#         #     # Split secondary batch between pos/neg
#         #     self.pos_batch_size = int(self.batch_size * self.xi_pos)
#         #     self.neg_batch_size = int(self.batch_size * self.xi_neg)

#         #     # self.config["buffer"]["dual_buffer"]["xi"]["primary_ratio"] = self.xi_primary_ratio
#         #     # self.config["buffer"]["dual_buffer"]["xi"]["pos_ratio"] = self.xi_pos
#         #     # self.config["buffer"]["dual_buffer"]["xi"]["neg_ratio"] = self.xi_neg
#         #     # self.config["buffer"]["dual_buffer"]["adaptive"]["primary_batch_size"] = self.primary_batch_size
#         #     # self.config["buffer"]["dual_buffer"]["adaptive"]["pos_batch_size"] = self.pos_batch_size
#         #     # self.config["buffer"]["dual_buffer"]["adaptive"]["neg_batch_size"] = self.neg_batch_size
#         #     # self.config["buffer"]["dual_buffer"]["adaptive"]["secondary_batch_size"] = self.secondary_batch_size

#         #     print(f"Primary Buffer Size: {self.primary_batch_size}")
#         #     print(f"Positive Buffer Size: {self.pos_batch_size}, Negative Buffer Size: {self.neg_batch_size}")
#         #     print(f"Primary Buffer Ratio: {self.xi_primary_ratio:.2f}, Positive Buffer Ratio: {self.xi_pos:.2f}, Negative Buffer Ratio: {self.xi_neg:.2f}")
#         # elif self.xi_mode == 'adaptive':
#         #     batch_priorities = np.abs(priorities)
         

#         #     # Split priorities into primary, pos, neg segments
#         #     prio_primary = np.mean(batch_priorities[:self.primary_batch_size]) if batch_priorities[:self.primary_batch_size].size > 0 else 0.0
#         #     prio_pos = np.mean(batch_priorities[self.primary_batch_size : self.primary_batch_size + self.pos_batch_size]) if batch_priorities[self.primary_batch_size : self.primary_batch_size + self.pos_batch_size].size > 0 else 0.0
#         #     prio_neg = np.mean(batch_priorities[self.primary_batch_size + self.pos_batch_size :]) if batch_priorities[self.primary_batch_size + self.pos_batch_size :].size > 0 else 0.0

#         #     # Compute weighted priorities
#         #     # Example: Success-driven adaptation
#         #     min_success = 0.2
#         #     max_success = 0.8

#         #     # Normalize success rate within bounds
#         #     normalized_success = np.clip((success_rate - min_success) / (max_success - min_success), 0, 1)

#         #     # Linearly interpolate alpha between 0.2 and 0.8
#         #     self.xi_prioritized_alpha = min_success + (max_success-min_success) * normalized_success
    

#         #     weighted_primary = prio_primary ** self.xi_prioritized_alpha
#         #     weighted_pos = prio_pos ** self.xi_prioritized_alpha
#         #     weighted_neg = prio_neg ** self.xi_prioritized_alpha

#         #     total_priority = weighted_primary + weighted_pos + weighted_neg
            
#         #     # Compute probabilities (clipped)
#         #     prob_primary = np.clip(weighted_primary / total_priority, self.xi_min, self.xi_max)


#         #     prob_pos = np.clip(weighted_pos / total_priority, self.xi_min, 0.35)
#         #     prob_neg = np.clip(weighted_neg / total_priority, 0.25, self.xi_max)
            

#         #     # Update sampling ratios
#         #     # self.xi_primary_ratio = prob_primary  # Primary buffer ratio
#         #     self.xi_primary_ratio = max(self.xi_min, 1.0 - self.xi_pos - self.xi_neg)

#         #     self.xi_pos= prob_pos                # Positive buffer ratio
#         #     self.xi_neg = prob_neg                # Negative buffer ratio

#         #     # Calculate batch sizes
            
      

#         #     # Split secondary batch between pos/neg
#         #     self.pos_batch_size = int(self.batch_size * self.xi_pos)
#         #     self.neg_batch_size = int(self.batch_size * self.xi_neg)

#         #     # self.config["buffer"]["dual_buffer"]["xi"]["primary_ratio"] = self.xi_primary_ratio
#         #     # self.config["buffer"]["dual_buffer"]["xi"]["pos_ratio"] = self.xi_pos
#         #     # self.config["buffer"]["dual_buffer"]["xi"]["neg_ratio"] = self.xi_neg
#         #     # self.config["buffer"]["dual_buffer"]["adaptive"]["primary_batch_size"] = self.primary_batch_size
#         #     # self.config["buffer"]["dual_buffer"]["adaptive"]["pos_batch_size"] = self.pos_batch_size
#         #     # self.config["buffer"]["dual_buffer"]["adaptive"]["neg_batch_size"] = self.neg_batch_size
#         #     # self.config["buffer"]["dual_buffer"]["adaptive"]["secondary_batch_size"] = self.secondary_batch_size

#         #     print(f"Primary Buffer Size: {self.primary_batch_size}")
#         #     print(f"Positive Buffer Size: {self.pos_batch_size}, Negative Buffer Size: {self.neg_batch_size}")
#         #     print(f"Primary Buffer Ratio: {self.xi_primary_ratio:.2f}, Positive Buffer Ratio: {self.xi_pos:.2f}, Negative Buffer Ratio: {self.xi_neg:.2f}")
#         #     print(f"Adaptive Alpha: {self.xi_prioritized_alpha:.2f}")
#         #     print(f"Success Rate: {success_rate:.2f}")
            
#         # elif self.xi_mode == 'selfpacedadaptive':
            
#         #     batch_priorities = np.abs(priorities)

#         #     # Split priorities
#         #     prio_primary = np.mean(batch_priorities[:self.primary_batch_size]) if batch_priorities[:self.primary_batch_size].size > 0 else 0.0
#         #     prio_pos = np.mean(batch_priorities[self.primary_batch_size : self.primary_batch_size + self.pos_batch_size]) if batch_priorities[self.primary_batch_size : self.primary_batch_size + self.pos_batch_size].size > 0 else 0.0
#         #     prio_neg = np.mean(batch_priorities[self.primary_batch_size + self.pos_batch_size :]) if batch_priorities[self.primary_batch_size + self.pos_batch_size :].size > 0 else 0.0

#         #     # === SELF-PACED PART ===

#         #     # Track max success rate achieved so far
#         #     if not hasattr(self, 'best_success_rate'):
#         #         self.best_success_rate = success_rate
#         #     else:
#         #         self.best_success_rate = max(self.best_success_rate, success_rate)

#         #     # Interpolate alpha only based on *best success rate*
#         #     min_success = 0.2
#         #     max_success = 0.8

#         #     normalized_best_success = np.clip((self.best_success_rate - min_success) / (max_success - min_success), 0, 1)
#         #     self.xi_prioritized_alpha = min_success + (max_success-min_success)* normalized_best_success  # Range: 0.2 to 0.8

#         #     # === PRIORITY CALCULATION ===
#         #     weighted_primary = prio_primary ** self.xi_prioritized_alpha
#         #     weighted_pos = prio_pos ** self.xi_prioritized_alpha
#         #     weighted_neg = prio_neg ** self.xi_prioritized_alpha

#         #     total_priority = weighted_primary + weighted_pos + weighted_neg

#         #     # Compute ratios
#         #     # prob_primary = np.clip(weighted_primary / total_priority, self.xi_min, self.xi_max)
#         #     # prob_pos = np.clip(weighted_pos / total_priority, self.xi_min, self.xi_max)
#         #     # prob_neg = np.clip(weighted_neg / total_priority, self.xi_min, self.xi_max)
#         #     prob_pos = np.clip(weighted_pos / total_priority, self.xi_min, 0.35)
#         #     prob_neg = np.clip(weighted_neg / total_priority, 0.25, self.xi_max)
            

#         #     # Update sampling ratios
#         #     # self.xi_primary_ratio = prob_primary  # Primary buffer ratio
#         #     self.xi_primary_ratio = max(self.xi_min, 1.0 - self.xi_pos - self.xi_neg)


#         #     self.xi_pos = prob_pos
#         #     self.xi_neg = prob_neg

#         #     # Update batch sizes
#         #     self.pos_batch_size = int(self.batch_size * self.xi_pos)
#         #     self.neg_batch_size = int(self.batch_size * self.xi_neg)

#             # Logging
#             # print(f"[SELF-PACED] Best Success Rate: {self.best_success_rate:.2f}")
#             # print(f"Alpha: {self.xi_prioritized_alpha:.2f}")
#             # print(f"Ratios → Primary: {prob_primary:.2f}, Pos: {prob_pos:.2f}, Neg: {prob_neg:.2f}")


        
# #     def update_priorities(
# #     self, 
# #     priorities: np.ndarray, success_rate: float
# # ) -> None:
# #         """ Self-Paced Adaptive Prioritization using Logarithmic & Exponential Scaling (Without Sigmoid). """
        
# #         self.primary_batch_size = self.get_primary_batch_size()
# #         if self.xi_mode == 'fix':
# #             return

# #         elif self.xi_mode == 'prioritized':
# #             batch_priorities = np.abs(priorities)

# #             # **1️⃣ Compute Mean Priorities for Each Buffer**
# #             prio_primary = np.mean(batch_priorities[:self.primary_batch_size]) if self.primary_batch_size > 0 else 0.0
# #             prio_pos = np.mean(batch_priorities[self.primary_batch_size:self.primary_batch_size + self.pos_batch_size]) if self.pos_batch_size > 0 else 0.0
# #             prio_neg = np.mean(batch_priorities[self.primary_batch_size + self.pos_batch_size:]) if self.neg_batch_size > 0 else 0.0

# #             # **2️⃣ Compute Weighted Priorities**
# #             weighted_primary = prio_primary ** self.xi_prioritized_alpha
# #             weighted_pos = prio_pos ** self.xi_prioritized_alpha
# #             weighted_neg = prio_neg ** self.xi_prioritized_alpha

# #             total_priority = weighted_primary + weighted_pos + weighted_neg + 1e-8  # Avoid division by zero

# #             # **3️⃣ Compute Base Probabilities**
# #             prob_primary = np.clip(weighted_primary / total_priority, self.xi_min, self.xi_max)
            
# #             # **4️⃣ Apply Logarithmic Scaling for `prob_pos` (Smooth Increase)**
# #             log_factor = np.log1p(success_rate) / np.log1p(1)  # Normalized log scaling
# #             prob_pos = self.xi_min + (self.xi_max - self.xi_min) * log_factor  # Log interpolation

# #             # **5️⃣ Apply Exponential Decay for `prob_neg` (Smooth Reduction)**
# #             prob_neg = np.exp(-self.log_scale * success_rate)  # Exponential decay to remove negative samples
# #             prob_neg = max(0.0, prob_neg)  # Ensure `prob_neg` never goes negative

# #             # **6️⃣ Prevent Difficulty Reduction (Self-Paced Locking)**
# #             self.max_achieved_prob = max(self.max_achieved_prob, prob_pos)  # Track max difficulty
# #             prob_pos = max(self.max_achieved_prob * 0.98, prob_pos)  # Prevent difficulty reduction
# #             prob_neg = max(0.0, prob_neg)  # Ensure negative probability fully reduces when success is high

# #             # **7️⃣ Normalize Probabilities**
# #             prob_primary = max(0.15, 1 - (prob_pos + prob_neg))  # Ensure primary buffer is maintained
# #             total_prob = prob_primary + prob_pos + prob_neg
# #             prob_primary /= total_prob
# #             prob_pos /= total_prob
# #             prob_neg /= total_prob

# #             # **8️⃣ Update Sampling Ratios**
# #             self.xi_primary_ratio = prob_primary
# #             self.xi_pos = prob_pos
# #             self.xi_neg = prob_neg

# #             # **9️⃣ Recalculate Batch Sizes**
# #             self.pos_batch_size = int(self.batch_size * self.xi_pos)
# #             self.neg_batch_size = int(self.batch_size * self.xi_neg)
# #             self.primary_batch_size = max(1, self.batch_size - self.pos_batch_size - self.neg_batch_size)

# #             # **🔟 Debugging Output**
# #             print(f"📊 Self-Paced Prioritization → Primary: {self.xi_primary_ratio:.2f}, Pos: {self.xi_pos:.2f}, Neg: {self.xi_neg:.2f}")
# #             print(f"🔄 Batch Sizes → Primary: {self.primary_batch_size}, Pos: {self.pos_batch_size}, Neg: {self.neg_batch_size}")
# #             print(f"✅ Smoothed Success Rate: {self.success_rate_wma:.3f} | Max Achieved Pos Prob: {self.max_achieved_prob:.2f}")



     

            
           


#         else:
#             raise ValueError(f"Unknown xi_mode: {self.xi_mode}")
#     # def update_success_rate(self, new_success: float):
#     #     """ Smooth success rate tracking using a weighted moving average (WMA). """


#     # def update_priorities(self, priorities: np.ndarray, success_rate: float) -> None:
#     #     """ Log-Based Adaptive Prioritization with Difficulty Scaling. """

#     #     self.primary_batch_size = self.get_primary_batch_size()
#     #     if self.xi_mode == 'fix':
#     #         return

#     #     elif self.xi_mode == 'prioritized':
#     #         # **1️⃣ Update Success Rate with Weighted Moving Average**
#     #         beta = 0.9  # Smoothing factor
#     #         self.success_rate_wma = beta * self.success_rate_wma + (1 - beta) * success_rate  # Smoothed success rate

#     #         batch_priorities = np.abs(priorities)

#     #         # **2️⃣ Compute Mean Priorities for Each Buffer**
#     #         prio_primary = np.mean(batch_priorities[:self.primary_batch_size]) if self.primary_batch_size > 0 else 0.0
#     #         prio_pos = np.mean(batch_priorities[self.primary_batch_size:self.primary_batch_size + self.pos_batch_size]) if self.pos_batch_size > 0 else 0.0
#     #         prio_neg = np.mean(batch_priorities[self.primary_batch_size + self.pos_batch_size:]) if self.neg_batch_size > 0 else 0.0

#     #         # **3️⃣ Compute Weighted Priorities**
#     #         weighted_primary = prio_primary ** self.xi_prioritized_alpha
#     #         weighted_pos = prio_pos ** self.xi_prioritized_alpha
#     #         weighted_neg = prio_neg ** self.xi_prioritized_alpha

#     #         total_priority = weighted_primary + weighted_pos + weighted_neg + 1e-8  # Avoid division by zero

#     #         # **4️⃣ Compute Base Probabilities**
#     #         prob_primary = np.clip(weighted_primary / total_priority, self.xi_min, self.xi_max)
#     #         prob_pos = np.clip(weighted_pos / total_priority, self.xi_min, self.xi_max)
#     #         prob_neg = np.clip(weighted_neg / total_priority, self.xi_min, self.xi_max)

#     #         # **5️⃣ Apply Sigmoid-Based Probability Transition for `prob_pos`**
#     #         adaptive_success = max(self.success_rate_wma, self.base_success_rate)  # Ensure base stability
#     #         prob_pos = self.xi_min + (self.xi_max - self.xi_min) * (1 / (1 + np.exp(-self.sigmoid_scale * (adaptive_success - self.sigmoid_shift))))

#     #         # **6️⃣ Apply Logarithmic Scaling for `prob_neg`**
#     #         prob_neg = np.exp(-self.log_scale * prob_pos)  # Log-decay instead of hard clipping
#     #         prob_neg = prob_neg / (prob_neg + prob_pos + 1e-8)  # Normalize

#     #         # **7️⃣ Prevent Difficulty Reduction (Self-Paced Locking)**
#     #         self.max_achieved_prob = max(self.max_achieved_prob, prob_pos)  # Track max difficulty
#     #         prob_pos = max(self.max_achieved_prob * 0.98, prob_pos)  # Prevent difficulty reduction
#     #         prob_neg = max(0.1, prob_neg)  # Ensure some negative exploration

#     #         # **8️⃣ Normalize Probabilities**
#     #         prob_primary = max(0.15, 1 - (prob_pos + prob_neg))  # Ensure primary buffer is maintained
#     #         total_prob = prob_primary + prob_pos + prob_neg
#     #         prob_primary /= total_prob
#     #         prob_pos /= total_prob
#     #         prob_neg /= total_prob

#     #         # **9️⃣ Update Sampling Ratios**
#     #         self.xi_primary_ratio = prob_primary
#     #         self.xi_pos = prob_pos
#     #         self.xi_neg = prob_neg

#     #         # **🔟 Recalculate Batch Sizes**
#     #         self.pos_batch_size = int(self.batch_size * self.xi_pos)
#     #         self.neg_batch_size = int(self.batch_size * self.xi_neg)
#     #         self.primary_batch_size = max(1, self.batch_size - self.pos_batch_size - self.neg_batch_size)

#     #         # **Debugging Output**
#     #         print(f"📊 Log-Based Prioritization → Primary: {self.xi_primary_ratio:.2f}, Pos: {self.xi_pos:.2f}, Neg: {self.xi_neg:.2f}")
#     #         print(f"🔄 Batch Sizes → Primary: {self.primary_batch_size}, Pos: {self.pos_batch_size}, Neg: {self.neg_batch_size}")
#     #         print(f"✅ Smoothed Success Rate: {self.success_rate_wma:.3f} | Max Achieved Pos Prob: {self.max_achieved_prob:.2f}")


        
#     # def update_priorities(self, priorities: np.ndarray, success_rate: float) -> None:
#     #     """ Hybrid Adaptive Prioritization: Fully Adaptive → Self-Paced Transition. """

#     #     # **1️⃣ Update Success Rate with Weighted Moving Average (Smooth Transitions)**
#     #     beta = 0.9  # Smoothing factor for gradual adaptation
#     #     self.success_rate_wma = beta * self.success_rate_wma + (1 - beta) * success_rate  # Smoothed success rate

#     #     # **2️⃣ Compute Base Priorities**
#     #     batch_priorities = np.abs(priorities)
#     #     weighted_priority = batch_priorities ** 0.5  # xi_prioritized_alpha
#     #     total_priority = np.sum(weighted_priority) + 1e-8  # Avoid division by zero
#     #     prob_base = weighted_priority / total_priority  # Normalize
#     #     self.prob_positive=0.0

#     #     # **3️⃣ Fully Adaptive Phase (Exploration)**
#     #     if self.success_rate_wma < 0.8:  # Allow difficulty to change in both directions
#     #         prob_pos = self.xi_min + (self.xi_max - self.xi_min) * (1 / (1 + np.exp(-self.sigmoid_scale * (self.success_rate_wma - self.sigmoid_shift))))
#     #         prob_neg = 1 - prob_pos  # Ensure total sum = 1
#     #         self.max_achieved_prob = max(self.max_achieved_prob, prob_pos)  # Track highest difficulty
#     #         self.prob_positive=prob_pos

#     #     # **4️⃣ Self-Paced Phase (Mastery)**
#     #     else:  

#     #         prob_pos = max(self.max_achieved_prob, self.prob_positive)  # Never decrease difficulty
#     #         prob_neg = 1 - prob_pos  # Adjust neg accordingly

#     #     # **5️⃣ Normalize Probabilities**
#     #     prob_primary = max(0.1, 1 - (prob_pos + prob_neg))  # Ensure primary isn't zero
#     #     total_prob = prob_primary + prob_pos + prob_neg
#     #     prob_primary /= total_prob
#     #     prob_pos /= total_prob
#     #     prob_neg /= total_prob

#     #     # **6️⃣ Update Sampling Ratios**
#     #     self.xi_primary_ratio = prob_primary
#     #     self.xi_pos = prob_pos
#     #     self.xi_neg = prob_neg

#     #     # **7️⃣ Recalculate Batch Sizes**
#     #     self.pos_batch_size = int(self.batch_size * self.xi_pos)
#     #     self.neg_batch_size = int(self.batch_size * self.xi_neg)
#     #     self.primary_batch_size = max(1, self.batch_size - self.pos_batch_size - self.neg_batch_size)

#     #     # **8️⃣ Debugging Output**
#     #     print(f"📊 Hybrid Adaptive Prioritization → Primary: {self.xi_primary_ratio:.2f}, Pos: {self.xi_pos:.2f}, Neg: {self.xi_neg:.2f}")
#     #     print(f"🔄 Batch Sizes → Primary: {self.primary_batch_size}, Pos: {self.pos_batch_size}, Neg: {self.neg_batch_size}")
#     #     print(f"✅ Smoothed Success Rate: {self.success_rate_wma:.3f} | Max Achieved Pos Prob: {self.max_achieved_prob:.2f}")
#     # def update_priorities(self, priorities: np.ndarray, success_rate: float) -> None:
#     #     """ Self-Paced Adaptive Prioritization with Difficulty Locking & Exploration Control. """
        
#     #     # **1️⃣ Update Success Rate with Weighted Moving Average**
#     #     beta = 0.9  # Smoothing factor
#     #     self.success_rate_wma = beta * self.success_rate_wma + (1 - beta) * success_rate

#     #     # **2️⃣ Compute Base Priorities**
#     #     batch_priorities = np.abs(priorities)
#     #     weighted_priority = batch_priorities ** 0.5  # Adaptive scaling factor
#     #     total_priority = np.sum(weighted_priority) + 1e-8  # Avoid division by zero
#     #     prob_base = weighted_priority / total_priority

#     #     # **3️⃣ Apply Self-Paced Sigmoid Transition**
#     #     adaptive_success = max(self.success_rate_wma, self.base_success_rate)  # Ensure base stability
#     #     prob_pos = self.xi_min + (self.xi_max - self.xi_min) * (1 / (1 + np.exp(-self.sigmoid_scale * (adaptive_success - self.sigmoid_shift))))
        
#     #     # **4️⃣ Prevent Difficulty Reduction (Self-Paced Locking)**
#     #     self.max_achieved_prob = max(self.max_achieved_prob, prob_pos)  # Keep track of max difficulty
#     #     prob_pos = max(self.max_achieved_prob * 0.98, prob_pos)  # Prevent reduction
#     #     prob_neg = max(0.1, 1 - prob_pos)  # Keep exploration at least 10%

#     #     # **5️⃣ Stage-Based Adaptation for Smooth Transitions**
#     #     if self.success_rate_wma < 0.3:  # Early learning phase → increase exploration
#     #         prob_pos = max(0.3, prob_pos - 0.05)
#     #         prob_neg = min(0.4, prob_neg + 0.05)  # Ensure failures are explored
        
#     #     elif self.success_rate_wma > 0.7:  # Mastery phase → focus on positive reinforcement
#     #         prob_pos = min(0.85, prob_pos + 0.05)
#     #         prob_neg = max(0.05, 1 - prob_pos)

#     #     # **6️⃣ Normalize Probabilities**
#     #     prob_primary = max(0.15, 1 - (prob_pos + prob_neg))  # Ensure primary buffer is maintained
#     #     total_prob = prob_primary + prob_pos + prob_neg
#     #     prob_primary /= total_prob
#     #     prob_pos /= total_prob
#     #     prob_neg /= total_prob

#     #     # **7️⃣ Update Sampling Ratios**
#     #     self.xi_primary_ratio = prob_primary
#     #     self.xi_pos = prob_pos
#     #     self.xi_neg = prob_neg

#     #     # **8️⃣ Recalculate Batch Sizes**
#     #     self.pos_batch_size = int(self.batch_size * self.xi_pos)
#     #     self.neg_batch_size = int(self.batch_size * self.xi_neg)
#     #     self.primary_batch_size = max(1, self.batch_size - self.pos_batch_size - self.neg_batch_size)

#     #     # **9️⃣ Debugging Output**
#     #     print(f"📊 Self-Paced Prioritization → Primary: {self.xi_primary_ratio:.2f}, Pos: {self.xi_pos:.2f}, Neg: {self.xi_neg:.2f}")
#     #     print(f"🔄 Batch Sizes → Primary: {self.primary_batch_size}, Pos: {self.pos_batch_size}, Neg: {self.neg_batch_size}")
#     #     print(f"✅ Smoothed Success Rate: {self.success_rate_wma:.3f} | Max Achieved Pos Prob: {self.max_achieved_prob:.2f}")


#     # def update_priorities(self, priorities: np.ndarray, success_rate: float) -> None:
#     #     """ Self-Paced Adaptive Prioritization using a Base Level & Sigmoid-Based Transition. """
        
#     #     # **1️⃣ Update Success Rate with Weighted Moving Average**
#     #     beta = 0.9  # Smoothing factor
#     #     self.success_rate_wma = beta * self.success_rate_wma + (1 - beta) * success_rate

#     #     # **2️⃣ Compute Base Priorities**
#     #     batch_priorities = np.abs(priorities)
#     #     weighted_priority = batch_priorities ** 0.5  # xi_prioritized_alpha
#     #     total_priority = np.sum(weighted_priority) + 1e-8  # Avoid div by zero
#     #     prob_base = weighted_priority / total_priority

#     #     # **3️⃣ Apply Self-Paced Sigmoid Transition**
#     #     adaptive_success = self.success_rate_wma 

#     #     prob_pos = self.xi_min + (self.xi_max - self.xi_min) * (1 / (1 + np.exp(-self.sigmoid_scale * (adaptive_success - self.sigmoid_shift))))
#     #     prob_neg = 1 - prob_pos  # Ensure sum = 1

#     #     # # **4️⃣ Prevent Difficulty Reduction (Self-Paced Mechanism)**
#     #     # self.max_achieved_prob = max(self.max_achieved_prob, prob_pos)  # Keep track of max difficulty
#     #     # prob_pos = max(self.max_achieved_prob, prob_pos)  # Prevent reduction
#     #     # prob_neg = 1 - prob_pos  # Adjust neg accordingly

#     #     # **5️⃣ Normalize Probabilities**
#     #     prob_primary = max(0.15, 1 - (prob_pos + prob_neg))  # Ensure primary is not 0
#     #     total_prob = prob_primary + prob_pos + prob_neg
#     #     prob_primary /= total_prob
#     #     prob_pos /= total_prob
#     #     prob_neg /= total_prob

#     #     # **6️⃣ Update Sampling Ratios**
#     #     self.xi_primary_ratio = prob_primary
#     #     self.xi_pos = prob_pos
#     #     self.xi_neg = prob_neg

#     #     # **7️⃣ Recalculate Batch Sizes**
#     #     self.pos_batch_size = int(self.batch_size * self.xi_pos)
#     #     self.neg_batch_size = int(self.batch_size * self.xi_neg)
#     #     self.primary_batch_size = max(1, self.batch_size - self.pos_batch_size - self.neg_batch_size)

#     #     # **8️⃣ Debugging Output**
#     #     print(f"📊 Self-Paced Prioritization → Primary: {self.xi_primary_ratio:.2f}, Pos: {self.xi_pos:.2f}, Neg: {self.xi_neg:.2f}")
#     #     print(f"🔄 Batch Sizes → Primary: {self.primary_batch_size}, Pos: {self.pos_batch_size}, Neg: {self.neg_batch_size}")
#     #     print(f"✅ Smoothed Success Rate: {self.success_rate_wma:.3f} | Max Achieved Pos Prob: {self.max_achieved_prob:.2f}")



#     # def update_priorities(self, priorities: np.ndarray, success_rate: float) -> None:
#     #     """ 
#     #     Adaptive update of priorities based on success rate trends to optimize learning.
#     #     - If success rate is high, prioritize positive experiences.
#     #     - If success rate is low, allow more negative samples for better exploration.
#     #     """

#     #     self.primary_batch_size = self.get_primary_batch_size()

#     #     if self.xi_mode == 'fix':
#     #         return

#     #     elif self.xi_mode == 'prioritized':
#     #         batch_priorities = np.abs(priorities)

#     #         # Split priorities into primary, pos, neg segments
#     #         prio_primary = np.mean(batch_priorities[:self.primary_batch_size]) if self.primary_batch_size > 0 else 0.0
#     #         prio_pos = np.mean(batch_priorities[self.primary_batch_size : self.primary_batch_size + self.pos_batch_size]) if self.pos_batch_size > 0 else 0.0
#     #         prio_neg = np.mean(batch_priorities[self.primary_batch_size + self.pos_batch_size :]) if self.neg_batch_size > 0 else 0.0

#     #         # Compute weighted priorities
#     #         weighted_primary = prio_primary ** self.xi_prioritized_alpha
#     #         weighted_pos = prio_pos ** self.xi_prioritized_alpha
#     #         weighted_neg = prio_neg ** self.xi_prioritized_alpha

#     #         total_priority = weighted_primary + weighted_pos + weighted_neg + 1e-8  # Avoid division by zero

#     #         # Compute initial probabilities (clipped)
#     #         prob_primary = np.clip(weighted_primary / total_priority, self.xi_min, self.xi_max)
#     #         prob_pos = np.clip(weighted_pos / total_priority, self.xi_min, self.xi_max)
#     #         prob_neg = np.clip(weighted_neg / total_priority, self.xi_min, self.xi_max)
#     #         # **Smooth scaling for positive experience sampling**
#     #         prob_pos = 0.5 + 0.3 * np.tanh(3 * (success_rate - 0.5))  # Smooth transition
#     #         prob_neg = 1 - prob_pos  # Ensure sum is 1

#     #         # ## **Adaptive Adjustments Based on Success Rate**
#     #         # if success_rate >= 0.7:  # **High Success → Focus on positive samples**
#     #         #     prob_pos = min(0.85, prob_pos + 0.10)  # Increase positive sampling
#     #         #     prob_neg = max(0.05, prob_neg - 0.05)  # Reduce negative sampling

#     #         # elif success_rate <= 0.4:  # **Low Success → Increase exploration**
#     #         #     prob_pos = max(0.50, prob_pos - 0.10)  # Reduce positive influence slightly
#     #         #     prob_neg = min(0.30, prob_neg + 0.10)  # Allow more failure exploration

#     #         ## **Normalize Probabilities (Ensure Sum = 1)**
#     #         total_prob = prob_primary + prob_pos + prob_neg
#     #         prob_primary /= total_prob
#     #         prob_pos /= total_prob
#     #         prob_neg /= total_prob

#     #         # **Update sampling ratios**
#     #         self.xi_primary_ratio = prob_primary
#     #         self.xi_pos = prob_pos
#     #         self.xi_neg = prob_neg

#     #         # **Recalculate batch sizes**
#     #         self.pos_batch_size = int(self.batch_size * self.xi_pos)
#     #         self.neg_batch_size = int(self.batch_size * self.xi_neg)
#     #         self.primary_batch_size = max(1, self.batch_size - self.pos_batch_size - self.neg_batch_size)  # Ensure nonzero batch

#     #         print(f"📊 Updated Ratios → Primary: {self.xi_primary_ratio:.2f}, Pos: {self.xi_pos:.2f}, Neg: {self.xi_neg:.2f}")
#     #         print(f"🔄 Updated Batch Sizes → Primary: {self.primary_batch_size}, Pos: {self.pos_batch_size}, Neg: {self.neg_batch_size}")


#     #         print(f"Success Rate: {success_rate:.2f} | Buffer Ratios -> Primary: {prob_primary:.2f}, Pos: {prob_pos:.2f}, Neg: {prob_neg:.2f}")

    
#     # def update_priorities(self, priorities: np.ndarray, success_rate: float) -> None:
#     #     """
#     #     Self-Paced Dual Buffer Update:
#     #     - When success rate is high, increase the sampling of positive experiences.
#     #     - When success rate is low, increase the sampling of negative experiences.
#     #     - Prevents drastic shifts in difficulty.
#     #     """
#     #     self.primary_batch_size = self.get_primary_batch_size()
#     #     if self.xi_mode == 'fix':
#     #         return

#     #     elif self.xi_mode == 'prioritized':
#     #         batch_priorities = np.abs(priorities)

#     #         # Split priorities into primary, pos, neg segments
#     #         prio_primary = np.mean(batch_priorities[:self.primary_batch_size]) if batch_priorities[:self.primary_batch_size].size > 0 else 0.0
#     #         prio_pos = np.mean(batch_priorities[self.primary_batch_size : self.primary_batch_size + self.pos_batch_size]) if batch_priorities[self.primary_batch_size : self.primary_batch_size + self.pos_batch_size].size > 0 else 0.0
#     #         prio_neg = np.mean(batch_priorities[self.primary_batch_size + self.pos_batch_size :]) if batch_priorities[self.primary_batch_size + self.pos_batch_size :].size > 0 else 0.0

#     #         # Compute weighted priorities
#     #         weighted_primary = prio_primary ** self.xi_prioritized_alpha
#     #         weighted_pos = prio_pos ** self.xi_prioritized_alpha
#     #         weighted_neg = prio_neg ** self.xi_prioritized_alpha

#     #         total_priority = weighted_primary + weighted_pos + weighted_neg

#     #         # Compute probabilities (clipped)
#     #         prob_primary = np.clip(weighted_primary / total_priority, self.xi_min, self.xi_max)
#     #         prob_pos = np.clip(weighted_pos / total_priority, self.xi_min, self.xi_max)
#     #         prob_neg = np.clip(weighted_neg / total_priority, self.xi_min, self.xi_max)

#     #         # Self-Paced Adjustment Based on Success Rate
#     #         min_success, max_success = 0.2, 0.7  # Competence range for adaptation
#     #         scaling_factor = np.clip((success_rate - min_success) / (max_success - min_success), 0, 1)

#     #         # Adaptive ratios based on competence level
#     #         self.xi_primary_ratio = prob_primary  
#     #         self.xi_pos = np.clip(prob_pos * (1 + scaling_factor), self.xi_min, self.xi_max)  # More positive samples
#     #         self.xi_neg = np.clip(prob_neg * (1 - scaling_factor), self.xi_min, self.xi_max)  # Fewer negative samples

#     #         # Ensure probabilities sum to 1 (normalize)
#     #         total_prob = self.xi_primary_ratio + self.xi_pos + self.xi_neg
#     #         self.xi_primary_ratio /= total_prob
#     #         self.xi_pos /= total_prob
#     #         self.xi_neg /= total_prob

#     #         # Calculate new batch sizes
#     #         self.pos_batch_size = int(self.batch_size * self.xi_pos)
#     #         self.neg_batch_size = int(self.batch_size * self.xi_neg)

          

#     #     else:
#     #         raise ValueError(f"Unknown xi_mode: {self.xi_mode}")

#     def is_sampling_possible(self) -> bool:
#         return True if self.replay_buffer.size > 0 else False
    
    