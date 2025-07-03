from typing import Dict, List, Tuple
import numpy as np
import math
import csv
from buffer.replay_buffer import ReplayBuffer
from dual_buffer.dual_buffer import DualBuffer

Transition = Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]

class PredefinedDualBuffer(DualBuffer):
    """Stores experiences in a primary buffer and separate positive and negative buffers.
    
    lambda thresholds for the positive and negative buffers are updated based on the training progress.
    """

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.profile=self.config["buffer"]["dual_buffer"]["predefined"]["profile"]

        # Positive lambda configuration
        pos_config = config['buffer']['dicl']['lambda']['predefined']
        self.lambda_pos_start = pos_config['lambda_pos_start']
        self.lambda_pos_end = pos_config['lambda_pos_end']
        self.current_lambda_pos = self.lambda_pos_start

        # Negative lambda configuration
        self.lambda_neg_start = pos_config['lambda_neg_start']
        self.lambda_neg_end = pos_config['lambda_neg_end']
        self.current_lambda_neg = self.lambda_neg_start
        
        # Total training steps
        self.total_timesteps = float(config['trainer']['total_timesteps'])
        self.lambda_pos=self.config["buffer"]["dual_buffer"]["predefined"]["lambda_pos"]
        self.lambda_neg=self.config["buffer"]["dual_buffer"]["predefined"]["lambda_neg"]
        # Metrics bookkeeping
        self._counts = {'pos': 0, 'neg': 0, 'primary': 0}
        self.metrics = {
            'timestep': [],
            't_rel_pct': [],
            'pos_count': [],
            'neg_count': [],
            'primary_count': []
        }
        # Event log per episode
        self.events: List[Tuple[int, float, str]] = []  # (timestep, t_rel_pct, buffer_used)

    # def _record(self, t: int, buffer_used: str) -> None:
    #         """Increment counters and snapshot metrics (including timestep) for CSV export."""
    #         t_rel_pct = (min(max(float(t) / self.total_timesteps, 0.0), 1.0)) * 100
        

    #         if buffer_used == "Positive Buffer":
    #             self._counts['pos'] += 1
    #             self._counts['primary'] += 1
    #         elif buffer_used == "Negative Buffer":
    #             self._counts['neg'] += 1
    #         else:
    #             self._counts['primary'] += 1
    #         self.events.append((t, t_rel_pct, buffer_used))



    #         # record raw timestep and counts
    #         self.metrics['timestep'].append(t)
    #         self.metrics['t_rel_pct'].append(t_rel_pct)
    #         self.metrics['pos_count'].append(self._counts['pos'])
    #         self.metrics['neg_count'].append(self._counts['neg'])
    #         self.metrics['primary_count'].append(self._counts['primary'])

    def _compute_lambda(self, lambda_start: float, lambda_end: float, t_rel: float) -> Tuple[float,float]: # type: ignore
        """Compute the updated lambda based on the selected profile."""
      
        print(f"Using:{self.profile}")
        if self.profile=="linear":
            lambda_pos=lambda_start + t_rel * (lambda_end - lambda_start)
            lambda_neg=lambda_start
        elif self.profile == "exp":
            lambda_pos=lambda_start * (lambda_end / lambda_start) ** t_rel
            lambda_neg=lambda_start
            return lambda_pos, lambda_neg

    def update_lambda(self, t: int) -> None:
        """Update both positive and negative lambda thresholds based on training progress."""
        print(t)
        t_rel = float(t) / self.total_timesteps

        # t_rel=min(max(t_rel,0.0),1.0)
        
        self.current_lambda_pos,_= self._compute_lambda(
   
            self.lambda_pos_start,
            self.lambda_pos_end,
            t_rel
        )
        
        _,self.current_lambda_neg = self._compute_lambda(
            
            self.lambda_neg_start,
            self.lambda_neg_end,
            t_rel
        )

    def _store_episode_in_buffer(self, episode: List[Transition], buffer, condition: bool) -> None:
        """Helper method to store an episode into the given buffer if the condition holds."""
        if condition:
            for (o, a, r, o2, d) in episode:
                buffer.store(o, a, r, o2, d)

    def store_episode_pos(self, episode: List[Transition], t: int, success: bool) -> Tuple[float, float]:
        """Store the episode into the positive buffer if success is True and reward sum exceeds threshold."""
        self.update_lambda(t)
        sum_rew = sum(r for (_, _, r, _, _) in episode)
        # Check if success is True (or non-zero) and meets the lambda threshold
        if success and sum_rew >= self.current_lambda_pos:
            self.lambda_pos=self.current_lambda_pos
            self._store_episode_in_buffer(episode, self.pos_buffer, True)
            buffer_used = "Positive Buffer"
           
        else:
            buffer_used = "Primary Buffer"
        self._record(t, buffer_used)

        print(f"Stored episode | Sum reward: {sum_rew:.2f} | "
              f"lambda pos: {self.current_lambda_pos:.2f} | lambda neg: {self.current_lambda_neg:.2f} | "
              f"Buffer used: {buffer_used}")
        return self.current_lambda_pos, self.current_lambda_neg

    def store_episode_neg(self, episode: List[Transition], t: int, success: bool) -> Tuple[float, float]:
        """Store the episode into the negative buffer if success is False and reward sum is below threshold."""
        self.update_lambda(t)
        sum_rew = sum(r for (_, _, r, _, _) in episode)
        if (not success) and sum_rew <= self.current_lambda_neg:
            self.lambda_neg=self.current_lambda_neg
            self._store_episode_in_buffer(episode, self.neg_buffer, True)
            buffer_used = "Negative Buffer"
        else:
            buffer_used = "Primary Buffer"
        self._record(t, buffer_used)
        print(f"Stored episode | Sum reward: {sum_rew:.2f} | "
              f"lambda neg: {self.current_lambda_neg:.2f} | lambda pos: {self.current_lambda_pos:.2f} | "
              f"Buffer used: {buffer_used}")
        return self.current_lambda_pos, self.current_lambda_neg

    def is_sampling_possible(self) -> bool:
        """Check if sampling is possible from the primary replay buffer."""
        return self.replay_buffer.size > 0
    # def save_metrics_csv(self, filename: str = "buffer_metrics.csv") -> None:
    #     """Write out timestep, t_rel_pct, pos_count, neg_count, primary_count to a CSV."""
    #     if not self.metrics['timestep']:
    #         raise RuntimeError("No metrics recorded yet!")
    #     with open(filename, "w", newline="") as f:
    #         writer = csv.writer(f)
    #         writer.writerow([
    #             "timestep",
    #             "t_rel_pct",
    #             "pos_count",
    #             "neg_count",
    #             "primary_count"
    #         ])
    #         for t, rel, p, n, m in zip(
    #             self.metrics['timestep'],
    #             self.metrics['t_rel_pct'],
    #             self.metrics['pos_count'],
    #             self.metrics['neg_count'],
    #             self.metrics['primary_count']
    #         ):
    #             writer.writerow([t, rel, p, n, m])
    #     print(f"Metrics saved to {filename}")

    # def save_event_log_csv(self, filename: str = "buffer_event_log.csv") -> None:
    #     """Write out event log CSV: timestep, t_rel_pct, buffer_used per episode."""
    #     if not self.events:
    #         raise RuntimeError("No events recorded yet!")
    #     with open(filename, "w", newline="") as f:
    #         writer = csv.writer(f)
    #         writer.writerow(["timestep", "t_rel_pct", "buffer_used"])
    #         for t, rel, buf in self.events:
    #             writer.writerow([t, rel, buf])
    #     print(f"Event log saved to {filename}")
