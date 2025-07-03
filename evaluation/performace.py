import numpy as np
import matplotlib.pyplot as plt

def record_evaluation_data(evaluation_data, current_timestep, success_rate):
    """
    Store the timestep and success rate in a list or array.
    
    Args:
        evaluation_data (list): A list of tuples (timesteps, success_rate).
        current_timestep (int): Current training timestep.
        success_rate (float): Evaluation success rate at this timestep.
    """
    evaluation_data.append((current_timestep, success_rate))

def plot_evaluation_success_rate(evaluation_data_list, labels):
    """
    Plot the evaluation success rate vs. timesteps for multiple runs or algorithms.
    
    Args:
        evaluation_data_list (list of lists): Each element is a list of (timestep, success_rate) pairs.
        labels (list): Labels for each run/algorithm. Optional.
    """
    plt.figure(figsize=(8,5))
    
   
    labels = [f"Run {i+1}" for i in range(len(evaluation_data_list))]
    
    for i, evaluation_data in enumerate(evaluation_data_list):
        # Sort by timestep 
        evaluation_data = sorted(evaluation_data, key=lambda x: x[0])
        
        # Separate timesteps and success rates
        timesteps = [d[0] for d in evaluation_data]
        success_rates = [d[1] for d in evaluation_data]
        
        plt.plot(timesteps, success_rates, label=labels[i])
    
    plt.xlabel("Timesteps")
    plt.ylabel("Evaluation Success Rate")
    plt.title("Evaluation Success Rate vs. Timesteps")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

