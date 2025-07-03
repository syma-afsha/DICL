"""Logger for tracking training metrics with TensorBoard and CSV logging."""
from typing import Dict, Any, Tuple
from collections import defaultdict
import os
import pandas as pd
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator



class Logger:
    """Logger for tracking training metrics with TensorBoard and CSV logging."""
    def __init__(self, config: Dict) -> None:
        """Initialize the logger.
        Args:
            config (Dict): Configuration dictionary.
        """
        self.config = config
        self.log_dir = config["general"]["logdir"]
        self.seed = config["general"]["seed"]
        self.exp_name = config["general"]["exp_name"]
        self.exp_folder = os.path.join(self.log_dir, self.exp_name)
        os.makedirs(self.exp_folder, exist_ok=True)
        self.seed_id = str(len(os.listdir(self.exp_folder)))
        self.log_folder = os.path.join(self.exp_folder, self.seed_id)
        os.makedirs(self.log_folder, exist_ok=True)
        self.tb_logdir = os.path.join(self.log_folder, "runs")
        os.makedirs(self.tb_logdir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.tb_logdir)

        # Backup model
        self.backup_folder = os.path.join(self.log_folder, "model_backup")
        os.makedirs(self.backup_folder, exist_ok=True)

        # Save config
        self.config_path = os.path.join(self.log_folder, "run_config.yaml")
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)
        

    def backup_model_save_path(self,model_name: str) -> str:
        """Return the path to save the model."""
        return os.path.join(self.backup_folder,(model_name))
    

    # def load_config(self, config_path: str) -> Dict[str, Any]:
    #     """Load YAML configuration file."""
    #     with open(self.config_path, "r", encoding="utf-8") as f:
    #         return yaml.safe_load(f)
        
    def tb_writer_and_scalar(self, name: str, value: float, timestep: int) -> None:
        """Write scalar to tensorboard"""
        self.writer.add_scalar(name, value, timestep)

    def tb_tabulate_events(self, dpath: str):
        """Tabulate events from TensorBoard logs."""
        summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in os.listdir(dpath)]
        
        # Filter out logs that have no scalar metrics
        valid_iterators = [it for it in summary_iterators if it.Tags().get('scalars', [])]

        if not valid_iterators:
            return {}, []

        # Get common scalar tags across all valid logs
        common_tags = set(valid_iterators[0].Tags().get('scalars', []))
        for it in valid_iterators[1:]:
            common_tags.intersection_update(it.Tags().get('scalars', []))

        if not common_tags:
            return {}, []

        out = defaultdict(list)
        steps = []

        for tag in common_tags:
            steps = [e.step for e in valid_iterators[0].Scalars(tag)]
            for events in zip(*[acc.Scalars(tag) for acc in valid_iterators]):
                assert len(set(e.step for e in events)) == 1
                out[tag].append([e.value for e in events])

        return out, steps




    def tb_to_csv(self, dpath: str) -> None:
        """Convert TensorBoard logs to CSV."""
        # Get the TensorBoard data and steps
        d, steps = self.tb_tabulate_events(dpath)
        
        # Ensure steps is a sized object by converting it to a list.
        try:
            steps = list(steps)
        except TypeError:
            # print("Error: 'steps' is not convertible to a list. Type of steps:", type(steps))
            return

        if not d:
            return

        try:
            tags, values = zip(*d.items())  # Unpack dictionary safely
        except ValueError:
            return

        for index, tag in enumerate(tags):
            tag_short_name = tag.split("/")[-1]  # Use only the last part of the tag

            # Process the values: if an element is a list, take its mean; else use the value.
            # Wrap the result with np.array and then np.atleast_1d to ensure it's a 1D array.
            tag_values = np.atleast_1d(np.array([
                np.mean(v) if isinstance(v, list) else v 
                for v in values[index]
            ]))
            
            # Squeeze any extra dimensions and flatten if necessary.
            tag_values = np.squeeze(tag_values)
            if tag_values.ndim > 1:
                tag_values = tag_values.flatten()

            # # Debug prints to inspect types and shapes.
            # print(f"Processing tag: {tag}")
            # print("Type of steps:", type(steps), "Length of steps:", len(steps))
            # print("Type of tag_values:", type(tag_values), "Shape of tag_values:", tag_values.shape)

            try:
                formatted_tag_values = [f"{val:.2f}" for val in tag_values]
                min_length = min(len(steps), len(formatted_tag_values))

                # min_length = min(len(steps), len(tag_values))
            except TypeError as e:
                print("Error computing lengths:", e)
                continue

            if len(steps) != len(tag_values):
                print(f"Length mismatch for tag {tag}: using only first {min_length} entries.")
            
            steps_trim = steps[:min_length]
            tag_values_trim = tag_values[:min_length]

            # Create a DataFrame using the trimmed arrays.
            df = pd.DataFrame({
                "Timestep": steps_trim,
                tag_short_name: tag_values_trim
            })

            # Save the DataFrame to a CSV file.
            file_path = self.tb_get_file_path(dpath, tag_short_name)
            df.to_csv(file_path, index=False)




    def tb_get_file_path(self, dpath: str, tag) -> str:
        """Get the file path for the csv file"""
        file_name = tag.replace("/", "_") + '.csv'
        folder_path = os.path.join(dpath, 'csv')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return os.path.join(folder_path, file_name)
    

    def tb_close(self) -> None:
        """Close the tensorboard writer"""
        self.writer.flush()
        self.writer.close()
        # self.tb_to_csv(self.tb_logdir)

 

    def get_best_seed(self) -> int:
        """Get the best seed"""
        best_value = float("-inf")
        best_seed = 0
        seeds=sorted(os.listdir(self.exp_folder))
        for i in seeds:
            path=os.path.join(self.exp_folder,str(i),"runs","csv","EvalSuccessRate.csv")
            df=pd.read_csv(path)
            max_value=df.iloc[:,1].max()
            if max_value>best_value:
                best_value=max_value
                best_seed=int(i)
        print("BEST: [" + str(best_seed) + "] : " + str(best_value))
        return best_seed
        
    



        

        



    

    
    


        
