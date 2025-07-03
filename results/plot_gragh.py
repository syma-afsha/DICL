import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import numpy as np

# OpenAI-style boxcar smoothing
def simple_moving_avg(x: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    padded = np.concatenate([
        np.full(half, x[0]),
        x,
        np.full(half, x[-1])
    ])
    kernel = np.ones(window) / window
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed[:len(x)]

def load_data_for_seaborn(log_dir: str, exp_name: str, plotdata: str, num_seeds: int) -> pd.DataFrame:
    all_data = []
    for seed in range(num_seeds):
        path = os.path.join(log_dir, exp_name, str(seed), "runs", "csv", f"{plotdata}.csv")
        if not os.path.exists(path):
            print(f"File not found for seed {seed}: {path}")
            continue
        df = pd.read_csv(path)
        df[df.columns[1]] = -df[df.columns[1]] # Ensure the data is positive
        df = df.rename(columns={df.columns[0]: "timestep", df.columns[1]: plotdata})
        df["seed"] = seed
        df["method"] = exp_name
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True) if all_data else None

def plot_sns_smooth(df: pd.DataFrame, plotdata: str, title: str, outpath: str, window: int = 100):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 8))
    df["timestep_k"] = df["timestep"] / 1000

    method_legend_map = {
        # "FetchSlide_IGC+HER": "Incremental_Graded_Curriculum+HER",
        # "FetchSlide_IGC+HER+PER": "Incremental_Graded_Curriculum+HER+PER",
        # "FetchSlide_IGC+PER": "Incremental_Graded_Curriculum+PER",
        # "FetchSlide_baseline": "Baseline",
        # "FetchSlide_baseline+HER+PER": "Baseline+HER+PER",
        # "FetchSlide_baseline_PER": "Baseline+PER",
        # "FetchSlide_baseline_HER": "Baseline+HER",
        # "FetchPickAndPlace-v4_Incremental_with_Per_with_HER": "PER+HER+IGC",
        # "FetchPickAndPlace-v4_Incremental_with_Per_without_HER": "PER+IGC",
        # "FetchPickAndPlace-v4_Incremental_without_Per_with_HER": "HER+IGC",
        # "Baseline+HER+PER": "Baseline+HER+PER",
        # "Baseline+HER_":"Baseline+HER",
        # "Pick_And_Place_Baseline": "Baseline",
        # "Fetch_PickAndPlace_new_linear":"Linear",
        # "Fetch_PickAndPlace_test_new":"Exponential"
        # "Baseline+Penalty":"Baseline",
        # "Baseline+Penalty+PER":"Baseline+PER",
        # "Baseline+Penalty+HER_Penalty":"Baseline+HER",  
        # "Baseline+Penalty+HER+PER":"Baseline+HER+PER",
        # "Dual_Buffer_PickAndPlace":"Dual_Buffer+HER",
        # "FetchPickAndPlace_Dual_Buffer+without_HER":"Dual_Buffer"
        # # "FetchPush_Dual_Buffer+without_HER":"Dual_Buffer"
        
        

        
    }
    df["method"] = df["method"].map(method_legend_map)

    df = df.sort_values(['method','seed','timestep'])
    df[f'{plotdata}_smooth'] = (
        df.groupby(['method','seed'])[plotdata]
          .transform(lambda x: simple_moving_avg(x.values, window))
    )

    palette = {
    #    "PER+IGC": "#1f77b4",
    # "PER+HER+IGC":"#2ca02c",
    # "HER+IGC": "#d62728",
    # "Baseline+HER": "orange",
    # "Baseline+HER+PER": "blue",
    # "Baseline": "gray",
    # "Linear": "#1f77b4",
    # "Exponential": "#2ca02c",
    #     "Baseline":"gray",
    #     "Baseline+PER": "#1f77b4",
    #    "Baseline+HER": "#2ca02c",  
    #    "Baseline+HER+PER": "#d62728",
    #    "Dual_Buffer+HER": "magenta",
    #    "Dual_Buffer":"orange"
    "Fully Adaptive" : "magenta",
    "Self-Paced Adaptive":"blue",
    "Predefined":"green"
  
    }

    ax = sns.lineplot(
        data=df,
        x="timestep_k",
        y=f'{plotdata}_smooth',
        hue="method",
        errorbar="sd",
        palette=palette,
        linewidth=1.5
    )

    plt.xlabel("Timesteps (×10³)", fontsize=12)
    plt.ylabel("Eval_episode_len", fontsize=12)
    plt.title(title, fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.9)
    plt.ylim(0, 1.01)
    plt.xlim(0, 500)

    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=20,
        frameon=False
    )

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"→ Plot saved to: {outpath}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logs/", help="Base folder where logs are stored")
    parser.add_argument("--outdir", default="result_output_data/output/plots", help="Folder to save the plots")
    parser.add_argument("--exp_name", default="", help="Unused (multi-experiment handled in list)")
    parser.add_argument("--data_name", default="eval_success_rate", help="Comma-separated list of CSV column names")
    parser.add_argument("--num_seeds", type=int, default=3, help="Number of seeds per experiment")
    args = parser.parse_args()

    exp_names = [
        #  "FetchPickAndPlace-v4_Incremental_with_Per_with_HER",
        # "FetchPickAndPlace-v4_Incremental_with_Per_without_HER",
        # "FetchPickAndPlace-v4_Incremental_without_Per_with_HER",
        # "Baseline+HER+PER",
        # "Baseline+HER_",
        # "Pick_And_Place_Baseline",
        # "Baseline+Penalty",
        # "Baseline+Penalty+PER",
        # "Baseline+Penalty+HER_Penalty",  
        # "Baseline+Penalty+HER+PER",
        # "FetchPickAndPlace_Dual_Buffer+without_HER"
        # "FetchPush_Dual_Buffer+without_HER"

    ]

    data_names = [name.strip() for name in args.data_name.split(",")]

    for plotdata in data_names:
        all_df = []
        for exp in exp_names:
            df = load_data_for_seaborn(args.logdir, exp, plotdata, args.num_seeds)
            if df is not None:
                all_df.append(df)

        if all_df:
            final_df = pd.concat(all_df, ignore_index=True)
            plot_filename = os.path.join("3variantsFetchPickAndPlace.png")
            plot_sns_smooth(
                final_df,
                plotdata=plotdata,
                title="FetchPickAndPlace",
                outpath=plot_filename,
                window=80  # You can change this to control smoothing
            )
