#!/usr/bin/env python3
import os
import pandas as pd

# ─── USER CONFIG ──────────────────────────────────────────────────────────────
LOG_DIR = "./logs/exp_0"
DATA_NAMES = ["eval_success_rate"]       # list all metrics you want
NUM_SEEDS  = 3                           # number of seed subfolders (0,1,2,…)
OUTDIR     = "./output_tables"           # where to save the summary table
# ──────────────────────────────────────────────────────────────────────────────

def ensure_folder(path):
    os.makedirs(path, exist_ok=True)
    print(f"→ ensured folder exists: {path}")

def load_full_run_stats(log_dir, metric, num_seeds):
    """
    Returns three values:
      full_max: highest-ever across all seeds & timesteps
      full_mean: average over all seeds & all timesteps
      full_std:  std deviation of per-seed means over full runs
    """
    per_seed_means = []
    per_seed_maxs = []
    for seed in range(num_seeds):
        p = os.path.join(log_dir, str(seed), "runs", "csv", f"{metric}.csv")
        if not os.path.isfile(p):
            print(f"[WARN] missing full-run file for seed {seed}: {p}")
            continue
        df = pd.read_csv(p)
        # rename columns
        df = df.rename(columns={df.columns[0]: "timestep", df.columns[1]: metric})
        per_seed_maxs.append(df[metric].max())
        per_seed_means.append(df[metric].mean())

    if not per_seed_means:
        raise RuntimeError(f"No data found under {log_dir} for metric '{metric}'")

    full_max  = max(per_seed_maxs)
    full_mean = sum(per_seed_means) / len(per_seed_means)
    full_std  = pd.Series(per_seed_means).std(ddof=0)
    return full_max, full_mean, full_std

def load_final_step_stats(log_dir, metric, num_seeds):
    """
    Returns two values:
      final_mean: mean of the last logged metric across seeds
      final_std:  std deviation of those last values
    """
    finals = []
    for seed in range(num_seeds):
        p = os.path.join(log_dir, str(seed), "runs", "csv", f"{metric}.csv")
        if not os.path.isfile(p):
            print(f"[WARN] missing final-step file for seed {seed}: {p}")
            continue
        df = pd.read_csv(p)
        finals.append(df.iloc[-1, 1])  # second column is metric
    if not finals:
        raise RuntimeError(f"No final values found under {log_dir} for metric '{metric}'")
    final_mean = sum(finals) / len(finals)
    final_std  = pd.Series(finals).std(ddof=0)
    return final_mean, final_std

if __name__ == "__main__":
    ensure_folder(OUTDIR)

    rows = []
    for metric in DATA_NAMES:
        print(f"\n→ Processing '{metric}' …")
        # full-run stats
        full_max, full_mean, full_std = load_full_run_stats(LOG_DIR, metric, NUM_SEEDS)
        print(f"   • full_max  = {full_max:.3f}")
        print(f"   • full_mean = {full_mean:.3f}")
        print(f"   • full_std  = {full_std:.3f}")

        # final-step stats
        final_mean, final_std = load_final_step_stats(LOG_DIR, metric, NUM_SEEDS)
        print(f"   • final_mean = {final_mean:.3f}")
        print(f"   • final_std  = {final_std:.3f}")

        rows.append({
            "metric":              metric,
            "max_value":           full_max,
            "mean_value":          full_mean,
            "standard_deviation":  full_std,
            "final_mean":          final_mean,
            "final_std":           final_std
        })

    # build DataFrame & save
    summary_df = pd.DataFrame(rows)
    csv_out = os.path.join(OUTDIR, "summary_all_stats.csv")
    summary_df.to_csv(csv_out, index=False)
    print(f"\n→ wrote combined CSV → {csv_out}")

    tex_out = os.path.join(OUTDIR, "summary_all_stats.tex")
    with open(tex_out, "w", encoding="utf-8") as f:
        f.write(summary_df.to_latex(index=False, float_format="%.2f"))
    print(f"→ wrote combined LaTeX → {tex_out}")

    # print final table to console
    print("\n=== Complete Summary Table ===")
    print(summary_df.to_string(index=False))
