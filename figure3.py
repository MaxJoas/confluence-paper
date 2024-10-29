import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from collections import defaultdict
import sys


def plot_movie_vs_pos_2x2_grid(dataframes, titles, output_path):
    # Set style and font size
    sns.set_style("darkgrid")  # Apply darkgrid style for the entire figure
    plt.rcParams.update({"font.size": 22, "font.weight": "bold"})

    pvalue_dict = defaultdict(list)
    fig, axes = plt.subplots(2, 2, figsize=(24, 18))
    palette = sns.color_palette("viridis", 2)

    for idx, (df, title) in enumerate(zip(dataframes, titles)):
        ax = axes[idx // 2, idx % 2]

        # Prepare data
        al_data = df[df["Method"] == "al"].groupby("Step")["Value"].apply(list)
        rand_data = df[df["Method"] == "rand"].groupby("Step")["Value"].apply(list)
        steps = sorted(df["Step"].unique())

        offsets = np.linspace(-0.1, 0.1, 2)

        for step in steps:
            al_values = al_data[step]
            rand_values = rand_data[step]

            # Calculate percentiles for AL method
            al_median = np.median(al_values)
            al_25th = np.percentile(al_values, 25)
            al_75th = np.percentile(al_values, 75)
            al_error_lower = abs(al_median - al_25th)
            al_error_upper = abs(al_75th - al_median)

            # Calculate percentiles for Rand method
            rand_median = np.median(rand_values)
            rand_25th = np.percentile(rand_values, 25)
            rand_75th = np.percentile(rand_values, 75)
            rand_error_lower = abs(rand_median - rand_25th)
            rand_error_upper = abs(rand_75th - rand_median)

            # Plot with asymmetric error bars
            ax.errorbar(
                step + offsets[0],
                al_median,
                yerr=[[al_error_lower], [al_error_upper]],  # Asymmetric error bars
                fmt="o",
                color=palette[1],
                capsize=5,
                alpha=0.9,
                label="AL" if step == steps[0] else "",
                linewidth=3,
                markersize=6,
            )
            ax.errorbar(
                step + offsets[1],
                rand_median,
                yerr=[[rand_error_lower], [rand_error_upper]],  # Asymmetric error bars
                fmt="x",
                color=palette[0],
                capsize=5,
                alpha=0.9,
                label="Rand" if step == steps[0] else "",
                linewidth=4,
                markersize=8,
            )
            # Statistical test and marker placement
            _, p_value = mannwhitneyu(al_values, rand_values, alternative="two-sided")
            pvalue_dict[title].append(p_value)
            if p_value < 0.05:
                y_min, y_max = ax.get_ylim()
                marker_y = y_max - (y_max - y_min) * 0.1  # 10% below the top limit
                ax.text(
                    step,
                    marker_y,
                    "*",
                    ha="center",
                    va="bottom",
                    color="black",
                    fontsize=22,
                    weight="bold",
                )

        # Set titles, labels, and bold tick labels
        ax.set_title(title, fontweight="bold", fontsize=22)
        ax.set_xlabel("Step", fontweight="bold", fontsize=22)
        ax.set_ylabel("Mean Δ Position", fontweight="bold", fontsize=22)
        ax.legend(loc="upper left", fontsize=22)

        # Make tick labels bold
        ax.tick_params(axis="both", which="major", labelsize=16)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

    p_values_df = pd.DataFrame.from_dict(pvalue_dict, orient="index").transpose()
    p_values_df.to_csv(os.path.join(output_path, "03_figure_3_p_values.csv"))

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "03_figure_3_movie.png"))
    plt.close()


if __name__ == "__main__":

    base_path = sys.argv[1]
    relevant_dirs = ["cp-sc", "d2-sc", "sam-sc", "unet-sc"]
    dataframes = []
    for folder in relevant_dirs:
        path = os.path.join(base_path, folder, "999_DATASOURCE2_movie_plot_data.csv")
        if not os.path.exists(path):
            print(f"Path {path} does not exist")
            raise FileNotFoundError("Path does not exist")
        df = pd.read_csv(path)
        dataframes.append(df)

    titles = ["Cellpose", "Detectron2", "SAM", "U-Net"]
    output_path = os.path.join(base_path, "PAPER")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plot_movie_vs_pos_2x2_grid(dataframes, titles, output_path)
    # save as pickle
    with open(os.path.join(output_path, "03_figure_3_data.pkl"), "wb") as f:
        pickle.dump(dataframes, f)

    if not os.path.exists(os.path.join(output_path, "03_figure_3_data_human_readable")):
        os.makedirs(os.path.join(output_path, "03_figure_3_data_human_readable"))
    for idx, df in enumerate(dataframes):
        df.to_csv(
            os.path.join(
                output_path, "03_figure_3_data_human_readable", f"df_{titles[idx]}"
            )
        )
    print("Saved dataframes to disk")
