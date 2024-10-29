import os
import pandas as pd
import sys
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import seaborn as sns
import pickle
from collections import defaultdict


def plot_metric_grid(data_dict, metric, outpath_base, share_y=True):
    """
    Create a grid of plots for the specified metric, comparing AL vs. random
    methods across all model-dataset combinations in data_dict.

    Args:
        data_dict: Dictionary with model-dataset keys and lists of result DataFrames.
        metric: The metric to plot (e.g., 'iou', 'accuracy').
        outpath: Path to save the grid plot image.
    """
    # Define consistent colors using viridis colormap

    palette = sns.color_palette("viridis", 2)
    sns.set_style("darkgrid")
    plt.rcParams.update({"font.size": 22, "font.weight": "bold"})
    METHOD_COLORS = {"al": palette[1], "rand": palette[0]}

    # Define consistent markers for methods

    METHOD_MARKERS = {"al": "o", "rand": "x"}
    METRIC_FRIENDLY_NAME = {"iou": "Mean IoU", "dAbsAbsConf": "Mean Δ Confluence"}

    EXPERIMENT_FRIENDLY_NAME = {
        "unet-lc-internal-al": "U-Net LC Internal",
        "d2-lc-internal-al": "D2 LC Internal",
        "sam-lc-internal-al": "SAM LC Internal",
        "cp-lc-internal-al": "CP LC Internal",
        "unet-lc-external-al": "U-Net LC External",
        "d2-lc-external-al": "D2 LC External",
        "sam-lc-external-al": "SAM LC External",
        "cp-lc-external-al": "CP LC External",
        "unet-sc-al": "U-Net SC",
        "d2-sc-al": "D2 SC",
        "sam-sc-al": "SAM SC",
        "cp-sc-al": "CP SC",
        "unet-lc-internallazy-al": "U-Net LC Internal Lazy",
        "d2-lc-internallazy-al": "D2 LC Internal Lazy",
        "sam-lc-internallazy-al": "SAM LC Internal Lazy",
        "cp-lc-internallazy-al": "CP LC Internal Lazy",
    }
    OUTPATH_FILE = os.path.join(outpath_base, f"01_figure_1_{metric}_sharedy_{share_y}")
    P_VALUE_DICT = defaultdict(list)

    model_dict = {}
    for key, df_list in data_dict.items():
        model_name = key.split("-")[0]
        if model_name not in model_dict:
            model_dict[model_name] = {}
        model_dict[model_name][key] = df_list
    num_models = len(model_dict)
    num_datasets = max(len(datasets) for datasets in model_dict.values())

    fig, axes = plt.subplots(
        num_models, num_datasets, figsize=(40, num_models * 7), constrained_layout=True
    )

    if num_models == 1:
        axes = axes.reshape(1, -1)

    # First pass: collect all y values and determine y-limits for each row
    row_ylims = {}
    for i, (model, datasets) in enumerate(model_dict.items()):
        row_values = []
        for df_list in datasets.values():
            combined_df = pd.concat(df_list, ignore_index=True)
            row_values.extend(combined_df[metric].values)
        # Calculate row-specific y-limits with padding

        y_min = min(row_values)
        y_max = max(row_values)
        y_range = y_max - y_min
        padding = y_range * 0.08  # 15% padding for markers
        row_ylims[i] = (y_min - padding * 0.5, y_max + padding)
    # Second pass: create plots

    for i, (model, datasets) in enumerate(model_dict.items()):
        # Get y-limits for this row
        row_values = []
        for df_list in datasets.values():
            combined_df = pd.concat(df_list, ignore_index=True)
            row_values.extend(combined_df[metric].values)
        y_min = min(row_values)
        y_max = max(row_values)

        for j, (key, df_list) in enumerate(datasets.items()):
            combined_df = pd.concat(df_list, ignore_index=True)
            steps = sorted(combined_df["step"].unique())
            ax = axes[i, j]
            if share_y:
                axes[i, j].set_ylim(y_min, y_max)
                ax = axes[i, j]
                ax.set_ylim(row_ylims[i])
            for step in steps:
                step_data = combined_df[combined_df["step"] == step]

                for k, method in enumerate(["al", "rand"]):
                    offset = k * 0.1
                    method_data = step_data[step_data["method"] == method]

                    mean = method_data[metric].mean()
                    std = method_data[metric].std()
                    q25 = np.percentile(method_data[metric], 25)
                    q75 = np.percentile(method_data[metric], 75)

                    # Calculate absolute distances from mean for lower and upper errors
                    lower_err = abs(mean - q25)  # Distance from mean to 25th percentile
                    upper_err = abs(q75 - mean)  # Distance from mean to 75th percentile

                    label = method.upper() if step == steps[0] else ""
                    ax.errorbar(
                        step + offset,
                        mean,
                        yerr=[[lower_err], [upper_err]],
                        fmt=METHOD_MARKERS[method],
                        color=METHOD_COLORS[method],
                        linewidth=5,
                        markersize=10,
                        capsize=5,
                        alpha=0.8,
                        label=label,
                    )
                    offset = 0.0
                # Statistical significance test

                al_values = step_data[step_data["method"] == "al"][metric].values
                rand_values = step_data[step_data["method"] == "rand"][metric].values

                if len(al_values) > 0 and len(rand_values) > 0:
                    _, p_value = stats.mannwhitneyu(
                        al_values, rand_values, alternative="two-sided"
                    )
                    P_VALUE_DICT[key].append(p_value)

                    if p_value < 0.05:
                        # Plot the data first and let matplotlib set y-limits
                        ax.draw_artist(
                            ax.patch
                        )  # Ensures limits are updated before getting ylim
                        (
                            y_min,
                            y_max,
                        ) = (
                            ax.get_ylim()
                        )  # Get the current y-axis limits for this subplot

                        # Set marker_y slightly below the y-axis max

                        marker_y = (
                            y_max - (y_max - y_min) * 0.02
                        )  # 2% below the top limit

                        # Place the significance marker just below the y-axis max

                        ax.text(
                            step,
                            marker_y,
                            "*",
                            ha="center",
                            va="top",
                            color="black",
                            fontsize=25,
                            weight="bold",
                        )

                ax.set_title(EXPERIMENT_FRIENDLY_NAME[key], fontweight="bold")
                ax.set_ylabel(
                    METRIC_FRIENDLY_NAME[metric], fontweight="bold"
                )  # Y-axis label for every subplot
                ax.set_xlabel("Step", fontweight="bold")

                # Show legend in every subplot

                ax.legend(title="Method", loc="upper right", fontsize=15)
    # Hide any unused subplots

    for j in range(len(datasets), num_datasets):
        axes[i, j].set_visible(False)
    # Save the plot if outpath is provided

    if outpath_base:
        plt.savefig(OUTPATH_FILE, bbox_inches="tight", dpi=500)
    plt.show()
    plt.close()
    print(f"Saved plot to {OUTPATH_FILE}")
    # p_values to dataframe
    p_values_df = pd.DataFrame.from_dict(P_VALUE_DICT, orient="index").transpose()
    p_values_df.to_csv(OUTPATH_FILE.replace("png", "") + "_p_values.csv")

    return fig, axes


if __name__ == "__main__":
    # data_dicts = pd.read_pickle("/home/majo158f/development/confluence-results/final-results/R1/AGGREGATED_RESULTS/all_dicts.pkl")

    data_file = sys.argv[1]
    data_dicts = pd.read_pickle(data_file)
    # outpath_base = "/data/horse/ws/majo158f-work_systems/results/final-results/R1/AGGREGATED_RESULTS"

    outpath_base = sys.argv[2]
    # Create a new dictionary with modified keys

    new_data_dict = {key.split("/")[-1]: value for key, value in data_dicts.items()}

    # Sort the new dictionary by keys

    sorted_data_dict = dict(sorted(new_data_dict.items()))
    with open(os.path.join(outpath_base, "01_figure_1_data_dict.pkl"), "wb") as f:
        pickle.dump(sorted_data_dict, f)
    # structure of dict {"key": [df1, df2, df3, ...], "key2": [df1, df2, df3, ...], ...}
    # how to save this as csv?
    if not os.path.exists(
        os.path.join(outpath_base, "01_figure_1_data_human_readable")
    ):
        os.makedirs(os.path.join(outpath_base, "01_figure_1_data_human_readable"))
    save_dict = {
        key: [
            df.to_csv(
                os.path.join(
                    outpath_base, "01_figure_1_data_human_readable", f"df_{key}_{i}"
                )
            )
            for i, df in enumerate(value)
        ]
        for key, value in sorted_data_dict.items()
    }
    for to_share in [True, False]:
        for metric in ["iou", "dAbsAbsConf"]:

            plot_metric_grid(
                data_dict=sorted_data_dict,
                metric=metric,
                outpath_base=outpath_base,
                share_y=to_share,
            )
