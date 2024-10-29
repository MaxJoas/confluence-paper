import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.stats import mannwhitneyu
from collections import defaultdict
import os
import sys


def plot_lazy_vs_nonlazy(data_dict, metric1, metric2, method, outpath_base):
    """
    Create a custom grid plot comparing 'lazy' vs. 'non-lazy' labeling for two metrics,
    focusing on a single method (either 'al' or 'rand') at each step.

    Args:
        data_dict: Dictionary with model-dataset keys and lists of result DataFrames.
        metric1: First metric to plot (e.g., 'iou').
        metric2: Second metric to plot (e.g., 'dAbsAbsConf').
        method: The method to focus on ('al' or 'rand').
        output_path: Path to save the grid plot image.
    """
    # Filter data for only lc-internal datasets with lazy and non-lazy versions

    relevant_keys = [k for k in data_dict.keys() if "lc-internal" in k]
    models = sorted(set(k.split("-")[0] for k in relevant_keys))
    palette = sns.color_palette("viridis", 2)
    sns.set_style("darkgrid")
    plt.rcParams.update({"font.size": 22, "font.weight": "bold"})
    METRIC_FRIENDLY_NAME = {"iou": "Mean IoU", "dAbsAbsConf": "Mean Δ Confluence"}

    EXPERIMENT_FRIENDLY_NAME = {
        "unet": "U-Net",
        "cp": "Cellpose",
        "d2": "Detectron2",
        "sam": "Segment Anything",
    }
    OUTPATH_FILE = os.path.join(outpath_base, f"02_figure_2_method_{method}.png")
    P_VALUE_DICT = defaultdict(list)

    # Set up the custom grid layout
    _, axes = plt.subplots(len(models), 2, figsize=(24, 5 * len(models)))

    for idx, model in enumerate(models):
        # Retrieve lazy and non-lazy dataframes for the model

        nonlazy_key = f"{model}-lc-internal-{method}"
        lazy_key = f"{model}-lc-internallazy-{method}"

        if nonlazy_key not in data_dict or lazy_key not in data_dict:
            continue

        # Concatenate dataframes for replicates and filter by method
        nonlazy_df = pd.concat(
            [df[df["method"] == method] for df in data_dict[nonlazy_key]],
            ignore_index=True,
        )
        lazy_df = pd.concat(
            [df[df["method"] == method] for df in data_dict[lazy_key]],
            ignore_index=True,
        )
        steps = sorted(nonlazy_df["step"].unique())

        for metric, ax in zip([metric1, metric2], axes[idx]):
            lazy_means, lazy_errs = [], []
            nonlazy_means, nonlazy_errs = [], []

            for k, step in enumerate(steps):
                # Filter data by step
                offset = k * 0.1
                lazy_step = lazy_df[lazy_df["step"] == step][metric]
                nonlazy_step = nonlazy_df[nonlazy_df["step"] == step][metric]

                # Calculate means and absolute error bars
                lazy_means.append(lazy_step.mean())
                lazy_q1, lazy_q3 = np.percentile(lazy_step, [25, 75])
                lazy_errs.append([lazy_means[-1] - lazy_q1, lazy_q3 - lazy_means[-1]])

                nonlazy_means.append(nonlazy_step.mean())
                nonlazy_q1, nonlazy_q3 = np.percentile(nonlazy_step, [25, 75])
                nonlazy_errs.append(
                    [nonlazy_means[-1] - nonlazy_q1, nonlazy_q3 - nonlazy_means[-1]]
                )

                # Ensure error bars are non-negative
                lazy_errs[-1] = [abs(e) for e in lazy_errs[-1]]
                nonlazy_errs[-1] = [abs(e) for e in nonlazy_errs[-1]]

                _, p_value = mannwhitneyu(
                    lazy_step, nonlazy_step, alternative="two-sided"
                )
                P_VALUE_DICT[model].append(p_value)
                sig_marker = "*" if p_value < 0.05 else ""

                ax.errorbar(
                    step,
                    lazy_means[-1],
                    yerr=[[lazy_errs[-1][0]], [lazy_errs[-1][1]]],
                    fmt="o",
                    capsize=6,
                    markersize=9,
                    linewidth=5,
                    color=palette[0],
                    label="Lazy" if step == steps[0] else "",
                )
                ax.errorbar(
                    step + offset,
                    nonlazy_means[-1],
                    yerr=[[nonlazy_errs[-1][0]], [nonlazy_errs[-1][1]]],
                    fmt="x",
                    capsize=6,
                    markersize=9,
                    linewidth=5,
                    color=palette[1],
                    label="Non-Lazy" if step == steps[0] else "",
                )
                offset = 0.0

                if sig_marker:
                    ax.draw_artist(
                        ax.patch
                    )  # Ensures limits are updated before getting ylim
                    (
                        y_min,
                        y_max,
                    ) = ax.get_ylim()  # Get the current y-axis limits for this subplot

                    marker_y = y_max - (y_max - y_min) * 0.1  # 10% below the top limit

                    ax.text(
                        step,
                        marker_y,
                        sig_marker,
                        ha="center",
                        color="black",
                        fontsize=20,
                        weight="bold",
                    )
            ax.set_title(
                f"{EXPERIMENT_FRIENDLY_NAME[model]} - {METRIC_FRIENDLY_NAME[metric]}",
                fontsize=20,
                fontweight="bold",
            )
            # ax.set_title(f"{model} - {METRIC_FRIENDLY_NAME[metric]}", fontsize=20, fontweight="bold")
            ax.set_xlabel("Step", fontweight="bold", fontsize=20)
            ax.set_ylabel(
                f"{METRIC_FRIENDLY_NAME[metric]}", fontweight="bold", fontsize=20
            )

            ax.tick_params(axis="both", which="major", labelsize=16)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight("bold")
                ax.legend(loc="best", fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.savefig(OUTPATH_FILE)
    plt.close()
    p_values_df = pd.DataFrame.from_dict(P_VALUE_DICT, orient="index").transpose()
    p_values_df.to_csv(os.path.join(outpath_base, "02_figure_2_p_values.csv"))





if __name__ == "__main__":
    datapath = sys.argv[1]
    outpath_base = sys.argv[2]
    # data_dicts = pd.read_pickle("/home/majo158f/development/confluence-results/final-results/R1/AGGREGATED_RESULTS/all_dicts.pkl")
    # outpath_base = "/data/horse/ws/majo158f-work_systems/results/final-results/R1/AGGREGATED_RESULTS/Paper"
    data_dicts = pd.read_pickle(datapath)
    new_data_dict = {key.split('/')[-1]: value for key, value in data_dicts.items()}

    sorted_data_dict = dict(sorted(new_data_dict.items()))
    with open(os.path.join(outpath_base, "02_figure_2_data_dict.pkl"), "wb") as f:
        pickle.dump(sorted_data_dict, f)
    # save the data dict to a file
    if not os.path.exists(
        os.path.join(outpath_base, "02_figure_2_data_human_readable")
    ):
        os.makedirs(os.path.join(outpath_base, "02_figure_2_data_human_readable"))
    save_dict = {
        key: [
            df.to_csv(
                os.path.join(
                    outpath_base, "02_figure_2_data_human_readable", f"df_{key}_{i}"
                )
            )
            for i, df in enumerate(value)
        ]
        for key, value in sorted_data_dict.items()
    }
 
    plot_lazy_vs_nonlazy(
        sorted_data_dict,
        "iou",
        "dAbsAbsConf",
        "al",
        outpath_base,
    )
    plot_lazy_vs_nonlazy(
        sorted_data_dict,
        "iou",
        "dAbsAbsConf",
        "rand",
        outpath_base
    )
