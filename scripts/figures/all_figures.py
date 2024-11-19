import os
import matplotlib.colors as mcolors
import pandas as pd
import scipy.stats as stats
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import seaborn as sns
import argparse
import pickle
from collections import defaultdict
import matplotlib.cm as cm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process data from different sources.")

    # Create a mutually exclusive argument group
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--from-aggregation",
        action="store_true",
        help="Load data from aggregation source",
    )
    source_group.add_argument(
        "--from-figure-data",
        action="store_true",
        help="Load data from figure data source",
    )
    figure_group = parser.add_mutually_exclusive_group()
    figure_group.add_argument(
        "--figure-1",
        action="store_true",
        help="Create figure 1",
    )
    figure_group.add_argument(
        "--figure-2",
        action="store_true",
        help="Create figure 2",
    )
    figure_group.add_argument(
        "--figure-3",
        action="store_true",
        help="Create figure 3",
    )
    figure_group.add_argument(
        "--figure-4",
        action="store_true",
        help="Create figure 4",
    ),
    figure_group.add_argument(
        "--all-figures",
        action="store_true",
        help="Create all figures",
    )

    return parser.parse_args()


# FIGURE 1 ----------------------------------------------
def figure_1(data_dict, metric, outpath, share_y=True):
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

    # always 1 except for external datasets there it is 10
    STEP_SIZE_DICT = {
        "unet-lc-internal-al": 1,
        "d2-lc-internal-al": 1,
        "sam-lc-internal-al": 1,
        "cp-lc-internal-al": 1,
        "unet-lc-external-al": 10,
        "d2-lc-external-al": 10,
        "sam-lc-external-al": 10,
        "cp-lc-external-al": 10,
        "unet-sc-al": 1,
        "d2-sc-al": 1,
        "sam-sc-al": 1,
        "cp-sc-al": 1,
        "unet-lc-internallazy-al": 1,
        "d2-lc-internallazy-al": 1,
        "sam-lc-internallazy-al": 1,
        "cp-lc-internallazy-al": 1,
    }
    INITIAL_DATSET_SIZE_DICT = {
        "unet-lc-internal-al": 2,
        "d2-lc-internal-al": 2,
        "sam-lc-internal-al": 2,
        "cp-lc-internal-al": 2,
        "unet-lc-external-al": 10,
        "d2-lc-external-al": 10,
        "sam-lc-external-al": 10,
        "cp-lc-external-al": 10,
        "unet-sc-al": 2,
        "d2-sc-al": 2,
        "sam-sc-al": 2,
        "cp-sc-al": 2,
        "unet-lc-internallazy-al": 2,
        "d2-lc-internallazy-al": 2,
        "sam-lc-internallazy-al": 2,
        "cp-lc-internallazy-al": 2,
    }

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

    # First pass: collect ALL values to determine global y-limits
    all_values = []
    for model, datasets in model_dict.items():
        for df_list in datasets.values():
            combined_df = pd.concat(df_list, ignore_index=True)
            all_values.extend(combined_df[metric].values)

    # Calculate global y-limits with padding
    y_min = min(all_values)
    y_max = max(all_values)
    y_range = y_max - y_min
    padding = y_range * 0.08  # 8% padding for markers
    global_ylims = (y_min - padding * 0.5, y_max + padding)

    # Second pass: create plots
    for i, (model, datasets) in enumerate(model_dict.items()):
        for j, (key, df_list) in enumerate(datasets.items()):
            combined_df = pd.concat(df_list, ignore_index=True)
            steps = sorted(combined_df["step"].unique())
            ax = axes[i, j]

            # Set the global y-limits for every subplot
            if share_y:
                ax.set_ylim(global_ylims)

            for stepcounter, step in enumerate(steps):
                stepcounter += 1
                step_data = combined_df[combined_df["step"] == step]

                for k, method in enumerate(["al", "rand"]):
                    offset = k * 0.1
                    method_data = step_data[step_data["method"] == method]
                    # n_images = (stepcounter + INITIAL_DATSET_SIZE_DICT[key]) * STEP_SIZE_DICT[key]
                    n_images = step + int(INITIAL_DATSET_SIZE_DICT[key]) + 1

                    mean = method_data[metric].mean()
                    std = method_data[metric].std()
                    q25 = np.percentile(method_data[metric], 25)
                    q75 = np.percentile(method_data[metric], 75)

                    # Calculate absolute distances from mean for lower and upper errors
                    lower_err = abs(mean - q25)  # Distance from mean to 25th percentile
                    upper_err = abs(q75 - mean)  # Distance from mean to 75th percentile

                    label = method.upper() if step == steps[0] else ""
                    ax.errorbar(
                        n_images + offset,
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
                        # Calculate marker_y position based on global y-limits
                        marker_y = (
                            global_ylims[1] - (global_ylims[1] - global_ylims[0]) * 0.02
                        )

                        ax.text(
                            n_images,
                            marker_y,
                            "*",
                            ha="center",
                            va="top",
                            color="black",
                            fontsize=25,
                            weight="bold",
                        )

            ax.set_title(EXPERIMENT_FRIENDLY_NAME[key], fontweight="bold")
            ax.set_ylabel(METRIC_FRIENDLY_NAME[metric], fontweight="bold")
            ax.set_xlabel("N Images", fontweight="bold")
            ax.legend(title="Method", loc="upper right", fontsize=15)

    # Hide any unused subplots
    for j in range(len(datasets), num_datasets):
        axes[i, j].set_visible(False)

    # Save the plot
    if outpath:
        plt.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()
    print(f"Saved plot to {outpath}")

    # Save p-values
    p_values_df = pd.DataFrame.from_dict(P_VALUE_DICT, orient="index").transpose()
    p_values_df.to_csv(outpath.replace("png", "") + "_p_values.csv")

    return fig, axes


# FIGURE 2 ----------------------------------------------
def figure_2(data_dict, metric1, metric2, method, outpath):
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
    plt.savefig(outpath)
    plt.close()
    p_values_df = pd.DataFrame.from_dict(P_VALUE_DICT, orient="index").transpose()
    p_values_df.to_csv(os.path.join(outpath.replace(".png", "_p_values.csv")))


# FIGURE 3 ----------------------------------------------
def figure_3(dataframes, titles, output_path):
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
                # this has a numpy to long data type error convert
                y_min, y_max = ax.get_ylim()
                y_min, y_max = float(y_min), float(y_max)
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
    p_values_df.to_csv(os.path.join(output_path.replace(".png", "_p_values.csv")))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def figure_4(df_all_epochs, df_zeroshot, respath, metric, plot_errorbars=True):
    """
    Create a 1x2 grid of bar plots comparing metrics across two DataFrames.
    Excludes specific models from the second plot by hardcoding each subplot.

    Args:
        res_df1 (pd.DataFrame): The first DataFrame for the first plot.
        res_df2 (pd.DataFrame): The second DataFrame for the second plot.
        respath (str): Path to save the grid plot image.
        metric (str): Metric to plot (e.g., 'iou' or 'absabsDelConf').
        plot_errorbars (bool): Whether to plot error bars.
    """
    sns.set_style("darkgrid")
    plt.rcParams.update({"font.size": 18, "font.weight": "bold"})
    METRIC_FRIENDLY_NAME = {"iou": "Mean IoU", "absabsDelConf": "Mean Δ Confluence"}
    DATA_GROUP_FRIENDLY_NAME = {
        "-sc": "SC",
        "-lc-internal": "LC Internal",
        "-lc-external": "LC External",
        "-lc-internallazy": "LC Internal Lazy",
    }
    MODEL_NAMES_FRIENDLY = {
        "cp": "Cellpose",
        "d2": "Detectron2",
        "sam": "SAM",
        "unet": "U-Net",
        "base": "Baseline",
    }

    data_names = df_all_epochs["data(group)"].unique()
    friendly_data_names = [DATA_GROUP_FRIENDLY_NAME[name] for name in data_names]

    # Set up a 1x2 grid for the two plots

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    norm = mcolors.Normalize(vmin=0, vmax=len(data_names)-1)
    cmap = cm.viridis
    
    # Create group colors dictionary
    group_colors = {
        group: cmap(norm(i)) for i, group in enumerate(data_names)
    }
    # First subplotuu

    barWidth = 0.9
    offset = 0
    model_names_1 = df_all_epochs["model(col)"].unique()

    for i, model_name in enumerate(model_names_1):
        bars = df_all_epochs[df_all_epochs["model(col)"] == model_name][metric]
        yerrs = df_all_epochs[df_all_epochs["model(col)"] == model_name][
            f"{metric}_std"
        ]
        yerrs = yerrs.values.tolist()
        if not plot_errorbars:
            yerrs = [None for _ in range(len(yerrs))]
        groups = df_all_epochs[df_all_epochs["model(col)"] == model_name]["data(group)"]
        r = np.arange(len(bars))
        for j, (bar, group) in enumerate(zip(bars, groups)):
            ax1.bar(
                r[j] + i * 0.5 * barWidth + offset,
                bar,
                color=group_colors[group],
                width=barWidth,
                error_kw=dict(capthick=2, elinewidth=2),
                yerr=yerrs[j] if yerrs[j] is not None else None,
            )
        offset += 8
    ax1.set_xticks([1.0, 9.5, 18.5, 27.0, 35.5])
    ax1.set_xticklabels([MODEL_NAMES_FRIENDLY[model] for model in model_names_1])
    ax1.set_ylabel(METRIC_FRIENDLY_NAME[metric], fontweight="bold", fontsize=18)
    ax1.tick_params(axis="both", which="major", labelsize=16)

    # Second subplot (excluding specific models)

    barWidth = 0.9
    offset = 0
    model_names_2 = [
        name
        for name in df_zeroshot["model(col)"].unique()
        if name not in ["base", "unet"]
    ]

    for i, model_name in enumerate(model_names_2):
        bars = df_zeroshot[df_zeroshot["model(col)"] == model_name][metric]
        yerrs = df_zeroshot[df_zeroshot["model(col)"] == model_name][f"{metric}_std"]
        yerrs = yerrs.values.tolist()
        if not plot_errorbars:
            yerrs = [None for _ in range(len(yerrs))]
        groups = df_zeroshot[df_zeroshot["model(col)"] == model_name]["data(group)"]
        r = np.arange(len(bars))
        for j, (bar, group) in enumerate(zip(bars, groups)):
            ax2.bar(
                r[j] + i * 0.5 * barWidth + offset,
                bar,
                color=group_colors[group],
                width=barWidth,
                error_kw=dict(capthick=2, elinewidth=2),
                yerr=yerrs[j] if yerrs[j] is not None else None,
            )
        offset += 8
    ax2.set_xticks([1.5, 10.0, 18.5])
    ax2.set_xticklabels([MODEL_NAMES_FRIENDLY[model] for model in model_names_2])
    ax2.set_ylabel(METRIC_FRIENDLY_NAME[metric], fontweight="bold", fontsize=18)
    ax2.tick_params(axis="both", which="major", labelsize=16)

    # Create a legend for the data groups

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=group_colors[group]) for group in data_names
    ]
    ax1.legend(
        handles,
        friendly_data_names,
        title="Data Groups",
        fontsize=14,
        title_fontsize=14,
    )

    ax2.legend(
        handles,
        friendly_data_names,
        title="Data Groups",
        fontsize=14,
        title_fontsize=14,
    )

    plt.tight_layout()
    plt.show()
    plt.savefig(
        os.path.join(respath)
    )
    plt.close()



def figure_1_main(sorted_data_dict):

    outpath_dict = {
        "iou_True": os.path.join("output", "supplement", "Figure_S1.png"),
        "iou_False": os.path.join(
            "output", "additional_figures_and_tables", "Figure_S1_no_global_y_axis.png"
        ),
        "dAbsAbsConf_True": os.path.join("output", "main", "Figure_1.png"),
        "dAbsAbsConf_False": os.path.join(
            "output", "additional_figures_and_tables", "Figure_1_no_global_y_axis.png"
        ),
    }
    if not os.path.exists(os.path.join("output", "main")):
        os.makedirs(
            os.path.join(
                "output",
                "main",
            ),
            exist_ok=True,
        )
    if not os.path.exists(os.path.join("output", "additional_figures_and_tables")):
        os.makedirs(
            os.path.join("output", "additional_figures_and_tables"), exist_ok=True
        )
    if not os.path.exists(os.path.join("output", "supplement")):
        os.makedirs(os.path.join("output", "supplement"), exist_ok=True)
    for to_share in [True, False]:
        for metric in ["iou", "dAbsAbsConf"]:
            outpath = outpath_dict[f"{metric}_{to_share}"]
            print(f"Outpath base: {outpath}")

            figure_1(
                data_dict=sorted_data_dict,
                metric=metric,
                outpath=outpath,
                share_y=to_share,
            )


if __name__ == "__main__":
    # make output folders
    titles = ["Cellpose", "Detectron2", "SAM", "U-Net"]
    if not os.path.exists(os.path.join("output", "data", "LEVEL_1")):
        os.makedirs(os.path.join("output", "data", "LEVEL_1"), exist_ok=True)
    if not os.path.exists(os.path.join("output", "main")):
        os.makedirs(os.path.join("output", "main"), exist_ok=True)
    if not os.path.exists(os.path.join("output", "additional_figures_and_tables")):
        os.makedirs(
            os.path.join("output", "additional_figures_and_tables"), exist_ok=True
        )
    if not os.path.exists(os.path.join("output", "supplement")):
        os.makedirs(os.path.join("output", "supplement"), exist_ok=True)
    args = parse_arguments()
    if args.from_figure_data:
        if not os.path.exists(
            os.path.join("data", "LEVEL_1", "01_figure_1", "01_figure_1_data_dict.pkl")
        ):
            raise FileNotFoundError(
                "Data file not found. Please run data processing script first This should be in data/01_figure_1/01_figure_1_data_dict.pkl \
            and come directly with the repository"
            )
        sorted_data_dict = pd.read_pickle(
            os.path.join("data", "LEVEL_1", "01_figure_1", "01_figure_1_data_dict.pkl")
        )
        if not os.path.exists(
            os.path.join("data", "LEVEL_1", "03_figure_3", "03_figure_3_data.pkl")
        ):
            raise FileNotFoundError(
                "Data should be in data/03_figure_3/03_figure_3_data.pkl \
            and come directly with the repository"
            )
        dataframes = pd.read_pickle(
            os.path.join("data", "LEVEL_1", "03_figure_3", "03_figure_3_data.pkl")
        )

        # figure 4
        folder_4 = os.path.join("data", "LEVEL_1", "04_figure_4")
        filepath_normal = os.path.join(folder_4, "04_Figure_4_results_fulltraining.csv")
        file_path_zeroshot = os.path.join(folder_4, "04_Figure_4_results_zeroshot.csv")
        df_figure_4_full = pd.read_csv(filepath_normal)
        df_figure_4_zeroshot = pd.read_csv(file_path_zeroshot)
        df_figure_4_zeroshot["data(group)"] = df_figure_4_zeroshot["data(group)"].str.replace(
            "_0shot", "", regex=False
        )


        


    if args.from_aggregation:
        # datafile = os.path.join("output", "data", "LEVEL_2", "all_dicts.pkl")
        # print(datafile)
        # if not os.path.exists(datafile):
        #     raise FileNotFoundError(
        #         f"Data {datafile} file not found. Please run data processing scripts/aggregate/analyse_res.py first"
        #     ) 
        # data_dicts = pd.read_pickle(datafile)
        # # Create a new dictionary with modified keys
        # new_data_dict = {key.split("/")[-1]: value for key, value in data_dicts.items()}
        # sorted_data_dict = dict(sorted(new_data_dict.items()))
        # with open(
        #     os.path.join("output", "data", "LEVEL_1", "01_figure_1_data_dict.pkl"), "wb"
        # ) as f:
        #     pickle.dump(sorted_data_dict, f)
        # # same for figure 2
        # with open(
        #     os.path.join("output", "data", "LEVEL_1", "02_figure_2_data_dict.pkl"), "wb"
        # ) as f:
        #     pickle.dump(sorted_data_dict, f)
        # if not os.path.exists(os.path.join("output", "01_figure_1_data_human_readable")):
        #     os.makedirs(
        #         os.path.join(
        #             "output", "data", "LEVEL_1", "01_figure_1_data_human_readable"
        #         ),
        #         exist_ok=True,
        #     )
        # save_dict = {
        #     key: [
        #         df.to_csv(
        #             os.path.join(
        #                 "output",
        #                 "data",
        #                 "LEVEL_1",
        #                 "01_figure_1_data_human_readable",
        #                 f"df_{key}_{i}",
        #             )
        #         )
        #         for i, df in enumerate(value)
        #     ]
        #     for key, value in sorted_data_dict.items()
        # }
        # # same for figure 2

        # if not os.path.exists(
        #     os.path.join("output", "data", "LEVEL_1", "02_figure_2_data_human_readable")
        # ):
        #     os.makedirs(
        #         os.path.join(
        #             "output", "data", "LEVEL_1", "02_figure_2_data_human_readable"
        #         ),
        #         exist_ok=True,
        #     )
        # save_dict = {
        #     key: [
        #         df.to_csv(
        #             os.path.join(
        #                 "output",
        #                 "data",
        #                 "LEVEL_1",
        #                 "02_figure_2_data_human_readable",
        #                 f"df_{key}_{i}",
        #             )
        #         )
        #         for i, df in enumerate(value)
        #     ]
        #     for key, value in sorted_data_dict.items()
        # }
        # # same for figure 3
        # relevant_dirs = ["cp-sc", "d2-sc", "sam-sc", "unet-sc"]
        # dataframes = []
        # base_path = os.path.join("output/data/LEVEL_2/AGGREGATED_RESULTS/")
        # for folder in relevant_dirs:
        #     path = os.path.join(base_path, folder, "999_DATASOURCE2_movie_plot_data.csv")
        #     if not os.path.exists(path):
        #         print(f"Path {path} does not exist")
        #         raise FileNotFoundError("Path does not exist")
        #     df = pd.read_csv(path)
        #     dataframes.append(df)
        # with open(
        #     os.path.join("output", "data", "LEVEL_1", "03_figure_3_data.pkl"), "wb"
        # ) as f:
        #     pickle.dump(dataframes, f)

        # if not os.path.exists(
        #     os.path.join("output", "data", "LEVEL_3", "03_figure_3_data_human_readable")
        # ):
        #     os.makedirs(
        #         os.path.join(
        #             "output", "data", "LEVEL_1", "03_figure_3_data_human_readable"
        #         ),
        #         exist_ok=True,
        #     )
        # for idx, df in enumerate(dataframes):
        #     df.to_csv(
        #         os.path.join(
        #             "output",
        #             "data",
        #             "LEVEL_1",
        #             "03_figure_3_data_human_readable",
        #             f"df_{titles[idx]}",
        #         )
        #     )
        # print("Saved dataframes to disk")

        # figure
        folder_4 = os.path.join("output", "data", "LEVEL_1", "04_figure_4")
        if not os.path.exists(folder_4):
            os.makedirs(folder_4, exist_ok=True)
        filepath_normal = os.path.join(folder_4, "04_Figure_4_results_fulltraining.csv")
        file_path_zeroshot = os.path.join(folder_4, "04_Figure_4_results_zeroshot.csv")

        input_folder_aggr = os.path.join("output", "data", "LEVEL_2")
        if not os.path.exists(input_folder_aggr):
            raise FileNotFoundError(f"Path {input_folder_aggr} does not exist. Run scripts/aggregate/ana_normal.py first")
        input_file_normal = os.path.join(input_folder_aggr, "results_normal", "results.csv")
        input_file_0shot = os.path.join(input_folder_aggr, "results_0shot", "results.csv")
        df_figure_4_full = pd.read_csv(input_file_normal)
        df_figure_4_zeroshot = pd.read_csv(input_file_0shot)
        df_figure_4_zeroshot["data(group)"] = df_figure_4_zeroshot["data(group)"].str.replace(
            "_0shot", "", regex=False
        )
        # save to disk for regular level 1
        df_figure_4_full.to_csv(filepath_normal)
        df_figure_4_zeroshot.to_csv(file_path_zeroshot)




    # ACTUAL PLOTTING ----------------------------------------------
    # figure 1 and supplementary figure 1
    if args.figure_1:
        figure_1_main(sorted_data_dict)
    # figure 2 and supplementary figure 2
    if args.figure_2:
        figure_2(
            data_dict=sorted_data_dict,
            metric1="iou",
            metric2="dAbsAbsConf",
            method="al",
            outpath=os.path.join("output", "main", "Figure_2.png"),
        )
    # figure 3
    if args.figure_3:
        figure_3(
            titles=titles,
            dataframes=dataframes,
            output_path=os.path.join("output", "main", "Figure_3.png"),
        )
    if args.figure_4:
        figure_4(
            df_all_epochs=df_figure_4_full,
            df_zeroshot=df_figure_4_zeroshot,
            respath=os.path.join("output", "main", "Figure_4.png"),
            metric="absabsDelConf",
            plot_errorbars=True,
        )
        figure_4(
            df_all_epochs=df_figure_4_full,
            df_zeroshot=df_figure_4_zeroshot,
            respath=os.path.join("output", "supplement", "Figure_S2.png"),
            metric="iou",
            plot_errorbars=True,
        )

    if args.all_figures:
        figure_1_main(sorted_data_dict)
        figure_2(
            data_dict=sorted_data_dict,
            metric1="iou",
            metric2="dAbsAbsConf",
            method="al",
            outpath=os.path.join("output", "main", "Figure_2.png"),
        )
        figure_3(
            titles=titles,
            dataframes=dataframes,
            output_path=os.path.join("output", "main", "Figure_3.png"),
        )
        figure_4(
            df_all_epochs=df_figure_4_full,
                df_zeroshot=df_figure_4_zeroshot,
                respath=os.path.join("output", "main", "Figure_4.png"),
                metric="absabsDelConf",
                plot_errorbars=True,
            )
        figure_4(
            df_all_epochs=df_figure_4_full,
            df_zeroshot=df_figure_4_zeroshot,
            respath=os.path.join("output", "supplement", "Figure_S2_iou.png"),
            metric="iou",
            plot_errorbars=True,
        )
