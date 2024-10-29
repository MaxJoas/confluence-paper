import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import pandas as pd
import os


def plot_bars(df_all_epochs, df_zeroshot, respath, metric, plot_errorbars=True):
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
    cmap = cm.get_cmap("viridis", len(data_names))
    group_colors = {
        group: cmap(i / len(data_names)) for i, group in enumerate(data_names)
    }

    # First subplot

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
        os.path.join(respath, f"04_Figure_4_{metric}_by_model_and_data_wError.png")
    )
    plt.close()


if __name__ == "__main__":
    # filepath_zeroshot = "/home/majo158f/development/confluence-results/final-results/normal_and_0shot_res/NORMAL_CONFLUENCE_AGGREGATED/0SHOT_AGGREGATED_RESULTS/results.csv"
    # filepath = "/home/majo158f/development/NORMAL_CONFLUENCE_AGGREGATED/results.csv"
    # outpath_base = /home/majo158f/development/confluence-results/final-results/R1/AGGREGATED_RESULTS/PAPER

    filepath = sys.argv[1]
    filepath_zeroshot = sys.argv[2]
    outpath_base = sys.argv[3]

    df = pd.read_csv(filepath)
    df.to_csv(os.path.join(outpath_base, "04_Figure_4_results_fulltraining.csv"))
    df_zero = pd.read_csv(filepath_zeroshot)
    df_zero["data(group)"] = df_zero["data(group)"].str.replace(
        "_0shot", "", regex=False
    )
    df_zero.to_csv(os.path.join(outpath_base, "04_Figure_4_results_zeroshot.csv"))
    print(df_zero.head())
    print(df.head())
    plot_bars(
        df_all_epochs=df,
        df_zeroshot=df_zero,
        respath=outpath_base,
        metric="absabsDelConf",
        plot_errorbars=True,
    )
    print("Plotted absabsDelConf")
    plot_bars(
        df_all_epochs=df,
        df_zeroshot=df_zero,
        respath=outpath_base,
        metric="iou",
        plot_errorbars=True,
    )
    print("done")
