import os
import pickle
import sys
import pandas as pd
import csv
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import logging
from collections import defaultdict

from matplotlib.cm import viridis
from matplotlib.colors import to_rgba

import scipy.stats as stats
from typing import Tuple, List, Dict

def perform_statistical_tests(all_run_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Perform statistical tests comparing AL vs random at each step across all runs.
    
    Args:
        all_run_dfs: List of DataFrames, each containing results from one run
        
    Returns:
        DataFrame with statistical test results for each step
    """
    # Combine all runs
    combined_df = pd.concat(all_run_dfs, ignore_index=True)
    steps = sorted(combined_df['step'].unique())
    
    results = []
    metrics_to_test = ['iou', 'precision', 'recall', 'f1', 'accuracy']
    
    for step in steps:
        step_data = combined_df[combined_df['step'] == step]
        step_result = {'step': step}
        
        for metric in metrics_to_test:
            al_scores = step_data[step_data['method'] == 'al'][metric]
            rand_scores = step_data[step_data['method'] == 'rand'][metric]
            
            # Perform Mann-Whitney U test
            statistic, p_value = stats.mannwhitneyu(
                al_scores, 
                rand_scores,
                alternative='two-sided'
            )
            
            # Calculate effect size (Cliff's Delta)
            d = cliff_delta(al_scores.values, rand_scores.values)
            
            step_result.update({
                f'{metric}_p_value': p_value,
                f'{metric}_effect_size': d,
                f'{metric}_significant': p_value < 0.05,
                f'{metric}_al_median': al_scores.median(),
                f'{metric}_rand_median': rand_scores.median()
            })
        
        results.append(step_result)
    
    return pd.DataFrame(results)

def cliff_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Cliff's Delta effect size"""
    nx = len(x)
    ny = len(y)
    dominance = 0
    
    for i in x:
        for j in y:
            if i > j:
                dominance += 1
            elif i < j:
                dominance -= 1
                
    return dominance / (nx * ny)

def plot_statistical_results(stats_df: pd.DataFrame, metric: str, save_path: str = None):
    """
    Plot statistical test results for a specific metric.
    
    Args:
        stats_df: DataFrame containing statistical test results
        metric: Metric to plot (e.g., 'iou', 'precision')
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot medians
    steps = stats_df['step']
    ax1.plot(steps, stats_df[f'{metric}_al_median'], 'b-', label='AL')
    ax1.plot(steps, stats_df[f'{metric}_rand_median'], 'r-', label='Random')
    ax1.set_title(f'{metric.upper()} Medians by Step')
    ax1.set_xlabel('Step')
    ax1.set_ylabel(metric.upper())
    ax1.legend()
    
    # Add significance markers
    sig_steps = stats_df[stats_df[f'{metric}_significant']]['step']
    y_max = max(stats_df[f'{metric}_al_median'].max(), stats_df[f'{metric}_rand_median'].max())
    for step in sig_steps:
        ax1.text(step, y_max * 1.05, '*', horizontalalignment='center', size=20)
    
    # Plot effect sizes
    ax2.plot(steps, stats_df[f'{metric}_effect_size'], 'g-', marker='o')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    ax2.set_title("Effect Size (Cliff's Delta)")
    ax2.set_xlabel('Step')
    ax2.set_ylabel("Cliff's Delta")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def check_paths(path_tuple):
    """
    Args:
        path_tuple (tuple): A tuple containing the paths to the directories containing evaluation metrics CSV files.
        The first element of the tuple should be the path to the AL directory,
        and the second element should be the path to the random sampling directory.

    Returns:
        non_existing_paths (list) - all paths where there are no eval_metrics.csv
    """
    al_path = path_tuple[0]
    random_path = path_tuple[1]
    dirs = os.listdir(al_path)
    step_dirs = [d for d in dirs if "step_" in d]
    dirs_rand = os.listdir(random_path)
    step_dirs_rand = [d for d in dirs_rand if "step_" in d]
    non_existing_paths = []
    for al_dir, rand_dir in zip(step_dirs, step_dirs_rand):
        al_step = int(al_dir.split("_")[-1])
        rand_step = int(rand_dir.split("_")[-1])
        cur_al_path = os.path.join(al_path, al_dir, "eval_metrics.csv")
        cur_rand_path = os.path.join(random_path, rand_dir, "eval_metrics.csv")

        if not os.path.exists(cur_al_path):
            print(cur_al_path, "does not exist")
            non_existing_paths.append(cur_al_path)
        if not os.path.exists(cur_rand_path):
            print(cur_rand_path, "does not exist")
            non_existing_paths.append(cur_rand_path)
    return non_existing_paths


def aggregate_dfs(path_tuple):
    """
    Aggregate evaluation metrics from multiple steps of active learning (AL) and random sampling methods.

    Args:
        path_tuple (tuple): A tuple containing the paths to the directories containing evaluation metrics CSV files.
        The first element of the tuple should be the path to the AL directory,
        and the second element should be the path to the random sampling directory.

    Returns:
        pandas.DataFrame: A DataFrame containing the aggregated evaluation metrics.
        Each row represents a step, and the columns include the evaluation metrics,
        the step number, and the method used (AL or random sampling).
    """
    df = pd.DataFrame()
    al_path = path_tuple[0]
    random_path = path_tuple[1]
    dirs = os.listdir(al_path)
    step_dirs = [d for d in dirs if "step_" in d]
    dirs_rand = os.listdir(random_path)
    step_dirs_rand = [d for d in dirs_rand if "step_" in d]
    if not len(step_dirs) == len(step_dirs_rand):
        logger.info(f"WARNING: DIFFERENT NUMBER OF STEPS IN {path_tuple}")
        logger.info(step_dirs)
        logger.info(step_dirs_rand)

    for al_dir, rand_dir in zip(step_dirs, step_dirs_rand):
        al_step = int(al_dir.split("_")[-1])
        rand_step = int(rand_dir.split("_")[-1])
        cur_al_path = os.path.join(al_path, al_dir, "eval_metrics.csv")
        cur_rand_path = os.path.join(random_path, rand_dir, "eval_metrics.csv")
        df_al = pd.read_csv(cur_al_path)
        del df_al["iou_mask_wise"]
        rel_row_al = pd.DataFrame(df_al.iloc[-1, :])
        rel_row_al = rel_row_al.T
        rel_row_al["step"] = al_step
        # logger.info(f"al_dir: {al_dir}, rand_dir: {rand_dir}, al_step:{al_step}")
        rel_row_al["method"] = "al"
        rel_row_al["dAbsConf"] = (
            rel_row_al["pred_confluence"] - rel_row_al["gt_confluence"]
        )
        rel_row_al["dAbsAbsConf"] = abs(
            rel_row_al["pred_confluence"] - rel_row_al["gt_confluence"]
        )
        df_rand = pd.read_csv(cur_rand_path)
        del df_rand["iou_mask_wise"]
        rel_row_rand = pd.DataFrame(df_rand.iloc[-1, :])
        rel_row_rand = rel_row_rand.T
        rel_row_rand["step"] = rand_step
        rel_row_rand["method"] = "rand"
        rel_row_rand["dAbsConf"] = (
            rel_row_rand["pred_confluence"] - rel_row_rand["gt_confluence"]
        )
        rel_row_rand["dAbsAbsConf"] = abs(
            rel_row_rand["pred_confluence"] - rel_row_rand["gt_confluence"]
        )
        df_temp = pd.concat([rel_row_rand, rel_row_al], axis=0)
        df = pd.concat([df, df_temp])

    return df.sort_values(by=["step"])


def plot_rand_vs_al_aggr(merged_df, outpath):
    metrics = [
        "iou",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "relative_delta_confluence",
        "dAbsConf",
        "dAbsAbsConf",
    ]
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        steps = merged_df["step"].unique()
        methods = merged_df["method"].unique()
        palette = sns.color_palette("viridis", len(methods))
        for i, method in enumerate(methods):
            df_method = merged_df[merged_df["method"] == method]
            offsets = np.linspace(-0.1, 0.1, len(methods))
            marker = "o"
            if method == "al":
                marker = "x"
            for j, step in enumerate(steps):
                x_offset = step + offsets[i]
                y_mean = df_method[df_method["step"] == step][f"{metric}_mean"].values[
                    0
                ]
                y_std = df_method[df_method["step"] == step][f"{metric}_std"].values[0]

                plt.errorbar(
                    x_offset,
                    y_mean,
                    yerr=y_std,
                    fmt=marker,
                    color=palette[i],
                    elinewidth=2,
                    capsize=5,
                    alpha=0.9,
                    label=f"{method}" if j == 0 else "",
                )
        plt.xlabel("Step")
        plt.ylabel(f"Mean {metric} Value")
        plt.xticks(steps, rotation=45)  # Set x-ticks to steps
        plt.legend(title="Method", loc="best")
        plt.savefig(os.path.join(outpath, f"01_{metric}_werror_DATASOURCE_1.png"))
        plt.close()
        plt.clf()
    plt.close()
    plt.clf()


def plot_rand_vs_al_aggr_no_error(merged_df, outpath):
    metrics = [
        "iou",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "relative_delta_confluence",
        "dAbsConf",
        "dAbsAbsConf",
    ]
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        steps = merged_df["step"].unique()
        methods = merged_df["method"].unique()
        palette = sns.color_palette("viridis", len(methods))
        for i, method in enumerate(methods):
            df_method = merged_df[merged_df["method"] == method]
            offsets = np.linspace(-0.1, 0.1, len(methods))
            marker = "o"
            if method == "al":
                marker = "x"
            for j, step in enumerate(steps):
                x_offset = step + offsets[i]
                y_mean = df_method[df_method["step"] == step][f"{metric}_mean"].values[
                    0
                ]

                plt.scatter(
                    x_offset,
                    y_mean,
                    marker=marker,
                    color=palette[i],
                    alpha=0.9,
                    label=f"{method}" if j == 0 else "",
                )
        plt.xlabel("Step")
        plt.ylabel(f"Mean {metric} Value")
        plt.xticks(steps, rotation =45)  # Set x-ticks to steps
        plt.legend(title="Method", loc="best")
        plt.savefig(os.path.join(outpath, f"01_{metric}_noerror_DATASOURCE1.png"))
        plt.close()
        plt.clf()
    plt.close()
    plt.clf()


def plot_rand_vs_al(df, outpaths):
    try:

        sns.lineplot(data=df, x="step", y="iou", hue="method", palette="viridis")
        plt.savefig(os.path.join(outpaths[0], "iou.png"))
        plt.savefig(os.path.join(outpaths[1], "iou.png"))
        plt.close()
        sns.lineplot(
            data=df,
            x="step",
            y="relative_delta_confluence",
            palette="viridis",
            hue="method",
        )
        plt.savefig(os.path.join(outpaths[0], "relative_delta_confluence.png"))
        plt.savefig(os.path.join(outpaths[1], "relative_delta_confluence.png"))
        plt.close()
        fig, ax = plt.subplots(3, 2, figsize=(15, 15))
        sns.lineplot(
            data=df, x="step", y="iou", hue="method", palette="viridis", ax=ax[0, 0]
        )
        sns.lineplot(
            data=df,
            x="step",
            y="precision",
            hue="method",
            palette="viridis",
            ax=ax[0, 1],
        )
        sns.lineplot(
            data=df, x="step", y="recall", hue="method", palette="viridis", ax=ax[1, 0]
        )
        sns.lineplot(
            data=df, x="step", y="f1", hue="method", palette="viridis", ax=ax[1, 1]
        )
        sns.lineplot(
            data=df,
            x="step",
            y="accuracy",
            hue="method",
            palette="viridis",
            ax=ax[2, 0],
        )
        sns.lineplot(
            data=df,
            x="step",
            y="relative_delta_confluence",
            hue="method",
            ax=ax[2, 1],
            palette="viridis",
        )
        plt.savefig(os.path.join(outpaths[0], "all_metrics.png"))
        plt.savefig(os.path.join(outpaths[1], "all_metrics.png"))
        plt.close()
        plt.clf()
    except Exception as e:
        logger.info(e)
        logger.warning(f"ERROR PATH \n {outpaths} \n")
        logger.info(df)
        plt.close()
        plt.clf()
        pass


def plot_moviepos_vs_steps(res_dict, res_dict_rand, outpath):
    """
    Plot the positions of a movie vs. the steps.

    Args:
        res_dict (dict): A dictionary containing the movie positions for each step.
        outpath (str): The path to save the plot.

    Returns:
        None
    """

    steps = []
    positions = []
    positions_rand = []
    for step, pos_list in res_dict.items():
        steps.extend([step] * len(pos_list))
        positions.extend(pos_list)
    for step, pos_list in res_dict_rand.items():
        positions_rand.extend(pos_list)
    plot_path = os.path.join(outpath, "moviepos_vs_steps.png")
    logger.info(f"plot_path for movieplot: {plot_path}")
    if not len(steps) == len(positions):
        logger.warning(f"steps {steps} and positions: {positions} differ in len")
        logger.info("error in :")
        logger.info(f"plot_path for movieplot: {plot_path}")
        return
    plt.scatter(steps, positions, label="Active Learning", cmap="viridis")
    # logger.info(f"plot_path for movieplot: {plot_path}")
    if not len(steps) == len(positions_rand):
        logger.warning(
            f"steps {steps} and positions_rand: {positions_rand} differ in len"
        )
        logger.info("error in :")
        logger.info(f"plot_path for movieplot: {plot_path}")
        plt.xlabel("Step")
        plt.ylabel("Position")
        plt.legend()
        plt.savefig(plot_path)
        plt.close()
        return

    plt.scatter(
        steps,
        positions_rand,
        marker="X",
        label="Random",
        color="red",
        alpha=0.5,
        cmap="viridis",
    )
    plt.xlabel("Step")
    plt.ylabel("Position")
    plt.legend()
    plt.savefig(plot_path)
    plt.close()
    plt.clf()


def prepare_moviepos_vs_steps(path):
    """
    Prepare movie position vs steps data.

    Args:
        path (str): A path containing the input folder path.

    Returns:
        None
    """
    input_folder = os.path.join(path, "AL_LOGGING")
    # logger.info(f"input_folder for movie plot: {input_folder}")
    files = os.listdir(input_folder)
    jsonfiles = [f for f in files if f.endswith(".json")]
    txtfiles = [f.replace("_files", "") for f in files if f.endswith(".txt")]
    usecase = "json" if len(jsonfiles) > len(txtfiles) else "txt"
    # logger.info(
    #     f"usecase: {usecase}, len(jsonfiles): {len(jsonfiles)}, len(txtfiles): {len(txtfiles)}"
    # )
    res_dict = {}
    if usecase == "json":
        jsonfiles.sort(key=lambda x: int(x.split("_")[-1].replace(".json", "")))

        step_files = [
            (
                f"step_{int(f.split('_')[-1].replace('.json', ''))+1}.json"
                if "coco" not in f
                else f
            )
            for f in jsonfiles
        ]
        previous_imgfilenames = []
        # logger.info(f"sorted step_files: {step_files}")
        for f, stepf in zip(jsonfiles, step_files):
            # get all image filenames in the coco object
            coco = COCO(os.path.join(input_folder, f))
            imgids = coco.getImgIds()
            imgfilenames = [coco.loadImgs(i)[0]["file_name"] for i in imgids]
            step = int(stepf.split("_")[-1].replace(".json", ""))
            # res_dict[step] = {"files":[], "newimage":[], "movieposition":[]}

            newimages = [
                img for img in imgfilenames if img not in previous_imgfilenames
            ]
            # logger.info(f'len newimage: {len(newimage)}')

            if len(newimages) > 0:
                movieposition = [
                    int(img.split("Image")[1].replace(".jpg", "")) for img in newimages
                ]
            previous_imgfilenames = imgfilenames.copy()
            # logger.info(
            #     f" json:{f}, step:{int(step)}, newimage:{newimages}, movieposition:{movieposition}"
            # )
            # logger.info(f"all images: {imgfilenames}")
            res_dict[step] = movieposition
    elif usecase == "txt":
        txtfiles.sort(key=lambda x: int(x.split("_")[-1].replace(".txt", "")))
        # logger.info(f"sorted txtfiles: {txtfiles}")
        previous_imgfilenames = []
        for f in txtfiles:
            with open(os.path.join(input_folder, f), "r") as file:
                lines = file.readlines()
                # remove empty lines
                lines = [l.strip() for l in lines if l != "\n"]
                newimages = [l for l in lines if l not in previous_imgfilenames]
                movieposition = [
                    int(img.split("Image")[1].replace(".jpg", "")) for img in newimages
                ]
                previous_imgfilenames = lines.copy()
                step = int(f.split("_")[-1].replace(".txt", ""))
                res_dict[step] = movieposition
                # logger.info(
                #     f"file: {f}, nreimages: {newimages}, movieposition: {movieposition}"
                # )
    else:
        logger.info("No valid usecase found.")
    logger.info(f"final res_dict in prepare_movie: {res_dict}")

    return res_dict


def object_to_float(aggr_df):
    for col in aggr_df.columns:
        if col == "images":
            continue
        if col == "method":
            continue
        aggr_df[col] = aggr_df[col].astype(np.float32)
    return aggr_df


def aggregate_movie_dicts(all_movie_dicts):
    res = defaultdict(list)
    for d in all_movie_dicts:
        for k, v in d.items():
            res[k].extend(v)
    return dict(res)


def get_delta_dict(res_dict):
    sorted_d = dict(sorted(res_dict.items()))
    delta_dict = {}
    for second in list(sorted_d.keys())[1::]:
        first = second - 1

        print(first, second)
        first_mean = np.mean(sorted_d[first])
        second_mean = np.mean(sorted_d[second])
        delta = abs(second_mean - first_mean)
        delta_dict[second] = [delta]
    return delta_dict

def plot_movie_vs_pos_aggr(al_data, rand_data, outpath):
    # Save data as pickle
    with open(os.path.join(outpath, "999_DATASOURCE2_movie_plot_data.pkl"), "wb") as f:
        pickle.dump((al_data, rand_data), f)
    
    # Save data as CSV
    csv_path = os.path.join(outpath, "999_DATASOURCE2_movie_plot_data.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Step", "Method", "Value"])

        for step in al_data.keys():
            for value in al_data[step]:
                writer.writerow([step, "al", value])
            for value in rand_data[step]:
                writer.writerow([step, "rand", value])

    # Plotting
    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("viridis", 2)
    
    steps = list(al_data.keys())
    al_means = [np.mean(al_data[k]) for k in steps]
    rand_means = [np.mean(rand_data[k]) for k in steps]

    plt.plot(steps, al_means, 'o-', color=palette[0], label="al", alpha=0.9)
    plt.plot(steps, rand_means, 'x-', color=palette[1], label="rand", alpha=0.9)

    plt.axhline(y=0, color="red", linestyle="--", linewidth=1)
    plt.text(
        steps[-1] + 1,
        0,
        "Previous \n Position",
        color="red",
        verticalalignment="center",
    )

    plt.xlabel("Step")
    plt.ylabel("Mean delta Position")

    plt.xticks(steps, rotation=45)  # Set x-ticks to steps
    plt.legend(title="Method", loc="best")
    plot_path = os.path.join(outpath, "02_moviepos_X_steps_DATASOURCE2_no_errorbars.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Data saved in human-readable format: {csv_path}")
    print(f"Plot saved: {plot_path}")

def plot_movie_vs_pos_aggr_with_errorbars(al_data, rand_data, outpath):
    import pickle

    with open(os.path.join(outpath, "999_DATASOURCE2_movie_plot_data.pkl"), "wb") as f:
        pickle.dump((al_data, rand_data), f)
    csv_path = os.path.join(outpath, "999_DATASOURCE2_movie_plot_data.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Step", "Method", "Value"])

        for step in al_data.keys():
            for value in al_data[step]:
                writer.writerow([step, "al", value])
            for value in rand_data[step]:
                writer.writerow([step, "rand", value])

    offsets = np.linspace(-0.1, 0.1, 2)
    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("viridis", 2)
    first = True
    for k, v in al_data.items():
        x_offset = k + offsets[0]
        plt.errorbar(
            x_offset,
            np.mean(al_data[k]),
            yerr=np.std(al_data[k]),
            fmt="o",
            color=palette[0],
            capsize=5,
            alpha=0.9,
            label="al" if first else "",
        )
        x_offset = k + offsets[1]
        plt.errorbar(
            x_offset,
            np.mean(rand_data[k]),
            yerr=np.std(rand_data[k]),
            fmt="x",
            color=palette[1],
            capsize=5,
            alpha=0.9,
            label="rand" if first else "",
        )
        first = False
    plt.axhline(y=0, color="red", linestyle="--", linewidth=1)
    plt.text(
        list(al_data.keys())[-1] + 1,
        0,
        "Previous \n Position",
        color="red",
        verticalalignment="center",
    )

    plt.xlabel("Step")
    plt.ylabel("Mean delta Position")

    plt.xticks(list(al_data.keys()), rotation=45)  # Set x-ticks to steps
    plt.legend(title="Method", loc="best")
    plot_path = os.path.join(outpath, "02_moviepos_X_steps_DATASOURCE2.png")
    plt.savefig(plot_path)
    plt.close()


def path_exists_helper(path):
    if not os.path.exists(path):
        logger.warning(f"Path {path} does not exist")
    return path if os.path.exists(path) else None


def prepare_goal_dep_aggrplot(all_aggr_res_paths):
    base_dict = {
        "unet": {"lazy": None, "exact": None},
        "d2": {"lazy": None, "exact": None},
        "sam": {"lazy": None, "exact": None},
        "cp": {"lazy": None, "exact": None},
    }
    for p in all_aggr_res_paths:
        if "lazy" in p:
            id_helper = p.split("/")[-1]
            logger.info(f"ID_HELPER: {id_helper}")
            method_helper = id_helper.split("-")[0]
            basepath = p.replace(id_helper, "")
            logger
            exact_df_mean_path = path_exists_helper(
                os.path.join(
                    basepath, id_helper.replace("lazy", ""), "999_DATASOURCE1_PART1_meta_mean_al_vs_rand.csv"
                )
            )
            exact_df_std_path = path_exists_helper(
                os.path.join(
                    basepath, id_helper.replace("lazy", ""), "999_DATASOURCE1_PART2_meta_std_al_vs_rand.csv"
                )
            )

            lazy_df_mean_path = path_exists_helper(
                os.path.join(basepath, id_helper, "999_DATASOURCE1_PART1_meta_mean_al_vs_rand.csv")
            )
            lazy_df_std_path = path_exists_helper(
                os.path.join(basepath, id_helper, "999_DATASOURCE1_PART2_meta_std_al_vs_rand.csv")
            )

            base_dict[method_helper]["lazy"] = [lazy_df_mean_path, lazy_df_std_path]
            base_dict[method_helper]["exact"] = [exact_df_mean_path, exact_df_std_path]
    return base_dict


def plot_goal_dep_agg_with_errorbars(base_dict, method):
    logger.info("Starting to plot goal dependent errorbars")
    logger.info('BASE_DICT: ', base_dict)
    metrics = ["iou", "dAbsAbsConf"]
    color_palette = sns.color_palette("viridis", len(metrics))
    markers = ["o", "x"]  # Different markers for different metrics
    logger.info("Starting to plot goal dependent errorbars")
    skips = 0

    for k, v in base_dict.items():
        if v["lazy"] is None or v["exact"] is None:
            logger.debug(f"Skipping {k}, because one of the files is None")
            skips += 1
            continue

        logger.debug(k)
        df_lazy_mean = pd.read_csv(v["lazy"][0], index_col=0)
        df_lazy_std = pd.read_csv(v["lazy"][1], index_col=0)
        df_lazy = pd.merge(
            df_lazy_mean, df_lazy_std, on=["step", "method"], suffixes=("_mean", "_std")
        )
        df_lazy["label"] = "lazy"

        df_exact_mean = pd.read_csv(v["exact"][0], index_col=0)
        df_exact_std = pd.read_csv(v["exact"][1], index_col=0)
        df_exact = pd.merge(
            df_exact_mean,
            df_exact_std,
            on=["step", "method"],
            suffixes=("_mean", "_std"),
        )
        df_exact["label"] = "exact"

        merged_df = pd.concat([df_exact, df_lazy], axis=0)
        merged_df = merged_df[merged_df["method"] != method]
        # save merged_df as csv

        path_helper = base_dict[k]["lazy"][0].split("/")[:-2]
        res_base_path = "/".join(path_helper)

        merged_df.to_csv(os.path.join(res_base_path, f"999_DATASOURCE3{k}_merged_df.csv"))
        logger.info(f"res_base_path: {res_base_path}")
        filename = f"03_{k}-{method}-goal-dep-enhanced-errorbars_DATASOURCE3.png"
        with open(
            os.path.join(res_base_path, filename.replace("png", "pkl").replace("03","999_DATASOURCE3")), "wb"
        ) as f:
            pickle.dump(merged_df, f)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        for metric, ax in zip(metrics, axes.flatten()):
            # Calculate unique steps and methods
            steps = merged_df["step"].unique()
            methods = merged_df["label"].unique()

            # Assign colors to each method
            palette = sns.color_palette("viridis", len(methods))

            # Create a plot
            for i, method in enumerate(methods):
                df_method = merged_df[merged_df["label"] == method]
                offsets = np.linspace(-0.1, 0.1, len(methods))
                marker = "o"
                if method == "al":
                    marker = "x"
                for j, step in enumerate(steps):
                    x_offset = step + offsets[i]
                    y_mean = df_method[df_method["step"] == step][
                        f"{metric}_mean"
                    ].values[0]
                    y_std = df_method[df_method["step"] == step][
                        f"{metric}_std"
                    ].values[0]

                    ax.errorbar(
                        x_offset,
                        y_mean,
                        yerr=y_std,
                        fmt=marker,
                        color=palette[i],
                        elinewidth=2,
                        capsize=5,
                        alpha=0.9,
                        label=f"{method}" if j == 0 else "",
                    )

            # Customize the plot
            plt.xlabel("Step")
            plt.ylabel(f"Mean {metric} Value")
            plt.xticks(steps, rotation=45)  # Set x-ticks to steps
            plt.legend(title="Method", loc="best")

        plt.tight_layout()
        plt.savefig(os.path.join(res_base_path, filename))
        plt.close()
        plt.clf()
    plt.close()
    plt.clf()

    logger.info(f"Total: {len(base_dict)} plots, Skips: {skips}")


if __name__ == "__main__":
    global logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    fh = logging.FileHandler("analyse_res.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    respath = sys.argv[1]
    paths_df = pd.read_csv(respath, index_col=0)
    logger.info(f"paths_df: {paths_df}")

    mapping = {k: k.replace("-al", "-rand") for k in paths_df.index if "-al" in k}
    all_tuples = [
        (paths_df.loc[k, "path"], paths_df.loc[v, "path"]) for k, v in mapping.items()
    ]
    logger.info(f"All tuples: {all_tuples}")

    all_aggr_res_paths = []
    all_dicts = {}
    for tup in all_tuples:
        if not (os.path.exists(tup[0]) and os.path.exists(tup[1])):
            logger.info(f"Path {tup[0]} or path {tup[1]} does not exist, skipping")
            logger.info(f"Whole path: {tup}")
            continue

        all_aggregations, all_rand_movie_dicts, all_al_movie_dicts = [], [], []

        for i in range(int(sys.argv[2])):
            base1 = tup[0].split("/")[-1] + f"_{i}"
            base2 = tup[1].split("/")[-1] + f"_{i}"
            sub1 = os.path.join(tup[0], base1)
            sub2 = os.path.join(tup[1], base2)
            if not (os.path.exists(sub1) and os.path.exists(sub2)):
                print(f"path {base1}{sub1} or {base2}{sub2} does not exist")
                continue

            # Dropna on aggr_df and log shapes before and after
            aggr_df = object_to_float(aggregate_dfs((sub1, sub2)))
            aggr_df_copy = aggr_df.copy()
            before_shape = aggr_df_copy.shape
            logger.debug(f"aggr_df before dropna: {before_shape}")
            # aggr_df = aggr_df.dropna(axis=0)
            aggr_df = aggr_df.fillna(aggr_df.mean(numeric_only=True))
            after_shape = aggr_df.shape
            logger.debug(f"aggr_df after dropna: {aggr_df.shape}")
            if not before_shape == after_shape:
                logger.warning(aggr_df_copy)
                logger.warning(aggr_df)
                aggr_df_copy.to_csv("before_drop-nan.csv")

            # Save aggregations and log
            if not os.path.exists(os.path.join(sub1, "REPORTS")):
                os.makedirs(os.path.join(sub1, "REPORTS"), exist_ok=True)
            if not os.path.exists(os.path.join(sub2, "REPORTS")):
                os.makedirs(os.path.join(sub2, "REPORTS"), exist_ok=True)

            aggr_df.to_csv(os.path.join(sub1, "REPORTS", "aggregated.csv"))
            aggr_df.to_csv(os.path.join(sub2, "REPORTS", "aggregated.csv"))
            logger.debug("Calling plot_rand_vs_al")
            plot_rand_vs_al(
                aggr_df, [os.path.join(sub1, "REPORTS"), os.path.join(sub2, "REPORTS")]
            )
            all_aggregations.append(aggr_df)

            if "-sc" in tup[0]:
                res_dict_al = prepare_moviepos_vs_steps(sub1)
                res_dict_rand = prepare_moviepos_vs_steps(sub2)
                fig = plot_moviepos_vs_steps(
                    res_dict_al, res_dict_rand, os.path.join(sub1, "REPORTS")
                )
                all_al_movie_dicts.append(get_delta_dict(res_dict_al))
                all_rand_movie_dicts.append(get_delta_dict(res_dict_rand))
        all_dicts[tup[0]] = all_aggregations

        # Stack arrays
        try:
            arr = np.stack(
                [
                    df.drop(columns=["images", "method"]).values
                    for df in all_aggregations
                ],
                axis=-1,
            )
            logger.info(f"Shape of stacked array 'arr': {arr.shape}")
        except Exception as e:
            logger.error(f"Error during np.stack: {e}")
            with open("plotting_dfs.pkl", "wb") as f:
                pickle.dump(all_aggregations, f)
            continue
        with open(os.path.join("plotting_dfs_all.pkl"), "wb") as f:
            pickle.dump(all_dicts, f)

        # Compute mean and std data
        mean_data = arr.mean(axis=-1, dtype=np.float32)
        std_data = arr.std(axis=-1, dtype=np.float32)

        # Create meta_df and meta_df_std, and log shapes
        meta_df = pd.DataFrame(
            mean_data, columns=aggr_df.drop(columns=["images", "method"]).columns
        )
        meta_df_std = pd.DataFrame(
            std_data, columns=aggr_df.drop(columns=["images", "method"]).columns
        )
        logger.info(
            f"meta_df shape: {meta_df.shape}, meta_df_std shape: {meta_df_std.shape}"
        )

        # Try to add images and method columns
        try:
            logger.debug("Adding images and method columns")
            meta_df["images"] = aggr_df["images"].values
            meta_df["method"] = aggr_df["method"].values
        except Exception as e:
            logger.warning(f"Exception when adding 'images' and 'method': {e}")
            meta_df.to_csv("debug-meta-df.csv")
            aggr_df.to_csv("aggr_df-debug.csv")

        # Cast step column to uint32 and log final meta_df shape
        meta_df["step"] = meta_df["step"].astype(np.uint32)
        meta_df_std["images"] = aggr_df["images"].values
        meta_df_std["method"] = aggr_df["method"].values
        meta_df_std["step"] = meta_df["step"].values
        logger.info(f"Final meta_df shape: {meta_df.shape}")

        # Save results to file
        h = tup[0].split("/")
        base_id = h[-1].replace("-al", "")
        aggregated_res_path = os.path.join(
            h[0], h[1], h[2], h[3], "AGGREGATED_RESULTS", base_id
        )
        logger.info(f"aggregated_res_path: {aggregated_res_path}")
        all_aggr_res_paths.append(aggregated_res_path)
        if not os.path.exists(aggregated_res_path):
            os.makedirs(aggregated_res_path, exist_ok=True)

        logger.info(f"Saving meta_mean_al_vs_rand.csv to {aggregated_res_path}")
        meta_df.to_csv(
            os.path.join(
                aggregated_res_path, "999_DATASOURCE1_PART1_meta_mean_al_vs_rand.csv"
            )
        )
        meta_df_std.to_csv(
            os.path.join(
                aggregated_res_path, "999_DATASOURCE1_PART2_meta_std_al_vs_rand.csv"
            )
        )
        np.save(
            os.path.join(
                aggregated_res_path, "999_DATASOURCE1_ARRAY_aggregated_al_ran.npy"
            ),
            arr,
        )

        # Merge and log final merged_df shape
        merged_df = pd.merge(
            meta_df, meta_df_std, on=["step", "method"], suffixes=("_mean", "_std")
        )
        merged_df.to_csv(
            os.path.join(aggregated_res_path, "999_DATASOURCE1_FINAL_merged_al_vs_rand.csv")
        )
        logger.info(f"Final merged_df shape: {merged_df.shape}")

        # Plot based on the aggregated movie data
        if "-sc" in tup[0]:
            aggr_movie_al = aggregate_movie_dicts(all_al_movie_dicts)
            aggr_movie_rand = aggregate_movie_dicts(all_rand_movie_dicts)
            plot_movie_vs_pos_aggr_with_errorbars(
                aggr_movie_al, aggr_movie_rand, aggregated_res_path
            )
            plot_movie_vs_pos_aggr(
                aggr_movie_al, aggr_movie_rand, aggregated_res_path
            )
        plot_rand_vs_al_aggr(merged_df=merged_df, outpath=aggregated_res_path)
        plot_rand_vs_al_aggr_no_error(merged_df=merged_df, outpath=aggregated_res_path)

    aggregated_all_path = os.path.join(h[0], h[1], h[2], h[3], "AGGREGATED_RESULTS", "all_dicts.pkl")
    with open(aggregated_all_path, "wb") as f:
        pickle.dump(all_dicts, f)

    # Goal dependent labeling
    # all_aggr_res_paths = ['../confluence-results/final-results/R1/AGGREGATED_RESULTS/unet-lc-internal', '../confluence-results/final-results/R1/AGGREGATED_RESULTS/d2-lc-internal', '../confluence-results/final-results/R1/AGGREGATED_RESULTS/sam-lc-internal', '../confluence-results/final-results/R1/AGGREGATED_RESULTS/cp-lc-internal', '../confluence-results/final-results/R1/AGGREGATED_RESULTS/unet-lc-external', '../confluence-results/final-results/R1/AGGREGATED_RESULTS/d2-lc-external', '../confluence-results/final-results/R1/AGGREGATED_RESULTS/sam-lc-external', '../confluence-results/final-results/R1/AGGREGATED_RESULTS/cp-lc-external', '../confluence-results/final-results/R1/AGGREGATED_RESULTS/unet-sc', '../confluence-results/final-results/R1/AGGREGATED_RESULTS/d2-sc', '../confluence-results/final-results/R1/AGGREGATED_RESULTS/sam-sc', '../confluence-results/final-results/R1/AGGREGATED_RESULTS/cp-sc', '../confluence-results/final-results/R1/AGGREGATED_RESULTS/unet-lc-internallazy', '../confluence-results/final-results/R1/AGGREGATED_RESULTS/d2-lc-internallazy', '../confluence-results/final-results/R1/AGGREGATED_RESULTS/sam-lc-internallazy', '../confluence-results/final-results/R1/AGGREGATED_RESULTS/cp-lc-internallazy']
    logger.info(f"all_aggr_res_paths: {all_aggr_res_paths}")
    base_dict = prepare_goal_dep_aggrplot(all_aggr_res_paths)
    plot_goal_dep_agg_with_errorbars(base_dict, "al")
    plot_goal_dep_agg_with_errorbars(base_dict, "rand")
