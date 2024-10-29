import seaborn as sns
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import matplotlib.cm as cm


def process_results(folders):
    res_dict = {
        "model(col)": [],
        "data(group)": [],
        "iou": [],
        "relDelConf": [],
        "absDelConf": [],
        "absabsDelConf": [],
        "iou_std": [],
        "relDelConf_std": [],
        "absDelConf_std": [],
        "absabsDelConf_std": [],
    }
    print("folders", folders)
    for subfolder in folders:
        print(subfolder)
        subfolders = os.listdir(subfolder)
        print(f"len subfolders before filter: {len(subfolders)}")
        subfolders = [f for f in subfolders if "." not in f]
        print(f"len subfolders after filter: {len(subfolders)}")
        for f in subfolders:
            print(f"folder: {f}")
            df = pd.read_csv(os.path.join(subfolder, f, "eval_metrics.csv"))
            model_name = f.split("-")[0]
            dataset_name = f.replace(model_name, "").replace("_normal", "")
            res_dict["data(group)"].append(dataset_name)
            res_dict["model(col)"].append(model_name)

            df["absDelConf"] = df["gt_confluence"] - df["pred_confluence"]
            df["absabsDelConf"] = abs(df["gt_confluence"] - df["pred_confluence"])
            last_row = df.iloc[-1]
            res_dict["iou"].append(last_row["iou"])
            res_dict["relDelConf"].append(last_row["relative_delta_confluence"])
            res_dict["absDelConf"].append(last_row["absDelConf"])
            res_dict["absabsDelConf"].append(last_row["absabsDelConf"])
            # get standard deviation of iou, relDelConf, absDelConf
            res_dict["iou_std"].append(df["iou"].std())
            res_dict["relDelConf_std"].append(df["relative_delta_confluence"].std())
            res_dict["absDelConf_std"].append(df["absDelConf"].std())
            res_dict["absabsDelConf_std"].append(df["absabsDelConf"].std())
    return res_dict


def plot_bars(res_df, respath, plot_errorbars=True):
    data_names = res_df["data(group)"].unique()
    model_names = res_df["model(col)"].unique()
    metrics = ["iou", "absabsDelConf"]
    offset = 0
    sns.set_style("darkgrid")
    plt.rcParams.update({"font.size": 18, "font.weight": "bold"})
    METRIC_FRIENDLY_NAME = {"iou": "Mean IoU", "absabsDelConf": "Mean Δ Confluence"}
    DATA_GROUP_FRIENDLY_NAME = {
        "-sc": "SC",
        "-lc-internal": "LC Internal",
        "-lc-external": "LC External",
        "-lc-internallazy": "LC Internal Lazy",
    }
    friendly_data_names = [DATA_GROUP_FRIENDLY_NAME[name] for name in data_names]

    for metric in metrics:
        cmap = cm.get_cmap("viridis", len(data_names))
        group_colors = {
            group: cmap(i / len(data_names)) for i, group in enumerate(data_names)
        }

        models = []
        barWidth = 0.5
        offset = 0

        for i, model_name in enumerate(model_names):
            print(f'i: {i}, "model_name: {model_name}')
            models.append(model_name)

            bars = res_df[res_df["model(col)"] == model_name][metric]
            yerrs = res_df[res_df["model(col)"] == model_name][f"{metric}_std"]
            yerrs = yerrs.values.tolist()
            if not plot_errorbars:
                print("not plot errobars")
                yerrs = [None for _ in range(len(yerrs))]
            groups = res_df[res_df["model(col)"] == model_name]["data(group)"]
            r = np.arange(len(bars))
            for j, (bar, group) in enumerate(zip(bars, groups)):
                plt.bar(
                    r[j] + i * 0.5 * barWidth + offset,
                    bar,
                    color=group_colors[group],
                    width=barWidth,
                    yerr=yerrs[j],
                )

            offset += 8

        plt.xticks([1.5, 9.5, 17.5, 25.5, 33.5], models)
        # Create a legend for the data groups
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=group_colors[group])
            for group in data_names
        ]
        plt.legend(
            handles,
            friendly_data_names,
            title="Data Groups",
            fontsize=16,
            title_fontsize=16,
        )
        plt.ylabel(METRIC_FRIENDLY_NAME[metric], fontweight="bold", fontsize=18)
        plt.tick_params(axis="both", which="major", labelsize=16)
        # for label in plt.get_xticklabels() + plt.get_yticklabels():
        #     label.set_fontweight("bold")
        #     plt.legend(loc="best", fontsize=16)
        plt.show()
        plt.tight_layout()  # Adjust layout to fit labels and legend
        file_base_name = "by_model_and_data_wError"
        if not plot_errorbars:
            file_base_name = file_base_name.replace("_wError", "")
        plt.savefig(os.path.join(respath, f"{metric}_{file_base_name}"))
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outpath",
        help="output path",
        type=str,
        default="NORMAL_CONFLUENCE_AGGREGATED",
    )
    parser.add_argument(
        "--result-base-dir",
        default="results_normal",
        dest="basedir",
        type=str,
        help="root folder where the results are store, usually results_normal or results_0shot",
    )

    args = parser.parse_args()
    result_base_dir = args.basedir
    respath = args.outpath
    folders = [
        os.path.join("confluence", result_base_dir),
        os.path.join("confluence-sam", result_base_dir),
        os.path.join("confluence-unet", result_base_dir),
    ]
    if not os.path.exists(respath):
        os.makedirs(respath)
    res_dict = process_results(folders)
    res_df = pd.DataFrame(res_dict)
    data_names = res_df["data(group)"].unique()
    model_names = res_df["model(col)"].unique()
    res_df.to_csv(os.path.join(respath, "results.csv"))
    model_names = model_names.tolist()
    print(f"model_names: {model_names}")
    print(res_df)
    plot_bars(res_df, respath)
    plot_bars(res_df, respath, False)
