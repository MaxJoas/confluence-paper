import seaborn as sns
from urllib.parse import urlparse, urlencode
import requests
import zipfile
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import matplotlib.cm as cm
import os
import zipfile
import requests



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
        "-sc_0shot": "SC",
        "-lc-internal_0shot": "LC Internal",
        "-lc-external_0shot": "LC External",
        "-lc-internallazy_0shot": "LC Internal Lazy",
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

        if len(models) == 5:
            plt.xticks([1.5, 9.5, 17.5, 25.5, 33.5], models)
        if len(models) == 3:    
            plt.xticks([1.5, 10.0, 18.5], models)
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




def download_and_process_normal():
    url = "https://cloud.scadsai.uni-leipzig.de/index.php/s/tdFAfDSojDB6C42/download/normal_and_zero_shot_results.zip"
    outpath = os.path.join("output", "data", "LEVEL_3")
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)

    try:
        # Download the ZIP file
        filepath = os.path.join(outpath, "normal_and_zero_shot_results.zip")
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print(f"Downloaded file saved to {filepath}")

        # Unzip the file
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(outpath)

        # Rename the extracted folder if necessary
        extracted_folder = os.path.join(outpath, "normal_and_zero_shot_results (1)")
        expected_folder = os.path.join(outpath, "normal_and_zero_shot_results")
        
        if os.path.exists(extracted_folder) and not os.path.exists(expected_folder):
            os.rename(extracted_folder, expected_folder)
            print(f"Renamed folder: {extracted_folder} -> {expected_folder}")
        elif os.path.exists(expected_folder):
            print(f"Folder already exists: {expected_folder}")
        else:
            print(f"Extracted folder does not need renaming: {extracted_folder}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid ZIP file.")


def aggregate(result_base_dir):
    respath = os.path.join("output", "data", "LEVEL_2", result_base_dir)
    inbase = os.path.join("output", "data", "LEVEL_3", "normal_and_zero_shot_results")
    if not os.path.exists(respath):
        os.makedirs(respath)
    folders = [
        os.path.join(inbase, "confluence", result_base_dir),
        os.path.join(inbase, "confluence-sam", result_base_dir),
        os.path.join(inbase, "confluence-unet", result_base_dir),
    ]
    if not os.path.exists(respath):
        os.makedirs(respath)
    res_dict = process_results(folders)
    res_df = pd.DataFrame(res_dict)
    data_names = res_df["data(group)"].unique()
    model_names = res_df["model(col)"].unique()
    res_df.to_csv(os.path.join(respath, "results.csv"))
    model_names = model_names.tolist()
    plot_bars(res_df, respath)
    plot_bars(res_df, respath, False)



if __name__ == "__main__":

    # url = "https://www.dropbox.com/scl/fi/isiiysz2qmrxmtttjtls1/normal_and_zero_shot_results.zip?rlkey=u0ejj3dowwqwftj28llrlfcab&st=qojk38eq&dl=1"
    outpath = os.path.join("output", "data", "LEVEL_3")
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)
    # download_from_dropbox(url=url, extract_path=outpath)

    # download_and_process_normal()
    aggregate("results_normal")
    aggregate("results_0shot")