import pandas as pd
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Table 1")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--from-agg", action="store_true", help="Use aggregated results")
    group.add_argument("--from-table-data", action="store_true", help="Use table data")
    return parser.parse_args()


def get_min_max_steps(sorted_data_dict, metric, method="al"):
    outpath_base = "output"
    results = {
        'key': [],
        'model': [],
        'dataset': [],
        'min': [],
        'max': [],
        'min_step': [],
        'max_rel_step': [],
        'min_rel_step': [],
        'best_step': [],
        'best_rel_step': [],
        'max_step': []
    }

    for k, v in sorted_data_dict.items():
        model = k.split("-")[0]
        dataset = k.replace(f"{model}-", "")
        
        # Concatenate all dataframes in the list, filter by method, then aggregate by mean
        combined_df = pd.concat(v, ignore_index=True)
        combined_df = combined_df[combined_df['method'] == method]
        
        # Aggregate the list of dataframes by mean, only for numeric columns
        numeric_df = combined_df.select_dtypes(include='number')
        numeric_df['step'] = combined_df['step']  # Ensure 'step' is included for grouping
        aggregated_df = numeric_df.groupby('step').mean().reset_index()

        # Calculate min and max for the specified metric
        max_delta = aggregated_df[metric].max()
        max_step = aggregated_df.loc[aggregated_df[metric] == max_delta, 'step'].values[0]
        max_steps = aggregated_df['step'].max()
        max_rel_step = max_step / max_steps
        
        min_delta = aggregated_df[metric].min()
        min_step = aggregated_df.loc[aggregated_df[metric] == min_delta, 'step'].values[0]
        min_rel_step = min_step / max_steps

        # Append results to the dictionary
        results['key'].append(k)
        results['model'].append(model)
        results['dataset'].append(dataset)
        results['min'].append(min_delta)
        results['max'].append(max_delta)
        results['min_step'].append(min_step)
        results['max_rel_step'].append(max_rel_step)
        results['min_rel_step'].append(min_rel_step)
        results['max_step'].append(max_step)
        if metric == "iou":
            results['best_step'].append(max_step)
            results['best_rel_step'].append(max_rel_step)
        else:
            results['best_step'].append(min_step)
            results['best_rel_step'].append(min_rel_step)



    # Convert results dictionary to DataFrame
    del results['min_rel_step']
    del results['max_rel_step']
    del results['min_step']
    del results['max_step']
    result_df = pd.DataFrame(results)
    result_df = result_df.round(2)

    # Save the DataFrame to a CSV file
    outpath = os.path.join(outpath_base, f"T1_{metric}__{method}_min_max_steps.csv") # table S2, when ababsDelConf,table S3 when iou
    if metric == "iou":
        outpath = os.path.join(outpath_base, "supplement", "Table_S3_iou_min_max_steps.csv")
        result_df.to_csv(outpath, index=False)
    elif metric == "dAbsAbsConf":
        outpath = os.path.join(outpath_base, "supplement", "Table_S2_dAbsAbsConf_min_max_steps.csv")
        result_df.to_csv(outpath, index=False)
    else:
        raise ValueError(f"Invalid metric: {metric}, use 'iou' or 'dAbsAbsConf'")
    mean_df = result_df.groupby('model').mean()
    std_df = result_df.groupby('model').std()
    mean_and_std_df = pd.concat([mean_df, std_df], axis=1)
    dataset_df = result_df.groupby('dataset').mean()
    dataset_std_df = result_df.groupby('dataset').std()
    dataset_mean_and_std_df = pd.concat([dataset_df, dataset_std_df], axis=1)
    if metric == "iou":
        outpath1 = os.path.join(outpath_base, "supplement", "Table_S1_a.csv")
        outpath2 = os.path.join(outpath_base, "supplement", "Table_S1_b.csv")
        mean_and_std_df.to_csv(outpath1)
        dataset_mean_and_std_df.to_csv(outpath2)
    elif metric == "dAbsAbsConf":
        outpath1 = os.path.join(outpath_base, "main", "Table_1_a.csv")
        outpath2 = os.path.join(outpath_base, "main", "Table_1_b.csv")
        mean_and_std_df.to_csv(outpath1)
        dataset_mean_and_std_df.to_csv(outpath2)
    else:
        raise ValueError(f"Invalid metric: {metric}, use 'iou' or 'dAbsAbsConf'")

    return result_df


if __name__ == "__main__":
    args = parse_args()
    if args.from_table_data:
        input_file = os.path.join(os.path.join("data","LEVEL_1"), "00_tables", "00_raw_data_for_all_tables.pkl")
        data_dicts = pd.read_pickle(input_file)
        new_data_dict = {key.split('/')[-1]: value for key, value in data_dicts.items()}
    if args.from_agg:
        # TODO
        pass


    # Sort the new dictionary by keys
    metrics = ["iou", "dAbsAbsConf"]
    methods = ["al", "rand"]
    out_dict = {}
    for metric in metrics:
        for method in methods:
            model_df = get_min_max_steps(new_data_dict, metric, method)
            # round all values to 2 decimal places
            model_df = model_df.round(2)
            #aggregated based on model
            # get best score for each model (by metric)
            if metric == "iou":
                # use max for iou
                best_model = model_df.groupby('model').max()
                worst_model_df =  model_df.groupby('model').min()
                # read step info
                # best_model.to_csv(os.path.join(outpath_base, f"T1_best_{metric}__{method}_min_max_steps.csv"))
                # worst_model_df.to_csv(os.path.join(outpath_base, f"T1_worst_{metric}__{method}_min_max_steps.csv"))
                # print(f"best_model for {metric} and {method}: {best_model}")
                # print(f"worst_model for {metric} and {method}: {worst_model_df}")
            else:
                # use min for dAbsAbsConf
                best_model_df = model_df.groupby('model').min()
                worst_model_df =  model_df.groupby('model').max()
                # best_model_df.to_csv(os.path.join(outpath_base, f"T1_best_{metric}__{method}_min_max_steps.csv"))
                # worst_model_df.to_csv(os.path.join(outpath_base, f"T1_worst_{metric}__{method}_min_max_steps.csv"))
                print(f"best_model for {metric} and {method}: {best_model}")
                print(f"worst_model for {metric} and {method}: {worst_model_df}")

            
            # print(f"model_df for {metric} and {method}: {model_df}")