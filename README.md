# confluence-paper
This repository accompanies our paper "How to estimate confluence lazily".
We show how to obtain the in the paper shown results on three levels:
1. LEVEL 1: How to get the plots and tables from specifically for the plotting aggregated data
2. LEVEL 2: How to get the plots and figures from the aggregation of the raw results
3. LEVEL 3: How to aggregate the raw results

# Requirements



# Getting started with Level 1
We pre-prepared the data to generate the plots and made it available with this repo.
To generate the figures as shown in the paper and supplement run:
```python scripts/tables/tables.py --from-table-data```
This will generate the following output
```
output/
|-- additional_figures_and_tables
|-- main
|   |-- Table_1_a.csv
|   `-- Table_1_b.csv
`-- supplement
    |-- Table_S1_a.csv
    |-- Table_S1_b.csv
    |-- Table_S2_dAbsAbsConf_min_max_steps.csv
    |-- Table_S3_dAbsAbsConf_min_max_steps.csv
    |-- Table_S4_iou_min_max_steps.csv
    `-- Table_S5_iou_min_max_steps.csv
```

```python scripts/figures/all_figures.py --from-figure-data --all-figures```

This will result in:
```
output/
|-- additional_figures_and_tables
|   |-- Figure_1_no_global_y_axis._p_values.csv
|   |-- Figure_1_no_global_y_axis.png
|   |-- Figure_S1_no_global_y_axis._p_values.csv
|   `-- Figure_S1_no_global_y_axis.png
|-- main
|   |-- Figure_1._p_values.csv
|   |-- Figure_1.png
|   |-- Figure_2.png
|   |-- Figure_2_p_values.csv
|   |-- Figure_3.png
|   |-- Figure_3_p_values.csv
|   `-- Figure_4.png
`-- supplement
    |-- Figure_S1._p_values.csv
    |-- Figure_S1.png
    `-- Figure_S2_iou.png|`output
|-- main
|   |-- Figure_1._p_values.csv
|   `-- Figure_1.png
|-- main_tables
|-- supplement
|   |-- Figure_S1._p_values.csv
|   `-- Figure_S1.png
`-- supplement_tables
```
Note if you want to only generate a specific figure then use `--figure-<n>` instead of `--all-figures`. This will generate the figure and the corresponding supplement figures
# Pipeline
tldr:
- do_all.sh
- get raw results file:
- ```wget TODO```vv
- and here ``wget nromal res`
- run analyse_res.py
- run ana_normal.py
- run_paper figure.py
- notebooks
