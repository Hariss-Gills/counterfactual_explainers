"""
A module for visualizing and statistically analyzing counterfactual
explanation metrics.

This module provides functionality for:
- Loading and aggregating counterfactual evaluation metrics from CSV files.
- Performing statistical tests (ANOVA and Tukey HSD) on the metric scores.
- Generating interactive line plots for each metric to compare different
counterfactual generation methods.
- Creating parallel coordinates plots for a multi-dimensional visualization
of the dataset, query instances, and counterfactuals.

Key components:
- plot_metrics: Reads metric CSV files, computes statistics, and generates
metric
comparison plots.
- plot_parallel_coordinates: Preprocesses dataset features and renders a
parallel
coordinates plot.
- main: Orchestrates the execution workflow for visualizing counterfactual
explanation results.

Typical usage: TODO
"""

import math
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from plotly.subplots import make_subplots
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from counterfactual_explainers.data.preprocess_data import (
    clean_config,
    create_data_transformer,
    get_output_path,
    read_config,
    read_dataset,
)

RESULTS_PATH = get_output_path()
PLOTS_PATH = get_output_path("plots")
STATS_PATH = get_output_path("stats")


def plot_metrics(dataset: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Plot counterfactual evaluation metrics for a given dataset.

    This function reads CSV files for three methods (DICE gradient-based,
    DICE genetic, and AIDE) if available, extracts metric values, performs
    ANOVA and Tukey HSD statistical tests, and creates a subplot for each
    metric.

    Args:
        dataset: Name of the dataset to process.

    Returns:
        A tuple containing:
            - A concatenated DataFrame of metric scores from available methods.
            - A list of metric names plotted.
    """
    dnn_csv = RESULTS_PATH / f"cf_dice_dnn_{dataset}_metrics.csv"
    rf_csv = RESULTS_PATH / f"cf_dice_rf_{dataset}_metrics.csv"
    aide_csv = RESULTS_PATH / f"cf_aide_dnn_{dataset}_metrics.csv"

    try:
        metrics_dnn_df = pd.read_csv(dnn_csv)
        metrics_dnn_exists = True
    except FileNotFoundError:
        metrics_dnn_exists = False

    try:
        metrics_rf_df = pd.read_csv(rf_csv)
        metrics_rf_exists = True
    except FileNotFoundError:
        metrics_rf_exists = False

    try:
        metrics_aide_df = pd.read_csv(aide_csv)
        metrics_aide_exists = True
    except FileNotFoundError:
        metrics_aide_exists = False

    if metrics_rf_exists:
        base_df = metrics_rf_df
    elif metrics_dnn_exists:
        base_df = metrics_dnn_df
    elif metrics_aide_exists:
        base_df = metrics_aide_df
    else:
        raise FileNotFoundError("No metrics CSV file is available.")

    metrics = base_df.drop("Number of Required CFS", axis=1).columns.tolist()

    num_metrics = len(metrics)
    num_cols = 3
    num_rows = math.ceil(num_metrics / num_cols)

    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=metrics,
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    metrics_df = []
    for i, metric in enumerate(metrics):
        metric_df = []
        if metrics_aide_exists:
            metric_df.append(
                pd.DataFrame(
                    {
                        "score": metrics_aide_df[metric],
                        "method": "AIDE",
                        "metric": metric,
                    }
                )
            )
        if metrics_dnn_exists:
            metric_df.append(
                pd.DataFrame(
                    {
                        "score": metrics_dnn_df[metric],
                        "method": "DICE_gradient",
                        "metric": metric,
                    }
                )
            )
        if metrics_rf_exists:
            metric_df.append(
                pd.DataFrame(
                    {
                        "score": metrics_rf_df[metric],
                        "method": "DICE_genetic",
                        "metric": metric,
                    }
                )
            )

        metrics_df.extend(metric_df)
        scores = pd.concat(metric_df, ignore_index=True)
        model = ols("score ~ C(method)", data=scores).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        tukey = pairwise_tukeyhsd(
            endog=scores["score"], groups=scores["method"], alpha=0.05
        )
        print(f"Dataset {dataset}, Metric: {metric}")
        print("ANOVA results:")
        print(anova_table)
        print("\nTukey HSD test results:")
        print(tukey.summary())
        tukey_df = pd.DataFrame(
            tukey._results_table.data[1:], columns=tukey._results_table.data[0]
        )
        tukey_df.to_csv(
            STATS_PATH / f"{dataset}_{metric}_tukey.csv", index=False
        )

        row = (i // num_cols) + 1
        col = (i % num_cols) + 1
        if metrics_dnn_exists:
            fig.add_trace(
                go.Scatter(
                    x=metrics_dnn_df["Number of Required CFS"],
                    y=metrics_dnn_df[metric],
                    mode="lines+markers",
                    name=f"DICE Gradient-Based {metric}",
                    line=dict(color="blue"),
                ),
                row=row,
                col=col,
            )
        if metrics_rf_exists:
            fig.add_trace(
                go.Scatter(
                    x=metrics_rf_df["Number of Required CFS"],
                    y=metrics_rf_df[metric],
                    mode="lines+markers",
                    name=f"DICE Genetic {metric}",
                    line=dict(color="red"),
                ),
                row=row,
                col=col,
            )
        if metrics_aide_exists:
            fig.add_trace(
                go.Scatter(
                    x=metrics_aide_df["Number of Required CFS"],
                    y=metrics_aide_df[metric],
                    mode="lines+markers",
                    name=f"AIDE Local {metric}",
                    line=dict(color="green"),
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        title_text=f"Counterfactual Metrics for {dataset}",
    )

    for i in range(1, num_metrics + 1):
        axis_key = "xaxis" if i == 1 else f"xaxis{i}"
        if axis_key in fig.layout:
            fig.layout[axis_key].update(title_text="Number of Required CFS")

    fig.show()
    fig.write_html(PLOTS_PATH / f"metrics_plot_{dataset}.html")
    return pd.concat(metrics_df, ignore_index=True), metrics


def plot_parallel_coordinates(config: dict[str, Any], dataset: str) -> None:
    """
    Generate and display a parallel coordinates plot for a given dataset.

    This function reads and preprocesses the dataset, transforms categorical
    features to numerical codes, and creates a parallel coordinates plot with
    Plotly. It distinguishes between the original dataset, the query instance,
    and counterfactuals generated using different methods.

    Args:
        config: Configuration dictionary containing dataset and model parameters.
        dataset: Name of the dataset to process.

    Returns:
        None
    """
    data = read_dataset(config, dataset)

    params_dataset = config["dataset"][dataset]

    continuous_features = data["continuous_features"]
    categorical_features = data["categorical_features"]
    features = data["features"]
    target = data["target"]
    preprocessor, target_encoder = create_data_transformer(
        continuous_features=continuous_features,
        categorical_features=categorical_features,
    )

    # impute only
    preprocessor.fit(features)

    cont_imputer = preprocessor.named_transformers_["continuous"].named_steps[
        "imputer"
    ]
    cat_imputer = preprocessor.named_transformers_["categorical"].named_steps[
        "imputer"
    ]

    new_feat_cont = cont_imputer.fit_transform(features[continuous_features])
    new_feat_cont = pd.DataFrame(
        new_feat_cont,
        columns=continuous_features,
        index=features.index,
    )
    if dataset != "fico":
        new_feat_cat = cat_imputer.fit_transform(
            features[categorical_features]
        )

        new_feat_cat = pd.DataFrame(
            new_feat_cat,
            columns=categorical_features,
            index=features.index,
        )
        features = pd.concat([new_feat_cont, new_feat_cat], axis=1)
    else:
        features = new_feat_cont

    df = pd.concat([features, target], axis=1)

    df["dummy_target"] = target_encoder.fit_transform(
        df[params_dataset["target_name"]]
    )

    label_mapping = {
        class_label: index
        for index, class_label in enumerate(target_encoder.classes_)
    }
    print(label_mapping)

    cfs_dice_gradient_path = RESULTS_PATH / f"cf_dice_dnn_{dataset}_20.csv"
    cfs_dice_genetic_path = RESULTS_PATH / f"cf_dice_rf_{dataset}_20.csv"
    cfs_aide_path = RESULTS_PATH / f"cf_aide_dnn_{dataset}.csv"

    cfs_dice_gradient_exists = True
    cfs_dice_genetic_exists = True
    cfs_aide_exists = True

    try:
        cfs_dice_gradient = pd.read_csv(cfs_dice_gradient_path)
    except FileNotFoundError:
        cfs_dice_gradient_exists = False

    try:
        cfs_dice_genetic = pd.read_csv(cfs_dice_genetic_path)
    except FileNotFoundError:
        cfs_dice_genetic_exists = False

    try:
        cfs_aide = pd.read_csv(cfs_aide_path)
    except FileNotFoundError:
        cfs_aide_exists = False

    query_instance = pd.read_csv(
        RESULTS_PATH / f"cf_dice_rf_{dataset}_query_instance_combined.csv"
    )

    query_instance["dummy_target"] = target_encoder.transform(
        query_instance["target"]
    )

    df["type"] = 0
    query_instance["type"] = 0.80

    dfs_to_concat = [df, query_instance]

    if cfs_aide_exists:
        cfs_aide["type"] = 0.20
        cfs_aide["dummy_target"] = 1 - query_instance["dummy_target"].iloc[0]
        dfs_to_concat.append(cfs_aide)

    if cfs_dice_gradient_exists:
        cfs_dice_gradient["type"] = 0.40
        cfs_dice_gradient["dummy_target"] = (
            1 - query_instance["dummy_target"].iloc[0]
        )
        dfs_to_concat.append(cfs_dice_gradient)

    if cfs_dice_genetic_exists:
        cfs_dice_genetic["type"] = 0.60
        cfs_dice_genetic["dummy_target"] = (
            1 - query_instance["dummy_target"].iloc[0]
        )
        if dataset == "compas":
            cfs_dice_genetic["dummy_target"] = (
                1 + query_instance["dummy_target"].iloc[0]
            )

        dfs_to_concat.append(cfs_dice_genetic)

    combined_df = pd.concat(dfs_to_concat, axis=0)

    mappings = {}
    for feature in categorical_features:
        combined_df[feature], mappings[feature] = pd.factorize(
            combined_df[feature]
        )

    dimensions = []
    for feature in features:
        if feature in categorical_features:
            labels = mappings[feature]
            tickvals = list(range(len(labels)))
            ticktext = list(labels)
            dimensions.append(
                {
                    "label": feature,
                    "values": combined_df[feature],
                    "tickvals": tickvals,
                    "ticktext": ticktext,
                }
            )
        else:
            dimensions.append(
                {
                    "label": feature,
                    "values": combined_df[feature],
                }
            )
    dimensions.append(
        {
            "label": params_dataset["target_name"],
            "values": combined_df["dummy_target"].values,
            "tickvals": list(label_mapping.values()),
            "ticktext": list(target_encoder.classes_),
        }
    )

    fig = go.Figure(
        layout=go.Layout(
            title=go.layout.Title(
                text=f"Parallel coordinate Plot for {dataset}"
            )
        ),
        data=go.Parcoords(
            line=dict(
                color=combined_df["type"],
                colorscale=[
                    (0.00, "grey"),
                    (0.20, "grey"),
                    (0.20, "green"),
                    (0.40, "green"),
                    (0.40, "blue"),
                    (0.60, "blue"),
                    (0.60, "red"),
                    (0.80, "red"),
                    (0.80, "magenta"),
                    (1.00, "magenta"),
                ],
                colorbar=dict(
                    title="Legend",
                    tickvals=[0.1, 0.25, 0.4, 0.55, 0.725],
                    ticktext=[
                        "Dataset",
                        "Counterfactuals AIDE",
                        "Counterfactuals DICE gradient",
                        "Counterfactuals DICE genetic",
                        "Query Instance",
                    ],
                ),
                showscale=True,
            ),
            dimensions=dimensions,
            unselected=dict(line=dict(opacity=0)),
        ),
    )

    fig.show()
    fig.write_html(PLOTS_PATH / f"parallel_coordinate_plot_{dataset}.html")


def main() -> None:
    """
    Main function for plotting and statistical analysis.
    """
    config = clean_config(read_config())

    across_dataset_metrics = []
    for dataset in config["dataset"]:
        metrics_df, metrics = plot_metrics(dataset)
        across_dataset_metrics.append(metrics_df)
        plot_parallel_coordinates(config, dataset)

    for metric in metrics:
        all_scores = pd.concat(across_dataset_metrics, ignore_index=True)
        scores = all_scores[all_scores["metric"] == metric]
        group_means = scores.groupby("method")["score"].mean()
        print("Group Means:")
        print(group_means)

        model = ols("score ~ C(method)", data=scores).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        tukey = pairwise_tukeyhsd(
            endog=scores["score"], groups=scores["method"], alpha=0.05
        )
        print(f"ANOVA results for metric: {metric}")
        print(anova_table)
        print("\nTukey HSD test results:")
        print(tukey.summary())
        tukey_df = pd.DataFrame(
            tukey._results_table.data[1:], columns=tukey._results_table.data[0]
        )
        tukey_df.to_csv(STATS_PATH / f"{metric}_tukey.csv", index=False)


if __name__ == "__main__":
    main()
