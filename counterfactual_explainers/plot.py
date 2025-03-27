import math
from importlib.resources import files
from tomllib import load

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from counterfactual_explainers.data.preprocess_data import (
    clean_config,
    create_data_transformer,
    read_config,
    read_dataset,
)

# TODO:plot parallel coordinate plot


def main():
    config = clean_config(read_config())
    for dataset in config["dataset"]:
        if dataset == "german_credit":
            metrics_dnn_df = pd.read_csv(
                f"counterfactual_explainers/results/cf_dice_DNN_{dataset}_metrics_2.csv"
            )
            metrics_rf_df = pd.read_csv(
                f"counterfactual_explainers/results/cf_dice_RF_{dataset}_metrics_2.csv"
            )

            metrics_aide = pd.read_csv(
                f"counterfactual_explainers/results/cf_aide_DNN_{dataset}_metrics_2.csv"
            )

            metrics = list(
                metrics_rf_df.drop("num_required_cfs", axis=1).columns
            )
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

            for i, metric in enumerate(metrics):
                row = (i // num_cols) + 1
                col = (i % num_cols) + 1
                fig.add_trace(
                    go.Scatter(
                        x=metrics_dnn_df["num_required_cfs"],
                        y=metrics_dnn_df[metric],
                        mode="lines+markers",
                        name=f"DICE Gradient-Based {metric}",
                        line=dict(color="blue"),
                    ),
                    row=row,
                    col=col,
                )

                fig.add_trace(
                    go.Scatter(
                        x=metrics_rf_df["num_required_cfs"],
                        y=metrics_rf_df[metric],
                        mode="lines+markers",
                        name=f"DICE Genetic {metric}",
                        line=dict(color="red"),
                    ),
                    row=row,
                    col=col,
                )

                fig.add_trace(
                    go.Scatter(
                        x=metrics_aide["num_required_cfs"],
                        y=metrics_aide[metric],
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
                    fig.layout[axis_key].update(
                        title_text="Number of Required CFs"
                    )

            fig.show()
            fig.write_html(
                f"counterfactual_explainers/plots/metrics_plot_{dataset}.html"
            )
            # Add parrallel coordinate plots for k=20 for all methods
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

            cont_imputer = preprocessor.named_transformers_[
                "continuous"
            ].named_steps["imputer"]
            cat_imputer = preprocessor.named_transformers_[
                "categorical"
            ].named_steps["imputer"]

            new_feat_cont = cont_imputer.fit_transform(
                features[continuous_features]
            )
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

            cfs_dice_gradient = pd.read_csv(
                f"counterfactual_explainers/results/cf_dice_DNN_{dataset}_20.csv"
            )

            cfs_dice_genetic = pd.read_csv(
                f"counterfactual_explainers/results/cf_dice_RF_{dataset}_20.csv"
            )

            cfs_aide = pd.read_csv(
                f"counterfactual_explainers/results/cf_aide_DNN_{dataset}.csv"
            )

            query_instance = pd.read_csv(
                f"counterfactual_explainers/results/cf_dice_DNN_{dataset}_query_instance.csv"
            )

            df["type"] = 0

            cfs_aide["type"] = 0.20
            cfs_aide["dummy_target"] = 1

            cfs_dice_gradient["type"] = 0.40
            cfs_dice_gradient["dummy_target"] = 1

            cfs_dice_genetic["type"] = 0.60
            cfs_dice_genetic["dummy_target"] = 1

            query_instance["type"] = 0.80
            query_instance["dummy_target"] = 0

            combined_df = pd.concat(
                [
                    df,
                    cfs_dice_gradient,
                    cfs_dice_genetic,
                    cfs_aide,
                    query_instance,
                ],
                axis=0,
            )

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
                    "tickvals": [0, 1],
                    "ticktext": list(target_encoder.classes_),
                }
            )

            fig = go.Figure(
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
                )
            )

            fig.show()
            fig.write_html(
                f"counterfactual_explainers/plots/parallel_coordinate_plot_{dataset}.html"
            )


if __name__ == "__main__":
    main()
