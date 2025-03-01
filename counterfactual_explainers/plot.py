import math
from importlib.resources import files
from tomllib import load

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from counterfactual_explainers.data.preprocess_data import clean_config

# TODO:plot metrics for aide


def main():
    package = files("counterfactual_explainers")
    toml_path = package / "config.toml"
    with toml_path.open("rb") as file:
        config = load(file)

    config = clean_config(config)

    for dataset in config["dataset"]:
        if dataset == "compas":
            metrics_dnn_df = pd.read_csv(
                f"counterfactual_explainers/results/cf_dice_DNN_{dataset}_metrics_2.csv"
            )
            metrics_rf_df = pd.read_csv(
                f"counterfactual_explainers/results/cf_dice_RF_{dataset}_metrics_2.csv"
            )

            # metrics_aide = pd.read_csv(
            #     f"counterfactual_explainers/results/cf_aide_DNN_{dataset}_metrics.csv"
            # )

            metrics = list(
                metrics_dnn_df.drop("num_required_cfs", axis=1).columns
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
                        name=f"DNN {metric}",
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
                        name=f"RF {metric}",
                        line=dict(color="red"),
                    ),
                    row=row,
                    col=col,
                )

                # fig.add_trace(
                #     go.Scatter(
                #         x=metrics_aide["num_required_cfs"],
                #         y=metrics_aide[metric],
                #         mode="lines+markers",
                #         name=f"DNN {metric}",
                #         line=dict(color="green"),
                #     ),
                #     row=row,
                #     col=col,
                # )

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


if __name__ == "__main__":
    main()
