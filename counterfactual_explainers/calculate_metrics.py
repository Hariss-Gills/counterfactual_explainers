from importlib.resources import files
from pathlib import Path
from tomllib import load

import numpy as np
import pandas as pd

from counterfactual_explainers.data.preprocess_data import (
    clean_config,
    read_compas_dataset,
    read_dataset,
)

# TODO: calculate runtime and figure out Euclidean-Jaccard/Hamming distance
# in paper
# NOTE: later calculate Instability, Implausibility, and Discriminative Power


def median_absolute_deviation(cf_row):
    mad = (cf_row - cf_row.median()).abs().median()
    return mad if mad else 1


def distance(
    cf_row, query_instance, mad, continuous_features, categorical_features
):

    cf_cont = cf_row[continuous_features].to_numpy()
    query_cont = query_instance[continuous_features].to_numpy()

    cf_cat = cf_row[categorical_features].to_numpy()
    query_cat = query_instance[categorical_features].to_numpy()

    mad = mad.to_numpy()

    cont_diff = np.abs(cf_cont - query_cont)
    cont_dist = np.sum(cont_diff / mad)

    cat_dist = np.sum(cf_cat != query_cat)

    return (cont_dist / len(continuous_features)) + (
        cat_dist / len(categorical_features)
    )


def calc_changes(cf_row, query_instance, features):
    cf_values = cf_row[features].to_numpy()
    query_values = query_instance[features].to_numpy()
    return (cf_values != query_values).sum()


def calc_size(num_required_cfs, cfs_df):
    return len(cfs_df) / num_required_cfs


def actionable(cf_row, query_instance, non_act_features):
    cf_values = cf_row[non_act_features].to_numpy()
    query_values = query_instance.iloc[0][non_act_features].to_numpy()
    non_action_changes = (cf_values != query_values).any()
    return 0 if non_action_changes else 1


def calc_diversity(
    cfs, continuous_features, categorical_features, feature_cols, mad
):
    normalization_factor = len(cfs) ** 2

    pairwise_dists = cfs.apply(
        lambda row: cfs.apply(
            lambda row2: distance(
                row, row2, mad, continuous_features, categorical_features
            ),
            axis=1,
        ),
        axis=1,
    )
    diversity_distance = (
        np.sum(pairwise_dists.to_numpy()) / normalization_factor
    )

    pairwise_change_counts = cfs.apply(
        lambda row: cfs.apply(
            lambda row2: calc_changes(row, row2, feature_cols), axis=1
        ),
        axis=1,
    )
    diversity_count = np.sum(pairwise_change_counts.to_numpy()) / (
        normalization_factor * len(feature_cols)
    )

    return diversity_distance, diversity_count


def main():
    package = files("counterfactual_explainers")
    toml_path = package / "config.toml"
    with toml_path.open("rb") as file:
        config = load(file)

    config = clean_config(config)

    for dataset in config["dataset"]:
        if dataset == "compas":
            data = read_compas_dataset()
        else:
            data = read_dataset(dataset)

        for model_name in config["model"]:
            results = []
            for num_required_cfs in range(1, 21):
                if dataset == "adult":
                    try:
                        continuous_features = data["continuous_features"]
                        categorical_features = data["categorical_features"]
                        non_act_features = data["non_act_features"]
                        features = data["features"]
                        cfs = pd.read_csv(
                            f"counterfactual_explainers/results/cf_dice_{model_name}_{dataset}_{num_required_cfs}.csv"
                        )
                        query_instance = pd.read_csv(
                            f"counterfactual_explainers/results/cf_dice_{model_name}_{dataset}_query_instance.csv"
                        )

                        print(f"{model_name}, {dataset}, {num_required_cfs}")

                        size = calc_size(num_required_cfs, cfs)
                        mad = cfs[continuous_features].apply(
                            median_absolute_deviation
                        )
                        dist = cfs.apply(
                            distance,
                            axis=1,
                            mad=mad,
                            query_instance=query_instance,
                            continuous_features=continuous_features,
                            categorical_features=categorical_features,
                        )
                        dis_dist = dist.mean()
                        dis_changes = cfs.apply(
                            calc_changes,
                            axis=1,
                            query_instance=query_instance,
                            features=features.columns,
                        )
                        dis_count = (dis_changes.mean()) / len(
                            features.columns
                        )
                        act = (
                            cfs.apply(
                                actionable,
                                axis=1,
                                query_instance=query_instance,
                                non_act_features=non_act_features,
                            ).sum()
                            / num_required_cfs
                        )

                        div_dist, div_count = calc_diversity(
                            cfs,
                            continuous_features,
                            categorical_features,
                            features.columns,
                            mad,
                        )

                        results.append(
                            {
                                "num_required_cfs": num_required_cfs,
                                "Size": size,
                                "Dissimilarity_distance": dis_dist,
                                "Dissimilarity_count": dis_count,
                                "Actionability": act,
                                "Diversity_distance": div_dist,
                                "Diversity_count": div_count,
                            }
                        )

                    except FileNotFoundError:
                        print("Not avaliable")

            results_df = pd.DataFrame(results).set_index("num_required_cfs")
            print("\nMetrics Summary DataFrame:")
            print(results_df)

            results_path = Path("./counterfactual_explainers/results")
            results_path.mkdir(parents=True, exist_ok=True)

            results_df.to_csv(
                results_path / f"cf_dice_{model_name}_{dataset}"
                f"_metrics.csv",
            )


if __name__ == "__main__":
    main()
