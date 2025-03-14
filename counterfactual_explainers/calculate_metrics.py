from importlib.resources import files
from pathlib import Path
from tomllib import load

import numpy as np
import pandas as pd
from scipy.sparse import isspmatrix_csr
from sklearn.model_selection import train_test_split

from counterfactual_explainers.data.preprocess_data import (
    clean_config,
    create_data_transformer,
    read_compas_dataset,
    read_dataset,
)

# TODO: calculate runtime and figure out Euclidean-Jaccard/Hamming distance
# in paper.
# TODO: Implement calculation with CSR matrices instead
# NOTE: later calculate Instability, Implausibility, and Discriminative Power
# NOTE: So After some checks, it seems that the paper divides counts by non-encoded lengths
# this is weird since distances I think are divided by encoded lengths. Or maybe
# my results simply deviate a lot.


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

    if continuous_features.empty:
        cont_term = 0
    else:
        cont_term = cont_dist / len(continuous_features)

    if categorical_features.empty:
        cat_term = 0
    else:
        cat_term = cat_dist / len(categorical_features)

    return cont_term + cat_term


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
            for explainer in ["aide", "dice"]:
                results = []
                for num_required_cfs in range(1, 21):
                    try:
                        if dataset == "german_credit":

                            print(
                                f"{model_name}, {dataset}, {num_required_cfs}, {explainer}"
                            )
                            params_model = config["model"][model_name]
                            params_dataset = config["dataset"][dataset]
                            #
                            seed = params_model["classifier__random_state"][0]
                            continuous_features = data["continuous_features"]
                            categorical_features = data["categorical_features"]
                            non_act_features = data["non_act_features"]
                            features = data["features"]
                            target = data["target"]
                            #
                            preprocessor, _ = create_data_transformer(
                                continuous_features=continuous_features,
                                categorical_features=categorical_features,
                            )

                            encoded_features = preprocessor.fit_transform(
                                features
                            )

                            if isspmatrix_csr(encoded_features):
                                encoded_features = encoded_features.toarray()

                            feature_names = (
                                preprocessor.get_feature_names_out()
                            )
                            features_encoded_df = pd.DataFrame(
                                encoded_features,
                                columns=feature_names,
                                index=features.index,
                            )

                            X_train, _, _, _ = train_test_split(
                                features,
                                target,
                                test_size=params_dataset["test_size"],
                                random_state=seed,
                                stratify=target,
                            )

                            cfs = pd.read_csv(
                                f"counterfactual_explainers/results/cf_{explainer}_{model_name}_{dataset}.csv"
                            )

                            cfs = cfs.iloc[:num_required_cfs]
                            print(cfs)

                            cfs_encoded = preprocessor.transform(
                                cfs[features.columns]
                            )

                            if isspmatrix_csr(cfs_encoded):
                                cfs_encoded = cfs_encoded.toarray()

                            cfs_encoded_df = pd.DataFrame(
                                cfs_encoded,
                                columns=preprocessor.get_feature_names_out(),
                                index=cfs.index,
                            )

                            query_instance = pd.read_csv(
                                f"counterfactual_explainers/results/cf_{explainer}_{model_name}_{dataset}_query_instance.csv"
                            )

                            query_encoded = preprocessor.transform(
                                query_instance[features.columns]
                            )

                            if isspmatrix_csr(query_encoded):
                                query_encoded = query_encoded.toarray()

                            query_encoded_df = pd.DataFrame(
                                query_encoded,
                                columns=preprocessor.get_feature_names_out(),
                                index=query_instance.index,
                            )

                            X_train_scaled = features_encoded_df.loc[
                                X_train.index
                            ]

                            continuous_columns = features_encoded_df.columns[
                                features_encoded_df.columns.str.startswith(
                                    "continuous__"
                                )
                            ]

                            categorical_columns = features_encoded_df.columns[
                                features_encoded_df.columns.str.startswith(
                                    "categorical__"
                                )
                            ]

                            non_act_columns = features_encoded_df.columns[
                                features_encoded_df.columns.str.contains(
                                    "|".join(non_act_features)
                                )
                            ]

                            size = calc_size(num_required_cfs, cfs)
                            mad = X_train_scaled[continuous_columns].apply(
                                median_absolute_deviation
                            )

                            dist = cfs_encoded_df.apply(
                                distance,
                                axis=1,
                                mad=mad,
                                query_instance=query_encoded_df,
                                continuous_features=continuous_columns,
                                categorical_features=categorical_columns,
                            )
                            dis_dist = dist.mean()

                            dis_changes = cfs_encoded_df.apply(
                                calc_changes,
                                axis=1,
                                query_instance=query_encoded_df,
                                features=features_encoded_df.columns,
                            )
                            dis_count = (dis_changes.mean()) / len(
                                features.columns
                            )
                            act = (
                                cfs_encoded_df.apply(
                                    actionable,
                                    axis=1,
                                    query_instance=query_encoded_df,
                                    non_act_features=non_act_columns,
                                ).sum()
                                / num_required_cfs
                            )

                            div_dist, div_count = calc_diversity(
                                cfs_encoded_df,
                                continuous_columns,
                                categorical_columns,
                                features_encoded_df.columns,
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

                if results:
                    results_df = pd.DataFrame(results).set_index(
                        "num_required_cfs"
                    )
                    print("Metrics Summary DataFrame:")
                    print(results_df)
                    results_path = Path("./counterfactual_explainers/results")
                    results_path.mkdir(parents=True, exist_ok=True)

                    results_df.to_csv(
                        results_path
                        / f"cf_{explainer}_{model_name}_{dataset}_"
                        f"metrics_2.csv",
                    )


if __name__ == "__main__":
    main()
