"""
A module for calculating counterfactual explanation metrics.

This module provides functionality for:
- Computing various counterfactual quality metrics (distance, diversity,
actionability)
- Encoding counterfactual data for metric calculation
- Loading pre-generated counterfactual results
- Generating comprehensive metric reports for different models and datasets

Key components:
- calc_mad: Computes Median Absolute Deviation for continuous features
- calc_distance: Calculates normalized distance between instances
- calc_diversity: Measures diversity among counterfactual explanations
- encode_cfs_to_dfs: Preprocesses data for metric calculation
- calculate_metrics_for_dataset: Orchestrates metric calculation workflow

Typical usage: TODO
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import isspmatrix_csr
from sklearn.model_selection import train_test_split

from counterfactual_explainers.data.preprocess_data import (
    clean_config,
    create_data_transformer,
    get_output_path,
    read_config,
    read_dataset,
)

# NOTE: later calculate Instability, Implausibility, and Discriminative Power

RESULTS_PATH = get_output_path()


def calc_mad(cf_row: pd.Series) -> float:
    """Calculate Median Absolute Deviation (MAD) for a given data row.

    Args:
        cf_row: Pandas Series representing a single data row

    Returns:
        MAD value as float. Returns 1.0 if MAD is zero to avoid division by
        zero.
    """
    mad = (cf_row - cf_row.median()).abs().median()
    return mad if mad else 1


def calc_distance(
    cf_row: pd.Series,
    query_instance: pd.Series,
    mad: pd.Series,
    continuous_features: list[str],
    categorical_features: list[str],
) -> float:
    """Calculate normalized distance between counterfactual and query instance.

    Args:
        cf_row: Counterfactual instance as pandas Series
        query_instance: Original query instance as pandas Series
        mad: MAD values for continuous features
        continuous_features: List of continuous feature names
        categorical_features: List of categorical feature names

    Returns:
        Combined normalized distance (continuous + categorical) as float
    """
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


def calc_changes(
    cf_row: pd.Series, query_instance: pd.Series, features: list[str]
) -> int:
    """Count number of feature changes between counterfactual and query
    instance.

    Args:
        cf_row: Counterfactual instance as pandas Series
        query_instance: Original query instance as pandas Series
        features: List of feature names to consider

    Returns:
        Number of changed features as integer
    """
    cf_values = cf_row[features].to_numpy()
    query_values = query_instance[features].to_numpy()
    return (cf_values != query_values).sum()


def calc_size(num_required_cfs: int, cfs_df: pd.DataFrame) -> float:
    """Calculate size metric as ratio of generated CFs to required CFs.

    Args:
        num_required_cfs: Requested number of counterfactuals
        cfs_df: DataFrame containing generated counterfactuals

    Returns:
        Size metric as float
    """
    return len(cfs_df) / num_required_cfs


def calc_actionability(
    cf_row: pd.Series,
    query_instance: pd.DataFrame,
    non_act_features: list[str],
) -> int:
    """Check if counterfactual makes changes to non-actionable features.

    Args:
        cf_row: Counterfactual instance as pandas Series
        query_instance: Original query instance as DataFrame
        non_act_features: List of non-actionable feature names

    Returns:
        1 if no changes to non-actionable features, 0 otherwise
    """
    cf_values = cf_row[non_act_features].to_numpy()
    query_values = query_instance.iloc[0][non_act_features].to_numpy()
    print(cf_values)
    print(query_values)
    print(non_act_features)
    non_action_changes = (cf_values != query_values).any()
    return 0 if non_action_changes else 1


def calc_diversity(
    cfs: pd.DataFrame,
    continuous_features: list[str],
    categorical_features: list[str],
    feature_cols: list[str],
    mad: pd.Series,
) -> tuple[float, float]:
    """Calculate diversity metrics for a set of counterfactuals.

    Args:
        cfs: DataFrame of counterfactual instances
        continuous_features: List of continuous feature names
        categorical_features: List of categorical feature names
        feature_cols: All feature names
        mad: MAD values for continuous features

    Returns:
        Tuple containing:
            - diversity_distance: Normalized pairwise distance metric
            - diversity_count: Normalized feature change count metric
    """
    normalization_factor = len(cfs) ** 2

    pairwise_dists = cfs.apply(
        lambda row: cfs.apply(
            lambda row2: calc_distance(
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


def encode_cfs_to_dfs(
    query_instance: pd.DataFrame,
    cfs: pd.DataFrame,
    continuous_features: list[str],
    categorical_features: list[str],
    features: pd.DataFrame,
    non_act_features: list[str],
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]
]:
    """Preprocess and encode data for metric calculations.

    Args:
        query_instance: Original query instance
        cfs: Counterfactual instances
        continuous_features: List of continuous feature names
        categorical_features: List of categorical feature names
        features: Original dataset features
        non_act_features: List of non-actionable feature names

    Returns:
        Tuple containing:
            - Encoded counterfactuals DataFrame
            - Encoded query instance DataFrame
            - Encoded features DataFrame
            - Continuous feature columns
            - Categorical feature columns
            - Non-actionable feature columns
    """

    preprocessor, _ = create_data_transformer(
        continuous_features=continuous_features,
        categorical_features=categorical_features,
    )

    encoded_features = preprocessor.fit_transform(features)

    if isspmatrix_csr(encoded_features):
        encoded_features = encoded_features.toarray()

    feature_names = preprocessor.get_feature_names_out()
    features_encoded_df = pd.DataFrame(
        encoded_features,
        columns=feature_names,
        index=features.index,
    )

    cfs_encoded = preprocessor.transform(cfs[features.columns])

    if isspmatrix_csr(cfs_encoded):
        cfs_encoded = cfs_encoded.toarray()

    cfs_encoded_df = pd.DataFrame(
        cfs_encoded,
        columns=preprocessor.get_feature_names_out(),
        index=cfs.index,
    )

    query_encoded = preprocessor.transform(query_instance[features.columns])

    if isspmatrix_csr(query_encoded):
        query_encoded = query_encoded.toarray()

    query_encoded_df = pd.DataFrame(
        query_encoded,
        columns=preprocessor.get_feature_names_out(),
        index=query_instance.index,
    )

    continuous_columns = features_encoded_df.columns[
        features_encoded_df.columns.str.startswith("continuous__")
    ]

    categorical_columns = features_encoded_df.columns[
        features_encoded_df.columns.str.startswith("categorical__")
    ]

    non_act_columns = features_encoded_df.columns[
        features_encoded_df.columns.str.contains("|".join(non_act_features))
    ]

    return (
        cfs_encoded_df,
        query_encoded_df,
        features_encoded_df,
        continuous_columns,
        categorical_columns,
        non_act_columns,
    )


def load_counterfactual_data(
    dataset: str,
    model_name: str,
    explainer: str,
    num_required_cfs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Load pre-generated counterfactual data and runtime information.

    Args:
        dataset: Dataset name
        model_name: Model type name
        explainer: Explainer method name
        num_required_cfs: Number of requested counterfactuals

    Returns:
        Tuple containing:
            - Counterfactuals DataFrame
            - Query instance DataFrame
            - Runtime in seconds
    """
    append = "" if explainer == "aide" else f"_{num_required_cfs}"
    index = 20 if explainer == "aide" else num_required_cfs

    cfs = pd.read_csv(
        RESULTS_PATH / f"cf_{explainer}_{model_name}_{dataset}{append}.csv"
    ).iloc[:num_required_cfs]

    query_instance = pd.read_csv(
        RESULTS_PATH
        / f"cf_{explainer}_{model_name}_{dataset}_query_instance.csv"
    )

    runtimes = pd.read_csv(
        RESULTS_PATH / f"runtime_{explainer}_{model_name}_{dataset}.csv"
    )
    runtimes.set_index("Number of Required CFS", inplace=True)
    runtime = runtimes.loc[index, "Runtime"]

    return cfs, query_instance, runtime


def calculate_metrics_for_dataset(
    config: dict[str, Any], dataset: str
) -> None:
    """Calculate metrics for a dataset across models and explainers.

    Args:
        config: Configuration dictionary
        dataset: Dataset name to process
    """
    data = read_dataset(config, dataset)
    params_dataset = config["dataset"][dataset]

    continuous_features = data["continuous_features"]
    categorical_features = data["categorical_features"]
    non_act_features = data["non_act_features"]
    features = data["features"]
    target = data["target"]

    for model_name in config["model"]:
        params_model = config["model"][model_name]
        seed = params_model["classifier__random_state"][0]
        for explainer in ["aide", "dice"]:
            results = []
            for num_required_cfs in range(1, 21):
                try:
                    print(
                        f"{model_name}, {dataset}, {num_required_cfs}, {explainer}"
                    )

                    cfs, query_instance, runtime = load_counterfactual_data(
                        dataset,
                        model_name,
                        explainer,
                        num_required_cfs,
                    )

                    (
                        cfs_encoded_df,
                        query_encoded_df,
                        features_encoded_df,
                        continuous_columns,
                        categorical_columns,
                        non_act_columns,
                    ) = encode_cfs_to_dfs(
                        query_instance,
                        cfs,
                        continuous_features,
                        categorical_features,
                        features,
                        non_act_features,
                    )

                    X_train, _, _, _ = train_test_split(
                        features,
                        target,
                        test_size=params_dataset["test_size"],
                        random_state=seed,
                        stratify=target,
                    )

                    X_train_scaled = features_encoded_df.loc[X_train.index]

                    size = calc_size(num_required_cfs, cfs)
                    mad = X_train_scaled[continuous_columns].apply(calc_mad)

                    dist = cfs_encoded_df.apply(
                        calc_distance,
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
                    dis_count = (dis_changes.mean()) / len(features.columns)
                    act = (
                        cfs_encoded_df.apply(
                            calc_actionability,
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
                            "Number of Required CFS": num_required_cfs,
                            "Size": size,
                            "Dissimilarity Distance": dis_dist,
                            "Dissimilarity Count": dis_count,
                            "Actionability": act,
                            "Diversity Distance": div_dist,
                            "Diversity Count": div_count,
                            "Runtime (in seconds)": runtime,
                        }
                    )

                except FileNotFoundError:
                    print("Not avaliable")

            if results:
                results_df = pd.DataFrame(results).set_index(
                    "Number of Required CFS"
                )
                print("Metrics Summary DataFrame:")
                print(results_df)

                results_df.to_csv(
                    RESULTS_PATH / f"cf_{explainer}_{model_name}_{dataset}_"
                    f"metrics.csv",
                )


def main() -> None:
    """Main execution function for metric calculation workflow."""
    config = clean_config(read_config())
    for dataset in config["dataset"]:
        calculate_metrics_for_dataset(config, dataset)


if __name__ == "__main__":
    main()
